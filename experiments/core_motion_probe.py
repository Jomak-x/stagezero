"""Bounded, isolated ARDY Core human-rig inference comparison.

Run on the existing CUDA Pod, never in the live StageZero backend. One common
Core-native standing prefix is generated, then 4 prompts x 2 text CFG values
x 3 seeds = 24 clips of two 40-frame horizons each. The prefix and each clip
are saved for visual review; geometric proxies are not semantic ground truth.

Example:
  python experiments/core_motion_probe.py --output motion-research/core-v1
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback


MODEL = "ARDY-Core-RP-20FPS-Horizon40"
PROMPTS = {
    "wave": "A person waves with their right hand while standing in place.",
    "overhead": "A person raises both arms overhead.",
    "squat": "A person does a squat.",
    "turn_react": "A person turns around suddenly as if startled.",
}
SEEDS = (11, 22, 33)
TEXT_CFG = (2.0, 4.0)
START_PROMPT = "A person stands still in a relaxed neutral pose."
START_SEED = 8171


def save_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def decode(model, motion, np):
    with __import__("torch").inference_mode():
        output = model.motion_rep.inverse(motion, is_normalized=True)
    arrays = {
        "motion": motion[0].detach().cpu().numpy().astype(np.float32, copy=True),
        "positions": output["posed_joints"][0].detach().cpu().numpy().astype(np.float32, copy=True),
        "rotations": output["global_rot_mats"][0].detach().cpu().numpy().astype(np.float32, copy=True),
    }
    joint_count = model.skeleton.nbjoints
    frames = motion.shape[1]
    expected = {
        "motion": (frames, model.motion_rep.motion_rep_dim),
        "positions": (frames, joint_count, 3),
        "rotations": (frames, joint_count, 3, 3),
    }
    for name, array in arrays.items():
        if array.shape != expected[name] or not np.isfinite(array).all():
            raise ValueError(f"Invalid {name}: shape {array.shape}, expected {expected[name]}")
    rotations = arrays["rotations"]
    ortho = rotations @ np.swapaxes(rotations, -1, -2)
    if not np.allclose(ortho, np.eye(3), atol=0.025) or not np.allclose(np.linalg.det(rotations), 1.0, atol=0.025):
        raise ValueError("Non-orthonormal joint rotation")
    return arrays


def metrics(positions, rotations, prefix_positions, prefix_rotations, joints, np):
    idx = {name: joints.index(name) for name in
           ("Hips", "LeftShoulder", "RightShoulder", "LeftHand", "RightHand", "Head", "LeftFoot", "RightFoot")}
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    seam = np.linalg.norm(positions[0] - prefix_positions[-1], axis=-1)
    midpoint = np.linalg.norm(positions[40] - positions[39], axis=-1)
    shoulder = np.maximum(positions[:, idx["LeftShoulder"], 1], positions[:, idx["RightShoulder"], 1])
    both_up = (positions[:, idx["LeftHand"], 1] > shoulder + .15) & (positions[:, idx["RightHand"], 1] > shoulder + .15)
    right_up = positions[:, idx["RightHand"], 1] > positions[:, idx["RightShoulder"], 1] + .10
    initial_hip_y = float(np.median(prefix_positions[-4:, idx["Hips"], 1]))
    hip_drop = initial_hip_y - positions[:, idx["Hips"], 1]
    # The head turn proxy is an orientation change of the skeleton root, not
    # a facial or gaze measurement. Rotation-matrix trace handles wraparound.
    relative = rotations[:, idx["Hips"]] @ prefix_rotations[-1, idx["Hips"]].T
    root_angle = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    feet = positions[:, [idx["LeftFoot"], idx["RightFoot"]]]
    foot_speed = np.linalg.norm(np.diff(feet, axis=0), axis=-1) * 20
    return {
        "history_seam_mean_joint_m": float(seam.mean()),
        "history_seam_max_joint_m": float(seam.max()),
        "internal_horizon_seam_mean_joint_m": float(midpoint.mean()),
        "mean_joint_step_m": float(steps.mean()),
        "max_joint_step_m": float(steps.max()),
        "overhead_both_hands_up_frames": int(both_up.sum()),
        "right_hand_above_shoulder_frames": int(right_up.sum()),
        "max_hip_drop_m": float(hip_drop.max()),
        "max_root_rotation_deg": float(np.degrees(root_angle.max())),
        "mean_foot_speed_m_s": float(foot_speed.mean()),
        "min_foot_height_m": float(feet[..., 1].min()),
        "root_path_m": float(np.linalg.norm(np.diff(positions[:, idx["Hips"], :][:, [0, 2]], axis=0), axis=-1).sum()),
    }


def sample(model, history, embedding, cfg, seed, torch, seed_everything):
    seed_everything(seed)
    chunks = []
    step_seconds = []
    current = history
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(2):
            initial = 0 if current is None else current.shape[1]
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = model.autoregressive_step(
                num_frames=initial + 40,
                num_denoising_steps=model.diffusion.num_base_steps,
                motion_mask=None,
                observed_motion=None,
                cfg_weight=(cfg, 2.0),
                text_feat=embedding[0],
                text_pad_mask=embedding[1],
                init_history_sequence=current,
            )
            chunks.append(result[:, initial:initial + 40])
            current = result[:, -4:]
            torch.cuda.synchronize()
            step_seconds.append(time.perf_counter() - started)
    return torch.cat(chunks, dim=1), step_seconds, torch.cuda.max_memory_allocated() / 2**30


def serve_review(directory: Path, port: int) -> None:
    """Play saved Core poses with the official human-shaped CoreSkin mesh."""
    import numpy as np
    import torch
    import viser
    from ardy.skeleton import CoreSkeleton27
    from ardy.viz.viser_utils import Character

    files = sorted(directory.glob("*__cfg*__seed*.npz"))
    if not files:
        raise ValueError(f"No Core probe clips in {directory}")
    with np.load(directory / "shared_prefix.npz", allow_pickle=False) as data:
        prefix_positions = data["positions"].copy()
        prefix_rotations = data["rotations"].copy()

    def load_clip(path):
        with np.load(path, allow_pickle=False) as data:
            positions = data["positions"].copy()
            rotations = data["rotations"].copy()
        if positions.shape != (80, 27, 3) or rotations.shape != (80, 27, 3, 3):
            raise ValueError(f"Unexpected Core clip dimensions: {path.name}")
        return np.concatenate((prefix_positions, positions)), np.concatenate((prefix_rotations, rotations))

    torch.set_num_threads(2)
    skeleton = CoreSkeleton27().to("cpu")
    server = viser.ViserServer(host="127.0.0.1", port=port, label="ARDY Core motion review")
    server.gui.configure_theme(dark_mode=True, control_layout="floating", show_logo=False, show_share_button=False)
    server.scene.set_up_direction("+y")
    server.scene.world_axes.visible = False
    server.scene.add_box("/floor", color=(28, 36, 46), dimensions=(200, .1, 200), position=(0, -.1, 0), cast_shadow=False)
    actor = Character("core_actor", server, skeleton, create_skeleton_mesh=False,
                      create_skinned_mesh=True, mesh_mode="core_skin", show_foot_contacts=False)
    positions, rotations = load_clip(files[0])
    state = {"positions": positions, "rotations": rotations, "frame": -1, "playing": False,
             "started": 0.0, "root": positions[0, 0].copy()}
    server.gui.add_markdown("# ARDY Core motion review\n**Saved 27-joint human rig clips · no inference**")
    chosen = server.gui.add_dropdown("Case", tuple(x.name for x in files), initial_value=files[0].name)
    info = server.gui.add_markdown("")
    play = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
    pause = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
    follow = server.gui.add_checkbox("Follow actor", initial_value=True)
    slider = server.gui.add_slider("Frame", min=0, max=83, step=1, initial_value=0)
    status = server.gui.add_markdown("")

    def reset_camera(client):
        root = state["positions"][0, 0]
        client.camera.position = (float(root[0] + 2.5), 1.6, float(root[2] + 3.2))
        client.camera.look_at = (float(root[0]), .85, float(root[2]))
        client.camera.up_direction = (0, 1, 0)
        client.camera.fov = np.deg2rad(43)

    @server.on_client_connect
    def connected(client):
        reset_camera(client)

    def show_frame(frame):
        frame = max(0, min(int(frame), 83))
        root = state["positions"][frame, 0].copy()
        delta = root - state["root"]
        delta[1] = 0
        with server.atomic():
            actor.set_pose(torch.from_numpy(state["positions"][frame]),
                           torch.from_numpy(state["rotations"][frame]))
            if follow.value and np.any(delta):
                for client in server.get_clients().values():
                    client.camera.position = np.asarray(client.camera.position) + delta
        state["frame"] = frame
        state["root"] = root
        slider.value = frame
        status.content = f"**{'Playing' if state['playing'] else 'Paused'}** · {frame / 20:.2f} s · frame {frame}"

    def show_case(path):
        state["playing"] = False
        state["positions"], state["rotations"] = load_clip(path)
        state["root"] = state["positions"][0, 0].copy()
        info.content = f"**Case:** {path.name}  \n**Rig:** Core 27 · 20 fps · four-frame generated prefix"
        for client in server.get_clients().values():
            reset_camera(client)
        show_frame(0)

    @chosen.on_update
    def choose(_):
        show_case(directory / chosen.value)

    @slider.on_update
    def scrub(_):
        if int(slider.value) != state["frame"]:
            state["playing"] = False
            show_frame(slider.value)

    @play.on_click
    def start(_):
        if state["frame"] >= 83:
            show_frame(0)
        state["started"] = time.perf_counter() - state["frame"] / 20
        state["playing"] = True

    @pause.on_click
    def stop(_):
        state["playing"] = False

    show_case(files[0])
    print(f"Core review: http://127.0.0.1:{port}", flush=True)
    try:
        while True:
            if state["playing"]:
                frame = min(int((time.perf_counter() - state["started"]) * 20), 83)
                if frame != state["frame"]:
                    show_frame(frame)
                if frame == 83:
                    state["playing"] = False
            time.sleep(.025)
    except KeyboardInterrupt:
        server.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--serve-review", action="store_true", help="Open saved Core clips in a local Viser viewer; no model loading")
    parser.add_argument("--port", type=int, default=2338, help="Local review port (default: 2338)")
    args = parser.parse_args()
    if args.serve_review:
        if not 1 <= args.port <= 65535:
            parser.error("--port must be between 1 and 65535")
        serve_review(args.output, args.port)
        return 0
    import numpy as np
    import torch
    from ardy.model import load_model
    from ardy.tools import seed_everything

    if not torch.cuda.is_available():
        parser.error("Requires the existing CUDA Pod")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report = {
        "schema_version": 1, "status": "loading", "started_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL, "experiment": "Core human rig, one shared generated prefix; 24 cases",
        "design": {"prompts": PROMPTS, "seeds": SEEDS, "text_cfg": TEXT_CFG, "constraint_cfg": 2.0,
                   "prefix_prompt": START_PROMPT, "prefix_seed": START_SEED, "history_frames": 4,
                   "generated_frames_per_case": 80, "fps": 20},
        "caveat": "Geometric proxies and shared-prefix generation cannot establish acting quality; inspect saved clips.",
        "cases": [],
    }
    save_json(report_path, report)
    try:
        torch.cuda.synchronize()
        started = time.perf_counter()
        model = load_model(MODEL, device="cuda", text_encoder_mode="local")
        torch.cuda.synchronize()
        if model.skeleton.name != "cskel27" or model.skeleton.nbjoints != 27 or model.motion_rep.fps != 20 or model.gen_horizon_len != 40:
            raise ValueError("Loaded checkpoint metadata does not match Core 27 / 20 fps / horizon 40")
        if model.num_frames_per_token != 4:
            raise ValueError("This probe requires 4-frame tokens")
        joints = list(model.skeleton.bone_order_names)
        report["model"] = {
            "skeleton": model.skeleton.name, "joint_count": model.skeleton.nbjoints,
            "joint_names": joints, "parents": [p for _, p in model.skeleton.bone_order_names_with_parents],
            "fps": model.motion_rep.fps, "horizon_frames": model.gen_horizon_len,
            "feature_count": model.motion_rep.motion_rep_dim,
            "diffusion_steps": int(model.diffusion.num_base_steps),
            "cfg_wrapper": type(model.denoiser).__name__,
            "gpu": torch.cuda.get_device_name(), "load_seconds": time.perf_counter() - started,
        }
        report["status"] = "initializing"
        save_json(report_path, report)
        with torch.inference_mode():
            start_embedding = model._encode_text([START_PROMPT])
            seed_everything(START_SEED)
            start_motion = model.autoregressive_step(
                num_frames=40, num_denoising_steps=model.diffusion.num_base_steps,
                motion_mask=None, observed_motion=None, cfg_weight=(2.0, 2.0),
                text_feat=start_embedding[0], text_pad_mask=start_embedding[1], init_history_sequence=None,
            )
            prefix_motion = start_motion[:, -4:].detach().clone()
            prefix = decode(model, prefix_motion, np)
        np.savez_compressed(args.output / "shared_prefix.npz", **prefix,
                            metadata=np.array(json.dumps({"prompt": START_PROMPT, "seed": START_SEED, "model": MODEL})))
        report["status"] = "running"
        save_json(report_path, report)
        for category, prompt in PROMPTS.items():
            try:
                with torch.inference_mode():
                    embedding = model._encode_text([prompt])
            except Exception as exc:
                embedding = None
                report.setdefault("encoding_failures", {})[category] = f"{type(exc).__name__}: {exc}"[:500]
            for cfg in TEXT_CFG:
                for seed in SEEDS:
                    case_id = f"{category}__cfg{cfg:g}__seed{seed}"
                    case = {"id": case_id, "category": category, "prompt": prompt,
                            "seed": seed, "text_cfg": cfg, "status": "running"}
                    report["cases"].append(case)
                    save_json(report_path, report)
                    try:
                        if embedding is None:
                            raise RuntimeError("text encoding failed")
                        torch.cuda.synchronize()
                        case_started = time.perf_counter()
                        motion, step_seconds, peak_gib = sample(
                            model, prefix_motion, embedding, cfg, seed, torch, seed_everything)
                        arrays = decode(model, motion, np)
                        values = metrics(arrays["positions"], arrays["rotations"],
                                         prefix["positions"], prefix["rotations"], joints, np)
                        values.update(step_seconds=step_seconds,
                                      total_seconds=time.perf_counter() - case_started,
                                      peak_gpu_allocated_gib=peak_gib)
                        np.savez_compressed(args.output / f"{case_id}.npz", **arrays,
                                            metadata=np.array(json.dumps({"model": MODEL, "fps": 20,
                                                                          "joint_names": joints, "case": case_id,
                                                                          "prefix_file": "shared_prefix.npz"})))
                        case.update(status="ok", file=f"{case_id}.npz", metrics=values)
                    except Exception as exc:
                        case.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:500],
                                    traceback=traceback.format_exc()[-2000:])
                    save_json(report_path, report)
                    print(json.dumps({"case": case_id, "status": case["status"]}), flush=True)
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:500],
                      traceback=traceback.format_exc()[-3000:])
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["counts"] = {state: sum(c["status"] == state for c in report["cases"]) for state in ("ok", "failed")}
    save_json(report_path, report)
    return 0 if report["status"] == "complete" and report["counts"] == {"ok": 24, "failed": 0} else 1


if __name__ == "__main__":
    raise SystemExit(main())
