"""Core hand-contact probe using only native ARDY conditioning.

Runs on a CUDA research Pod with the pinned ARDY Core checkpoint. Each case
shares its first 40 generated frames and RNG state between baseline and
conditioned second horizons. The contact target comes from a coherent Core
pose in a *different* saved sample; its entire pose is rigidly aligned to the
held-out actor's current root and heading. Nothing edits generated poses.

Example::

  python experiments/core_contact_probe.py \
      --reference-dir /workspace/stagezero/motion-research/core-v1 \
      --output /workspace/stagezero/motion-research/core-contact-v1

The exported ``build_contact_constraint`` also works in a trusted Python-only
``condition_hook``; never accept arbitrary pose tensors from an HTTP client.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np

from core_motion_probe import MODEL, decode


SCENARIOS = {
    "right_touch": {
        "prompt": "A person reaches their right hand to touch an object and holds their hand there.",
        "reference": "wave__cfg4__seed33.npz",
        "reference_frame": 72,
        "hands": ("RightHand",),
    },
    "two_hand_hold": {
        "prompt": "A person raises both hands to touch an object above them and holds it.",
        "reference": "overhead__cfg4__seed33.npz",
        "reference_frame": 72,
        "hands": ("LeftHand", "RightHand"),
    },
}
KEYFRAMES = (56, 68, 76)  # generated frames within horizon two
DEFAULT_SEEDS = (101, 102, 103, 104, 105)
YAW_OFFSETS_DEG = (-20.0, 0.0, 20.0, -10.0, 10.0)


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def save_npz(path: Path, arrays: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp, path)


def load_reference(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        result = {name: np.asarray(data[name], dtype=np.float32).copy()
                  for name in ("positions", "rotations")}
    if result["positions"].shape != (80, 27, 3) or result["rotations"].shape != (80, 27, 3, 3):
        raise ValueError(f"Reference clip has incompatible Core shape: {path}")
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError(f"Reference clip has nonfinite values: {path}")
    return result


def _heading(pose: np.ndarray, right_hip: int, left_hip: int) -> float:
    diff = pose[right_hip] - pose[left_hip]
    if np.linalg.norm(diff[[0, 2]]) < .05:
        raise ValueError("Hip vector is too short for yaw alignment")
    return float(np.arctan2(diff[2], -diff[0]))


def _y_rotation(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=np.float32)


def build_contact_constraint(
    model, current_history, reference: dict, *, hands: tuple[str, ...],
    source_frame: int = 72, yaw_offset_deg: float = 0.0,
    generated_offset: int = 40, keyframes: tuple[int, ...] = KEYFRAMES,
):
    """Return (official EndEffectorConstraintSet, inspectable target metadata).

    The target is a static hand location from one FK-coherent generated pose.
    Yaw changes rotate the *whole reference pose*, never its isolated hand.
    The current carry can move independently of the original prefix.
    """
    import torch
    from ardy.constraints import EndEffectorConstraintSet

    if model.skeleton.name != "cskel27" or model.gen_horizon_len != 40 or model.motion_rep.motion_rep_dim != 330:
        raise ValueError("Requires the pinned Core Horizon40 model")
    if not isinstance(current_history, torch.Tensor) or current_history.ndim != 3 or current_history.shape[0] != 1:
        raise ValueError("current_history must be (1,H,330)")
    history_len = int(current_history.shape[1])
    if current_history.shape[2] != 330 or history_len < 4 or history_len > 40 or history_len % 4:
        raise ValueError("Core history length must be 4..40 in four-frame tokens")
    if not bool(torch.isfinite(current_history).all()):
        raise ValueError("History contains nonfinite values")
    if generated_offset != 40 or not 0 <= source_frame < len(reference["positions"]):
        raise ValueError("Only the second Core horizon and a valid source frame are supported")
    if not keyframes or any(not generated_offset <= f < generated_offset + 40 for f in keyframes):
        raise ValueError("Contact frames must be inside horizon two")
    if hands not in (("RightHand",), ("LeftHand", "RightHand")):
        raise ValueError("Unsupported hand set")
    if not math.isfinite(yaw_offset_deg) or abs(yaw_offset_deg) > 30:
        raise ValueError("Yaw offset must stay within 30 degrees")

    with torch.inference_mode():
        decoded = model.motion_rep.inverse(current_history, is_normalized=True)
    current = decoded["posed_joints"][0, -1].detach().cpu().numpy()
    skeleton = model.skeleton
    root, right_hip, left_hip = int(skeleton.root_idx), *(int(x) for x in skeleton.hip_joint_idx)
    source_p = reference["positions"][source_frame]
    source_r = reference["rotations"][source_frame]
    source_yaw = _heading(source_p, right_hip, left_hip)
    target_yaw = _heading(current, right_hip, left_hip) + math.radians(yaw_offset_deg)
    delta_yaw = math.atan2(math.sin(target_yaw - source_yaw), math.cos(target_yaw - source_yaw))
    yrot = _y_rotation(delta_yaw)
    source_root = source_p[root]
    aligned_p = (source_p - source_root) @ yrot.T + source_root
    offset_xz = current[root, [0, 2]] - source_root[[0, 2]]
    aligned_p[:, 0] += offset_xz[0]
    aligned_p[:, 2] += offset_xz[1]
    aligned_r = yrot @ source_r
    frames = np.repeat(aligned_p[None], len(keyframes), axis=0).astype(np.float32)
    rotations = np.repeat(aligned_r[None], len(keyframes), axis=0).astype(np.float32)
    indices = torch.as_tensor([history_len + f - generated_offset for f in keyframes], dtype=torch.long)
    p = torch.from_numpy(frames).to(current_history.device)
    r = torch.from_numpy(rotations).to(current_history.device)
    root_2d = p[:, root, :][:, (0, 2)]
    constraint = EndEffectorConstraintSet(
        skeleton, indices, p, r, root_2d, joint_names=[*hands, "Hips"],
    )
    metadata = {
        "reference_frame": source_frame,
        "generated_keyframes": list(keyframes),
        "window_keyframes": indices.tolist(),
        "yaw_offset_deg": yaw_offset_deg,
        "alignment_yaw_rad": delta_yaw,
        "alignment_translation_xz_m": offset_xz.tolist(),
        "root_xz_m": aligned_p[root, [0, 2]].tolist(),
        "hand_targets_m": {hand: aligned_p[skeleton.bone_index[hand]].tolist() for hand in hands},
        "native_constraint": "EndEffectorConstraintSet",
        "caveat": "Targets are reference-assisted reachable poses, not arbitrary object geometry or a learned grasp.",
    }
    return constraint, metadata


def contact_metrics(positions: np.ndarray, rotations: np.ndarray, prior_positions: np.ndarray,
                    target: dict, skeleton, hands: tuple[str, ...]) -> dict:
    root = int(skeleton.root_idx)
    feet = [skeleton.bone_index[name] for name in ("LeftFoot", "RightFoot")]
    errors = {}
    target_frames = list(KEYFRAMES)
    for hand in hands:
        idx = skeleton.bone_index[hand]
        point = np.asarray(target["hand_targets_m"][hand], dtype=np.float32)
        distance = np.linalg.norm(positions[:, idx] - point, axis=-1)
        errors[hand] = {
            "keyframe_error_m": [float(distance[f]) for f in target_frames],
            "mean_keyframe_error_m": float(distance[target_frames].mean()),
            "min_after_first_goal_m": float(distance[target_frames[0]:].min()),
            "frames_within_10cm_after_first_goal": int((distance[target_frames[0]:] <= .10).sum()),
        }
    all_near = np.ones(len(positions), dtype=bool)
    for hand in hands:
        idx = skeleton.bone_index[hand]
        point = np.asarray(target["hand_targets_m"][hand], dtype=np.float32)
        all_near &= np.linalg.norm(positions[:, idx] - point, axis=-1) <= .10
    seam0 = float(np.linalg.norm(positions[0] - prior_positions[-1], axis=-1).mean())
    seam1 = float(np.linalg.norm(positions[40] - positions[39], axis=-1).mean())
    det = np.linalg.det(rotations)
    ortho = rotations @ np.swapaxes(rotations, -1, -2)
    floor = float(min(positions[:, feet, 1].min(), prior_positions[:, feet, 1].min()))
    return {
        "hands": errors,
        "simultaneous_contact_frames_after_first_goal": int(all_near[target_frames[0]:].sum()),
        "contact_at_all_three_keyframes": bool(all(all_near[f] for f in target_frames)),
        "history_seam_mean_joint_m": seam0,
        "internal_horizon_seam_mean_joint_m": seam1,
        "minimum_foot_height_m": floor,
        "root_path_m": float(np.linalg.norm(np.diff(positions[:, root, :][:, [0, 2]], axis=0), axis=-1).sum()),
        "quality_guard_met": bool(np.isfinite(positions).all() and np.isfinite(rotations).all()
                                  and np.allclose(ortho, np.eye(3), atol=.025)
                                  and np.allclose(det, 1.0, atol=.025)
                                  and max(seam0, seam1) <= .15 and floor >= -.05),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--cfg", type=float, default=4.0)
    args = parser.parse_args()
    try:
        seeds = tuple(int(piece.strip()) for piece in args.seeds.split(","))
    except ValueError:
        parser.error("Seeds must be comma-separated integers")
    if not seeds or len(seeds) > 5 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        parser.error("Choose one to five distinct nonnegative seeds")
    if not math.isfinite(args.cfg) or not 0 <= args.cfg <= 10:
        parser.error("CFG must be a finite number from 0 to 10")
    refs = {name: load_reference(args.reference_dir / spec["reference"])
            for name, spec in SCENARIOS.items()}
    with np.load(args.reference_dir / "shared_prefix.npz", allow_pickle=False) as data:
        prefix = {name: np.asarray(data[name], dtype=np.float32).copy()
                  for name in ("motion", "positions", "rotations")}
    if (prefix["motion"].shape != (4, 330) or prefix["positions"].shape != (4, 27, 3)
            or not all(np.isfinite(x).all() for x in prefix.values())):
        raise ValueError("Invalid Core shared prefix")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report = {
        "schema_version": 1, "status": "starting", "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "reference_directory": str(args.reference_dir),
        "design": "Five held-out seeds x two source-pose contact scenarios x paired same-RNG baseline/native-conditioned second horizon",
        "seeds": seeds, "text_cfg": args.cfg, "yaw_offsets_deg": YAW_OFFSETS_DEG,
        "scenarios": SCENARIOS, "keyframes": KEYFRAMES,
        "caveat": "An object marker is placed at a hand location from a prior coherent Core motion. This tests native target following, not arbitrary prop awareness or physical grasp.",
        "cases": [],
    }
    atomic_json(report_path, report)
    import torch
    from ardy.model import load_model
    from ardy.tools import seed_everything

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(4)
    started = time.perf_counter()
    model = load_model(MODEL, device="cuda", text_encoder_mode="local")
    torch.cuda.synchronize()
    if (model.skeleton.name != "cskel27" or model.skeleton.nbjoints != 27
            or model.gen_horizon_len != 40 or model.motion_rep.motion_rep_dim != 330):
        raise RuntimeError("Unexpected Core checkpoint")
    report.update(status="running", model_load_seconds=time.perf_counter() - started,
                  gpu=torch.cuda.get_device_name(), diffusion_steps=int(model.diffusion.num_base_steps))
    atomic_json(report_path, report)
    prefix_t = torch.from_numpy(prefix["motion"]).unsqueeze(0).to("cuda")
    for scenario, spec in SCENARIOS.items():
        embed = model._encode_text([spec["prompt"]])
        for case_index, seed in enumerate(seeds):
            case = {"scenario": scenario, "seed": seed, "status": "running",
                    "yaw_offset_deg": YAW_OFFSETS_DEG[case_index], "variants": {}}
            report["cases"].append(case)
            atomic_json(report_path, report)
            try:
                seed_everything(seed)
                with torch.inference_mode():
                    first_result = model.autoregressive_step(
                        num_frames=44, num_denoising_steps=model.diffusion.num_base_steps,
                        motion_mask=None, observed_motion=None, cfg_weight=(args.cfg, 2.0),
                        text_feat=embed[0], text_pad_mask=embed[1], init_history_sequence=prefix_t,
                    )
                    first = first_result[:, 4:]
                    current = first[:, -4:].detach().clone()
                cpu_rng = torch.random.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state()
                constraint, target = build_contact_constraint(
                    model, current, refs[scenario], hands=spec["hands"],
                    source_frame=spec["reference_frame"], yaw_offset_deg=YAW_OFFSETS_DEG[case_index],
                )
                case["target"] = target
                for variant in ("baseline", "conditioned"):
                    torch.random.set_rng_state(cpu_rng)
                    torch.cuda.set_rng_state(cuda_rng)
                    observed = mask = None
                    if variant == "conditioned":
                        lengths = torch.full((1,), 44, device="cuda", dtype=torch.long)
                        observed, mask = model.motion_rep.create_conditions_from_constraints_batched(
                            [[constraint]], lengths, to_normalize=True, device="cuda",
                        )
                        if observed.shape != (1, 44, 330) or mask.shape != (1, 44, 330):
                            raise ValueError("Wrong native condition tensor shape")
                        if int(mask[:, :4].count_nonzero()) or not int(mask.count_nonzero()):
                            raise ValueError("Native mask overlaps immutable history or is empty")
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    with torch.inference_mode():
                        result = model.autoregressive_step(
                            num_frames=44, num_denoising_steps=model.diffusion.num_base_steps,
                            motion_mask=mask, observed_motion=observed, cfg_weight=(args.cfg, 2.0),
                            text_feat=embed[0], text_pad_mask=embed[1], init_history_sequence=current,
                        )
                        second = result[:, 4:]
                        arrays = decode(model, torch.cat((first, second), dim=1), np)
                    torch.cuda.synchronize()
                    values = contact_metrics(arrays["positions"], arrays["rotations"],
                                             prefix["positions"], target, model.skeleton, spec["hands"])
                    row = {"status": "ok", "metrics": values,
                           "conditioned_channels": 0 if mask is None else int(mask.count_nonzero()),
                           "seconds_second_horizon_and_decode": time.perf_counter() - start,
                           "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
                    filename = f"{scenario}__seed{seed}__{variant}.npz"
                    save_npz(args.output / filename, {**arrays, "metadata": np.array(json.dumps({
                        "scenario": scenario, "seed": seed, "variant": variant,
                        "target": target, "metrics": values, "fps": 20,
                    }))})
                    case["variants"][variant] = {**row, "file": filename}
                case["status"] = "ok"
            except Exception as exc:
                case.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:500],
                            traceback=traceback.format_exc()[-2000:])
            atomic_json(report_path, report)
            print(json.dumps({"scenario": scenario, "seed": seed, "status": case["status"]}), flush=True)
    report["status"] = "complete"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["counts"] = {state: sum(c["status"] == state for c in report["cases"])
                        for state in ("ok", "failed")}
    report["summary"] = {}
    for scenario in SCENARIOS:
        completed = [case for case in report["cases"]
                     if case["scenario"] == scenario and case["status"] == "ok"]
        report["summary"][scenario] = {"completed_pairs": len(completed)}
        for variant in ("baseline", "conditioned"):
            rows = [case["variants"][variant]["metrics"] for case in completed]
            errors = [sum(hand["mean_keyframe_error_m"] for hand in row["hands"].values())
                      / len(row["hands"]) for row in rows]
            report["summary"][scenario][variant] = {
                "contact_at_all_three_keyframes": sum(row["contact_at_all_three_keyframes"] for row in rows),
                "quality_guard_met": sum(row["quality_guard_met"] for row in rows),
                "mean_hand_keyframe_error_m": float(np.mean(errors)) if errors else None,
            }
    atomic_json(report_path, report)
    return 0 if report["counts"] == {"ok": 2 * len(seeds), "failed": 0} else 1


if __name__ == "__main__":
    raise SystemExit(main())
