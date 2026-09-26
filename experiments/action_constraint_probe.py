"""Bounded, isolated ARDY G1 hand and full-body conditioning experiment.

Uses one original 104-frame StageZero prefix, a successful generated exemplar
for each action, and five held-out seeds.  Each seed's first 52-frame horizon
is identical for the unconditioned and conditioned second horizon.  No pose
array is patched after inference.  This is research code, not a backend path.

Run in the existing Pod's ARDY environment::

    python experiments/action_constraint_probe.py --source-project PROJECT \
      --exemplars DIR --output DIR
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from motion_ablation import MODEL_NAME, PROMPTS, _load_prefix


ACTION_SPECS = {
    "overhead": {
        "profile": "h4cfg2", "history": 4, "cfg": 2.0,
        "exemplar": "overhead__h4cfg2__seed33.npz",
        "keyframes": (80, 90, 100), "constraint": "both_hands",
    },
    "squat": {
        "profile": "h12cfg4", "history": 12, "cfg": 4.0,
        "exemplar": "squat__h12cfg4__seed33.npz",
        "keyframes": (88, 96, 103), "constraint": "full_body",
    },
}
SEEDS = (101, 102, 103, 104, 105)


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def save_npz(path: Path, arrays: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp, path)


def longest_run(flags: np.ndarray) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def score(action: str, positions: np.ndarray, prefix_positions: np.ndarray) -> dict:
    seam0 = float(np.linalg.norm(positions[0] - prefix_positions[-1], axis=-1).mean())
    seam1 = float(np.linalg.norm(positions[52] - positions[51], axis=-1).mean())
    floor_depth = float(max(0.0, -positions[:, (7, 14), 1].min()))
    if action == "overhead":
        margins = positions[:, (25, 33), 1] - positions[:, (18, 26), 1]
        run = longest_run((margins > .15).all(axis=1))
        proxy = {"longest_both_hands_above_shoulders_15cm_frames": run,
                 "minimum_run_frames": 5, "action_proxy_met": run >= 5,
                 "max_simultaneous_margin_m": float(np.max(np.min(margins, axis=1)))}
    else:
        baseline = float(np.median(prefix_positions[-4:, 0, 1]))
        drop = baseline - float(positions[:, 0, 1].min())
        proxy = {"pelvis_drop_from_prefix_m": drop, "minimum_drop_m": .15,
                 "action_proxy_met": drop >= .15}
    return {
        **proxy, "prefix_seam_mean_joint_m": seam0,
        "horizon_seam_mean_joint_m": seam1,
        "floor_penetration_max_depth_m": floor_depth,
        "quality_guard_met": bool(np.isfinite(positions).all() and max(seam0, seam1) <= .15 and floor_depth <= .05),
    }


def build_constraint(model, torch, spec: dict, exemplar: dict, second_history_positions: np.ndarray):
    from ardy.constraints import EndEffectorConstraintSet, FullBodyConstraintSet

    frames = np.asarray(spec["keyframes"], dtype=int)
    if not np.all((52 <= frames) & (frames < 104)):
        raise ValueError("All keyframes must occur in the second 52-frame horizon")
    history_len = int(spec["history"])
    # ARDY's EndEffector/FullBody constructors pair these indices with CPU
    # skeleton joint indices before moving the assembled mask to CUDA.
    indices = torch.as_tensor(history_len + frames - 52, dtype=torch.long)
    positions = np.asarray(exemplar["positions"][frames], dtype=np.float32).copy()
    rotations = np.asarray(exemplar["rotations"][frames], dtype=np.float32).copy()
    # Both cases share the original prefix but their first generated horizon
    # can move to a different world XZ. Preserve a coherent FK pose while
    # aligning the target pose's root to this seed's second-horizon start.
    origin = np.asarray(exemplar["positions"][51, 0, (0, 2)], dtype=np.float32)
    actual = np.asarray(second_history_positions[-1, 0, (0, 2)], dtype=np.float32)
    offset = actual - origin
    positions[..., 0] += offset[0]
    positions[..., 2] += offset[1]
    p = torch.from_numpy(positions).to("cuda")
    r = torch.from_numpy(rotations).to("cuda")
    root = p[:, model.skeleton.root_idx, :][:, (0, 2)]
    if spec["constraint"] == "both_hands":
        constraint = EndEffectorConstraintSet(
            model.skeleton, indices, p, r, root,
            joint_names=["LeftHand", "RightHand", "Hips"],
        )
    else:
        constraint = FullBodyConstraintSet(model.skeleton, indices, p, r, root)
    return constraint, {
        "window_indices": indices.cpu().tolist(),
        "original_clip_frames": frames.tolist(),
        "world_xz_translation_m": offset.tolist(),
        "target_root_y_m": positions[:, 0, 1].tolist(),
        "target_left_hand_y_m": positions[:, 25, 1].tolist(),
        "target_right_hand_y_m": positions[:, 33, 1].tolist(),
    }


def decode(model, torch, motion):
    decoded = model.motion_rep.inverse(motion, is_normalized=True)
    arrays = {
        "motion": motion[0].detach().cpu().numpy().copy(),
        "positions": decoded["posed_joints"][0].detach().cpu().numpy().copy(),
        "rotations": decoded["global_rot_mats"][0].detach().cpu().numpy().copy(),
    }
    if arrays["motion"].shape != (104, 414) or arrays["positions"].shape != (104, 34, 3):
        raise ValueError("Model returned wrong dimensions")
    if not all(np.isfinite(x).all() for x in arrays.values()):
        raise ValueError("Model returned nonfinite data")
    return arrays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", required=True, type=Path)
    parser.add_argument("--exemplars", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args()
    seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
    if not seeds or len(seeds) > 5 or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        parser.error("--seeds requires one to five distinct nonnegative integers")
    args.output.mkdir(parents=True, exist_ok=True)
    prefix, source_meta = _load_prefix(args.source_project, np)
    exemplars = {}
    for action, spec in ACTION_SPECS.items():
        with np.load(args.exemplars / spec["exemplar"], allow_pickle=False) as data:
            exemplars[action] = {key: data[key].copy() for key in ("positions", "rotations")}
        if exemplars[action]["positions"].shape != (104, 34, 3):
            raise ValueError(f"Invalid {action} exemplar")

    result = {
        "schema_version": 1, "status": "starting", "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME, "original_prefix": source_meta, "seeds": seeds,
        "actions": {a: {k: v for k, v in spec.items()} for a, spec in ACTION_SPECS.items()},
        "design": "Paired second-horizon unconditioned vs native pose-conditioned inference; same first horizon and second-horizon RNG state",
        "caveat": "Geometric action proxies are not human ratings; pose constraints are soft model inputs and are not exact guarantees.",
        "cases": [],
    }
    report_path = args.output / "report.json"
    atomic_json(report_path, result)

    import torch
    from ardy.model import load_model
    from ardy.tools import seed_everything

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.set_num_threads(4)
    load_start = time.perf_counter()
    model = load_model(MODEL_NAME, device="cuda", text_encoder_mode="local")
    torch.cuda.synchronize()
    if model.skeleton.name != "g1skel34" or model.gen_horizon_len != 52 or model.num_frames_per_token != 4:
        raise RuntimeError("Unexpected checkpoint")
    result.update(status="running", gpu=torch.cuda.get_device_name(),
                  model_load_seconds=time.perf_counter() - load_start,
                  checkpoint_cfg_wrapper=type(model.denoiser).__name__,
                  diffusion_base_steps=int(model.diffusion.num_base_steps))
    atomic_json(report_path, result)
    with torch.inference_mode():
        prefix_tensor = torch.from_numpy(prefix).unsqueeze(0).to("cuda")
        prefix_positions = model.motion_rep.inverse(prefix_tensor, is_normalized=True)["posed_joints"][0].cpu().numpy()

    for action, spec in ACTION_SPECS.items():
        text_feat, text_mask = model._encode_text([PROMPTS[action]])
        for seed in seeds:
            case = {"action": action, "seed": seed, "status": "running", "variants": {}}
            result["cases"].append(case)
            atomic_json(report_path, result)
            try:
                history_len = spec["history"]
                first_history = prefix_tensor[:, -history_len:]
                seed_everything(seed)
                with torch.inference_mode():
                    first_result = model.autoregressive_step(
                        num_frames=history_len + 52,
                        num_denoising_steps=model.diffusion.num_base_steps,
                        motion_mask=None, observed_motion=None,
                        cfg_weight=(spec["cfg"], 2.0),
                        text_feat=text_feat, text_pad_mask=text_mask,
                        init_history_sequence=first_history,
                    )
                    first = first_result[:, history_len:]
                    second_history = first[:, -history_len:].clone()
                    second_decoded = model.motion_rep.inverse(second_history, is_normalized=True)
                    second_history_positions = second_decoded["posed_joints"][0].cpu().numpy()
                torch.cuda.synchronize()
                cpu_state = torch.random.get_rng_state()
                cuda_state = torch.cuda.get_rng_state()
                constraint, target_info = build_constraint(
                    model, torch, spec, exemplars[action], second_history_positions,
                )
                case["target"] = target_info

                for variant in ("baseline", "conditioned"):
                    torch.random.set_rng_state(cpu_state)
                    torch.cuda.set_rng_state(cuda_state)
                    observed = motion_mask = None
                    if variant == "conditioned":
                        lengths = torch.full((1,), history_len + 52, dtype=torch.long, device="cuda")
                        observed, motion_mask = model.motion_rep.create_conditions_from_constraints_batched(
                            [[constraint]], lengths, to_normalize=True, device="cuda",
                        )
                        if int(motion_mask[:, :history_len].count_nonzero()) != 0:
                            raise RuntimeError("Conditioning overlaps immutable history")
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    with torch.inference_mode():
                        second_result = model.autoregressive_step(
                            num_frames=history_len + 52,
                            num_denoising_steps=model.diffusion.num_base_steps,
                            motion_mask=motion_mask, observed_motion=observed,
                            cfg_weight=(spec["cfg"], 2.0),
                            text_feat=text_feat, text_pad_mask=text_mask,
                            init_history_sequence=second_history,
                        )
                        second = second_result[:, history_len:]
                        arrays = decode(model, torch, torch.cat((first, second), dim=1))
                    torch.cuda.synchronize()
                    metrics = score(action, arrays["positions"], prefix_positions)
                    row = {
                        "status": "ok", "seconds_second_horizon_and_decode": time.perf_counter() - start,
                        "conditioned_channels": 0 if motion_mask is None else int(motion_mask.count_nonzero()),
                        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                        "metrics": metrics,
                    }
                    path = args.output / f"{action}__seed{seed}__{variant}.npz"
                    save_npz(path, {**arrays, "metadata": np.array(json.dumps({
                        "action": action, "seed": seed, "variant": variant, "profile": spec["profile"],
                        "target": target_info, "metrics": metrics,
                    }))})
                    row["npz"] = path.name
                    case["variants"][variant] = row
                    atomic_json(report_path, result)
                    print(action, seed, variant, metrics, flush=True)
                case["status"] = "ok"
            except Exception as exc:
                case["status"] = "failed"
                case["error"] = f"{type(exc).__name__}: {exc}"
                case["traceback"] = traceback.format_exc()
                print(action, seed, "FAILED", case["error"], flush=True)
            atomic_json(report_path, result)
    result["status"] = "complete" if all(c["status"] == "ok" for c in result["cases"]) else "partial_failure"
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(report_path, result)
    print(result["status"], len(result["cases"]), flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
