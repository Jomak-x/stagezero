"""Isolated ARDY G1 interaction-conditioning probe. Run on the existing GPU Pod.

This measures independent two-actor inference throughput and one-actor root
conditioning. It does not claim joint interaction or modify StageZero inference.

Example:
  python experiments/interaction_probe.py \
      --source-project /private/take.stagezero.npz \
      --json /private/interaction-probe.json \
      --npz /private/interaction-probe.npz

The JSON is replaced after every case so partial and failed runs remain visible.
The NPZ is written after the cases finish. Default: two seeds, ten model calls,
twelve generated actor samples. At most two seeds are allowed by this probe.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

# The script is kept outside the app modules, but accepts the same project file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takes import decode_project  # noqa: E402

MODEL_NAME = "ARDY-G1-RP-25FPS-Horizon52"
PROMPTS = (
    "A person waves their right hand while standing in place.",
    "A person raises both arms overhead while standing in place.",
)
APPROACH_PROMPT = "A person walks to the marked location and stops there."


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", required=True, type=Path,
                        help="Private saved .stagezero.npz project used for motion history")
    parser.add_argument("--json", required=True, type=Path, help="Rolling JSON result path")
    parser.add_argument("--npz", required=True, type=Path, help="Generated array archive path")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1],
                        help="One or two paired random seeds (default: 0 1)")
    parser.add_argument("--take-id", help="Use a specific take instead of the saved active take")
    parser.add_argument("--frame", type=int, help="Use this 0-based frame instead of the saved playhead")
    parser.add_argument("--target-dx", type=float, default=0.0,
                        help="Root target X offset in world meters (default: 0)")
    parser.add_argument("--target-dz", type=float, default=0.8,
                        help="Root target Z offset in world meters (default: 0.8)")
    args = parser.parse_args()
    if not 1 <= len(args.seeds) <= 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("use one or two distinct seeds")
    if any(not np.isfinite(v) or abs(v) > 3 for v in (args.target_dx, args.target_dz)):
        parser.error("target offsets must be finite and within 3 m")
    for output in (args.json, args.npz):
        if output.resolve() == args.source_project.resolve():
            parser.error("output paths must not overwrite the source project")
    if args.json.resolve() == args.npz.resolve():
        parser.error("JSON and NPZ paths must differ")
    return args


def save_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def load_history(args: argparse.Namespace, patch: int) -> tuple[np.ndarray, int, str]:
    takes, active, saved_frame, _ = decode_project(args.source_project.read_bytes())
    take_id = args.take_id or active
    if take_id not in takes:
        raise ValueError("requested take is not in the source project")
    take = takes[take_id]
    frame = saved_frame if args.frame is None else args.frame
    # Saved projects often have the playhead rewound to zero. In that case use
    # the selected take's end, which supplies a valid continuation history.
    if args.frame is None and frame + 1 < patch:
        frame = len(take.motion) - 1
    if not 0 <= frame < len(take.motion):
        raise ValueError("source frame is outside the selected take")
    history_length = min(52, frame + 1) // patch * patch
    if history_length < patch:
        raise ValueError(f"source frame needs at least {patch} history frames")
    history = take.motion[frame + 1 - history_length:frame + 1].astype(np.float32, copy=True)
    if history.shape != (history_length, 414) or not np.isfinite(history).all():
        raise ValueError("source history is incompatible with ARDY G1")
    return history, frame, take_id


def seed_all(torch, seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_arrays(motion: np.ndarray, positions: np.ndarray, rotations: np.ndarray,
                    actors: int, horizon: int) -> None:
    if motion.shape != (actors, horizon, 414):
        raise ValueError(f"invalid motion shape {motion.shape}")
    if positions.shape != (actors, horizon, 34, 3):
        raise ValueError(f"invalid position shape {positions.shape}")
    if rotations.shape != (actors, horizon, 34, 3, 3):
        raise ValueError(f"invalid rotation shape {rotations.shape}")
    if not all(np.isfinite(x).all() for x in (motion, positions, rotations)):
        raise ValueError("model returned nonfinite values")
    orthogonality = rotations @ np.swapaxes(rotations, -1, -2)
    if not np.allclose(orthogonality, np.eye(3), atol=0.02):
        raise ValueError("model returned non-orthogonal joint rotations")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=0.02):
        raise ValueError("model returned improper joint rotations")


def generate(model, torch, histories, prompts, *, seed, constraints=None):
    """One official autoregressive step; result is the generated horizon only."""
    actors, history_length, _ = histories.shape
    horizon = model.gen_horizon_len
    total_frames = history_length + horizon
    if constraints is not None and len(constraints) != actors:
        raise ValueError("one constraint list is required per actor")

    seed_all(torch, seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    text_feat, text_mask = model._encode_text(list(prompts))
    torch.cuda.synchronize()
    encoded = time.perf_counter()
    observed = motion_mask = None
    if constraints is not None:
        lengths = torch.full((actors,), total_frames, device="cuda", dtype=torch.long)
        observed, motion_mask = model.motion_rep.create_conditions_from_constraints_batched(
            constraints, lengths, to_normalize=True, device="cuda"
        )
        if observed.shape != (actors, total_frames, model.motion_rep.motion_rep_dim):
            raise ValueError("invalid observed-motion shape")
        if motion_mask.shape != observed.shape:
            raise ValueError("invalid constraint-mask shape")
        if motion_mask[:, :history_length].count_nonzero().item():
            raise ValueError("constraints overlap the immutable history")
    conditioned = time.perf_counter()

    with torch.inference_mode():
        samples = model.autoregressive_step(
            num_frames=total_frames,
            num_denoising_steps=model.diffusion.num_base_steps,
            motion_mask=motion_mask,
            observed_motion=observed,
            cfg_weight=(2.0, 2.0),
            text_feat=text_feat,
            text_pad_mask=text_mask,
            init_history_sequence=histories,
        )
        generated = samples[:, history_length:history_length + horizon]
        decoded = model.motion_rep.inverse(generated, is_normalized=True)
    torch.cuda.synchronize()
    generated_at = time.perf_counter()
    motion = generated.detach().cpu().numpy().copy()
    positions = decoded["posed_joints"].detach().cpu().numpy().copy()
    rotations = decoded["global_rot_mats"].detach().cpu().numpy().copy()
    copied_at = time.perf_counter()
    validate_arrays(motion, positions, rotations, actors, horizon)
    metrics = {
        "actors": actors,
        "seed": seed,
        "text_encoding_seconds": encoded - began,
        "condition_build_seconds": conditioned - encoded,
        "generation_decode_seconds": generated_at - conditioned,
        "total_seconds": copied_at - began,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "conditioned_channels": 0 if motion_mask is None else int(motion_mask.count_nonzero().item()),
    }
    return motion, positions, rotations, metrics


def boundary_metrics(source_positions: np.ndarray, output_positions: np.ndarray) -> dict:
    delta = np.linalg.norm(output_positions[0] - source_positions[-1], axis=-1)
    return {
        "boundary_mean_joint_jump_m": float(delta.mean()),
        "boundary_max_joint_jump_m": float(delta.max()),
        "boundary_root_jump_m": float(delta[0]),
    }


def main() -> int:
    args = arguments()
    result = {
        "schema": 1,
        "status": "starting",
        "model": MODEL_NAME,
        "experiment": "independent_two_actor_batch_and_root2d_approach",
        "caveat": "Two batch samples are independent single-actor motions, not learned joint interaction.",
        "seeds": args.seeds,
        "target_offset_xz_m": [args.target_dx, args.target_dz],
        "cases": [],
        "comparisons": [],
        "npz_path": str(args.npz),
    }
    save_json(args.json, result)
    arrays: dict[str, np.ndarray] = {}

    try:
        import torch
        from ardy.constraints import Root2DConstraintSet
        from ardy.model import load_model

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for this isolated probe")
        torch.set_num_threads(4)
        loaded_at = time.perf_counter()
        model = load_model(MODEL_NAME, device="cuda", text_encoder_mode="local")
        model_load_seconds = time.perf_counter() - loaded_at
        if model.skeleton.name != "g1skel34" or model.skeleton.nbjoints != 34:
            raise RuntimeError("loaded checkpoint is not the pinned G1 skeleton")
        if model.motion_rep.fps != 25 or model.gen_horizon_len != 52:
            raise RuntimeError("loaded checkpoint has unexpected FPS or horizon")
        patch = model.num_frames_per_token
        history_np, source_frame, _ = load_history(args, patch)
        history_a = torch.from_numpy(history_np).unsqueeze(0).to("cuda")
        # World-space shift gives actor B its own start position; ARDY still
        # generates the two samples independently, without mutual perception.
        history_b_world = model.motion_rep.translate_2d(
            model.motion_rep.unnormalize(history_a),
            torch.tensor([[2.0, 0.0]], dtype=history_a.dtype, device="cuda"),
        )
        history_b = model.motion_rep.normalize(history_b_world)
        histories = torch.cat([history_a, history_b], dim=0)
        source = model.motion_rep.inverse(histories, is_normalized=True)["posed_joints"].detach().cpu().numpy()
        if histories.shape != (2, len(history_np), 414) or not torch.isfinite(histories).all():
            raise RuntimeError("invalid prepared actor histories")
        result.update(status="running", gpu=torch.cuda.get_device_name(),
                      torch_version=torch.__version__, history_frames=len(history_np),
                      source_frame=source_frame,
                      model_load_seconds=model_load_seconds,
                      horizon_frames=model.gen_horizon_len,
                      checkpoint_cfg_wrapper=type(model.denoiser).__name__)
        save_json(args.json, result)

        def run_case(name, case_seed, actor_histories, source_positions,
                     prompts, constraints=None, target=None):
            case = {"name": name, "seed": case_seed, "status": "running",
                    "actor_count": len(prompts), "prompts": list(prompts)}
            result["cases"].append(case)
            save_json(args.json, result)
            try:
                motion, positions, rotations, metrics = generate(
                    model, torch, actor_histories, prompts,
                    seed=case_seed, constraints=constraints,
                )
                case.update(status="ok", **metrics)
                case["actor_metrics"] = [
                    boundary_metrics(source_positions[i], positions[i])
                    for i in range(len(prompts))
                ]
                if target is not None:
                    measured = positions[0, -1, 0, [0, 2]]
                    case["target_xz_m"] = target.tolist()
                    case["endpoint_root_xz_m"] = measured.tolist()
                    case["endpoint_root_error_m"] = float(np.linalg.norm(measured - target))
                for i in range(len(prompts)):
                    key = f"{name}_actor{i}"
                    arrays[f"{key}_motion"] = motion[i]
                    arrays[f"{key}_positions"] = positions[i]
                    arrays[f"{key}_rotations"] = rotations[i]
                case["array_prefixes"] = [f"{name}_actor{i}" for i in range(len(prompts))]
            except Exception as exc:
                case.update(status="failed", error_type=type(exc).__name__,
                            error=str(exc)[:1000], traceback=traceback.format_exc(limit=6)[-3000:])
            save_json(args.json, result)
            return case

        # The order alternates across seeds to expose cache/warmup order effects.
        for index, seed in enumerate(args.seeds):
            pair = {}
            modes = ("sequential", "batched") if index % 2 == 0 else ("batched", "sequential")
            for mode in modes:
                if mode == "sequential":
                    pair["sequential"] = [
                        run_case(f"seed{seed}_sequential_{actor}", seed,
                                 histories[actor:actor + 1], source[actor:actor + 1],
                                 [PROMPTS[actor]])
                        for actor in range(2)
                    ]
                else:
                    pair["batched"] = run_case(f"seed{seed}_batched", seed,
                                               histories, source, PROMPTS)

            seq = pair["sequential"]
            bat = pair["batched"]
            comparison = {"seed": seed, "status": "incomplete"}
            if bat["status"] == "ok" and all(x["status"] == "ok" for x in seq):
                seq_total = sum(x["total_seconds"] for x in seq)
                bat_total = bat["total_seconds"]
                comparison.update(
                    status="ok", sequential_total_seconds=seq_total,
                    batched_total_seconds=bat_total,
                    sequential_peak_allocated_gib=max(x["gpu_peak_allocated_gib"] for x in seq),
                    batched_peak_allocated_gib=bat["gpu_peak_allocated_gib"],
                    speedup_sequential_over_batch=seq_total / bat_total,
                )
            result["comparisons"].append(comparison)
            save_json(args.json, result)

            # Same seed and starting history for unconstrained/conditioned pair.
            # The final generated frame is index H+51 in the model's visible
            # history+generation window; position [-1] scores exactly that frame.
            case_seed = seed + 1000
            target = source[0, -1, 0, [0, 2]].copy()
            target += np.array([args.target_dx, args.target_dz], dtype=np.float32)
            target = target.astype(np.float32)
            final_index = len(history_np) + model.gen_horizon_len - 1
            root_constraint = Root2DConstraintSet(
                model.skeleton,
                frame_indices=torch.tensor([final_index], dtype=torch.long, device="cuda"),
                root_2d=torch.tensor(target[None, :], dtype=torch.float32, device="cuda"),
            )
            baseline = run_case(f"seed{seed}_approach_baseline", case_seed,
                                histories[0:1], source[0:1], [APPROACH_PROMPT], target=target)
            conditioned = run_case(f"seed{seed}_approach_root2d", case_seed,
                                   histories[0:1], source[0:1], [APPROACH_PROMPT],
                                   constraints=[[root_constraint]], target=target)
            comparison = {"seed": seed, "kind": "root2d_approach", "status": "incomplete",
                          "requested_frame_in_window": final_index}
            if baseline["status"] == conditioned["status"] == "ok":
                comparison.update(status="ok",
                                  baseline_error_m=baseline["endpoint_root_error_m"],
                                  conditioned_error_m=conditioned["endpoint_root_error_m"],
                                  error_reduction_m=baseline["endpoint_root_error_m"] - conditioned["endpoint_root_error_m"])
            result["comparisons"].append(comparison)
            save_json(args.json, result)

        arrays["source_history_actor0_motion"] = history_np
        arrays["source_history_actor0_positions"] = source[0]
        arrays["source_history_actor1_positions"] = source[1]
        save_npz(args.npz, arrays)
        failures = sum(case["status"] != "ok" for case in result["cases"])
        result.update(status="complete" if failures == 0 else "partial_failure",
                      successful_cases=len(result["cases"]) - failures,
                      failed_cases=failures,
                      archived_arrays=len(arrays))
        save_json(args.json, result)
        return 0 if failures == 0 else 1
    except Exception as exc:
        result.update(status="setup_failed", error_type=type(exc).__name__,
                      error=str(exc)[:1000], traceback=traceback.format_exc(limit=8)[-4000:])
        save_json(args.json, result)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
