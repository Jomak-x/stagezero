"""Isolated ARDY G1 motion ablation on the existing Pod.

Run with the Pod's official ``ardy`` environment. This script loads its own
model and never calls or modifies the live backend. Each case generates two
native 52-frame horizons from the same stored 104-frame prefix. The metrics
are timing and geometric checks; judging whether a gesture obeys its prompt
requires inspecting the saved motion.

Example:
    python experiments/motion_ablation.py --output review/motion-ablation
    python experiments/motion_ablation.py --output review/motion-quick --quick
    python experiments/motion_ablation.py --output review/motion-followup \
        --configs h52cfg2,h4carry52,h4cfg2 --prompts overhead,wave
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


MODEL_NAME = "ARDY-G1-RP-25FPS-Horizon52"
DEFAULT_PROJECT = (
    Path(__file__).resolve().parents[1]
    / ".runtime/projects/StageZero-two-endings-20260926-015459-6a268b.stagezero.npz"
)
PROMPTS = {
    "overhead": "A person raises both arms overhead.",
    "wave": "A person waves with their right hand.",
    "squat": "A person does a squat.",
    "stop": "A person stops and stands still.",
}


@dataclass(frozen=True)
class Config:
    name: str
    initial_history: int
    carry_history: int
    text_cfg: float


# These are deliberate comparisons, not a Cartesian product. Constraint CFG
# remains 2.0 as in pod_backend.py; there are no constraints in these cases.
CONFIGS = (
    Config("h52cfg2", 52, 52, 2.0),
    Config("h24cfg2", 24, 24, 2.0),
    Config("h12cfg2", 12, 12, 2.0),
    Config("h4cfg2", 4, 4, 2.0),
    Config("h4carry52", 4, 52, 2.0),
    Config("h52cfg3", 52, 52, 3.0),
    Config("h52cfg4", 52, 52, 4.0),
    Config("h24cfg4", 24, 24, 4.0),
    Config("h12cfg4", 12, 12, 4.0),
    Config("h52cfg6", 52, 52, 6.0),
    Config("coldcfg2", 0, 52, 2.0),
)
QUICK_CONFIGS = ("h52cfg2", "h4cfg2", "h4carry52", "h52cfg3")
DEFAULT_CONFIGS = (
    "h52cfg2", "h24cfg2", "h12cfg2", "h4cfg2", "h4carry52",
    "h52cfg4", "h24cfg4", "h12cfg4", "h52cfg6",
)


def _selection(value: str | None, available: dict, label: str) -> list[str]:
    if value is None:
        return list(available)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError(f"--{label} needs distinct comma-separated names")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(unknown)}; choices: {', '.join(available)}")
    return selected


def _write_report(path: Path, report: dict) -> None:
    """Replace atomically so interrupted runs retain every completed case."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _load_prefix(path: Path, np):
    if not path.is_file():
        raise FileNotFoundError(f"History project is missing: {path}; pass --history-project")
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"]))
        if manifest.get("version") != 1 or manifest.get("model") != MODEL_NAME or manifest.get("fps") != 25:
            raise ValueError("History project is not a compatible G1 25 FPS project")
        takes = manifest.get("takes", [])
        if not takes:
            raise ValueError("History project contains no takes")
        key = takes[0]["key"]
        first = np.asarray(data[f"{key}_motion"][:104], dtype=np.float32)
        if first.shape != (104, 414) or not np.isfinite(first).all():
            raise ValueError("First take has no valid 104-frame G1 motion prefix")
        # The comparison project should have the same original prefix in every
        # branch. Refuse an ambiguous source rather than silently choosing one.
        for take in takes[1:]:
            other = np.asarray(data[f"{take['key']}_motion"][:104], dtype=np.float32)
            if other.shape == first.shape and not np.array_equal(other, first):
                raise ValueError("Project takes disagree on the first 104 motion frames")
    return first, {"file": path.name, "take_key": key, "source_frames": 104}


def _geometry_metrics(positions, prefix_positions, np) -> dict:
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    metrics = {
        "mean_joint_step_m": float(steps.mean()),
        "max_joint_step_m": float(steps.max()),
        "horizon_boundary_mean_joint_step_m": float(steps[51].mean()),
        "joint0_path_m": float(np.linalg.norm(np.diff(positions[:, 0], axis=0), axis=-1).sum()),
    }
    if prefix_positions is not None:
        metrics["history_boundary_mean_joint_step_m"] = float(
            np.linalg.norm(positions[0] - prefix_positions[-1], axis=-1).mean()
        )
    return metrics


def _case(model, config: Config, prompt: str, seed: int, text_features, prefix, np, torch, seed_everything):
    seed_everything(seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    source = None if config.initial_history == 0 else prefix[-config.initial_history :]
    current = None if source is None else torch.from_numpy(source).unsqueeze(0).to("cuda")
    chunks = []
    step_seconds = []
    feat, mask = text_features
    with torch.inference_mode():
        for step in range(2):
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            history_len = 0 if current is None else current.shape[1]
            result = model.autoregressive_step(
                num_frames=history_len + 52,
                num_denoising_steps=model.diffusion.num_base_steps,
                motion_mask=None,
                observed_motion=None,
                cfg_weight=(config.text_cfg, 2.0),
                text_feat=feat,
                text_pad_mask=mask,
                init_history_sequence=current,
            )
            chunks.append(result[:, history_len : history_len + 52])
            if step == 0:
                current = result[:, -config.carry_history :]
            torch.cuda.synchronize()
            step_seconds.append(time.perf_counter() - step_started)
        motion = torch.cat(chunks, dim=1)
        output = model.motion_rep.inverse(motion, is_normalized=True)
        arrays = {
            "motion": motion[0].cpu().numpy(),
            "positions": output["posed_joints"][0].cpu().numpy(),
            "rotations": output["global_rot_mats"][0].cpu().numpy(),
        }
    torch.cuda.synchronize()
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise RuntimeError(f"Non-finite {name} from model")
    if arrays["motion"].shape != (104, model.motion_rep.motion_rep_dim):
        raise RuntimeError(f"Unexpected motion shape {arrays['motion'].shape}")
    if arrays["positions"].shape != (104, 34, 3) or arrays["rotations"].shape != (104, 34, 3, 3):
        raise RuntimeError("Unexpected G1 joint array dimensions")
    prefix_positions = None
    if source is not None:
        with torch.inference_mode():
            source_output = model.motion_rep.inverse(torch.from_numpy(source).unsqueeze(0).to("cuda"), is_normalized=True)
            prefix_positions = source_output["posed_joints"][0].cpu().numpy()
    torch.cuda.synchronize()
    metrics = {
        "step_seconds": step_seconds,
        "case_seconds": time.perf_counter() - started,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        **_geometry_metrics(arrays["positions"], prefix_positions, np),
    }
    return arrays, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Directory for per-case NPZ and rolling report.json")
    parser.add_argument("--history-project", type=Path, default=DEFAULT_PROJECT, help="Stored .stagezero.npz project containing the shared original 104-frame prefix")
    parser.add_argument("--seeds", default="11,22,33", help="Comma-separated integer seeds (default: 11,22,33)")
    parser.add_argument("--configs", help="Comma-separated preset names; use --list to inspect")
    parser.add_argument("--prompts", help="Comma-separated prompt categories: overhead,wave,squat,stop")
    parser.add_argument("--quick", action="store_true", help="One seed, four focused configs, all four prompts")
    parser.add_argument("--list", action="store_true", help="List configs and prompts without loading the model")
    args = parser.parse_args()
    configs_by_name = {config.name: config for config in CONFIGS}
    if args.list:
        for config in CONFIGS:
            print(f"{config.name}: initial={config.initial_history}, carry={config.carry_history}, cfg=({config.text_cfg:g},2)")
        for category, prompt in PROMPTS.items():
            print(f"{category}: {prompt}")
        return 0
    if args.output is None:
        parser.error("--output is required unless --list is used")
    try:
        seeds = [int(part.strip()) for part in args.seeds.split(",")]
        if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
            raise ValueError("--seeds needs distinct nonnegative integers")
        selected_configs = _selection(args.configs, configs_by_name, "configs") if args.configs is not None else list(DEFAULT_CONFIGS)
        selected_prompts = _selection(args.prompts, PROMPTS, "prompts")
        if args.quick:
            if args.configs is None:
                selected_configs = list(QUICK_CONFIGS)
            if args.seeds == "11,22,33":
                seeds = seeds[:1]
        if any(configs_by_name[name].initial_history for name in selected_configs) and not args.history_project.is_file():
            raise FileNotFoundError(f"History project is missing: {args.history_project}; pass --history-project")
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    # Heavy dependencies are imported only after CLI validation, so --help and
    # --list work on the development machine without the Pod's CUDA environment.
    import numpy as np
    import torch
    from ardy.model import load_model
    from ardy.tools import seed_everything

    if not torch.cuda.is_available():
        parser.error("CUDA is required; run this in the existing Pod's official ardy environment")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = None
    history_metadata = None
    if any(configs_by_name[name].initial_history for name in selected_configs):
        prefix, history_metadata = _load_prefix(args.history_project, np)
    torch.cuda.synchronize()
    load_started = time.perf_counter()
    model = load_model(MODEL_NAME, device="cuda", text_encoder_mode="local")
    torch.cuda.synchronize()
    if model.skeleton.name != "g1skel34" or model.skeleton.nbjoints != 34 or model.motion_rep.fps != 25 or model.gen_horizon_len != 52:
        raise RuntimeError("Loaded checkpoint is not the expected G1 25 FPS horizon-52 model")
    if model.num_frames_per_token != 4:
        raise RuntimeError(f"Preset histories require four-frame tokens; got {model.num_frames_per_token}")
    report = {
        "schema_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "model": {
            "name": MODEL_NAME,
            "skeleton": model.skeleton.name,
            "fps": model.motion_rep.fps,
            "generation_horizon_frames": model.gen_horizon_len,
            "features": model.motion_rep.motion_rep_dim,
            "frames_per_token": model.num_frames_per_token,
            "diffusion_num_base_steps": int(model.diffusion.num_base_steps),
            "cfg_wrapper": type(model.denoiser).__name__,
            "text_encoder": type(model.text_encoder).__name__,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "load_seconds": time.perf_counter() - load_started,
        },
        "history_source": history_metadata,
        "design": {"seeds": seeds, "configs": selected_configs, "prompts": selected_prompts, "constraint_cfg": 2.0, "generated_frames": 104},
        "encoding": {},
        "cases": [],
    }
    report_path = args.output / "report.json"
    _write_report(report_path, report)

    # Text is encoded once per prompt, then reused by every config and seed.
    embeddings = {}
    for category in selected_prompts:
        prompt = PROMPTS[category]
        try:
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                embeddings[category] = model._encode_text([prompt])
            torch.cuda.synchronize()
            report["encoding"][category] = {"prompt": prompt, "seconds": time.perf_counter() - started, "status": "ok"}
        except Exception as exc:
            report["encoding"][category] = {"prompt": prompt, "status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:500]}
        _write_report(report_path, report)

    for category in selected_prompts:
        for config_name in selected_configs:
            config = configs_by_name[config_name]
            for seed in seeds:
                case_id = f"{category}__{config_name}__seed{seed}"
                record = {
                    "id": case_id,
                    "prompt_category": category,
                    "prompt": PROMPTS[category],
                    "config": config_name,
                    "initial_history_frames": config.initial_history,
                    "carry_history_frames": config.carry_history,
                    "cfg_weight": [config.text_cfg, 2.0],
                    "seed": seed,
                }
                if category not in embeddings:
                    record.update(status="skipped", error="Text encoding failed")
                else:
                    try:
                        arrays, metrics = _case(model, config, PROMPTS[category], seed, embeddings[category], prefix, np, torch, seed_everything)
                        metadata = {**record, "model": MODEL_NAME, "fps": 25, "frames": 104, "metrics": metrics}
                        output_name = case_id + ".npz"
                        np.savez_compressed(args.output / output_name, **arrays, metadata=np.array(json.dumps(metadata, allow_nan=False)))
                        record.update(status="ok", file=output_name, metrics=metrics)
                    except Exception as exc:
                        record.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:500])
                report["cases"].append(record)
                _write_report(report_path, report)
                print(json.dumps({"id": case_id, "status": record["status"]}), flush=True)
    report["status"] = "complete"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["counts"] = {status: sum(case["status"] == status for case in report["cases"]) for status in ("ok", "failed", "skipped")}
    _write_report(report_path, report)
    return 0 if report["counts"]["failed"] == 0 and report["counts"]["skipped"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
