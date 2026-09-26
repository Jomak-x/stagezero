"""Summarize every ARDY G1 ablation case from saved CPU-readable arrays.

Example:
    python experiments/summarize_motion_ablation.py \
        --input review/motion-ablation \
        --history-project .runtime/projects/StageZero-two-endings-20260926-015459-6a268b.stagezero.npz \
        --output review/motion-ablation-summary.json

The four action checks below are fixed geometric proxies, not judgments of
whether a motion actually follows its instruction. They do not detect a wave's
oscillation, a squat's knee form, or a stop's pose stability.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from motion_quality import HANDS, ROOT, SHOULDERS, analyze_motion  # noqa: E402


MODEL_NAME = "ARDY-G1-RP-25FPS-Horizon52"
FPS = 25
GENERATED_FRAMES = 104
DEFAULT_PROJECT = PROJECT_ROOT / ".runtime/projects/StageZero-two-endings-20260926-015459-6a268b.stagezero.npz"
PROXY_RULES = {
    "overhead": "Both hands > same-side shoulders + 0.15 m in at least 5 generated frames (simultaneously)",
    "wave": "Right hand > right shoulder + 0.15 m in at least 5 generated frames",
    "squat": "Minimum generated pelvis height >= 0.15 m below median pelvis height in the original prefix's last 4 frames",
    "stop": "Mean planar pelvis speed across the 24 intervals in the final 25 generated frames <= 0.15 m/s",
}


def load_history_prefix(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read the same original 104 frames used by the runner, without a model."""
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"]))
        if (manifest.get("version") != 1 or manifest.get("model") != MODEL_NAME
                or manifest.get("fps") != FPS or not manifest.get("takes")):
            raise ValueError("History project is not a compatible 25 fps G1 project with takes")
        takes = manifest["takes"]
        key = takes[0]["key"]
        positions = np.asarray(data[f"{key}_positions"][:GENERATED_FRAMES]).copy()
        rotations = np.asarray(data[f"{key}_rotations"][:GENERATED_FRAMES]).copy()
        motion = np.asarray(data[f"{key}_motion"][:GENERATED_FRAMES])
        if (positions.shape != (GENERATED_FRAMES, 34, 3)
                or rotations.shape != (GENERATED_FRAMES, 34, 3, 3)
                or motion.shape != (GENERATED_FRAMES, 414)
                or not np.isfinite(positions).all() or not np.isfinite(rotations).all()
                or not np.isfinite(motion).all()):
            raise ValueError("Original history prefix has invalid G1 arrays")
        for take in takes[1:]:
            other = np.asarray(data[f"{take['key']}_motion"][:GENERATED_FRAMES])
            if other.shape == motion.shape and not np.array_equal(other, motion):
                raise ValueError("Project takes disagree on the original 104-frame motion prefix")
    return positions, rotations, {"file": str(path), "take_key": key, "source_frames": GENERATED_FRAMES}


def _seam(prior: np.ndarray, current: np.ndarray, fps: float) -> dict:
    """Position step and root-velocity change at one known frame boundary."""
    joint_step = np.linalg.norm(current[0] - prior[-1], axis=1)
    boundary_velocity = (current[0, ROOT] - prior[-1, ROOT]) * fps
    entry_jump = None if len(prior) < 2 else float(np.linalg.norm(
        boundary_velocity - (prior[-1, ROOT] - prior[-2, ROOT]) * fps))
    exit_jump = None if len(current) < 2 else float(np.linalg.norm(
        (current[1, ROOT] - current[0, ROOT]) * fps - boundary_velocity))
    return {
        "mean_joint_position_jump_m": float(joint_step.mean()),
        "max_joint_position_jump_m": float(joint_step.max()),
        "root_position_jump_m": float(joint_step[ROOT]),
        "root_boundary_speed_mps": float(np.linalg.norm(boundary_velocity)),
        "root_entry_velocity_jump_mps": entry_jump,
        "root_exit_velocity_jump_mps": exit_jump,
    }


def _proxy(category: str, positions: np.ndarray, original_prefix: np.ndarray) -> dict:
    hand_margin = positions[:, HANDS, 1] - positions[:, SHOULDERS, 1]
    if category == "overhead":
        frames = int(np.count_nonzero((hand_margin > .15).all(axis=1)))
        return {"success": frames >= 5, "observed_frames": frames, "required_frames": 5}
    if category == "wave":
        frames = int(np.count_nonzero(hand_margin[:, 1] > .15))
        return {"success": frames >= 5, "observed_frames": frames, "required_frames": 5}
    if category == "squat":
        baseline = float(np.median(original_prefix[-4:, ROOT, 1]))
        minimum = float(positions[:, ROOT, 1].min())
        drop = baseline - minimum
        return {"success": bool(drop >= .15), "baseline_pelvis_height_m": baseline,
                "minimum_generated_pelvis_height_m": minimum, "drop_m": float(drop), "required_drop_m": .15}
    if category == "stop":
        last = positions[-25:, ROOT][:, (0, 2)]
        speed = np.linalg.norm(np.diff(last, axis=0), axis=1) * FPS
        mean = float(speed.mean())
        return {"success": bool(mean <= .15), "mean_last_25_root_speed_mps": mean,
                "maximum_mean_speed_mps": .15}
    raise ValueError(f"Unsupported prompt category: {category}")


def _case_file(input_dir: Path, record: dict) -> Path:
    name = record.get("file")
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".npz"):
        raise ValueError("Case has no valid NPZ filename")
    return input_dir / name


def _score_case(input_dir: Path, record: dict, prefix_positions: np.ndarray,
                prefix_rotations: np.ndarray) -> dict:
    row = {key: record.get(key) for key in ("id", "prompt_category", "prompt", "config",
                                             "initial_history_frames", "carry_history_frames", "cfg_weight", "seed")}
    row["runner_status"] = record.get("status")
    if record.get("status") != "ok":
        row["status"] = record.get("status", "missing")
        row["error"] = record.get("error", "Runner did not produce a successful case")
        return row
    try:
        path = _case_file(input_dir, record)
        with np.load(path, allow_pickle=False) as data:
            positions = np.asarray(data["positions"]).copy()
            rotations = np.asarray(data["rotations"]).copy()
        if positions.shape != (GENERATED_FRAMES, 34, 3) or rotations.shape != (GENERATED_FRAMES, 34, 3, 3):
            raise ValueError("Case NPZ has incompatible G1 array shapes")
        initial_history = record.get("initial_history_frames")
        if type(initial_history) is not int or initial_history < 0 or initial_history > len(prefix_positions):
            raise ValueError("Case has invalid initial history length")
        prior_p = prefix_positions[-initial_history:] if initial_history else None
        prior_r = prefix_rotations[-initial_history:] if initial_history else None
        quality = analyze_motion(positions, rotations, fps=FPS,
                                 prior_positions=prior_p, prior_rotations=prior_r)
        row["quality"] = quality
        if not quality["positions_finite"]:
            raise ValueError("Case positions contain non-finite values")
        if not quality["rotations_valid"]:
            raise ValueError("Case rotations are not valid G1 rotation matrices")
        category = record["prompt_category"]
        row.update(status="ok", file=path.name,
                   proxy=_proxy(category, positions, prefix_positions),
                   history_seam=_seam(prior_p, positions, FPS) if prior_p is not None else None,
                   horizon_seam=_seam(positions[:52], positions[52:], FPS),
                   timing=record.get("metrics", {}))
    except (OSError, KeyError, ValueError, TypeError) as exc:
        row.update(status="invalid_output", error=f"{type(exc).__name__}: {exc}")
    return row


def _mean(rows: list[dict], path: tuple[str, ...]) -> float | None:
    values = []
    for row in rows:
        value = row
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else None


def summarize(input_dir: Path, history_project: Path) -> dict:
    report = json.loads((input_dir / "report.json").read_text())
    model = report.get("model", {})
    design = report.get("design", {})
    if (report.get("schema_version") != 1 or model.get("name") != MODEL_NAME
            or model.get("fps") != FPS or design.get("generated_frames") != GENERATED_FRAMES):
        raise ValueError("Runner report does not describe the expected G1 25 fps ablation")
    prefix_p, prefix_r, history_info = load_history_prefix(history_project)
    if report.get("history_source"):
        source = report["history_source"]
        if source.get("take_key") != history_info["take_key"] or source.get("file") != history_project.name:
            raise ValueError("History project file or take key differs from the runner report")
    records = report.get("cases", [])
    if not isinstance(records, list):
        raise ValueError("Runner cases must be a list")
    by_id = {}
    for record in records:
        case_id = record.get("id")
        if not isinstance(case_id, str) or case_id in by_id:
            raise ValueError("Runner case IDs must be unique strings")
        by_id[case_id] = record
    prompts, configs, seeds = (design.get(name) for name in ("prompts", "configs", "seeds"))
    if not all(isinstance(value, list) and value for value in (prompts, configs, seeds)):
        raise ValueError("Runner design must list prompts, configs, and seeds")
    if any(prompt not in PROXY_RULES for prompt in prompts):
        raise ValueError("Runner design contains an unsupported prompt category")
    cases = []
    used = set()
    for category in prompts:
        for config in configs:
            for seed in seeds:
                case_id = f"{category}__{config}__seed{seed}"
                if case_id in by_id:
                    record = by_id[case_id]
                    if (record.get("prompt_category"), record.get("config"), record.get("seed")) != (category, config, seed):
                        raise ValueError(f"Case identity disagrees with design: {case_id}")
                    cases.append(_score_case(input_dir, record, prefix_p, prefix_r))
                    used.add(case_id)
                else:
                    cases.append({"id": case_id, "prompt_category": category, "config": config,
                                  "seed": seed, "status": "missing", "runner_status": "missing",
                                  "error": "No case record in runner report"})
    # Preserve unexpected records too; they must not disappear from the review.
    for record in records:
        if record["id"] not in used:
            extra = _score_case(input_dir, record, prefix_p, prefix_r)
            extra["outside_design"] = True
            cases.append(extra)
    aggregates = []
    for category in prompts:
        for config in configs:
            group = [row for row in cases if row["prompt_category"] == category and row["config"] == config
                     and not row.get("outside_design")]
            assessed = [row for row in group if row["status"] == "ok"]
            successes = sum(row["proxy"]["success"] for row in assessed)
            aggregates.append({
                "prompt_category": category, "config": config,
                "planned_seeds": list(seeds), "planned_cases": len(group),
                "assessed_cases": len(assessed), "successful_proxies": successes,
                "proxy_success_rate_all_planned": successes / len(group),
                "proxy_success_rate_assessed": successes / len(assessed) if assessed else None,
                "status_counts": dict(Counter(row["status"] for row in group)),
                "metric_case_counts": {
                    "history_seam": sum(row["history_seam"] is not None for row in assessed),
                    "foot_sliding_proxy": sum(row["quality"]["foot_sliding_proxy_mean_mps"] is not None for row in assessed),
                },
                "means_of_assessed_cases": {
                    "root_mean_speed_mps": _mean(assessed, ("quality", "root_mean_speed_mps")),
                    "floor_penetration_max_depth_m": _mean(assessed, ("quality", "floor_penetration_max_depth_m")),
                    "foot_sliding_proxy_mean_mps": _mean(assessed, ("quality", "foot_sliding_proxy_mean_mps")),
                    "history_seam_mean_joint_position_jump_m": _mean(assessed, ("history_seam", "mean_joint_position_jump_m")),
                    "history_seam_max_joint_position_jump_m": _mean(assessed, ("history_seam", "max_joint_position_jump_m")),
                    "history_seam_root_entry_velocity_jump_mps": _mean(assessed, ("history_seam", "root_entry_velocity_jump_mps")),
                    "horizon_seam_mean_joint_position_jump_m": _mean(assessed, ("horizon_seam", "mean_joint_position_jump_m")),
                    "horizon_seam_max_joint_position_jump_m": _mean(assessed, ("horizon_seam", "max_joint_position_jump_m")),
                    "horizon_seam_root_entry_velocity_jump_mps": _mean(assessed, ("horizon_seam", "root_entry_velocity_jump_mps")),
                },
            })
    return {
        "schema_version": 1, "source_report": str(input_dir / "report.json"),
        "runner_status": report.get("status"), "model": model, "design": design,
        "history_source": history_info, "proxy_rules": PROXY_RULES,
        "proxy_caveat": "Geometric threshold checks only; prompt adherence and motion quality require visual review.",
        "status_counts": dict(Counter(row["status"] for row in cases)),
        "cases": cases, "aggregates": aggregates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Runner output directory containing report.json and case NPZ files")
    parser.add_argument("--history-project", type=Path, default=DEFAULT_PROJECT,
                        help="Original .stagezero.npz project used as the runner's history source")
    parser.add_argument("--output", type=Path, required=True, help="Summary JSON path")
    args = parser.parse_args()
    if args.output.resolve() == (args.input / "report.json").resolve():
        parser.error("--output must not overwrite the runner report")
    try:
        result = summarize(args.input, args.history_project)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, KeyError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print("prompt       config       proxy/planned  assessed  history seam mean (m)  floor max depth (m)")
    for row in result["aggregates"]:
        means = row["means_of_assessed_cases"]
        seam = means["history_seam_mean_joint_position_jump_m"]
        floor = means["floor_penetration_max_depth_m"]
        seam_text = "n/a" if seam is None else f"{seam:.3f}"
        floor_text = "n/a" if floor is None else f"{floor:.3f}"
        print(f"{row['prompt_category']:<12} {row['config']:<12} "
              f"{row['successful_proxies']:>2}/{row['planned_cases']:<8} "
              f"{row['assessed_cases']:>3}/{row['planned_cases']:<5} "
              f"{seam_text:>21} {floor_text:>20}")
    print(f"Saved {len(result['cases'])} case records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
