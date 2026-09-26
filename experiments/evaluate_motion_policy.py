"""Evaluate the production pose selector on every held-out ARDY sample.

This is an offline, CPU-only replay of *saved* model outputs. It neither
generates new poses nor edits them. Geometric action checks are proxies; visual
review remains necessary to judge acting quality.

Usage:
    python experiments/evaluate_motion_policy.py \
      --input .runtime/motion-research/validation-v1 \
      --summary review/motion-validation-summary.json \
      --history-project .runtime/projects/StageZero-two-endings-20260926-015459-6a268b.stagezero.npz \
      --output review/motion-policy-validation.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from motion_policy import select_candidate  # noqa: E402
from summarize_motion_ablation import load_history_prefix  # noqa: E402

SEEDS = tuple(range(101, 111))
PROMPTS = {
    "overhead": "A person raises both arms overhead.",
    "wave": "A person waves with their right hand.",
    "squat": "A person does a squat.",
    "stop": "A person stops and stands still.",
}
LEGACY = "h52cfg2"
RESPONSIVE = "h4cfg2"
EXPRESSIVE = "h12cfg4"
DEFAULT_PROJECT = ROOT / ".runtime/projects/StageZero-two-endings-20260926-015459-6a268b.stagezero.npz"


def _require_case(index: dict, category: str, config: str, seed: int) -> dict:
    key = (category, config, seed)
    if key not in index:
        raise ValueError(f"Missing planned case {key}")
    case = index[key]
    if case.get("status") != "ok":
        raise ValueError(f"Failed planned case {key}: {case.get('status')}")
    if case.get("prompt") != PROMPTS[category]:
        raise ValueError(f"Prompt mismatch for {key}: {case.get('prompt')!r}")
    return case


def _load_poses(directory: Path, case: dict) -> dict:
    filename = case.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"Invalid filename in {case.get('id')}")
    with np.load(directory / filename, allow_pickle=False) as archive:
        positions = archive["positions"].copy()
        rotations = archive["rotations"].copy()
    if positions.shape != (104, 34, 3) or rotations.shape != (104, 34, 3, 3):
        raise ValueError(f"Invalid pose shape in {filename}")
    return {"positions": positions, "rotations": rotations}


def _brief_assessment(row: dict) -> dict:
    q = row.get("quality") or {}
    return {
        "safe": row["safe"],
        "reasons": row["reasons"],
        "selector_proxy_met": None if row["proxy"] is None else row["proxy"]["met"],
        "selector_proxy_score": row["score"],
        "selector_proxy_details": row["proxy"],
        "worst_seam_mean_joint_m": row["seam_mean_joint_m"],
        "floor_penetration_max_depth_m": row["floor_penetration_max_depth_m"],
        "root_mean_speed_mps": q.get("root_mean_speed_mps"),
        "foot_sliding_proxy_mean_mps": q.get("foot_sliding_proxy_mean_mps"),
    }


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--history-project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads((args.input / "report.json").read_text())
    summary = json.loads(args.summary.read_text())
    if source.get("status") != "complete" or len(source.get("cases", [])) != 120:
        raise ValueError("Held-out source must contain all 120 completed cases")
    if len(summary.get("cases", [])) != 120:
        raise ValueError("Held-out summary must contain all 120 assessed cases")
    report_index = {(r["prompt_category"], r["config"], r["seed"]): r for r in source["cases"]}
    score_index = {(r["prompt_category"], r["config"], r["seed"]): r for r in summary["cases"]}
    if len(report_index) != 120 or len(score_index) != 120:
        raise ValueError("Duplicate or missing case keys")
    prior_p, prior_r, history = load_history_prefix(args.history_project)

    cases = []
    for category in PROMPTS:
        for seed in SEEDS:
            # The deployed policy considers two same-seed settings for actions
            # that need stronger movement. Wave uses responsive alone; stop
            # uses expressive alone to avoid responsive's held-out regression.
            configs = ([RESPONSIVE, EXPRESSIVE] if category in ("overhead", "squat")
                       else [RESPONSIVE] if category == "wave" else [EXPRESSIVE])
            generated = [_require_case(report_index, category, config, seed) for config in configs]
            arrays = [_load_poses(args.input, row) for row in generated]
            decision = select_candidate(PROMPTS[category], arrays, prior_positions=prior_p,
                                        prior_rotations=prior_r, fps=25)
            legacy = _require_case(report_index, category, LEGACY, seed)
            legacy_poses = _load_poses(args.input, legacy)
            legacy_check = select_candidate(PROMPTS[category], [legacy_poses],
                                            prior_positions=prior_p, prior_rotations=prior_r, fps=25)
            selected_index = decision["chosen_index"]
            selected_config = None if selected_index is None else configs[selected_index]
            selected_score = None if selected_config is None else score_index[(category, selected_config, seed)]
            legacy_score = score_index[(category, LEGACY, seed)]
            if selected_score is not None and selected_score["status"] != "ok":
                raise ValueError(f"Summary missing selected case {category} {seed}")
            if legacy_score["status"] != "ok":
                raise ValueError(f"Summary missing legacy case {category} {seed}")
            timings = [float(row["metrics"]["case_seconds"]) for row in generated]
            if any(t < 0 for t in timings):
                raise ValueError("Negative case duration")
            cases.append({
                "action": category,
                "seed": seed,
                "candidate_configs": configs,
                "chosen_index": selected_index,
                "chosen_config": selected_config,
                "selector_action": decision["action"],
                "candidate_assessments": {config: _brief_assessment(assessment)
                                          for config, assessment in zip(configs, decision["assessments"])},
                "legacy_assessment": _brief_assessment(legacy_check["assessments"][0]),
                "legacy_selector_proxy_met": legacy_check["assessments"][0]["proxy"]["met"],
                "selected_selector_proxy_met": None if selected_index is None else decision["assessments"][selected_index]["proxy"]["met"],
                "legacy_summary_proxy_met": bool(legacy_score["proxy"]["success"]),
                "selected_summary_proxy_met": None if selected_score is None else bool(selected_score["proxy"]["success"]),
                "legacy_case_seconds": float(legacy["metrics"]["case_seconds"]),
                "candidate_case_seconds": dict(zip(configs, timings)),
                "total_candidate_case_seconds": sum(timings),
            })

    aggregates = {}
    for category in PROMPTS:
        rows = [row for row in cases if row["action"] == category]
        assert len(rows) == len(SEEDS)
        selected_proxy = sum(row["selected_selector_proxy_met"] is True for row in rows)
        legacy_proxy = sum(row["legacy_selector_proxy_met"] is True for row in rows)
        selected_summary = sum(row["selected_summary_proxy_met"] is True for row in rows)
        legacy_summary = sum(row["legacy_summary_proxy_met"] is True for row in rows)
        candidate_times = [row["total_candidate_case_seconds"] for row in rows]
        legacy_times = [row["legacy_case_seconds"] for row in rows]
        aggregates[category] = {
            "planned_seeds": list(SEEDS),
            "planned_cases": len(rows),
            "selected_configs": dict(Counter(row["chosen_config"] or "none" for row in rows)),
            "no_safe_selection_count": sum(row["chosen_index"] is None for row in rows),
            "legacy_selector_proxy_successes": legacy_proxy,
            "selected_selector_proxy_successes": selected_proxy,
            "legacy_summary_proxy_successes": legacy_summary,
            "selected_summary_proxy_successes": selected_summary,
            "selector_proxy_success_gain_count": selected_proxy - legacy_proxy,
            "summary_proxy_success_gain_count": selected_summary - legacy_summary,
            "legacy_mean_case_seconds": _mean(legacy_times),
            "selected_mean_total_candidate_seconds": _mean(candidate_times),
            "legacy_median_case_seconds": _median(legacy_times),
            "selected_median_total_candidate_seconds": _median(candidate_times),
            "mean_extra_case_seconds": _mean([a - b for a, b in zip(candidate_times, legacy_times)]),
            "mean_selected_worst_seam_joint_m": _mean([
                row["candidate_assessments"][row["chosen_config"]]["worst_seam_mean_joint_m"]
                for row in rows if row["chosen_config"] is not None]),
            "mean_selected_floor_penetration_depth_m": _mean([
                row["candidate_assessments"][row["chosen_config"]]["floor_penetration_max_depth_m"]
                for row in rows if row["chosen_config"] is not None]),
            "mean_selected_foot_sliding_proxy_mps": _mean([
                row["candidate_assessments"][row["chosen_config"]]["foot_sliding_proxy_mean_mps"]
                for row in rows if row["chosen_config"] is not None
                and row["candidate_assessments"][row["chosen_config"]]["foot_sliding_proxy_mean_mps"] is not None]),
        }

    result = {
        "schema_version": 1,
        "source_report": str(args.input / "report.json"),
        "source_summary": str(args.summary),
        "history_source": history,
        "model": source["model"],
        "design": "10 held-out seeds per action; same-seed responsive+expressive for overhead/squat, responsive for wave, expressive for stop; legacy independently assessed",
        "selector_proxy_rule": "Actual motion_policy.select_candidate: overhead/wave require 5 consecutive frames above shoulder, squat drop >= 0.15 m, stop final 25-frame mean planar root speed <= 0.15 m/s",
        "summary_proxy_rule": "Original summary counts 5 qualifying frames anywhere for hands; squat and stop thresholds are equivalent",
        "quality_rule": "Finite positions, valid rotations, <= 0.05 m toe penetration, <= 0.15 m mean joint seam at history and internal horizon",
        "timing_caveat": "Case seconds sum saved isolated model timings; excludes loading, RPC, queueing and selection overhead. First recorded case includes one-time warmup.",
        "proxy_caveat": "Geometric proxies do not prove the intended acting, hand oscillation, correct squat form, contact, or visual quality.",
        "aggregates": aggregates,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {x: v[x] for x in ("legacy_selector_proxy_successes", "selected_selector_proxy_successes", "selected_configs", "mean_extra_case_seconds")}
                      for k, v in aggregates.items()}, indent=2))


if __name__ == "__main__":
    main()
