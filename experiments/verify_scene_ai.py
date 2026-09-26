"""Real configured-gateway trial for grounded scene direction planning.

Uses the existing private object-gateway configuration. It writes only bounded
directions, validated actions, timings, and outcome labels; never credentials
or raw provider response bodies. No deterministic fallback masks AI failures.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

from scene_ai_planner import SceneAIPlanner, compile_actions
from scene_objects import make_object


def _base_scene() -> dict:
    gate = make_object("arch", 0)
    gate.update(name="Gate", position=[0., 1.5, 0.], size=[2.7, 3., .38])
    ball = make_object("ball", 1)
    ball.update(name="Ball", position=[2.5, .15, -.5])
    return {"version": 2, "name": "Trial stage", "objects": [gate, ball],
            "effects": [], "lighting": "neutral"}


def _rotated_gate() -> tuple[dict, dict]:
    scene = {"version": 3, "objects": [{"id": "gate-rotated", "name": "Side portal",
             "kind": "custom", "asset": "asset-1", "position": [0., 1.5, 0.],
             "size": [3., 3., .5], "yaw": 45.}], "assets": [{"id": "asset-1"}]}
    affordances = {"gate-rotated": {"kind": "passage", "verified_open": True,
                   "width_m": 2., "height_m": 2.5, "depth_m": .4,
                   "yaw_degrees": 45., "floor_y_m": 0.}}
    return scene, affordances


def scenarios() -> list[dict]:
    base = _base_scene()
    alex = {"id": "alex", "name": "Alex", "position": [0., 0., -3.]}
    bea = {"id": "bea", "name": "Bea", "position": [3.5, 0., -3.]}
    rotated_scene, rotated_affordances = _rotated_gate()
    two_gates = _base_scene()
    second = make_object("arch", 2)
    second.update(name="Gate", position=[4., 1.5, 0.], size=[2.7, 3., .38])
    two_gates["objects"].append(second)
    return [
        {"id": "gate_name", "prompt": "Alex, go through the gate.",
         "scene": base, "actors": [alex], "affordances": {}, "expected": "route"},
        {"id": "rotated_gate_id", "prompt": "Alex, go through gate-rotated.",
         "scene": rotated_scene, "actors": [alex], "affordances": rotated_affordances,
         "expected": "route"},
        {"id": "ambiguous_gates", "prompt": "Alex, go through the gate.",
         "scene": two_gates, "actors": [alex], "affordances": {}, "expected": "reject_ambiguous"},
        {"id": "approach_then_pass", "prompt": "Alex, first approach the gate, then go through it.",
         "scene": base, "actors": [alex], "affordances": {}, "expected": "two_routes"},
        {"id": "nonexistent_object", "prompt": "Alex, go through the rocket hatch.",
         "scene": base, "actors": [alex], "affordances": {}, "expected": "reject_unknown"},
        {"id": "two_actors_staging", "prompt": "Alex approach the gate while Bea approaches the ball.",
         "scene": base, "actors": [alex, bea], "affordances": {}, "expected": "two_independent_routes"},
        {"id": "face_partner", "prompt": "Alex face Bea.",
         "scene": base, "actors": [alex, bea], "affordances": {}, "expected": "symbolic_only"},
        {"id": "handoff_prop", "prompt": "Alex hand the ball to Bea.",
         "scene": base, "actors": [alex, bea], "affordances": {}, "expected": "symbolic_only"},
    ]


def run_case(planner: SceneAIPlanner, case: dict) -> dict:
    result = {"id": case["id"], "prompt": case["prompt"], "expected": case["expected"]}
    started = time.perf_counter()
    try:
        plan = planner.plan(case["prompt"], case["scene"], case["actors"], case["affordances"])
        result["actions"] = plan["actions"]
        result["source"] = plan["source"]
        try:
            routes = compile_actions(plan, case["scene"], case["actors"], case["affordances"])
        except ValueError as exc:
            result["outcome"] = "validated_symbolic_or_uncompilable"
            result["reason"] = str(exc)
        else:
            result["outcome"] = "compiled_routes"
            result["route_count"] = len(routes)
            result["routes"] = [{"actor_id": route["geometry"]["actor_id"],
                                 "verb": route["geometry"]["verb"],
                                 "target_id": route["geometry"]["target_id"],
                                 "waypoints": route["geometry"]["waypoints"],
                                 "motion_following_verified": False}
                                for route in routes]
    except ValueError as exc:
        result["outcome"] = "rejected"
        result["reason"] = str(exc)
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    triples = [(action["actor_id"], action["verb"], action["target_id"])
               for action in result.get("actions", [])]
    expected_triples = {
        "gate_name": [("alex", "go_through", "arch-0")],
        "rotated_gate_id": [("alex", "go_through", "gate-rotated")],
        "approach_then_pass": [("alex", "approach", "arch-0"),
                               ("alex", "go_through", "arch-0")],
        "two_actors_staging": [("alex", "approach", "arch-0"),
                               ("bea", "approach", "ball-1")],
        "face_partner": [("alex", "face", "bea")],
        "handoff_prop": [("alex", "handoff", "bea")],
    }
    if case["id"] in ("ambiguous_gates", "nonexistent_object"):
        result["expectation_met"] = (result["outcome"] == "rejected" and
            ("Several objects match" if case["id"] == "ambiguous_gates" else "was not named")
            in result.get("reason", ""))
    else:
        desired_outcome = ("validated_symbolic_or_uncompilable" if case["id"] in
                           ("face_partner", "handoff_prop") else "compiled_routes")
        result["expectation_met"] = result["outcome"] == desired_outcome and triples == expected_triples[case["id"]]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Existing private objects.env path; never written to output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", choices=[case["id"] for case in scenarios()])
    args = parser.parse_args()
    planner = SceneAIPlanner.from_env(config_path=args.config)
    cases = [case for case in scenarios() if not args.only or case["id"] == args.only]
    records = []
    for case in cases:
        record = run_case(planner, case)
        records.append(record)
        print(f"{record['id']}: {record['outcome']} in {record['elapsed_seconds']:.3f}s")
    api_times = [case["elapsed_seconds"] for case in records if case["id"] != "ambiguous_gates"]
    report = {"experiment": "real_scene_ai_gateway_v1", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "provenance": "real configured project gateway; no fallback, no simulated transport",
              "model": planner.model, "cases": records,
              "summary": {"expectations_met": sum(case["expectation_met"] for case in records),
                          "cases": len(records),
                          "api_trial_median_seconds": round(statistics.median(api_times), 3) if api_times else None}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {len(records)} case outcomes to {args.output}")


if __name__ == "__main__":
    main()
