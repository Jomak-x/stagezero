"""Re-evaluate one saved native take in a distinct authored background.

This is a controlled visual-background comparison. It reuses identical ARDY
native arrays and features; no second native generation is performed or claimed.
Run as ``python -m experiments.observatory_background_reuse``.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from core_scene_reactions import evaluated_scene, object_states
from core_spatial_commands import measure_completion, parse_commands, plan_command
from studio_interaction_scene import adapt_studio_scene
from terrain_assisted_session import (
    NativeTerrainResult, _native_digest, _validate_native_body,
    assist_native_terrain_result, load_native_terrain_result,
    save_assisted_result, save_native_terrain_result,
)
from traversal_kit import TEMPLE_START_ROOT_XYZ, traversable_observatory_scene


OBSERVATORY_COMMAND = ("walk up the shallow observatory steps, "
                       "cross the northern copper walkway, "
                       "open observatory door, and enter")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "review/terrain-assisted/full-route/native_terrain.npz"
DEFAULT_OUTPUT = ROOT / ".runtime/terrain-assisted/observatory"


def revalidate_observatory_background(source: NativeTerrainResult) -> NativeTerrainResult:
    """Rebuild alternate scene IDs, routes, measurements, and gate reactions."""
    if not isinstance(source, NativeTerrainResult) or source.committed_prefix is not None:
        raise ValueError("Background comparison needs a complete zero-prefix native take")
    native = source.native_clip
    if (native.actor_ids != ("actor_1",) or native.native_features is None or
            len(source.action_spans) != 4 or source.action_spans[0][0] != 0 or
            source.action_spans[-1][1] != native.frames):
        raise ValueError("Saved native take does not cover one four-action route")
    scene = traversable_observatory_scene()
    adapted = adapt_studio_scene(scene)
    adapted.update(original_scene=scene, terrain_active=True)
    actions = parse_commands(OBSERVATORY_COMMAND, adapted)
    if [item["verb"] for item in actions] != ["ascend", "cross", "open", "go_through"]:
        raise ValueError("Observatory command no longer describes the four required actions")
    placements = {"actor_1": {"position_xz": [TEMPLE_START_ROOT_XYZ[0],
                                               TEMPLE_START_ROOT_XYZ[2]],
                               "yaw": math.pi}}
    routes, measurements = [], []
    planning_heading = None
    for action, (start, end) in zip(actions, source.action_spans):
        if not (0 <= start < end <= native.frames):
            raise ValueError("Saved native action span is outside the clip")
        prefix = None if start == 0 else native.slice_frames(0, start)
        evaluated = evaluated_scene(scene, prefix, enabled=True, terrain=True)
        adapted = adapt_studio_scene(evaluated)
        adapted.update(original_scene=scene, terrain_active=True)
        if planning_heading is not None:
            adapted["terrain_planning_heading"] = planning_heading
        stages, route = plan_command(action, adapted, native.actor_ids,
                                     "actor_1", prefix, placements)
        if (not stages or route.get("terrain_navigation_version") != 1 or
                route["schedule"]["frames"] != end-start):
            raise ValueError("Alternate authored geometry changes the native action timing")
        current = native.slice_frames(0, end)
        measurement = measure_completion(route, current, start)
        measurement.update(_validate_native_body(native.slice_frames(start, end),
                                                  scene, current, start, 0))
        if action["verb"] == "open":
            gate = next(row for row in object_states(scene, current, enabled=True,
                                                      terrain=True)
                        if row["id"] == route["target_id"])
            measurement["automatic_door_open_verified"] = gate["opening_fraction"] >= .99
            measurement["completed"] &= measurement["automatic_door_open_verified"]
        if not measurement["completed"]:
            raise ValueError(f"Saved native take fails the alternate {action['verb']} geometry: {measurement}")
        routes.append(deepcopy(route))
        measurements.append(measurement)
        # Match the relative-heading state used when the native take was made.
        # It is derived from this alternate route, so later plans still test
        # the authored observatory geometry against the saved action spans.
        path = np.asarray(route["support_xyz"], dtype=float)
        for segment in np.diff(path[:, [0, 2]], axis=0)[::-1]:
            if np.linalg.norm(segment) > .05:
                planning_heading = math.atan2(float(segment[0]), float(segment[1]))
                break
    reactions = object_states(scene, native, enabled=True, terrain=True)
    gate = next(row for row in reactions if row["id"] == "observatory-gate")
    if gate["opening_fraction"] < .99 or not measurements[-1]["crossing_verified"]:
        raise ValueError("Saved motion does not open and cross the observatory gate")
    return NativeTerrainResult(scene, native, tuple(routes), source.action_spans,
                               tuple(measurements), tuple(reactions), None)


def prepare_observatory_artifacts(source_path=DEFAULT_SOURCE, output_dir=DEFAULT_OUTPUT):
    """Save background-specific native evidence with unchanged GPU motion."""
    source_path, output_dir = Path(source_path), Path(output_dir)
    source = load_native_terrain_result(source_path)
    result = revalidate_observatory_background(source)
    if _native_digest(result.native_clip) != _native_digest(source.native_clip):
        raise AssertionError("Alternate background changed native ARDY arrays")
    output_dir.mkdir(parents=True, exist_ok=True)
    native_path = output_dir / "observatory.native.npz"
    scene_path = output_dir / "observatory.scene.json"
    report_path = output_dir / "observatory.report.json"
    save_native_terrain_result(result, native_path)
    scene_path.write_text(json.dumps(result.scene, indent=2), encoding="utf-8")
    report = {"version": 1, "comparison": "same native motion in alternate authored background",
              "no_second_gpu_generation": True,
              "source_native_archive": str(source_path),
              "source_native_sha256": _native_digest(source.native_clip),
              "observatory_native_sha256": _native_digest(result.native_clip),
              "scene_name": result.scene["name"],
              "new_object_ids": [obj["id"] for obj in result.scene["objects"]],
              "action_spans": result.action_spans,
              "routes": result.routes, "measurements": result.measurements,
              "reaction_states": result.reaction_states,
              "visual_review": "pending", "accepted": False}
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return result, report


def assist_observatory_artifacts(output_dir=DEFAULT_OUTPUT):
    """Use saved native-only proof for a CPU rig17 pass, without any GPU work."""
    output_dir = Path(output_dir)
    native = load_native_terrain_result(output_dir / "observatory.native.npz")
    report_path = output_dir / "observatory.report.json"
    try:
        result = assist_native_terrain_result(native)
    except Exception as exc:
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["assisted_archive_produced"] = False
            report["assistance_error"] = str(exc)[-800:]
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False),
                                   encoding="utf-8")
        raise
    result.report["native_reuse"] = {
        "same_gpu_motion_as_temple": True, "no_second_gpu_generation": True,
        "native_sha256": _native_digest(native.native_clip)}
    save_assisted_result(result, output_dir / "proof.assisted.npz")
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.pop("assistance_error", None)
        report["assisted_archive_produced"] = True
        report["assisted_archive"] = str(output_dir / "proof.assisted.npz")
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False),
                               encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-native", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--assist", action="store_true",
                        help="Run CPU rig17 assistance after native background revalidation")
    args = parser.parse_args(argv)
    result, report = prepare_observatory_artifacts(args.source_native, args.output_dir)
    print(json.dumps({"native_frames": result.native_clip.frames,
                      "native_sha256": report["observatory_native_sha256"],
                      "no_second_gpu_generation": True,
                      "native_archive": str(args.output_dir / "observatory.native.npz")}))
    if args.assist:
        assisted = assist_observatory_artifacts(args.output_dir)
        print(json.dumps({"assisted_frames": assisted.presentation.frames,
                          "accepted": False,
                          "archive": str(args.output_dir / "proof.assisted.npz")}))


if __name__ == "__main__":
    main()
