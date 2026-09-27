"""Offline station terrain comparison over a faithful Scene3 proxy of authored boxes.

Source geometry: studio_client/src/environments/EnvironmentScene.ts, station().
This is one actor's route feasibility, not native Core generation or crowd motion.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scene_composition import validate_scene
from scene_interaction_geometry import SceneInteractionGeometry
from scene_navigation import plan_navigation_route

SOURCE = "studio_client/src/environments/EnvironmentScene.ts:639-642"
START = (32.0, 0.0, -8.0)
GOAL = (50.0, 0.0, -8.0)
LOWER_RADIUS = .18
UPPER_RADIUS = .60


def _box_part(position=(0., 0., 0.), size=(1., 1., 1.)):
    return {"shape": "box", "position": list(position), "size": list(size), "color": [154, 159, 147]}


def station_scene():
    """Represent exact east terrace and six station step boxes in Scene3.

    The large plaza slab is an explicit local support proxy at rendered Y=0.
    It covers only this trial's corridor; it is not an invented route through a
    gap. No visual extraction or model inference is involved.
    """
    stair_parts = []
    for step in range(6):
        y = .125 + step * .125
        z = -.4 - step * .5
        h = .25 + step * .25
        stair_parts.append(_box_part(
            (0., (y-.75)/1.5, (z+1.65)/3.), (1., h/1.5, .5/3.)))
    return validate_scene({
        "version": 3, "name": "Hikari station east terrace route proxy",
        "lighting": "sunset", "effects": [],
        "assets": [
            {"id": "slab", "name": "Unit box", "parts": [_box_part()]},
            {"id": "station-stair-six", "name": "Station six rendered step boxes",
             "parts": stair_parts},
        ],
        "objects": [
            {"id": "plaza-support", "name": "Station plaza support", "kind": "custom",
             "asset": "slab", "position": [41., -.1, -6.], "size": [32., .2, 32.],
             "color": [154, 159, 147], "interaction": {"action": "none", "trigger": "none", "radius": 0}},
            {"id": "east-terrace", "name": "East raised terrace", "kind": "custom",
             "asset": "slab", "position": [41., .75, -11.], "size": [15., 1.5, 15.],
             "color": [154, 159, 147], "interaction": {"action": "none", "trigger": "none", "radius": 0}},
            {"id": "east-stairs", "name": "East station stairs", "kind": "custom",
             "asset": "station-stair-six", "position": [41., .75, -1.65],
             "size": [15., 1.5, 3.],
             "color": [154, 159, 147], "interaction": {"action": "none", "trigger": "none", "radius": 0}},
        ],
    })


def clearance_failures(geometry, points):
    """Sample support and both body envelopes on a proposed 3-D path."""
    failures = []
    for index, (x, y, z) in enumerate(points):
        support = geometry.support_height(x, z, y, max_step_up=.25, max_drop=.35)
        blocked = any(geometry.obstacle_at(
            x, y + offset, z,
            radius=LOWER_RADIUS if offset == .40 else UPPER_RADIUS)
            for offset in (.40, .90, 1.35))
        if support is None or abs(support-y) > .025 or blocked:
            failures.append({"sample": index, "xyz": [float(x), float(y), float(z)],
                             "support_y": None if support is None else float(support),
                             "body_blocked": bool(blocked)})
    return failures


def evaluate():
    scene = station_scene()
    geometry = SceneInteractionGeometry.from_scene(scene)
    direct = np.linspace(START, GOAL, 181)
    baseline_failures = clearance_failures(geometry, direct)
    route = plan_navigation_route(
        geometry, START, GOAL, radius=LOWER_RADIUS,
        upper_body_radius=UPPER_RADIUS, max_step_up=.25, max_drop=.35,
        max_expansions=16000)
    route_failures = clearance_failures(geometry, route)
    # A compact route can span a bend or riser; validate its interpolated sweep.
    swept = [route[0]]
    for a, b in zip(route[:-1], route[1:]):
        samples = max(1, math.ceil(np.linalg.norm((b-a)[[0, 2]]) / .1))
        swept.extend(a + (b-a) * (i/samples) for i in range(1, samples+1))
    sweep_failures = clearance_failures(geometry, np.asarray(swept))
    step_heights = [float(row[1]) for flight in geometry.stair_routes for row in flight["steps"]]
    # The top stair box ends at Z=-3.15, while the terrace begins at Z=-3.5.
    # Check the actual unsupported midpoint at the elevated sole height.
    stair_gap_support = geometry.support_height(41., -3.325, 1.5,
                                                  max_step_up=.25, max_drop=.35)
    return {
        "source": SOURCE,
        "scope": "one-actor CPU route feasibility; authored station geometry proxy",
        "scene3_valid": True,
        "coordinates": {"start_xyz": START, "goal_xyz": GOAL},
        "baseline": {"method": "straight line on Y=0", "sample_count": len(direct),
                     "collision_or_support_failure_count": len(baseline_failures),
                     "first_failures": baseline_failures[:3]},
        "terrain": {"method": "scene_navigation.plan_navigation_route over Scene3 rendered boxes",
                    "waypoints_xyz": route.round(4).tolist(),
                    "path_length_m": float(np.linalg.norm(np.diff(route[:, [0, 2]], axis=0), axis=1).sum()),
                    "waypoint_failures": route_failures,
                    "swept_sample_count": len(swept), "swept_failures": sweep_failures},
        "stairs": {"detected_tread_heights_m": step_heights,
                   "recognized_stair_flights": len(geometry.stair_routes),
                   "last_tread_back_z": -3.15, "terrace_front_z": -3.5,
                   "gap_m": .35, "elevated_gap_support_y": stair_gap_support,
                   "claim": "rendered station boxes are not recognized as a stair flight or certified connected to terrace"},
        "provenance": {"geometry": "hand-transcribed exact station box dimensions from source",
                       "vision_or_learned_scene_reconstruction": False,
                       "native_motion_generated": False,
                       "human17_display_solved": False,
                       "multi_actor_terrain": False},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("review/crowd-demos/terrain"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scene = station_scene()
    (args.output / "station-proxy.scene3.json").write_text(json.dumps(scene, indent=2) + "\n")
    result = evaluate()
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"baseline_failures": result["baseline"]["collision_or_support_failure_count"],
                      "terrain_sweep_failures": len(result["terrain"]["swept_failures"]),
                      "path_length_m": result["terrain"]["path_length_m"],
                      "stair_gap_support_y": result["stairs"]["elevated_gap_support_y"]},
                     allow_nan=False))


if __name__ == "__main__":
    main()
