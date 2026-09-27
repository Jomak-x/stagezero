"""Reusable industrial switchback scene with real, connected Scene3 support.

The route ascends toward -Z, bends across a +X gantry, passes an automatic
raised door, and descends a separate flight toward -Z.  No invisible floor or
route ramp is added to bridge the gap.
"""
from __future__ import annotations

import numpy as np


START_ROOT_XYZ = (0., .95, 2.)
START_YAW_RAD = float(np.pi)
LOADING_SUPPORT_Y_M = .20
APPROACH_RISE_M = .04
APPROACH_TREAD_RUN_M = .50
EXIT_RISE_M = .04
EXIT_TREAD_RUN_M = .75
STEP_COUNT = 5
GANTRY_NEAR_X = 4.
GANTRY_FAR_X = 8.
DOOR_X = 10.
EXIT_GROUND_Z = -11.75


def _asset_box(identifier, name, color):
    return {"id": identifier, "name": name,
            "parts": [{"shape": "box", "position": [0., 0., 0.],
                       "size": [1., 1., 1.], "color": list(color)}]}


def _flight_asset(identifier, name, top_heights, bottom, run, color):
    """Normalize filled treads to a unit asset; object size sets world units."""
    span = max(top_heights)-bottom
    parts = []
    for index, top in enumerate(top_heights):
        raw_top = min(.5, -.5+(top-bottom)/span)
        parts.append({"shape": "box",
                      "position": [0., (raw_top-.5)/2., .4-index/STEP_COUNT],
                      "size": [1., raw_top+.5, 1./STEP_COUNT],
                      "color": list(color)})
    return {"id": identifier, "name": name, "parts": parts}


def _slab(identifier, name, asset, position, size, color):
    return {"id": identifier, "name": name, "kind": "custom",
            "asset": asset, "position": list(position), "size": list(size),
            "color": list(color),
            "interaction": {"action": "none", "trigger": "none", "radius": 0.}}


def industrial_switchback_scene():
    """An authored route with two stair flights and a lateral gantry crossing."""
    concrete = _asset_box('yard-concrete', 'Concrete', (116, 125, 126))
    steel = _asset_box('yard-steel', 'Weathered steel', (78, 101, 111))
    chasm = _asset_box('yard-abyss', 'Dark ravine', (39, 53, 60))
    approach_tops = [(i+1)*APPROACH_RISE_M for i in range(STEP_COUNT)]
    exit_tops = [(STEP_COUNT-i-1)*EXIT_RISE_M for i in range(STEP_COUNT)]
    approach = _flight_asset('yard-approach-flight', 'Filled approach treads',
                             approach_tops, -.10, APPROACH_TREAD_RUN_M,
                             (135, 133, 121))
    descending = _flight_asset('yard-exit-flight', 'Filled exit treads',
                               exit_tops, -.10, EXIT_TREAD_RUN_M,
                               (129, 132, 125))
    objects = [
        _slab('yard-forecourt', 'Broad factory forecourt', 'yard-concrete',
              (0., -.10, 2.), (6., .20, 4.), (116, 125, 126)),
        _slab('yard-approach-stairs', 'Foundry approach stairs', 'yard-approach-flight',
              (0., .05, -1.25), (2., .30, 2.5), (135, 133, 121)),
        _slab('yard-turning-deck', 'Broad turning loading platform', 'yard-concrete',
              (.75, .10, -5.), (6.5, .20, 5.), (116, 125, 126)),
        _slab('yard-gantry', 'Service gantry bridge', 'yard-steel',
              (6., .10, -5.), (4., .20, 2.2), (78, 101, 111)),
        _slab('yard-far-deck', 'Workshop loading platform', 'yard-concrete',
              (10., .10, -5.25), (4., .20, 5.5), (116, 125, 126)),
        {"id": "yard-workshop-door", "name": "Workshop door", "kind": "door",
         "position": [DOOR_X, 1.30, -5.], "size": [1.8, 2.2, .18],
         "yaw": 90., "color": [118, 78, 57],
         "interaction": {"action": "open", "trigger": "proximity", "radius": 1.4}},
        _slab('yard-exit-stairs', 'Loading exit steps', 'yard-exit-flight',
              (10.5, .03, -9.875), (2., .26, STEP_COUNT*EXIT_TREAD_RUN_M),
              (129, 132, 125)),
        _slab('yard-exit-ground', 'Lower freight yard', 'yard-concrete',
              (10.5, -.10, -13.75), (5., .20, 4.), (116, 125, 126)),
        _slab('yard-chasm', 'Deep industrial ravine', 'yard-abyss',
              (6., -2.25, -5.), (7.5, 3.5, 5.5), (39, 53, 60)),
        {"id": "yard-forklift-crate", "name": "Forklift cargo crate", "kind": "crate",
         "position": [1.65, .75, -3.15], "size": [1.2, 1.1, 1.0],
         "color": [169, 114, 60],
         "interaction": {"action": "none", "trigger": "none", "radius": 0.}},
        {"id": "yard-stack", "name": "Crate stack", "kind": "crate",
         "position": [-3.6, .75, -5.5], "size": [1.2, 1.1, 1.1],
         "color": [132, 85, 48],
         "interaction": {"action": "none", "trigger": "none", "radius": 0.}},
        _slab('yard-header', 'Workshop steel header', 'yard-steel',
              (10., 3.12, -5.), (.30, .25, 4.5), (78, 101, 111)),
    ]
    return {"version": 3, "name": "Industrial switchback courtyard",
            "objects": objects, "assets": [concrete, steel, chasm, approach, descending],
            "effects": [], "lighting": "sunset",
            "camera": {"position": [-2., 8., 11.], "look_at": [6., 1., -6.]},
            "targets": []}


def switchback_route_metadata():
    return {"start_root_xyz": list(START_ROOT_XYZ), "start_yaw_rad": START_YAW_RAD,
            "approach_rise_m": APPROACH_RISE_M,
            "approach_tread_run_m": APPROACH_TREAD_RUN_M,
            "exit_rise_m": EXIT_RISE_M,
            "exit_tread_run_m": EXIT_TREAD_RUN_M,
            "step_count_each_flight": STEP_COUNT,
            "raised_support_y_m": LOADING_SUPPORT_Y_M,
            "gantry_near_x": GANTRY_NEAR_X, "gantry_far_x": GANTRY_FAR_X,
            "door_x": DOOR_X, "exit_ground_z": EXIT_GROUND_Z,
            "ordered_commands": [
                "walk up foundry approach stairs, cross service gantry bridge",
                "open workshop door, enter",
                "walk down loading exit steps"],
            "ids": {"approach_stairs": "yard-approach-stairs",
                    "gantry": "yard-gantry", "door": "yard-workshop-door",
                    "exit_stairs": "yard-exit-stairs"}}


def main():
    """Write a validated Scene3 JSON for app import or a CPU trial."""
    import argparse
    import json
    from pathlib import Path
    from scene_composition import validate_scene

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    scene = validate_scene(industrial_switchback_scene())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scene, indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
