"""Small, scene-independent procedural traversal fixture for native Core trials.

Every walkable surface below is a rendered box.  The smoothed pelvis target is
only a conditioning trajectory; it never adds a ramp to the Scene3 scene.
"""
from __future__ import annotations

import numpy as np

FPS = 20
FRAMES = 120
RISE_M = .04
TREAD_RUN_M = .50
STEP_COUNT = 5
WIDTH_M = 2.0
START_Z = .85
END_Z = -3.05
PELVIS_CLEARANCE_M = .95
TEMPLE_BRIDGE_NEAR_Z = -2.5
TEMPLE_BRIDGE_FAR_Z = -7.5
TEMPLE_GATE_Z = -9.3
TEMPLE_INTERIOR_FAR_Z = -11.5
TEMPLE_START_ROOT_XYZ = (0., PELVIS_CLEARANCE_M, START_Z)


def _box(object_id: str, name: str, *, z_near: float, z_far: float, top: float) -> dict:
    """A filled box with its top at ``top``; adjoining faces have no gaps."""
    bottom = -.10
    return {
        "id": object_id, "name": name, "kind": "custom",
        "asset": "traversal-solid-box",
        "position": [0., (bottom + top) / 2., (z_near + z_far) / 2.],
        "size": [WIDTH_M, top - bottom, z_near - z_far],
        "color": [255, 255, 255],
        "interaction": {"action": "none", "trigger": "none", "radius": 0.},
    }


def shallow_stair_scene() -> dict:
    """Scene3 boxes: 1 m entry, five .5 m treads, .75 m exit."""
    objects = [_box("entry", "Level entry landing", z_near=1., z_far=0., top=0.)]
    for step in range(STEP_COUNT):
        objects.append(_box(f"step-{step + 1}", f"Riser {step + 1}",
                            z_near=-step*TREAD_RUN_M,
                            z_far=-(step+1)*TREAD_RUN_M,
                            top=(step+1)*RISE_M))
    objects.append(_box("exit", "Level exit landing", z_near=-2.5,
                        z_far=-3.25, top=STEP_COUNT*RISE_M))
    return {
        "version": 3, "name": "Shallow stair traversal feasibility fixture",
        "objects": objects,
        "effects": [], "lighting": "neutral",
        "assets": [{"id": "traversal-solid-box", "name": "Solid box",
                    "parts": [{"shape": "box", "position": [0., 0., 0.],
                               "size": [1., 1., 1.], "color": [151, 147, 119]}]}],
        "camera": {"position": [5., 2.8, 4.], "look_at": [0., 1., -1.2]},
        "targets": [],
    }


def traversable_temple_scene() -> dict:
    """Connected temple approach with actual treads, bridge, and raised gate.

    All support is rendered geometry.  The broad forecourt disables the studio's
    implicit y=0 floor, leaving a real gap beside and beneath the bridge.
    """
    stair_parts = []
    for index in range(STEP_COUNT):
        top = (index+1)*RISE_M
        # Asset part coordinates are normalized to a unit bounding box.
        # The object scales that box to 2 x .30 x 2.5 m in world space.
        raw_top = min(.5, -.5+(top+.10)/.30)
        stair_parts.append({"shape": "box",
                            "position": [0., (raw_top-.5)/2., .4-index/STEP_COUNT],
                            "size": [1., raw_top+.5, 1./STEP_COUNT],
                            "color": [152+index*3, 139+index*3, 111+index*3]})
    stone = {"id": "temple-solid-stone", "name": "Stone slab",
             "parts": [{"shape": "box", "position": [0., 0., 0.],
                        "size": [1., 1., 1.], "color": [143, 137, 115]}]}
    wood = {"id": "temple-solid-wood", "name": "Timber deck",
            "parts": [{"shape": "box", "position": [0., 0., 0.],
                       "size": [1., 1., 1.], "color": [116, 84, 54]}]}
    stairs = {"id": "temple-shallow-flight", "name": "Five shallow stone stairs",
              "parts": stair_parts}

    def slab(identifier, name, asset, position, size, color):
        return {"id": identifier, "name": name, "kind": "custom", "asset": asset,
                "position": list(position), "size": list(size), "color": list(color),
                "interaction": {"action": "none", "trigger": "none", "radius": 0.}}

    objects = [
        slab("temple-forecourt", "Broad stone forecourt", "temple-solid-stone",
             (0., -.10, 2.5), (6., .20, 4.), (143, 137, 115)),
        slab("temple-entry", "Level stair entry", "temple-solid-stone",
             (0., -.05, .25), (WIDTH_M, .10, .50), (151, 145, 120)),
        slab("temple-stairs", "Shallow temple stairs", "temple-shallow-flight",
             (0., .05, -1.25), (WIDTH_M, .30, 2.5), (156, 143, 116)),
        slab("temple-bridge", "Suspended temple bridge", "temple-solid-wood",
             (0., .10, -5.), (WIDTH_M, .20, 5.), (116, 84, 54)),
        slab("temple-far-landing", "Far bridge landing and temple floor", "temple-solid-stone",
             (0., .10, -9.5), (WIDTH_M, .20, 4.), (143, 137, 115)),
        {"id": "temple-gate", "name": "Temple gate", "kind": "door",
         "position": [0., 1.3, TEMPLE_GATE_Z], "size": [1.5, 2.2, .18],
         "color": [124, 85, 56],
         "interaction": {"action": "open", "trigger": "proximity", "radius": 1.4}},
        slab("temple-chasm", "Deep ravine abyss", "temple-solid-stone",
             (0., -2.25, -5.), (7., 3.5, 5.), (55, 59, 50)),
        slab("temple-lintel", "Carved temple lintel", "temple-solid-stone",
             (0., 3.15, TEMPLE_GATE_Z), (5.2, .28, .55), (167, 156, 126)),
    ]
    for side in (-1, 1):
        objects.append({"id": f"temple-column-{'left' if side < 0 else 'right'}",
                        "name": "Carved temple column", "kind": "pillar",
                        "position": [side*2.25, 1.65, TEMPLE_GATE_Z],
                        "size": [.55, 3.3, .55], "color": [158, 149, 124],
                        "interaction": {"action": "none", "trigger": "none", "radius": 0.}})
    return {"version": 3, "name": "Shallow temple bridge crossing",
            "objects": objects, "effects": [], "lighting": "warm",
            "assets": [stone, wood, stairs],
            "camera": {"position": [8., 5., 5.], "look_at": [0., 1., -6.]},
            "targets": []}


def traversable_observatory_scene() -> dict:
    """A wider copper background with distinct object and asset identifiers."""
    from copy import deepcopy
    from scene_composition import validate_scene

    scene = deepcopy(traversable_temple_scene())
    scene["name"] = "Copper observatory crossing"
    asset_ids = {}
    for asset in scene["assets"]:
        old = asset["id"]
        asset["id"] = old.replace("temple-", "observatory-")
        asset_ids[old] = asset["id"]
        for part in asset["parts"]:
            part["color"] = [163, 117, 82]
    names = {"temple-stairs": "Shallow observatory steps",
             "temple-bridge": "Northern copper walkway",
             "temple-gate": "Observatory door"}
    for obj in scene["objects"]:
        old = obj["id"]
        obj["id"] = old.replace("temple-", "observatory-")
        obj["name"] = names.get(old, obj["name"])
        if obj["kind"] == "custom":
            obj["asset"] = asset_ids[obj["asset"]]
        if old in ("temple-forecourt", "temple-entry", "temple-stairs",
                   "temple-bridge", "temple-far-landing"):
            obj["size"][0] *= 1.2
        if old == "temple-gate":
            obj["size"][0] = 1.7
    return validate_scene(scene)


def temple_route_metadata() -> dict:
    return {"start_root_xyz": list(TEMPLE_START_ROOT_XYZ),
            "stair_risers": STEP_COUNT, "stair_rise_m": RISE_M,
            "stair_tread_run_m": TREAD_RUN_M, "path_width_m": WIDTH_M,
            "bridge_near_z": TEMPLE_BRIDGE_NEAR_Z,
            "bridge_far_z": TEMPLE_BRIDGE_FAR_Z,
            "bridge_and_interior_support_y_m": STEP_COUNT*RISE_M,
            "gate_z": TEMPLE_GATE_Z,
            "interior_far_z": TEMPLE_INTERIOR_FAR_Z,
            "scene_object_ids": {"stairs": "temple-stairs", "bridge": "temple-bridge",
                                 "door": "temple-gate"}}


def selftest() -> dict:
    """Validate the actual Scene3 schema and its rendered support surfaces."""
    from scene_composition import validate_scene
    from scene_interaction_geometry import SceneInteractionGeometry

    scene = validate_scene(shallow_stair_scene())
    geometry = SceneInteractionGeometry.from_scene(scene)
    for height, z in ((0., .5), (.04, -.25), (.08, -.75),
                      (.12, -1.25), (.16, -1.75), (.20, -2.25),
                      (.20, -2.85)):
        got = geometry.support_height(0., z, height, max_step_up=.1, max_drop=.1)
        if got is None or abs(got-height) > 1e-5:
            raise AssertionError(f'rendered support mismatch at z={z}: {got} != {height}')
    return scene


def _smoothstep(value: float) -> float:
    t = float(np.clip(value, 0., 1.))
    return t*t*(3. - 2.*t)


def travel_distance(time_s: float) -> float:
    """1.2 m/s cruise with .35 s ease at either end; then hold the exit."""
    length = START_Z - END_Z
    cruise = 1.2
    ease = .35
    duration = length/cruise + ease
    t = float(np.clip(time_s, 0., duration))
    if t < ease:
        return cruise * ease * (t/ease)**2 / 2.
    if t > duration-ease:
        remaining = duration-t
        return length - cruise * remaining**2 / (2.*ease)
    return cruise*(t-ease/2.)


def target_surface_y(z: float) -> float:
    """Smoothed *pelvis target* across risers; only boxes define contact."""
    return RISE_M * sum(_smoothstep(((-z) - i*TREAD_RUN_M + .15)/.30)
                        for i in range(STEP_COUNT))


def planned_root(frame: int) -> np.ndarray:
    z = START_Z - travel_distance((frame+1)/FPS)
    return np.asarray([0., target_surface_y(z)+PELVIS_CLEARANCE_M, z], dtype=float)


def route_metadata() -> dict:
    return {"fps": FPS, "generated_frames": FRAMES, "width_m": WIDTH_M,
            "step_count": STEP_COUNT, "rise_m": RISE_M, "tread_run_m": TREAD_RUN_M,
            "entry_landing_m": 1., "exit_landing_m": .75,
            "start_z": START_Z, "end_z": END_Z, "distance_m": START_Z-END_Z,
            "cruise_m_s": 1.2, "ease_seconds": .35,
            "pelvis_clearance_m": PELVIS_CLEARANCE_M}
