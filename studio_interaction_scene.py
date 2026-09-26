"""Adapt the current studio scene for measured Core navigation.

The returned document preserves actual rendered geometry. Names never create
openings, and a catalog action is geometric eligibility, not motion success.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from scene_composition import validate_scene
from interaction_scene import scene_objects, passage_for, is_walkable_ground


def adapt_studio_scene(scene_document, *, object_states=None):
    """Return ``scene``, trusted ``affordances``, and an ``objects`` catalog.

    ``object_states`` may be the current output of evaluate_objects. Omit it
    for the authored layout. A built-in open door must match that evaluator's
    actual upward translation; an ``active`` label alone proves no passage.
    Arbitrary custom geometry receives no inferred passage affordance.
    """
    scene = validate_scene(scene_document)
    if object_states is not None:
        if not isinstance(object_states, (list, tuple)):
            raise ValueError("object_states must be current evaluated object states")
        originals = {obj["id"]: obj for obj in scene["objects"]}
        seen = set()
        for state in object_states:
            if not isinstance(state, Mapping) or state.get("id") not in originals or state["id"] in seen:
                raise ValueError("Object states contain duplicate or unknown IDs")
            seen.add(state["id"])
            obj = originals[state["id"]]
            position = state.get("position")
            if (not isinstance(position, (list, tuple)) or len(position) != 3
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in position)):
                raise ValueError("Object state position must contain finite world coordinates")
            if obj["kind"] == "door":
                expected = list(obj["position"])
                if state.get("active") is True:
                    expected[1] += obj["size"][1]
                if any(abs(a - b) > 1e-6 for a, b in zip(position, expected)):
                    raise ValueError("Door state does not match its rendered opening geometry")
            obj["position"] = list(position)
        if seen != set(originals):
            raise ValueError("Object states must cover the complete current scene")
        scene = validate_scene(scene)
    catalog = []
    for obj in scene_objects(scene):
        if is_walkable_ground(obj):
            continue
        actions = ["approach"]
        try:
            passage = passage_for(obj, None, actor_height_m=1.65)
            if passage.width_m >= 2 * (.28 + .06) + .1:
                actions.append("go_through")
        except ValueError:
            pass
        catalog.append({"id": obj.id, "name": obj.name, "kind": obj.kind,
                        "actions": actions})
    return {"scene": scene, "affordances": {}, "objects": catalog}


def resolve_target(adapted, text):
    """Resolve an exact stable ID or unique exact name without guessing."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Choose an object ID or exact object name")
    text = text.strip()
    objects = adapted["objects"]
    identifier = next((obj["id"] for obj in objects if obj["id"] == text), None)
    if identifier is not None:
        return identifier
    matches = [obj["id"] for obj in objects if obj["name"].casefold() == text.casefold()]
    if len(matches) > 1:
        raise ValueError("Object name is ambiguous; choose its exact ID")
    if not matches:
        raise ValueError("Unknown navigation target; choose an object from the current scene")
    return matches[0]


def recommend_placements(scene_document, count):
    """Find one or two clear ground anchors inside the Core worker bounds.

    Placement uses conservative rotated obstacle footprints. An authored floor
    limits the search to its usable footprint; no height or stepping is faked.
    """
    if type(count) is not int or not 1 <= count <= 2:
        raise ValueError("Placement supports one or two actors")
    from interaction_planner import _obstacles, _inside
    from interaction_scene import local_axes
    objects = scene_objects(validate_scene(scene_document))
    obstacles = _obstacles(objects, None, 1.65, .4)
    floors = [obj for obj in objects if is_walkable_ground(obj)]
    def on_floor(point):
        if not floors:
            return True
        for obj in floors:
            axes = local_axes(obj.yaw_degrees)
            delta = [point[0] - obj.x, point[1] - obj.z]
            local = [sum(a * b for a, b in zip(delta, axis)) for axis in axes]
            if abs(local[0]) <= obj.width / 2 - .4 and abs(local[1]) <= obj.depth / 2 - .4:
                return True
        return False
    candidates = [(x * .75, z * .75) for x in range(-32, 33) for z in range(-32, 33)]
    candidates.sort(key=lambda point: (point[0] ** 2 + point[1] ** 2, point[0], point[1]))
    chosen = []
    for point in candidates:
        if (not on_floor(point) or any(_inside(point, obstacle, .4) for obstacle in obstacles)
                or any(math.dist(point, prior) < 1.5 for prior in chosen)):
            continue
        chosen.append(point)
        if len(chosen) == count:
            return [{"position_xz": list(point), "yaw": 0.} for point in chosen]
    raise ValueError("No clear actor placement within the Core worker's ±25 m ground area")


def measure_scene_motion(positions, scene_document, *, skeleton="g1", fps=25,
                         object_states=None):
    """Measure one actor against a static snapshot; never label physics solved."""
    from interaction_scene_collision import scene_collision
    from interaction_metrics import floor_motion, continuity
    adapted = adapt_studio_scene(scene_document, object_states=object_states)
    return {"collision": scene_collision(positions, skeleton, adapted["scene"], adapted["affordances"]),
            "floor": floor_motion(positions, skeleton=skeleton, fps=fps),
            "continuity": continuity(positions, skeleton=skeleton, fps=fps)}
