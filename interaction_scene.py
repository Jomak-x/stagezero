"""Small, strict geometry adapter for StageZero scene documents.

Scene coordinates are metres with +Y up. A custom asset's visual mesh does
not imply a hole: a passage must be declared separately by trusted scene
metadata before a route may use it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from scene_objects import KINDS, MAX_OBJECTS
from scene_ground import has_authored_ground


@dataclass(frozen=True)
class SceneObject:
    id: str
    name: str
    kind: str
    x: float
    y: float
    z: float
    width: float
    height: float
    depth: float
    yaw_degrees: float

    @property
    def center_xz(self) -> tuple[float, float]:
        return self.x, self.z


@dataclass(frozen=True)
class Passage:
    object_id: str
    center_xz: tuple[float, float]
    yaw_degrees: float
    width_m: float
    height_m: float
    depth_m: float
    source: str


def _number(value: object, label: str, lower: float, upper: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{label} must be a finite metre/degree value in [{lower}, {upper}]")
    return float(value)


def _xz(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{label} requires [x, z] in metres")
    return (_number(value[0], label + ".x", -100, 100),
            _number(value[1], label + ".z", -100, 100))


def scene_objects(scene: Mapping) -> list[SceneObject]:
    """Read the route-relevant fields from version 2/3 scene documents."""
    if not isinstance(scene, Mapping) or type(scene.get("version")) is not int or scene["version"] not in (2, 3):
        raise ValueError("Expected a StageZero scene version 2 or 3")
    raw = scene.get("objects")
    if not isinstance(raw, list) or len(raw) > MAX_OBJECTS:
        raise ValueError(f"Scene objects must be a list of at most {MAX_OBJECTS}")
    asset_ids = set()
    if scene["version"] == 3:
        assets = scene.get("assets")
        if not isinstance(assets, list):
            raise ValueError("Scene version 3 requires assets")
        asset_ids = {asset["id"] for asset in assets if isinstance(asset, Mapping)
                     and isinstance(asset.get("id"), str)}
    result, ids = [], set()
    for index, obj in enumerate(raw):
        label = f"objects[{index}]"
        if not isinstance(obj, Mapping):
            raise ValueError(f"{label} must be an object")
        identifier, name, kind = obj.get("id"), obj.get("name"), obj.get("kind")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError(f"{label} requires a unique id")
        if not isinstance(name, str) or not name.strip() or not isinstance(kind, str):
            raise ValueError(f"{label} requires a name and kind")
        if kind not in KINDS and kind != "custom":
            raise ValueError(f"{label}.kind is unsupported")
        ids.add(identifier)
        position, size = obj.get("position"), obj.get("size")
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError(f"{label}.position requires [x, y, z] in metres")
        if not isinstance(size, (list, tuple)) or len(size) != 3:
            raise ValueError(f"{label}.size requires [width, height, depth] in metres")
        x, y, z = [_number(v, f"{label}.position[{i}]", -100, 100) for i, v in enumerate(position)]
        width, height, depth = [_number(v, f"{label}.size[{i}]", .05, 60) for i, v in enumerate(size)]
        yaw = _number(obj.get("yaw", 0), f"{label}.yaw", -360, 360)
        if kind not in ("custom", "door") and yaw != 0:
            # Door boxes and passage axes match the renderer yaw. Other
            # procedural passage shapes still require explicit rotation support.
            raise ValueError(f"{label} cannot rotate a procedural prop")
        if kind == "custom" and (scene["version"] != 3 or obj.get("asset") not in asset_ids):
            raise ValueError(f"{label} custom asset requires scene version 3 and an existing asset id")
        result.append(SceneObject(identifier, name.strip(), kind, x, y, z, width, height, depth, yaw))
    return result


def local_axes(yaw_degrees: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Local +X width and +Z travel axes under Viser's +Y quaternion yaw."""
    yaw = math.radians(yaw_degrees)
    return (math.cos(yaw), -math.sin(yaw)), (math.sin(yaw), math.cos(yaw))


def is_walkable_ground(obj: SceneObject) -> bool:
    """Exclude floor contact from horizontal navigation at the Y=0 plane.

    The studio may render higher authored slabs, but this motion planner cannot
    climb them. Only surfaces within its 10 cm floor tolerance are walkable.
    Yaw does not change a horizontal slab's height or local footprint area.
    """
    top = obj.y + obj.height / 2
    if obj.kind == "platform":
        return top <= .1
    return (abs(top) <= .1 and has_authored_ground([{
        "kind": obj.kind, "position": [obj.x, obj.y, obj.z],
        "size": [obj.width, obj.height, obj.depth]}]))


def passage_for(obj: SceneObject, affordances: Mapping | None, *, actor_height_m: float) -> Passage:
    if obj.kind == "door":
        # The built-in door lifts its entire rendered assembly straight up.
        # Its current bottom is the ceiling of the newly clear passage.
        height = obj.y - obj.height / 2
        if height < actor_height_m + .05:
            raise ValueError(f"{obj.id} door is closed or has insufficient overhead clearance")
        return Passage(obj.id, obj.center_xz, obj.yaw_degrees, obj.width, height,
                       obj.depth, "raised_procedural_door")
    if obj.kind == "arch":
        floor_y = obj.y - obj.height / 2
        # Include the wider floor bases: centres +/- .385 width, each .23 width.
        width = .54 * obj.width
        height = .81 * obj.height
        passage = Passage(obj.id, obj.center_xz, 0., width, height, .9 * obj.depth, "procedural_arch")
    else:
        raw = affordances.get(obj.id) if isinstance(affordances, Mapping) else None
        if obj.kind != "custom" or not isinstance(raw, Mapping) or raw.get("kind") != "passage" or raw.get("verified_open") is not True:
            raise ValueError(f"{obj.id} has no verified open passage; custom mesh geometry cannot be inferred")
        if set(raw) - {"kind", "verified_open", "width_m", "height_m", "depth_m", "center_xz", "yaw_degrees", "floor_y_m"}:
            raise ValueError(f"{obj.id} passage has unknown fields")
        if "floor_y_m" not in raw:
            raise ValueError(f"{obj.id} custom passage requires floor_y_m")
        floor_y = _number(raw["floor_y_m"], "passage.floor_y_m", -100, 100)
        if floor_y < obj.y - obj.height / 2 - .1 or floor_y > obj.y + obj.height / 2 + .1:
            raise ValueError(f"{obj.id} passage floor lies outside the object")
        width = _number(raw.get("width_m"), "passage.width_m", .05, obj.width)
        height = _number(raw.get("height_m"), "passage.height_m", .05, obj.height)
        if floor_y + height > obj.y + obj.height / 2 + .1:
            raise ValueError(f"{obj.id} passage height extends outside the object")
        depth = _number(raw.get("depth_m", obj.depth), "passage.depth_m", .05, obj.depth)
        center = _xz(raw.get("center_xz", obj.center_xz), "passage.center_xz")
        yaw = _number(raw.get("yaw_degrees", obj.yaw_degrees), "passage.yaw_degrees", -360, 360)
        # Off-centre openings must still lie within their containing prop.
        width_axis, normal_axis = local_axes(obj.yaw_degrees)
        delta = (center[0] - obj.x, center[1] - obj.z)
        if abs(delta[0] * width_axis[0] + delta[1] * width_axis[1]) + width / 2 > obj.width / 2 + 1e-6 or \
           abs(delta[0] * normal_axis[0] + delta[1] * normal_axis[1]) + depth / 2 > obj.depth / 2 + 1e-6:
            raise ValueError(f"{obj.id} passage lies outside the object bounds")
        if abs(math.sin(math.radians(yaw - obj.yaw_degrees))) > 1e-6:
            raise ValueError(f"{obj.id} passage yaw disagrees with its containing object")
        passage = Passage(obj.id, center, yaw, width, height, depth, "explicit_affordance")
    if passage.height_m < actor_height_m + .05:
        raise ValueError(f"{obj.id} passage is too low for the actor")
    if abs(floor_y) > .1:
        raise ValueError(f"{obj.id} passage is not grounded on the walkable Y=0 plane")
    return passage
