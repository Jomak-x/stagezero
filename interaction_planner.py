"""CPU-only routes for canonical scene actions against StageZero prop metadata.

`plan_action(action, scene, actor_position, *, affordances=None)` returns a
JSON-safe plan. Waypoints are world X/Z metres, times are scene seconds. The
plan is a goal proposal for the motion model; it does not assert that generated
feet or hands have followed the path. Callers must measure generated motion.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Mapping

from interaction_scene import Passage, SceneObject, _number, local_axes, passage_for, scene_objects, is_walkable_ground


Vec2 = tuple[float, float]


def _add(a: Vec2, b: Vec2, scale: float = 1.) -> Vec2:
    return a[0] + b[0] * scale, a[1] + b[1] * scale


def _dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _distance(a: Vec2, b: Vec2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(frozen=True)
class _Obstacle:
    object_id: str
    center: Vec2
    half_width: float
    half_depth: float
    yaw_degrees: float


def _relative(p: Vec2, obstacle: _Obstacle) -> Vec2:
    width, normal = local_axes(obstacle.yaw_degrees)
    delta = (p[0] - obstacle.center[0], p[1] - obstacle.center[1])
    return _dot(delta, width), _dot(delta, normal)


def _inside(p: Vec2, obstacle: _Obstacle, pad: float) -> bool:
    u, v = _relative(p, obstacle)
    return abs(u) < obstacle.half_width + pad - 1e-8 and abs(v) < obstacle.half_depth + pad - 1e-8


def _intersects(a: Vec2, b: Vec2, obstacle: _Obstacle, pad: float) -> bool:
    start, end = _relative(a, obstacle), _relative(b, obstacle)
    t0, t1 = 0., 1.
    for axis, limit in ((0, obstacle.half_width + pad), (1, obstacle.half_depth + pad)):
        d = end[axis] - start[axis]
        if abs(d) < 1e-12:
            if abs(start[axis]) < limit - 1e-8:
                continue
            return False
        lower, upper = (-limit - start[axis]) / d, (limit - start[axis]) / d
        if lower > upper:
            lower, upper = upper, lower
        t0, t1 = max(t0, lower), min(t1, upper)
        if t0 >= t1 - 1e-8:
            return False
    return t0 < t1 - 1e-8


def _corners(obstacle: _Obstacle, pad: float) -> list[Vec2]:
    width, normal = local_axes(obstacle.yaw_degrees)
    w, d = obstacle.half_width + pad + .025, obstacle.half_depth + pad + .025
    return [_add(_add(obstacle.center, width, u * w), normal, v * d)
            for u in (-1, 1) for v in (-1, 1)]


def _passage_obstacles(obj: SceneObject, passage: Passage) -> list[_Obstacle]:
    if obj.kind == "arch":
        width_axis, _ = local_axes(0.)
        return [_Obstacle(obj.id, _add(obj.center_xz, width_axis, side * .385 * obj.width),
                          .115 * obj.width, obj.depth / 2, 0.) for side in (-1, 1)]
    width_axis, _ = local_axes(obj.yaw_degrees)
    rel = (passage.center_xz[0] - obj.x, passage.center_xz[1] - obj.z)
    x = _dot(rel, width_axis)
    left, right = x - passage.width_m / 2, x + passage.width_m / 2
    result = []
    for lo, hi in ((-obj.width / 2, left), (right, obj.width / 2)):
        if hi - lo > 1e-6:
            result.append(_Obstacle(obj.id, _add(obj.center_xz, width_axis, (lo + hi) / 2),
                                    (hi - lo) / 2, obj.depth / 2, obj.yaw_degrees))
    return result


def _obstacles(objects: list[SceneObject], passage: Passage | None,
               actor_height: float, pad: float) -> list[_Obstacle]:
    result = []
    for obj in objects:
        if is_walkable_ground(obj) or obj.y - obj.height / 2 >= actor_height:
            continue
        if passage is not None and obj.id == passage.object_id:
            result.extend(_passage_obstacles(obj, passage))
        elif obj.kind == "arch":
            try:
                other_passage = passage_for(obj, None, actor_height_m=actor_height)
            except ValueError:
                other_passage = None
            if other_passage is not None and other_passage.width_m >= 2 * pad + .1:
                result.extend(_passage_obstacles(obj, other_passage))
            else:
                result.append(_Obstacle(obj.id, obj.center_xz, obj.width / 2,
                                        obj.depth / 2, 0.))
        else:
            result.append(_Obstacle(obj.id, obj.center_xz, obj.width / 2,
                                    obj.depth / 2, obj.yaw_degrees))
    return result


def _path(start: Vec2, goal: Vec2, obstacles: list[_Obstacle], pad: float,
          *, side: tuple[Vec2, Vec2, float] | None = None) -> list[Vec2]:
    def allowed(p: Vec2) -> bool:
        if side is None:
            return True
        center, normal, minimum = side
        return _dot((p[0] - center[0], p[1] - center[1]), normal) >= minimum - 1e-7

    if any(_inside(p, box, pad) for p in (start, goal) for box in obstacles):
        raise ValueError("Actor or route target overlaps scene geometry")
    if not allowed(start) or not allowed(goal):
        raise ValueError("Actor is already inside or across the requested passage")
    if all(not _intersects(start, goal, box, pad) for box in obstacles):
        return [start, goal]
    points = [start, goal]
    for box in obstacles:
        points.extend(corner for corner in _corners(box, pad)
                      if allowed(corner) and not any(_inside(corner, other, pad) for other in obstacles))
    links: list[list[tuple[int, float]]] = [[] for _ in points]
    for i, a in enumerate(points):
        for j in range(i + 1, len(points)):
            b = points[j]
            if (allowed(a) and allowed(b) and
                all(not _intersects(a, b, box, pad) for box in obstacles)):
                length = _distance(a, b)
                links[i].append((j, length))
                links[j].append((i, length))
    queue = [(0., 0)]
    distance, previous = {0: 0.}, {}
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distance[node]:
            continue
        if node == 1:
            chain = [1]
            while chain[-1] != 0:
                chain.append(previous[chain[-1]])
            return [points[i] for i in reversed(chain)]
        for neighbor, step in links[node]:
            candidate = cost + step
            if candidate + 1e-9 < distance.get(neighbor, math.inf):
                distance[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
    raise ValueError("No collision-free route to the target")


def _target(action: Mapping, objects: list[SceneObject]) -> SceneObject:
    identifier, name = action.get("target_id"), action.get("target_name")
    if identifier is None and name is None:
        raise ValueError("Action requires target_id or target_name")
    if identifier is not None and (not isinstance(identifier, str) or not identifier):
        raise ValueError("target_id must be a nonempty string")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError("target_name must be a nonempty string")
    if identifier is not None:
        match = next((obj for obj in objects if obj.id == identifier), None)
        if match is None:
            raise ValueError(f"Unknown target_id: {identifier}")
        if name is not None and name.strip().casefold() != match.name.casefold():
            raise ValueError("target_id and target_name disagree")
        return match
    normalized = name.strip().casefold()
    matches = [obj for obj in objects if obj.name.casefold() == normalized or obj.kind == normalized]
    if len(matches) != 1:
        raise ValueError(f"Target name {name!r} matched {len(matches)} objects; supply target_id")
    return matches[0]


def _waypoints(points: list[Vec2], roles: list[str], start_seconds: float, speed_mps: float) -> list[dict]:
    elapsed = start_seconds
    result = []
    for i, (point, role) in enumerate(zip(points, roles)):
        if i:
            elapsed += _distance(points[i - 1], point) / speed_mps
        result.append({"time_seconds": round(elapsed, 4),
                       "position_xz": [round(v, 5) for v in point], "role": role})
    return result


def plan_action(action: Mapping, scene: Mapping, actor_position: object, *,
                affordances: Mapping | None = None, actor_radius_m: float = .28,
                actor_height_m: float = 1.65, speed_mps: float = 1.) -> dict:
    """Plan `go_through` or `approach` using actual scene objects and IDs.

    `actor_position` is a ground anchor [x,y,z], not a pelvis joint. Vertical
    navigation is not implemented; its Y must be within 10 cm of the Y=0 floor.
    `affordances` is trusted geometry metadata keyed by object ID. Custom gate
    assets need `{kind:'passage', verified_open:true, width_m, height_m,
    floor_y_m, depth_m?, center_xz?, yaw_degrees?}`. The planner never derives a hole from
    a custom mesh bounding box or claims visual object recognition.
    """
    if not isinstance(action, Mapping) or set(action) - {"verb", "actor_id", "target_id", "target_name", "start_seconds"}:
        raise ValueError("Action has missing or unknown fields")
    verb, actor_id = action.get("verb"), action.get("actor_id")
    if verb not in ("go_through", "approach") or not isinstance(actor_id, str) or not actor_id.strip():
        raise ValueError("Action requires verb go_through/approach and actor_id")
    if not isinstance(actor_position, (tuple, list)) or len(actor_position) != 3:
        raise ValueError("actor_position requires world [x, y, z] in metres")
    start_xyz = [_number(v, f"actor_position[{i}]", -100, 100) for i, v in enumerate(actor_position)]
    if abs(start_xyz[1]) > .1:
        raise ValueError("actor_position must be a ground anchor near Y=0; vertical routes are unsupported")
    start: Vec2 = start_xyz[0], start_xyz[2]
    start_seconds = _number(action.get("start_seconds", 0), "start_seconds", 0, 3600)
    radius = _number(actor_radius_m, "actor_radius_m", .15, .75)
    height = _number(actor_height_m, "actor_height_m", 1., 2.5)
    speed = _number(speed_mps, "speed_mps", .2, 3.)
    objects = scene_objects(scene)
    obj = _target(action, objects)
    pad = radius + .06
    passage = passage_for(obj, affordances, actor_height_m=height) if verb == "go_through" else None
    obstacles = _obstacles(objects, passage, height, pad)
    if passage is not None:
        if passage.width_m < 2 * pad + .1:
            raise ValueError(f"{obj.id} passage is too narrow for the actor clearance")
        _, normal = local_axes(passage.yaw_degrees)
        origin = passage.center_xz
        direction = _dot((start[0] - origin[0], start[1] - origin[1]), normal)
        if abs(direction) < passage.depth_m / 2 + pad:
            raise ValueError("Actor starts within the passage or cannot determine approach side")
        side_sign = 1. if direction > 0 else -1.
        along = (normal[0] * side_sign, normal[1] * side_sign)
        reach = passage.depth_m / 2 + pad + .25
        entry, exit_point = _add(origin, along, reach), _add(origin, along, -reach)
        approach = _path(start, entry, obstacles, pad,
                         side=(origin, along, passage.depth_m / 2 + pad))
        crossing = [entry, origin, exit_point]
        if any(_intersects(a, b, box, pad) for a, b in zip(crossing, crossing[1:]) for box in obstacles):
            raise ValueError(f"{obj.id} passage is blocked")
        points = approach + [origin, exit_point]
        roles = ["start"] + ["route"] * (len(approach) - 2) + ["entry", "center", "exit"]
        clearance = round(passage.width_m / 2 - pad, 4)
        geometry = {"passage_width_m": passage.width_m, "passage_height_m": passage.height_m,
                    "passage_depth_m": passage.depth_m, "yaw_degrees": passage.yaw_degrees,
                    "minimum_lateral_clearance_m": clearance, "source": passage.source}
    else:
        # Candidates lie past the inflated rectangular footprint on all four faces.
        width, normal = local_axes(obj.yaw_degrees)
        reach = .35
        candidates = [_add(obj.center_xz, width, sign * (obj.width / 2 + pad + reach)) for sign in (-1, 1)]
        candidates += [_add(obj.center_xz, normal, sign * (obj.depth / 2 + pad + reach)) for sign in (-1, 1)]
        choices = []
        for candidate in candidates:
            try:
                route = _path(start, candidate, obstacles, pad)
            except ValueError:
                continue
            choices.append((sum(_distance(a, b) for a, b in zip(route, route[1:])), route))
        if not choices:
            raise ValueError(f"No collision-free approach to {obj.id}")
        points = min(choices, key=lambda item: item[0])[1]
        roles = ["start"] + ["route"] * (len(points) - 2) + ["approach"]
        geometry = {"target_center_xz": [obj.x, obj.z], "target_size_xz_m": [obj.width, obj.depth],
                    "yaw_degrees": obj.yaw_degrees}
    waypoints = _waypoints(points, roles, start_seconds, speed)
    if waypoints[-1]["time_seconds"] - start_seconds > 60:
        raise ValueError("Route exceeds the 60-second planning horizon")
    return {"version": 1, "actor_id": actor_id.strip(), "verb": verb, "target_id": obj.id,
            "waypoints": waypoints, "geometry": geometry,
            "assumptions": {"units": "metres", "up_axis": "+Y", "ground_plane": "XZ",
                            "actor_radius_m": radius, "actor_height_m": height,
                            "obstacle_margin_m": .06, "speed_mps": speed,
                            "scene_source": "object_metadata", "motion_following_verified": False}}
