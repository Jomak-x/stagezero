"""Turn a verified scene route into bounded ARDY root-conditioning stages.

Only model targets are produced. This module never edits generated poses or
asserts that the resulting motion followed the planned route.
"""

from __future__ import annotations

import math
from typing import Mapping

from interaction_planner import plan_action
from interaction_scene import scene_objects
from realtime_clip import CanonicalClip
from realtime_director import HORIZON, StageSpec


FPS = 20
MAX_SECONDS = 30
HOLD_FRAMES = HORIZON
SAMPLE_FRAMES = (7, 15, 23, 31, 39)
HELD_ACTOR_CLEARANCE_M = .65


def _finite(value: object, label: str, *, limit: float = 100) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > limit:
        raise ValueError(f"{label} must be a finite number within ±{limit:g}")
    return float(value)


def _placements(actor_ids: tuple[str, ...], last_clip: CanonicalClip | None,
                initial_placements: Mapping | None) -> tuple[dict, dict[str, float]]:
    if last_clip is not None:
        if not isinstance(last_clip, CanonicalClip) or last_clip.actor_ids != actor_ids:
            raise ValueError("last clip must use the stable actor IDs in order")
        positions = {}
        yaws = {}
        for index, actor_id in enumerate(actor_ids):
            root = last_clip.positions[index, -1, 0]
            forward = last_clip.rotations[index, -1, 0, :, 2]
            positions[actor_id] = [float(root[0]), float(root[2])]
            yaws[actor_id] = math.atan2(float(forward[0]), float(forward[2]))
        return positions, yaws
    if not isinstance(initial_placements, Mapping) or set(initial_placements) != set(actor_ids):
        raise ValueError("initial_placements must cover all stable actor IDs")
    positions = {}
    yaws = {}
    for actor_id in actor_ids:
        item = initial_placements[actor_id]
        if not isinstance(item, Mapping) or set(item) - {"position_xz", "yaw"} or "position_xz" not in item:
            raise ValueError(f"{actor_id} needs position_xz and optional yaw")
        xz = item["position_xz"]
        if not isinstance(xz, (tuple, list)) or len(xz) != 2:
            raise ValueError(f"{actor_id}.position_xz must be [x, z]")
        positions[actor_id] = [_finite(xz[0], f"{actor_id}.x"),
                               _finite(xz[1], f"{actor_id}.z")]
        yaws[actor_id] = _finite(item.get("yaw", 0), f"{actor_id}.yaw", limit=math.pi)
    return positions, yaws


def _point_at(waypoints: list[dict], second: float) -> tuple[float, float]:
    if second <= waypoints[0]["time_seconds"]:
        return tuple(waypoints[0]["position_xz"])
    for before, after in zip(waypoints, waypoints[1:]):
        if second <= after["time_seconds"]:
            span = after["time_seconds"] - before["time_seconds"]
            alpha = 1. if span <= 1e-8 else (second - before["time_seconds"]) / span
            a, b = before["position_xz"], after["position_xz"]
            return (a[0] + alpha * (b[0] - a[0]),
                    a[1] + alpha * (b[1] - a[1]))
    return tuple(waypoints[-1]["position_xz"])


def _arrival_heading(waypoints: list[dict], fallback: float) -> float:
    for before, after in reversed(list(zip(waypoints, waypoints[1:]))):
        dx = after["position_xz"][0] - before["position_xz"][0]
        dz = after["position_xz"][1] - before["position_xz"][1]
        if math.hypot(dx, dz) > 1e-6:
            return math.atan2(dx, dz)
    return fallback


def _heading_at(waypoints: list[dict], second: float, fallback: float,
                arrival: float) -> float:
    if second >= waypoints[-1]["time_seconds"]:
        return arrival
    before = _point_at(waypoints, max(0, second - .1))
    after = _point_at(waypoints, min(waypoints[-1]["time_seconds"], second + .1))
    dx, dz = after[0] - before[0], after[1] - before[1]
    return math.atan2(dx, dz) if math.hypot(dx, dz) > 1e-6 else fallback


def _planning_scene(scene: Mapping, actor_ids: tuple[str, ...], actor_id: str,
                    positions: dict[str, list[float]]) -> tuple[dict, str | None]:
    """Add the stationary actor to route geometry without mutating the scene."""
    other = next((aid for aid in actor_ids if aid != actor_id), None)
    if other is None:
        return dict(scene), None
    objects = list(scene["objects"])
    if len(objects) == 40:
        # The planner's object budget is full. The later explicit clearance
        # check will reject a route that passes through the held actor.
        return dict(scene), other
    occupied = {item["id"] for item in objects}
    identifier = "__held_actor_obstacle"
    while identifier in occupied:
        identifier += "_"
    x, z = positions[other]
    # plan_action inflates obstacle half extents by radius .28 + margin .06.
    # .31 + .34 = .65 m minimum centre distance along either face.
    objects.append({"id": identifier, "name": "Stationary actor clearance",
                    "kind": "crate", "position": [x, .825, z],
                    "size": [.62, 1.65, .62]})
    return {**scene, "objects": objects}, other


def _segment_clearance(a: tuple[float, float] | list[float],
                       b: tuple[float, float] | list[float],
                       center: list[float]) -> float:
    ax, az = a[0] - center[0], a[1] - center[1]
    dx, dz = b[0] - a[0], b[1] - a[1]
    length_squared = dx * dx + dz * dz
    alpha = 0. if length_squared < 1e-12 else max(0., min(1., -(ax * dx + az * dz) / length_squared))
    return math.hypot(ax + alpha * dx, az + alpha * dz)


def plan_navigation(scene: Mapping, actor_ids: tuple[str, ...] | list[str], *,
                    actor_id: str, target_id: str, verb: str,
                    last_clip: CanonicalClip | None = None,
                    initial_placements: Mapping | None = None,
                    affordances: Mapping | None = None,
                    speed_mps: float = .65) -> tuple[list[StageSpec], dict]:
    """Plan a target by ID and sample root goals per 40-frame model window.

    Each stage contains five local-frame root targets per actor, including
    frame 39. The last horizon holds the terminal position and heading. A
    second actor receives stationary root targets throughout the route.
    """
    ids = tuple(actor_ids)
    if not 1 <= len(ids) <= 2 or any(not isinstance(x, str) or not 1 <= len(x) <= 64 for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("actor_ids must be one or two unique stable IDs")
    if actor_id not in ids:
        raise ValueError("selected actor is not in the stable actor set")
    if verb not in ("approach", "go_through"):
        raise ValueError("navigation supports approach or go_through")
    if not isinstance(target_id, str) or not target_id:
        raise ValueError("target_id is required")
    speed = _finite(speed_mps, "speed_mps", limit=3)
    if speed < .2:
        raise ValueError("speed_mps must be at least .2")
    objects = scene_objects(scene)
    target = next((obj for obj in objects if obj.id == target_id), None)
    if target is None:
        raise ValueError(f"Unknown target_id: {target_id}")
    positions, yaws = _placements(ids, last_clip, initial_placements)
    start = positions[actor_id]
    planning_scene, held_actor = _planning_scene(scene, ids, actor_id, positions)
    route = plan_action({"verb": verb, "actor_id": actor_id, "target_id": target_id},
                        planning_scene, [start[0], 0., start[1]], affordances=affordances,
                        speed_mps=speed)
    waypoints = route["waypoints"]
    travel_seconds = waypoints[-1]["time_seconds"] - waypoints[0]["time_seconds"]
    moving_windows = max(1, math.ceil(travel_seconds * FPS / HORIZON))
    total_frames = (moving_windows + 1) * HORIZON
    if total_frames > MAX_SECONDS * FPS:
        raise ValueError("Route plus terminal hold exceeds the 30-second generation cap")
    arrival = _arrival_heading(waypoints, yaws[actor_id])
    action = "Walk through" if verb == "go_through" else "Approach"
    active_prompt = f"{action} {target.name} along the planned clear route, then stop."
    actor_prompts = {aid: (active_prompt if aid == actor_id else "Stand in place and hold position.")
                     for aid in ids}
    stages = []
    all_active_goals = []
    for stage_index in range(total_frames // HORIZON):
        local_frames = set(SAMPLE_FRAMES)
        for waypoint in waypoints[1:-1]:
            global_frame = max(0, min(total_frames - 1,
                                      round(waypoint["time_seconds"] * FPS) - 1))
            if global_frame // HORIZON == stage_index:
                local_frames.add(global_frame % HORIZON)
        goals = {aid: [] for aid in ids}
        for local in sorted(local_frames):
            global_frame = stage_index * HORIZON + local
            second = (global_frame + 1) / FPS
            active_xz = _point_at(waypoints, second)
            active_yaw = _heading_at(waypoints, second, yaws[actor_id], arrival)
            for aid in ids:
                xz = active_xz if aid == actor_id else positions[aid]
                yaw = active_yaw if aid == actor_id else yaws[aid]
                # Decimal rounding can push exact pi to 3.141593, which the
                # Core request boundary correctly rejects as outside [-pi,pi].
                heading = max(-math.pi, min(math.pi, round(yaw, 6)))
                goals[aid].append({"frame": local,
                                   "position_xz": [round(xz[0], 5), round(xz[1], 5)],
                                   "heading": heading})
        if any(len(items) > 24 for items in goals.values()):
            raise ValueError("Route needs too many model root constraints in one horizon")
        all_active_goals.extend(goals[actor_id])
        metadata = {"root_targets": goals,
                    "navigation": {"verb": verb, "actor_id": actor_id,
                                   "target_id": target_id, "stage": stage_index,
                                   "planned_only": True}}
        if stage_index == 0 and last_clip is None:
            metadata["initial_placements"] = {
                aid: {"position_xz": positions[aid], "yaw": yaws[aid]} for aid in ids}
        stages.append(StageSpec(active_prompt, kind="approach", frames=HORIZON,
                                source="ardy_core", actor_prompts=actor_prompts,
                                metadata=metadata))
    if held_actor is not None:
        points = [start] + [goal["position_xz"] for goal in all_active_goals]
        clearance = min(_segment_clearance(a, b, positions[held_actor])
                        for a, b in zip(points, points[1:]))
        if clearance < HELD_ACTOR_CLEARANCE_M - .01:
            raise ValueError("Sampled route passes too close to the stationary actor")
    else:
        clearance = None
    result = {**route, "schedule": {"frames": total_frames, "fps": FPS,
                                    "seconds": total_frames / FPS,
                                    "travel_seconds": travel_seconds,
                                    "terminal_hold_frames": HOLD_FRAMES,
                                    "horizons": len(stages),
                                    "dense_root_targets_per_actor_per_horizon": len(SAMPLE_FRAMES),
                                    "held_actor_id": held_actor,
                                    "sampled_route_clearance_m": clearance}}
    return stages, result
