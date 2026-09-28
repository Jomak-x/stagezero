"""Explicit support-aware route and native height intent for Core commands.

Routes describe rendered sole support. They do not certify generated gait or
contacts; CoreStudioSession must reject unsupported native motion before commit.
"""
from __future__ import annotations

import math
import re

import numpy as np

from realtime_director import HORIZON, StageSpec
from realtime_navigation import (FPS, MAX_SECONDS, SAMPLE_FRAMES,
                                 SPATIAL_SETTLE_SECONDS, _placements,
                                 _spatial_fraction, _spatial_second_for_fraction)
from scene_interaction_geometry import build_scene_geometry
from scene_navigation import plan_navigation_route


ROOT_TO_SOLE_M = .95  # Official Core neutral pelvis-to-floor clearance.
SPEED_MPS = 1.2
# Terrain routes need room for articulated arms after world-facing assistance:
# a walking hand can extend .44 m in XZ, plus a .06 m proxy and path margin.
ACTOR_RADIUS_M = .60
# Keep the lower probe inside normal risers; the .60 m envelope applies at arm
# and torso heights, where a tall prop can strike the native upper body.
LOWER_BODY_RADIUS_M = .18
TARGET_LIMIT = 24


def terrain_scene(scene):
    """Only an explicit spatial command may opt into this branch."""
    return bool(build_scene_geometry(scene).stair_routes)


def _surface_candidates(geometry, x, z, object_id=None):
    """Find actual triangle heights, including separate floors at one XZ."""
    found = []
    for surface in geometry.walkable_surfaces:
        if object_id is not None and surface.object_id != object_id:
            continue
        a, b, c = (np.asarray(p, dtype=float) for p in surface.triangle)
        denominator = (b[2]-c[2])*(a[0]-c[0]) + (c[0]-b[0])*(a[2]-c[2])
        if abs(denominator) < 1e-10:
            continue
        u = ((b[2]-c[2])*(x-c[0]) + (c[0]-b[0])*(z-c[2]))/denominator
        v = ((c[2]-a[2])*(x-c[0]) + (a[0]-c[0])*(z-c[2]))/denominator
        w = 1-u-v
        if min(u, v, w) >= -1e-5:
            y = float(u*a[1] + v*b[1] + w*c[1])
            if not any(abs(y-other) < .025 for other in found):
                found.append(y)
    return sorted(found)


def _support_at(geometry, x, z, reference_y=None, *, object_id=None):
    candidates = _surface_candidates(geometry, x, z, object_id)
    if reference_y is None:
        if len(candidates) != 1:
            raise ValueError("Starting floor is missing or ambiguous; place the actor on a unique rendered surface")
        return candidates[0]
    if not candidates:
        raise ValueError("Target has no rendered walkable support")
    if object_id is None:
        reachable = geometry.support_height(x, z, reference_y, max_step_up=.25, max_drop=.35)
        if reachable is not None:
            return reachable
    near = sorted(candidates, key=lambda y: abs(y-reference_y))
    if len(near) > 1 and abs(abs(near[0]-reference_y)-abs(near[1]-reference_y)) < .06:
        raise ValueError("Target has ambiguous overlapping walkable floors")
    return near[0]


def _clear_destination_height(geometry, x, z):
    for y in _surface_candidates(geometry, x, z):
        if not any(geometry.obstacle_at(x, y+offset, z,
                                        radius=LOWER_BODY_RADIUS_M if offset == .40 else ACTOR_RADIUS_M)
                   for offset in (.40, .90, 1.35)):
            return y
    raise ValueError("Far side has no supported body clearance")


def _root_start(geometry, actor_ids, actor_id, last_clip, placements):
    positions, yaws = _placements(tuple(actor_ids), last_clip, placements)
    x, z = positions[actor_id]
    reference = None if last_clip is None else float(last_clip.positions[tuple(actor_ids).index(actor_id), -1, 0, 1] - ROOT_TO_SOLE_M)
    floor = _support_at(geometry, x, z, reference)
    if reference is not None and abs(floor-reference) > .55:
        raise ValueError("Committed actor root is too far from rendered support")
    return np.array((x, floor, z), dtype=float), yaws, positions


def _objects_for_alias(scene, alias):
    words = {"gate": ("door",), "door": ("door",),
             "bridge": ("bridge", "walkway"), "walkway": ("bridge", "walkway"),
             "stairs": ("stairs", "steps", "stair"), "steps": ("stairs", "steps", "stair")}
    needles = words.get(alias.casefold(), ())
    result = []
    for obj in scene["objects"]:
        name = obj.get("name", "")
        if alias.casefold() in ("bridge", "walkway"):
            # A bridge approach stair or landing names the destination, not
            # the deck to cross. Require the traversal noun itself.
            matches = obj["kind"] in needles or bool(re.search(r"\b(bridge|walkway)\s*$", name, re.I))
        else:
            matches = obj["kind"] in needles or any(
                re.search(r"\b" + re.escape(word) + r"\b", name, re.I) for word in needles)
        if matches:
            result.append(obj)
    return result


def resolve_terrain_object(action, adapted, start, geometry=None):
    scene = adapted["scene"]
    if action.get("target_id"):
        return next(obj for obj in scene["objects"] if obj["id"] == action["target_id"])
    alias = action.get("target_alias")
    options = _objects_for_alias(scene, alias or "")
    if action.get("verb") in ("ascend", "descend") and alias in ("stairs", "steps"):
        # A landing named "stair entry" describes proximity, not a rendered
        # flight. Exact object names remain authoritative and are checked by
        # _object_target so invalid geometry still fails explicitly.
        geometry = geometry if geometry is not None else build_scene_geometry(scene)
        flight_ids = {flight["object_id"] for flight in geometry.stair_routes}
        options = [obj for obj in options if obj["id"] in flight_ids]
    if not options:
        raise ValueError(f"No authored {alias or 'target'} affordance exists")
    if alias != "stairs" and len(options) > 1:
        raise ValueError(f"{alias} target is ambiguous; name the exact object")
    distances = sorted(((float(np.linalg.norm(np.asarray(obj["position"])[[0, 2]]-start[[0, 2]])), obj)
                        for obj in options), key=lambda row: row[0])
    if len(distances) > 1 and distances[1][0]-distances[0][0] < .35:
        raise ValueError(f"{alias} target is ambiguous; name the exact object")
    return distances[0][1]


def _concat_routes(geometry, points):
    pieces = []
    for a, b in zip(points, points[1:]):
        piece = plan_navigation_route(geometry, a, b, radius=LOWER_BODY_RADIUS_M,
                                      upper_body_radius=ACTOR_RADIUS_M,
                                      max_step_up=.25, max_drop=.35,
                                      max_expansions=800)
        pieces.extend(piece if not pieces else piece[1:])
    return np.asarray(pieces, dtype=float)


def _start_tread_index(geometry, start, object_id, steps):
    """Identify a committed root supported by this flight's rendered tread."""
    surfaces = _surface_candidates(geometry, start[0], start[2], object_id)
    matches = [index for index, step in enumerate(steps)
               if abs(start[1]-step[1]) < .015
               and any(abs(height-step[1]) < .015 for height in surfaces)]
    if not matches:
        return None
    return min(matches, key=lambda index: np.linalg.norm((start-steps[index])[[0, 2]]))


def _traverses_stair_treads(geometry, path, object_id, steps, verb, start_index=None):
    """Confirm ordered progress over rendered tread surfaces of this flight.

    Navigation compacts collinear support samples into riser brackets.  The
    stored route may have no vertex near the centre of a broad tread, so test
    its segments against the selected object's actual walkable triangles.
    """
    run = np.linalg.norm(np.diff(steps[:, [0, 2]], axis=0), axis=1)
    spacing = min(.08, float(np.min(run))/4.)
    direction = 1 if verb == "ascend" else -1
    order = range(len(steps)) if direction > 0 else range(len(steps)-1, -1, -1)
    progress = [] if start_index is None else [start_index]
    final_matches = set()
    for first, last in zip(path, path[1:]):
        distance = float(np.linalg.norm((last-first)[[0, 2]]))
        count = max(1, math.ceil(distance/spacing))
        for sample in range(1, count+1):
            point = first + (last-first)*(sample/count)
            surfaces = _surface_candidates(geometry, point[0], point[2], object_id)
            matches = {index for index in order for height in surfaces
                       if abs(height-steps[index, 1]) < .015
                       and abs(point[1]-height) <= .08}
            final_matches = matches
            for index in order:
                if index in matches and (not progress or index-progress[-1] == direction):
                    progress.append(index)
    destination = len(steps)-1 if direction > 0 else 0
    remaining = len(steps) if start_index is None else abs(destination-start_index)+1
    return (destination in final_matches and progress[-1:] == [destination]
            and len(progress) >= min(3, remaining))


def _door_points(action, adapted, start, obj, geometry):
    if action["verb"] == "go_through" and "original_scene" not in adapted:
        raise ValueError("Terrain crossing needs the authored gate scene to verify observed opening")
    source = adapted.get("original_scene", adapted["scene"])
    original = next(item for item in source["objects"] if item["id"] == obj["id"])
    floor = float(original["position"][1]-original["size"][1]/2)
    evaluated_lift = float(obj["position"][1]-original["position"][1])
    axis_angle = math.radians(original.get("yaw", 0))
    normal = np.array((math.sin(axis_angle), math.cos(axis_angle)))
    center = np.asarray(original["position"], dtype=float)
    side = float(np.dot(start[[0, 2]]-center[[0, 2]], normal))
    if abs(side) < original["size"][2]/2 + LOWER_BODY_RADIUS_M:
        raise ValueError("Actor starts inside the gate approach; crossing side is ambiguous")
    side_sign = 1. if side > 0 else -1.
    # Leave the articulated body envelope clear of a closed panel. Side
    # ambiguity above concerns the root footprint, including a repeated open.
    reach = original["size"][2]/2 + max(LOWER_BODY_RADIUS_M+.25, ACTOR_RADIUS_M) + .06
    entry_xz = center[[0, 2]] + normal*side_sign*reach
    exit_xz = center[[0, 2]] - normal*side_sign*reach
    entry = np.array((entry_xz[0], floor, entry_xz[1]))
    middle = np.array((center[0], floor, center[2]))
    exit_point = np.array((exit_xz[0], floor, exit_xz[1]))
    if action["verb"] == "open":
        path = _concat_routes(geometry, (start, entry))
        roles = {"approach": path[-1]}
    else:
        if evaluated_lift < original["size"][1]-.03:
            raise ValueError("Gate is not fully open from observed motion")
        if original["size"][0] < 2*(ACTOR_RADIUS_M+.06)+.1:
            raise ValueError("Gate passage is too narrow")
        path = _concat_routes(geometry, (start, entry, middle, exit_point))
        roles = {"entry": entry, "center": middle, "exit": exit_point}
    return path, roles, {"passage_depth_m": original["size"][2],
                         "minimum_lateral_clearance_m": original["size"][0]/2-ACTOR_RADIUS_M-.06,
                         "floor_y_m": floor, "source": "observed_raised_door"}


def _object_target(action, adapted, start, geometry, obj):
    verb = action["verb"]
    if verb in ("ascend", "descend"):
        flights = [flight for flight in geometry.stair_routes if flight["object_id"] == obj["id"]]
        if len(flights) != 1:
            raise ValueError(f"{obj['id']} has no unambiguous rendered stair flight")
        steps = np.asarray(flights[0]["steps"], dtype=float)
        first, last = steps[0], steps[-1]
        goal = last if verb == "ascend" else first
        departure = first if verb == "ascend" else last
        start_index = _start_tread_index(geometry, start, obj["id"], steps)
        if start_index == (len(steps)-1 if verb == "ascend" else 0):
            raise ValueError(f"Actor is already at the {verb} destination side of the stairs")
        if (start_index is None and
                np.linalg.norm((start-departure)[[0, 2]]) >
                np.linalg.norm((start-goal)[[0, 2]])+.2):
            raise ValueError(f"Actor is already at the {verb} destination side of the stairs")
        path = _concat_routes(geometry, (start, goal))
        if not _traverses_stair_treads(geometry, path, obj['id'], steps, verb, start_index):
            raise ValueError("Planned route does not traverse the selected stair flight")
        return path, {"arrival": path[-1]}, {"support_object_id": obj["id"]}
    if verb == "cross":
        if obj["kind"] != "custom":
            raise ValueError("Cross requires authored walkable bridge geometry")
        size = np.asarray(obj["size"], dtype=float)
        long_x = size[0] > size[2]
        theta = math.radians(obj.get("yaw", 0))
        axis = np.array((math.cos(theta), -math.sin(theta))) if long_x else np.array((math.sin(theta), math.cos(theta)))
        center = np.asarray(obj["position"], dtype=float)
        bridge_floor = _support_at(geometry, center[0], center[2], start[1]+.5, object_id=obj["id"])
        signed = float(np.dot(start[[0, 2]]-center[[0, 2]], axis))
        if abs(signed) < .2:
            raise ValueError("Actor starts at bridge centre; crossing direction is ambiguous")
        half_length = size[0]/2 if long_x else size[2]/2
        far_xz = center[[0, 2]] - np.sign(signed)*axis*(half_length+.22)
        far_y = _clear_destination_height(geometry, far_xz[0], far_xz[1])
        middle = np.array((center[0], bridge_floor, center[2]))
        far = np.array((far_xz[0], far_y, far_xz[1]))
        path = _concat_routes(geometry, (start, middle, far))
        return path, {"center": middle, "arrival": path[-1]}, {"support_object_id": obj["id"]}
    if obj["kind"] == "door" and verb in ("open", "go_through"):
        return _door_points(action, adapted, start, obj, geometry)
    if verb == "go_through":
        raise ValueError("Terrain passage needs an authored automatic door")
    # Approach a prop from the nearest clear supported side.
    theta = math.radians(obj.get("yaw", 0))
    axes = (np.array((math.cos(theta), -math.sin(theta))),
            np.array((math.sin(theta), math.cos(theta))))
    center = np.asarray(obj["position"], dtype=float)
    candidates = []
    for axis, half in zip(axes, (obj["size"][0]/2, obj["size"][2]/2)):
        for sign in (-1, 1):
            xz = center[[0, 2]] + sign*axis*(half+ACTOR_RADIUS_M+.3)
            try:
                y = _support_at(geometry, xz[0], xz[1], start[1])
                path = _concat_routes(geometry, (start, (xz[0], y, xz[1])))
            except ValueError:
                continue
            candidates.append((float(np.linalg.norm(np.diff(path[:, [0, 2]], axis=0), axis=1).sum()), path))
    if not candidates:
        raise ValueError(f"No supported approach to {obj['id']}")
    path = min(candidates, key=lambda row: row[0])[1]
    return path, {"approach": path[-1]}, {"target_center_xz": center[[0, 2]].tolist()}


def _route_waypoints(path, roles):
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path[:, [0, 2]], axis=0), axis=1))]
    result = []
    for i, point in enumerate(path):
        role = "start" if i == 0 else "arrival" if i == len(path)-1 else "route"
        for name, chosen in roles.items():
            if np.linalg.norm(point[[0, 2]]-chosen[[0, 2]]) < 1e-4:
                role = name
        result.append({"time_seconds": round(float(distance[i]/SPEED_MPS), 4),
                       "position_xz": [round(float(point[0]), 5), round(float(point[2]), 5)],
                       "support_y": round(float(point[1]), 5), "role": role})
    return result, distance


def _sample(path, distance, fraction):
    s = float(distance[-1])*fraction
    return np.array([np.interp(s, distance, path[:, axis]) for axis in (0, 1, 2)])


def plan_terrain_command(action, adapted, actor_ids, actor_id, last_clip, initial_placements):
    scene = adapted["scene"]
    geometry = build_scene_geometry(scene)
    start, yaws, positions = _root_start(geometry, actor_ids, actor_id, last_clip, initial_placements)
    # A committed terrain take has a separate, route-facing presentation pose.
    # Its heading is planning intent only: keep last_clip and its exact native
    # features as the sole source of history, root position, and validation.
    planning_heading = adapted.get("terrain_planning_heading")
    if planning_heading is not None:
        if (not adapted.get("terrain_active") or type(planning_heading) not in (int, float)
                or not math.isfinite(planning_heading)):
            raise ValueError("Terrain planning heading must be a finite terrain-only yaw")
        yaws[actor_id] = math.atan2(math.sin(planning_heading), math.cos(planning_heading))
    verb = action["verb"]
    resolved_target_id = None
    if verb == "move":
        offsets = {"forward": 0., "forwards": 0., "back": math.pi, "backward": math.pi,
                   "backwards": math.pi, "left": -math.pi/2, "right": math.pi/2}
        heading = yaws[actor_id]+offsets[action["direction"]]
        xz = start[[0, 2]] + action["distance_m"]*np.array((math.sin(heading), math.cos(heading)))
        y = _support_at(geometry, xz[0], xz[1], start[1])
        path = _concat_routes(geometry, (start, (xz[0], y, xz[1])))
        roles, details = {"arrival": path[-1]}, {"target_xz": xz.tolist()}
    else:
        obj = resolve_terrain_object(action, adapted, start, geometry)
        resolved_target_id = obj["id"]
        path, roles, details = _object_target(action, adapted, start, geometry, obj)
    waypoints, distance = _route_waypoints(path, roles)
    if distance[-1] < .2 and verb != "open":
        raise ValueError("Actor is already at the terrain destination")
    travel_seconds = float(distance[-1]/SPEED_MPS)
    moving_windows = max(1, math.ceil((travel_seconds+SPATIAL_SETTLE_SECONDS)/2.))
    hold_windows = 1 if verb == "open" else 0
    departure = next((math.atan2(b[0]-a[0], b[2]-a[2]) for a, b in zip(path, path[1:])
                      if np.linalg.norm((b-a)[[0, 2]]) > 1e-5), yaws[actor_id])
    delta = math.atan2(math.sin(departure-yaws[actor_id]), math.cos(departure-yaws[actor_id]))
    turn_windows = int(abs(delta) > math.radians(30))
    total_frames = (turn_windows+moving_windows+hold_windows)*HORIZON
    if total_frames > MAX_SECONDS*FPS:
        raise ValueError("Terrain route exceeds the 30-second native generation cap")
    if np.max(np.abs(path)) > 25:
        raise ValueError("Terrain route exceeds the Core worker's ±25 m bounds")
    moving_seconds = moving_windows*HORIZON/FPS
    # Smooth *pelvis intent* over short time; exact support samples remain in
    # route metadata for terrain contact validation. Planks do not bob the root.
    raw = np.asarray([_sample(path, distance, _spatial_fraction((f+1)/FPS, moving_seconds))
                      for f in range(moving_windows*HORIZON)])
    kernel = np.array((1., 2., 3., 4., 3., 2., 1.))/16.
    smoothed_y = np.convolve(np.pad(raw[:, 1], (3, 3), mode="edge"), kernel, mode="valid")
    smoothed_y[-1] = path[-1, 1]
    extra_frames = set()
    for i in range(1, len(path)-1):
        before, after = path[i]-path[i-1], path[i+1]-path[i]
        if min(np.linalg.norm(before[[0, 2]]), np.linalg.norm(after[[0, 2]])) < .08:
            continue
        bend = abs(math.atan2(before[0]*after[2]-before[2]*after[0],
                              before[0]*after[0]+before[2]*after[2]))
        if bend > .30:
            frame = max(0, min(len(raw)-1, round(_spatial_second_for_fraction(distance[i]/distance[-1], moving_seconds)*FPS)-1))
            extra_frames.add(frame)
    stages = []
    ids = tuple(actor_ids)
    for stage_index in range(turn_windows+moving_windows+hold_windows):
        turning = stage_index < turn_windows
        motion_index = stage_index-turn_windows
        holding = motion_index >= moving_windows
        extra = {f%HORIZON for f in extra_frames if f//HORIZON == motion_index} if not turning and not holding else set()
        frames = sorted(set(SAMPLE_FRAMES)|extra)
        if len(frames) > TARGET_LIMIT:
            raise ValueError("Terrain route needs too many native targets in one horizon")
        floor_origin = float(start[1] if turning else path[-1, 1] if holding else raw[motion_index*HORIZON, 1])
        goals = {}
        frames_y = {}
        for aid in ids:
            if aid == actor_id:
                targets = []
                for local in frames:
                    if turning:
                        point, height = start, start[1]
                        yaw = yaws[aid]+delta*(local/(HORIZON-1))
                    elif holding:
                        point, height = path[-1], path[-1, 1]
                    else:
                        frame = motion_index*HORIZON+local
                        point, height = raw[frame], smoothed_y[frame]
                    target = {"frame": local,
                              "position_xz": [round(float(point[0]), 5), round(float(point[2]), 5)],
                              "root_height": round(float(height+ROOT_TO_SOLE_M), 5)}
                    if turning:
                        # The Core request bounds heading to ±π. Preserve the
                        # intended turn across that boundary instead of
                        # flattening every target beyond it to the limit.
                        wrapped = math.atan2(math.sin(yaw), math.cos(yaw))
                        target["heading"] = max(-math.pi, min(math.pi, round(wrapped, 6)))
                    targets.append(target)
                goals[aid] = targets
                frames_y[aid] = round(floor_origin, 5)
            else:
                x, z = positions[aid]
                reference = None if last_clip is None else float(last_clip.positions[ids.index(aid), -1, 0, 1]-ROOT_TO_SOLE_M)
                held_floor = _support_at(geometry, x, z, reference)
                held_root = held_floor+ROOT_TO_SOLE_M if last_clip is None else float(last_clip.positions[ids.index(aid), -1, 0, 1])
                goals[aid] = [{"frame": f, "position_xz": [round(x, 5), round(z, 5)],
                               "heading": round(yaws[aid], 6), "root_height": round(held_root, 5)}
                              for f in SAMPLE_FRAMES]
                frames_y[aid] = round(held_floor, 5)
        prompt = ("A person turns in place, keeping an upright posture." if turning else
                  "A person stands upright and relaxed." if holding else
                  "A person walks forward naturally, then slows to a relaxed stop." if motion_index == moving_windows-1 else
                  "A person walks forward naturally.")
        metadata = {"root_targets": goals, "coordinate_frames_y": frames_y,
                    "terrain_navigation_version": 1,
                    "terrain_navigation": {"actor_id": actor_id, "verb": verb, "stage": stage_index,
                                           "raw_native_validation_required": True,
                                           "support_start_y": round(float(start[1]), 5),
                                           "support_end_y": round(float(path[-1, 1]), 5)}}
        if stage_index == 0 and last_clip is None:
            metadata["initial_placements"] = {
                aid: {"position_xz": positions[aid], "yaw": yaws[aid]} for aid in ids}
        actor_prompts = {aid: prompt if aid == actor_id else "Stand in place and hold position." for aid in ids}
        stages.append(StageSpec(prompt, kind="approach", frames=HORIZON, source="ardy_core",
                                actor_prompts=actor_prompts, metadata=metadata))
    route = {"version": 1, "terrain_navigation_version": 1, "actor_id": actor_id,
             "verb": "go_through" if verb == "enter" else verb,
             "target_id": resolved_target_id, "waypoints": waypoints,
             "support_xyz": np.round(path, 5).tolist(), "geometry": details,
             "assumptions": {"motion_following_verified": False,
                             "raw_native_contact_verified": False,
                             "actor_radius_m": ACTOR_RADIUS_M,
                             "lower_body_radius_m": LOWER_BODY_RADIUS_M,
                             "speed_mps": SPEED_MPS},
             "schedule": {"frames": total_frames, "fps": FPS,
                          "seconds": total_frames/FPS, "travel_seconds": travel_seconds,
                          "initial_turn_frames": turn_windows*HORIZON,
                          "horizons": len(stages), "terminal_settle_frames": round(SPATIAL_SETTLE_SECONDS*FPS),
                          "door_hold_frames": hold_windows*HORIZON,
                          "gait_profile": "terrain_experimental"}}
    return stages, route
