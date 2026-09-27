"""Plan a two-person city encounter with explicitly staged, contactless sparring.

The output is native ARDY conditioning, never a claim that generation followed
the route, made a believable gesture, touched, or maintained foot contact.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from core_action_library import CONTACT_MODE, PROMPTS, SAMPLE_FRAMES, make_action_candidate
from interaction_planner import _inside, _intersects, _obstacles, _path
from interaction_scene import scene_objects
from realtime_director import HORIZON, StageSpec
from realtime_navigation import validate_ground_path
from scene_composition import validate_scene
from studio_interaction_scene import recommend_placements


ACTOR_IDS = ("actor_1", "actor_2")
MIN_PLANNED_SEPARATION_M = .90
ENCOUNTER_SEPARATION_M = 1.60
ACTOR_PATH_PAD_M = .34
ROUTE_VARIANTS = ("direct", "west", "east")
TIMING_VARIANTS = ("measured", "brisk")
_SPEED = {"measured": .6, "brisk": 1.2}


def _xz(value, label):
    if (not isinstance(value, (tuple, list)) or len(value) != 2 or
            any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 25 for v in value)):
        raise ValueError(f"{label} must be finite [x, z] within ±25 m")
    return (float(value[0]), float(value[1]))


def _placement_map(scene, placements):
    if placements is None:
        if any(obj.get("asset") == "asphalt" for obj in scene["objects"]):
            # The city preset reserves a clear central avenue. These anchors
            # give both actors visible travel before their safe encounter.
            placements = {"actor_1": {"position_xz": [-1.2, 0.], "yaw": math.pi},
                          "actor_2": {"position_xz": [1.2, -6.], "yaw": 0.}}
        else:
            recommended = recommend_placements(scene, 2)
            placements = dict(zip(ACTOR_IDS, recommended))
            first = tuple(recommended[0]["position_xz"])
            if math.dist(first, recommended[1]["position_xz"]) < 2.2:
                obstacles = _obstacles(scene_objects(scene), None, 1.65, ACTOR_PATH_PAD_M)
                for dx, dz in ((0, -3), (0, 3), (-3, 0), (3, 0),
                               (-2.4, -2.4), (2.4, -2.4), (-2.4, 2.4), (2.4, 2.4)):
                    candidate = (first[0] + dx, first[1] + dz)
                    if (any(abs(value) > 25 for value in candidate) or
                            any(_inside(candidate, box, ACTOR_PATH_PAD_M) for box in obstacles)):
                        continue
                    try:
                        validate_ground_path(scene, [first, candidate])
                    except ValueError:
                        continue
                    placements[ACTOR_IDS[1]] = {"position_xz": list(candidate),
                                                 "yaw": recommended[1].get("yaw", 0.)}
                    break
    if not isinstance(placements, Mapping) or set(placements) != set(ACTOR_IDS):
        raise ValueError("placements must cover actor_1 and actor_2")
    result = {}
    for actor_id in ACTOR_IDS:
        item = placements[actor_id]
        if not isinstance(item, Mapping) or set(item) - {"position_xz", "yaw"} or "position_xz" not in item:
            raise ValueError(f"{actor_id} placement needs position_xz and optional yaw")
        point = _xz(item["position_xz"], actor_id)
        yaw = item.get("yaw", 0.)
        if type(yaw) not in (int, float) or not math.isfinite(yaw) or abs(yaw) > math.pi:
            raise ValueError(f"{actor_id} yaw must be finite radians within ±pi")
        result[actor_id] = {"position_xz": list(point), "yaw": float(yaw)}
    if math.dist(*[result[aid]["position_xz"] for aid in ACTOR_IDS]) < 2.2:
        raise ValueError("Actors need at least 2.2 m initial separation for this encounter")
    return result


def _heading(a, b, fallback):
    dx, dz = b[0] - a[0], b[1] - a[1]
    return math.atan2(dx, dz) if math.hypot(dx, dz) > 1e-8 else fallback


def _turn(a, b, alpha):
    delta = (b - a + math.pi) % (2 * math.pi) - math.pi
    return (a + delta * alpha + math.pi) % (2 * math.pi) - math.pi


def _path_lengths(path):
    lengths = [0.]
    for a, b in zip(path, path[1:]):
        lengths.append(lengths[-1] + math.dist(a, b))
    return lengths


def _path_point(path, lengths, fraction):
    distance = max(0., min(1., fraction)) * lengths[-1]
    for index in range(1, len(path)):
        if distance <= lengths[index] + 1e-9:
            span = lengths[index] - lengths[index - 1]
            alpha = 0. if span < 1e-9 else (distance - lengths[index - 1]) / span
            return (path[index - 1][0] + alpha * (path[index][0] - path[index - 1][0]),
                    path[index - 1][1] + alpha * (path[index][1] - path[index - 1][1]))
    return path[-1]


def _goal(frame, point, heading):
    return {"frame": frame, "position_xz": [round(point[0], 5), round(point[1], 5)],
            "heading": max(-math.pi, min(math.pi, round(heading, 6)))}


def _route_targets(path, windows, placement_yaw, *, departing=False):
    """Keep every obstacle-route corner as an extra native target."""
    lengths = _path_lengths(path)
    total_frames = windows * HORIZON
    corners = {}
    if lengths[-1] > 1e-9:
        for index in range(1, len(path) - 1):
            global_frame = min(total_frames - 2,
                               max(0, round(lengths[index] / lengths[-1] * total_frames) - 1))
            if global_frame in corners:
                raise ValueError("Route corners are too close for native frame targets")
            corners[global_frame] = index
    stages = []
    for stage_index in range(windows):
        frames = set(SAMPLE_FRAMES)
        frames.update(frame % HORIZON for frame in corners if frame // HORIZON == stage_index)
        if len(frames) > 24:
            raise ValueError("Route needs too many native constraints in one horizon")
        goals = []
        for local in sorted(frames):
            global_frame = stage_index * HORIZON + local
            fraction = (global_frame + 1) / total_frames
            point = (path[corners[global_frame]] if global_frame in corners
                     else _path_point(path, lengths, fraction))
            ahead = _path_point(path, lengths, min(1., fraction + .01))
            behind = _path_point(path, lengths, max(0., fraction - .01))
            yaw = _heading(behind, ahead, placement_yaw)
            if departing and stage_index == 0 and local < 15:
                yaw = _turn(placement_yaw, yaw, (local + 1) / 16)
            goals.append(_goal(local, point, yaw))
        stages.append(goals)
    return stages


def _hold_targets(start, end, start_yaw, end_yaw):
    return [_goal(frame, (start[0] + (end[0] - start[0]) * (frame + 1) / HORIZON,
                          start[1] + (end[1] - start[1]) * (frame + 1) / HORIZON),
                  _turn(start_yaw, end_yaw, (frame + 1) / HORIZON))
            for frame in SAMPLE_FRAMES]


def _point_on_targets(start, goals, local_frame):
    before_frame, before = -1, start
    for goal in goals:
        after_frame, after = goal["frame"], goal["position_xz"]
        if local_frame <= after_frame:
            alpha = (local_frame - before_frame) / (after_frame - before_frame)
            return (before[0] + alpha * (after[0] - before[0]),
                    before[1] + alpha * (after[1] - before[1]))
        before_frame, before = after_frame, after
    return tuple(goals[-1]["position_xz"])


def _min_segment_distance(a0, a1, b0, b1):
    rx, rz = a0[0] - b0[0], a0[1] - b0[1]
    dx, dz = (a1[0] - a0[0]) - (b1[0] - b0[0]), (a1[1] - a0[1]) - (b1[1] - b0[1])
    norm = dx * dx + dz * dz
    t = 0. if norm < 1e-12 else max(0., min(1., -(rx * dx + rz * dz) / norm))
    return math.hypot(rx + t * dx, rz + t * dz)


def _validate_tracks(scene, tracks, placements, obstacles):
    positions = {aid: tuple(placements[aid]["position_xz"]) for aid in ACTOR_IDS}
    paths = {aid: [positions[aid]] for aid in ACTOR_IDS}
    minimum = math.dist(*positions.values())
    for stage in tracks:
        starts = positions.copy()
        for aid in ACTOR_IDS:
            for goal in stage[aid]:
                point = tuple(goal["position_xz"])
                if any(abs(value) > 25 for value in point):
                    raise ValueError("Encounter target exceeds Core's ±25 m bounds")
                if any(_inside(point, box, ACTOR_PATH_PAD_M) or
                       _intersects(positions[aid], point, box, ACTOR_PATH_PAD_M)
                       for box in obstacles):
                    raise ValueError("Encounter route overlaps scene solids")
                paths[aid].append(point)
                positions[aid] = point
        for frame in range(HORIZON):
            a0 = _point_on_targets(starts[ACTOR_IDS[0]], stage[ACTOR_IDS[0]], frame - 1)
            a1 = _point_on_targets(starts[ACTOR_IDS[0]], stage[ACTOR_IDS[0]], frame)
            b0 = _point_on_targets(starts[ACTOR_IDS[1]], stage[ACTOR_IDS[1]], frame - 1)
            b1 = _point_on_targets(starts[ACTOR_IDS[1]], stage[ACTOR_IDS[1]], frame)
            minimum = min(minimum, _min_segment_distance(a0, a1, b0, b1))
    if minimum < MIN_PLANNED_SEPARATION_M - 1e-5:
        raise ValueError("Planned actors approach below the staged separation margin")
    ground = {aid: validate_ground_path(scene, paths[aid]) for aid in ACTOR_IDS}
    if any(not result.get("continuous_support_verified", False) for result in ground.values()):
        raise ValueError("City encounter requires authored, continuous ground beneath both routes")
    return {"planned_min_root_separation_m": round(minimum, 4),
            "planned_path_ground_support": ground,
            "planned_obstacle_clearance_margin_m": ACTOR_PATH_PAD_M,
            "planned_root_disc_overlap_proxy_frames": 0}


def build_city_encounter(scene, placements=None, *, route_variant="direct",
                         timing_variant="measured", seed=33):
    """Return ``(placements, StageSpec tuple, plan report)`` for a fresh cast.

    The report concerns requested root tracks and scene proxies only. Generated
    clips still need measured collision, floor, continuity and visual review.
    """
    if route_variant not in ROUTE_VARIANTS:
        raise ValueError(f"route_variant must be one of {ROUTE_VARIANTS}")
    if timing_variant not in TIMING_VARIANTS:
        raise ValueError(f"timing_variant must be one of {TIMING_VARIANTS}")
    if type(seed) is not int or not 0 <= seed < 2**32 - 32:
        raise ValueError("seed must leave room for per-stage uint32 seeds")
    document = validate_scene(scene)
    place = _placement_map(document, placements)
    starts = {aid: tuple(place[aid]["position_xz"]) for aid in ACTOR_IDS}
    a, b = starts.values()
    dx, dz = b[0] - a[0], b[1] - a[1]
    distance = math.hypot(dx, dz)
    direction = dx / distance, dz / distance
    center = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    lateral = {"direct": 0., "west": -.65, "east": .65}[route_variant]
    center = center[0] + lateral, center[1]
    meet = {ACTOR_IDS[0]: (center[0] - direction[0] * ENCOUNTER_SEPARATION_M / 2,
                           center[1] - direction[1] * ENCOUNTER_SEPARATION_M / 2),
            ACTOR_IDS[1]: (center[0] + direction[0] * ENCOUNTER_SEPARATION_M / 2,
                           center[1] + direction[1] * ENCOUNTER_SEPARATION_M / 2)}
    obstacles = _obstacles(scene_objects(document), None, 1.65, ACTOR_PATH_PAD_M)
    routes = {aid: _path(starts[aid], meet[aid], obstacles, ACTOR_PATH_PAD_M)
              for aid in ACTOR_IDS}
    lengths = {aid: _path_lengths(routes[aid])[-1] for aid in ACTOR_IDS}
    windows = max(1, math.ceil(max(lengths.values()) / (_SPEED[timing_variant] * 2)))
    if windows > 4:
        raise ValueError("Encounter route exceeds the 30-second staged sequence budget")
    walk = {aid: _route_targets(routes[aid], windows, place[aid]["yaw"])
            for aid in ACTOR_IDS}
    face = {ACTOR_IDS[0]: _heading(meet[ACTOR_IDS[0]], meet[ACTOR_IDS[1]], 0.),
            ACTOR_IDS[1]: _heading(meet[ACTOR_IDS[1]], meet[ACTOR_IDS[0]], math.pi)}
    arrival = {aid: walk[aid][-1][-1]["heading"] for aid in ACTOR_IDS}
    tracks = []
    actions = []
    for index in range(windows):
        tracks.append({aid: walk[aid][index] for aid in ACTOR_IDS})
        actions.append({aid: "walk" for aid in ACTOR_IDS})
    tracks.append({aid: _hold_targets(meet[aid], meet[aid], arrival[aid], arrival[aid]) for aid in ACTOR_IDS})
    actions.append({aid: "stop" for aid in ACTOR_IDS})
    tracks.append({aid: _hold_targets(meet[aid], meet[aid], arrival[aid], face[aid]) for aid in ACTOR_IDS})
    actions.append({aid: "turn" for aid in ACTOR_IDS})
    tracks.append({aid: _hold_targets(meet[aid], meet[aid], face[aid], face[aid]) for aid in ACTOR_IDS})
    actions.append({aid: "meet" for aid in ACTOR_IDS})
    tracks.append({aid: _hold_targets(meet[aid], meet[aid], face[aid], face[aid]) for aid in ACTOR_IDS})
    actions.append({aid: "guard" for aid in ACTOR_IDS})
    feint = (meet[ACTOR_IDS[0]][0] + .14 * direction[0],
             meet[ACTOR_IDS[0]][1] + .14 * direction[1])
    dodge = (meet[ACTOR_IDS[1]][0] + .22 * direction[1],
             meet[ACTOR_IDS[1]][1] - .22 * direction[0])
    tracks.append({ACTOR_IDS[0]: _hold_targets(meet[ACTOR_IDS[0]], feint, face[ACTOR_IDS[0]], face[ACTOR_IDS[0]]),
                   ACTOR_IDS[1]: _hold_targets(meet[ACTOR_IDS[1]], dodge, face[ACTOR_IDS[1]], face[ACTOR_IDS[1]])})
    actions.append({ACTOR_IDS[0]: "lunge", ACTOR_IDS[1]: "dodge"})
    reaction = (dodge[0] + .10 * direction[0], dodge[1] + .10 * direction[1])
    tracks.append({ACTOR_IDS[0]: _hold_targets(feint, feint, face[ACTOR_IDS[0]], face[ACTOR_IDS[0]]),
                   ACTOR_IDS[1]: _hold_targets(dodge, reaction, face[ACTOR_IDS[1]], face[ACTOR_IDS[1]])})
    actions.append({ACTOR_IDS[0]: "guard", ACTOR_IDS[1]: "hit_reaction"})
    tracks.append({ACTOR_IDS[0]: _hold_targets(feint, meet[ACTOR_IDS[0]], face[ACTOR_IDS[0]], face[ACTOR_IDS[0]]),
                   ACTOR_IDS[1]: _hold_targets(reaction, meet[ACTOR_IDS[1]], face[ACTOR_IDS[1]], face[ACTOR_IDS[1]])})
    actions.append({aid: "retreat" for aid in ACTOR_IDS})
    depart = {aid: _route_targets(list(reversed(routes[aid])), windows, face[aid], departing=True)
              for aid in ACTOR_IDS}
    for index in range(windows):
        tracks.append({aid: depart[aid][index] for aid in ACTOR_IDS})
        actions.append({aid: "depart" for aid in ACTOR_IDS})
    geometry = _validate_tracks(document, tracks, place, obstacles)
    stages = []
    for index, (track, beat) in enumerate(zip(tracks, actions)):
        stage_seed = seed + index
        candidates = {aid: make_action_candidate(beat[aid], aid, seed=stage_seed,
                       root_targets=track[aid], conditions={"contact_mode": CONTACT_MODE,
                       "route_variant": route_variant, "timing_variant": timing_variant,
                       "scene_name": document["name"], "stage_index": index})
                      for aid in ACTOR_IDS}
        prompts = {aid: candidates[aid]["prompt"] for aid in ACTOR_IDS}
        metadata = {"seed": stage_seed, "root_targets": track,
                    "action_candidates": candidates,
                    "city_encounter": {"stage_index": index, "actions": beat,
                                       "contact_mode": CONTACT_MODE,
                                       "planned_only": True}}
        if index == 0:
            metadata["initial_placements"] = place
        stages.append(StageSpec(" / ".join(f"{aid}: {PROMPTS[beat[aid]]}" for aid in ACTOR_IDS),
                                kind="approach" if index == 0 else "action",
                                frames=HORIZON, source="ardy_core",
                                actor_prompts=prompts, metadata=metadata))
    report = {"status": "planned_unobserved", "contact_mode": CONTACT_MODE,
              "model": "ARDY-Core-RP-20FPS-Horizon40", "source": "ardy_core",
              "route_variant": route_variant, "timing_variant": timing_variant,
              "seed": seed, "actors": list(ACTOR_IDS), "initial_placements": place,
              "meeting_positions_xz": {aid: list(meet[aid]) for aid in ACTOR_IDS},
              "planned_route_lengths_m": {aid: round(lengths[aid], 4) for aid in ACTOR_IDS},
              "planned_windows": len(stages), "planned_frames": len(stages) * HORIZON,
              "planned_seconds": len(stages) * HORIZON / 20,
              "planned_meeting_frame": (windows + 3) * HORIZON - 1,
              "planned_stage_actions": actions,
              "planned_stage_root_targets": tracks,
              "observed_output": None, "observed_measurement": None,
              **geometry}
    return place, tuple(stages), report


def measure_city_encounter(clip, scene, plan_report):
    """Measure a complete clip; report geometry separately from acting/contact.

    This measures sampled Core27 joints and proxies only. It cannot certify a
    convincing performance, physical contact, impulse or causal hit reaction.
    """
    from interaction_metrics import continuity, floor_motion, goal_endpoint, joint_index, pair_separation
    from interaction_scene_collision import scene_collision
    from realtime_clip import CanonicalClip
    from studio_interaction_scene import adapt_studio_scene

    if not isinstance(plan_report, Mapping) or plan_report.get("contact_mode") != CONTACT_MODE:
        raise ValueError("expected a staged city encounter plan report")
    planned_frames = plan_report.get("planned_frames")
    if type(planned_frames) is not int or planned_frames < HORIZON:
        raise ValueError("plan report has no valid frame count")
    if clip is None:
        return {"status": "not_verified", "reason": "no generated clip",
                "planned_frames": planned_frames, "observed_frames": 0,
                "contact_mode": CONTACT_MODE}
    if not isinstance(clip, CanonicalClip) or clip.actor_ids != ACTOR_IDS or clip.fps != 20:
        raise ValueError("expected a synchronized two-actor Core27 clip at 20 fps")
    observed_frames = clip.positions.shape[1]
    if observed_frames != planned_frames:
        return {"status": "not_verified", "reason": "encounter clip is incomplete" if observed_frames < planned_frames
                else "clip exceeds planned encounter", "planned_frames": planned_frames,
                "observed_frames": observed_frames, "contact_mode": CONTACT_MODE}
    adapted = adapt_studio_scene(scene)
    seams = tuple(range(HORIZON, observed_frames, HORIZON))
    cast = {}
    geometry_ok = clip.source == "ardy_core" and clip.native_features is not None
    meeting_frame = plan_report["planned_meeting_frame"]
    if type(meeting_frame) is not int or not 0 <= meeting_frame < observed_frames:
        raise ValueError("plan report has no valid meeting frame")
    facing = {}
    for index, aid in enumerate(ACTOR_IDS):
        poses = clip.positions[index]
        root = poses[:, 0, :][:, (0, 2)]
        try:
            support = validate_ground_path(adapted["scene"], root)
        except ValueError as exc:
            support = {"continuous_support_verified": False, "error": str(exc)}
            geometry_ok = False
        if not support.get("continuous_support_verified", False):
            geometry_ok = False
        collision = scene_collision(poses, "core27", adapted["scene"], adapted["affordances"])
        floor_range = support.get("floor_height_range_m")
        floor_height = (sum(floor_range) / 2 if floor_range is not None else 0.)
        floor = floor_motion(poses, skeleton="core27", fps=20, floor_y=floor_height)
        # This encounter has no airborne beats. Check each frame so one brief
        # toe touch cannot certify a clip that floats for most of the scene.
        toe_indices = (joint_index("core27", "LeftToeBase"),
                       joint_index("core27", "RightToeBase"))
        toe_heights = [min(float(poses[frame, toe, 1] - floor_height)
                           for toe in toe_indices) for frame in range(observed_frames)]
        near_floor = [(-.05 <= height <= .15) for height in toe_heights]
        unsupported_run = longest_unsupported_run = 0
        for grounded in near_floor:
            unsupported_run = 0 if grounded else unsupported_run + 1
            longest_unsupported_run = max(longest_unsupported_run, unsupported_run)
        near_floor_fraction = sum(near_floor) / observed_frames
        vertical_support_proxy_pass = (floor_range is not None and
                                       floor["toe_penetration_max_depth_m"] <= .05 and
                                       near_floor_fraction >= .85 and
                                       longest_unsupported_run <= 8)
        smoothness = continuity(poses, skeleton="core27", fps=20, horizon_boundaries=seams)
        # A metre-scale teleport within a horizon is invisible to seam-only
        # checks; 0.25 m per 20 Hz frame is a generous root-motion ceiling.
        root_continuity_proxy_pass = (smoothness["root_peak_step_m"] is not None and
                                      smoothness["root_peak_step_m"] <= .25)
        destination = plan_report["initial_placements"][aid]["position_xz"]
        endpoint = goal_endpoint(poses, skeleton="core27", target_xz=destination, tolerance_m=.35)
        meet_position = plan_report["meeting_positions_xz"][aid]
        meeting_error = math.dist(root[meeting_frame], meet_position)
        if (collision["total_collision_frames"] or not endpoint["within_tolerance"] or
                meeting_error > .45 or not vertical_support_proxy_pass or
                not root_continuity_proxy_pass):
            geometry_ok = False
        cast[aid] = {"scene_collision": collision, "ground_support": support,
                     "floor_motion": floor, "continuity": smoothness,
                     "vertical_support_proxy_pass": vertical_support_proxy_pass,
                     "near_floor_frame_fraction": near_floor_fraction,
                     "longest_unsupported_run_frames": longest_unsupported_run,
                     "root_continuity_proxy_pass": root_continuity_proxy_pass,
                     "departure_endpoint": endpoint,
                     "meeting_root_error_xz_m": float(meeting_error),
                     "observed_root_path_length_m": float(sum(
                         math.dist(before, after) for before, after in zip(root, root[1:]))),
                     "horizon_seam_max_root_step_m": max((x["root_step_m"] for x in smoothness["horizon_seams"]), default=None),
                     "horizon_seam_max_mean_joint_step_m": max((x["mean_joint_step_m"] for x in smoothness["horizon_seams"]), default=None)}
        other = ACTOR_IDS[1 - index]
        toward = clip.positions[1 - index, meeting_frame, 0, (0, 2)] - root[meeting_frame]
        intended = math.atan2(float(toward[0]), float(toward[1]))
        forward = clip.rotations[index, meeting_frame, 0, :, 2]
        actual = math.atan2(float(forward[0]), float(forward[2]))
        error = abs((actual - intended + math.pi) % (2 * math.pi) - math.pi)
        facing[aid] = {"toward_actor_id": other, "error_degrees": math.degrees(error),
                       "within_45_degrees": error <= math.pi / 4}
        if not facing[aid]["within_45_degrees"]:
            geometry_ok = False
    pair = pair_separation(clip.positions[0], clip.positions[1],
                           skeleton_a="core27", skeleton_b="core27",
                           radius_a_m=.325, radius_b_m=.325)
    if pair["root_disc_overlap_proxy_frames"] or pair["min_root_separation_xz_m"] < MIN_PLANNED_SEPARATION_M:
        geometry_ok = False
    action_measurements = []
    for index, (actions, targets) in enumerate(zip(
            plan_report["planned_stage_actions"], plan_report["planned_stage_root_targets"])):
        for actor_index, aid in enumerate(ACTOR_IDS):
            errors = [math.dist(clip.positions[actor_index, index * HORIZON + goal["frame"], 0, (0, 2)],
                                goal["position_xz"]) for goal in targets[aid]]
            max_error = max(errors)
            route_target_proxy_pass = max_error <= .45
            if not route_target_proxy_pass:
                geometry_ok = False
            action_measurements.append({"stage_index": index, "actor_id": aid,
                                        "action": actions[aid], "seed": plan_report["seed"] + index,
                                        "root_target_error_mean_m": sum(errors) / len(errors),
                                        "root_target_error_max_m": max_error,
                                        "route_target_proxy_pass": route_target_proxy_pass,
                                        "status": "measured_geometry_only",
                                        "visual_review_pass": None,
                                        "instruction_followed": None,
                                        "output": None})
    return {"status": "measured", "planned_frames": planned_frames,
            "observed_frames": observed_frames, "contact_mode": CONTACT_MODE,
            "geometry_checks_pass": geometry_ok,
            "native_source_recorded": clip.source == "ardy_core" and clip.native_features is not None,
            "actors": cast, "pair_separation": pair,
            "mutual_facing_at_meeting": facing,
            "action_measurements": action_measurements,
            "physical_contact_verified": False,
            "visual_realism_verified": False,
            "note": "Sampled geometry is a proxy; staged motions have no controlled physical contact."}
