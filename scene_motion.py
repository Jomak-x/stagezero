"""Scene-grounded locomotion for the pinned G1 motion representation.

The generator supplies the body performance. This layer routes locomotion over
rendered support geometry, plants alternating feet, solves the two leg chains,
and re-encodes the corrected pose as valid autoregressive history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import numpy as np

from take_sequencing import motion_statistics

FPS = 25
WALK_SPEED = .35
FEET = (6, 7, 13, 14)
PARENTS = (-1, 0, 1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13,
           0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 17, 26, 27, 28, 29, 30, 31, 32)


@dataclass
class SceneMotionPlan:
    geometry: object
    backend_prompt: str
    route: np.ndarray | None
    start: np.ndarray
    stair_name: str | None = None
    progress: float = 0.
    frames: int = 0
    anchors: dict = field(default_factory=dict)
    swing: dict = field(default_factory=dict)
    last_positions: np.ndarray | None = None
    last_heading: float | None = None
    stopped: bool = False
    settle_frame: int | None = None
    leg_pose: np.ndarray | None = None
    leg_rotations: np.ndarray | None = None

    @property
    def recommended_seconds(self):
        return None if self.route is None else float(_route_lengths(self.route)[-1] / WALK_SPEED + 1.92)


def _pose(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 3:
        value = value[-1]
    if value.shape != (34, 3) or not np.isfinite(value).all():
        raise ValueError('Scene interaction requires a finite G1 starting pose')
    return value.copy()


def _heading(p):
    d = p[8] - p[1]
    return float(np.arctan2(d[2], -d[0]))


def _yaw(a):
    c, s = np.cos(a), np.sin(a)
    return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))


def _route_lengths(route):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(route[:, [0, 2]], axis=0), axis=1))]


def _sample(route, distance):
    lengths = _route_lengths(route)
    distance = min(max(distance, 0.), lengths[-1])
    i = min(np.searchsorted(lengths, distance, side='right') - 1, len(route) - 2)
    i = max(i, 0)
    delta = route[i + 1] - route[i]
    t = (distance - lengths[i]) / max(lengths[i + 1] - lengths[i], 1e-9)
    direction = delta[[0, 2]] / max(np.linalg.norm(delta[[0, 2]]), 1e-9)
    return route[i] + delta * t, direction


def plan_scene_motion(scene, prompt, start_positions):
    """Capture scene geometry and a route; unrelated gestures remain untouched.

    Stair commands require actual stairs. Descending from the ground does not
    teleport an actor onto the upper landing. The caller keeps this plan for
    every generation chunk belonging to the request. Generic grounding is
    limited to walking; running/jogging keep their generated gait until a
    dedicated contact solver supports those flight phases. Explicit stair
    requests always use the controlled stair gait.
    """
    text = str(prompt).lower()
    stairs = bool(re.search(r'\b(stairs?|staircase|upstairs|downstairs)\b', text) or re.search(r'\b(?:up|down|climb\w*|ascend\w*|descend\w*|stone|temple)\b.{0,25}\bsteps\b', text))
    walking = bool(re.search(r'\bwalk\w*\b', text))
    if not stairs and not walking:
        return None
    from scene_interaction_geometry import SceneInteractionGeometry
    geometry = SceneInteractionGeometry.from_scene(scene)
    start = _pose(start_positions)
    if not stairs and not geometry.walkable_surfaces:
        return None
    if not stairs:
        return SceneMotionPlan(geometry, prompt, None, start)
    if not geometry.stair_routes:
        raise ValueError('This scene has no reachable stair geometry. Add stairs before asking the actor to climb them.')
    descending = bool(re.search(r'\b(down|downstairs|descend\w*)\b', text))
    candidates = []
    ground = float(np.min(start[list(FEET), 1]))
    for stair in geometry.stair_routes:
        steps = np.asarray(stair['steps'], dtype=float)
        if len(steps) < 2:
            continue
        if np.max(np.abs(np.diff(steps[:, 1]))) > .50:
            continue
        if descending:
            steps = steps[::-1]
        direction = steps[1, [0, 2]] - steps[0, [0, 2]]
        direction /= max(np.linalg.norm(direction), 1e-9)
        already_on_flight = False
        # Start at the nearest compatible tread when already on this flight.
        compatible = np.flatnonzero(np.abs(steps[:, 1] - ground) < .20)
        if len(compatible):
            k = compatible[np.argmin(np.linalg.norm(steps[compatible][:, [0, 2]] - start[0, [0, 2]], axis=1))]
            if np.linalg.norm(steps[k, [0, 2]] - start[0, [0, 2]]) < .65:
                steps = steps[k:]
                already_on_flight = True
        entry = steps[0].copy()
        entry[[0, 2]] -= direction * .60
        height = geometry.support_height(entry[0], entry[2], ground, max_step_up=.50, max_drop=.6)
        if height is None or (descending and abs(steps[0, 1] - ground) > .50):
            continue
        entry[1] = height
        approach = [] if already_on_flight else [entry]
        exit_point = steps[-1].copy()
        exit_point[[0, 2]] += direction * (.65 if descending else .35)
        exit_height = geometry.support_height(exit_point[0], exit_point[2], steps[-1, 1], max_step_up=.50, max_drop=.60)
        landing = []
        if exit_height is not None and not geometry.obstacle_at(exit_point[0], exit_height + .4, exit_point[2], radius=.16):
            exit_point[1] = exit_height
            landing = [exit_point]
        route = np.vstack([[start[0, 0], ground, start[0, 2]], *approach, *steps, *landing])
        # Refuse paths that cut through solid scenery or jump to tall platforms.
        valid = True
        for a, b in zip(route[:-1], route[1:]):
            previous_height = a[1]
            for t in np.linspace(0., 1., max(2, int(np.linalg.norm((b-a)[[0, 2]]) / .10) + 1)):
                p = a + t * (b-a)
                h = geometry.support_height(p[0], p[2], previous_height, max_step_up=.50, max_drop=.6)
                if h is None or geometry.obstacle_at(p[0], h + .4, p[2], radius=.16,
                                                    ignore_object_ids={stair['object_id']}):
                    valid = False
                    break
                previous_height = h
            if not valid:
                break
        if valid:
            candidates.append((_route_lengths(route)[-1], route, stair['name']))
    if not candidates:
        raise ValueError('The stairs cannot be reached safely from this position. Move the actor to the stair entrance first.')
    _, route, name = min(candidates, key=lambda item: item[0])
    gait = 'A person walks slowly with alternating steps and a relaxed upright body.'
    return SceneMotionPlan(geometry, gait, route, start, name)


def _between(a, b):
    """Shortest proper rotation mapping a onto b, including antiparallel bones."""
    a, b = np.asarray(a), np.asarray(b)
    if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-9:
        return np.eye(3)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.clip(a @ b, -1, 1))
    if c < -.999999:
        axis = np.cross(a, np.eye(3)[np.argmin(abs(a))])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    k = np.array(((0, -v[2], v[1]), (v[2], 0, -v[0]), (-v[1], v[0], 0)))
    return np.eye(3) + k + k @ k / max(1 + c, 1e-9)


def _rotate_subtree(p, r, joint, descendants, rotation):
    p[descendants] = (p[descendants] - p[joint]) @ rotation.T + p[joint]
    r[[joint] + descendants] = rotation @ r[[joint] + descendants]


def _leg_ik(p, r, hip, knee, ankle, end, target):
    """Two-bone IK using rigid subtree transforms, preserving all bone offsets."""
    a, b = np.linalg.norm(p[knee] - p[hip]), np.linalg.norm(p[ankle] - p[knee])
    if min(a, b) < 1e-5:
        return
    delta = target - p[hip]
    d = np.linalg.norm(delta)
    direction = delta / max(d, 1e-9)
    d = np.clip(d, abs(a - b) + 1e-5, a + b - 1e-5)
    bend = p[knee] - p[hip]
    bend -= (bend @ direction) * direction
    if np.linalg.norm(bend) < 1e-6:
        bend = np.cross(direction, (1., 0., 0.))
    bend /= max(np.linalg.norm(bend), 1e-9)
    along = (a*a - b*b + d*d) / (2*d)
    knee_target = p[hip] + along * direction + np.sqrt(max(a*a - along*along, 0.)) * bend
    foot_rotation = r[ankle].copy()
    _rotate_subtree(p, r, hip, list(range(knee, end + 1)), _between(p[knee] - p[hip], knee_target - p[hip]))
    _rotate_subtree(p, r, knee, list(range(ankle, end + 1)), _between(p[ankle] - p[knee], target - p[knee]))
    _rotate_subtree(p, r, ankle, list(range(ankle + 1, end + 1)), foot_rotation @ r[ankle].T)


def encode_scene_motion(positions, rotations, geometry):
    """Encode ARDY global rotations, ground-origin joints, velocities, contacts."""
    p, r = np.asarray(positions), np.asarray(rotations)
    n = len(p)
    f = np.zeros((n, 414), dtype=np.float32)
    f[:, :3] = p[:, 0]
    heading = np.arctan2((p[:, 8] - p[:, 1])[:, 2], -(p[:, 8] - p[:, 1])[:, 0])
    f[:, 3:5] = np.stack((np.cos(heading), np.sin(heading)), axis=-1)
    local = p[:, 1:].copy()
    local[..., 0] -= p[:, None, 0, 0]
    local[..., 2] -= p[:, None, 0, 2]
    f[:, 5:104] = local.reshape(n, 99)
    f[:, 104:308] = np.concatenate((r[..., 0], r[..., 1]), axis=-1).reshape(n, 204)
    velocity = np.zeros_like(p)
    if n > 1:
        velocity[:-1] = np.diff(p, axis=0) * FPS
        velocity[-1] = velocity[-2]
    f[:, 308:410] = velocity.reshape(n, 102)
    for frame in range(n):
        for col, joint in enumerate(FEET):
            point = p[frame, joint]
            h = geometry.support_height(point[0], point[2], point[1], max_step_up=.03, max_drop=.15)
            f[frame, 410 + col] = float(h is not None and abs(point[1] - h) < .05 and np.linalg.norm(velocity[frame, joint]) < .20)
    mean, scale = motion_statistics()
    return ((f - mean) / scale).astype(np.float32)


def apply_scene_motion(result, plan, prior_positions=None):
    """Correct only new frames; the supplied prior pose/history is never mutated."""
    if plan is None:
        return result
    p = np.asarray(result['positions'], dtype=np.float64).copy()
    r = np.asarray(result['rotations'], dtype=np.float64).copy()
    if p.ndim != 3 or p.shape[1:] != (34, 3) or r.shape != (*p.shape[:2], 3, 3):
        raise ValueError('Scene grounding requires G1 positions and global rotations')
    if plan.leg_pose is None:
        plan.leg_pose = p[0].copy()
        plan.leg_rotations = r[0].copy()
    prior = _pose(prior_positions) if prior_positions is not None else (plan.last_positions if plan.last_positions is not None else plan.start)
    root = prior[0].copy()
    heading = plan.last_heading if plan.last_heading is not None else _heading(prior)
    ground_start = np.min(prior[list(FEET), 1])
    body_height = float(np.clip(plan.start[0, 1] - np.min(plan.start[list(FEET), 1]), .5, 1.05))
    stride_frames = 16
    for i in range(len(p)):
        previous_progress = plan.progress
        source = p[i].copy()
        old_heading = _heading(source)
        if plan.route is not None:
            candidate, direction = _sample(plan.route, plan.progress + WALK_SPEED / FPS)
            next_root = candidate[[0, 2]]
            if np.linalg.norm(next_root - root[[0, 2]]) > WALK_SPEED / FPS + .002:
                next_root = root[[0, 2]] + (next_root - root[[0, 2]]) / np.linalg.norm(next_root - root[[0, 2]]) * WALK_SPEED / FPS
            desired_heading = np.arctan2(direction[0], direction[1])
            plan.progress = min(plan.progress + WALK_SPEED / FPS, _route_lengths(plan.route)[-1])
        else:
            displacement = source[0, [0, 2]] - (np.asarray(result['positions'])[max(i-1, 0), 0, [0, 2]])
            length = np.linalg.norm(displacement)
            displacement *= min(1., WALK_SPEED / FPS / max(length, 1e-9))
            next_root = root[[0, 2]] + displacement
            desired_heading = old_heading
        support = plan.geometry.support_height(next_root[0], next_root[1], ground_start, max_step_up=.50, max_drop=.6)
        if support is None or plan.geometry.obstacle_at(next_root[0], support + .4, next_root[1], radius=.16):
            next_root = root[[0, 2]]
            support = ground_start
            plan.stopped = True
            plan.progress = previous_progress
        heading += np.clip(np.arctan2(np.sin(desired_heading-heading), np.cos(desired_heading-heading)), -.10, .10)
        rotation = _yaw(heading - old_heading)
        p[i] = (source - source[0]) @ rotation.T
        root = np.array((next_root[0], root[1] + np.clip(support + body_height - root[1], -.025, .025), next_root[1]))
        p[i] += root
        r[i] = rotation @ r[i]
        leg_rotation = _yaw(heading - _heading(plan.leg_pose))
        hip_origins = p[i, [1, 8]].copy()
        p[i, 1:15] = (plan.leg_pose[1:15] - plan.leg_pose[0]) @ leg_rotation.T + root
        r[i, 1:15] = leg_rotation @ plan.leg_rotations[1:15]
        p[i, 1:8] += hip_origins[0] - p[i, 1]
        p[i, 8:15] += hip_origins[1] - p[i, 8]
        ground_start = support
        if plan.route is not None and plan.progress >= _route_lengths(plan.route)[-1] - .001 and plan.settle_frame is None:
            plan.settle_frame = ((plan.frames // stride_frames) + 3) * stride_frames
        settled = plan.settle_frame is not None and plan.frames >= plan.settle_frame
        phase = plan.frames % stride_frames
        moving_leg = (plan.frames // stride_frames) % 2
        targets = []
        for leg, (hip, knee, ankle, end) in enumerate(((3, 4, 5, 7), (10, 11, 12, 14))):
            # Level heel/toe together before IK. This is also a rigid subtree edit.
            toe = p[i, end] - p[i, end-1]
            flat = toe.copy(); flat[1] = 0
            _rotate_subtree(p[i], r[i], ankle, [ankle+1, end], _between(toe, flat))
            offsets = p[i, [ankle+1, end]] - p[i, ankle]
            if leg not in plan.anchors:
                plan.anchors[leg] = prior[ankle].copy()
            if phase == 0 and leg == moving_leg and not settled:
                if plan.route is not None:
                    future, tangent = _sample(plan.route, plan.progress + WALK_SPEED * stride_frames / FPS)
                    side = .10 if leg == 0 else -.10
                    target_xz = future[[0, 2]] + side * np.array((tangent[1], -tangent[0]))
                else:
                    target_xz = root[[0, 2]] + (_yaw(heading) @ np.array((.10 if leg == 0 else -.10, 0., .24)))[[0, 2]]
                if plan.geometry.obstacle_at(target_xz[0], support + .4, target_xz[1], radius=.12):
                    target_xz = plan.anchors[leg][[0, 2]]
                endpoint = p[i, ankle].copy(); endpoint[[0, 2]] = target_xz
                heights = [plan.geometry.support_height(*(endpoint[[0, 2]] + off[[0, 2]]), support, max_step_up=1.0, max_drop=.6) for off in offsets]
                if any(h is None for h in heights):
                    endpoint = plan.anchors[leg].copy()
                else:
                    endpoint[1] = max(h - off[1] for h, off in zip(heights, offsets)) + .015
                plan.swing[leg] = (plan.anchors[leg].copy(), endpoint)
            target = plan.anchors[leg].copy()
            if leg == moving_leg and leg in plan.swing and not settled:
                begin, end_point = plan.swing[leg]
                t = (phase + 1) / stride_frames
                blend = t*t*(3-2*t)
                target = begin*(1-blend) + end_point*blend
                target[1] += np.sin(np.pi*t) * (.08 + min(abs(end_point[1]-begin[1]) * .20, .06))
                if phase == stride_frames - 1:
                    plan.anchors[leg] = end_point.copy()
            # Clear the actual tread under the entire foot throughout swing.
            clearance = []
            for off in offsets:
                q = target + off
                h = plan.geometry.support_height(q[0], q[2], max(support, q[1]), max_step_up=.60, max_drop=.6)
                if h is not None:
                    clearance.append(h - off[1] + .015)
            if clearance:
                target[1] = max(target[1], max(clearance))
            targets.append((hip, knee, ankle, end, target))
        # A planted foot must remain reachable: lower the pelvis before solving.
        lower_y, upper_y = -np.inf, np.inf
        reach_constraints = []
        for hip, knee, ankle, end, target in targets:
            reach = np.linalg.norm(p[i, knee]-p[i, hip]) + np.linalg.norm(p[i, ankle]-p[i, knee]) - .001
            horizontal = np.linalg.norm((p[i, hip]-target)[[0, 2]])
            if reach <= horizontal:
                raise ValueError('The next scene foot placement is out of reach; move closer to the stair entrance.')
            min_reach = abs(np.linalg.norm(p[i, knee]-p[i, hip]) - np.linalg.norm(p[i, ankle]-p[i, knee])) + .001
            vertical = np.sqrt(reach*reach-horizontal*horizontal)
            offset_y = p[i, hip, 1] - p[i, 0, 1]
            reach_constraints.append((target[1] - offset_y, horizontal, min_reach, reach))
            lower_y = max(lower_y, target[1] - vertical - offset_y)
            upper_y = min(upper_y, target[1] + vertical - offset_y)
        if lower_y > upper_y:
            raise ValueError('The next stair rise is too large for the actor to maintain foot contact.')
        candidate_y = [float(np.clip(p[i, 0, 1], lower_y, upper_y)), lower_y, upper_y]
        for center_y, horizontal, minimum, maximum in reach_constraints:
            if horizontal < minimum:
                gap = np.sqrt(minimum*minimum-horizontal*horizontal)
                candidate_y.extend((center_y-gap, center_y+gap))
        feasible = [y for y in candidate_y if lower_y-1e-8 <= y <= upper_y+1e-8 and all(
            minimum-1e-8 <= np.hypot(y-center_y, horizontal) <= maximum+1e-8
            for center_y, horizontal, minimum, maximum in reach_constraints)]
        if not feasible:
            raise ValueError('The next stair pose cannot maintain both foot contacts safely.')
        # Prefer the upright IK branch whenever both planted legs allow it;
        # otherwise the minimum-reach circle can trap the pelvis below a foot.
        upright = [y for y in feasible if abs(y-p[i, 0, 1]) <= .06 and all(y >= center_y for center_y, _, _, _ in reach_constraints)]
        pelvis_y = min(upright or feasible, key=lambda y: abs(y-p[i, 0, 1]))
        p[i, :, 1] += pelvis_y - p[i, 0, 1]
        for args in targets:
            _leg_ik(p[i], r[i], *args)
        root = p[i, 0].copy()
        plan.frames += 1
    plan.last_positions = p[-1].copy()
    plan.last_heading = heading
    output = dict(result)
    output['positions'], output['rotations'] = p.astype(np.float32), r.astype(np.float32)
    output['motion'] = encode_scene_motion(p, r, plan.geometry)
    if plan.stopped:
        label = 'Stopped before obstacle'
    elif plan.settle_frame is not None and plan.frames >= plan.settle_frame:
        label = 'Reached the landing of ' + str(plan.stair_name)
    elif plan.route is None:
        label = 'Walking on scene surfaces'
    else:
        action = ('Descending ' if plan.route[-1, 1] < plan.route[0, 1] else 'Climbing ')
        label = (action if abs(ground_start - plan.route[0, 1]) > .15 else 'Approaching ') + str(plan.stair_name)
    output['metadata'] = dict(result.get('metadata', {}), scene_motion=label, scene_interaction={
        'mode': 'geometry_grounded_locomotion', 'stairs': plan.stair_name,
        'distance_along_route_m': round(plan.progress, 4), 'blocked': plan.stopped,
        'route_complete': bool(plan.route is not None and plan.progress >= _route_lengths(plan.route)[-1] - .001),
    })
    return output
