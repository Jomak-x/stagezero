"""Explicit OFFLINE assisted terrain presentation proof, never native ARDY output.

Uses the actual scene support mesh, an explicit alternating footstep schedule,
and neutral Core FK. Does not mutate native features/history; not streaming.
The output requires full-speed visual acceptance before any runtime use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

from motion_bridge import _layout, _swing, _two_bone
from scene_interaction_geometry import SceneInteractionGeometry

NAMES, PARENTS, NEUTRAL = _layout()
INDEX = {name: i for i, name in enumerate(NAMES)}
OFFSETS = NEUTRAL - NEUTRAL[np.maximum(PARENTS, 0)]
LEGS = tuple(tuple(INDEX[side + name] for name in ('UpLeg', 'Leg', 'Foot', 'ToeBase'))
             for side in ('Left', 'Right'))
LEG_JOINTS = {j for leg in LEGS for j in leg}


def _runs(mask):
    d = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _fk(root, rotations):
    p = np.empty((27, 3))
    p[0] = root
    for j in range(1, 27):
        p[j] = p[PARENTS[j]] + rotations[PARENTS[j]] @ OFFSETS[j]
    return p


def _sole(ankle, rotation):
    toe = OFFSETS[LEGS[0][3]]
    return ankle + np.array([[x, toe[1], z] for x in (-.045, .045)
                            for z in (-.035, toe[2])]) @ rotation.T


def _support(geometry, point):
    height = geometry.support_height(float(point[0]), float(point[2]), float(point[1]),
                                     max_step_up=1., max_drop=1.)
    if height is None:
        raise ValueError('Assisted foot path leaves authored support; no implicit ground')
    return float(height)


def _route_frames(native, rotations, fps):
    """Planning arclength and heading; never changes the native root path."""
    path = gaussian_filter1d(native[:, 0][:, [0, 2]], .8, axis=0)
    velocity = np.gradient(path, axis=0)*fps
    speed = np.linalg.norm(velocity, axis=1)
    moving = speed > .18
    for a, b in _runs(~moving):
        if a and b < len(path) and b-a < max(3, round(.65*fps)):
            moving[a:b] = True
    for a, b in _runs(moving):
        if b-a < 3:
            moving[a:b] = False
    distance = np.linalg.norm(np.diff(path, axis=0), axis=1)
    distance[~(moving[:-1] | moving[1:])] = 0.
    progress = np.r_[0., np.cumsum(distance)]
    active = np.flatnonzero(moving & (speed > max(.2, .3*float(speed.max()))))
    if len(active):
        angles = np.unwrap(np.arctan2(velocity[active, 0], velocity[active, 1]))
        yaw = np.interp(np.arange(len(path)), active, angles)
        yaw = gaussian_filter1d(yaw, 1.5)
    else:
        forward = rotations[:, 0, :, 2]
        yaw = np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))
    return path, progress, yaw, _runs(moving)


def _plan_contacts(native, rotations, fps, initial_p, initial_r, *, initial_is_continuation=False,
                   minimum_transfer_frames=None, touchdown_lead=.18, minimum_stance_frames=4,
                   heading_assistance=False):
    path, progress, yaw, bouts = _route_frames(native, rotations, fps)
    native_yaw = np.unwrap(np.arctan2(rotations[:, 0, 0, 2], rotations[:, 0, 2, 2]))
    count = len(native)
    swing_frames = max(4, round(.4*fps))
    separation = max(3, round(.3*fps))
    if minimum_transfer_frames is not None:
        separation = max(separation, int(minimum_transfer_frames))
    contacts = [[(0, initial_p[leg[3]].copy(), initial_r[leg[2]].copy())] for leg in LEGS]
    # For an unconstrained first pose, flatten the initial foot using its own
    # heading. A supplied accepted pose keeps the full foot rotation exactly.
    next_side = 1
    last_event = -separation
    unique, indices = np.unique(progress, return_index=True)
    def sample(along):
        clipped = np.clip(along, unique[0], unique[-1])
        point = np.array([np.interp(clipped, unique, path[indices, axis]) for axis in (0, 1)])
        angle = float(np.interp(clipped, unique, yaw[indices]))
        if along > unique[-1]:
            point += (along-unique[-1])*np.array([np.sin(angle), np.cos(angle)])
        return point, angle
    def pivot_during_hold(a, b, *, followed_by_walk=False):
        nonlocal last_event, next_side
        if b-a < 3:
            return
        # Native yaw may contain brief oscillations while the character is
        # standing. Smooth for planning only; native rotations stay unchanged.
        turn = gaussian_filter1d(native_yaw[a:b], 1.5)
        foot_angles = [np.arctan2(entries[-1][2][0, 2], entries[-1][2][2, 2])
                       for entries in contacts]
        anchor_yaw = float(np.angle(np.mean(np.exp(1j*np.array(foot_angles)))))
        route_directed = heading_assistance and followed_by_walk
        if route_directed:
            # Explicit world-heading assistance gives travel direction priority
            # over native pelvis overshoot during a buffered walking lead-in.
            # Turn from the inherited contacts along the shortest arc, keeping
            # ordinary/native-heading pivots subject to the reversal check.
            delta = float(np.arctan2(np.sin(yaw[b]-anchor_yaw), np.cos(yaw[b]-anchor_yaw)))
            if np.isclose(abs(delta), np.pi, atol=1e-7, rtol=0):
                raise ValueError('Stationary pivot has an ambiguous half-turn route heading')
            phase = np.linspace(0., 1., b-a)
            turn = anchor_yaw+delta*phase*phase*(3.-2.*phase)
        delta = float(turn[-1]-turn[0])
        if abs(delta) < np.radians(30):
            if route_directed:
                yaw[a:b] = turn
            return
        if not followed_by_walk and np.ptp(turn[-min(5, len(turn)):]) > np.radians(5):
            raise ValueError('Stationary pivot needs a settled final heading')
        direction = np.sign(delta)
        if np.any(np.diff(turn)*direction < -np.radians(2)):
            raise ValueError('Stationary pivot reverses direction within one hold')
        # Native pelvis orientation can differ from the accepted foot heading
        # (for example while reaching). Transport its turn delta from the
        # current contacts; treating it as absolute foot yaw snaps the legs.
        anchor_yaw += 2*np.pi*round((turn[0]-anchor_yaw)/(2*np.pi))
        turn = turn+(anchor_yaw-turn[0])
        yaw[a:b] = turn
        if followed_by_walk and not route_directed:
            # Velocity gives the next walking heading. Join it during the
            # stationary tail rather than switching the knee pole at b.
            destination = yaw[b]+2*np.pi*round((turn[-1]-yaw[b])/(2*np.pi))
            width = min(b-a, max(4, round(.6*fps)))
            phase = np.linspace(0., 1., width)
            blend = phase*phase*(3-2*phase)
            yaw[b-width:b] = (1-blend)*yaw[b-width:b]+blend*destination
        # Admission follows lifted-contact timing below and the actual rig's
        # unchanged rotation/reach/contact gates, not a blanket pelvis rate.
        # Plant each foot after a lifted transfer. If walking immediately
        # follows, its first alternating steps complete the reorientation.
        # Otherwise the trailing foot needs one final settling event.
        sections = max(1, int(np.ceil(abs(delta)/np.radians(30))))
        angles = np.linspace(turn[0], turn[-1], sections+1)[1:]
        targets = list(angles[:-1]) if followed_by_walk else [*angles, turn[-1]]
        for event_index, target_angle in enumerate(targets):
            target_t = a+int(np.argmin(abs(turn-target_angle)))
            t = max(target_t, last_event+separation, swing_frames+2,
                    contacts[next_side][-1][0]+swing_frames+3)
            lag = swing_frames+2 if not followed_by_walk and event_index == len(angles) else round(.2*fps)
            if t > target_t+lag or t >= b-2:
                raise ValueError('Stationary pivot needs more time for alternating lifted contacts')
            angle = float(turn[t-a]) if target_angle != turn[-1] else float(turn[-1])
            lateral = np.array([-np.cos(angle), np.sin(angle)])
            point = native[t, 0].copy()
            point[1] += float(NEUTRAL[LEGS[next_side][3], 1])
            point[[0, 2]] += (-1 if next_side == 0 else 1)*.13*lateral
            contacts[next_side].append((t, point, Rotation.from_euler('y', angle).as_matrix()))
            last_event, next_side = t, 1-next_side
    previous_end = 0
    for bout_index, (start, end) in enumerate(bouts):
        pivot_during_hold(previous_end, start, followed_by_walk=True)
        previous_end = end
        first, last = float(progress[start]), float(progress[end-1])
        if last-first < .08:
            continue
        forward = np.array([np.sin(yaw[start]), np.cos(yaw[start])])
        lead = float(np.mean([(entries[-1][1][[0, 2]]-path[start])@forward for entries in contacts]))
        # A restarted two-foot hold may have its toes beneath the pelvis,
        # unlike the initial native stand. Shorten that first transfer using
        # the actual current anchors so the trailing leg remains reachable.
        first_stride = .6 if bout_index == 0 and not initial_is_continuation and last_event < 0 else .4+float(np.clip(lead, 0., .2))
        alongs = list(np.arange(first+first_stride, last-.15, .5))+[last+.015]
        # Redistribute a large final remainder instead of either dropping the
        # penultimate landing or blindly adding one that overruns the tail.
        # Short approaches get one extra contact only when genuinely needed.
        if len(alongs) > 1 and max(np.diff(alongs)) > .52:
            needed = max(len(alongs), 1+int(np.ceil((alongs[-1]-alongs[0])/.52)))
            alongs = np.linspace(alongs[0], alongs[-1], needed).tolist()
        limit = bouts[bout_index+1][0]-2 if bout_index+1 < len(bouts) else count-3
        for along in alongs:
            candidates = np.arange(start, end)
            t = int(candidates[np.argmin(abs(progress[candidates]-(along-touchdown_lead)))])
            t = max(t, last_event+separation, swing_frames+2,
                    contacts[next_side][-1][0]+swing_frames+minimum_stance_frames-1)
            if t >= limit:
                raise ValueError('Insufficient buffered time for alternating planted contacts')
            point2, angle = sample(along)
            lateral = np.array([-np.cos(angle), np.sin(angle)])
            point = native[t, 0].copy()
            point[1] += float(NEUTRAL[LEGS[next_side][3], 1])
            point[[0, 2]] = point2+(-1 if next_side == 0 else 1)*.13*lateral
            rot = Rotation.from_euler('y', angle).as_matrix()
            contacts[next_side].append((t, point, rot))
            last_event, next_side = t, 1-next_side
        # Settle the other foot beside the final support before a hold/restart.
        t = max(last_event+swing_frames, contacts[next_side][-1][0]+swing_frames+3)
        if t >= limit:
            raise ValueError('Clip needs a longer stationary tail to finish both feet')
        point2, angle = sample(last+.015)
        lateral = np.array([-np.cos(angle), np.sin(angle)])
        point = native[t, 0].copy()
        point[1] += float(NEUTRAL[LEGS[next_side][3], 1])
        point[[0, 2]] = point2+(-1 if next_side == 0 else 1)*.13*lateral
        contacts[next_side].append((t, point, Rotation.from_euler('y', angle).as_matrix()))
        last_event, next_side = t, 1-next_side
    pivot_during_hold(previous_end, count)
    runs = []
    all_events = sorted(t for entries in contacts for t, _, _ in entries if t)
    for entries in contacts:
        side_runs = []
        for i, (t, point, rot) in enumerate(entries):
            end = count
            if i+1 < len(entries):
                landing = entries[i+1][0]
                preceding = [event for event in all_events if event < landing]
                # Match swing time to route cadence. A fixed .4s swing makes
                # slower routes keep the trailing foot planted beyond reach.
                gap = landing-preceding[-1] if preceding else 0
                duration = min(round(.7*fps), gap) if preceding and gap <= round(.9*fps) else swing_frames
                duration = max(4, duration)
                end = landing-duration+1
                if end-t < 3:
                    raise ValueError('Route cadence leaves no stable support interval')
            side_runs.append((t, end, point, rot))
        runs.append(side_runs)
    return runs, yaw, bouts


def assist_clip(native_positions, native_rotations, geometry, *, fps=20.,
                initial_assisted_positions=None, initial_assisted_rotations=None):
    """Return FK-consistent presentation poses and numerical evidence, separately.

    Alternating contacts follow native root progress, replacing native cadence.
    Full-clip lookahead allows smooth swing endpoints.
    A global smooth root-Y solve enforces reach without greedy frame snapping.
    """
    native = np.asarray(native_positions, dtype=float)
    original_r = np.asarray(native_rotations, dtype=float)
    if native.shape[1:] != (27, 3) or original_r.shape != (len(native), 27, 3, 3):
        raise ValueError('Expected Core27 positions and global rotation matrices')
    if len(native) < 3 or not np.isfinite(native).all() or not np.isfinite(original_r).all():
        raise ValueError('Need at least three finite frames')
    fk_error = max(np.max(np.linalg.norm(_fk(native[t, 0], original_r[t])-native[t], axis=1))
                   for t in range(len(native)))
    if fk_error > .001:
        raise ValueError('Input positions and rotations do not describe neutral Core FK')
    count = len(native)
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError('fps must be positive and finite')
    initial_p, initial_r = native[0].copy(), original_r[0].copy()
    has_boundary = initial_assisted_positions is not None or initial_assisted_rotations is not None
    if has_boundary:
        if initial_assisted_positions is None or initial_assisted_rotations is None:
            raise ValueError('Provide both initial assisted positions and rotations')
        initial_p = np.asarray(initial_assisted_positions, dtype=float).copy()
        initial_r = np.asarray(initial_assisted_rotations, dtype=float).copy()
        if initial_p.shape != (27, 3) or initial_r.shape != (27, 3, 3):
            raise ValueError('Initial assisted boundary must be one Core27 pose')
        nonlegs = [j for j in range(27) if j not in LEG_JOINTS]
        if not np.allclose(initial_p[0, [0, 2]], native[0, 0, [0, 2]], atol=1e-7, rtol=0) or not np.allclose(initial_r[nonlegs], original_r[0, nonlegs], atol=1e-7, rtol=0):
            raise ValueError('Initial assisted root XZ and upper-body rotations must match native first frame')
        if np.max(abs(_fk(initial_p[0], initial_r)-initial_p)) > 1e-5:
            raise ValueError('Initial assisted boundary is not neutral Core FK')
    else:
        for _, _, foot, _ in LEGS:
            angle = np.arctan2(initial_r[foot, 0, 2], initial_r[foot, 2, 2])
            initial_r[foot] = Rotation.from_euler('y', float(angle)).as_matrix()
    stance = np.zeros((count, 2), bool)
    targets = np.zeros((count, 2, 3))
    foot_rotations = np.zeros((count, 2, 3, 3))
    anchors = []
    planned_runs, route_yaw, moving_bouts = _plan_contacts(native, original_r, fps, initial_p, initial_r, initial_is_continuation=has_boundary)
    for side, (_, _, foot, toe) in enumerate(LEGS):
        path = native[:, toe]
        mask = np.zeros(count, bool)
        for a, b, _, _ in planned_runs[side]:
            mask[a:b] = True
        for t in range(1, count-1):
            if mask[t-1] and mask[t+1]:
                mask[t] = True
        for a, b in _runs(mask):
            if b-a < 3:
                mask[a:b] = False
        runs = _runs(mask)
        if not runs or runs[0][0] != 0 or runs[-1][1] != count:
            raise ValueError('First offline proof requires planted clip endpoints')
        stance[:, side] = mask
        foot_rotations[:, side] = Rotation.from_euler('y', route_yaw[:, None]).as_matrix()
        for a, b in runs:
            _, _, anchor, rot = next(entry for entry in planned_runs[side] if entry[0] == a)
            anchor, rot = anchor.copy(), rot.copy()
            direction = rot[[0, 2], 2]
            direction /= np.linalg.norm(direction)
            # Move an edge-straddling footprint a few cm onto a real flat tread.
            candidates = []
            for dz in np.linspace(-.13, .13, 27):
                point = anchor.copy(); point[[0, 2]] += dz*direction
                sole = _sole(point - rot @ OFFSETS[toe], rot)
                try:
                    # Four corners plus interior samples prevent corner-only
                    # acceptance over a narrow gap or thin unsupported tread.
                    samples = np.concatenate([sole, (sole+np.roll(sole, 1, axis=0))*.5, sole.mean(axis=0)[None]])
                    heights = np.array([_support(geometry, q) for q in samples])
                except ValueError:
                    continue
                if np.ptp(heights) < .005:
                    candidates.append((abs(dz), point, float(heights.max())))
            if not candidates:
                raise ValueError('No supported flat footprint near native contact')
            _, anchor, height = min(candidates, key=lambda x: x[0])
            anchor[1] = height + .004
            if has_boundary and a == 0:
                anchor = initial_p[toe].copy()
                points = _sole(anchor-rot@OFFSETS[toe], rot)
                if max(abs(q[1]-_support(geometry, q)) for q in points) > .015:
                    raise ValueError('Initial assisted boundary must have both feet planted on support')
            targets[a:b, side] = anchor
            foot_rotations[a:b, side] = rot
            if b < count:
                for t in range(max(a, b-4), b):
                    amount = (t-(b-4)+1)/4
                    foot_rotations[t, side] = rot @ Rotation.from_euler('x', .48*amount*amount).as_matrix()
            anchors.append({'side': side, 'first': int(a), 'last_exclusive': int(b),
                            'toe_xyz': anchor.tolist()})
        for (_, end), (start, _) in zip(runs, runs[1:]):
            first, last = end-1, start
            u = np.linspace(0., 1., last-first+1)
            a, b = targets[first, side], targets[last, side]
            mix = u*u*(3.-2.*u)
            start_rotation = foot_rotations[first, side].copy()
            end_rotation = foot_rotations[last, side].copy()
            interpolated = Slerp([0., 1.], Rotation.from_matrix(np.stack([start_rotation, end_rotation])))(mix).as_matrix()
            for i, t in enumerate(range(first, last+1)):
                foot_rotations[t, side] = interpolated[i]
            path = (1.-mix[:, None])*a + mix[:, None]*b
            # Clearance height follows real treads; sine-squared arc has zero
            # vertical endpoint speed and does not retain floating native feet.
            bump = np.sin(np.pi*u)**2
            amplitude = .10
            for i, t in enumerate(range(first, last+1)):
                ankle = path[i] - foot_rotations[t, side] @ OFFSETS[toe]
                clearance = .004+.014*min(1., bump[i]/.10)
                need = max(_support(geometry, q)-q[1]+clearance for q in _sole(ankle, foot_rotations[t, side]))
                if bump[i] > .02:
                    amplitude = max(amplitude, need/bump[i])
            path[:, 1] += amplitude*bump
            targets[first:last+1, side] = path
    ankles = np.stack([targets[:, s] - np.einsum('tij,j->ti', foot_rotations[:, s], OFFSETS[leg[3]])
                      for s, leg in enumerate(LEGS)], axis=1)
    # Root XZ and all non-leg orientation are exactly native. Only root Y is
    # adjusted, jointly across the complete clip with smoothness and reach.
    upper_bounds = np.full(count, np.inf)
    preferred = np.empty(count)
    for t in range(count):
        floor = min(targets[t, :, 1])
        preferred[t] = floor + .927
        for s, (hip, knee, foot, _) in enumerate(LEGS):
            horizontal = np.linalg.norm(native[t, hip, [0, 2]]-ankles[t, s, [0, 2]])
            reach = np.linalg.norm(OFFSETS[knee])+np.linalg.norm(OFFSETS[foot])-.008
            if horizontal >= reach:
                raise ValueError('Native root XZ cannot reach planned anchor')
            hip_offset = native[t, hip, 1]-native[t, 0, 1]
            upper_bounds[t] = min(upper_bounds[t], ankles[t, s, 1]+np.sqrt(reach**2-horizontal**2)-hip_offset)
    preferred = gaussian_filter1d(preferred, 2)
    if np.any(upper_bounds < preferred-.30):
        bad = np.flatnonzero(upper_bounds < preferred-.30)
        raise ValueError(f'Unnatural root lowering required at {bad.tolist()}; upper={upper_bounds[bad].tolist()}, preferred={preferred[bad].tolist()}')
    d1 = np.diff(np.eye(count), axis=0)
    d2 = np.diff(np.eye(count), n=2, axis=0)
    matrix = np.eye(count) + 12*d1.T@d1 + 55*d2.T@d2
    def objective(y):
        return float(y@matrix@y-2*preferred@y), 2*(matrix@y-preferred)
    lower_bounds = preferred-.30
    if has_boundary:
        if not lower_bounds[0]-1e-6 <= initial_p[0, 1] <= upper_bounds[0]+1e-6:
            raise ValueError('Initial assisted pelvis height cannot satisfy planted leg reach')
        lower_bounds[0] = upper_bounds[0] = initial_p[0, 1]
    fit = minimize(objective, np.minimum(preferred, upper_bounds), jac=True,
                   bounds=list(zip(lower_bounds, upper_bounds)), method='L-BFGS-B',
                   options={'ftol': 1e-12, 'gtol': 1e-7, 'maxiter': 500})
    if not fit.success:
        raise ValueError('Root solve did not converge: '+str(fit.message))
    output_r = original_r.copy()
    output_p = np.empty_like(native)
    for t in range(count):
        gait_rotation = Rotation.from_euler('y', float(route_yaw[t])).as_matrix()
        knee_forward = gait_rotation @ np.array([0., 0., 1.])
        root = native[t, 0].copy(); root[1] = fit.x[t]
        p = _fk(root, output_r[t])
        for s, (hip, knee, foot, toe) in enumerate(LEGS):
            # Native near-straight knee planes flipped when cadence changed.
            # The explicit gait uses one continuous anatomical forward plane.
            pole = knee_forward
            blend = min(1., t/max(1., .3*fps))
            blend = blend*blend*(3.-2.*blend)
            if has_boundary and blend < 1.:
                old_axis = initial_p[foot]-initial_p[hip]
                old_axis /= np.linalg.norm(old_axis)
                previous_plane = initial_p[knee]-initial_p[hip]
                previous_plane -= previous_plane@old_axis*old_axis
                if np.linalg.norm(previous_plane) > 1e-6:
                    previous_plane /= np.linalg.norm(previous_plane)
                    pole = (1-blend)*previous_plane+blend*knee_forward
            new_knee, endpoint = _two_bone(p[hip], ankles[t, s], p[hip]+pole,
                                           np.linalg.norm(OFFSETS[knee]), np.linalg.norm(OFFSETS[foot]),
                                           knee_forward)
            output_r[t, hip] = _swing(gait_rotation@OFFSETS[knee], new_knee-p[hip])@gait_rotation
            output_r[t, knee] = _swing(gait_rotation@OFFSETS[foot], endpoint-new_knee)@gait_rotation
            if has_boundary and blend < 1.:
                for joint, child, vector in ((hip, knee, new_knee-p[hip]), (knee, foot, endpoint-new_knee)):
                    interpolated = Slerp([0., 1.], Rotation.from_matrix(np.stack([initial_r[joint], output_r[t, joint]])))([blend]).as_matrix()[0]
                    output_r[t, joint] = _swing(interpolated@OFFSETS[child], vector)@interpolated
            output_r[t, foot] = foot_rotations[t, s]
            output_r[t, toe] = foot_rotations[t, s]
        output_p[t] = _fk(root, output_r[t])
    if has_boundary:
        output_r[0, sorted(LEG_JOINTS)] = initial_r[sorted(LEG_JOINTS)]
        output_p[0] = _fk(initial_p[0], output_r[0])
    penetration, spread, gap, slip, knee_angles, local_changes = [], [], [], [], [], []
    flat_gap, swept_penetration = [], []
    sole_paths = np.empty((count, 2, 4, 3))
    for t in range(count):
        for s, (hip, knee, foot, toe) in enumerate(LEGS):
            points = _sole(output_p[t, foot], output_r[t, foot])
            sole_paths[t, s] = points
            heights = np.array([_support(geometry, q) for q in points])
            penetration.extend((heights-points[:, 1]).tolist())
            if stance[t, s]:
                spread.append(float(np.ptp(heights)))
                gap.extend(abs(points[:, 1]-heights).tolist())
                if np.ptp(points[:, 1]) < .001:
                    flat_gap.extend(abs(points[:, 1]-heights).tolist())
                slip.append(float(np.linalg.norm(output_p[t, toe]-targets[t, s])))
            v1, v2 = output_p[t, knee]-output_p[t, hip], output_p[t, foot]-output_p[t, knee]
            knee_angles.append(float(np.degrees(np.arccos(np.clip(v1@v2/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1., 1.)))))
    for t in range(count-1):
        for fraction in (.125, .25, .375, .5, .625, .75, .875):
            points = (1-fraction)*sole_paths[t]+fraction*sole_paths[t+1]
            swept_penetration.extend(_support(geometry, q)-q[1] for q in points.reshape(-1, 3))
    frame_rotation_changes = []
    for j in LEG_JOINTS:
        old = original_r[:, PARENTS[j]].transpose(0, 2, 1)@original_r[:, j]
        new = output_r[:, PARENTS[j]].transpose(0, 2, 1)@output_r[:, j]
        local_changes.extend(np.degrees(Rotation.from_matrix(old.transpose(0, 2, 1)@new).magnitude()).tolist())
        frame_rotation_changes.extend(np.degrees(Rotation.from_matrix(new[:-1].transpose(0, 2, 1)@new[1:]).magnitude()).tolist())
    delta = output_p[:, 0]-native[:, 0]
    toe_delta = output_p[:, [leg[3] for leg in LEGS]]-native[:, [leg[3] for leg in LEGS]]-delta[:, None]
    report = {'provenance': 'explicit offline terrain-assisted presentation; NOT native ARDY output',
              'accepted': False, 'visual_review': 'pending; numerical results alone never accept this proof',
              'native_cadence_preserved': False,
              'cadence_policy': 'explicit alternating footsteps from native route arclength and local heading; both feet settle for holds',
              'moving_intervals': [[int(a), int(b)] for a, b in moving_bouts],
              'initial_assisted_boundary_supplied': has_boundary,
              'initial_boundary_position_error_m': float(np.max(np.linalg.norm(output_p[0]-initial_p, axis=1))) if has_boundary else None,
              'knee_plane_policy': 'continuous route-forward anatomical plane; native near-extension poles rejected for discontinuity',
              'frames': count, 'fps': fps, 'stance_anchors': anchors,
              'native_positions_and_rotations_unchanged': True, 'input_fk_error_m': float(fk_error),
              'non_leg_global_rotations_unchanged': bool(np.array_equal(output_r[:, [j for j in range(27) if j not in LEG_JOINTS]], original_r[:, [j for j in range(27) if j not in LEG_JOINTS]])),
              'max_root_y_change_m': float(abs(delta[:, 1]).max()),
              'max_root_xz_change_m': float(abs(delta[:, [0, 2]]).max()),
              'max_root_frame_y_step_m': float(abs(np.diff(output_p[:, 0, 1])).max()),
              'max_foot_relative_root_change_m': float(np.linalg.norm(toe_delta, axis=2).max()),
              'max_local_leg_rotation_change_deg': max(local_changes),
              'max_local_leg_rotation_frame_step_deg': max(frame_rotation_changes),
              'max_knee_bend_deg': max(knee_angles),
              'max_stance_gap_m': max(gap), 'max_stance_anchor_error_m': max(slip),
              'max_flat_stance_gap_m': max(flat_gap or [0.]),
              'stance_gap_note': 'Toe-off raises the heel about 9cm; toe remains fixed. Flat stance is reported separately.',
              'max_sole_penetration_m': max(0., max(penetration)),
              'max_swept_sole_proxy_penetration_m': max(0., max(swept_penetration)),
              'max_stance_footprint_support_spread_m': max(spread),
              'old_rejected_adapter_budgets': {'root_y_m': .08, 'foot_relative_m': .12, 'local_leg_deg': 35},
              'limitations': ['whole clip phase lookahead, not streaming', 'four point sole proxy, not skinned mesh collision', 'foot and root changes can exceed rejected v2 budgets and are explicitly reported']}
    report['old_budget_violations'] = [label for value, limit, label in (
        (report['max_root_y_change_m'], .08, 'root Y exceeds .08m'),
        (report['max_foot_relative_root_change_m'], .12, 'foot relative correction exceeds .12m'),
        (report['max_local_leg_rotation_change_deg'], 35., 'local leg rotation exceeds 35deg')) if value > limit]
    report['motion_derivatives'] = {}
    for label, values in [('assisted_root', output_p[:, [0]]), ('native_root', native[:, [0]]),
                          ('assisted_toes', output_p[:, [leg[3] for leg in LEGS]]),
                          ('native_toes', native[:, [leg[3] for leg in LEGS]])]:
        speed = np.linalg.norm(np.diff(values, axis=0)*fps, axis=-1)
        acceleration = np.linalg.norm(np.diff(values, n=2, axis=0)*fps**2, axis=-1)
        report['motion_derivatives'][label] = {
            'max_speed_m_s': float(speed.max()), 'p95_speed_m_s': float(np.percentile(speed, 95)),
            'max_acceleration_m_s2': float(acceleration.max()),
            'p95_acceleration_m_s2': float(np.percentile(acceleration, 95))}
    return output_p, output_r, stance, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.input.read_bytes()
    with np.load(args.input, allow_pickle=False) as archive:
        native = {key: archive[key].copy() for key in archive.files}
    p = native.get('with_prefix_positions', native['positions'])
    r = native.get('with_prefix_rotations', native['rotations'])
    geometry = SceneInteractionGeometry.from_scene(json.loads(args.scene.read_text()))
    poses, rotations, stance, report = assist_clip(p, r, geometry)
    report['input_sha256'] = hashlib.sha256(source).hexdigest()
    report['source_file_unchanged'] = args.input.read_bytes() == source
    np.savez_compressed(args.output, positions=poses.astype(np.float32), rotations=rotations.astype(np.float32),
                        native_positions=p, native_rotations=r, assisted_stance=stance,
                        native_features=native.get('with_prefix_native_features', native['native_features']),
                        assisted_presentation=np.array(True), accepted=np.array(False))
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in report.items() if k != 'stance_anchors'}, indent=2))


if __name__ == '__main__':
    main()
