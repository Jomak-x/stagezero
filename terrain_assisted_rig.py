"""Offline, explicitly assisted GroundedCharacter17 terrain presentation.

Native Core27 is retargeted exactly once. This controller solves the actual
rendered rig, and its output must be drawn directly without retargeting or IK.
Numerical validity does not constitute visual acceptance.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

from grounded_character import _PARENTS, _SHOULDER_BLENDS
from motion_bridge import _swing, _two_bone
from terrain_assisted_motion import LEGS as CORE_LEGS, _plan_contacts, _support

LEGS = ((9, 10, 11), (12, 13, 14))
LEG_JOINTS = {j for leg in LEGS for j in leg}
MAX_KNEE_BEND_DEG = 110.


def _fk(root, rotations, rest):
    positions = np.empty((17, 3)); positions[0] = root
    for j in range(1, 17):
        positions[j] = positions[_PARENTS[j]]+rotations[_PARENTS[j]]@(rest[j]-rest[_PARENTS[j]])
    for arm, blend in _SHOULDER_BLENDS:
        positions[blend] = positions[arm]
    return positions


def assist_rig_clip(native_positions, native_rotations, geometry, character, *, fps=20.,
                    initial_assisted_positions=None, initial_assisted_rotations=None):
    """Return rig17 positions, rotations, L/R stance, and a never-accepted report.

    Optional initial boundary must be a planted previous rig17 pose with the
    same native root XZ and retargeted non-leg rotations. This is a buffered
    whole-action solver; it does not write native features or generation state.
    """
    native = np.asarray(native_positions, dtype=float)
    native_r = np.asarray(native_rotations, dtype=float)
    if native.shape[1:] != (27, 3) or native_r.shape != (len(native), 27, 3, 3) or len(native) < 3:
        raise ValueError('Expected at least three native Core27 frames')
    if not np.isfinite(native).all() or not np.isfinite(native_r).all() or fps <= 0:
        raise ValueError('Finite native poses and positive fps required')
    baseline = [character.retarget(p, r, preserve_root_height=False, preserve_wrists=False)
                for p, r in zip(native, native_r)]
    base_p = np.array([row['positions'] for row in baseline])
    base_r = np.array([row['rotations'] for row in baseline])
    rest = np.asarray(character.rest, dtype=float)
    count = len(native)
    initial_p, initial_r = base_p[0].copy(), base_r[0].copy()
    boundary = initial_assisted_positions is not None or initial_assisted_rotations is not None
    nonlegs = [j for j in range(17) if j not in LEG_JOINTS]
    if boundary:
        if initial_assisted_positions is None or initial_assisted_rotations is None:
            raise ValueError('Provide both rig17 boundary positions and rotations')
        initial_p, initial_r = np.asarray(initial_assisted_positions, float).copy(), np.asarray(initial_assisted_rotations, float).copy()
        if initial_p.shape != (17, 3) or initial_r.shape != (17, 3, 3):
            raise ValueError('Boundary must use this exact 17-bone character rig')
        if not np.isfinite(initial_p).all() or not np.isfinite(initial_r).all():
            raise ValueError('Rig17 boundary must be finite')
        if not np.allclose(initial_p[0, [0, 2]], base_p[0, 0, [0, 2]], atol=1e-6, rtol=0) or not np.allclose(initial_r[nonlegs], base_r[0, nonlegs], atol=1e-6, rtol=0):
            raise ValueError('Boundary root XZ and native upper-body rotations disagree')
        if np.max(abs(_fk(initial_p[0], initial_r, rest)-initial_p)) > 1e-5:
            raise ValueError('Boundary does not match the character rest rig')
    ids, contact_local, footprint = [], [], []
    for side, (_, _, foot) in enumerate(LEGS):
        surface = character.foot_surface_indices[side]
        sole = surface[character.vertices[surface, 1] <= character.vertices[surface, 1].min()+.015]
        if len(sole) < 3:
            raise ValueError('Character has no measurable sole surface')
        ids.append(sole)
        local = character.vertices[sole]-rest[foot]
        lo, hi = local.min(axis=0), local.max(axis=0)
        # Toe-edge contact pivot; flat sole height comes from actual mesh minY.
        pivot = np.array([(lo[0]+hi[0])*.5, lo[1], hi[2]])
        contact_local.append(pivot)
        footprint.append(np.array([[x, lo[1], z] for x in np.linspace(lo[0]-.003, hi[0]+.003, 3)
                                   for z in np.linspace(lo[2]-.003, hi[2]+.003, 5)]))
    fake_p, fake_r = native[0].copy(), native_r[0].copy()
    for s, (_, _, foot) in enumerate(LEGS):
        if not boundary:
            heading = np.arctan2(initial_r[foot, 0, 2], initial_r[foot, 2, 2])
            initial_r[foot] = Rotation.from_euler('y', float(heading)).as_matrix()
        fake_p[CORE_LEGS[s][3]] = initial_p[foot]+initial_r[foot]@contact_local[s]
        fake_r[CORE_LEGS[s][2]] = initial_r[foot]
    # A constrained previous endpoint can have less toe lead than a native
    # initial stand; avoid compressed transfers when resuming that contact.
    runs, route_yaw, bouts = _plan_contacts(native, native_r, fps, fake_p, fake_r,
                                          initial_is_continuation=boundary,
                                          minimum_transfer_frames=max(4, round(.4*fps)) if boundary else None)
    targets = np.empty((count, 2, 3))
    foot_r = np.empty((count, 2, 3, 3))
    stance = np.zeros((count, 2), bool)
    flat_stance = np.zeros_like(stance)
    anchors = []
    for side, entries in enumerate(runs):
        pivot = contact_local[side]
        samples = footprint[side]
        for a, b, proposed, rotation in entries:
            heading = rotation[[0, 2], 2]; heading /= np.linalg.norm(heading)
            candidates = []
            for shift in np.r_[0., np.linspace(-.16, .16, 33)]:
                point = proposed.copy(); point[[0, 2]] += shift*heading
                points = point+(samples-pivot)@rotation.T
                try:
                    heights = np.array([_support(geometry, q) for q in points])
                except ValueError:
                    continue
                if np.ptp(heights) < .005:
                    candidates.append((abs(shift), point, heights.max()))
            if not candidates:
                raise ValueError('Actual mesh sole footprint cannot fit a supported tread')
            _, anchor, height = min(candidates, key=lambda v: v[0])
            anchor[1] = height+.004
            if boundary and a == 0:
                anchor = fake_p[CORE_LEGS[side][3]].copy()
                vertices = character._sole_vertices(ids[side], initial_p, initial_r)
                clear = np.array([q[1]-_support(geometry, q) for q in vertices])
                if clear.min() < -.008 or clear.min() > .025:
                    raise ValueError('Previous rig17 endpoint is not a supported planted pose')
            targets[a:b, side] = anchor
            foot_r[a:b, side] = rotation
            stance[a:b, side] = True
            flat_stance[a:b, side] = True
            if b < count:
                for t in range(max(a, b-4), b):
                    u = (t-b+5)/4
                    foot_r[t, side] = rotation@Rotation.from_euler('x', .40*u*u).as_matrix()
                    flat_stance[t, side] = False
            anchors.append({'side': side, 'first': int(a), 'last_exclusive': int(b), 'contact_xyz': anchor.tolist()})
        for (_, end, _, _), (start, _, _, _) in zip(entries, entries[1:]):
            first, last = end-1, start
            u = np.linspace(0., 1., last-first+1); blend = u*u*(3-2*u)
            rotations = Slerp([0., 1.], Rotation.from_matrix(np.stack([foot_r[first, side], foot_r[last, side]])))(blend).as_matrix()
            path = (1-blend[:, None])*targets[first, side]+blend[:, None]*targets[last, side]
            bump = np.sin(np.pi*u)**2; amplitude = .09
            for i in range(len(path)):
                points = path[i]+(samples-pivot)@rotations[i].T
                # Contact clearance must taper to zero at takeoff/landing;
                # a fixed margin divided by a tiny endpoint bump produces
                # arbitrarily high arcs for slow steps or pauses.
                clearance = .004+.014*min(1., bump[i]/.10)
                needed = max(_support(geometry, q)-q[1]+clearance for q in points)
                if bump[i] > .02:
                    amplitude = max(amplitude, needed/bump[i])
            if amplitude > .24:
                raise ValueError('Mesh foot needs excessive swing clearance')
            path[:, 1] += amplitude*bump
            targets[first:last+1, side], foot_r[first:last+1, side] = path, rotations
    ankles = np.stack([targets[:, s]-np.einsum('tij,j->ti', foot_r[:, s], contact_local[s]) for s in range(2)], axis=1)
    upper = np.full(count, np.inf)
    knee_lower = np.full(count, -np.inf)
    # Calibrated upright pelvis height from this actual rig/sole relationship.
    comfortable = float(rest[0, 1]-min(character.vertices[ids[0], 1].min(), character.vertices[ids[1], 1].min())-.015)
    preferred = gaussian_filter1d(targets[:, :, 1].min(axis=1)+comfortable, 2)
    for t in range(count):
        for s, (hip, knee, foot) in enumerate(LEGS):
            thigh = np.linalg.norm(rest[knee]-rest[hip])
            shin = np.linalg.norm(rest[foot]-rest[knee])
            length = thigh+shin-.008
            horizontal = np.linalg.norm(base_p[t, hip, [0, 2]]-ankles[t, s, [0, 2]])
            if horizontal >= length:
                raise ValueError('Actual rig legs cannot reach the native root path and planned contacts')
            hip_offset = base_p[t, hip, 1]-base_p[t, 0, 1]
            upper[t] = min(upper[t], ankles[t, s, 1]+np.sqrt(length*length-horizontal*horizontal)-hip_offset)
            # Law of cosines with zero flexion = a straight leg. A maximum
            # flexion angle imposes a MINIMUM hip-to-ankle distance. Intersect
            # its root-Y lower bound with BOTH legs' upper reach bounds; do not
            # smooth or clamp the resulting knees after the solve.
            minimum_distance = np.sqrt(thigh*thigh+shin*shin+2*thigh*shin*np.cos(np.radians(MAX_KNEE_BEND_DEG)))+1e-5
            vertical_minimum = np.sqrt(max(0., minimum_distance*minimum_distance-horizontal*horizontal))
            knee_lower[t] = max(knee_lower[t], ankles[t, s, 1]+vertical_minimum-hip_offset)
    lower = np.maximum(preferred-.20, knee_lower)
    if np.any(upper < lower):
        frames = np.flatnonzero(upper < lower)
        raise ValueError(f'Actual rig has no root height satisfying both leg reach and {MAX_KNEE_BEND_DEG:g}-degree knee limit at frames {frames.tolist()}')
    if boundary:
        if not lower[0]-1e-6 <= initial_p[0, 1] <= upper[0]+1e-6:
            raise ValueError('Rig17 boundary height violates reach')
        lower[0] = upper[0] = initial_p[0, 1]
    d1, d2 = np.diff(np.eye(count), axis=0), np.diff(np.eye(count), n=2, axis=0)
    matrix = np.eye(count)+12*d1.T@d1+55*d2.T@d2
    fit = minimize(lambda y: (float(y@matrix@y-2*preferred@y), 2*(matrix@y-preferred)),
                   np.minimum(preferred, upper), jac=True, bounds=list(zip(lower, upper)), method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-7})
    if not fit.success:
        raise ValueError('Rig17 root optimization failed')
    poses, rotations = np.empty_like(base_p), base_r.copy()
    for t in range(count):
        root = base_p[t, 0].copy(); root[1] = fit.x[t]
        current = _fk(root, rotations[t], rest)
        heading = Rotation.from_euler('y', float(route_yaw[t])).as_matrix()
        forward = heading@np.array([0., 0., 1.])
        for s, (hip, knee, foot) in enumerate(LEGS):
            blend = min(1., t/max(1., fps*.3)); blend = blend*blend*(3-2*blend)
            pole = forward
            if boundary and blend < 1.:
                axis = initial_p[foot]-initial_p[hip]; axis /= np.linalg.norm(axis)
                previous = initial_p[knee]-initial_p[hip]; previous -= previous@axis*axis
                if np.linalg.norm(previous) > 1e-6:
                    pole = (1-blend)*previous/np.linalg.norm(previous)+blend*forward
            k, ankle = _two_bone(current[hip], ankles[t, s], current[hip]+pole,
                                np.linalg.norm(rest[knee]-rest[hip]), np.linalg.norm(rest[foot]-rest[knee]), forward)
            for joint, child, vector in ((hip, knee, k-current[hip]), (knee, foot, ankle-k)):
                offset = rest[child]-rest[joint]
                rotation = _swing(heading@offset, vector)@heading
                if boundary and blend < 1.:
                    rotation = Slerp([0., 1.], Rotation.from_matrix(np.stack([initial_r[joint], rotation])))([blend]).as_matrix()[0]
                    rotation = _swing(rotation@offset, vector)@rotation
                rotations[t, joint] = rotation
            rotations[t, foot] = foot_r[t, s]
        poses[t] = _fk(root, rotations[t], rest)
    if boundary:
        rotations[0, sorted(LEG_JOINTS)] = initial_r[sorted(LEG_JOINTS)]
        poses[0] = _fk(initial_p[0], rotations[0], rest)
    # Exact sparse LBS of every sole vertex, with real scene support queries.
    clearances, flat_clearances, flat_slip, flat_lowest = [], [], [], []
    sole_paths = [[], []]
    unsupported = 0
    support_cache = {}
    def cached_support(point):
        key = tuple(np.round(point, 7))
        if key not in support_cache:
            try: support_cache[key] = _support(geometry, point)
            except ValueError: support_cache[key] = None
        return support_cache[key]
    for t in range(count):
        for s in range(2):
            vertices = character._sole_vertices(ids[s], poses[t], rotations[t])
            sole_paths[s].append(vertices)
            frame_clearances = []
            for q in vertices:
                height = cached_support(q)
                if height is None:
                    unsupported += 1; continue
                clearance = float(q[1]-height); clearances.append(clearance)
                frame_clearances.append(clearance)
                if flat_stance[t, s]: flat_clearances.append(clearance)
            if flat_stance[t, s] and frame_clearances:
                flat_lowest.append(min(frame_clearances))
            if t and flat_stance[t, s] and flat_stance[t-1, s]:
                flat_slip.extend(np.linalg.norm(vertices[:, [0, 2]]-sole_paths[s][t-1][:, [0, 2]], axis=1)*fps)
    # Conservative bottom rectangle with seven temporal samples. This is
    # deliberately reported as a finite proxy, not swept triangle collision.
    swept_penetration, swept_unsupported = 0., 0
    fractions = np.arange(1, 8)/8
    for side, (_, _, foot) in enumerate(LEGS):
        samples = footprint[side]
        for t in range(count-1):
            if np.max(abs(poses[t+1, foot]-poses[t, foot])) < 1e-9 and np.max(abs(rotations[t+1, foot]-rotations[t, foot])) < 1e-9:
                continue
            interpolated = Slerp([0., 1.], Rotation.from_matrix(rotations[t:t+2, foot]))(fractions).as_matrix()
            for fraction, rotation in zip(fractions, interpolated):
                ankle = (1-fraction)*poses[t, foot]+fraction*poses[t+1, foot]
                for q in ankle+samples@rotation.T:
                    height = cached_support(q)
                    if height is None:
                        swept_unsupported += 1
                    else:
                        swept_penetration = max(swept_penetration, float(height-q[1]))
    angles, frame_angles, knee_bends = [], [], []
    for hip, knee, foot in LEGS:
        a, b = poses[:, knee]-poses[:, hip], poses[:, foot]-poses[:, knee]
        knee_bends.extend(np.degrees(np.arccos(np.clip(np.sum(a*b, axis=1)/(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)), -1., 1.))))
    for j in LEG_JOINTS:
        old = base_r[:, _PARENTS[j]].transpose(0, 2, 1)@base_r[:, j]
        new = rotations[:, _PARENTS[j]].transpose(0, 2, 1)@rotations[:, j]
        angles.extend(np.degrees(Rotation.from_matrix(old.transpose(0, 2, 1)@new).magnitude()))
        frame_angles.extend(np.degrees(Rotation.from_matrix(new[:-1].transpose(0, 2, 1)@new[1:]).magnitude()))
    report = {'representation': 'GroundedCharacter17 explicit assisted presentation', 'accepted': False,
              'visual_review': 'pending', 'rig_mesh_sha256': character.mesh_sha256, 'rig_bone_count': 17,
              'native_arrays_unchanged': True, 'retarget_count_per_native_frame': 1,
              'renderer_contract': 'render these rig17 positions/rotations directly; no retarget or second IK',
              'native_cadence_preserved': False, 'native_root_xz_preserved': bool(np.array_equal(poses[:, 0, [0, 2]], native[:, 0, [0, 2]])),
              'native_retargeted_upperbody_rotations_preserved': bool(np.array_equal(rotations[:, nonlegs], base_r[:, nonlegs])),
              'frames': count, 'fps': fps, 'stance_anchors': anchors,
              'moving_intervals': [[int(a), int(b)] for a, b in bouts],
              'initial_boundary_position_error_m': float(np.max(np.linalg.norm(poses[0]-initial_p, axis=1))) if boundary else None,
              'max_root_y_change_from_native_retarget_m': float(abs(poses[:, 0, 1]-base_p[:, 0, 1]).max()),
              'max_local_leg_rotation_change_deg': float(max(angles)),
              'max_local_leg_rotation_frame_step_deg': float(max(frame_angles)),
              'max_knee_bend_deg': float(max(knee_bends)),
              'knee_flexion_limit_deg': MAX_KNEE_BEND_DEG,
              'knee_limit_enforced_in_root_optimizer': True,
              'actual_mesh_sole_vertex_counts': [len(v) for v in ids],
              'max_actual_mesh_sole_penetration_m': max(0., -min(clearances)) if clearances else None,
              'max_flat_stance_mesh_gap_m': max(flat_clearances) if flat_clearances else None,
              'max_flat_stance_lowest_vertex_gap_m': max(flat_lowest) if flat_lowest else None,
              'sole_gap_note': 'Sole cloud includes vertices within15mm of bind outsole; lowest-vertex gap measures contact separately.',
              'max_flat_stance_mesh_vertex_slip_m_s': float(max(flat_slip or [0.])),
              'unsupported_actual_mesh_sole_vertices': unsupported,
              'max_swept_sole_envelope_penetration_m': swept_penetration,
              'unsupported_swept_sole_envelope_samples': swept_unsupported,
              'swept_sole_proxy': {'interframe_samples': 7, 'footprint_grid': [3, 5], 'margin_m': .003,
                                  'note': 'finite rigid bottom-envelope samples using linear foot translation and rotation slerp; not continuous triangle collision'},
              'limitations': ['buffered whole action, not streaming', 'finite sole-envelope sweep; no dynamics or full body collision certificate']}
    report['motion_derivatives'] = {}
    for label, values in [('root', poses[:, [0]]), ('feet', poses[:, [11, 14]])]:
        speed = np.linalg.norm(np.diff(values, axis=0)*fps, axis=-1)
        acceleration = np.linalg.norm(np.diff(values, n=2, axis=0)*fps**2, axis=-1)
        report['motion_derivatives'][label] = {'max_speed_m_s': float(speed.max()),
            'max_acceleration_m_s2': float(acceleration.max())}
    report['numerical_rejections'] = []
    if unsupported: report['numerical_rejections'].append('actual sole vertices leave authored support')
    if report['max_actual_mesh_sole_penetration_m'] is None or report['max_actual_mesh_sole_penetration_m'] > .008:
        report['numerical_rejections'].append('actual mesh sole penetration exceeds8mm')
    if report['max_flat_stance_mesh_vertex_slip_m_s'] > .05:
        report['numerical_rejections'].append('flat stance mesh vertex slip exceeds5cm/s')
    if swept_unsupported:
        report['numerical_rejections'].append('swept sole envelope leaves authored support')
    if swept_penetration > .008:
        report['numerical_rejections'].append('swept sole envelope penetration exceeds8mm')
    if max(knee_bends) > MAX_KNEE_BEND_DEG:
        report['numerical_rejections'].append('knee flexion exceeds110deg')
    if max(frame_angles) > 35:
        report['numerical_rejections'].append('local leg rotation step exceeds35deg/frame')
    return poses, rotations, stance, report
