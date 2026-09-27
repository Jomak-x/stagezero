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
from terrain_assisted_motion import LEGS as CORE_LEGS, _plan_contacts, _route_frames, _support

LEGS = ((9, 10, 11), (12, 13, 14))
LEG_JOINTS = {j for leg in LEGS for j in leg}
MAX_KNEE_BEND_DEG = 110.


def _boundary_heading(desired, inherited, fps, bouts, runs):
    """Join a settled boundary without magnifying a short startup yaw wobble."""
    desired = np.asarray(desired, float)
    inherited += 2*np.pi*round((desired[0]-inherited)/(2*np.pi))
    width = min(len(desired), max(2, round(.6*fps)))
    phase = np.linspace(0., 1., width)
    blend = phase*phase*(3-2*phase)
    baseline = desired.copy()
    baseline[:width] += (inherited-desired[0])*(1-blend)
    # A long intentional pivot already has its own contact timing. Walking
    # immediately also gives no evidence of a settled startup interval.
    if (not bouts or not 0 < bouts[0][0] < width or
            any(0 < entry[0] < bouts[0][0] for side in runs for entry in side)):
        return baseline, False
    candidate = desired.copy()
    candidate[:width] = inherited+(desired[:width]-inherited)*blend

    def peaks(curve):
        # Include the settled incoming heading and unchanged samples after
        # the fade, so shifting a spike to the join cannot count as a repair.
        window = np.r_[inherited, inherited, curve[:width+2]]
        return np.array([np.max(abs(np.diff(window)))*fps,
                         np.max(abs(np.diff(window, n=2)))*fps*fps])

    before, after = peaks(baseline), peaks(candidate)
    if np.all(after <= before+1e-12) and np.any(after < before-1e-12):
        return candidate, True
    return baseline, False


def _fk(root, rotations, rest):
    positions = np.empty((17, 3)); positions[0] = root
    for j in range(1, 17):
        positions[j] = positions[_PARENTS[j]]+rotations[_PARENTS[j]]@(rest[j]-rest[_PARENTS[j]])
    for arm, blend in _SHOULDER_BLENDS:
        positions[blend] = positions[arm]
    return positions


def _swing_path(first, last, first_r, last_r, blend, follow_turn):
    path = (1-blend[:, None])*first+blend[:, None]*last
    chord = last[[0, 2]]-first[[0, 2]]
    length = np.linalg.norm(chord)
    if follow_turn and length > .05:
        headings = [rotation[[0, 2], 2].copy() for rotation in (first_r, last_r)]
        headings = [v/np.linalg.norm(v) for v in headings]
        if all(v@chord > .1*length for v in headings):
            # Follow the entering/exiting route tangents instead of cutting
            # a straight foot chord through the inside of a prop corner.
            controls = [first[[0, 2]], first[[0, 2]]+headings[0]*length/3,
                        last[[0, 2]]-headings[1]*length/3, last[[0, 2]]]
            u = blend[:, None]
            path[:, [0, 2]] = (1-u)**3*controls[0]+3*(1-u)**2*u*controls[1]+3*(1-u)*u*u*controls[2]+u**3*controls[3]
    return path


def _clearance_profile(geometry, first, last, first_r, last_r, samples, pivot, count, follow_turn=False,
                       _landing_hold=False):
    """Lift before leaving a nearby riser; solve a bounded smooth vertical arc.

    A global sine amplitude overreacts to a riser crossed just after takeoff.
    One sample of vertical takeoff leaves room to clear that riser without
    inflating the entire swing. Linear constraints use the same finite swept
    rigid-foot envelope as final validation, including rotation interpolation.
    """
    u = np.linspace(0., 1., count)
    # Descending contacts need their heel beyond the upper tread before the
    # final lowering phase. A feasible late crossing can still demand a large
    # last-frame drop (and an abrupt knee extension), so reserve the landing
    # sample for every descent, not only when the lift solver is infeasible.
    _landing_hold = _landing_hold or (count >= 4 and first[1]-last[1] > .005)
    finish = u[-2] if _landing_hold else 1.
    travel = np.clip((u-u[1])/(finish-u[1]), 0., 1.)
    blend = travel*travel*(3-2*travel)
    rotations = Slerp([0., 1.], Rotation.from_matrix(np.stack([first_r, last_r])))(blend).as_matrix()
    path = _swing_path(first, last, first_r, last_r, blend, follow_turn)
    ankles = path-np.einsum('tij,j->ti', rotations, pivot)
    rows, bounds = [], []
    def require(a, b, amount):
        rotation = Slerp([0., 1.], Rotation.from_matrix(np.stack([rotations[a], rotations[b]])))([amount]).as_matrix()[0]
        points = (1-amount)*ankles[a]+amount*ankles[b]+samples@rotation.T
        phase = (a*(1-amount)+b*amount)/(count-1)
        clearance = .004+.014*min(1., np.sin(np.pi*phase)**2/.10)
        needed = max(_support(geometry, q)-q[1]+clearance for q in points)
        row = np.zeros(count); row[a] += 1-amount; row[b] += amount
        rows.append(row); bounds.append(needed)
    for index in range(1, count-1):
        require(index, index, 0.)
    for index in range(count-1):
        for amount in np.linspace(0., 1., 9)[1:-1]:
            require(index, index+1, amount)
    matrix, lower = np.asarray(rows), np.asarray(bounds)
    desired = .09*np.sin(np.pi*u)**2
    curvature = np.diff(np.eye(count), n=2, axis=0)
    smooth = np.eye(count)+3*curvature.T@curvature
    fit = minimize(lambda y: (float(y@smooth@y-2*desired@y), 2*(smooth@y-desired)),
                   desired, jac=True, method='SLSQP',
                   bounds=[(0., 0.)]+[(0., .24)]*(count-2)+[(0., 0.)],
                   constraints=[{'type': 'ineq', 'fun': lambda y: matrix@y-lower,
                                 'jac': lambda y: matrix}],
                   options={'maxiter': 150, 'ftol': 1e-10})
    if not fit.success or np.min(matrix@fit.x-lower) < -1e-6:
        if not _landing_hold and count >= 4:
            # On descent, the heel may cross the upper tread just before
            # touchdown. Finish horizontal travel one sample early so the
            # foot can lower vertically after its entire envelope clears.
            return _clearance_profile(geometry, first, last, first_r, last_r,
                                      samples, pivot, count, follow_turn, _landing_hold=True)
        raise ValueError('Mesh foot cannot clear the swept riser within the 24cm lift budget')
    path[:, 1] += fit.x
    return path, rotations, float(fit.x.max())


def assist_rig_clip(native_positions, native_rotations, geometry, character, *, fps=20.,
                    initial_assisted_positions=None, initial_assisted_rotations=None,
                    _anchor_advances=None, _repair_attempt=0, _clearance_mode=False,
                    _minimum_stance_frames=4, _follow_turn_swing=False, heading_assistance=False):
    """Return rig17 positions, rotations, L/R stance, and a never-accepted report.

    Optional initial boundary must be a planted previous rig17 pose with the
    same native root XZ and retargeted non-leg rotations. This is a buffered
    whole-action solver; it does not write native features or generation state.
    """
    _anchor_advances = dict(_anchor_advances or {})
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
    raw_base_r = base_r.copy()
    rest = np.asarray(character.rest, dtype=float)
    count = len(native)
    if heading_assistance:
        _, _, initial_route_yaw, _ = _route_frames(native, native_r, fps)
        raw_yaw = np.arctan2(base_r[0, 0, 0, 2], base_r[0, 0, 2, 2])
        alignment = Rotation.from_euler('y', float(initial_route_yaw[0]-raw_yaw)).as_matrix()
        base_r[0] = alignment@base_r[0]
        base_p[0] = _fk(base_p[0, 0], base_r[0], rest)
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
        if heading_assistance:
            inherited = initial_r[0]@raw_base_r[0, 0].T
            angle = np.arctan2(inherited[0, 2], inherited[2, 2])
            common_yaw = Rotation.from_euler('y', float(angle)).as_matrix()
            if not np.allclose(inherited, common_yaw, atol=2e-6, rtol=0):
                raise ValueError('Boundary heading correction is not a common world-Y rotation')
            base_r[0] = common_yaw@raw_base_r[0]
        if not np.allclose(initial_p[0, [0, 2]], base_p[0, 0, [0, 2]], atol=1e-6, rtol=0) or not np.allclose(initial_r[nonlegs], base_r[0, nonlegs], atol=2e-6, rtol=0):
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
        footprint.append(np.array([[x, y, z] for x in np.linspace(lo[0]-.003, hi[0]+.003, 3)
                                   for y in ([lo[1], hi[1]] if _clearance_mode else [lo[1]])
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
                                          heading_assistance=heading_assistance,
                                          # Center ankle stance about the pelvis. The
                                          # planner targets a toe pivot, not an ankle;
                                          # this actual mesh has a much longer toe
                                          # offset than the neutral Core skeleton.
                                          touchdown_lead=max(p[2] for p in contact_local)+.25,
                                          minimum_stance_frames=_minimum_stance_frames,
                                          minimum_transfer_frames=max(4, round(.4*fps)) if boundary else None)
    heading_correction = np.zeros(count)
    boundary_heading_target_fade = False
    if heading_assistance:
        desired = np.unwrap(route_yaw.copy())
        if boundary and not bouts and all(len(entries) == 1 for entries in runs):
            # A stationary append has no new directional intent. Preserve the
            # displayed heading instead of restoring native global pelvis yaw.
            desired[:] = np.arctan2(initial_r[0, 0, 2], initial_r[0, 2, 2])
            route_yaw[:] = desired
        if boundary:
            inherited_yaw = np.arctan2(initial_r[0, 0, 2], initial_r[0, 2, 2])
            desired, boundary_heading_target_fade = _boundary_heading(
                desired, inherited_yaw, fps, bouts, runs)
        native_heading = np.unwrap(np.arctan2(raw_base_r[:, 0, 0, 2], raw_base_r[:, 0, 2, 2]))
        heading_correction = desired-native_heading
        common = Rotation.from_euler('y', heading_correction[:, None]).as_matrix()
        base_r = common[:, None]@raw_base_r
        base_p = np.array([_fk(p[0], r, rest) for p, r in zip(base_p, base_r)])
        if boundary:
            # The committed pose is authoritative down to the stored floats.
            base_r[0, nonlegs] = initial_r[nonlegs]
    targets = np.empty((count, 2, 3))
    foot_r = np.empty((count, 2, 3, 3))
    stance = np.zeros((count, 2), bool)
    flat_stance = np.zeros_like(stance)
    anchors = []
    clearance_profiles = []
    for side, entries in enumerate(runs):
        pivot = contact_local[side]
        samples = footprint[side]
        for a, b, proposed, rotation in entries:
            heading = rotation[[0, 2], 2]; heading /= np.linalg.norm(heading)
            proposed = proposed.copy()
            proposed[[0, 2]] += _anchor_advances.get((side, a), 0.)*heading
            candidates = []
            for shift in np.r_[0., np.linspace(-.16, .16, 33)]:
                point = proposed.copy(); point[[0, 2]] += shift*heading
                points = point+(samples-pivot)@rotation.T
                try:
                    heights = np.array([_support(geometry, q) for q in points])
                except ValueError:
                    continue
                if np.ptp(heights) < .005:
                    if _clearance_mode:
                        planted = point.copy(); planted[1] = heights.max()+.004
                        try:
                            safe = all(_support(geometry, q)-q[1] <= .001
                                for angle in np.linspace(0., .4, 5)
                                for q in planted+(samples-pivot)@(rotation@Rotation.from_euler('x', angle).as_matrix()).T)
                        except ValueError:
                            safe = False
                        if not safe:
                            continue
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
        for (begin, end, _, _), (start, _, _, _) in zip(entries, entries[1:]):
            first, last = (max(begin+2, end-2) if _clearance_mode else end-1), start
            u = np.linspace(0., 1., last-first+1); blend = u*u*(3-2*u)
            rotations = Slerp([0., 1.], Rotation.from_matrix(np.stack([foot_r[first, side], foot_r[last, side]])))(blend).as_matrix()
            path = _swing_path(targets[first, side], targets[last, side],
                               foot_r[first, side], foot_r[last, side], blend, _follow_turn_swing)
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
            if amplitude > .24 and not _clearance_mode:
                return assist_rig_clip(native_positions, native_rotations, geometry, character, fps=fps,
                    initial_assisted_positions=initial_assisted_positions,
                    initial_assisted_rotations=initial_assisted_rotations,
                    _anchor_advances=_anchor_advances, _repair_attempt=_repair_attempt, _clearance_mode=True,
                    _minimum_stance_frames=_minimum_stance_frames, _follow_turn_swing=_follow_turn_swing, heading_assistance=heading_assistance)
            if _clearance_mode:
                path, rotations, lift = _clearance_profile(geometry,
                    targets[first, side], targets[last, side], foot_r[first, side],
                    foot_r[last, side], samples, pivot, len(path), _follow_turn_swing)
                clearance_profiles.append({'side': side, 'first': first, 'last': last,
                                            'maximum_lift_m': lift})
                stance[first+1:last, side] = False
                flat_stance[first+1:last, side] = False
                next(row for row in anchors if row['side'] == side and row['first'] == begin)['last_exclusive'] = first+1
            else:
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
        # A boundary-constrained first landing may occur later than its
        # nominal route crossing. Repair only the resulting trailing anchor,
        # by the exact horizontal reach deficit at the unchanged root floor.
        # Initial committed feet never move. Rebuild every affected swing and
        # rerun the full mesh/support/pose checks, with bounded attempts.
        advances = dict(_anchor_advances)
        required = {}
        initial_conflict = False
        for t in frames:
            for side, (hip, knee, foot) in enumerate(LEGS):
                length = np.linalg.norm(rest[knee]-rest[hip])+np.linalg.norm(rest[foot]-rest[knee])-.008
                hip_offset = base_p[t, hip, 1]-base_p[t, 0, 1]
                vertical = lower[t]+hip_offset-ankles[t, side, 1]
                available = length*length-vertical*vertical
                if available <= 0:
                    continue
                entries = runs[side]
                entry = max(i for i, row in enumerate(entries) if row[0] <= t)
                a, b, _, rotation = entries[entry]
                if a == 0:
                    offset = base_p[t, hip, [0, 2]]-ankles[t, side, [0, 2]]
                    initial_conflict |= float(offset@offset) > available+1e-8
                    continue
                heading = rotation[[0, 2], 2]
                heading = heading/np.linalg.norm(heading)
                offset = base_p[t, hip, [0, 2]]-ankles[t, side, [0, 2]]
                along = float(offset@heading)
                cross_squared = float(offset@offset-along*along)
                if available <= cross_squared:
                    continue
                deficit = along-np.sqrt(available-cross_squared)
                weight = 1.
                if t >= b and entry+1 < len(entries):
                    u = (t-(b-1))/(entries[entry+1][0]-(b-1))
                    weight = 1-u*u*(3-2*u)
                if deficit > 1e-5 and weight > .1:
                    key = (side, a)
                    required[key] = max(required.get(key, 0.), (deficit+.001)/weight)
        for key, amount in required.items():
            advances[key] = advances.get(key, 0.)+amount
        if _repair_attempt < 3 and required and max(abs(x) for x in advances.values()) <= .16:
            return assist_rig_clip(native_positions, native_rotations, geometry, character, fps=fps,
                initial_assisted_positions=initial_assisted_positions,
                initial_assisted_rotations=initial_assisted_rotations,
                _anchor_advances=advances, _repair_attempt=_repair_attempt+1, _clearance_mode=_clearance_mode,
                _minimum_stance_frames=_minimum_stance_frames, _follow_turn_swing=_follow_turn_swing, heading_assistance=heading_assistance)
        if boundary and initial_conflict and _minimum_stance_frames > 3:
            # Three planted samples meet the planner's existing stability
            # minimum. The nominal four-sample initial stance can lag a fast
            # inherited path; shorten it by one without moving its anchor.
            return assist_rig_clip(native_positions, native_rotations, geometry, character, fps=fps,
                initial_assisted_positions=initial_assisted_positions,
                initial_assisted_rotations=initial_assisted_rotations,
                _clearance_mode=_clearance_mode, _minimum_stance_frames=3,
                _follow_turn_swing=_follow_turn_swing, heading_assistance=heading_assistance)
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
              'native_retargeted_upperbody_rotations_preserved': bool(np.array_equal(rotations[:, nonlegs], raw_base_r[:, nonlegs])),
              'heading_assistance_enabled': bool(heading_assistance),
              'boundary_heading_target_fade': boundary_heading_target_fade,
              'heading_assistance_scope': 'common world-Y alignment to contact/route heading; native data unchanged' if heading_assistance else None,
              'max_display_heading_rate_deg_s': float(np.degrees(abs(np.diff(np.unwrap(np.arctan2(base_r[:, 0, 0, 2], base_r[:, 0, 2, 2]))))).max()*fps),
              'max_display_heading_correction_deg': float(np.degrees(abs(np.angle(np.exp(1j*heading_correction)))).max()),
              'native_relative_upperbody_rotations_preserved': bool(all(np.allclose(
                  rotations[:, _PARENTS[j]].transpose(0, 2, 1)@rotations[:, j],
                  raw_base_r[:, _PARENTS[j]].transpose(0, 2, 1)@raw_base_r[:, j], atol=2e-6, rtol=0)
                  for j in nonlegs if j)),
              'frames': count, 'fps': fps, 'stance_anchors': anchors,
              'bounded_clearance_profiles': clearance_profiles,
              'swept_clearance_contact_mode': _clearance_mode,
              'minimum_stance_samples': _minimum_stance_frames,
              'route_tangent_swing_paths': _follow_turn_swing,
              'contact_reach_repair_passes': _repair_attempt,
              'contact_reach_advances_m': [{'side': side, 'first': first, 'advance_m': amount}
                                          for (side, first), amount in sorted(_anchor_advances.items())],
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
              'swept_sole_proxy': {'interframe_samples': 7, 'footprint_grid': [3, 2, 5] if _clearance_mode else [3, 5], 'margin_m': .003,
                                  'note': 'finite rigid sole-envelope samples using linear foot translation and rotation slerp; not continuous triangle collision'},
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
    actual_penetration = report['max_actual_mesh_sole_penetration_m']
    penetration_failed = ((actual_penetration is not None and actual_penetration > .008)
                          or swept_penetration > .008)
    if not _clearance_mode and penetration_failed:
        # Samplewise sine arcs can clear every stored pose yet strike a riser
        # between poses. Retry with the existing bounded swept-envelope solve
        # before accepting or rejecting; its output reruns every quality gate.
        return assist_rig_clip(native_positions, native_rotations, geometry, character, fps=fps,
            initial_assisted_positions=initial_assisted_positions,
            initial_assisted_rotations=initial_assisted_rotations,
            _anchor_advances=_anchor_advances, _repair_attempt=_repair_attempt, _clearance_mode=True,
            _minimum_stance_frames=_minimum_stance_frames, _follow_turn_swing=_follow_turn_swing,
            heading_assistance=heading_assistance)
    if not _follow_turn_swing and penetration_failed:
        return assist_rig_clip(native_positions, native_rotations, geometry, character, fps=fps,
            initial_assisted_positions=initial_assisted_positions,
            initial_assisted_rotations=initial_assisted_rotations,
            _anchor_advances=_anchor_advances, _repair_attempt=_repair_attempt, _clearance_mode=_clearance_mode,
            _minimum_stance_frames=_minimum_stance_frames, _follow_turn_swing=True, heading_assistance=heading_assistance)
    return poses, rotations, stance, report
