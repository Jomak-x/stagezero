"""Offline, rejectable Core27 terrain-contact experiment.

This module never changes native motion features or model history. Its output is
only a presentation candidate. Whole-clip phase inference and anchors are not a
streaming implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from motion_bridge import _layout, _swing
from scene_interaction_geometry import SceneInteractionGeometry


MAX_FOOT_RELATIVE_ROOT_M = .12
MAX_ROOT_Y_M = .08
MAX_ROOT_XZ_M = .06
MAX_LEG_ROTATION_DEG = 35.
MAX_ROOT_Y_FRAME_STEP_M = .04
MAX_FOOT_CORRECTION_FRAME_STEP_M = .08
SOLE_PENETRATION_TOLERANCE_M = .01
STANCE_GAP_TOLERANCE_M = .025
STANCE_ANCHOR_SLIP_TOLERANCE_M = .025
FOOTPRINT_HEIGHT_SPREAD_M = .035

NAMES, PARENTS, NEUTRAL = _layout()
INDEX = {name: index for index, name in enumerate(NAMES)}
LEGS = ((INDEX['LeftUpLeg'], INDEX['LeftLeg'], INDEX['LeftFoot'], INDEX['LeftToeBase']),
        (INDEX['RightUpLeg'], INDEX['RightLeg'], INDEX['RightFoot'], INDEX['RightToeBase']))
LEG_JOINTS = {joint for leg in LEGS for joint in leg}
OFFSETS = NEUTRAL - NEUTRAL[np.maximum(PARENTS, 0)]


def _support(geometry, point, reference_y):
    value = geometry.support_height(float(point[0]), float(point[2]), float(reference_y),
                                    max_step_up=.4, max_drop=.4)
    return None if value is None else float(value)


def _sole_points(ankle, foot_rotation):
    """Four Core foot proxies; toe is calibrated to the Core toe landmark."""
    toe = OFFSETS[INDEX['LeftToeBase']].copy()
    # Both Core feet share their ankle-to-toe bind offset.
    samples = np.asarray([[side, toe[1], z] for z in (-.035, toe[2])
                          for side in (-.045, .045)])
    return ankle + samples @ foot_rotation.T


def _phase(positions, fps=20.):
    """Observed slow, low foot intervals; no alternating-foot schedule."""
    stance = np.zeros((len(positions), 2), dtype=bool)
    speed = np.zeros_like(stance, dtype=float)
    for side, (_, _, _, toe) in enumerate(LEGS):
        path = positions[:, toe]
        speed[:, side] = np.linalg.norm(np.gradient(path, axis=0) * fps, axis=1)
        relative_y = path[:, 1] - positions[:, 0, 1]
        low_limit = float(np.quantile(relative_y, .35) + .085)
        active = False
        for frame in range(len(path)):
            low = relative_y[frame] <= low_limit
            active = bool(low and speed[frame, side] < (.75 if active else .42))
            stance[frame, side] = active
        # A single ambiguous frame cannot create or break a planted anchor.
        for frame in range(1, len(path) - 1):
            if stance[frame-1, side] and stance[frame+1, side]:
                stance[frame, side] = True
        for first, last in _runs(stance[:, side]):
            if last - first < 3:
                stance[first:last, side] = False
    return stance, speed


def _runs(mask):
    edge = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))


def _fk(root, globals_):
    positions = np.empty((len(globals_), 3), dtype=float)
    positions[0] = root
    for joint in range(1, len(globals_)):
        parent = PARENTS[joint]
        positions[joint] = positions[parent] + globals_[parent] @ OFFSETS[joint]
    return positions


def _strict_two_bone(hip, target, native_knee, native_ankle, upper_length, lower_length):
    vector = target - hip
    distance = float(np.linalg.norm(vector))
    if not abs(upper_length-lower_length)+1e-5 < distance < upper_length+lower_length-1e-5:
        raise ValueError('leg endpoint is beyond native two-bone reach')
    direction = vector / distance
    bend = native_knee - hip
    bend = bend - np.dot(bend, direction)*direction
    if np.linalg.norm(bend) < .015:
        raise ValueError('native knee bend plane is ambiguous')
    bend /= np.linalg.norm(bend)
    along = (upper_length**2-lower_length**2+distance**2)/(2*distance)
    height = np.sqrt(max(0., upper_length**2-along**2))
    knee = hip + along*direction + height*bend
    return knee


def adapt_clip(native_positions, native_global_rotations, geometry, *, reference_sole_y=None):
    """Return a candidate and a strict acceptance report for one complete clip.

    `reference_sole_y` chooses the reachable tread in overlapping geometry. It
    is the planned sole height, not an IK target; when absent the native pelvis
    height minus the Core neutral pelvis-to-toe distance is used.
    """
    native = np.asarray(native_positions, dtype=float)
    rotations = np.asarray(native_global_rotations, dtype=float)
    if native.ndim != 3 or native.shape[1:] != (27, 3) or rotations.shape != (len(native), 27, 3, 3):
        raise ValueError('Expected Core27 positions[T,27,3] and global rotations[T,27,3,3]')
    if len(native) < 3 or not np.isfinite(native).all() or not np.isfinite(rotations).all():
        raise ValueError('Need at least three finite Core27 frames')
    if not callable(getattr(geometry, 'support_height', None)):
        raise TypeError('Geometry must provide support_height')
    reference = (native[:, 0, 1] + float(NEUTRAL[[leg[3] for leg in LEGS], 1].mean())
                 if reference_sole_y is None else np.asarray(reference_sole_y, dtype=float))
    if reference.shape != (len(native),) or not np.isfinite(reference).all():
        raise ValueError('reference_sole_y must have one finite value per frame')
    fk_error = max(float(np.linalg.norm(_fk(native[t, 0], rotations[t])-native[t], axis=1).max())
                   for t in range(len(native)))
    if fk_error > 1e-3:
        raise ValueError('Core positions and global rotations are not FK-consistent')

    stance, speed = _phase(native)
    anchor = np.full((len(native), 2, 3), np.nan)
    anchor_count = 0
    for side, (_, _, _, toe) in enumerate(LEGS):
        for first, last in _runs(stance[:, side]):
            point = native[first, toe].copy()
            point[[0, 2]] = np.median(native[first:min(last, first+3), toe][:, [0, 2]], axis=0)
            support = _support(geometry, point, reference[first])
            if support is None:
                continue
            point[1] = support
            anchor[first:last, side] = point
            anchor_count += 1

    # Prescribe the smallest toe correction that clears sampled sole geometry.
    desired = np.zeros((len(native), 2, 3), dtype=float)
    invalid = []
    for t in range(len(native)):
        for side, (_, _, foot, toe) in enumerate(LEGS):
            if np.isfinite(anchor[t, side]).all():
                desired[t, side] = anchor[t, side] - native[t, toe]
            else:
                points = _sole_points(native[t, foot], rotations[t, foot])
                penetrations = [height-point[1] for point in points
                                if (height := _support(geometry, point, reference[t])) is not None]
                desired[t, side, 1] = max(0., max(penetrations or [0.]))
    # Select the minimum signed root correction that keeps both foot-relative
    # requests and two-bone reaches feasible. Downward correction is allowed
    # only for observed stance contact; this bound prevents a large squat.
    root_delta = np.zeros(len(native), dtype=float)
    root_infeasible = 0
    candidates = sorted(np.linspace(-MAX_ROOT_Y_M, MAX_ROOT_Y_M, 33), key=lambda value: abs(value))
    def feasible_root(t, height):
        for side, (hip, knee, foot, _) in enumerate(LEGS):
            relative = desired[t, side]-[0., height, 0.]
            if np.linalg.norm(relative) > MAX_FOOT_RELATIVE_ROOT_M+1e-7:
                return False
            moved_hip = native[t, hip]+[0., height, 0.]
            target_ankle = native[t, foot]+desired[t, side]
            if abs(height) < 1e-9 and np.linalg.norm(desired[t, side]) < 1e-9:
                continue
            distance = np.linalg.norm(target_ankle-moved_hip)
            upper = np.linalg.norm(OFFSETS[knee]); lower = np.linalg.norm(OFFSETS[foot])
            if not abs(upper-lower)+1e-5 < distance < upper+lower-1e-5:
                return False
        return True
    for t in range(len(native)):
        fit = next((height for height in candidates if feasible_root(t, height)), None)
        if fit is None:
            root_infeasible += 1
        else:
            root_delta[t] = fit
    # Smooth only the correction. Keep smoothed values only if all hard
    # per-frame endpoint and reach constraints remain satisfied.
    smoothed = np.convolve(np.pad(root_delta, (1, 1), mode='edge'),
                           [1/4, 1/2, 1/4], mode='valid')
    for t in range(len(native)):
        if feasible_root(t, smoothed[t]):
            root_delta[t] = smoothed[t]
    if root_infeasible:
        invalid.append(f'{root_infeasible} frames lack a root-Y solution within .08 m')
    candidate_p = np.empty_like(native)
    candidate_r = rotations.copy()
    reach_failures = 0
    for t in range(len(native)):
        root = native[t, 0].copy()
        root[1] += root_delta[t]
        # Every non-leg global rotation is native; modified local rotations
        # occur only in Core hip/knee/ankle/toe nodes.
        p = _fk(root, candidate_r[t])
        for side, (hip, knee, foot, toe) in enumerate(LEGS):
            correction = desired[t, side].copy()
            correction[1] -= root_delta[t]
            if np.linalg.norm(correction) < 1e-9:
                continue
            target_ankle = p[foot] + correction
            upper = float(np.linalg.norm(OFFSETS[knee]))
            lower = float(np.linalg.norm(OFFSETS[foot]))
            try:
                new_knee = _strict_two_bone(p[hip], target_ankle, p[knee], p[foot], upper, lower)
            except ValueError:
                reach_failures += 1
                continue
            candidate_r[t, hip] = _swing(p[knee]-p[hip], new_knee-p[hip]) @ candidate_r[t, hip]
            candidate_r[t, knee] = _swing(p[foot]-p[knee], target_ankle-new_knee) @ candidate_r[t, knee]
            p[knee], p[foot] = new_knee, target_ankle
            p[toe] = p[foot] + candidate_r[t, foot] @ OFFSETS[toe]
        candidate_p[t] = _fk(root, candidate_r[t])

    # Hard acceptance checks on recomputed FK, never on requested endpoints.
    root_correction = candidate_p[:, 0]-native[:, 0]
    toe_correction = candidate_p[:, [leg[3] for leg in LEGS]]-native[:, [leg[3] for leg in LEGS]]-root_correction[:, None]
    foot_magnitude = np.linalg.norm(toe_correction, axis=2)
    if foot_magnitude.max() > MAX_FOOT_RELATIVE_ROOT_M + 1e-5:
        invalid.append('foot-relative-root correction exceeds .12 m')
    if np.linalg.norm(root_correction[:, [0, 2]], axis=1).max() > MAX_ROOT_XZ_M + 1e-9:
        invalid.append('root XZ correction exceeds .06 m')
    if np.max(np.abs(np.diff(root_delta))) > MAX_ROOT_Y_FRAME_STEP_M:
        invalid.append('root correction rate exceeds .04 m/frame')
    if np.max(np.linalg.norm(np.diff(toe_correction, axis=0), axis=2)) > MAX_FOOT_CORRECTION_FRAME_STEP_M:
        invalid.append('foot correction rate exceeds .08 m/frame')
    if reach_failures:
        invalid.append(f'{reach_failures} native knee planes or leg targets are infeasible')
    leg_angle = 0.
    for joint in LEG_JOINTS:
        parent = PARENTS[joint]
        native_local = rotations[:, parent].transpose(0, 2, 1) @ rotations[:, joint]
        fitted_local = candidate_r[:, parent].transpose(0, 2, 1) @ candidate_r[:, joint]
        difference = native_local.transpose(0, 2, 1) @ fitted_local
        leg_angle = max(leg_angle, float(np.degrees(Rotation.from_matrix(difference).magnitude().max())))
    if leg_angle > MAX_LEG_ROTATION_DEG:
        invalid.append('leg local rotation correction exceeds 35 degrees')
    if not np.array_equal(candidate_r[:, 0], rotations[:, 0]):
        invalid.append('root orientation changed')
    if any(not np.array_equal(candidate_r[:, j], rotations[:, j]) for j in range(27) if j not in LEG_JOINTS):
        invalid.append('non-leg rotation changed')

    support_gap = []
    anchor_slip = []
    footprint_spread = []
    penetration = []
    sole_paths = np.empty((len(native), 2, 4, 3), dtype=float)
    for t in range(len(native)):
        for side, (_, _, foot, toe) in enumerate(LEGS):
            points = _sole_points(candidate_p[t, foot], candidate_r[t, foot])
            sole_paths[t, side] = points
            heights = [_support(geometry, point, reference[t]) for point in points]
            penetration.extend([max(0., height-point[1]) for point, height in zip(points, heights)
                                if height is not None])
            if np.isfinite(anchor[t, side]).all():
                gap = abs(candidate_p[t, toe, 1]-anchor[t, side, 1])
                slip = np.linalg.norm(candidate_p[t, toe, [0, 2]]-anchor[t, side, [0, 2]])
                support_gap.append(gap)
                anchor_slip.append(slip)
                if any(height is None for height in heights):
                    invalid.append('stance sole footprint leaves its tread')
                else:
                    footprint_spread.append(max(heights)-min(heights))
    max_penetration = max(penetration or [0.])
    max_gap = max(support_gap or [np.inf])
    max_slip = max(anchor_slip or [np.inf])
    max_spread = max(footprint_spread or [np.inf])
    if max_penetration > SOLE_PENETRATION_TOLERANCE_M:
        invalid.append('corrected sole penetrates rendered geometry')
    if max_gap > STANCE_GAP_TOLERANCE_M:
        invalid.append('stance support gap exceeds .025 m')
    if max_slip > STANCE_ANCHOR_SLIP_TOLERANCE_M:
        invalid.append('stance anchor slip exceeds .025 m')
    if max_spread > FOOTPRINT_HEIGHT_SPREAD_M:
        invalid.append('stance footprint straddles multiple tread heights')
    swept_penetration = 0.
    for t in range(len(native)-1):
        for alpha in (.25, .5, .75):
            points = sole_paths[t]*(1-alpha)+sole_paths[t+1]*alpha
            ref = reference[t]*(1-alpha)+reference[t+1]*alpha
            for point in points.reshape(-1, 3):
                height = _support(geometry, point, ref)
                if height is not None:
                    swept_penetration = max(swept_penetration, height-point[1])
    if swept_penetration > SOLE_PENETRATION_TOLERANCE_M:
        invalid.append('swept sole path penetrates rendered tread/riser')
    report = {
        'accepted': not invalid,
        'rejection_reasons': sorted(set(invalid)),
        'native_fk_error_max_m': fk_error,
        'stance_anchor_count': anchor_count,
        'stance_fraction': stance.mean(axis=0).tolist(),
        'native_toe_speed_median_m_s': np.median(speed, axis=0).tolist(),
        'max_foot_relative_root_correction_m': float(foot_magnitude.max()),
        'max_root_y_correction_m': float(np.max(np.abs(root_delta))),
        'max_root_xz_correction_m': float(np.linalg.norm(root_correction[:, [0,2]], axis=1).max()),
        'max_leg_rotation_change_deg': leg_angle,
        'max_stance_gap_m': float(max_gap),
        'max_anchor_slip_m': float(max_slip),
        'max_footprint_height_spread_m': float(max_spread),
        'max_sole_penetration_m': float(max_penetration),
        'max_swept_sole_penetration_m': float(max(0., swept_penetration)),
        'max_root_correction_frame_step_m': float(np.max(np.abs(np.diff(root_delta)))),
        'max_foot_correction_frame_step_m': float(np.max(np.linalg.norm(np.diff(toe_correction, axis=0), axis=2))),
        'budgets_m': {'foot_relative_root': MAX_FOOT_RELATIVE_ROOT_M,
                      'root_y': MAX_ROOT_Y_M, 'root_xz': MAX_ROOT_XZ_M},
        'note': 'Whole-clip experimental presentation candidate; native features and history remain untouched.',
    }
    return candidate_p, candidate_r, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    geometry = SceneInteractionGeometry.from_scene(json.loads(args.scene.read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.input:
        with np.load(path, allow_pickle=False) as native:
            planned = native['planned_root'][:, 1]-.95 if 'planned_root' in native else None
            poses, rotations, report = adapt_clip(native['positions'], native['rotations'], geometry,
                                                  reference_sole_y=planned)
        (args.output/(path.stem+'.json')).write_text(json.dumps(report, indent=2, allow_nan=False))
        np.savez_compressed(args.output/(path.stem+'.candidate.npz'),
                            positions=poses.astype(np.float32), rotations=rotations.astype(np.float32),
                            accepted=np.array(report['accepted']))
        print(path.stem, 'ACCEPT' if report['accepted'] else 'REJECT',
              ', '.join(report['rejection_reasons']))


if __name__ == '__main__':
    main()
