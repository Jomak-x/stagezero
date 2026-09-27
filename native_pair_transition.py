"""Direct Core pose display and honest boundaries around native paired motion.

Core output uses a semantic joint subset, never InterGen→Core→skin retargeting.
One shared rigid placement preserves every inter-actor distance. Optional bridge
frames are separately authored positions, never joint model output or contact IK.
No function here mutates, resamples, or overwrites the native interaction frames.
"""
from __future__ import annotations

import hashlib
import math
import uuid

import numpy as np

from experiments.native_pair_rig import NAMES as NATIVE22_NAMES, PARENTS


# Exact vendor/ardy/ardy/skeleton/definitions.py CoreSkeleton27 bone order.
CORE27_NAMES = (
    'Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandEnd',
    'RightHandThumb1', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'LeftHandEnd', 'LeftHandThumb1', 'RightUpLeg', 'RightLeg', 'RightFoot',
    'RightToeBase', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
)
CORE_TO_NATIVE = tuple(CORE27_NAMES.index(name) for name in NATIVE22_NAMES)


def core27_to_native22(positions, *, joint_names=CORE27_NAMES):
    """Copy exact named Core endpoints into authored rig's 22-joint order.

    Accepts arbitrary leading dimensions, including one pose or T,actors.
    Names are required to cover the exact Core schema (permutations allowed).
    Core's extra Spine3 and hand endpoints/thumbs are omitted. Thus the upper
    torso hierarchy is approximated, but no source elbow, wrist or root moves.
    This changes display representation only, not the source model identity.
    """
    points = np.asarray(positions)
    names = tuple(joint_names)
    if len(names) != 27 or set(names) != set(CORE27_NAMES):
        raise ValueError('Expected the 27 unique semantic Core joint names')
    if points.ndim < 2 or points.shape[-2:] != (27, 3) or not np.issubdtype(points.dtype, np.number) or not np.isfinite(points).all():
        raise ValueError('Expected finite Core positions[...,27,3]')
    return points[..., [names.index(name) for name in NATIVE22_NAMES], :].copy()


def core_adapter_provenance():
    return {
        'source': 'ARDY Core27', 'display': 'authored native22 rig',
        'method': 'exact semantic position subset; no FK or proportion fit',
        'joint_mapping': dict(zip(NATIVE22_NAMES, CORE_TO_NATIVE)),
        'omitted_core_joints': [name for name in CORE27_NAMES if name not in NATIVE22_NAMES],
        'native_pair_model_output': False,
        'limitations': ['Core Spine3 is omitted; upper torso hierarchy differs.',
                       'Core proportions remain Core proportions; they can differ at native pair boundaries.',
                       'Position-only display does not preserve all source joint twist.'],
    }


def _pair(value, *, min_frames=1):
    points = np.asarray(value, dtype=float)
    if points.ndim != 4 or points.shape[1:] != (2, 22, 3) or len(points) < min_frames or not np.isfinite(points).all():
        raise ValueError(f'Expected at least {min_frames} finite pair poses[T,2,22,3]')
    return points


def _fps(value):
    if isinstance(value, bool) or not np.isscalar(value) or not math.isfinite(value) or not 1 <= value <= 240:
        raise ValueError('FPS must be finite and between 1 and 240')
    return float(value)


def shared_place_pair(joints, *, yaw=0., translation=(0., 0., 0.)):
    """Rotate about world +Y (radians) and translate both actors identically.

    No scale, per-actor offset, root recentering, or floor snapping is allowed.
    Returned coordinates are a copy; native features must remain source data.
    """
    points = _pair(joints)
    offset = np.asarray(translation, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all() or not np.isscalar(yaw) or not math.isfinite(yaw):
        raise ValueError('One finite yaw and shared XYZ translation are required')
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    return points @ rotation.T + offset


def _summary(vectors):
    values = np.linalg.norm(vectors, axis=-1)
    return {'mean': float(values.mean()), 'max': float(values.max()),
            'per_actor_max': values.reshape(-1, 2, 22).max(axis=(0, 2)).tolist()}


def boundary_diagnostics(left, right, *, left_fps, right_fps):
    """Expose raw position and measured velocity mismatch before any blending."""
    a, b = _pair(left, min_frames=2), _pair(right, min_frames=2)
    fa, fb = _fps(left_fps), _fps(right_fps)
    va, vb = (a[-1] - a[-2]) * fa, (b[1] - b[0]) * fb
    return {
        'position_gap_m': _summary(b[0] - a[-1]),
        'root_gap_m': np.linalg.norm(b[0, :, 0] - a[-1, :, 0], axis=-1).tolist(),
        'root_relative_pose_gap_m': _summary((b[0] - b[0, :, :1]) - (a[-1] - a[-1, :, :1])),
        'velocity_gap_m_s': _summary(vb - va),
        'left_endpoint_speed_m_s': _summary(va),
        'right_endpoint_speed_m_s': _summary(vb),
        'left_fps': fa, 'right_fps': fb,
        'seamless_verified': False,
        'interpretation': 'Geometric boundary diagnostics only; skin, feet, obstacles and contact require review.',
    }


def _lengths(poses):
    return np.linalg.norm(poses[..., 1:, :] - poses[..., np.asarray(PARENTS[1:]), :], axis=-1)


def authored_boundary_bridge(left, right, *, left_fps, right_fps,
                             output_fps=30., frames=15, max_speed_m_s=5.,
                             max_bone_length_change_fraction=.15,
                             max_endpoint_velocity_error_m_s=1e-6):
    """Return separate authored frames and mechanical gate report.

    The cubic passes through four specified samples: the preceding endpoint,
    its one-frame velocity extrapolation, the following endpoint's one-frame
    velocity backprojection, and that endpoint. Therefore finite-difference
    endpoint velocities match measured source velocities at output_fps.

    This is NOT a rigid skeleton, contact solver, or physically valid transition.
    Gates can reject speed, bone distortion and endpoint velocity; a passing
    report is still not animation acceptance. Do not insert these frames inside
    a native interaction or claim either actor was jointly generated with Core.
    """
    a, b = _pair(left, min_frames=2), _pair(right, min_frames=2)
    fa, fb, fps = _fps(left_fps), _fps(right_fps), _fps(output_fps)
    if type(frames) is not int or not 2 <= frames <= 300:
        raise ValueError('Authored bridge requires 2–300 inserted frames')
    for value in (max_speed_m_s, max_bone_length_change_fraction, max_endpoint_velocity_error_m_s):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('Mechanical gate thresholds must be finite and positive')
    va, vb = (a[-1] - a[-2]) * fa, (b[1] - b[0]) * fb
    # Normalize polynomial time for numerical stability over long bridges.
    last = frames + 1
    times = np.array([0., 1. / last, frames / last, 1.])
    samples = np.stack([a[-1], a[-1] + va / fps, b[0] - vb / fps, b[0]])
    coefficients = np.linalg.solve(np.vander(times, 4), samples.reshape(4, -1))
    bridge = (np.vander(np.arange(1, last) / last, 4) @ coefficients).reshape(frames, 2, 22, 3)
    # Keep the promised endpoint finite differences exact to ordinary roundoff.
    bridge[0], bridge[-1] = samples[1], samples[2]
    path = np.concatenate([a[-1:], bridge, b[:1]])
    speed = float(np.linalg.norm(np.diff(path, axis=0) * fps, axis=-1).max())
    errors = np.stack([(bridge[0] - a[-1]) * fps - va, (b[0] - bridge[-1]) * fps - vb])
    endpoint_error = float(np.linalg.norm(errors, axis=-1).max())
    # Compare lengths with the interpolated endpoint lengths, explicitly exposing
    # source anatomy differences separately below. No division by zero segments.
    alpha = np.arange(1, last)[:, None, None] / last
    start_lengths, end_lengths = _lengths(a[-1]), _lengths(b[0])
    reference_lengths = (1 - alpha) * start_lengths + alpha * end_lengths
    if np.any(reference_lengths < 1e-6):
        raise ValueError('Cannot bridge degenerate anatomical segments')
    distortion = float(np.max(abs(_lengths(bridge) / reference_lengths - 1)))
    endpoint_lengths = float(np.max(abs(end_lengths / np.maximum(start_lengths, 1e-6) - 1)))
    reasons = []
    if speed > max_speed_m_s:
        reasons.append('authored bridge exceeds maximum joint speed')
    if distortion > max_bone_length_change_fraction:
        reasons.append('authored bridge distorts anatomical segment lengths')
    if endpoint_lengths > max_bone_length_change_fraction:
        reasons.append('source endpoint skeleton proportions differ beyond the configured limit')
    if endpoint_error > max_endpoint_velocity_error_m_s:
        reasons.append('authored bridge does not meet endpoint velocity tolerance')
    report = {
        'source': 'authored boundary interpolation', 'model_generated': False,
        'method': 'cubic position interpolation through endpoint velocity samples',
        'frames': frames, 'fps': fps, 'inserted_duration_seconds': frames / fps,
        'source_pair_frames_modified': False, 'native_features_available': False,
        'input_boundary': boundary_diagnostics(a, b, left_fps=fa, right_fps=fb),
        'max_joint_speed_m_s': speed,
        'max_endpoint_velocity_error_m_s': endpoint_error,
        'max_segment_distortion_fraction': distortion,
        'max_source_endpoint_length_change_fraction': endpoint_lengths,
        'thresholds': {'max_speed_m_s': max_speed_m_s,
                       'max_bone_length_change_fraction': max_bone_length_change_fraction,
                       'max_endpoint_velocity_error_m_s': max_endpoint_velocity_error_m_s},
        'mechanical_gate_passed': not reasons, 'rejection_reasons': reasons,
        'visual_acceptance': 'unverified', 'contact_preserved_in_bridge': False,
        'source_endpoint_sha256': hashlib.sha256(samples[[0, 3]].tobytes()).hexdigest(),
        'limitations': ['No collision, floor, foot locking, contact or rigid-bone solve.',
                       'Endpoint velocity continuity does not establish acceleration continuity.',
                       'Bridge has its own timing; preserve adjacent clips at their original FPS.',
                       'Use only as a separately disclosed entry or exit; native interaction stays untouched.'],
    }
    return bridge, report


def core_to_pair_anatomy(positions, reference_pair):
    """Explicit Core-only direction retarget to each native actor's median lengths.

    This deliberately moves Core display endpoints, unlike core27_to_native22.
    Every bone retains its Core world direction; native pair frames are never
    passed through this operation. The root XZ target path is retained. A Y-only
    shift preserves the minimum source ankle/toe height, not physical foot locks.
    """
    core = _pair(core27_to_native22(positions))
    reference = _pair(reference_pair)
    lengths = np.median(_lengths(reference), axis=0)
    vectors = core[:, :, 1:] - core[:, :, np.asarray(PARENTS[1:])]
    core_lengths = np.linalg.norm(vectors, axis=-1)
    if np.any(core_lengths < 1e-6) or np.any(lengths < 1e-6):
        raise ValueError('Cannot retarget degenerate anatomical segments')
    directions = vectors / core_lengths[..., None]
    out = np.empty_like(core)
    out[:, :, 0] = core[:, :, 0]
    for joint, parent in enumerate(PARENTS[1:], 1):
        out[:, :, joint] = out[:, :, parent] + directions[:, :, joint-1] * lengths[None, :, joint-1, None]
    feet = [7, 8, 10, 11]
    vertical = core[:, :, feet, 1].min(axis=-1) - out[:, :, feet, 1].min(axis=-1)
    out[:, :, :, 1] += vertical[:, :, None]
    report = {
        'source': 'ARDY Core', 'method': 'Core world bone directions with native actor median segment lengths',
        'reference_frames': len(reference), 'median_lengths_m': lengths.tolist(),
        'core_endpoint_displacement_m': _summary(out-core),
        'root_xz_modified': False, 'native_pair_modified': False,
        'root_y_correction_range_m': [float(vertical.min()), float(vertical.max())],
        'floor_policy': 'Preserve source minimum ankle/toe height with one Y-only root translation each frame.',
        'foot_lock': False,
        'limitations': ['Core wrist and elbow endpoints move as part of this explicit morphology retarget.',
                       'No FK from the old 17-bone rig and no InterGen-to-Core conversion.',
                       'Core foot sliding is not repaired; ground contact and mesh penetration remain unverified.',
                       'Native source lengths vary while Core display lengths use actor-specific medians.'],
    }
    return out, report


def _hermite(p0, p1, v0, v1, t, duration):
    return ((2*t**3-3*t**2+1)*p0 + (t**3-2*t**2+t)*duration*v0
            + (-2*t**3+3*t**2)*p1 + (t**3-t**2)*duration*v1)


def authored_direction_bridge(left, right, *, left_fps, right_fps, output_fps=30.,
                              frames=21, max_speed_m_s=5.,
                              max_endpoint_velocity_error_m_s=.35,
                              max_bone_length_change_fraction=.20,
                              match_sampled_endpoints=False):
    """Author a length-preserving direction bridge; report real sampled seams.

    Uses normalized Hermite interpolation of bone directions with tangential
    endpoint velocity, and linear interpolation of each endpoint bone length.
    Root Hermite motion uses measured velocity. This avoids Cartesian limb
    shortening. Finite-difference velocity still differs and is explicitly gated.
    Optional sampled endpoint correction matches one measured sample per seam,
    with radial correction bounded to 15% of interpolated segment length.
    No contact frame is edited and there is no foot-lock or collision solve.
    """
    a, b = _pair(left, min_frames=2), _pair(right, min_frames=2)
    fa, fb, fps = _fps(left_fps), _fps(right_fps), _fps(output_fps)
    if type(frames) is not int or not 2 <= frames <= 300:
        raise ValueError('Authored bridge requires 2–300 inserted frames')
    for value in (max_speed_m_s, max_endpoint_velocity_error_m_s, max_bone_length_change_fraction):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('Mechanical gate thresholds must be finite and positive')
    def directions(p):
        vector = p[:, :, 1:] - p[:, :, np.asarray(PARENTS[1:])]
        length = np.linalg.norm(vector, axis=-1)
        if np.any(length < 1e-6):
            raise ValueError('Cannot bridge degenerate anatomical segments')
        return vector/length[..., None], length
    da, la = directions(a[-2:]); db, lb = directions(b[:2])
    va = (da[1]-da[0])*fa; vb = (db[1]-db[0])*fb
    va -= (va*da[1]).sum(-1, keepdims=True)*da[1]
    vb -= (vb*db[0]).sum(-1, keepdims=True)*db[0]
    alpha = np.arange(1, frames+1)/(frames+1)
    duration = (frames+1)/fps
    raw = _hermite(da[1], db[0], va, vb, alpha[:, None, None, None], duration)
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    if np.any(norms < .1):
        raise ValueError('Direction bridge crosses an ambiguous opposite-bone orientation')
    blended = raw/norms
    lengths = (1-alpha[:, None, None])*la[1] + alpha[:, None, None]*lb[0]
    reference_lengths = lengths.copy()
    root = _hermite(a[-1, :, 0], b[0, :, 0], (a[-1, :, 0]-a[-2, :, 0])*fa,
                    (b[1, :, 0]-b[0, :, 0])*fb, alpha[:, None, None], duration)
    if match_sampled_endpoints:
        # Native positions have measured radial as well as angular velocity.
        # Tangent-only interpolation cannot reproduce those finite differences.
        # Match one extrapolated sample at each seam, fading its correction over
        # five inserted frames. The interior retains the original interpolation;
        # bounded support avoids duration-dependent Hermite length overshoot.
        samples = np.stack([a[-1]+(a[-1]-a[-2])*fa/fps,
                            b[0]-(b[1]-b[0])*fb/fps])
        sample_directions, sample_lengths = directions(samples)
        width = min(5, frames/2)
        phase = np.clip(1-np.arange(frames)/width, 0, 1)
        weight = phase*phase*(3-2*phase)
        for edge, fade in ((0, weight), (-1, weight[::-1])):
            sample = 0 if edge == 0 else 1
            blended += fade[:, None, None, None]*(sample_directions[sample]-blended[edge])
            lengths += fade[:, None, None]*(sample_lengths[sample]-lengths[edge])
            root += fade[:, None, None]*(samples[sample, :, 0]-root[edge])
        direction_norms = np.linalg.norm(blended, axis=-1, keepdims=True)
        if np.any(direction_norms < .1) or np.any(lengths < 1e-6):
            raise ValueError('Sampled endpoint correction creates degenerate anatomical segments')
        blended /= direction_norms
    out = np.empty((frames, 2, 22, 3))
    out[:, :, 0] = root
    for joint, parent in enumerate(PARENTS[1:], 1):
        out[:, :, joint] = out[:, :, parent] + blended[:, :, joint-1]*lengths[:, :, joint-1, None]
    path = np.concatenate([a[-1:], out, b[:1]])
    speed = float(np.linalg.norm(np.diff(path, axis=0)*fps, axis=-1).max())
    errors = np.stack([(out[0]-a[-1])*fps-(a[-1]-a[-2])*fa,
                       (b[0]-out[-1])*fps-(b[1]-b[0])*fb])
    error = float(np.linalg.norm(errors, axis=-1).max())
    proportion_change = float(np.max(abs(lb[0]/la[1]-1)))
    distortion = float(np.max(abs(lengths/reference_lengths-1)))
    reasons = []
    if speed > max_speed_m_s: reasons.append('authored bridge exceeds maximum joint speed')
    if error > max_endpoint_velocity_error_m_s: reasons.append('sampled endpoint velocity exceeds tolerance')
    if proportion_change > max_bone_length_change_fraction: reasons.append('source endpoint proportions differ beyond tolerance')
    if distortion > .15: reasons.append('sampled endpoint correction distorts anatomical segment lengths')
    feet = path[:, :, [7, 8, 10, 11]]
    foot_step = np.linalg.norm(np.diff(feet[..., [0, 2]], axis=0), axis=-1)
    report = {
        'source': 'authored direction transition', 'model_generated': False,
        'method': 'normalized Hermite bone directions, endpoint lengths, and root Hermite',
        'sampled_endpoint_correction': bool(match_sampled_endpoints),
        'sampled_endpoint_correction_support_frames': min(5, frames/2) if match_sampled_endpoints else 0,
        'frames': frames, 'fps': fps, 'source_pair_frames_modified': False,
        'input_boundary': boundary_diagnostics(a, b, left_fps=fa, right_fps=fb),
        'max_joint_speed_m_s': speed, 'max_endpoint_velocity_error_m_s': error,
        'max_source_endpoint_length_change_fraction': proportion_change,
        'max_segment_distortion_fraction': distortion,
        'max_bone_length_interpolation_error_m': float(np.max(abs(_lengths(out)-lengths))),
        'min_ankle_toe_height_m': float(feet[..., 1].min()),
        'max_ankle_toe_xz_travel_m': float(foot_step.sum(axis=0).max()),
        'thresholds': {'max_speed_m_s': max_speed_m_s,
                       'max_endpoint_velocity_error_m_s': max_endpoint_velocity_error_m_s,
                       'max_segment_distortion_fraction': .15,
                       'max_bone_length_change_fraction': max_bone_length_change_fraction},
        'mechanical_gate_passed': not reasons, 'rejection_reasons': reasons,
        'foot_lock': False, 'contact_preserved_in_bridge': False, 'visual_acceptance': 'unverified',
        'limitations': ['Authored motion, not jointly generated with Core.',
                       'Bone directions are normalized; optional endpoint radial corrections are separately bounded.',
                       'Endpoint velocity is measured at sampled frames; continuous derivatives alone are not proof.',
                       'No foot locking, floor, partner-contact or obstacle solve.'],
    }
    return out, report


def compose_ardy_pair_context(approach_core, placed_pair, exit_core, *, blend_frames=21):
    """Compose an explicitly authored 30fps review timeline from raw arrays.

    Core inputs are T,2,27,3 at 20fps. The pair is T,2,22,3 at 30fps and is
    copied unchanged into the result. Source arrays remain separately available
    to the caller; there are no valid unified model features for this timeline.
    """
    pair = _pair(placed_pair, min_frames=4)
    paths, refinement = {}, {}
    for phase, value in [('approach', approach_core), ('exit', exit_core)]:
        core = np.asarray(value, dtype=float)
        if core.ndim != 4 or core.shape[1:] != (2, 27, 3) or len(core) < 4 or not np.isfinite(core).all():
            raise ValueError('Expected finite Core poses[T,2,27,3] with at least four frames')
        count = math.ceil(len(core)*1.5)
        old_time, new_time = np.arange(len(core))/20., np.arange(count)/30.
        flat = core.reshape(len(core), -1)
        resampled = np.stack([np.interp(new_time, old_time, c) for c in flat.T], axis=-1).reshape(count, 2, 27, 3)
        paths[phase], refinement[phase] = core_to_pair_anatomy(resampled, pair)
        refinement[phase]['resampling'] = {'source_fps': 20, 'display_fps': 30,
            'method': 'linear Core positions then normalized bone directions and fixed actor lengths',
            'final_sample_hold_seconds': max(0., float(new_time[-1]-old_time[-1]))}
    entry, entry_report = authored_direction_bridge(paths['approach'], pair, left_fps=30, right_fps=30, frames=blend_frames)
    exit_frames, exit_report = authored_direction_bridge(pair, paths['exit'], left_fps=30, right_fps=30, frames=blend_frames)
    pieces = [
        ('ARDY Core approach; anatomy retarget and 20 to 30 fps interpolation', 'ardy_core', 'approach', paths['approach']),
        ('Authored direction entry; no foot lock', 'authored_transition', 'transition', entry),
        ('Native30 paired interaction; shared rigid placement only', 'intergen', 'paired_action', pair),
        ('Authored direction exit; no foot lock', 'authored_transition', 'transition', exit_frames),
        ('ARDY Core departure; anatomy retarget and 20 to 30 fps interpolation', 'ardy_core', 'departure', paths['exit']),
    ]
    segments = []; start = 0
    for label, source, kind, data in pieces:
        segments.append({'label': label, 'source': source, 'kind': kind,
                         'start_frame': start, 'end_frame_exclusive': start+len(data), 'frames': len(data)})
        start += len(data)
    combined = np.concatenate([data for *_, data in pieces])
    bounds = segments[2]
    if not np.array_equal(combined[bounds['start_frame']:bounds['end_frame_exclusive']], pair):
        raise RuntimeError('Native pair preservation check failed')
    provenance = {'refinement': refinement, 'boundaries': {'entry': entry_report, 'exit': exit_report},
                  'source_pair_frames_modified': False,
                  'all_mechanical_gates_passed': entry_report['mechanical_gate_passed'] and exit_report['mechanical_gate_passed'],
                  'visual_acceptance': 'unverified'}
    metadata = {'model': 'ARDY Core + InterGen', 'fps': 30, 'frames': len(combined),
                'prompt': 'Core approach, paired interaction, Core departure; authored transitions.',
                'segments': segments, 'transition_provenance': provenance,
                'source_pair_preserved_exactly_after_shared_placement': True,
                'animation_accepted': False,
                'features': 'No unified model features; preserve raw source archives separately.'}
    return combined, metadata


def build_ardy_pair_context(pair_clip, client, scene_document, *, yaw=0.,
                           translation=(0., 0., 0.), approach_offset=.65,
                           seed=92642, cancelled=lambda: False, on_core_chunk=None):
    """Generate four bounded warm-Core windows and compose a review candidate.

    `pair_clip` must be an InterGen NativePairClip. This function does not load a
    model or restart services. The supplied client owns authentication and job
    deadlines. `on_core_chunk(phase, window, clip, request)` can preserve every
    raw completed chunk before the next job. Return includes raw Core positions,
    rotations and native features separately from composed display positions.

    Exit deliberately starts a fresh independent Core sample; InterGen never
    masquerades as Core native history. A successful result still requires full
    scene visual review, especially source foot motion and authored transitions.
    """
    from realtime_navigation import validate_ground_path
    from studio_interaction_scene import adapt_studio_scene

    if getattr(pair_clip, 'fps', None) != 30 or getattr(pair_clip, 'metadata', {}).get('model') != 'InterGen':
        raise ValueError('ARDY context currently requires an original InterGen native30 clip')
    raw_pair = _pair(pair_clip.joints, min_frames=4)
    if len(raw_pair) > 718:
        raise ValueError('Pair exceeds the 1000-frame composed display limit after adding Core and transitions')
    if not math.isfinite(approach_offset) or not .1 <= approach_offset <= 1.5:
        raise ValueError('Approach offset must be between 0.1 and 1.5 metres')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be a uint32 integer')
    pair = shared_place_pair(raw_pair, yaw=yaw, translation=translation)
    scene = adapt_studio_scene(scene_document)['scene']
    ids = list(getattr(pair_clip, 'actor_ids', ('actor_1', 'actor_2')))
    if len(ids) != 2:
        raise ValueError('ARDY paired context requires exactly two actors')
    generated = {}; floor_checks = {}; requests = []
    for phase, edge in [('approach', 0), ('exit', -1)]:
        roots = pair[edge, :, 0][:, [0, 2]]
        across = pair[edge, :, 1]-pair[edge, :, 2]
        yaws = np.arctan2(-across[:, 2], across[:, 0])
        outward = roots-roots.mean(0)
        norms = np.linalg.norm(outward, axis=-1, keepdims=True)
        if np.any(norms < .1):
            raise ValueError('Pair roots are too close for a safe radial approach plan')
        outward /= norms
        start = roots+outward*approach_offset if phase == 'approach' else roots
        end = roots if phase == 'approach' else roots+outward*approach_offset
        floor_checks[phase+'_planned'] = [validate_ground_path(scene, [a, b]) for a, b in zip(start, end)]
        clips = []
        for window in range(2):
            if cancelled():
                raise RuntimeError('ARDY context generation cancelled')
            prompt = ('A person walks slowly forward and stops in a relaxed standing pose.'
                      if phase == 'approach' and window == 0 else
                      'A person stands still in a relaxed neutral pose.' if phase == 'approach' or window == 0 else
                      'A person turns away and walks slowly forward.')
            goals = {}
            for i, aid in enumerate(ids):
                goals[aid] = []
                for frame in (0, 7, 15, 23, 31, 39):
                    progress = min(1., frame/31.) if window == (0 if phase == 'approach' else 1) else (1. if phase == 'approach' else 0.)
                    point = start[i]*(1-progress)+end[i]*progress
                    angle = np.arctan2(outward[i, 0], outward[i, 1]) if phase == 'exit' and window == 1 else yaws[i]
                    goals[aid].append({'frame': frame, 'position_xz': point.tolist(), 'heading': float(angle)})
            request = {'request_id': 'native-context-'+uuid.uuid4().hex,
                       'stage_kind': 'approach' if window == 0 else 'continuation',
                       'frames': 40, 'prompt': prompt, 'actor_ids': ids, 'seed': seed, 'root_targets': goals}
            if window == 0:
                request['initial_placements'] = {aid: {'position_xz': start[i].tolist(), 'yaw': float(yaws[i])} for i, aid in enumerate(ids)}
            else:
                request['history'] = {'native_features': clips[-1].native_features[:, -40:].tolist()}
            returned = client.wait(request, cancelled=cancelled)
            if len(returned) != 1 or returned[0].native_features is None:
                raise ValueError('Expected one complete native Core window per job')
            clip = returned[0]
            clips.append(clip); requests.append(request)
            if on_core_chunk is not None:
                on_core_chunk(phase, window, clip, request)
        generated[phase] = {name: np.concatenate([getattr(c, name) for c in clips], axis=1)
                            for name in ('positions', 'rotations', 'native_features')}
        floor_checks[phase+'_actual'] = [validate_ground_path(scene, p[:, 0, :][:, [0, 2]]) for p in generated[phase]['positions']]
    joints, metadata = compose_ardy_pair_context(generated['approach']['positions'].transpose(1, 0, 2, 3), pair,
                                                generated['exit']['positions'].transpose(1, 0, 2, 3))
    metadata['transition_provenance'].update({'gpu_jobs': 4, 'floor_checks': floor_checks,
        'shared_placement': {'yaw': float(yaw), 'translation': np.asarray(translation).tolist()},
        'exit_policy': 'Fresh independent Core sample at pair final roots; no fabricated native history.'})
    return {'joints': joints, 'metadata': metadata, 'raw_core': generated,
            'source_pair': pair_clip, 'requests': requests}


def stabilize_authored_feet(joints, segments, *, fps=30., speed_limit_m_s=.25,
                           height_band_m=.05, max_correction_m=.08):
    """Optional bounded ankle-XZ planting on Core/authored segments only.

    Low/slow intervals are heuristic, not model contacts. Knee position follows
    analytic two-bone IK with original segment lengths and bend half-plane.
    Unreachable anchors are skipped. Pelvis, root path, arms, source interaction,
    and first/last two frames of every segment stay exactly unchanged. Toe
    vectors retain their original orientation; this is ankle planting, not a
    complete contact or collision solver. Always compare full playback visually.
    """
    p = _pair(joints, min_frames=4); rate = _fps(fps)
    from native_pair_clip import validated_segments
    normalized = validated_segments({'segments': segments}, len(p))
    if normalized is None:
        raise ValueError('Explicit source segments are required for selective foot planting')
    for v in (speed_limit_m_s, height_band_m, max_correction_m):
        if isinstance(v, bool) or not math.isfinite(v) or v <= 0:
            raise ValueError('Foot planting thresholds must be finite and positive')
    out = p.copy(); records = []; changes = 0; unreachable = 0
    for segment in normalized:
        if segment['source'] not in ('ardy_core', 'authored_transition'):
            continue
        start, end = segment['start'], segment['end']; count = end-start
        if count < 9:
            continue
        for actor in range(2):
            for hip, knee, ankle, toe in ((1, 4, 7, 10), (2, 5, 8, 11)):
                foot = p[start:end, actor, ankle]
                steps = np.linalg.norm(np.diff(foot[:, [0, 2]], axis=0), axis=-1)*rate
                speed = np.maximum(np.r_[steps[0], steps], np.r_[steps, steps[-1]])
                low = foot[:, 1] <= np.percentile(foot[:, 1], 10)+height_band_m
                eligible = low & (speed <= speed_limit_m_s)
                eligible[:2] = False; eligible[-2:] = False
                edges = np.flatnonzero(np.diff(np.r_[False, eligible, False]))
                for lo, hi in zip(edges[::2], edges[1::2]):
                    if hi-lo < 5:
                        continue
                    anchor = np.median(foot[lo:hi][:, [0, 2]], axis=0)
                    record = {'segment_start': start, 'actor': actor, 'ankle_joint': ankle,
                              'start_frame': start+int(lo), 'end_frame_exclusive': start+int(hi),
                              'anchor_xz_m': anchor.tolist(), 'applied_frames': 0, 'unreachable_frames': 0,
                              'over_correction_limit_frames': 0}
                    for local in range(lo, hi):
                        frame = start+local
                        # Ease the first/last three eligible samples, so a source
                        # segment boundary never inherits a contact correction.
                        u = min(1., (local-lo)/3., (hi-1-local)/3.)
                        weight = u*u*(3-2*u)
                        target = p[frame, actor, ankle].copy()
                        delta = anchor-target[[0, 2]]
                        distance = np.linalg.norm(delta)
                        if weight == 0:
                            continue
                        if distance > max_correction_m:
                            record['over_correction_limit_frames'] += 1
                            continue
                        target[[0, 2]] += weight*delta
                        h = p[frame, actor, hip]; k = p[frame, actor, knee]
                        original_ankle = p[frame, actor, ankle]
                        upper, lower = np.linalg.norm(k-h), np.linalg.norm(original_ankle-k)
                        axis = target-h; reach = np.linalg.norm(axis)
                        if not abs(upper-lower)+1e-6 < reach < upper+lower-1e-6:
                            unreachable += 1; record['unreachable_frames'] += 1
                            continue
                        axis /= reach
                        bend = k-h; bend -= np.dot(bend, axis)*axis
                        if np.linalg.norm(bend) < 1e-6:
                            unreachable += 1; record['unreachable_frames'] += 1
                            continue
                        bend /= np.linalg.norm(bend)
                        along = (upper*upper-lower*lower+reach*reach)/(2*reach)
                        radius = math.sqrt(max(0., upper*upper-along*along))
                        out[frame, actor, knee] = h+along*axis+radius*bend
                        out[frame, actor, ankle] = target
                        out[frame, actor, toe] = target+p[frame, actor, toe]-original_ankle
                        record['applied_frames'] += 1; changes += 1
                    # Never toggle a foot constraint on/off within an interval
                    # because reachability changed. Reject the entire interval.
                    if record['unreachable_frames'] or record['over_correction_limit_frames']:
                        for joint in (knee, ankle, toe):
                            out[start+lo:start+hi, actor, joint] = p[start+lo:start+hi, actor, joint]
                        changes -= record['applied_frames']
                        record['applied_frames'] = 0
                        record['rejected_whole_interval'] = True
                    else:
                        record['rejected_whole_interval'] = False
                    records.append(record)
    modified = np.linalg.norm(out-p, axis=-1)
    return out, {'source': 'authored heuristic ankle plant refinement', 'model_contacts': False,
                 'method': 'low/slow ankle XZ anchors; bounded two-bone IK; original toe offset',
                 'applied_foot_frames': changes, 'unreachable_foot_frames_skipped': unreachable,
                 'maximum_joint_correction_m': float(modified.max()), 'intervals': records,
                 'max_bone_length_error_m': float(np.max(abs(_lengths(out)-_lengths(p)))),
                 'root_positions_modified': False, 'native_pair_modified': False,
                 'visual_acceptance': 'unverified',
                 'thresholds': {'speed_limit_m_s': speed_limit_m_s, 'height_band_m': height_band_m,
                                'max_ankle_xz_correction_m': max_correction_m},
                 'limitations': ['Heuristic contact; no vertical ground solve or collision checks.',
                                'Toe orientation retains source motion, so an anchored ankle can still have toe drift.',
                                'Transition endpoints remain unchanged; incompatible source foot placements cannot both lock.',
                                'IK can alter knee bend; full sequence visual comparison is required.']}
