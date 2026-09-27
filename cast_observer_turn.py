"""Bounded fresh-Core turns for a released native22 observer.

This creates candidates, never edits a paired take or certifies a whole scene.
The caller's client.wait proxy must archive raw requests/results, including
rejections. Only actual Core native330 outputs may initialize continuation.
"""
from __future__ import annotations

import math
import uuid
import numpy as np

from native_pair_transition import authored_direction_bridge, core_to_pair_anatomy

MIN_INTERVAL_FRAMES = 150
SOURCE_FPS, OUTPUT_FPS, HORIZON = 20, 30, 40
LIMITS = {
    'maximum_root_excursion_m': .30,
    'maximum_root_height_gap_m': .30,
    'maximum_heading_error_degrees': 30.,
    'maximum_tail_foot_speed_m_s': .25,
    'maximum_tail_root_speed_m_s': .10,
    'maximum_tail_joint_speed_m_s': .35,
    'maximum_tail_foot_xz_travel_m': .06,
    'maximum_support_height_gap_m': .10,
    'maximum_wrist_above_root_m': .35,
    'maximum_joint_speed_m_s': 5.,
    'maximum_entry_wrist_rise_m': .10,
}
FEET = [7, 8, 10, 11]


class ObserverTurnRejected(ValueError):
    """A rejected candidate is retained for diagnosis, never for publication."""
    def __init__(self, message, *, candidate=None, report=None):
        super().__init__(message)
        self.candidate = None if candidate is None else np.asarray(candidate).copy()
        self.candidate_joints = self.candidate
        self.report = report if report is not None else {}


class ObserverTurnIneligible(ObserverTurnRejected):
    """Skip this interval without calling Core."""


def _angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def _heading(pose):
    across = pose[1]-pose[2]
    if np.linalg.norm(across[[0, 2]]) < 1e-6:
        raise ValueError('Observer pose has ambiguous hip heading')
    return math.atan2(-across[2], across[0])


def _pair(track):
    # Existing pair helpers need two actors; these are computational copies,
    # never a second actor in a Core request or a jointly generated pair.
    return np.repeat(track[:, None], 2, axis=1)


def _display(raw, left):
    fitted, report = core_to_pair_anatomy(_pair(raw), _pair(left))
    fitted = fitted[:, 0]
    old = np.arange(len(fitted))/SOURCE_FPS
    new = np.arange(len(fitted)*OUTPUT_FPS//SOURCE_FPS)/OUTPUT_FPS
    shown = np.stack([np.interp(new, old, c) for c in fitted.reshape(len(fitted), -1).T], axis=-1)
    return shown.reshape(-1, 22, 3), fitted, report


def _measure(motion, left, target, initial_error, *, fps):
    root = left[-1, 0]
    tail = motion[-10:]  # Measure real source samples before interpolation/hold.
    speed = np.linalg.norm(np.diff(tail, axis=0)*fps, axis=-1)
    foot_travel = np.linalg.norm(np.diff(tail[:, FEET][:, :, [0, 2]], axis=0), axis=-1).sum(0)
    end_delta = target-motion[-1, 0, [0, 2]]
    if np.linalg.norm(end_delta) < .05:
        raise ValueError('Observer reached the target; facing is ambiguous')
    error = abs(math.degrees(_angle(_heading(motion[-1])-math.atan2(end_delta[0], end_delta[1]))))
    root_excursion = float(np.linalg.norm(motion[:, 0][:, [0, 2]]-root[[0, 2]], axis=-1).max())
    height = float(np.abs(motion[:, 0, 1]-root[1]).max())
    support_gap = float(np.max(np.abs(tail[:, FEET, 1].min(axis=1)-left[-1, FEET, 1].min())))
    measured = {
        'maximum_root_excursion_m': root_excursion,
        'maximum_root_height_gap_m': height,
        'initial_heading_error_degrees': initial_error,
        'final_heading_error_degrees': error,
        'heading_improvement_degrees': initial_error-error,
        'maximum_tail_foot_speed_m_s': float(speed[:, FEET].max()),
        'maximum_tail_root_speed_m_s': float(speed[:, 0].max()),
        'maximum_tail_joint_speed_m_s': float(speed.max()),
        'maximum_tail_foot_xz_travel_m': float(foot_travel.max()),
        'maximum_support_height_gap_m': support_gap,
        'maximum_wrist_above_root_m': float((tail[:, [20, 21], 1]-tail[:, :1, 1]).max()),
        'source_tail_frames': len(tail), 'source_tail_fps': fps,
    }
    reasons = [key+' exceeds limit' for key in (
        'maximum_root_excursion_m', 'maximum_root_height_gap_m',
        'maximum_tail_foot_speed_m_s', 'maximum_tail_root_speed_m_s',
        'maximum_tail_joint_speed_m_s', 'maximum_tail_foot_xz_travel_m',
        'maximum_support_height_gap_m', 'maximum_wrist_above_root_m')
        if measured[key] > LIMITS[key]]
    if error > LIMITS['maximum_heading_error_degrees']:
        reasons.append('observer does not face the next active pair')
    if initial_error > 30 and initial_error-error < min(30., initial_error*.5):
        reasons.append('observer does not attain meaningful heading improvement')
    measured['settled_tail_passed'] = not any('tail_' in x or 'support_height' in x or 'wrist_' in x for x in reasons)
    measured['rejection_reasons'] = reasons
    return measured


def generate_observer_turn(left, client, *, actor_id, target_xz, duration_frames,
                           seed, cancelled=lambda: False):
    """Return (full interval track, report), or a typed failure with evidence.

    left is >=2 exact preceding native22 frames at 30fps. The returned interval
    starts AFTER left[-1]. No input pose or raw Core output is modified. At most
    three sequential horizons are requested: an unplayed neutral warmup,
    its genuine native330 continuation turn, and only if needed one settling
    continuation. Every output is retained by the archiving client.
    Collision/scene validation and visual acceptance remain caller obligations.
    """
    left = np.asarray(left, dtype=float)
    target = np.asarray(target_xz, dtype=float)
    if left.ndim != 3 or left.shape[1:] != (22, 3) or len(left) < 2 or not np.isfinite(left).all():
        raise ValueError('left must contain at least two finite native22 poses')
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError('target_xz must contain two finite coordinates')
    if type(duration_frames) is not int or duration_frames < 1:
        raise ValueError('duration_frames must be a positive integer')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be a uint32 integer')
    if not isinstance(actor_id, str) or not 1 <= len(actor_id) <= 64:
        raise ValueError('actor_id must be a nonempty string of at most 64 characters')
    report = {'actor_id': actor_id, 'duration_frames': duration_frames, 'fps': OUTPUT_FPS,
              'source_fps': SOURCE_FPS, 'limits': dict(LIMITS), 'windows': [], 'spans': [],
              'source_pair_frames_modified': False, 'fresh_initialization': True,
              'native_history_policy': 'Only actual prior Core native330 features; no InterGen conversion',
              'raw_output_archiving': 'caller client.wait proxy', 'foot_lock': False,
              'scene_validation': 'required by caller', 'visual_acceptance': 'unverified',
              'mechanical_gate_passed': False, 'animation_accepted': False}
    if duration_frames < MIN_INTERVAL_FRAMES:
        report['rejection_reasons'] = ['interval shorter than 150 frames']
        raise ObserverTurnIneligible('Observer turn needs at least 150 inactive frames', report=report)
    root = left[-1, 0, [0, 2]].copy()
    delta = target-root
    if np.linalg.norm(delta) < .05:
        raise ObserverTurnIneligible('Observer target is too close to define heading', report=report)
    initial = _heading(left[-1])
    desired = math.atan2(delta[0], delta[1])
    change = _angle(desired-initial)
    initial_error = abs(math.degrees(change))
    previous, candidate, chunks = None, None, []
    try:
        for window in range(3):
            if cancelled():
                raise RuntimeError('Observer turn generation cancelled')
            warmup = window == 0
            prompt = ('A person stands still with arms hanging relaxed at their sides and both feet comfortably planted.'
                      if warmup else
                      'A person turns with small natural steps in place to watch two other people while keeping '
                      'their arms relaxed at their sides, then stands still with both feet comfortably planted.')
            if window == 2:
                prompt = 'A person stands still watching two other people, with arms relaxed at sides and both feet comfortably planted.'
            goals = []
            start_yaw = initial if warmup else _heading(fitted[-1])
            remaining = 0. if warmup else _angle(desired-start_yaw)
            for frame in (0, 7, 15, 23, 31, 39):
                alpha = min(1., frame/27.)
                heading = _angle(start_yaw+alpha*alpha*(3-2*alpha)*remaining)
                goals.append({'frame': frame, 'position_xz': root.tolist(), 'heading': heading})
            request = {'request_id': 'observer-turn-'+uuid.uuid4().hex,
                       'stage_kind': 'approach' if window == 0 else 'continuation',
                       'frames': HORIZON, 'prompt': prompt, 'actor_prompts': {actor_id: prompt},
                       'actor_ids': [actor_id], 'seed': seed, 'root_targets': {actor_id: goals}}
            if previous is None:
                request['initial_placements'] = {actor_id: {'position_xz': root.tolist(), 'yaw': initial}}
            else:
                request['history'] = {'native_features': previous.native_features.copy().tolist()}
            returned = client.wait(request, cancelled=cancelled)
            if len(returned) != 1:
                raise ValueError('Expected exactly one Core observer horizon')
            clip = returned[0]
            if (tuple(clip.actor_ids) != (actor_id,) or clip.fps != SOURCE_FPS or clip.frames != HORIZON
                    or clip.positions.shape != (1, HORIZON, 27, 3)
                    or clip.native_features is None or clip.native_features.shape != (1, HORIZON, 330)
                    or not np.isfinite(clip.positions).all() or not np.isfinite(clip.native_features).all()):
                raise ValueError('Core returned incompatible observer actors, timing, positions or native history')
            previous = clip
            if not warmup:
                chunks.append(clip.positions[0].copy())
            raw = clip.positions[0].copy() if warmup else np.concatenate(chunks)
            candidate, fitted, anatomy = _display(raw, left)
            report['anatomy'] = anatomy
            report['spans'] = [{'start_frame': 0, 'end_frame_exclusive': len(candidate),
                                'source': 'ardy_core', 'model_generated': True,
                                'display_processing': 'native anatomy direction fit; linear 20 to 30fps interpolation',
                                'source_frames': len(fitted)}]
            measure_target = root+3*np.array([math.sin(initial), math.cos(initial)]) if warmup else target
            measured = _measure(fitted, left, measure_target, 0. if warmup else initial_error, fps=SOURCE_FPS)
            if not warmup:
                relative_wrists = fitted[:, [20, 21], 1]-fitted[:, :1, 1]
                release_wrists = left[-1, [20, 21], 1]-left[-1, 0, 1]
                measured['maximum_entry_wrist_rise_m'] = float((relative_wrists[0]-release_wrists).max())
                measured['maximum_played_wrist_above_root_m'] = float(relative_wrists.max())
                if measured['maximum_entry_wrist_rise_m'] > LIMITS['maximum_entry_wrist_rise_m']:
                    measured['rejection_reasons'].append('entry wrist rise exceeds limit')
                if measured['maximum_played_wrist_above_root_m'] > LIMITS['maximum_wrist_above_root_m']:
                    measured['rejection_reasons'].append('played wrist height exceeds limit')
            report['windows'].append({'index': window, 'request_id': request['request_id'],
                                      'role': 'unplayed_neutral_warmup' if warmup else ('played_continuation_turn' if window == 1 else 'played_settling_continuation'),
                                      'used_in_performance': not warmup, **measured})
            if cancelled():
                raise RuntimeError('Observer turn generation cancelled')
            if warmup:
                report['warmup'] = measured
                report['warmup_played'] = False
                if measured['rejection_reasons']:
                    raise ValueError('Neutral warmup rejected: '+'; '.join(measured['rejection_reasons']))
            else:
                reasons = measured['rejection_reasons']
                # A settling continuation cannot repair spatial or arm violations.
                if not reasons or any(not reason.startswith('maximum_tail_') for reason in reasons):
                    break
        report['generated_motion'] = measured
        if measured['rejection_reasons']:
            raise ValueError('; '.join(measured['rejection_reasons']))
        report['bridge_attempts'] = []
        generated = candidate
        for frames in (21, 24, 30):
            try:
                bridge, boundary = authored_direction_bridge(_pair(left), _pair(generated),
                                                             left_fps=30, right_fps=30, frames=frames)
            except ValueError as exc:
                report['bridge_attempts'].append({'frames': frames, 'error': str(exc), 'mechanical_gate_passed': False})
                continue
            report['bridge_attempts'].append(boundary)
            candidate = np.concatenate([bridge[:, 0], generated])
            report['spans'] = [
                {'start_frame': 0, 'end_frame_exclusive': frames,
                 'source': 'authored_transition', 'model_generated': False},
                {'start_frame': frames, 'end_frame_exclusive': len(candidate),
                 'source': 'ardy_core', 'model_generated': True,
                 'display_processing': 'native anatomy direction fit; linear 20 to 30fps interpolation',
                 'source_frames': len(fitted)},
            ]
            if boundary['mechanical_gate_passed']:
                break
        else:
            raise ValueError('No 21/24/30 frame observer bridge passes existing strict mechanical gates')
        hold_frames = duration_frames-len(candidate)
        if hold_frames < 0:
            raise ValueError('Inactive interval cannot contain the measured observer turn')
        generated_end = len(candidate)
        candidate = np.concatenate([candidate, np.repeat(candidate[-1:], hold_frames, axis=0)])
        path = np.concatenate([left[-1:], candidate])
        max_speed = float(np.linalg.norm(np.diff(path, axis=0)*OUTPUT_FPS, axis=-1).max())
        excursion = float(np.linalg.norm(candidate[:, 0][:, [0, 2]]-root, axis=-1).max())
        height = float(np.abs(candidate[:, 0, 1]-left[-1, 0, 1]).max())
        wrist_height = float((candidate[:, [20,21],1]-candidate[:,:1,1]).max())
        report['maximum_composed_wrist_above_root_m'] = wrist_height
        if wrist_height > LIMITS['maximum_wrist_above_root_m']:
            raise ValueError('Composed observer wrist height exceeds limit')
        report.update(maximum_joint_speed_m_s=max_speed, maximum_root_excursion_m=excursion,
                      maximum_root_height_gap_m=height)
        report['spans'] = [
            {'start_frame': 0, 'end_frame_exclusive': frames, 'source': 'authored_transition', 'model_generated': False},
            {'start_frame': frames, 'end_frame_exclusive': generated_end, 'source': 'ardy_core', 'model_generated': True,
             'display_processing': 'native anatomy direction fit; linear 20 to 30fps interpolation',
             'source_frames': len(fitted)},
        ]
        if hold_frames:
            report['spans'].append({'start_frame': generated_end, 'end_frame_exclusive': duration_frames,
                                    'source': 'stationary_hold', 'model_generated': False,
                                    'settled_tail_passed': True})
        report['span_end_policy'] = 'exclusive'
        if max_speed > LIMITS['maximum_joint_speed_m_s'] or excursion > LIMITS['maximum_root_excursion_m'] or height > LIMITS['maximum_root_height_gap_m']:
            raise ValueError('Composed observer exceeds speed/root/height limits')
        report['mechanical_gate_passed'] = True
        report['rejection_reasons'] = []
        return candidate, report
    except Exception as exc:
        report['rejection_reasons'] = [str(exc)]
        if isinstance(exc, ValueError):
            raise ObserverTurnRejected(str(exc), candidate=candidate, report=report) from exc
        # Cancellation and worker/transport failures must stop the whole job.
        exc.observer_report = report
        exc.observer_candidate = candidate
        raise
