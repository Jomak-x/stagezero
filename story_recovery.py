"""Shared, bounded recovery policy for generated G1 scenes and action edits.

These geometric proxies detect sustained upright recovery, not stunt accuracy.
They never synthesize, translate, or replace model-generated poses.
"""

import re

import numpy as np

from motion_quality import JOINT_INDEX, ROOT, SHOULDERS


RECOVERY_PATTERN = (r'\b(?:get(?:s|ting)? up|stand(?:s|ing)? up|'
                    r'ris(?:e|es|ing) (?:back )?to (?:their |his |her |the )?feet|'
                    r'push(?:es|ing)? up from lying)\b')
NEGATION = re.compile(r"\b(?:not|never|without|avoid|don't|doesn't)\b", re.I)
EXPLICIT_TIMING = re.compile(r'\b(?:\d+(?:\.\d+)?|a|one|two|three|four|five|six|seven|eight|nine|ten|'
                             r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|'
                             r'twenty|thirty|forty|fifty|sixty|ninety)\s*(?:seconds?|secs?|s|minutes?|mins?)\b', re.I)
MAX_RECOVERY_EXTENSIONS = 2
MAX_RECOVERY_BEAT_FRAMES = 750
FLOOR_RECOVERY_PROMPT = 'A person pushes up from lying on the floor and stands upright.'
_QUALITY_FAILURE = re.compile(
    r'^(?:RuntimeError: )?Generated candidates failed motion-quality checks(?: \(([a-z_, ]+)\))?'
    r'(?:; no motion committed\. Retry the instruction\.)?$', re.I)
_QUALITY_REASONS = frozenset(('intra_clip_jump', 'foot_slide', 'joint_jump', 'horizon_seam',
                              'history_seam', 'floor_penetration', 'invalid_rotations',
                              'nonfinite_positions'))


def quality_failure_reasons(error):
    """Recognize only the backend's bounded quality rejection, never transport errors."""
    match = _QUALITY_FAILURE.fullmatch(str(error))
    if match is None:
        return None
    reasons = tuple(part.strip().lower() for part in match.group(1).split(',')) if match.group(1) else ()
    return reasons if all(reason in _QUALITY_REASONS for reason in reasons) else None


def is_recovery_motion(text):
    return not NEGATION.search(text) and bool(re.search(RECOVERY_PATTERN, text, re.I))


def recovery_prompt(prompt, context=''):
    """Use a tested terminal-state instruction for an unambiguous floor get-up."""
    floor = re.search(r'\b(?:floor|ground|lying|fall(?:s|ing)?|fell)\b', context + ' ' + prompt, re.I)
    seated = re.search(r'\b(?:chair|seat|bench)\b', prompt, re.I)
    return FLOOR_RECOVERY_PROMPT if is_recovery_motion(prompt) and floor and not seated else prompt


def _upright_frames(positions):
    """Scale/translation-independent standing proxy, with G1's Y-up axis."""
    p = np.asarray(positions)
    if p.ndim != 3 or p.shape[1:] != (34, 3) or not np.isfinite(p).all():
        raise ValueError('Recovery poses must be finite G1 (frames, 34, 3) positions')
    torso = p[:, SHOULDERS, :].mean(axis=1) - p[:, ROOT]
    torso_length = np.linalg.norm(torso, axis=1)
    leg_lengths, ankle_heights = [], []
    for side in ('left', 'right'):
        hip, knee, ankle = (p[:, JOINT_INDEX[f'{side}_{name}_skel']]
                            for name in ('hip_yaw', 'knee', 'ankle_roll'))
        leg_lengths.append(np.linalg.norm(hip - knee, axis=1)
                           + np.linalg.norm(knee - ankle, axis=1))
        ankle_heights.append(ankle[:, 1])
    leg_length = np.mean(leg_lengths, axis=0)
    elevation = p[:, ROOT, 1] - np.mean(ankle_heights, axis=0)
    return ((leg_length > 1e-5) & (torso_length > 1e-5)
            & (elevation >= .80 * leg_length)
            & (torso[:, 1] >= .80 * torso_length))


def recovered_upright(position_parts):
    """Require the final 10 frames (0.4 seconds) to be upright."""
    if not position_parts:
        return False
    positions = np.concatenate(position_parts[-3:], axis=0)[-10:]
    return len(positions) == 10 and bool(_upright_frames(positions).all())


def recovery_completion_frame(positions, min_frames=50):
    """Earliest exclusive upright endpoint, in the supplied array's coordinates.

    Auto can stop there instead of asking the model to repeat a completed rise.
    Callers preserve any duration explicitly specified by the user.
    """
    run = 0
    for end, upright in enumerate(_upright_frames(positions), 1):
        run = run + 1 if upright else 0
        if end >= max(10, min_frames) and run >= 10:
            return end
    return None
