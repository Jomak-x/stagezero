"""Rigidly place stored ARDY takes without GPU inference or changing their poses.

ARDY motion arrays are normalized, not world coordinates. Match the official
ArdyMotionRep.rotate/translate_2d operations on all features before normalizing
again so the appended result remains valid continuation history.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from live_motion import MODEL


@lru_cache(maxsize=1)
def motion_statistics():
    path = Path(__file__).resolve().parent / 'assets' / 'motion-normalization.npz'
    with np.load(path, allow_pickle=False) as data:
        mean, std = data['mean'].copy(), data['std'].copy()
        metadata = json.loads(str(data['metadata']))
    if (metadata.get('model') != MODEL or metadata.get('epsilon') != 1e-5
            or mean.shape != (414,) or std.shape != (414,)
            or not np.isfinite(mean).all() or not np.isfinite(std).all()
            or np.any(std < 0)):
        raise ValueError('Incompatible motion normalization data')
    scale = np.sqrt(std ** 2 + np.float32(1e-5))
    mean.flags.writeable = scale.flags.writeable = False
    return mean, scale


def align_take(source, target):
    """Return copied arrays with source start at target end in XZ and heading.

Keep source Y coordinates (including floor contact and jumps) unchanged. This
is a cut between recorded poses, not a generated or interpolated transition.
Feature ranges follow vendor/ardy/ardy/motion_rep/reps/ardy_motionrep.py:
root 0:3; heading 3:5; joints 5:104; rotations 104:308;
velocities 308:410; foot contacts 410:414.
"""
    mean, scale = motion_statistics()
    features = source.motion * scale + mean
    end = target.motion[-1] * scale + mean
    start_heading, end_heading = features[0, 3:5], end[3:5]
    if min(np.linalg.norm(start_heading), np.linalg.norm(end_heading)) < 1e-6:
        raise ValueError('Take has no usable facing direction')
    angle = np.arctan2(end_heading[1], end_heading[0]) - np.arctan2(start_heading[1], start_heading[0])
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=np.float32)
    offset = target.positions[-1, 0] - source.positions[0, 0] @ rotation.T
    offset[1] = 0
    positions = source.positions @ rotation.T + offset
    rotations = rotation @ source.rotations
    features[:, :3] = features[:, :3] @ rotation.T + offset
    # ARDY heading is [cos(theta), sin(theta)], not an ordinary XZ vector.
    features[:, 3:5] = features[:, 3:5] @ np.array(((c, s), (-s, c)), dtype=np.float32)
    features[:, 5:104] = (features[:, 5:104].reshape(-1, 33, 3) @ rotation.T).reshape(-1, 99)
    # 6D rotations store matrix columns 0 and 1, each a 3-vector.
    features[:, 104:308] = (features[:, 104:308].reshape(-1, 34, 2, 3) @ rotation.T).reshape(-1, 204)
    features[:, 308:410] = (features[:, 308:410].reshape(-1, 34, 3) @ rotation.T).reshape(-1, 102)
    motion = ((features - mean) / scale).astype(source.motion.dtype)
    if any(not np.isfinite(array).all() for array in (positions, rotations, motion)):
        raise ValueError('Take alignment produced invalid motion')
    return positions, rotations, motion
