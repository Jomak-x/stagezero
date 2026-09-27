"""Read-only geometric diagnostics for raw 30 fps paired InterGen samples.

Contact proximity is a screening measurement, not a contact/visual acceptance test.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def longest_interval(mask):
    starts = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)) == -1)
    if not len(starts):
        return None
    i = int(np.argmax(ends - starts))
    return {'start_frame': int(starts[i]), 'end_frame_exclusive': int(ends[i]),
            'duration_seconds': float((ends[i] - starts[i]) / 30)}


def measure(path):
    with np.load(path, allow_pickle=False) as data:
        joints = data['joints'].copy()
        metadata = json.loads(data['metadata'].item())
    if (joints.ndim != 4 or joints.shape[1:] != (2, 22, 3)
            or not np.isfinite(joints).all() or metadata.get('fps') != 30):
        raise ValueError('Expected finite native 30 fps joints[T,2,22,3]')
    root_gap = np.linalg.norm(joints[:, 0, 0] - joints[:, 1, 0], axis=-1)
    pairs = []
    for a in (20, 21):
        for b in (20, 21):
            distance = np.linalg.norm(joints[:, 0, a] - joints[:, 1, b], axis=-1)
            pairs.append({'joint_pair': [a, b], 'minimum_m': float(distance.min()),
                          'longest_under_15cm': longest_interval(distance < .15)})
    steps = np.linalg.norm(np.diff(joints, axis=0), axis=-1)
    return {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'prompt': metadata.get('prompt'), 'seed': metadata.get('seed'),
            'frames': len(joints), 'fps': 30, 'duration_seconds': len(joints) / 30,
            'raw_source_unmodified': True,
            'root_gap_m': {'minimum': float(root_gap.min()), 'median': float(np.median(root_gap)),
                           'first': float(root_gap[0]), 'last': float(root_gap[-1])},
            'wrist_pairs': pairs,
            'joint_step_m': {'maximum': float(steps.max()), 'p95': float(np.percentile(steps, 95))},
            'generation_seconds': metadata.get('generation_seconds'),
            'peak_cuda_bytes': metadata.get('peak_cuda_bytes'),
            'visual_acceptance': 'requires independent complete-scene review',
            'limitations': ['No mesh collision test', 'Wrist proximity is not palm or finger contact',
                            'No geometric threshold determines whether the requested action succeeded']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output already exists; preserve earlier measurements')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([measure(path) for path in args.archives], indent=2) + '\n')
