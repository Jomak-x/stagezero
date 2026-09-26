"""Verify native pose goals with an officially transformed normalized history.

The input NPZ contains motion, positions, and rotations after ARDY's own
unnormalize -> rotate(+pi/2) -> translate_2d([2,-1]) -> normalize path.
Never rotate individual normalized features as if they were world positions.
"""
import argparse
import json
from pathlib import Path
import time
import uuid

import numpy as np
import requests

from verify_motion_backend import PROMPTS, case_summary, decode_generation, save_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8767')
    parser.add_argument('--token', type=Path, default=Path('.runtime/api-token'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.history, allow_pickle=False) as data:
        history = data['motion'][-52:].copy()
        positions = data['positions'][-52:].copy()
    session = requests.Session()
    session.headers['Authorization'] = 'Bearer ' + args.token.read_text().strip()
    report = {'transform': {'rotation_y_rad': float(np.pi/2), 'translation_xz_m': [2, -1]},
              'seed': 303, 'cases': {}, 'status': 'running'}
    for action in ('overhead', 'squat'):
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        response = session.post(args.url + '/generate', timeout=(5, 60), json={
            'request_id': request_id, 'prompt': PROMPTS[action],
            'history': history.tolist(), 'generation_options': {'seed': 303}})
        result = decode_generation(response, request_id, 303, action, expect_pose_goal=True)
        row = case_summary(result, time.perf_counter()-start, positions)
        report['cases'][action] = row
        save_result(args.output / (action + '.npz'), result)
        print(action, row['action_proxy_met'], row['pose_goal']['world_yaw_rotation_rad'])
    report['status'] = 'passed' if all(c['action_proxy_met'] for c in report['cases'].values()) else 'action_failure'
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
