"""Bounded final holdout: two native pose goals, ten untouched seeds each."""
import argparse
import json
from pathlib import Path
import time
import uuid
import requests

from verify_motion_backend import (
    DEFAULT_PROJECT, PROMPTS, case_summary, decode_generation, save_result,
    source_history, write_report,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--token', type=Path, default=Path('.runtime/api-token'))
    parser.add_argument('--source-project', type=Path, default=DEFAULT_PROJECT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.url.startswith(('http://127.0.0.1:', 'http://localhost:')):
        parser.error('Use a private loopback endpoint')
    args.output.mkdir(parents=True, exist_ok=True)
    history, prior, _ = source_history(args.source_project)
    session = requests.Session()
    session.headers['Authorization'] = 'Bearer ' + args.token.read_text().strip()
    report = {'seeds': list(range(201, 211)), 'status': 'running', 'cases': {},
              'design': 'Native directed-v2 pose goals on seeds unused for profile/reference development',
              'caveat': 'Geometric cues on one walking prefix; not human ratings or arbitrary scene reliability'}
    for action in ('overhead', 'squat'):
        for seed in report['seeds']:
            key = f'{action}-seed{seed}'
            request_id = str(uuid.uuid4())
            try:
                start = time.perf_counter()
                response = session.post(args.url + '/generate', timeout=(5, 60), json={
                    'request_id': request_id, 'prompt': PROMPTS[action],
                    'history': history.tolist(), 'generation_options': {'seed': seed}})
                result = decode_generation(response, request_id, seed, action, expect_pose_goal=True)
                row = case_summary(result, time.perf_counter()-start, prior)
                save_result(args.output / (key + '.npz'), result)
            except Exception as exc:
                row = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}', 'action_proxy_met': False}
            report['cases'][key] = row
            write_report(args.output / 'report.json', report)
            print(key, row['status'], row['action_proxy_met'], flush=True)
    report['status'] = 'passed' if all(c['status']=='ok' and c['action_proxy_met'] for c in report['cases'].values()) else 'partial_failure'
    write_report(args.output / 'report.json', report)
    return 0 if report['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
