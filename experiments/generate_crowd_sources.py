"""Bounded fresh Core walking source capture, with native history continuity.

This script never provisions, restarts or mutates a worker.  The main crowd
agent coordinates the shared Core lane before it is run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from realtime_client import RealtimeClient


STYLES = {
    'relaxed': ('A person walks steadily straight ahead through a city crossing at a relaxed natural pace. '
                'Keep a continuous alternating gait and easy arm swing; do not stop or turn.', 1.0, 611),
    'brisk': ('A person walks briskly straight ahead through a busy city crossing. '
              'Keep continuous alternating footfalls and natural arm swing; do not run, stop or turn.', 1.35, 719),
}


def write_json(path: Path, item):
    path.write_text(json.dumps(item, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8769')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-jobs', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.max_jobs <= 12:
        parser.error('max-jobs must be 1–12')
    args.output.mkdir(parents=True, exist_ok=False)
    client = RealtimeClient(args.url, args.token_file.read_text().strip(), timeout=5, job_timeout=100)
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'service': args.url,
              'job_cap': args.max_jobs, 'jobs': [], 'status': 'running', 'health_before': client.health()}
    write_json(args.output / 'report.json', report)
    submitted = 0
    try:
        for style, (prompt, speed, seed) in STYLES.items():
            history = None
            for horizon in range(2):
                if submitted >= args.max_jobs:
                    break
                first_z = horizon * speed * 2
                body = {
                    'request_id': 'crowd-' + uuid.uuid4().hex,
                    'stage_kind': 'approach' if horizon == 0 else 'continuation',
                    'frames': 40, 'prompt': prompt, 'actor_ids': ['walker'], 'seed': seed,
                    'actor_prompts': {'walker': prompt},
                    'root_targets': {'walker': [
                        {'frame': frame, 'position_xz': [0., first_z + speed * frame / 20.], 'heading': 0.}
                        for frame in (0, 7, 15, 23, 31, 39)]},
                }
                if horizon == 0:
                    body['initial_placements'] = {'walker': {'position_xz': [0., 0.], 'yaw': 0.}}
                else:
                    body['history'] = {'native_features': history.tolist()}
                request_path = args.output / f'{style}-{horizon}-request.json'
                write_json(request_path, body)
                job = {'style': style, 'horizon': horizon, 'request_id': body['request_id'],
                       'request_file': request_path.name, 'status': 'submitted',
                       'submitted_at': datetime.now(timezone.utc).isoformat()}
                report['jobs'].append(job)
                write_json(args.output / 'report.json', report)
                started = time.monotonic()
                submitted += 1
                clip = client.wait(body)[0]
                job['generation_seconds'] = time.monotonic() - started
                job['status'] = 'complete'
                archive_path = args.output / f'{style}-{horizon}-core.npz'
                np.savez_compressed(archive_path, positions=clip.positions, rotations=clip.rotations,
                                    native_features=clip.native_features,
                                    metadata=np.array(json.dumps(clip.metadata)))
                job['source_file'] = archive_path.name
                job['source_sha256'] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
                history = clip.native_features[:, -40:]
                write_json(args.output / 'report.json', report)
        report['status'] = 'complete'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        if report['jobs'] and report['jobs'][-1]['status'] == 'submitted':
            report['jobs'][-1]['status'] = 'failed_or_uncertain'
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report['jobs_submitted'] = submitted
        try:
            report['health_after'] = client.health()
        except Exception as exc:
            report['health_after_error'] = str(exc)
        write_json(args.output / 'report.json', report)
        print(json.dumps({'status': report['status'], 'jobs_submitted': submitted,
                          'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
