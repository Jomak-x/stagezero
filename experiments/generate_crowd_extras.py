"""Capture fresh idle, walk start and walk stop Core source horizons."""
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


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def request(name, *, history=None):
    prompt = {
        'idle': 'A person waits calmly at a crossing, standing upright with subtle breathing and small natural weight shifts. Keep both feet grounded.',
        'start': 'A person begins from a still standing pose, then starts walking forward at a relaxed city pace with natural alternating steps and arm swing.',
        'stop': 'A person smoothly slows from walking, takes a last natural step, and settles into a relaxed upright standing pose with both feet grounded.',
    }[name]
    speed = {'idle': 0., 'start': 1., 'stop': .25}[name]
    start_z = 4.0 if name == 'stop' else 0.
    goals = []
    for frame in (0, 7, 15, 23, 31, 39):
        f = frame / 39.
        if name == 'start':
            z = 1.2 * f * f
        elif name == 'stop':
            z = start_z + .5 * (2 * f - f * f)
        else:
            z = 0.
        goals.append({'frame': frame, 'position_xz': [0., z], 'heading': 0.})
    body = {'request_id': 'crowd-' + name + '-' + uuid.uuid4().hex,
            'stage_kind': 'continuation' if history is not None else 'approach',
            'frames': 40, 'prompt': prompt, 'actor_ids': ['walker'],
            'actor_prompts': {'walker': prompt},
            'seed': {'idle': 823, 'start': 827, 'stop': 829}[name],
            'root_targets': {'walker': goals}}
    if history is None:
        body['initial_placements'] = {'walker': {'position_xz': [0., 0.], 'yaw': 0.}}
    else:
        body['history'] = {'native_features': history.tolist()}
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--token-file', required=True, type=Path)
    ap.add_argument('--source-walk', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--url', default='http://127.0.0.1:8769')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    client = RealtimeClient(args.url, args.token_file.read_text().strip(), timeout=5, job_timeout=100)
    with np.load(args.source_walk, allow_pickle=False) as d:
        history = d['native_features'][:, -40:].copy()
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'source_walk': str(args.source_walk),
              'source_walk_sha256': hashlib.sha256(args.source_walk.read_bytes()).hexdigest(),
              'health_before': client.health(), 'jobs': []}
    write(args.output / 'report.json', report)
    try:
        for name in ('idle', 'start', 'stop'):
            body = request(name, history=history if name == 'stop' else None)
            write(args.output / f'{name}-request.json', body)
            job = {'name': name, 'request_id': body['request_id'], 'status': 'submitted'}
            report['jobs'].append(job)
            write(args.output / 'report.json', report)
            began = time.monotonic()
            clip = client.wait(body)[0]
            job['generation_seconds'] = time.monotonic() - began
            archive = args.output / f'{name}-core.npz'
            np.savez_compressed(archive, positions=clip.positions,
                                rotations=clip.rotations,
                                native_features=clip.native_features,
                                metadata=np.array(json.dumps(clip.metadata)))
            job['source_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
            job['status'] = 'complete'
            write(args.output / 'report.json', report)
        report['status'] = 'complete'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        try:
            report['health_after'] = client.health()
        except Exception as exc:
            report['health_after_error'] = str(exc)
        write(args.output / 'report.json', report)
        print(json.dumps({'status': report['status'], 'jobs': len(report['jobs'])}), flush=True)


if __name__ == '__main__':
    main()
