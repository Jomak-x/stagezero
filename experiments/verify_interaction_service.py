"""Exercise the isolated real Core service's auth, failure recovery and save/load."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import time
import numpy as np
import requests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8768')
    p.add_argument('--token-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    headers = {'Authorization': 'Bearer ' + a.token_file.read_text().strip()}
    checks = {}
    checks['unauthorized_rejected'] = requests.get(a.url + '/health', timeout=10).status_code == 401
    invalid = {'request_id': 'invalid-recovery-probe', 'frames': 41,
               'actors': [{'id': 'actor', 'prompt': 'A person walks forward.', 'seed': 701}]}
    checks['malformed_rejected'] = requests.post(a.url + '/generate', json=invalid, headers=headers, timeout=20).status_code == 400
    checks['unknown_cancellation_false'] = requests.post(a.url + '/cancel', json={'request_id': 'never-submitted'}, headers=headers, timeout=10).json() == {'cancelled': False}
    body = {**invalid, 'request_id': 'valid-recovery-probe', 'frames': 40}
    started = time.perf_counter()
    response = requests.post(a.url + '/generate', json=body, headers=headers, timeout=120)
    response.raise_for_status()
    a.output.mkdir(parents=True, exist_ok=True)
    archive_path = a.output / 'recovery-motion.npz'
    archive_path.write_bytes(response.content)
    with np.load(io.BytesIO(response.content), allow_pickle=False) as before, np.load(archive_path, allow_pickle=False) as after:
        checks['save_load_exact'] = before.files == after.files and all(np.array_equal(before[key], after[key]) for key in before.files)
        meta = json.loads(str(after['metadata'].item()))
        checks['successful_generation_after_failure'] = after['actor_0_positions'].shape == (40, 27, 3) and np.isfinite(after['actor_0_positions']).all().item()
        checks['baseline_conditions_false'] = meta['native_conditions'] is False
        checks['unmodified_model_output'] = meta['post_generation_pose_edits'] is False
    report = {'checks': checks, 'passed': all(checks.values()), 'round_trip_seconds': time.perf_counter()-started, 'runtime': meta}
    (a.output / 'recovery-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
