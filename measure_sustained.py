"""Ten-minute real inference soak. No cached or simulated results."""
import json
from pathlib import Path
import time
import uuid
import numpy as np
from live_motion import Backend

backend = Backend('.runtime/api-token')
prompts = ['A person waves with their right hand.', 'A person slowly walks forward.',
           'A person turns left while walking.', 'A person stops and stands still.',
           'A person does a squat.', 'A person raises both arms overhead.']
records, failures = [], []
history = None
started = time.perf_counter()
for i in range(144):
    if i % 6 == 0:
        history = None
    before = time.perf_counter()
    try:
        result = backend.generate(str(uuid.uuid4()), prompts[i % 6], history)
        records.append({**result['metadata'], 'roundtrip_seconds': time.perf_counter() - before})
        history = result['motion'][-52:]
    except Exception as exc:
        failures.append({'iteration': i, 'error': str(exc)})
        history = None
    Path('review/directing-soak.json').write_text(json.dumps(dict(
        wall_seconds=time.perf_counter() - started, completed=i + 1,
        records=records, failures=failures), indent=2))
    if i % 12 == 0:
        print('Completed', i + 1, 'failures', len(failures), flush=True)
    time.sleep(max(0, 4.2 - (time.perf_counter() - before)))
print('FINISHED', time.perf_counter() - started, 'failures', len(failures), flush=True)
