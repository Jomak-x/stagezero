"""Real Pod soak test; does not use cached motion or substitute synthetic output."""
import json
from pathlib import Path
import time
import uuid
import numpy as np
from live_motion import Backend

prompts = ["A person waves with their right hand.", "A person slowly walks forward.",
           "A person turns left while walking.", "A person stops and stands still.",
           "A person does a squat.", "A person raises both arms overhead."]
b = Backend('.runtime/api-token')
records = []
history = None
last = None
start = time.perf_counter()
for i in range(24):
    if i % 6 == 0:
        history = None
        last = None
    before = time.perf_counter()
    r = b.generate(str(uuid.uuid4()), prompts[i % len(prompts)], history)
    meta = {**r['metadata'], 'roundtrip_seconds': time.perf_counter() - before}
    p = r['positions']
    meta['root_travel_m'] = float(np.linalg.norm(p[-1, 0] - p[0, 0]))
    meta['min_joint_y_m'] = float(p[:, :, 1].min())
    if last is not None:
        meta['boundary_mean_joint_step_m'] = float(np.linalg.norm(p[0] - last, axis=-1).mean())
    records.append(meta)
    history = r['motion'][-52:]
    last = p[-1]
    print(i + 1, meta['prompt'], round(meta['generation_seconds'], 3), flush=True)
    time.sleep(max(0, 4.16 - (time.perf_counter() - before)))
out = {'wall_seconds': time.perf_counter() - start, 'records': records}
Path('review/backend-soak.json').write_text(json.dumps(out, indent=2))
print('DONE', out['wall_seconds'], flush=True)
