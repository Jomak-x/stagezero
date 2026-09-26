#!/usr/bin/env python3
"""Save three unmodified native ARDY Core clips from the existing warm service.

No model is loaded or provisioned here. The service runs one 40-frame horizon at
a time and returns Core27 positions, global rotations, and native features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realtime_client import RealtimeClient


CASES = (
    ("swing", 1201, "A person swings forward through the air while holding a rope high overhead with the right hand, legs bent and body leaning back."),
    ("hang_reach", 1202, "A person hangs by the right hand from an overhead rope and reaches forward with the left hand, knees tucked up."),
    ("land_carry", 1203, "A person lands from a jump, bends the knees to absorb the impact, stands up and carries another person in both arms."),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8769")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    token = args.token_file.read_text().strip()
    client = RealtimeClient(args.url, token, timeout=15, job_timeout=180)
    health = client.health()
    if health.get("model") != "ARDY-Core-RP-20FPS-Horizon40" or health.get("horizon") != 40:
        raise RuntimeError("Unexpected model service; refusing to record clips")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"service": {k: health[k] for k in ("model", "fps", "joints", "horizon")}, "cases": {}}
    for name, seed, prompt in CASES:
        request_id = f"swing-probe-{name}-{uuid.uuid4().hex[:12]}"
        body = {"request_id": request_id, "stage_kind": "approach", "frames": 80,
                "prompt": prompt, "actor_ids": ["actor"], "seed": seed}
        started = time.perf_counter()
        chunks = client.wait(body)
        elapsed = time.perf_counter() - started
        positions = np.concatenate([c.positions[0] for c in chunks], axis=0)
        rotations = np.concatenate([c.rotations[0] for c in chunks], axis=0)
        native = np.concatenate([c.native_features[0] for c in chunks], axis=0)
        if positions.shape != (80, 27, 3) or rotations.shape != (80, 27, 3, 3) or native.shape != (80, 330):
            raise RuntimeError("Unexpected Core27 output shapes")
        if not np.isfinite(positions).all() or not np.isfinite(rotations).all() or not np.isfinite(native).all():
            raise RuntimeError("Nonfinite Core27 output")
        metadata = {"source": "unmodified ARDY Core model output", "model": health["model"],
                    "fps": 20, "horizon": 40, "frames": 80, "seed": seed,
                    "prompt": prompt, "request_id": request_id, "chunk_metadata": [dict(c.metadata) for c in chunks],
                    "client_round_trip_seconds": elapsed}
        path = args.output_dir / f"{name}.npz"
        np.savez_compressed(path, positions=positions, rotations=rotations,
                            native_features=native, metadata=json.dumps(metadata, allow_nan=False))
        report["cases"][name] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "seed": seed, "prompt": prompt, "frames": len(positions),
                                "client_round_trip_seconds": round(elapsed, 4),
                                "root_horizontal_range_m": round(float(np.ptp(positions[:, 0, [0, 2]], axis=0).max()), 4),
                                "root_vertical_range_m": round(float(np.ptp(positions[:, 0, 1])), 4)}
        print(json.dumps({"case": name, **report["cases"][name]}), flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
