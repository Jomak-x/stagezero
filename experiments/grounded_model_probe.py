#!/usr/bin/env python3
"""Record bounded, unedited grounded motion from the existing warm service.

InterGen is noncommercial research output. Its service path retargets native
SMPL22 pair joints into Core27; this probe does not alter the published poses.
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
    ("pair_shove_dodge", "paired", 120, 6101,
     "Two people face each other. One shoves the other's shoulder, the other sidesteps, blocks with an arm, and steps back."),
    ("pair_help_up", "paired", 120, 6102,
     "One person is crouched on the floor. A second person reaches down, takes their hand, pulls them to standing, and they steady each other."),
    ("pair_embrace", "paired", 120, 6103,
     "Two people walk toward each other, stop, and hug warmly with both arms, then step back."),
    ("pair_dance", "paired", 120, 6104,
     "Two people dance together face to face, holding hands, stepping side to side and turning together."),
    ("core_guard_dodge", "approach", 80, 6105,
     "A boxer stands in a guarded stance, dodges to the left, blocks a punch with both arms, then steps back."),
    ("core_grounded_dance", "approach", 80, 6106,
     "A person dances in place, shifting weight from foot to foot, stepping sideways, turning the torso and moving both arms."),
)


def record(client: RealtimeClient, output: Path, selection: set[str]) -> dict:
    health = client.health()
    if (health.get("model") != "ARDY-Core-RP-20FPS-Horizon40"
            or health.get("horizon") != 40 or health.get("fps") != 20
            or health.get("pair_research_enabled") is not True):
        raise RuntimeError("Expected the already warm Core/InterGen research worker")
    output.mkdir(parents=True, exist_ok=True)
    report = {"health": {key: health.get(key) for key in
                         ("model", "horizon", "fps", "joints", "pair_research_enabled")},
              "cases": {}, "license_note": "InterGen: CC BY-NC-SA 4.0; research preview only"}
    for name, kind, frames, seed, prompt in CASES:
        if selection and name not in selection:
            continue
        request_id = f"grounded-{name}-{uuid.uuid4().hex[:12]}"
        ids = ["person_a", "person_b"] if kind == "paired" else ["person_a"]
        body = {"request_id": request_id, "stage_kind": kind, "frames": frames,
                "prompt": prompt, "actor_ids": ids, "seed": seed}
        if kind == "paired":
            body.update(pair_sequence_id=request_id, source_start_frame=0,
                        source_total_frames=frames)
        start = time.perf_counter()
        chunks = client.wait(body)
        elapsed = time.perf_counter() - start
        positions = np.concatenate([chunk.positions for chunk in chunks], axis=1)
        rotations = np.concatenate([chunk.rotations for chunk in chunks], axis=1)
        native = (np.concatenate([chunk.native_features for chunk in chunks], axis=1)
                  if kind != "paired" else None)
        if positions.shape != (len(ids), frames, 27, 3) or rotations.shape != (len(ids), frames, 27, 3, 3):
            raise RuntimeError(f"Unexpected pose shape in {name}")
        if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
            raise RuntimeError(f"Nonfinite pose in {name}")
        if kind != "paired" and (native is None or native.shape != (1, frames, 330)
                                  or not np.isfinite(native).all()):
            raise RuntimeError(f"Missing native Core features in {name}")
        result = client.status(request_id)
        metadata = {"name": name, "request_id": request_id, "stage_kind": kind,
                    "model": "InterGen (SMPL22 retargeted to Core27)" if kind == "paired"
                             else health["model"],
                    "source": "Unedited published service poses",
                    "fps": 20, "frames": frames, "actor_ids": ids, "seed": seed,
                    "prompt": prompt, "service_result": result,
                    "chunk_metadata": [dict(c.metadata) for c in chunks],
                    "client_round_trip_seconds": elapsed}
        path = output / f"{name}.npz"
        arrays = {"positions": positions, "rotations": rotations,
                  "metadata": np.array(json.dumps(metadata, allow_nan=False))}
        if native is not None:
            arrays["native_features"] = native
        np.savez_compressed(path, **arrays)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        row = {"file": path.name, "sha256": digest, "prompt": prompt,
               "model": metadata["model"], "seed": seed, "frames": frames,
               "actors": len(ids), "seconds": round(elapsed, 3),
               "source": metadata["source"]}
        report["cases"][name] = row
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"case": name, **row}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8769")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=[c[0] for c in CASES])
    args = parser.parse_args()
    client = RealtimeClient(args.url, args.token_file.read_text().strip(),
                            timeout=15, job_timeout=180)
    record(client, args.output_dir, set(args.case or []))


if __name__ == "__main__":
    main()
