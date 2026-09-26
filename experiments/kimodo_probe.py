#!/usr/bin/env python3
"""Bounded native Kimodo inference; run only in an isolated model environment.

The launcher must set TEXT_ENCODER_DEVICE=cpu, TEXT_ENCODER_MODE=local,
HF_HOME to a RAM-backed cache, and PYTHONPATH to the official source checkout.
No existing worker or model service is modified by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="kimodo-soma-rp-v1.1")
    parser.add_argument("--profile", choices=("initial4", "expressive8", "stunts8"), default="initial4")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--gpu-fraction", type=float, default=0.10)
    args = parser.parse_args()
    if not 0 < args.seconds <= 8 or not 25 <= args.steps <= 100:
        parser.error("Bounded trial requires ≤8 seconds and 25–100 steps")
    if not 0 < args.gpu_fraction <= .12:
        parser.error("GPU allocator fraction must be ≤0.12")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, device=0)
    torch.set_num_threads(8)

    from kimodo import load_model
    from kimodo.tools import seed_everything

    initial_cases = (
        ("kimodo_guard_dodge", 6105,
         "A boxer stands in a guarded stance, dodges to the left, blocks a punch with both arms, then steps back."),
        ("kimodo_grounded_dance", 6106,
         "A person dances in place, shifting weight from foot to foot, stepping sideways, turning the torso and moving both arms."),
    )
    expressive_cases = (
        ("kimodo_hiphop_8s", 6200,
         "A person performs an energetic hip hop dance with deep knee bends, alternating side steps, quick torso twists and broad rhythmic arm swings."),
        ("kimodo_martial_combo_8s", 6201,
         "A martial artist performs a sharp combination: a left jab, a right cross, a front kick, then resets into a low guarded stance and repeats the combination."),
    )
    stunt_cases = (
        ("kimodo_cartwheel_8s", 6301,
         "A trained acrobat takes two quick steps, performs a cartwheel, lands on both feet and stands balanced."),
        ("kimodo_spinning_roundhouse_8s", 6302,
         "A martial artist performs a fast spinning roundhouse kick, lands solidly, then punches twice and returns to guard."),
        ("kimodo_breakdance_sweep_8s", 6303,
         "A dancer performs energetic breakdance footwork, crouches into a low floor sweep, then rises with arms spread."),
        ("kimodo_shoulder_roll_8s", 6304,
         "A stunt performer ducks, performs a forward shoulder roll, gets back onto their feet and runs two steps."),
    )
    cases = {"initial4": initial_cases, "expressive8": expressive_cases,
             "stunts8": stunt_cases}[args.profile]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    model = load_model(args.model, device="cuda")
    load_seconds = time.perf_counter() - total_start
    if int(model.fps * args.seconds) > 240:
        raise RuntimeError("Unexpected duration/fps")
    rows = []
    for name, seed, prompt in cases:
        seed_everything(seed)
        torch.cuda.reset_peak_memory_stats(0)
        start = time.perf_counter()
        result = model([prompt], [int(model.fps * args.seconds)], constraint_lst=[],
                       num_denoising_steps=args.steps, num_samples=1,
                       multi_prompt=False, post_processing=True, return_numpy=True)
        torch.cuda.synchronize(0)
        elapsed = time.perf_counter() - start
        positions = np.asarray(result["posed_joints"][0], dtype=np.float32)
        rotations = np.asarray(result["global_rot_mats"][0], dtype=np.float32)
        if positions.ndim != 3 or positions.shape[-1] != 3 or not np.isfinite(positions).all():
            raise RuntimeError("Kimodo returned invalid joint positions")
        if rotations.shape != positions.shape[:2] + (3, 3) or not np.isfinite(rotations).all():
            raise RuntimeError("Kimodo returned invalid joint rotations")
        meta = {"source": "unmodified native Kimodo output", "model": args.model,
                "fps": model.fps, "seed": seed, "prompt": prompt,
                "seconds": args.seconds, "steps": args.steps,
                "post_processing": True, "generation_seconds": elapsed,
                "model_load_seconds": load_seconds,
                "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "gpu_fraction_cap": args.gpu_fraction}
        path = args.output_dir / f"{name}.npz"
        np.savez_compressed(path, positions=positions, rotations=rotations,
                            metadata=np.array(json.dumps(meta, allow_nan=False)))
        row = {"file": path.name, "sha256": sha256_file(path),
               "frames": int(len(positions)), "joints": int(positions.shape[1]),
               "generation_seconds": elapsed, "peak_gpu_allocated_bytes": meta["peak_gpu_allocated_bytes"]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {"model": args.model, "profile": args.profile,
              "model_load_seconds": load_seconds,
              "cases": rows, "source": "official nv-tlabs/kimodo checkout"}
    (args.output_dir / "kimodo-manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
