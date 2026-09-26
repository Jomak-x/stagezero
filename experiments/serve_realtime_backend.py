#!/usr/bin/env python3
"""Launch a separate, private Core27 realtime worker on port 8769.

InterGen is a research-only optional mode. It uses the published checkpoint,
never downloads CLIP weights, and is disabled unless explicitly requested.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from realtime_backend import CoreStageAdapter, InterGenStageAdapter, JobManager, serve
from interaction_runtime import InteractionRuntime, MODEL_NAME


def load_pair(args):
    import torch
    from experiments.intergen_probe import inspect, prepare_model

    if not args.intergen_repo or not args.intergen_checkpoint:
        raise ValueError("Research mode requires --intergen-repo and --intergen-checkpoint")
    class PairArgs:
        repo = args.intergen_repo
        checkpoint = args.intergen_checkpoint
        clip_cache = args.clip_cache
        text_only_clip = True
        seed = 0

    report = inspect(PairArgs)
    if not report["ready"]:
        raise RuntimeError("InterGen research assets are incomplete: " + json.dumps(report))
    # The published InterGen inference path constructs normalizers lazily and
    # reads ./data/global_mean.npy during forward_test, so retain its repo cwd.
    pair_model, _ = prepare_model(PairArgs, report)
    from utils.utils import MotionNormalizer
    normalizer = MotionNormalizer()
    pair_model = pair_model.to("cuda:0").eval()
    return InterGenStageAdapter(pair_model, normalizer, source_revision=report["source_revision"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--token-file", type=Path, default=ROOT / ".runtime/api-token")
    parser.add_argument("--research-intergen", action="store_true")
    parser.add_argument("--intergen-repo", type=Path)
    parser.add_argument("--intergen-checkpoint", type=Path)
    parser.add_argument("--clip-cache", type=Path, default=Path("/workspace/stagezero/intergen-lab/CLIP"))
    parser.add_argument("--min-free-gib", type=float, default=20.0)
    parser.add_argument("--cuda-memory-fraction", type=float, default=.68)
    args = parser.parse_args()
    token = args.token_file.read_text().strip()
    if not token:
        raise RuntimeError("API token file is empty")
    if args.port == 8765:
        raise ValueError("Port 8765 belongs to the existing G1 worker")
    if not 0 < args.cuda_memory_fraction <= 1:
        raise ValueError("Invalid CUDA memory fraction")

    import torch
    from ardy.model import load_model

    if not torch.cuda.is_available():
        raise RuntimeError("Realtime Core requires CUDA")
    free, total = torch.cuda.mem_get_info()
    if free < args.min_free_gib * 2**30:
        raise RuntimeError(f"Only {free / 2**30:.2f} GiB free; require {args.min_free_gib:.2f} GiB")
    torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device=0)
    torch.set_num_threads(4)
    started = time.perf_counter()
    core_model = load_model(MODEL_NAME, device="cuda", text_encoder_mode="local")
    core = CoreStageAdapter(InteractionRuntime(core_model, device="cuda"))
    pair = load_pair(args) if args.research_intergen else None
    manager = JobManager(core, pair_adapter=pair, pair_enabled=args.research_intergen)
    print(json.dumps({"event": "realtime_ready", "pid": os.getpid(), "port": args.port,
                      "model": MODEL_NAME, "pair_research_enabled": args.research_intergen,
                      "load_seconds": time.perf_counter() - started,
                      "gpu": torch.cuda.get_device_name(),
                      "free_gib_before_load": free / 2**30,
                      "total_gib": total / 2**30}), flush=True)
    serve(manager, token, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
