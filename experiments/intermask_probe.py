#!/usr/bin/env python3
"""Bounded isolated InterMask inference preserving raw native paired features.

Run from a separate lab with the official unmodified checkout and checkpoints.
No download/install logic, Core conversion, motion smoothing, foot IK or scene
placement is performed here. Default is CPU model/accounting inspection only;
--run explicitly enables at most three seeds for each of two fixed prompts.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

GIB = 1024 ** 3
PROMPTS = {"handshake": "Two people shake hands and step apart.", "embrace": "Two people embrace each other."}
INTERGEN_SHA256 = "bab341123c27af9d8c3ee7ebc43c2d341bfd0f4103425d06f8af6c717a086f8d"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gpu_status():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"], text=True)
    rows = [list(map(int, line.split(","))) for line in raw.strip().splitlines()]
    if len(rows) != 1:
        raise RuntimeError("Probe requires exactly one visible GPU")
    return {"total_mib": rows[0][0], "free_mib": rows[0][1]}


def prepare(args):
    import numpy as np
    import torch
    import clip
    from clip.model import LayerNorm, Transformer
    from torch import nn
    from models.vq.model import RVQVAE
    from models.mask_transformer.transformer import MaskTransformer
    from utils.get_opt import get_opt

    root = args.repo / "checkpoints/interhuman"
    trans_file = root / "trans_default/model/best_fid.tar"
    vq_file = root / "vq_default/model/best_fid.tar"
    paths = {"transformer": trans_file, "vq": vq_file, "clip_text_source_intergen": args.clip_checkpoint}
    hashes = {name: sha256(path) for name, path in paths.items()}
    if hashes["clip_text_source_intergen"] != INTERGEN_SHA256:
        raise RuntimeError("Existing InterGen checkpoint does not match the previously verified official SHA256")
    # The previously verified Lightning artifact contains NumPy metadata that
    # is incompatible with weights_only=True. Its exact SHA is pinned above.
    pretrained = torch.load(args.clip_checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    pretrained = {k.removeprefix("model."): v for k, v in pretrained.items()}
    text_state = {}
    for key, value in pretrained.items():
        if key.startswith("clip_transformer."):
            text_state[key] = value
        elif key.startswith("token_embedding.") or key.startswith("ln_final.") or key == "positional_embedding":
            text_state["clip_" + key] = value
    del pretrained
    vocab, width = text_state["clip_token_embedding.weight"].shape
    context = text_state["clip_positional_embedding"].shape[0]
    layers = {int(k.split(".")[2]) for k in text_state if k.startswith("clip_transformer.resblocks.")}
    if width != 768 or layers != set(range(12)):
        raise RuntimeError("CLIP text checkpoint does not match official ViT-L/14@336px text architecture")

    class TextOnlyClip:
        def __init__(self):
            self.token_embedding = nn.Embedding(vocab, width)
            self.transformer = Transformer(width, len(layers), width // 64, torch.full((context, context), float("-inf")).triu_(1))
            self.positional_embedding = nn.Parameter(torch.empty(context, width))
            self.ln_final = LayerNorm(width)
            self.dtype = torch.float32

    opt = get_opt(str(root / "trans_default/opt.txt"), "cpu")
    vopt = get_opt(str(root / "vq_default/opt.txt"), "cpu")
    opt.num_tokens = vopt.nb_code
    opt.code_dim = vopt.code_dim
    original_load = clip.load
    clip.load = lambda *_a, **_kw: (TextOnlyClip(), None)
    try:
        trans = MaskTransformer(code_dim=opt.code_dim, cond_mode="text", latent_dim=opt.latent_dim,
            ff_size=opt.ff_size, num_layers=opt.n_layers, num_heads=opt.n_heads, dropout=opt.dropout,
            clip_dim=768, cond_drop_prob=opt.cond_drop_prob, clip_version="ViT-L/14@336px", opt=opt)
    finally:
        clip.load = original_load
    state = torch.load(trans_file, map_location="cpu", weights_only=True)["t2m_transformer"]
    overlap = set(state) & set(text_state)
    if overlap:
        raise RuntimeError("Unexpected checkpoint overlap: do not overwrite model text weights")
    state.update(text_state)
    trans.load_state_dict(state, strict=True)
    trans_keys = len(state)
    del state, text_state
    vq = RVQVAE(vopt, 12, vopt.nb_code, vopt.code_dim, vopt.code_dim, vopt.down_t,
                vopt.stride_t, vopt.width, vopt.depth, vopt.dilation_growth_rate, activation=vopt.vq_act, norm=vopt.vq_norm)
    state = torch.load(vq_file, map_location="cpu", weights_only=True)["vq_model"]
    incompatible = vq.load_state_dict(state, strict=False)
    # The official released checkpoint predates unused decoder conv/resnet keys.
    if incompatible.unexpected_keys or any(not k.startswith(("decoder.conv", "decoder.resnets")) for k in incompatible.missing_keys):
        raise RuntimeError(f"VQ checkpoint mismatch: {incompatible}")
    vq_keys = len(state)
    del state
    gc.collect()
    report = {
        "model": "InterMask", "research_only": True,
        "source_revision": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint_sha256": hashes, "checkpoint_bytes": {k: v.stat().st_size for k, v in paths.items()},
        "torch": torch.__version__, "python": sys.version,
        "transformer_loaded_keys": trans_keys, "vq_loaded_keys": vq_keys,
        "vq_missing_unused_keys": incompatible.missing_keys,
        "text_weights": "Unmodified official ViT-L/14@336px text weights reused from hash-verified existing InterGen checkpoint; no visual encoder loaded",
        "native_postprocessing": "Denormalization only; no Gaussian smoothing, foot IK, root offsets or Core retargeting",
        "license_provenance": "InterMask repository MIT; generated research samples derive from InterHuman pretrained weights; InterHuman noncommercial research terms remain relevant",
    }
    return trans.eval(), vq.eval(), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[37, 42, 43])
    parser.add_argument("--frames", type=int, default=208)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not 1 <= len(args.seeds) <= 3 or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s <= 2**32-1 for s in args.seeds):
        parser.error("Provide one to three distinct uint32 seeds")
    if not 40 <= args.frames <= 208 or args.frames % 4:
        parser.error("Frames must be a multiple of4 between40 and208")
    args.repo = args.repo.resolve(); args.clip_checkpoint = args.clip_checkpoint.resolve(); args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(args.repo); sys.path.insert(0, str(args.repo))
    import numpy as np
    import torch
    torch.set_num_threads(2)
    trans, vq, report = prepare(args)
    report["gpu_before"] = gpu_status()
    (args.output / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not args.run:
        return
    if report["gpu_before"]["free_mib"] < 8192:
        raise RuntimeError("Less than8GiB GPU memory free; refusing run")
    if any((args.output / f"{name}_seed{seed}.npz").exists() for name in PROMPTS for seed in args.seeds):
        raise RuntimeError("Refusing to overwrite existing native samples")
    stop = threading.Event()
    guard = {"minimum_free_mib": report["gpu_before"]["free_mib"], "maximum_process_mib": 0}
    started = time.monotonic()

    def watchdog():
        while not stop.wait(.5):
            try:
                status = gpu_status(); guard["minimum_free_mib"] = min(guard["minimum_free_mib"], status["free_mib"])
                raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True)
                own = [int(row.split(",")[1]) for row in raw.strip().splitlines() if row.split(",")[0].strip() == str(os.getpid())]
                used = max(own, default=0); guard["maximum_process_mib"] = max(guard["maximum_process_mib"], used)
                if status["free_mib"] < 8192 or used > 6144 or time.monotonic() - started > 180:
                    raise RuntimeError("GPU free/process memory or180s runtime bound exceeded")
            except Exception as exc:
                (args.output / "guard_stop.json").write_text(json.dumps({**guard, "error": str(exc)}))
                print(f"Own probe stopped: {exc}", file=sys.stderr, flush=True)
                os._exit(72)

    threading.Thread(target=watchdog, daemon=True).start()
    torch.cuda.set_per_process_memory_fraction(6 * GIB / torch.cuda.get_device_properties(0).total_memory, 0)
    trans = trans.cuda(); vq = vq.cuda()
    from data.utils import MotionNormalizer
    normalizer = MotionNormalizer()
    samples = []
    try:
        for name, prompt in PROMPTS.items():
            for seed in args.seeds:
                if gpu_status()["free_mib"] < 8192:
                    raise RuntimeError("Less than8GiB free before sample")
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                torch.cuda.reset_peak_memory_stats(); tick = time.monotonic()
                with torch.inference_mode():
                    ids = trans.generate([prompt], torch.tensor([args.frames // 4], device="cuda"), 20, 2, temperature=1, topk_filter_thres=.9)
                    a, b = ids[:, :ids.shape[1] // 2], ids[:, ids.shape[1] // 2:]
                    ma = vq.forward_decoder(a.unsqueeze(-1)); mb = vq.forward_decoder(b.unsqueeze(-1))
                    native = torch.cat([ma, mb], dim=-1).reshape(1, -1, 2, 262)
                    features = normalizer.backward(native.cpu().numpy())[0]
                torch.cuda.synchronize(); elapsed = time.monotonic() - tick
                if features.shape != (args.frames, 2, 262) or not np.isfinite(features).all():
                    raise RuntimeError(f"Unexpected or nonfinite native output {features.shape}")
                metadata = {**report, "prompt": prompt, "seed": seed, "frames": args.frames, "fps": 30,
                    "sampling": {"steps": 20, "guidance": 2, "temperature": 1, "topk_filter_threshold": .9},
                    "generation_seconds": elapsed, "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(), "allocator_limit_bytes": 6 * GIB,
                    "foot_channels": "VQ postprocess emits normalized zeros; these are not predicted contact labels"}
                np.savez_compressed(args.output / f"{name}_seed{seed}.npz", features=features.astype(np.float32),
                    joints=features[..., :66].reshape(args.frames, 2, 22, 3).astype(np.float32), metadata=np.array(json.dumps(metadata)))
                samples.append({"name": name, "seed": seed, "seconds": elapsed, "peak_cuda_bytes": metadata["peak_cuda_bytes"]})
                print(json.dumps(samples[-1]), flush=True)
    finally:
        stop.set()
        (args.output / "run_summary.json").write_text(json.dumps({"samples": samples, "guard": guard}, indent=2) + "\n")


if __name__ == "__main__":
    main()
