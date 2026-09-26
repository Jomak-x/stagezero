#!/usr/bin/env python3
"""Inspect or run one official InterGen pair sample without its video renderer.

The caller supplies an unmodified InterGen source checkout and the published
checkpoint.  This script never downloads weights or creates synthetic motion.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


CLIP_FILE = "ViT-L-14-336px.pt"
CLIP_BYTES = 934_088_680
CHECKPOINT_BYTES = 2_701_617_525
SOURCE_FILES = (
    "configs/model.yaml",
    "models/intergen.py",
    "data/global_mean.npy",
    "data/global_std.npy",
)


def source_revision(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def inspect(args: argparse.Namespace) -> dict:
    repo = args.repo.resolve()
    checkpoint = args.checkpoint.resolve()
    clip_file = args.clip_cache.expanduser().resolve() / CLIP_FILE
    files = {name: (repo / name).is_file() for name in SOURCE_FILES}
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in ("torch", "torchvision", "clip", "yacs", "numpy", "scipy", "matplotlib", "PIL")
    }
    checkpoint_size = checkpoint.stat().st_size if checkpoint.is_file() else None
    clip_size = clip_file.stat().st_size if clip_file.is_file() else None
    report = {
        "source": str(repo),
        "source_revision": source_revision(repo) if repo.is_dir() else None,
        "source_files": files,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint_size,
        "checkpoint_expected_bytes": CHECKPOINT_BYTES,
        "clip_file": str(clip_file),
        "clip_bytes": clip_size,
        "clip_expected_bytes": CLIP_BYTES,
        "python": sys.executable,
        "modules": modules,
    }
    report["ready"] = (
        all(files.values())
        and checkpoint_size == CHECKPOINT_BYTES
        and (args.text_only_clip or clip_size == CLIP_BYTES)
        and all(modules.values())
    )
    report["text_only_clip"] = args.text_only_clip
    return report


def prepare_model(args: argparse.Namespace, report: dict):
    if not report["ready"]:
        raise SystemExit("InterGen assets or Python modules are missing; inspect the check report first.")

    # The authors' MotionNormalizer reads ./data relative to the current directory.
    os.chdir(args.repo.resolve())
    sys.path.insert(0, str(args.repo.resolve()))
    os.environ["MPLBACKEND"] = "Agg"

    import torch
    import clip
    from configs import get_config
    from models import InterGen

    torch.manual_seed(args.seed)
    cfg = get_config("configs/model.yaml")
    # The official demo calls clip.load(), which downloads a 934 MB CLIP file.
    # Lightning's InterGen checkpoint can supply those text weights directly.
    # This path is accepted only when *every* model key loads from the checkpoint.
    checkpoint = torch.load(report["checkpoint"], map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    state = {key.removeprefix("model."): value for key, value in state.items()}
    if args.text_only_clip:
        from clip.model import LayerNorm, Transformer
        from torch import nn

        token_key = "token_embedding.weight"
        position_key = "positional_embedding"
        if token_key not in state or position_key not in state:
            raise SystemExit("Checkpoint lacks registered CLIP text weights; full CLIP weights are required.")
        vocab, width = state[token_key].shape
        context = state[position_key].shape[0]
        layer_indices = {
            int(key.split(".")[2])
            for key in state
            if key.startswith("clip_transformer.resblocks.")
        }
        if not layer_indices or layer_indices != set(range(len(layer_indices))):
            raise SystemExit("Checkpoint CLIP transformer layers are incomplete.")

        class ClipTextShell:
            def __init__(self):
                self.token_embedding = nn.Embedding(vocab, width)
                mask = torch.full((context, context), float("-inf")).triu_(1)
                self.transformer = Transformer(width, len(layer_indices), width // 64, mask)
                self.positional_embedding = nn.Parameter(torch.empty(context, width))
                self.ln_final = LayerNorm(width)
                self.dtype = torch.float32

        original_load = clip.load
        clip.load = lambda *_args, **_kwargs: (ClipTextShell(), None)
        try:
            model = InterGen(cfg)
        finally:
            clip.load = original_load
    else:
        model = InterGen(cfg)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise SystemExit(
            f"Checkpoint did not populate {len(incompatible.missing_keys)} model parameters; "
            f"first missing key: {incompatible.missing_keys[0]}"
        )
    if incompatible.unexpected_keys:
        raise SystemExit(
            f"Checkpoint contains {len(incompatible.unexpected_keys)} unexpected model parameters; "
            f"first extra key: {incompatible.unexpected_keys[0]}"
        )
    return model.eval(), len(state)


def sample(args: argparse.Namespace, report: dict) -> None:
    if not args.prompt.strip():
        raise SystemExit("A nonempty interaction prompt is required.")
    if not 1 <= args.frames <= 210:
        raise SystemExit("Frames must be between 1 and the official 210-frame demo length.")
    output_path = args.output.resolve()
    if output_path.exists() and not args.force:
        raise SystemExit(f"Output already exists: {output_path} (use --force to overwrite)")

    model, state_key_count = prepare_model(args, report)
    import numpy as np
    import torch
    from scipy.ndimage import gaussian_filter1d
    from utils.utils import MotionNormalizer

    if not torch.cuda.is_available():
        raise SystemExit("This probe requires one CUDA GPU, as in the official demo.")
    if not 0 < args.cuda_memory_fraction <= 1:
        raise SystemExit("CUDA memory fraction must be in (0, 1].")
    # Set the allocator ceiling before creating any CUDA tensors or moving the model.
    torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device=0)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    model = model.to(device)
    normalizer = MotionNormalizer()

    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    with torch.inference_mode():
        batch = {
            "motion_lens": torch.tensor([args.frames], dtype=torch.long, device=device),
            "text": [args.prompt],
        }
        output = model.forward_test(batch)["output"]
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - start
    if tuple(output.shape) != (1, args.frames, 524):
        raise SystemExit(f"Unexpected model output shape: {tuple(output.shape)}")

    features = normalizer.backward(output[0].reshape(args.frames, 2, 262).cpu().numpy())
    joints = features[..., :66].reshape(args.frames, 2, 22, 3)
    # Match the one-frame Gaussian smoothing in tools/infer.py for displayed joints.
    smoothed_joints = gaussian_filter1d(joints, sigma=1, axis=0, mode="nearest")
    if not np.isfinite(features).all():
        raise SystemExit("Generated features contain non-finite values.")

    metadata = {
        "model": "InterGen",
        "source_revision": report["source_revision"],
        "checkpoint": report["checkpoint"],
        "clip_file": None if args.text_only_clip else report["clip_file"],
        "text_only_clip_from_checkpoint": args.text_only_clip,
        "checkpoint_state_keys": state_key_count,
        "prompt": args.prompt,
        "seed": args.seed,
        "frames": args.frames,
        "fps": 30,
        "feature_shape": list(features.shape),
        "joint_shape": list(joints.shape),
        "generation_seconds": elapsed,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "missing_checkpoint_keys": 0,
        "unexpected_checkpoint_keys": 0,
        "license": "CC BY-NC-SA 4.0; noncommercial research preview only",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features.astype(np.float32),
        joints=joints.astype(np.float32),
        smoothed_joints=smoothed_joints.astype(np.float32),
        metadata=json.dumps(metadata),
    )
    print(json.dumps({"output": str(output_path), **metadata}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "validate", "sample"))
    parser.add_argument("--repo", type=Path, required=True, help="Official InterGen checkout")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Published intergen.ckpt")
    parser.add_argument(
        "--clip-cache", type=Path, default=Path("~/.cache/clip"), help="OpenAI CLIP cache directory"
    )
    parser.add_argument(
        "--text-only-clip", action="store_true",
        help="Build CLIP text modules from checkpoint keys without downloading CLIP weights",
    )
    parser.add_argument("--prompt", default="Two people embrace each other.")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--frames", type=int, default=210)
    parser.add_argument("--cuda-memory-fraction", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("results/intergen_pair.npz"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = inspect(args)
    if args.command == "check":
        print(json.dumps(report, indent=2))
        return
    if args.command == "validate":
        _, state_key_count = prepare_model(args, report)
        print(json.dumps({
            "checkpoint_state_keys": state_key_count,
            "missing_model_keys": 0,
            "unexpected_model_keys": 0,
            "text_only_clip_from_checkpoint": args.text_only_clip,
            "source_revision": report["source_revision"],
        }, indent=2))
        return
    sample(args, report)


if __name__ == "__main__":
    main()
