"""Generate a textured character GLB with Microsoft's TRELLIS image pipeline.

Run this in the isolated character environment on the GPU pod. The input should
show one full-body person with a clear silhouette; an RGBA cutout is preferred.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import time


MAX_GLB_BYTES = 40 * 1024 * 1024
GIB = 1024 ** 3
# Only the current model's weights are resident. Keep an additional GiB each
# for CUDA/native libraries and free headroom for the other GPU workers.
GPU_ALLOCATOR_BUDGET_BYTES = 12 * GIB
MIN_FREE_GPU_BYTES = GPU_ALLOCATOR_BUDGET_BYTES + 2 * GIB


def _gpu_memory(torch: object) -> dict:
    free_bytes, _ = torch.cuda.mem_get_info()
    return {
        "free_gib": round(free_bytes / GIB, 3),
        "allocated_gib": round(torch.cuda.memory_allocated() / GIB, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / GIB, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 3),
    }


def _configure_gpu_budget(torch: object, *, wait_seconds: float = 120) -> None:
    deadline = time.monotonic() + wait_seconds
    while True:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if free_bytes >= MIN_FREE_GPU_BYTES:
            break
        _stage("waiting_gpu_memory", free_gib=round(free_bytes / GIB, 2),
               required_gib=MIN_FREE_GPU_BYTES / GIB)
        if time.monotonic() >= deadline:
            raise RuntimeError("Waiting for GPU capacity: 14 GiB free is required; other workers were left running")
        time.sleep(min(5, max(0, deadline - time.monotonic())))
    # This bounds PyTorch's allocator, including its cache. Native CUDA
    # allocations are separate, hence the additional admission headroom above.
    torch.cuda.set_per_process_memory_fraction(GPU_ALLOCATOR_BUDGET_BYTES / total_bytes)
    torch.cuda.reset_peak_memory_stats()
    _stage("gpu_available", allocator_budget_gib=GPU_ALLOCATOR_BUDGET_BYTES / GIB,
           **_gpu_memory(torch))


class _StagedModelOffload:
    """Move root models only at phase boundaries; preserve all sampling settings."""

    def __init__(self, pipeline: object, torch: object):
        self.pipeline = pipeline
        self.torch = torch
        self.active = None
        self.handles = []
        self.original_class = type(pipeline)
        # TRELLIS infers its device from the first model. That assumption no
        # longer holds with CPU offload, while latents must stay on CUDA.
        pipeline.__class__ = type("StagedCudaPipeline", (self.original_class,), {
            "device": property(lambda _pipeline: torch.device("cuda")),
        })
        for name, model in pipeline.models.items():
            model.cpu()
            self.handles.append(model.register_forward_pre_hook(
                lambda module, _args, name=name: self.activate(name, module)
            ))

    def activate(self, name: str, model: object) -> None:
        if self.active is model:
            return
        if self.active is not None:
            self.active.cpu()
        self.torch.cuda.empty_cache()
        _stage("generating_phase", model=name, **_gpu_memory(self.torch))
        model.to(self.torch.device("cuda"))
        self.active = model

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.active is not None:
            self.active.cpu()
            self.active = None
        self.pipeline.__class__ = self.original_class
        self.torch.cuda.empty_cache()


def _stage(name: str, **details: object) -> None:
    print(json.dumps({"stage": name, **details}, separators=(",", ":")), flush=True)


def _verify_glb(path: Path) -> None:
    """Reject oversized, malformed, or externally referenced GLBs before publication."""
    size = path.stat().st_size
    if size < 1024 or size > MAX_GLB_BYTES:
        raise RuntimeError("TRELLIS GLB size is outside the supported range")
    data = path.read_bytes()
    if len(data) != size or size % 4 or data[:4] != b"glTF":
        raise RuntimeError("TRELLIS GLB has invalid framing")
    magic, version, declared = struct.unpack_from("<4sII", data)
    if magic != b"glTF" or version != 2 or declared != size:
        raise RuntimeError("TRELLIS GLB has an invalid header")
    offset = 12
    chunks = []
    while offset < size:
        if offset + 8 > size:
            raise RuntimeError("TRELLIS GLB has a truncated chunk")
        length, kind = struct.unpack_from("<I4s", data, offset)
        offset += 8
        if length % 4 or offset + length > size:
            raise RuntimeError("TRELLIS GLB has an invalid chunk")
        chunks.append((kind, data[offset:offset + length]))
        offset += length
    if not chunks or chunks[0][0] != b"JSON" or len(chunks) > 2 or (
        len(chunks) == 2 and chunks[1][0] != b"BIN\0"
    ):
        raise RuntimeError("TRELLIS GLB has an invalid chunk layout")
    document = json.loads(chunks[0][1].rstrip(b" "))
    if not isinstance(document, dict) or not isinstance(document.get("asset"), dict) or (
        document["asset"].get("version") != "2.0"
    ):
        raise RuntimeError("TRELLIS GLB is missing glTF 2.0 metadata")
    for collection in ("buffers", "images"):
        entries = document.get(collection, [])
        if not isinstance(entries, list) or any(
            not isinstance(item, dict) or (
                "uri" in item and not (isinstance(item["uri"], str) and item["uri"].startswith("data:"))
            ) for item in entries
        ):
            raise RuntimeError("TRELLIS GLB references an external resource")


def generate(input_path: Path, output_path: Path, *, seed: int) -> None:
    # These must be selected before TRELLIS imports its attention modules.
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # The pod's /usr/local/cuda points to CUDA 12.8.93 (verified with nvcc).
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

    import torch

    _stage("checking_gpu")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for TRELLIS")
    _configure_gpu_budget(torch)

    from PIL import Image
    import rembg
    from character_texture_detail import enhance_front_texture
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    _stage("loading_pipeline", model="microsoft/TRELLIS-image-large")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    # ONNX Runtime otherwise uses CUDA independently of PyTorch's memory cap.
    pipeline.rembg_session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
    offload = _StagedModelOffload(pipeline, torch)

    _stage("generating")
    with Image.open(input_path) as source:
        reference = source.convert("RGBA")
        try:
            outputs = pipeline.run(reference, seed=seed, formats=["mesh", "gaussian"])
        finally:
            offload.close()
        # Use a real foreground mask from the same reference that conditioned
        # TRELLIS. Its own preprocessing already initialized this U2NET session
        # for opaque inputs; reuse it instead of loading another model.
        if reference.getchannel("A").getextrema()[0] < 255:
            reference_cutout = reference
        else:
            _stage("removing_reference_background")
            reference_cutout = rembg.remove(reference.convert("RGB"), session=pipeline.rembg_session)
    del offload, pipeline
    torch.cuda.empty_cache()

    # The v1 exporter bakes Gaussian appearance onto the extracted mesh and
    # rotates Z-up geometry to glTF Y-up internally; do not rotate it again.
    source_mesh = outputs["mesh"][0]
    raw_faces = int(source_mesh.faces.shape[0])
    if raw_faces < 1_000:
        raise RuntimeError("TRELLIS produced an empty or unusable mesh")
    # In TRELLIS v1, simplify is the *fraction of faces to remove*. Keep about
    # 100k faces, without discarding detail from meshes already near that size.
    simplify = max(0.0, min(0.95, 1.0 - 100_000 / raw_faces))
    _stage("baking_texture")
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        source_mesh,
        simplify=simplify,
        texture_size=2048,
    )
    # TRELLIS bakes color only. Leaving metallicFactor unset means glTF's
    # default of 1 (metal), which makes skin and fabric unnaturally dark.
    glb.visual.material.metallicFactor = 0.0
    _stage("enhancing_front_texture")
    glb, detail = enhance_front_texture(glb, reference_cutout)
    _stage("front_texture_result", **detail)
    face_count = len(glb.faces)
    if face_count < 1_000:
        raise RuntimeError("TRELLIS produced an empty or unusable GLB mesh")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged = output_path.with_name(output_path.stem + ".partial.glb")
    try:
        # Trimesh's default GLB export embeds standard PNG textures in BIN;
        # extension_webp=True is avoided for broader viewer compatibility.
        _stage("exporting", faces=face_count, texture_size=2048)
        glb.export(str(staged))
        _verify_glb(staged)
        os.replace(staged, output_path)
    finally:
        staged.unlink(missing_ok=True)
    _stage("complete", faces=face_count, bytes=output_path.stat().st_size,
           **_gpu_memory(torch))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.input, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
