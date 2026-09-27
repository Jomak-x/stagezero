"""Exact native timeline and separate rig17 presentation for Core terrain mode.

The ordinary Core project format stays unchanged. Terrain projects add four
bounded NPZ members to that format, and loading checks their native digest
before a presentation can be shown or used for continuation.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from realtime_clip import CanonicalClip, FPS, MAX_CLIP_FRAMES
from realtime_director import RealtimeDirector, StageSpec
from scene_composition import validate_scene
from terrain_assisted_session import AssistedTerrainResult, PresentationClip


MAX_STUDIO_BYTES = 64_000_000
MAX_EXPANDED_BYTES = 180_000_000
TERRAIN_MEMBERS = frozenset({"terrain_manifest.npy", "terrain_positions.npy",
                             "terrain_rotations.npy", "terrain_stance.npy"})


def native_digest(clip: CanonicalClip) -> str:
    if clip.native_features is None:
        raise ValueError("Terrain continuation requires exact native features")
    digest = hashlib.sha256()
    for array in (clip.positions, clip.rotations, clip.native_features):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def presentation_digest(clip: PresentationClip) -> str:
    digest = hashlib.sha256()
    for array in (clip.positions, clip.rotations, clip.stance):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def director_from_assisted(result: AssistedTerrainResult, *, clock,
                           project_metadata: dict, playhead: int | None = None) -> RealtimeDirector:
    """Rebuild a playback clock from already verified Core frames, without inference."""
    native = result.native_clip
    display = result.presentation
    if (native.actor_ids != display.actor_ids or native.frames != display.frames
            or native.native_features is None or native.frames % 40
            or native.frames > MAX_CLIP_FRAMES
            or not np.allclose(display.positions[:, :, 0, [0, 2]],
                               native.positions[:, :, 0, [0, 2]], atol=1e-5)
            or result.report.get("native_sha256") != native_digest(native)
            or result.report.get("accepted") is not False):
        raise ValueError("Terrain native and presentation streams do not match")
    metadata = json.loads(json.dumps(project_metadata, allow_nan=False))
    studio = metadata.setdefault("studio_core", {})
    if validate_scene(studio.get("scene_document")) != validate_scene(result.scene):
        raise ValueError("Terrain result was generated against a different scene")
    studio.update(terrain_assisted_version=1, terrain_navigation_version=1,
                  terrain_navigation_start_frame=0, scene_reactions_version=1,
                  scene_reactions_start_frame=0)
    director = RealtimeDirector(native.actor_ids, clock=clock,
                                target_buffer_frames=MAX_CLIP_FRAMES,
                                max_buffer_frames=MAX_CLIP_FRAMES,
                                project_metadata=metadata)
    spans = [(int(item["native_span"][0]), int(item["native_span"][1]))
             for item in result.report.get("actions", ())]
    if len(spans) != len(result.routes) or any(
        item.get("route") != route
        for item, route in zip(result.report.get("actions", ()), result.routes)
    ):
        raise ValueError("Terrain action provenance differs from its route list")
    if not spans:
        spans = [(0, native.frames)]
    if spans[0][0] != 0 or spans[-1][1] != native.frames or any(
        a < 0 or b <= a or a % 40 or b % 40 or
        (index and spans[index-1][1] != a)
        for index, (a, b) in enumerate(spans)
    ):
        raise ValueError("Terrain action spans do not cover the native timeline")
    specs = tuple(StageSpec(
        str(result.report["actions"][index]["route"].get("verb", "Terrain action"))[:500] or "Terrain action",
        frames=end-start, metadata={"terrain_assisted_action_index": index})
        for index, (start, end) in enumerate(spans))
    director.queue_sequence(specs)
    for start in range(0, native.frames, 40):
        request = director.claim_request()
        if request is None:
            raise ValueError("Cannot restore verified terrain timeline")
        chunk = native.slice_frames(start, start+40)
        committed = CanonicalClip(chunk.positions, chunk.rotations, FPS, native.actor_ids,
                                  "ardy_core", {}, chunk.native_features)
        if not director.complete(request.request_id, committed):
            raise ValueError("Cannot commit verified terrain timeline")
    if playhead is not None:
        director.seek(min(max(0, playhead), native.frames-1))
    return director


def _member(value) -> bytes:
    output = io.BytesIO()
    np.save(output, value, allow_pickle=False)
    return output.getvalue()


def _preflight(content: bytes) -> bool:
    if not isinstance(content, bytes) or len(content) > MAX_STUDIO_BYTES:
        raise ValueError("Native studio archive must be at most 64 MB")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        entries = archive.infolist()
        names = [item.filename for item in entries]
        if (len(names) > 1510 or len(set(names)) != len(names)
                or sum(item.file_size for item in entries) > MAX_EXPANDED_BYTES
                or any(item.file_size > MAX_EXPANDED_BYTES or item.flag_bits & 1
                       for item in entries)):
            raise ValueError("Terrain project archive entries are oversized or duplicated")
    return TERRAIN_MEMBERS <= set(names)


def _validate_presentation_headers(content: bytes, actor_ids: tuple[str, ...], frames: int):
    """Bound every added NPY allocation before NumPy materializes it."""
    count = len(actor_ids)
    expected = {"terrain_positions.npy": ((count, frames, 17, 3), np.dtype("float32")),
                "terrain_rotations.npy": ((count, frames, 17, 3, 3), np.dtype("float32")),
                "terrain_stance.npy": ((count, frames, 2), np.dtype("bool"))}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for name in TERRAIN_MEMBERS:
            entry = archive.getinfo(name)
            with archive.open(entry) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    raise ValueError("Unsupported terrain project array encoding")
                size = math.prod(shape)*dtype.itemsize
                if (dtype.hasobject or len(shape) > 5 or size < 0
                        or size != entry.file_size-member.tell()):
                    raise ValueError("Terrain project array header disagrees with its bounded member")
                if name == "terrain_manifest.npy":
                    if shape != () or dtype.kind != "U" or size > 1_000_000:
                        raise ValueError("Terrain project manifest header is invalid")
                elif (shape, dtype) != expected[name]:
                    raise ValueError("Terrain presentation array shape or dtype is invalid")


def _verify_render_assets(hashes: tuple[str, ...]):
    # Import lazily: native-only Core work must not load the Studio renderer.
    from studio_core_renderer import DEFAULT_ASSETS
    expected = tuple(hashlib.sha256(Path(path).read_bytes()).hexdigest()
                     for path in DEFAULT_ASSETS[:len(hashes)])
    if hashes != expected:
        raise ValueError("Terrain presentation targets different character assets")


def pack_terrain_project(director: RealtimeDirector,
                         result: AssistedTerrainResult) -> bytes:
    native = director.timeline_clip()
    if (native is None or native_digest(native) != native_digest(result.native_clip)
            or native.frames != result.presentation.frames
            or director.project_metadata.get("studio_core", {}).get("terrain_assisted_version") != 1):
        raise ValueError("Terrain project streams are inconsistent")
    manifest = {"version": 1, "native_sha256": native_digest(native),
                "presentation_sha256": presentation_digest(result.presentation),
                "actor_ids": native.actor_ids, "fps": FPS,
                "rig_asset_sha256": result.presentation.rig_asset_sha256,
                "report": result.report, "routes": result.routes,
                "reaction_states": result.reaction_states,
                "presentation_report": result.presentation.report}
    base = director.save_project()
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(base)) as original, zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as target:
        for name in original.namelist():
            target.writestr(name, original.read(name))
        target.writestr("terrain_manifest.npy", _member(np.array(json.dumps(manifest, allow_nan=False))))
        target.writestr("terrain_positions.npy", _member(result.presentation.positions))
        target.writestr("terrain_rotations.npy", _member(result.presentation.rotations))
        target.writestr("terrain_stance.npy", _member(result.presentation.stance))
    content = output.getvalue()
    if len(content) > MAX_STUDIO_BYTES:
        raise ValueError("Native studio archive exceeds the 64 MB limit")
    return content


def unpack_terrain_project(content: bytes, director: RealtimeDirector) -> AssistedTerrainResult | None:
    has_terrain = _preflight(content)
    assisted = director.project_metadata.get("studio_core", {}).get("terrain_assisted_version")
    if not has_terrain:
        if assisted is not None:
            raise ValueError("Terrain presentation is missing from this project")
        return None
    if assisted != 1:
        raise ValueError("Terrain project marker is missing or unsupported")
    _validate_presentation_headers(content, director.actor_ids, director.total_frames)
    with np.load(io.BytesIO(content), allow_pickle=False) as data:
        raw = str(data["terrain_manifest"])
        if len(raw) > 1_000_000:
            raise ValueError("Terrain project manifest is too large")
        manifest = json.loads(raw)
        if (manifest.get("version") != 1 or manifest.get("fps") != FPS
                or tuple(manifest.get("actor_ids", ())) != director.actor_ids):
            raise ValueError("Unsupported terrain project presentation")
        _verify_render_assets(tuple(manifest["rig_asset_sha256"]))
        native = director.timeline_clip()
        if native is None or manifest.get("native_sha256") != native_digest(native):
            raise ValueError("Terrain presentation does not match exact native history")
        display = PresentationClip(data["terrain_positions"], data["terrain_rotations"],
                                   data["terrain_stance"], FPS, director.actor_ids,
                                   tuple(manifest["rig_asset_sha256"]),
                                   manifest.get("presentation_report"))
    report = manifest["report"]
    if (display.frames != native.frames or report.get("accepted") is not False
            or report.get("native_sha256") != native_digest(native)
            or manifest.get("presentation_sha256") != presentation_digest(display)
            or not np.allclose(display.positions[:, :, 0, [0, 2]],
                               native.positions[:, :, 0, [0, 2]], atol=1e-5)
            or tuple(report.get("rig_asset_sha256", ())) != display.rig_asset_sha256):
        raise ValueError("Terrain project has inconsistent presentation provenance")
    scene = validate_scene(director.project_metadata["studio_core"]["scene_document"])
    result = AssistedTerrainResult(scene, native, display, report,
                                   tuple(manifest["routes"]), tuple(manifest["reaction_states"]))
    spans = [item.get("native_span") for item in report.get("actions", ())]
    if (len(spans) != len(result.routes) or not spans or spans[0][0] != 0
            or spans[-1][1] != native.frames or any(
                not isinstance(span, list) or len(span) != 2 or
                type(span[0]) is not int or type(span[1]) is not int or
                span[0] < 0 or span[1] <= span[0] or
                span[0] % 40 or span[1] % 40 or
                (index and spans[index-1][1] != span[0])
                for index, span in enumerate(spans))
            or any(item.get("route") != route for item, route in zip(report["actions"], result.routes))):
        raise ValueError("Terrain action archive provenance is inconsistent")
    return result
