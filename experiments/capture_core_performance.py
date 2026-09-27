"""Capture every saved Native Core frame in its saved Studio scene.

This is an offline replay: no inference, pose blending, interpolation, frame
duplication, or G1 actor is involved. Connect a browser to the printed local
URL so Viser can render the StageZero Studio client, then the script writes a
20 fps MP4, contact sheets, and a provenance manifest.

Experimental --canonical plus --scene replays paired InterGen research arrays
retargeted to Core27 at 20 fps, without a native project or ARDY history claim.
The supplied scene is a backdrop; renderer body fitting may alter contacts.

Explicit --assisted plus --scene displays saved offline-assisted positions and
rotations as a visual proof. It builds a CanonicalClip without native features,
keeps the retargeter's fixed neutral root-height correction, uses no additional
clip floor shift, and disables the renderer's second foot-clearance pass.

Example:
    python experiments/capture_core_performance.py \
        --archive review/studio-core/browser-two-actors.core.stagezero.npz \
        --output-dir /private/tmp/core-pair-capture --port 24892
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from object_scene import ObjectSceneLayer  # noqa: E402
from scene_composition import validate_scene  # noqa: E402
from scene_ground import has_authored_ground  # noqa: E402
from realtime_clip import CanonicalClip  # noqa: E402
from studio_core_renderer import StudioCoreRenderer  # noqa: E402
from studio_core_session import CoreStudioSession  # noqa: E402
from studio_server import create_studio_server  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def sheet_indices(frames: int, count: int) -> set[int]:
    chosen = min(count, frames)
    if chosen == 1:
        return {0}
    return {round(i * (frames - 1) / (chosen - 1)) for i in range(chosen)}


def write_contact_sheet(samples: list[tuple[int, Image.Image]], path: Path) -> None:
    if not samples:
        return
    width, height = samples[0][1].size
    columns = min(4, len(samples))
    rows = (len(samples) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * (height + 28)), (18, 24, 31))
    draw = ImageDraw.Draw(sheet)
    for index, (frame, picture) in enumerate(samples):
        x = (index % columns) * width
        y = (index // columns) * (height + 28)
        sheet.paste(picture, (x, y + 28))
        draw.text((x + 8, y + 7), f"Frame {frame:04d} · {frame / 20:.2f} s", fill="white")
    sheet.save(path)


def get_render_with_timeout(client, *, width: int, height: int, timeout: float) -> np.ndarray:
    """Bound a stalled browser render without changing the native frame clock."""
    result: queue.Queue = queue.Queue(maxsize=1)

    def request() -> None:
        try:
            result.put((client.get_render(height=height, width=width), None))
        except BaseException as exc:
            result.put((None, exc))

    worker = threading.Thread(target=request, name="core-capture-webgl", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError("Browser did not answer the WebGL render request; reconnect and restart capture")
    pixels, error = result.get_nowait()
    if error is not None:
        raise error
    return pixels


class AssistedPresentationRenderer(StudioCoreRenderer):
    """Retarget assisted poses with the fixed neutral offset and no second IK."""

    def _fit_actor(self, index: int, clip: CanonicalClip, start: int,
                   offset: float | None) -> tuple[np.ndarray, np.ndarray, float, dict | None]:
        if offset not in (None, 0, 0.0):
            raise ValueError("Assisted presentation requires a fixed zero floor offset")
        character = self.characters[index]
        payload = character.clip_payload(
            clip.positions[index:index + 1, start:],
            clip.rotations[index:index + 1, start:],
            preserve_root_height=False,
            preserve_wrists=False,
            preserve_feet=False,
            floor_y=self.floor_y,
            floor_offsets=[0.0],
            fps=float(clip.fps),
        )
        floor_offset = float(payload["character_provenance"]["floor_offsets"][0])
        if floor_offset != 0.0:
            raise ValueError("Assisted presentation retarget unexpectedly changed its floor offset")
        return (np.asarray(payload["fitted_positions"][0], dtype=np.float32),
                np.asarray(payload["fitted_rotations"][0], dtype=np.float32),
                0.0, None)

    def _measure_fitting(self, clip: CanonicalClip) -> dict:
        result = super()._measure_fitting(clip)
        result.update({
            "floor_offsets_m": [0.0] * len(clip.actor_ids),
            "floor_method": "explicit fixed zero floor translation; no clip-derived calibration",
            "preserve_native_feet": False,
            "foot_clearance_pass": False,
            "retarget_root_height_method": "fixed neutral Core-to-mesh sole alignment",
            "retarget_root_height_offset_m": [
                float(self.characters[index].root_height_offset)
                for index in range(len(clip.actor_ids))
            ],
        })
        return result


def load_assisted_clip(archive: Path) -> tuple[CanonicalClip, dict]:
    """Load only the saved assisted pose arrays; never attach native history."""
    import zipfile
    from terrain_assisted_session import _validate_array_headers
    if archive.stat().st_size > 64_000_000:
        raise ValueError("Assisted archive exceeds the 64 MB limit")
    with zipfile.ZipFile(archive) as zipped:
        entries = zipped.infolist()
        if (len(entries) > 16 or len({entry.filename for entry in entries}) != len(entries)
                or sum(entry.file_size for entry in entries) > 200_000_000
                or any(entry.flag_bits & 1 or not entry.filename.endswith('.npy')
                       for entry in entries)):
            raise ValueError("Assisted archive has unexpected or oversized entries")
        _validate_array_headers(zipped, entries)
    with np.load(archive, allow_pickle=False) as data:
        required = {"positions", "rotations", "assisted_presentation"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Assisted archive is missing required arrays: {missing}")
        if np.asarray(data["assisted_presentation"]).shape != () or not bool(data["assisted_presentation"]):
            raise ValueError("Assisted archive must be explicitly marked assisted_presentation=true")
        positions = np.array(data["positions"], dtype=np.float32, copy=True)
        rotations = np.array(data["rotations"], dtype=np.float32, copy=True)
        source_fields = set(data.files)
        accepted = (bool(data["accepted"]) if "accepted" in source_fields
                    and np.asarray(data["accepted"]).shape == () else None)
    if positions.ndim != 3 or positions.shape[1:] != (27, 3):
        raise ValueError("Assisted positions must have shape [frames, 27, 3]")
    if rotations.shape != (positions.shape[0], 27, 3, 3):
        raise ValueError("Assisted rotations must have shape [frames, 27, 3, 3]")
    clip = CanonicalClip.from_arrays(
        positions[None], rotations[None], actor_ids=("assisted",),
        source="offline_assisted_presentation",
        metadata={
            "source_format": "saved assisted Core27 positions and rotations",
            "native_features_loaded": False,
            "accepted": accepted,
        },
        native_features=None,
    )
    return clip, {
        "source_fields": sorted(source_fields),
        "native_positions_present_in_source_archive": "native_positions" in source_fields,
        "native_rotations_present_in_source_archive": "native_rotations" in source_fields,
        "native_features_present_in_source_archive": "native_features" in source_fields,
        "accepted": accepted,
    }


def capture(args: argparse.Namespace) -> dict:
    paired_project = getattr(args, "paired_project", None)
    assisted_source = getattr(args, "assisted", None)
    assisted = assisted_source is not None
    research = paired_project is not None or getattr(args, "canonical", None) is not None
    offline = research or assisted
    source = paired_project or args.canonical or assisted_source or args.archive
    archive = source.resolve()
    content = archive.read_bytes()
    if len(content) > 64_000_000:
        raise ValueError("Motion archive exceeds the Studio's 64 MB limit")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    video = output / "performance.mp4"
    sheet = output / "contact-sheet.png"
    first_frame = output / "first-frame.png"
    last_frame = output / "last-frame.png"
    manifest_path = output / "capture-manifest.json"
    existing = [path for path in (video, sheet, first_frame, last_frame, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Capture output already exists: {existing[0]}")

    assisted_fields: dict | None = None
    with (nullcontext(None) if offline else CoreStudioSession()) as session:
        if offline:
            if paired_project is not None:
                from paired_scene import decode_project
                clip, scene, _ = decode_project(content)
                capture_label = "StageZero paired InterGen research capture"
                button_label = "Start exact research capture"
            elif research:
                from experiments.trial_paired_scene import load_research_clip
                clip, content = load_research_clip(archive)
                if args.scene is None or not args.scene.is_file():
                    raise ValueError("Research capture requires an explicit --scene JSON file")
                if args.scene.stat().st_size > 900_000:
                    raise ValueError("Research capture scene exceeds its size limit")
                scene = validate_scene(json.loads(args.scene.read_text()))
                capture_label = "StageZero paired InterGen research capture"
                button_label = "Start exact research capture"
            else:
                if args.scene is None or not args.scene.is_file():
                    raise ValueError("Assisted presentation capture requires an explicit --scene JSON file")
                if args.scene.stat().st_size > 900_000:
                    raise ValueError("Assisted presentation scene exceeds its size limit")
                clip, assisted_fields = load_assisted_clip(archive)
                scene = validate_scene(json.loads(args.scene.read_text()))
                capture_label = "StageZero assisted visual proof"
                button_label = "Start assisted visual proof"
        else:
            state = session.load(content)
            clip = session.timeline_clip()
            scene = validate_scene(session.scene_document)
            if not clip.frames:
                raise ValueError("Native Core archive has no committed frames")
            if state.get("scene_changed_since_motion"):
                raise ValueError("The archived scene changed after motion; capture against the original layout instead")
            capture_label = "StageZero Core performance capture"
            button_label = "Start exact Core capture"
        if clip.fps != 20:
            raise ValueError("Expected a canonical 20 fps timeline")
        captured_frames = clip.frames if args.max_frames is None else min(args.max_frames, clip.frames)

        server = create_studio_server(host="127.0.0.1", port=args.port,
                                      label=capture_label)
        try:
            server.gui.configure_theme(dark_mode=True, control_layout="floating",
                                       show_logo=False, show_share_button=False)
            server.scene.set_up_direction("+y")
            server.scene.world_axes.visible = False
            server.scene.configure_environment_map(None if args.environment == "none" else args.environment)
            server.scene.configure_default_lights(enabled=True, cast_shadow=True)
            server.scene.add_light_ambient("/fill", color=(191, 215, 239), intensity=.6)
            server.scene.add_box("/floor", color=(20, 28, 38), dimensions=(200, .1, 200),
                                 position=(0, -.07, 0), cast_shadow=False,
                                 visible=not has_authored_ground(scene["objects"]))
            terrain = None
            if getattr(args, "terrain_support", False):
                from scene_interaction_geometry import SceneInteractionGeometry
                terrain = SceneInteractionGeometry.from_scene(scene)
            renderer_type = AssistedPresentationRenderer if assisted else StudioCoreRenderer
            renderer = renderer_type(server, name_prefix="/core-cast",
                                     terrain_geometry=terrain,
                                     paired_retarget=bool(getattr(args, "paired_retarget", False)))
            renderer.set_clip(clip)
            renderer.tick(0)
            renderer.set_visible(True)
            layer = ObjectSceneLayer(server)
            objects = scene["objects"]
            static_states = [{"id": obj["id"], "position": obj["position"],
                              "color": obj["color"], "active": False} for obj in objects]
            # Publish every prop before the texture/geometry settling period;
            # otherwise the first recorded frame can contain a partial scene.
            layer.update(objects, {"objects": static_states,
                "assets": scene.get("assets", []), "effects": scene["effects"],
                "lighting": scene["lighting"], "seconds": 0.})

            camera = scene.get("camera", {"position": [3.0, 2.7, 8.0],
                                          "look_at": [0.0, 1.2, 0.0]})
            position = args.camera_position or camera["position"]
            look_at = args.look_at or camera["look_at"]
            if sum((a - b) ** 2 for a, b in zip(position, look_at)) < .01:
                raise ValueError("Camera position and look-at must differ")
            start_button = server.gui.add_button(button_label)
            chosen = {}
            ready = threading.Event()

            @start_button.on_click
            def choose_render_client(event):
                if event.client is not None and not ready.is_set():
                    chosen["client"] = event.client
                    ready.set()
                    start_button.disabled = True

            print(f"CONNECT BROWSER, ENTER STUDIO, THEN CLICK '{button_label}': http://127.0.0.1:{server.get_port()}", flush=True)
            deadline = time.monotonic() + args.wait_seconds
            while not ready.is_set():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"No browser clicked {button_label} before --wait-seconds elapsed")
                time.sleep(.2)
            # This exact client owns the visible Studio canvas; no stale welcome
            # socket or another open tab can receive the render request.
            client = chosen["client"]
            client.camera.up_direction = (0, 1, 0)
            client.camera.near = .05
            client.camera.far = 400.
            client.camera.position = tuple(position)
            client.camera.look_at = tuple(look_at)
            client.camera.fov = np.deg2rad(args.fov)
            time.sleep(args.settle_seconds)

            command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                       "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}",
                       "-r", "20", "-i", "-", "-an", "-c:v", "libx264",
                       "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
                       "-frames:v", str(captured_frames), "-movflags", "+faststart", str(video)]
            selected = sheet_indices(captured_frames, args.sheet_frames)
            thumbs: list[tuple[int, Image.Image]] = []
            frame_hashes = []
            render_times = []
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
            try:
                for frame in range(captured_frames):
                    renderer.tick(frame)
                    states = static_states
                    if not offline and state.get("scene_reactions_enabled"):
                        from core_scene_reactions import object_states
                        states = object_states(scene, clip, frame, enabled=True,
                            start_frame=state.get("scene_reactions_start_frame", 0),
                            terrain=state.get("terrain_navigation_enabled", False),
                            terrain_start_frame=state.get("terrain_navigation_start_frame", 0))
                    layer.update(objects, {"objects": states,
                        "assets": scene.get("assets", []), "effects": scene["effects"],
                        "lighting": scene["lighting"], "seconds": frame / 20.})
                    server.flush()
                    started = time.perf_counter()
                    pixels = get_render_with_timeout(client, height=args.height,
                                                     width=args.width, timeout=args.render_timeout)
                    render_times.append((time.perf_counter() - started) * 1000)
                    rgb = np.ascontiguousarray(pixels[:, :, :3], dtype=np.uint8)
                    if rgb.shape != (args.height, args.width, 3):
                        raise ValueError(f"Browser returned wrong frame shape: {rgb.shape}")
                    raw = rgb.tobytes()
                    frame_hashes.append(hashlib.sha256(raw).hexdigest())
                    process.stdin.write(raw)
                    if frame == 0:
                        Image.fromarray(rgb).save(first_frame)
                    if frame == captured_frames - 1:
                        Image.fromarray(rgb).save(last_frame)
                    if frame in selected:
                        thumbnail = Image.fromarray(rgb).resize((400, 225))
                        thumbs.append((frame, thumbnail))
                    if frame % 40 == 0 or frame == captured_frames - 1:
                        print(f"CAPTURED {frame + 1}/{captured_frames}", flush=True)
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError("ffmpeg failed to encode the Core capture")
            except BaseException:
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()
                if process.poll() is None:
                    process.terminate()
                process.wait()
                for partial in (video, first_frame, last_frame, sheet):
                    partial.unlink(missing_ok=True)
                raise
            write_contact_sheet(thumbs, sheet)
            scene_bytes = json.dumps(scene, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()
            manifest = {
                "terrain_support_fitting": bool(getattr(args, "terrain_support", False)),
                "fitting_provenance": renderer.fitting_provenance,
                "scene_name": scene["name"], "scene_sha256": hashlib.sha256(scene_bytes).hexdigest(),
                "scene_objects": len(objects), "scene_assets": len(scene.get("assets", [])),
                "actor_ids": list(clip.actor_ids),
                "frames": captured_frames, "duration_seconds": captured_frames / 20,
                "is_excerpt": captured_frames < clip.frames,
                "positions_sha256": sha256_array(clip.positions),
                "rotations_sha256": sha256_array(clip.rotations),
                "capture_camera": {"position": list(position), "look_at": list(look_at),
                                   "fov_degrees": args.fov},
                "capture_size": [args.width, args.height],
                "frame_indices": list(range(captured_frames)),
                "raw_rgb_frame_sha256": frame_hashes,
                "contact_sheet_frames": [index for index, _ in thumbs],
                "render_roundtrip_median_ms": round(statistics.median(render_times), 2),
                "render_roundtrip_scope": "Browser WebGL get_render roundtrip, not playback FPS",
                "video_sha256": sha256_file(video),
                "contact_sheet_sha256": sha256_file(sheet),
                "first_frame_sha256": sha256_file(first_frame),
                "last_frame_sha256": sha256_file(last_frame),
            }
            if assisted:
                manifest.update({
                    "capture_kind": "Offline assisted terrain presentation WebGL replay",
                    "source": "assisted_presentation",
                    "research_only": True,
                    "assisted_presentation": True,
                    "accepted": assisted_fields["accepted"],
                    "source_archive": str(archive),
                    "source_archive_sha256": hashlib.sha256(content).hexdigest(),
                    "source_archive_fields": assisted_fields["source_fields"],
                    "source_positions_sha256": sha256_array(clip.positions),
                    "source_rotations_sha256": sha256_array(clip.rotations),
                    "native_positions_present_in_source_archive": assisted_fields["native_positions_present_in_source_archive"],
                    "native_rotations_present_in_source_archive": assisted_fields["native_rotations_present_in_source_archive"],
                    "native_features_present_in_source_archive": assisted_fields["native_features_present_in_source_archive"],
                    "native_features_loaded_into_clip": False,
                    "native_features_used_for_render": False,
                    "native_core_project": False,
                    "assisted_canonical_clip": {
                        "source": clip.source,
                        "fps": clip.fps,
                        "frames": clip.frames,
                        "native_features": None,
                    },
                    "canonical_total_frames": clip.frames,
                    "fps": clip.fps,
                    "scene_source": str(args.scene.resolve()),
                    "scene_conditioned": False,
                    "physical_contact_verified": False,
                    "rendering_note": "Saved assisted positions and rotations were retargeted to the existing skinned character. The fixed neutral root-height correction was applied; clip floor translation is fixed at zero; no second foot-clearance or foot-IK pass was used.",
                    "fitting_provenance": renderer.fitting_provenance,
                    "capture_note": "Offline visual presentation only. One render per saved assisted frame in order; no interpolation. This does not claim native ARDY features, native Core generation, streaming behavior, or verified physical contact.",
                })
            elif research:
                manifest.update({
                    "capture_kind": "InterGen research retargeted canonical WebGL replay",
                    "source": "intergen", "research_only": True,
                    "canonical_clip": str(archive), "canonical_sha256": hashlib.sha256(content).hexdigest(),
                    "canonical_total_frames": clip.frames, "fps": 20,
                    "source_format": clip.metadata.get("source_format"),
                    "clip_metadata": dict(clip.metadata),
                    "native_features_available": False, "native_core_project": False,
                    "scene_source": "saved paired project" if paired_project else str(args.scene.resolve()), "scene_conditioned": False,
                    "physical_contact_verified": False,
                    "rendering_note": "Paired world-wrist preserving fit with common floor translation" if getattr(args, "paired_retarget", False) else "Baseline fitting does not preserve native wrists",
                    "fitting_provenance": renderer.fitting_provenance,
                    "capture_note": "One render per listed saved retargeted canonical frame, in order. No extra interpolation. Explicit scene is a rendering backdrop, not a generation constraint. No claim of native ARDY features or preserved physical contact.",
                })
            else:
                manifest.update({
                    "capture_kind": ("Native Core archive WebGL excerpt" if captured_frames < clip.frames
                                     else "Exact native Core archive WebGL replay"),
                    "archive": str(archive), "archive_sha256": hashlib.sha256(content).hexdigest(),
                    "archive_total_frames": clip.frames, "native_fps": 20,
                    "native_features_sha256": sha256_array(clip.native_features),
                    "capture_note": "One WebGL render per listed saved native frame in order; no interpolation or inferred contact. Opt-in props use the archived native root trajectory for reactions; fitting is disclosed separately.",
                })
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"CAPTURE COMPLETE: {video}", flush=True)
            return manifest
        finally:
            server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--archive", type=Path, help="Saved .core.stagezero.npz (native default)")
    inputs.add_argument("--paired-project", type=Path, help="Saved separate paired research project from Studio")
    inputs.add_argument("--canonical", type=Path,
                        help="Experimental .intergen.canonical.npz from trial_paired_scene.py")
    inputs.add_argument("--assisted", type=Path,
                        help="Saved offline-assisted presentation .npz; no native features are loaded")
    parser.add_argument("--scene", type=Path, help="Explicit scene JSON required with --canonical or --assisted")
    parser.add_argument("--terrain-support", action="store_true", help="Explicit terrain-relative sole fitting; native arrays remain unchanged")
    parser.add_argument("--paired-retarget", action="store_true", help="Preserve paired world wrists and a common floor translation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=24892)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--wait-seconds", type=float, default=180.)
    parser.add_argument("--settle-seconds", type=float, default=8.,
                        help="Wait for the browser's Studio canvas and textures to settle")
    parser.add_argument("--render-timeout", type=float, default=45.)
    parser.add_argument("--sheet-frames", type=int, default=12)
    parser.add_argument("--max-frames", type=int,
                        help="Capture only the first N frames; manifest labels this as an excerpt")
    parser.add_argument("--camera-position", type=float, nargs=3)
    parser.add_argument("--look-at", type=float, nargs=3)
    parser.add_argument("--fov", type=float, default=48.)
    parser.add_argument("--environment", choices=("studio", "warehouse", "none"), default="studio")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be 1–65535")
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        parser.error("--width and --height must be positive even numbers")
    if not 1 <= args.sheet_frames <= 64 or not 0 < args.wait_seconds <= 3600:
        parser.error("--sheet-frames must be 1–64 and --wait-seconds must be 0–3600")
    if not 0 <= args.settle_seconds <= 60 or not 1 <= args.render_timeout <= 120:
        parser.error("--settle-seconds must be 0–60 and --render-timeout must be 1–120")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if not 1 <= args.fov <= 120:
        parser.error("--fov must be 1–120 degrees")
    if args.paired_retarget and args.canonical is None and args.paired_project is None:
        parser.error("--paired-retarget requires research motion")
    if args.assisted is not None and (args.terrain_support or args.paired_retarget):
        parser.error("--assisted cannot be combined with terrain-support or paired-retarget fitting")
    source = args.paired_project or args.canonical or args.assisted or args.archive
    if not source.is_file():
        parser.error(f"Motion input does not exist: {source}")
    if (args.canonical is not None or args.assisted is not None) and (args.scene is None or not args.scene.is_file()):
        parser.error("--canonical and --assisted require an existing --scene JSON file")
    if (args.archive is not None or args.paired_project is not None) and args.scene is not None:
        parser.error("Native --archive uses its saved scene; --scene is only for research --canonical")
    capture(args)


if __name__ == "__main__":
    main()
