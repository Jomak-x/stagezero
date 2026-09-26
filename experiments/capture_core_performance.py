"""Capture every saved Native Core frame in its saved Studio scene.

This is an offline replay: no inference, pose blending, interpolation, frame
duplication, or G1 actor is involved. Connect a browser to the printed local
URL so Viser can render the StageZero Studio client, then the script writes a
20 fps MP4, contact sheets, and a provenance manifest.

Example:
    python experiments/capture_core_performance.py \
        --archive review/studio-core/browser-two-actors.core.stagezero.npz \
        --output-dir /private/tmp/core-pair-capture --port 24892
"""

from __future__ import annotations

import argparse
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


def capture(args: argparse.Namespace) -> dict:
    archive = args.archive.resolve()
    content = archive.read_bytes()
    if len(content) > 64_000_000:
        raise ValueError("Native Core archive exceeds the Studio's 64 MB limit")
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

    with CoreStudioSession() as session:
        state = session.load(content)
        clip = session.timeline_clip()
        scene = validate_scene(session.scene_document)
        if not clip.frames:
            raise ValueError("Native Core archive has no committed frames")
        if state.get("scene_changed_since_motion"):
            raise ValueError("The archived scene changed after motion; capture against the original layout instead")
        if clip.fps != 20:
            raise ValueError("Expected the native Core 20 fps timeline")
        captured_frames = clip.frames if args.max_frames is None else min(args.max_frames, clip.frames)

        server = create_studio_server(host="127.0.0.1", port=args.port,
                                      label="StageZero Core performance capture")
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
            renderer = StudioCoreRenderer(server, name_prefix="/core-cast")
            renderer.set_clip(clip)
            renderer.tick(0)
            renderer.set_visible(True)
            layer = ObjectSceneLayer(server)
            objects = scene["objects"]
            static_states = [{"id": obj["id"], "position": obj["position"],
                              "color": obj["color"], "active": False} for obj in objects]

            camera = scene.get("camera", {"position": [3.0, 2.7, 8.0],
                                          "look_at": [0.0, 1.2, 0.0]})
            position = args.camera_position or camera["position"]
            look_at = args.look_at or camera["look_at"]
            if sum((a - b) ** 2 for a, b in zip(position, look_at)) < .01:
                raise ValueError("Camera position and look-at must differ")
            start_button = server.gui.add_button("Start exact Core capture")
            chosen = {}
            ready = threading.Event()

            @start_button.on_click
            def choose_render_client(event):
                if event.client is not None and not ready.is_set():
                    chosen["client"] = event.client
                    ready.set()
                    start_button.disabled = True

            print(f"CONNECT BROWSER, ENTER STUDIO, THEN CLICK 'Start exact Core capture': http://127.0.0.1:{server.get_port()}", flush=True)
            deadline = time.monotonic() + args.wait_seconds
            while not ready.is_set():
                if time.monotonic() >= deadline:
                    raise TimeoutError("No browser clicked Start exact Core capture before --wait-seconds elapsed")
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
                    layer.update(objects, {"objects": static_states,
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
                "capture_kind": ("Native Core archive WebGL excerpt" if captured_frames < clip.frames
                                 else "Exact native Core archive WebGL replay"),
                "archive": str(archive), "archive_sha256": hashlib.sha256(content).hexdigest(),
                "scene_name": scene["name"], "scene_sha256": hashlib.sha256(scene_bytes).hexdigest(),
                "scene_objects": len(objects), "scene_assets": len(scene.get("assets", [])),
                "actor_ids": list(clip.actor_ids), "native_fps": 20,
                "frames": captured_frames, "duration_seconds": captured_frames / 20,
                "archive_total_frames": clip.frames,
                "is_excerpt": captured_frames < clip.frames,
                "positions_sha256": sha256_array(clip.positions),
                "rotations_sha256": sha256_array(clip.rotations),
                "native_features_sha256": sha256_array(clip.native_features),
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
                "capture_note": "One WebGL render per listed saved native frame in order; no interpolation or inferred contact. Props remain at their archived static positions.",
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"CAPTURE COMPLETE: {video}", flush=True)
            return manifest
        finally:
            server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Saved .core.stagezero.npz")
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
    if not args.archive.is_file():
        parser.error(f"Archive does not exist: {args.archive}")
    capture(args)


if __name__ == "__main__":
    main()
