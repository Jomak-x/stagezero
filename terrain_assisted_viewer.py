"""Separate local viewer for an offline terrain-assisted Core candidate.

Generation uses the existing warm Core HTTP service. The native feature stream
is retained in the paired archive; the skinned actor displays only the distinct
assisted pose stream. This viewer never publishes into a Studio/Core project.
"""

from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path
import queue
import re
import threading
import time

import numpy as np

from core_scene_reactions import object_states
from object_scene import ObjectSceneLayer
from realtime_client import RealtimeClient
from scene_composition import MAX_SCENE_FILE_BYTES, validate_scene
from scene_ground import has_authored_ground
from studio_server import create_studio_server
from terrain_assisted_renderer import TerrainAssistedRenderer
from terrain_assisted_session import (load_assisted_result,
                                      run_assisted_terrain_commands,
                                      save_assisted_result,
                                      save_native_terrain_result)
from traversal_kit import traversable_temple_scene


ROOT = Path(__file__).resolve().parent
DEFAULT_COMMAND = ("walk up Shallow temple stairs then cross Suspended temple bridge "
                   "then open Temple gate then enter")


def validate_presentation(result):
    """Check the separate mesh-rig clock without passing native poses to display."""
    display = result.presentation
    if result.report.get("committed_prefix_frames", 0) != 0:
        raise ValueError("Replay needs the full native prefix for correct gate reactions")
    if (result.native_clip.actor_ids != display.actor_ids or
            result.native_clip.frames != display.frames):
        raise ValueError("Native and assisted presentation clocks differ")
    if display.positions.shape[2:] != (17, 3):
        raise ValueError("Expected an assisted mesh17 presentation")
    return display


def _safe(value):
    # Viser markdown is MDX; errors may include service-controlled text.
    marked = re.sub(r"([\\`*_\[\]()#+.!|~-])", r"\\\1", str(value))
    return escape(marked, quote=False).replace("{", "&#123;").replace("}", "&#125;")


def _load_scene(path):
    if path is None:
        return validate_scene(traversable_temple_scene())
    if path.stat().st_size > MAX_SCENE_FILE_BYTES:
        raise ValueError("Scene JSON exceeds the 1 MB scene limit")
    return validate_scene(json.loads(path.read_text(encoding="utf-8")))


def checkpoint_native_result(native_result, path):
    """Persist a complete native route before the CPU-assisted solve starts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_native_terrain_result(native_result, path)
    return path


def start_placement(x, z, yaw):
    """Validate the immutable generation start in the planner's world units."""
    try:
        values = tuple(float(value) for value in (x, z, yaw))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Start X, Z, and yaw must be numbers") from exc
    x, z, yaw = values
    if (not all(math.isfinite(value) for value in values) or
            abs(x) > 200 or abs(z) > 200 or abs(yaw) > math.pi):
        raise ValueError("Start X/Z must be within ±200 m and yaw within ±π radians")
    return {"actor_1": {"position_xz": [x, z], "yaw": yaw}}


def initial_start_values(result, args):
    """Use the replay's first native root unless a CLI value overrides it."""
    if result is None:
        x, z, yaw = 0., .85, math.pi
    else:
        root = result.native_clip.positions[0, 0, 0]
        forward = result.native_clip.rotations[0, 0, 0, :, 2]
        x, z = float(root[0]), float(root[2])
        yaw = math.atan2(float(forward[0]), float(forward[2]))
    return (x if args.start_x is None else args.start_x,
            z if args.start_z is None else args.start_z,
            yaw if args.start_yaw is None else args.start_yaw)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, help="Authored Scene3 JSON; defaults to the temple fixture")
    parser.add_argument("--backend", default="http://127.0.0.1:8769",
                        help="Existing warm Core HTTP service; no service is started")
    parser.add_argument("--token-path", type=Path, default=ROOT / ".runtime/api-token",
                        help="Core bearer token file, read only when Generate is clicked")
    parser.add_argument("--port", type=int, default=24996, help="Dedicated local viewer port")
    parser.add_argument("--project", type=Path,
                        help="Replay an existing paired assisted NPZ, and save successful generation here")
    parser.add_argument("--output", type=Path,
                        help="Save new paired candidates here instead of --project or the private default")
    parser.add_argument("--start-x", type=float, help="Initial actor world X in metres")
    parser.add_argument("--start-z", type=float, help="Initial actor world Z in metres")
    parser.add_argument("--start-yaw", type=float, help="Initial actor heading in radians")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    initial_result = load_assisted_result(args.project) if args.project and args.project.is_file() else None
    if initial_result is not None:
        validate_presentation(initial_result)
    scene = _load_scene(args.scene) if args.scene or initial_result is None else initial_result.scene
    if initial_result is not None and scene != initial_result.scene:
        raise ValueError("The replay archive belongs to a different authored scene")
    start_values = initial_start_values(initial_result, args)
    start_placement(*start_values)
    output = (args.output or args.project or
              ROOT / ".runtime/terrain-assisted/candidate.assisted.npz")
    native_checkpoint_path = output.with_name(output.stem + ".native.npz")

    server = create_studio_server(host="127.0.0.1", port=args.port,
                                  label="Terrain-assisted traversal",
                                  enable_camera_keyboard_controls=False)
    server.gui.configure_theme(dark_mode=True, control_layout="collapsible",
                               control_width="large", show_logo=False,
                               show_share_button=False, brand_color=(181, 163, 119))
    server.scene.set_up_direction("+y")
    server.scene.world_axes.visible = False
    server.scene.configure_environment_map(None)
    server.scene.configure_default_lights(enabled=True, cast_shadow=True)
    server.scene.add_light_ambient("/fill", color=(191, 215, 239), intensity=.6)
    server.scene.add_box("/floor", color=(20, 28, 38), dimensions=(200, .1, 200),
                         position=(0, -.07, 0), cast_shadow=False,
                         visible=not has_authored_ground(scene["objects"]))
    layer = ObjectSceneLayer(server)
    renderer = TerrainAssistedRenderer(server, name_prefix="/terrain-assisted")

    gui = server.gui
    gui.add_markdown("# Terrain-assisted traversal\n"
                     "Separate terrain-assisted mode: ARDY upper-body motion with geometry-guided "
                     "footsteps and pelvis height. Ordinary Studio movement is unchanged. "
                     "Each generation starts a new take at the configured position.")
    command = gui.add_text("Terrain commands", initial_value=DEFAULT_COMMAND, multiline=True)
    gui.add_markdown("Set the actor's world start before generating. The temple default is "
                     "X=0, Z=0.85, yaw=π; custom scenes may need different coordinates.")
    start_x = gui.add_text("Start X (m)", initial_value=str(start_values[0]))
    start_z = gui.add_text("Start Z (m)", initial_value=str(start_values[1]))
    start_yaw = gui.add_text("Start yaw (radians)", initial_value=str(start_values[2]))
    generate = gui.add_button("Generate new assisted take")
    cancel = gui.add_button("Cancel generation", color="gray")
    play = gui.add_button("Play", color="gray")
    pause = gui.add_button("Pause", color="gray")
    seek = gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    status = gui.add_markdown("Ready to generate an offline candidate.")
    gui.add_markdown("Paired candidate output: `" + _safe(output) + "`  \n"
                     "Completed native route checkpoint: `" + _safe(native_checkpoint_path) + "`")

    events = queue.Queue()
    cancellation = threading.Event()
    current = initial_result
    frame = 0
    playing = False
    busy = False
    native_checkpoint = None
    last_tick = time.monotonic()
    suppress_seek = False

    def scene_frame(result, number):
        states = (object_states(result.scene, result.native_clip, number,
                                enabled=True, terrain=True) if result is not None else
                  object_states(scene, None))
        layer.update(scene["objects"], {"objects": states,
                     "assets": scene.get("assets", []), "effects": scene["effects"],
                     "lighting": scene["lighting"], "seconds": number / 20.0})

    def show_frame(number):
        nonlocal frame, suppress_seek
        if current is None:
            return
        frame = max(0, min(int(number), current.presentation.frames - 1))
        renderer.tick(frame)
        scene_frame(current, frame)
        suppress_seek = True
        if seek.value != frame:
            seek.value = frame
        suppress_seek = False

    def install(result):
        nonlocal current, playing
        display = validate_presentation(result)
        renderer.set_presentation(display, actor_ids=result.native_clip.actor_ids)
        renderer.set_visible(True)
        current = result
        playing = False
        seek.max = max(1, display.frames - 1)
        show_frame(0)
        review = result.report.get("visual_review", "pending")
        label = "Reviewed at normal speed" if review == "reviewed_at_1x" else "Visual review pending"
        status.content = f"{label} · {display.frames} frames · original ARDY motion saved separately"

    @server.on_client_connect
    def connected(client):
        camera = scene.get("camera", {"position": [8., 5., 5.],
                                      "look_at": [0., 1., -6.]})
        client.camera.up_direction = (0, 1, 0)
        client.camera.position = tuple(camera["position"])
        client.camera.look_at = tuple(camera["look_at"])
        client.camera.near = .05
        client.camera.far = 400.
        client.camera.fov = np.deg2rad(42.)

    @generate.on_click
    def generate_clicked(_):
        nonlocal busy, playing, native_checkpoint
        if busy:
            return
        try:
            if not args.token_path.is_file():
                raise ValueError("Core token file is missing; replay remains available")
            client = RealtimeClient(args.backend, args.token_path.read_text(encoding="utf-8"))
            text = command.value.strip()
            if not text:
                raise ValueError("Enter terrain commands")
            placement = start_placement(start_x.value, start_z.value, start_yaw.value)
        except Exception as exc:
            status.content = "Generation unavailable: " + _safe(exc)
            return
        cancellation.clear()
        busy = True
        playing = False
        native_checkpoint = None
        status.content = "Generating privately; the last complete candidate remains visible."

        def work():
            try:
                def native_ready(native_result):
                    saved = checkpoint_native_result(native_result, native_checkpoint_path)
                    events.put(("native_saved", saved))

                result = run_assisted_terrain_commands(
                    scene, text, actor_ids=("actor_1",), actor_id="actor_1",
                    initial_placements=placement,
                    client=client, cancelled=cancellation.is_set,
                    on_native_ready=native_ready)
                if cancellation.is_set():
                    events.put(("cancelled", None))
                else:
                    events.put(("ready", result))
            except Exception as exc:
                events.put(("error", str(exc)))

        threading.Thread(target=work, name="terrain-assisted-generate", daemon=True).start()

    @cancel.on_click
    def cancel_clicked(_):
        if busy:
            cancellation.set()
            status.content = "Cancelling private generation; last complete candidate retained."

    @play.on_click
    def play_clicked(_):
        nonlocal playing, last_tick
        if current is not None:
            playing = True
            last_tick = time.monotonic()

    @pause.on_click
    def pause_clicked(_):
        nonlocal playing
        playing = False

    @seek.on_update
    def seek_changed(event):
        nonlocal playing
        if current is not None and event.client is not None and not suppress_seek:
            playing = False
            show_frame(int(seek.value))

    scene_frame(current, 0)
    if current is not None:
        install(current)
    print(f"Terrain-assisted traversal: http://127.0.0.1:{server.get_port()}", flush=True)
    try:
        while True:
            try:
                kind, payload = events.get_nowait()
            except queue.Empty:
                kind = None
            if kind is not None:
                if kind == "native_saved":
                    native_checkpoint = payload
                    status.content = ("Native route checkpoint saved at " + _safe(payload) +
                                      "; solving the assisted mesh poses.")
                else:
                    busy = False
                if kind == "ready" and not cancellation.is_set():
                    try:
                        install(payload)
                        output.parent.mkdir(parents=True, exist_ok=True)
                        save_assisted_result(payload, output)
                        status.content += " · paired archive saved at " + _safe(output)
                    except Exception as exc:
                        status.content = "Candidate display/save failed: " + _safe(exc)
                elif kind == "cancelled" or cancellation.is_set():
                    status.content = "Generation cancelled; last complete candidate retained."
                    if native_checkpoint is not None:
                        status.content += " Native checkpoint: " + _safe(native_checkpoint)
                elif kind == "error":
                    status.content = "Generation failed: " + _safe(payload)
                    if native_checkpoint is not None:
                        status.content += " Native checkpoint: " + _safe(native_checkpoint)
            now = time.monotonic()
            if playing and current is not None and now - last_tick >= 1 / 20:
                steps = max(1, int((now - last_tick) * 20))
                last_tick += steps / 20
                target = frame + steps
                if target >= current.presentation.frames:
                    target = current.presentation.frames - 1
                    playing = False
                show_frame(target)
            time.sleep(.02)
    except KeyboardInterrupt:
        cancellation.set()
    finally:
        renderer.remove()
        server.stop()


if __name__ == "__main__":
    main()
