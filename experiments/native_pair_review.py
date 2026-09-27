"""Inspect exact native paired 22-joint output at its source 30 fps.

The solid mannequin is geometry drawn around the original joints, not a fitted
human mesh or a Core retarget. The Raw bones mode exposes every source endpoint.
Optional scene JSON is a visual backdrop; it did not condition generation.

Example::

    python experiments/native_pair_review.py \
      --input .runtime/intergen-lab/handshake_seed42.npz --port 2373 \
      --capture-dir /private/tmp/native-pair-review

Open the printed URL in a browser and click "Capture all 30 fps frames". The
capture writes one WebGL render for every source frame, in source order.
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

FPS = 30
CHAINS = ((0, 2, 5, 8, 11), (0, 1, 4, 7, 10), (0, 3, 6, 9, 12, 15),
          (9, 14, 17, 19, 21), (9, 13, 16, 18, 20))
EDGES = np.asarray([(a, b) for chain in CHAINS for a, b in zip(chain, chain[1:])], dtype=np.int32)
COLORS = ((52, 209, 220), (250, 178, 78))
LIMBS = ((0, 1, .105), (1, 4, .09), (4, 7, .07), (7, 10, .055),
         (0, 2, .105), (2, 5, .09), (5, 8, .07), (8, 11, .055),
         (9, 13, .075), (13, 16, .07), (16, 18, .055), (18, 20, .043),
         (9, 14, .075), (14, 17, .07), (17, 19, .055), (19, 21, .043),
         (12, 15, .068))


def load_native(path: Path) -> tuple[np.ndarray, dict]:
    """Read only unsmoothed source joints; never substitute fitted coordinates."""
    with np.load(path, allow_pickle=False) as archive:
        joints = np.asarray(archive["joints"], dtype=np.float32).copy()
        raw_meta = archive["metadata"].item() if "metadata" in archive else "{}"
    if joints.ndim != 4 or joints.shape[1:] != (2, 22, 3) or not 1 <= len(joints) <= 30_000:
        raise ValueError("Expected native joints[T,2,22,3] with 1–30000 frames")
    if not np.isfinite(joints).all():
        raise ValueError("Native joints must all be finite")
    meta = json.loads(str(raw_meta))
    if not isinstance(meta, dict):
        raise ValueError("Archive metadata must be an object")
    source_fps = meta.get("fps", FPS)
    if float(source_fps) != FPS:
        raise ValueError(f"Expected native 30 fps source; archive says {source_fps}")
    return joints, meta


def load_native_features(path: Path, frames: int) -> np.ndarray:
    """Read optional native rotation data only for an explicit audited experiment."""
    with np.load(path, allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32).copy()
    if features.shape != (frames, 2, 262) or not np.isfinite(features).all():
        raise ValueError("Native rotation prior needs finite features[T,2,262]")
    return features


def source_model(meta: dict) -> str:
    """Use only known model labels in UI and provenance, never arbitrary metadata."""
    model = meta.get("model")
    return model if model in ("InterGen", "InterMask") else "Native paired model"


def handshake_contact_weights(poses: np.ndarray, contact_joints: tuple[int, int],
                              *, threshold_m: float = .15, ramp_frames: int = 6
                              ) -> tuple[np.ndarray, dict]:
    """Author finger weights only inside the longest measured wrist-proximity run.

    Output is [actor, source frame, left20/right21]. It never changes `poses`.
    """
    raw = np.asarray(poses)
    if raw.ndim != 4 or raw.shape[1:] != (2, 22, 3) or not np.isfinite(raw).all():
        raise ValueError("Expected finite native joints[T,2,22,3]")
    if len(contact_joints) != 2 or any(joint not in (20, 21) for joint in contact_joints):
        raise ValueError("Contact joints must select source wrists 20 or 21 for each actor")
    distances = np.linalg.norm(raw[:, 0, contact_joints[0]] - raw[:, 1, contact_joints[1]], axis=1)
    within = distances < threshold_m
    runs: list[tuple[int, int]] = []
    start = None
    for frame in range(len(within) + 1):
        active = frame < len(within) and bool(within[frame])
        if active and start is None:
            start = frame
        elif not active and start is not None:
            runs.append((start, frame))
            start = None
    if not runs:
        raise ValueError("Selected source wrists never come within 0.15 m; no finger pose was authored")
    first, end = max(runs, key=lambda run: run[1] - run[0])
    if end - first < max(12, 2 * ramp_frames + 1):
        raise ValueError("Measured wrist proximity is too brief for a handshake finger pose")
    weights = np.zeros((2, len(raw), 2), dtype=np.float64)
    for frame in range(first, end):
        entry = min(1., (frame - first) / ramp_frames)
        exit_ = min(1., (end - 1 - frame) / ramp_frames)
        progress = min(entry, exit_)
        eased = progress * progress * (3. - 2. * progress)
        weights[0, frame, contact_joints[0] - 20] = eased
        weights[1, frame, contact_joints[1] - 20] = eased
    report = {
        "hand_pose": "handshake", "source": "authored finger curl from measured source wrist proximity",
        "contact_joints_actor0_actor1": list(contact_joints),
        "distance_threshold_m": threshold_m,
        "longest_run_start_frame": first, "longest_run_end_frame_exclusive": end,
        "ramp_frames_inside_run": ramp_frames,
        "minimum_wrist_distance_m": float(distances[first:end].min()),
        "source_joints_modified": False,
        "learned_finger_animation": False,
        "physical_contact_verified": False,
    }
    return weights, report


def _basis(up: np.ndarray, across: np.ndarray) -> np.ndarray:
    """Right-handed x/y/z basis using observed body landmarks only."""
    y = np.asarray(up, dtype=np.float64)
    y /= max(np.linalg.norm(y), 1e-8)
    x = np.asarray(across, dtype=np.float64) - y * np.dot(across, y)
    if np.linalg.norm(x) < 1e-8:
        reference = np.array([0., 0., 1.]) if abs(y[2]) < .9 else np.array([1., 0., 0.])
        x = np.cross(y, reference)
    x /= max(np.linalg.norm(x), 1e-8)
    z = np.cross(x, y)
    return np.column_stack((x, y, z))


def capsule_between(a: np.ndarray, b: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Rounded limb with its terminal vertices exactly on source endpoints."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    direction = b - a
    length = float(np.linalg.norm(direction))
    axis = direction / max(length, 1e-8)
    reference = np.array([0., 1., 0.]) if abs(axis[1]) < .9 else np.array([0., 0., 1.])
    basis = _basis(axis, np.cross(axis, reference))
    angles = np.arange(12, dtype=np.float64) * (2 * np.pi / 12)
    rings = ((0., 0.), (.035, .7), (.12, 1.), (.88, 1.), (.965, .7), (1., 0.))
    vertices = np.asarray([
        a + direction * t + radius * r * (basis[:, 0] * np.cos(angle) + basis[:, 2] * np.sin(angle))
        for t, r in rings for angle in angles
    ], dtype=np.float32)
    faces = []
    for ring in range(len(rings) - 1):
        for edge in range(len(angles)):
            aa = ring * len(angles) + edge
            bb = ring * len(angles) + (edge + 1) % len(angles)
            cc = aa + len(angles)
            dd = bb + len(angles)
            faces.extend(((aa, cc, bb), (bb, cc, dd)))
    return vertices, np.asarray(faces, dtype=np.uint32)


def ellipsoid(center: np.ndarray, basis: np.ndarray, radii: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    # Low-poly UV surface is intentionally smooth-looking without a textured rig.
    sides, latitudes = 16, 10
    vertices = []
    for latitude in range(latitudes + 1):
        phi = np.pi * latitude / latitudes
        for longitude in range(sides):
            theta = 2 * np.pi * longitude / sides
            local = np.array([np.sin(phi) * np.cos(theta), np.cos(phi),
                              np.sin(phi) * np.sin(theta)]) * radii
            vertices.append(center + basis @ local)
    faces = []
    for latitude in range(latitudes):
        for longitude in range(sides):
            a = latitude * sides + longitude
            b = latitude * sides + (longitude + 1) % sides
            c, d = a + sides, b + sides
            faces.extend(((a, c, b), (b, c, d)))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


def mannequin_mesh(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Articulated display geometry derived separately for each raw source frame."""
    j = np.asarray(joints, dtype=np.float64)
    if j.shape != (22, 3) or not np.isfinite(j).all():
        raise ValueError("Expected finite joints[22,3]")
    shoulder_mid = (j[16] + j[17]) / 2
    hip_mid = (j[1] + j[2]) / 2
    torso_basis = _basis(j[12] - hip_mid, j[17] - j[16])
    pelvis_basis = _basis(j[9] - hip_mid, j[2] - j[1])
    head_basis = _basis(j[15] - j[12], j[17] - j[16])
    torso_height = float(np.linalg.norm(j[12] - hip_mid))
    shoulder_width = float(np.linalg.norm(j[17] - j[16]))
    hip_width = float(np.linalg.norm(j[2] - j[1]))
    pieces = [capsule_between(j[a], j[b], radius) for a, b, radius in LIMBS]
    pieces.extend((
        ellipsoid((j[12] + hip_mid) / 2, torso_basis,
                  (max(.12, shoulder_width * .54), max(.15, torso_height * .49), .115)),
        ellipsoid((j[0] + hip_mid) / 2, pelvis_basis,
                  (max(.105, hip_width * .66), .105, .105)),
        ellipsoid(j[15], head_basis, (.105, .145, .105)),
    ))
    vertices = []
    faces = []
    offset = 0
    for part_vertices, part_faces in pieces:
        vertices.append(part_vertices)
        faces.append(part_faces + offset)
        offset += len(part_vertices)
    return np.concatenate(vertices), np.concatenate(faces)


def get_render_with_timeout(client, *, width: int, height: int, timeout: float) -> np.ndarray:
    result: queue.Queue = queue.Queue(maxsize=1)

    def request() -> None:
        try:
            result.put((client.get_render(height=height, width=width), None))
        except BaseException as exc:
            result.put((None, exc))

    worker = threading.Thread(target=request, daemon=True, name="native-pair-webgl-render")
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError("Browser did not answer the WebGL render request")
    pixels, error = result.get_nowait()
    if error is not None:
        raise error
    return pixels


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_contact_sheet(frames: list[tuple[int, Image.Image]], path: Path) -> None:
    if not frames:
        return
    width, height = frames[0][1].size
    cols = min(4, len(frames))
    sheet = Image.new("RGB", (width * cols, (height + 28) * ((len(frames) + cols - 1) // cols)), (17, 23, 30))
    draw = ImageDraw.Draw(sheet)
    for count, (index, image) in enumerate(frames):
        x, y = count % cols * width, count // cols * (height + 28)
        sheet.paste(image, (x, y + 28))
        draw.text((x + 8, y + 7), f"Source frame {index} · {index / FPS:.2f} s", fill="white")
    sheet.save(path)


def _safe_caption(meta: dict) -> str:
    prompt = str(meta.get("prompt", meta.get("caption", "See source archive")))
    return prompt.replace("<", "&lt;").replace(">", "&gt;").replace("{", "&#123;").replace("}", "&#125;")


def serve(args: argparse.Namespace) -> None:
    import viser
    from object_scene import ObjectSceneLayer
    from scene_composition import validate_scene
    from scene_ground import has_authored_ground

    source = args.input.resolve()
    paths = sorted(source.glob("*.npz")) if source.is_dir() else [source]
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("--input must be a native InterGen .npz file or directory of files")
    if args.take is not None:
        selected = next((path for path in paths if path.name == args.take), None)
        if selected is None:
            raise ValueError(f"--take {args.take!r} is not in --input")
        paths = [selected] + [path for path in paths if path != selected]
    poses, meta = load_native(paths[0])
    scene = None
    if args.scene is not None:
        if args.scene.stat().st_size > 900_000:
            raise ValueError("Scene JSON exceeds 900 KB")
        scene = validate_scene(json.loads(args.scene.read_text()))
    server = viser.ViserServer(host="127.0.0.1", port=args.port,
                              label="Native paired motion · 30 fps")
    if server.get_port() != args.port:
        server.stop()
        raise OSError(f"Requested port {args.port} is occupied")
    try:
        server.gui.configure_theme(dark_mode=True, show_logo=False, show_share_button=False)
        server.scene.set_up_direction("+y")
        server.scene.world_axes.visible = False
        server.scene.configure_default_lights(enabled=True, cast_shadow=True)
        server.scene.add_light_ambient("/native/fill", color=(195, 215, 232), intensity=.75)
        server.scene.add_box("/native/floor", dimensions=(40, .08, 40), position=(0, -.07, 0),
                             color=(25, 33, 43), cast_shadow=False,
                             visible=scene is None or not has_authored_ground(scene["objects"]))
        layer = ObjectSceneLayer(server) if scene is not None else None
        static_states = ([{"id": obj["id"], "position": obj["position"], "color": obj["color"],
                          "active": False} for obj in scene["objects"]] if scene is not None else [])
        title = server.gui.add_markdown("")
        choice = server.gui.add_dropdown("Take", [p.name for p in paths], initial_value=paths[0].name)
        modes = ["Solid mannequin", "Raw bones", "Both"]
        rig_actors = []
        if args.rig_asset is not None:
            from experiments.native_pair_rig import NativeRigActor, NativeRigAsset
            rig_asset = NativeRigAsset(args.rig_asset)
            rig_actors = [NativeRigActor(server, f"/native/actor{actor}/authored-rig",
                                         rig_asset, color) for actor, color in enumerate(COLORS)]
            modes.append("Authored rig")
        mode = server.gui.add_dropdown("Display", modes, initial_value=args.mode)
        caption = server.gui.add_markdown("")
        play = server.gui.add_button("Play")
        pause = server.gui.add_button("Pause")
        reset = server.gui.add_button("Reset view")
        slider = server.gui.add_slider("Source frame", min=0, max=len(poses) - 1, step=1, initial_value=0)
        capture_button = server.gui.add_button("Capture all 30 fps frames") if args.capture_dir else None
        capture_status = server.gui.add_markdown("") if args.capture_dir else None
        lock = threading.RLock()
        state = {"poses": poses, "meta": meta, "path": paths[0], "mode": args.mode, "playing": False,
                 "frame": 0, "started": 0., "capturing": False, "handshake_contact": None}
        handles = []
        for actor, color in enumerate(COLORS):
            line = server.scene.add_line_segments(f"/native/actor{actor}/bones",
                points=poses[0, actor][EDGES], colors=color, line_width=5)
            dots = server.scene.add_point_cloud(f"/native/actor{actor}/joints",
                points=poses[0, actor], colors=color, point_size=.047, point_shape="circle")
            vertices, faces = mannequin_mesh(poses[0, actor])
            solid = server.scene.add_mesh_simple(f"/native/actor{actor}/solid", vertices=vertices,
                faces=faces, color=color, material="toon5", flat_shading=False, cast_shadow=True)
            handles.append((line, dots, solid))

        def prepare_rigs(raw_poses: np.ndarray, path: Path) -> dict | None:
            if not rig_actors:
                return None
            features = (load_native_features(path, len(raw_poses))
                        if args.native_rotation_prior else None)
            finger_weights, contact_report = (handshake_contact_weights(
                raw_poses, tuple(args.contact_joints)) if args.handshake_fingers else (None, None))
            for actor, rig in enumerate(rig_actors):
                rig.prepare_clip(raw_poses[:, actor],
                                 native_features=None if features is None else features[:, actor],
                                 hand_pose="handshake" if args.handshake_fingers else None,
                                 contact_weights=None if finger_weights is None else finger_weights[actor])
            return contact_report

        state["handshake_contact"] = prepare_rigs(poses, paths[0])

        def visibility() -> None:
            for line, dots, solid in handles:
                line.visible = dots.visible = mode.value in ("Raw bones", "Both")
                solid.visible = mode.value in ("Solid mannequin", "Both")
            for rig in rig_actors:
                rig.set_visible(mode.value == "Authored rig")

        def show(frame: int) -> None:
            state["frame"] = frame
            for actor, (line, dots, solid) in enumerate(handles):
                raw = state["poses"][frame, actor]
                line.points = raw[EDGES]
                dots.points = raw
                if solid.visible:
                    solid.vertices = mannequin_mesh(raw)[0]
                if rig_actors and rig_actors[actor].visible:
                    rig_actors[actor].set_frame(frame)
            slider.value = frame
            if layer is not None:
                layer.update(scene["objects"], {"objects": static_states,
                    "assets": scene.get("assets", []), "effects": scene["effects"],
                    "lighting": scene["lighting"], "seconds": frame / FPS})

        def describe() -> None:
            model = source_model(state["meta"])
            title.content = f"# {model} paired motion\n30 fps native 22-joint source · research diagnostic"
            display_note = {
                "Solid mannequin": "Solid mannequin volumes are display geometry around those joints.",
                "Raw bones": "Raw bones expose the original joint endpoints and connections.",
                "Both": "Raw bones expose exact endpoints overlaid on display-only mannequin volumes.",
                "Authored rig": "Authored Xbot skin follows the original joint endpoints. Its visible mesh surface and contact are experimental.",
            }[mode.value]
            caption.content = (f"**Prompt:** {_safe_caption(state['meta'])}\n\n"
                "**Source:** exact unsmoothed 22 joints per actor, 30 fps, shared native coordinates. "
                "No Core27 conversion, smoothing, pair offsets, or inferred contact. "
                f"{display_note} "
                "Scene, if loaded, is only a backdrop."
                + (" Explicit authored finger curls follow a measured source-wrist proximity interval; "
                   "they are not model-generated finger animation or a contact solver."
                   if state["handshake_contact"] else ""))

        def camera(client) -> None:
            points = state["poses"].reshape(-1, 3)
            center = (points.min(axis=0) + points.max(axis=0)) / 2
            client.camera.position = tuple(args.camera_position or (center + np.array([3.5, 1.6, .55])))
            client.camera.look_at = tuple(args.look_at or center)
            client.camera.up_direction = (0, 1, 0)
            client.camera.fov = np.deg2rad(args.fov)

        @server.on_client_connect
        def connected(client):
            camera(client)

        @reset.on_click
        def reset_view(_):
            for client in server.get_clients().values():
                camera(client)

        @mode.on_update
        def change_mode(_):
            state["mode"] = mode.value
            visibility()
            show(state["frame"])
            describe()

        @slider.on_update
        def scrub(event):
            if event.client is None or state["capturing"]:
                return
            with lock:
                show(int(slider.value))
                state["started"] = time.perf_counter() - state["frame"] / FPS

        @play.on_click
        def begin(_):
            with lock:
                if state["capturing"]:
                    return
                if state["frame"] >= len(state["poses"]) - 1:
                    show(0)
                state["playing"] = True
                state["started"] = time.perf_counter() - state["frame"] / FPS

        @pause.on_click
        def stop(_):
            state["playing"] = False

        @choice.on_update
        def change(_):
            if state["capturing"]:
                return
            with lock:
                state["playing"] = False
                path = next(path for path in paths if path.name == choice.value)
                if path == state["path"]:
                    return
                next_poses, next_meta = load_native(path)
                try:
                    next_contact = prepare_rigs(next_poses, path)
                except Exception as exc:
                    prepare_rigs(state["poses"], state["path"])
                    choice.value = state["path"].name
                    caption.content = f"Cannot use that take in this display mode: `{type(exc).__name__}: {exc}`"
                    return
                state["poses"], state["meta"] = next_poses, next_meta
                state["handshake_contact"] = next_contact
                state["path"] = path
                slider.max = len(state["poses"]) - 1
                show(0)
                describe()

        if args.mode not in modes:
            raise ValueError("Authored rig mode requires --rig-asset")
        visibility()
        describe()
        show(0)
        print(f"Native paired viewer http://127.0.0.1:{args.port}/", flush=True)

        if capture_button is not None:
            @capture_button.on_click
            def request_capture(event):
                if event.client is None or state["capturing"]:
                    return
                state["capturing"] = True
                state["playing"] = False
                capture_button.disabled = True
                choice.disabled = True
                mode.disabled = True
                capture_status.content = "Capturing every native source frame at 30 fps…"

                def run_capture() -> None:
                    try:
                        outcome = _capture(event.client, args, state, server, show, rig_actors)
                        capture_status.content = outcome
                    finally:
                        state["capturing"] = False
                        capture_button.disabled = False
                        choice.disabled = False
                        mode.disabled = False

                thread = threading.Thread(target=run_capture,
                                          daemon=True, name="native-pair-capture")
                thread.start()

        while True:
            with lock:
                if state["playing"] and not state["capturing"]:
                    frame = min(int((time.perf_counter() - state["started"]) * FPS), len(state["poses"]) - 1)
                    if frame != state["frame"]:
                        show(frame)
                    if frame == len(state["poses"]) - 1:
                        state["playing"] = False
            time.sleep(1 / 60)
    finally:
        server.stop()


def _capture(client, args, state, server, show, rig_actors=()) -> str:
    """Encode all source frames, one synchronous WebGL render per frame."""
    output = args.capture_dir.resolve()
    if getattr(args, "capture_per_take", False):
        output = output / (state["path"].stem + "-" + state["mode"].lower().replace(" ", "-"))
    output.mkdir(parents=True, exist_ok=True)
    video = output / "native-pair.mp4"
    first = output / "first-frame.png"
    last = output / "last-frame.png"
    sheet = output / "contact-sheet.png"
    manifest_path = output / "capture-manifest.json"
    failure_path = output / "capture-failure.json"
    existing = [path for path in (video, first, last, sheet, manifest_path, failure_path) if path.exists()]
    if existing:
        print(f"CAPTURE FAILED: Output exists: {existing[0]}", flush=True)
        return f"Capture was not started: output already exists at `{existing[0]}`. Choose another --capture-dir."
    poses = state["poses"]
    path = state["path"]
    count = len(poses)
    camera = {"position": list(client.camera.position), "look_at": list(client.camera.look_at),
              "fov_degrees": float(np.rad2deg(client.camera.fov))}
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{args.width}x{args.height}", "-r", str(FPS), "-i", "-", "-an",
               "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
               "-frames:v", str(count), "-movflags", "+faststart", str(video)]
    thumbnails = []
    selected = {round(k * (count - 1) / (min(12, count) - 1)) for k in range(min(12, count))} if count > 1 else {0}
    hashes = []
    times = []
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        time.sleep(args.settle_seconds)
        for frame in range(count):
            show(frame)
            server.flush()
            started = time.perf_counter()
            pixels = get_render_with_timeout(client, width=args.width, height=args.height,
                                             timeout=args.render_timeout)
            times.append((time.perf_counter() - started) * 1000)
            rgb = np.ascontiguousarray(pixels[:, :, :3], dtype=np.uint8)
            if rgb.shape != (args.height, args.width, 3):
                raise ValueError(f"Wrong browser render shape: {rgb.shape}")
            raw = rgb.tobytes()
            hashes.append(hashlib.sha256(raw).hexdigest())
            process.stdin.write(raw)
            if frame == 0:
                Image.fromarray(rgb).save(first)
            if frame == count - 1:
                Image.fromarray(rgb).save(last)
            if frame in selected:
                thumbnails.append((frame, Image.fromarray(rgb).resize((400, 225))))
            if frame % 30 == 0 or frame == count - 1:
                print(f"CAPTURED {frame + 1}/{count}", flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed")
        write_contact_sheet(thumbnails, sheet)
        source_bytes = path.read_bytes()
        model = source_model(state["meta"])
        manifest = {
            "capture_kind": f"{model} raw 22-joint diagnostic WebGL replay",
            "source_model": model,
            "source_archive": str(path), "source_archive_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "source_joints_sha256": hashlib.sha256(np.ascontiguousarray(poses).tobytes()).hexdigest(),
            "source_array": "joints", "smoothed_joints_used": False,
            "frames": count, "fps": FPS, "duration_seconds": count / FPS,
            "frame_indices": list(range(count)), "raw_rgb_frame_sha256": hashes,
            "display_mode": state["mode"], "rendered_scene_backdrop": str(args.scene.resolve()) if args.scene else None,
            "authored_rig": [rig.provenance for rig in rig_actors] if rig_actors else None,
            "authored_rig_metrics_last_frame": [rig.metrics for rig in rig_actors] if rig_actors else None,
            "native_rotation_prior_requested": bool(args.native_rotation_prior),
            "authored_handshake_fingers_requested": bool(args.handshake_fingers),
            "authored_handshake_contact": state["handshake_contact"],
            "scene_conditioned": False, "retargeted": False, "pair_offsets_added": False,
            "motion_smoothed": False, "physical_contact_verified": False,
            "capture_camera": camera, "capture_size": [args.width, args.height],
            "render_roundtrip_median_ms": round(statistics.median(times), 2),
            "render_roundtrip_scope": "Browser WebGL get_render roundtrip; not playback FPS",
            "video_sha256": sha256_file(video), "contact_sheet_sha256": sha256_file(sheet),
            "first_frame_sha256": sha256_file(first), "last_frame_sha256": sha256_file(last),
            "note": ("One render per exact source frame in order. Mannequin is display geometry around raw joints; "
                     "the authored rig and optional finger pose are display experiments, not native Core animation."),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"CAPTURE COMPLETE: {video}", flush=True)
        return f"Capture complete: {count} exact source frames at 30 fps. Video: `{video}`"
    except BaseException as exc:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
        process.wait()
        failure_path.write_text(json.dumps({
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            "source_archive": str(path), "source_frames": count, "fps": FPS,
            "captured_frame_count": len(hashes), "captured_frame_indices": list(range(len(hashes))),
            "partial_artifacts_preserved": [str(item) for item in (video, first, last, sheet)
                                            if item.exists()],
        }, indent=2) + "\n")
        print(f"CAPTURE FAILED: {exc}", flush=True)
        return f"Capture failed after {len(hashes)} frames: {type(exc).__name__}. See `{failure_path}`."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Native raw paired .npz or directory, using unsmoothed joints key")
    parser.add_argument("--take", help="Initial .npz filename when --input is a directory")
    parser.add_argument("--scene", type=Path, help="Optional visual scene JSON backdrop")
    parser.add_argument("--mode", choices=("Solid mannequin", "Raw bones", "Both", "Authored rig"),
                        default="Solid mannequin")
    parser.add_argument("--rig-asset", type=Path,
                        help="Optional authored Xbot GLB experiment; enables Authored rig display")
    parser.add_argument("--native-rotation-prior", action="store_true",
                        help="Audit native features and use only eligible arm/hand twist as a rig experiment")
    parser.add_argument("--handshake-fingers", action="store_true",
                        help="Explicit authored finger pose during measured wrist proximity")
    parser.add_argument("--contact-joints", type=int, nargs=2, metavar=("ACTOR0", "ACTOR1"),
                        help="Source wrist IDs (20=left, 21=right) used with --handshake-fingers")
    parser.add_argument("--port", type=int, default=2373)
    parser.add_argument("--capture-dir", type=Path, help="Write exact full 30 fps MP4 after browser button click")
    parser.add_argument("--capture-per-take", action="store_true", help="Keep separate capture folders for each take/display mode")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--settle-seconds", type=float, default=3.)
    parser.add_argument("--render-timeout", type=float, default=45.)
    parser.add_argument("--camera-position", type=float, nargs=3)
    parser.add_argument("--look-at", type=float, nargs=3)
    parser.add_argument("--fov", type=float, default=48.)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be 1–65535")
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        parser.error("Capture width/height must be positive even numbers")
    if not 0 <= args.settle_seconds <= 60 or not 1 <= args.render_timeout <= 120:
        parser.error("Capture settle/render timeouts are out of range")
    if not 1 <= args.fov <= 120:
        parser.error("--fov must be 1–120 degrees")
    if args.scene is not None and not args.scene.is_file():
        parser.error("--scene must be an existing JSON file")
    if args.rig_asset is not None and not args.rig_asset.is_file():
        parser.error("--rig-asset must be an existing GLB file")
    if args.mode == "Authored rig" and args.rig_asset is None:
        parser.error("--mode 'Authored rig' requires --rig-asset")
    if args.native_rotation_prior and args.rig_asset is None:
        parser.error("--native-rotation-prior requires --rig-asset")
    if args.handshake_fingers and (args.rig_asset is None or args.contact_joints is None):
        parser.error("--handshake-fingers requires --rig-asset and explicit --contact-joints 20|21 20|21")
    if args.contact_joints is not None and not args.handshake_fingers:
        parser.error("--contact-joints requires --handshake-fingers")
    if args.contact_joints is not None and any(joint not in (20, 21) for joint in args.contact_joints):
        parser.error("--contact-joints must contain only wrist IDs 20 or 21")
    serve(args)


if __name__ == "__main__":
    main()
