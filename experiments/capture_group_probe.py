"""Offline browser capture of an independent native22 research group.

Input is a research NPZ containing joints[T,N,22,3] at 30 fps. This script
does not decode a production cast archive or demonstrate real-time Studio
support for ten actors. Open the printed URL and click Start group capture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.capture_core_performance import get_render_with_timeout  # noqa: E402
from experiments.native_pair_rig import NativeRigActor, NativeRigAsset  # noqa: E402
from object_scene import ObjectSceneLayer  # noqa: E402
from scene_composition import validate_scene  # noqa: E402
from scene_ground import has_authored_ground  # noqa: E402
from studio_server import create_studio_server  # noqa: E402

DEFAULT_CLIENT = Path(os.environ.get('STAGEZERO_CLIENT_BUILD', ROOT / 'studio_client' / 'build'))
DEFAULT_RIG = ROOT / 'assets/paired/Xbot.glb'
COLORS = ((240, 111, 92), (78, 178, 230), (248, 195, 75), (165, 122, 230),
          (100, 204, 158), (241, 135, 191), (115, 149, 245), (228, 166, 98),
          (173, 218, 102), (203, 150, 221))
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_input(path: Path) -> np.ndarray:
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError('Research archive exceeds 64 MiB')
    with np.load(path, allow_pickle=False) as data:
        if 'joints' not in data.files:
            raise ValueError('Research NPZ must contain joints')
        joints = data['joints']
        if 'fps' in data.files and (data['fps'].size != 1 or float(data['fps'].item()) != 30):
            raise ValueError('Research NPZ fps must be 30')
    if (joints.ndim != 4 or joints.shape[2:] != (22, 3)
            or not 4 <= joints.shape[0] <= 300 or not 1 <= joints.shape[1] <= 10
            or joints.dtype.kind != 'f' or not np.isfinite(joints).all()):
        raise ValueError('Expected finite float joints[T,N,22,3], T=4..300, N=1..10')
    return np.ascontiguousarray(joints)


def camera_for(joints: np.ndarray, position, look_at):
    if (position is None) != (look_at is None):
        raise ValueError('--camera-position and --camera-look-at must be supplied together')
    if position is not None:
        if not all(math.isfinite(x) for x in (*position, *look_at)) or np.linalg.norm(np.subtract(position, look_at)) < .1:
            raise ValueError('Camera vectors must be finite and separated')
        return tuple(position), tuple(look_at), math.radians(42), 'explicit'
    # Fit all source poses, including the arms and height, in a fixed 16:9 view.
    low = joints.min(axis=(0, 1, 2)) - .45
    high = joints.max(axis=(0, 1, 2)) + .45
    center = (low + high) / 2
    extent = high - low
    fov = math.radians(42)
    distance = max(5., float(extent[1] / (2 * math.tan(fov / 2))),
                   float(extent[0] / (2 * math.tan(fov / 2) * (16 / 9)))) * 1.3 + extent[2] / 2
    position = center + np.array([distance * .22, distance * .32, distance])
    return tuple(position.tolist()), tuple(center.tolist()), fov, 'full_motion_bounds'


def scene_state(scene: dict, seconds: float) -> dict:
    return {'objects': [{'id': obj['id'], 'position': obj['position'],
                         'color': obj['color'], 'active': False}
                        for obj in scene['objects']],
            'assets': scene.get('assets', []), 'effects': scene['effects'],
            'targets': scene.get('targets', []), 'lighting': scene['lighting'],
            'seconds': seconds}


def write_sheet(samples, output: Path) -> None:
    sheet = Image.new('RGB', (1600, 253 * math.ceil(len(samples) / 4)), (18, 24, 31))
    draw = ImageDraw.Draw(sheet)
    for n, (frame, picture) in enumerate(samples):
        x, y = (n % 4) * 400, (n // 4) * 253
        sheet.paste(picture, (x, y + 28))
        draw.text((x + 8, y + 7), f'Frame {frame} · {frame / 30:.2f}s', fill='white')
    sheet.save(output)


def capture(args) -> None:
    archive, scene_path, output = args.archive.resolve(), args.scene.resolve(), args.output.resolve()
    if not archive.is_file() or archive.suffix != '.npz':
        raise ValueError('Input must be an existing .npz research archive')
    if not scene_path.is_file() or scene_path.stat().st_size > 1_000_000:
        raise ValueError('Scene JSON is missing or exceeds 1 MiB')
    if not (args.client_build / 'index.html').is_file():
        raise FileNotFoundError(f'Built client is missing: {args.client_build}')
    if output.exists():
        raise FileExistsError(f'Output directory must be new: {output}')
    joints = read_input(archive)
    scene = validate_scene(json.loads(scene_path.read_text()))
    camera_position, camera_target, fov, camera_mode = camera_for(joints, args.camera_position, args.camera_look_at)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        'status': 'preparing', 'purpose': 'offline CPU browser capture of independent research tracks',
        'real_time_studio_support_verified': False, 'physical_contact_verified': False,
        'archive_sha256': file_hash(archive), 'scene_sha256': file_hash(scene_path),
        'frames': int(joints.shape[0]), 'actor_count': int(joints.shape[1]), 'fps': 30,
        'capture_size': [1280, 720], 'camera': {'mode': camera_mode,
            'position': camera_position, 'look_at': camera_target, 'fov_radians': fov},
        'source_archive': str(archive), 'source_scene': str(scene_path),
        'client_build': str(args.client_build.resolve()), 'captured_frames': 0,
    }
    manifest_path = output / 'capture-manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    shutil.copyfile(archive, output / 'source-research.npz')
    shutil.copyfile(scene_path, output / 'source-scene.json')
    os.environ['STAGEZERO_CLIENT_BUILD'] = str(args.client_build.resolve())
    server = None
    actors = []
    encoder = None
    try:
        asset = NativeRigAsset(args.rig_asset)
        manifest['rig_sha256'] = asset.sha256
        server = create_studio_server(host='127.0.0.1', port=args.port,
                                      label='StageZero · group research capture',
                                      enable_camera_keyboard_controls=False)
        if server.get_port() != args.port:
            raise OSError(f'Requested port {args.port} is occupied')
        server.gui.configure_theme(dark_mode=True, control_layout='floating',
                                   show_logo=False, show_share_button=False)
        server.scene.set_up_direction('+y')
        server.scene.world_axes.visible = False
        server.scene.configure_default_lights(enabled=True, cast_shadow=True)
        server.scene.add_light_ambient('/fill', color=(191, 215, 239), intensity=.6)
        floor = server.scene.add_box('/floor', color=(25, 33, 43),
                                     dimensions=(200, .1, 200), position=(0, -.07, 0),
                                     cast_shadow=False)
        floor.visible = not has_authored_ground(scene['objects'])
        layer = ObjectSceneLayer(server)
        layer.update(scene['objects'], scene_state(scene, 0.))
        for index in range(joints.shape[1]):
            actor = NativeRigActor(server, f'/group-probe/actor-{index:02d}', asset, COLORS[index])
            actors.append(actor)
            actor.prepare_clip(joints[:, index])
            actor.set_frame(0)
        selected, clicked = {}, threading.Event()
        button = server.gui.add_button('Start group capture')

        @button.on_click
        def select(event):
            if event.client is not None and not clicked.is_set():
                selected['client'] = event.client
                button.disabled = True
                clicked.set()

        print(f'CONNECT BROWSER, THEN CLICK "Start group capture": http://127.0.0.1:{args.port}', flush=True)
        if not clicked.wait(args.wait_seconds):
            raise TimeoutError('No browser selected the group capture tab in time')
        client = selected['client']
        client.camera.up_direction = (0, 1, 0)
        client.camera.near, client.camera.far = .05, 400.
        client.camera.position = camera_position
        client.camera.look_at = camera_target
        client.camera.fov = fov
        server.flush()
        if args.settle_seconds:
            time.sleep(args.settle_seconds)
        video = output / 'playback.mp4'
        encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x720', '-r', '30',
            '-i', '-', '-an', '-c:v', 'libx264', '-crf', '19', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE)
        indices = set(np.linspace(0, len(joints)-1, min(12, len(joints))).round().astype(int))
        samples, hashes = [], []
        for frame in range(len(joints)):
            for actor in actors:
                actor.set_frame(frame)
            layer.update(scene['objects'], scene_state(scene, frame / 30))
            server.flush()
            pixels = get_render_with_timeout(client, width=1280, height=720, timeout=args.render_timeout)
            pixels = np.ascontiguousarray(pixels[:, :, :3], dtype=np.uint8)
            if pixels.shape != (720, 1280, 3):
                raise ValueError(f'Browser returned invalid frame shape {pixels.shape}')
            raw = pixels.tobytes()
            encoder.stdin.write(raw)
            hashes.append(hashlib.sha256(raw).hexdigest())
            if frame == 0:
                Image.fromarray(pixels).save(output / 'first-frame.png')
            if frame == len(joints) - 1:
                Image.fromarray(pixels).save(output / 'last-frame.png')
            if frame in indices:
                samples.append((frame, Image.fromarray(pixels).resize((400, 225))))
            manifest['captured_frames'] = frame + 1
            if frame == 0 or (frame + 1) % 30 == 0 or frame + 1 == len(joints):
                print(f'CAPTURED {frame + 1}/{len(joints)}', flush=True)
        encoder.stdin.close()
        if encoder.wait(timeout=30) != 0:
            raise RuntimeError('ffmpeg failed to encode group capture')
        write_sheet(samples, output / 'contact-sheet.png')
        manifest.update(status='complete', frame_indices=list(range(len(joints))),
                        contact_sheet_frames=[int(i) for i in sorted(indices)], raw_rgb_frame_sha256=hashes,
                        video_sha256=file_hash(video),
                        first_frame_sha256=file_hash(output / 'first-frame.png'),
                        last_frame_sha256=file_hash(output / 'last-frame.png'),
                        contact_sheet_sha256=file_hash(output / 'contact-sheet.png'),
                        completed_at=time.time())
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'CAPTURE COMPLETE: {video}', flush=True)
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}', failed_at=time.time())
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        raise
    finally:
        if encoder is not None:
            if encoder.stdin is not None and not encoder.stdin.closed:
                encoder.stdin.close()
            if encoder.poll() is None:
                encoder.terminate()
                try:
                    encoder.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    encoder.kill()
                    encoder.wait(timeout=10)
        for actor in actors:
            actor.remove()
        if server is not None:
            server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True, help='Research NPZ with joints[T,N,22,3]')
    parser.add_argument('--scene', type=Path, required=True, help='Validated Studio scene JSON')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--port', type=int, default=24975)
    parser.add_argument('--client-build', type=Path, default=DEFAULT_CLIENT)
    parser.add_argument('--rig-asset', type=Path, default=DEFAULT_RIG)
    parser.add_argument('--camera-position', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--camera-look-at', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--wait-seconds', type=float, default=180.)
    parser.add_argument('--render-timeout', type=float, default=30.)
    parser.add_argument('--settle-seconds', type=float, default=.5)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be in 1–65535')
    if any(not math.isfinite(v) or v <= 0 for v in (args.wait_seconds, args.render_timeout)):
        parser.error('Wait and render timeouts must be positive and finite')
    if not math.isfinite(args.settle_seconds) or not 0 <= args.settle_seconds <= 30:
        parser.error('--settle-seconds must be in 0–30')
    capture(args)


if __name__ == '__main__':
    main()
