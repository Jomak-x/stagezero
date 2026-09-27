"""Capture complete saved cast performances with the StageZero browser client.

The JSON manifest is a list of {"archive": "...npz", "output_dir": "..."}
objects. Paths are resolved relative to the manifest. An optional "camera"
value of "closer" uses --closer-factor for that case; the default is the
full-take prompt-scene camera. No model or live Studio service is used.

Example:
    python experiments/capture_cast_review.py --manifest review/cast-cases.json \
        --port 24971 --wait-seconds 120

Open the printed URL in a browser and click "Start cast batch capture". The
same tab is used for every case, with a fresh local-playback acknowledgement.
Each case gets every source frame, an MP4, contact sheet, first/last frames,
the saved cast archive, and capture provenance. Failed cases retain their
partial output and a failure report; the batch exits nonzero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cast_performance_session import CastPerformanceSession  # noqa: E402
from cast_performance_renderer import CastPerformanceRenderer  # noqa: E402
from native_pair_capture import capture_pair  # noqa: E402
from native_pair_playback import NativePairPlaybackController  # noqa: E402
from object_scene import ObjectSceneLayer  # noqa: E402
from prompt_scene_camera import prompt_scene_camera_view  # noqa: E402
from scene_ground import has_authored_ground  # noqa: E402
from studio_server import create_studio_server  # noqa: E402

DEFAULT_CLIENT = Path(os.environ.get('STAGEZERO_CLIENT_BUILD', ROOT / 'studio_client' / 'build'))
ARCHIVE_LIMIT = 64 * 1024 * 1024


def read_cases(manifest: Path, output_root: Path | None = None) -> list[dict]:
    raw = json.loads(manifest.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError('Manifest must be a nonempty JSON list')
    cases = []
    outputs = set()
    for number, item in enumerate(raw, 1):
        if not isinstance(item, dict) or not isinstance(item.get('archive'), str) or not isinstance(item.get('output_dir'), str):
            raise ValueError(f'Case {number} needs archive and output_dir strings')
        if item.get('camera', 'full') not in ('full', 'closer'):
            raise ValueError(f'Case {number} camera must be full or closer')
        archive = (manifest.parent / item['archive']).resolve()
        output = (manifest.parent / item['output_dir']).resolve()
        if output_root is not None:
            output = output_root.resolve() / output.name
        if not archive.is_file() or archive.suffix != '.npz':
            raise FileNotFoundError(f'Case {number} archive is missing: {archive}')
        if archive.stat().st_size > ARCHIVE_LIMIT:
            raise ValueError(f'Case {number} archive exceeds 64 MiB')
        if output in outputs or output.exists():
            raise FileExistsError(f'Case {number} output must be new and unique: {output}')
        outputs.add(output)
        cases.append({'archive': archive, 'output': output, 'camera': item.get('camera', 'full')})
    return cases


def scene_states(scene: dict, seconds: float) -> dict:
    objects = scene.get('objects', [])
    return {'objects': [{'id': obj['id'], 'position': obj['position'],
                         'color': obj['color'], 'active': False} for obj in objects],
            'assets': scene.get('assets', []), 'effects': scene.get('effects', []),
            'targets': scene.get('targets', []),
            'lighting': scene.get('lighting', 'neutral'), 'seconds': seconds}


def set_camera(client, clip, scene: dict, mode: str, closer_factor: float) -> dict:
    position, center, fov = prompt_scene_camera_view(
        clip, scene, frame=0, aspect=getattr(client.camera, 'aspect', 16 / 9))
    if mode == 'closer':
        # An optional detail view. It can crop wide movements by design; the
        # standard full-take view remains the default and is always available.
        position = center + (position - center) * closer_factor
    client.camera.up_direction = (0, 1, 0)
    client.camera.near, client.camera.far = .05, 400.
    client.camera.position = tuple(float(v) for v in position)
    client.camera.look_at = tuple(float(v) for v in center)
    client.camera.fov = float(fov)
    return {'mode': mode, 'position': list(map(float, position)),
            'look_at': list(map(float, center)), 'fov_radians': float(fov)}


def run(args: argparse.Namespace) -> None:
    manifest = args.manifest.resolve()
    cases = read_cases(manifest, args.output_root)
    client_build = args.client_build.resolve()
    if not (client_build / 'index.html').is_file():
        raise FileNotFoundError(f'Built StageZero client is missing: {client_build}')
    os.environ['STAGEZERO_CLIENT_BUILD'] = str(client_build)

    session = CastPerformanceSession()
    server = create_studio_server(host='127.0.0.1', port=args.port,
                                  label='StageZero · cast review capture',
                                  enable_camera_keyboard_controls=False)
    if server.get_port() != args.port:
        server.stop()
        raise OSError(f'Requested port {args.port} is occupied')
    renderer = None
    try:
        server.gui.configure_theme(dark_mode=True, control_layout='floating',
                                   show_logo=False, show_share_button=False)
        server.scene.set_up_direction('+y')
        server.scene.world_axes.visible = False
        server.scene.configure_default_lights(enabled=True, cast_shadow=True)
        server.scene.add_light_ambient('/fill', color=(191, 215, 239), intensity=.6)
        floor = server.scene.add_box('/floor', color=(25, 33, 43),
                                     dimensions=(200, .1, 200), position=(0, -.07, 0),
                                     cast_shadow=False)
        layer = ObjectSceneLayer(server)
        renderer = CastPerformanceRenderer(server)
        playback = NativePairPlaybackController(
            server, get_state=lambda: dict(session.snapshot(), enabled=True))
        renderer.local_playback = playback
        selected = {}
        clicked = threading.Event()
        button = server.gui.add_button('Start cast batch capture')

        @button.on_click
        def choose_client(event):
            if event.client is not None and not clicked.is_set():
                selected['client'] = event.client
                button.disabled = True
                clicked.set()

        print(f'CONNECT BROWSER, THEN CLICK "Start cast batch capture": http://127.0.0.1:{args.port}', flush=True)
        if not clicked.wait(args.wait_seconds):
            raise TimeoutError('No browser selected the cast capture tab before --wait-seconds elapsed')
        client = selected['client']
        report = {'manifest': str(manifest), 'port': args.port,
                  'client_build': str(client_build), 'cases': []}
        failures = 0
        for number, case in enumerate(cases, 1):
            archive = case['archive']
            output = case['output']
            record = {'archive': str(archive), 'output_dir': str(output),
                      'input_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                      'camera_mode': case['camera']}
            print(f'[{number}/{len(cases)}] Loading {archive}', flush=True)
            try:
                session.load(archive.read_bytes())
                clip = session.timeline_clip()
                scene = session.scene_document
                objects = scene.get('objects', [])
                layer.update(objects, scene_states(scene, 0.))
                floor.visible = not has_authored_ground(objects)
                renderer.sync_cast(session.snapshot())
                renderer.set_clip(clip)
                renderer.set_visible(True)
                playback.update(session.snapshot(), enabled=True)
                record['camera'] = set_camera(client, clip, scene, case['camera'], args.closer_factor)
                server.flush()
                # A fresh clip must be acknowledged by this exact browser tab.
                # capture_pair also checks readiness after acquiring its lease.
                playback.require_ready(client, timeout=args.ready_seconds)
                if args.settle_seconds:
                    time.sleep(args.settle_seconds)

                def render_frame(frame, state):
                    layer.update(objects, scene_states(scene, frame / state['fps']))
                    if frame == 0 or (frame + 1) % 30 == 0 or frame + 1 == state['total_frames']:
                        print(f'[{number}/{len(cases)}] frame {frame + 1}/{state["total_frames"]}', flush=True)

                print(f'[{number}/{len(cases)}] Capturing all {clip.frames} frames to {output}', flush=True)
                video = capture_pair(session, renderer, client, output,
                                     flush=server.flush, render_frame=render_frame,
                                     archive_name='scene.cast.stagezero.npz')
                record.update(status='complete', frames=clip.frames, fps=clip.fps,
                              video=str(video))
                (output / 'source-provenance.json').write_text(json.dumps(record, indent=2) + '\n')
                print(f'[{number}/{len(cases)}] Complete: {video}', flush=True)
            except Exception as exc:
                failures += 1
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                output.mkdir(parents=True, exist_ok=True)
                (output / 'source-provenance.json').write_text(json.dumps(record, indent=2) + '\n')
                print(f'[{number}/{len(cases)}] FAILED: {record["error"]}', file=sys.stderr, flush=True)
            report['cases'].append(record)
        print(json.dumps(report, indent=2), flush=True)
        if failures:
            raise RuntimeError(f'{failures} of {len(cases)} captures failed; see each output/source-provenance.json')
    finally:
        if renderer is not None:
            renderer.remove()
        session.close()
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True,
                        help='JSON list of archive/output_dir cases; paths relative to manifest')
    parser.add_argument('--output-root', type=Path, help='New directory for repeat capture without replacing evidence')
    parser.add_argument('--port', type=int, default=24971, help='Dedicated local preview port (default 24971)')
    parser.add_argument('--client-build', type=Path, default=DEFAULT_CLIENT,
                        help='Built StageZero client directory')
    parser.add_argument('--wait-seconds', type=float, default=180.,
                        help='Maximum time to wait for browser button click')
    parser.add_argument('--ready-seconds', type=float, default=5.,
                        help='Browser local-playback acknowledgement timeout (0–5)')
    parser.add_argument('--settle-seconds', type=float, default=.5,
                        help='Pause after camera/scene update before each capture')
    parser.add_argument('--closer-factor', type=float, default=.75,
                        help='Camera distance multiplier for cases with camera=closer (default .75)')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be in 1–65535')
    if not math.isfinite(args.wait_seconds) or args.wait_seconds <= 0:
        parser.error('--wait-seconds must be positive and finite')
    if not math.isfinite(args.ready_seconds) or not 0 <= args.ready_seconds <= 5:
        parser.error('--ready-seconds must be in 0–5')
    if not math.isfinite(args.settle_seconds) or not 0 <= args.settle_seconds <= 30:
        parser.error('--settle-seconds must be in 0–30')
    if not math.isfinite(args.closer_factor) or not .25 <= args.closer_factor <= 1:
        parser.error('--closer-factor must be in .25–1')
    run(args)


if __name__ == '__main__':
    main()
