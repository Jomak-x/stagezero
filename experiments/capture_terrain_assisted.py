"""Capture the actual solved mesh17 presentation, with no further pose edits."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.capture_core_performance import get_render_with_timeout, sheet_indices, write_contact_sheet
from core_scene_reactions import object_states
from object_scene import ObjectSceneLayer
from scene_composition import validate_scene
from studio_server import create_studio_server
from terrain_assisted_renderer import TerrainAssistedRenderer


def _load_core_project(path: Path):
    from realtime_director import RealtimeDirector
    from studio_core_terrain_state import MAX_STUDIO_BYTES, unpack_terrain_project
    if path.stat().st_size > MAX_STUDIO_BYTES:
        raise ValueError('Core project exceeds the 64 MB archive limit')
    content = path.read_bytes()
    director = RealtimeDirector.load_project(content)
    result = unpack_terrain_project(content, director)
    if result is None:
        raise ValueError('Core project has no terrain-assisted presentation')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--project', type=Path)
    inputs.add_argument('--core-project', type=Path,
                        help='Saved terrain-aware Core studio project from the normal app')
    inputs.add_argument('--poses', type=Path)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=24993)
    parser.add_argument('--camera-position', type=float, nargs=3)
    parser.add_argument('--look-at', type=float, nargs=3)
    parser.add_argument('--follow-camera', action='store_true',
                        help='Keep the actual character and its foot contacts visible along the route')
    args = parser.parse_args()
    native = None
    if args.project or args.core_project:
        if args.core_project:
            result = _load_core_project(args.core_project)
        else:
            from terrain_assisted_session import load_assisted_result
            result = load_assisted_result(args.project)
        scene, native, presentation = result.scene, result.native_clip, result.presentation
        actor_ids = native.actor_ids
        report = result.report
    else:
        if args.scene is None:
            parser.error('--poses requires --scene')
        scene = validate_scene(json.loads(args.scene.read_text()))
        with np.load(args.poses, allow_pickle=False) as data:
            p, r = data['positions'], data['rotations']
            if p.ndim == 3:
                p, r = p[None], r[None]
            presentation = SimpleNamespace(positions=p, rotations=r,
                rig_asset_sha256=tuple(str(x) for x in np.atleast_1d(data['rig_asset_sha256'])))
        actor_ids = ('actor_1',)
        report = {'accepted': False, 'scope': 'isolated actual-rig proof'}
    args.output.mkdir(parents=True, exist_ok=True)
    video = args.output/'performance.mp4'
    if video.exists():
        raise FileExistsError(video)
    server = create_studio_server(host='127.0.0.1', port=args.port,
                                  label='Terrain-assisted actual-rig review')
    try:
        server.scene.set_up_direction('+y')
        server.scene.world_axes.visible = False
        server.scene.configure_environment_map(None)
        server.scene.configure_default_lights(enabled=True, cast_shadow=True)
        server.scene.add_light_ambient('/fill', color=(210, 220, 230), intensity=.7)
        renderer = TerrainAssistedRenderer(server)
        renderer.set_presentation(presentation, actor_ids=actor_ids)
        renderer.set_visible(True)
        layer = ObjectSceneLayer(server)
        def update_scene(frame):
            states = (object_states(scene, native, frame, enabled=True, terrain=True)
                      if native is not None else [{'id': o['id'], 'position': o['position'],
                                                   'color': o['color'], 'active': False} for o in scene['objects']])
            layer.update(scene['objects'], {'objects': states, 'assets': scene.get('assets', []),
                'effects': scene['effects'], 'lighting': scene['lighting'], 'seconds': frame/20.})
        update_scene(0)
        ready, selected = threading.Event(), {}
        button = server.gui.add_button('Capture actual-rig proof')
        @button.on_click
        def start(event):
            if event.client is not None and not ready.is_set():
                selected['client'] = event.client
                button.disabled = True
                ready.set()
        print(f'OPEN http://127.0.0.1:{server.get_port()} and click Capture actual-rig proof', flush=True)
        if not ready.wait(600):
            raise TimeoutError('No capture browser')
        client = selected['client']
        camera = scene['camera']
        client.camera.up_direction = (0, 1, 0)
        client.camera.position = tuple(args.camera_position or camera['position'])
        client.camera.look_at = tuple(args.look_at or camera['look_at'])
        client.camera.fov = np.deg2rad(48)
        time.sleep(8)
        frames = presentation.positions.shape[1]
        chosen, thumbs = sheet_indices(frames, 16), []
        command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', '1280x720', '-r', '20', '-i', '-', '-an', '-c:v', 'libx264',
                   '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for frame in range(frames):
                if args.follow_camera:
                    root = presentation.positions[0, frame, 0]
                    client.camera.position = tuple(root + np.asarray([3., 1.7, 2.5]))
                    client.camera.look_at = tuple(root + np.asarray([0., -.1, -.6]))
                renderer.tick(frame)
                update_scene(frame)
                server.flush()
                rgb = np.ascontiguousarray(get_render_with_timeout(client, width=1280, height=720, timeout=45)[:,:,:3])
                process.stdin.write(rgb.tobytes())
                if frame in chosen:
                    thumbs.append((frame, Image.fromarray(rgb).resize((400, 225))))
                if frame % 40 == 0:
                    print(f'CAPTURED {frame+1}/{frames}', flush=True)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError('Video encoding failed')
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
        write_contact_sheet(thumbs, args.output/'contact-sheet.png')
        (args.output/'manifest.json').write_text(json.dumps({
            'kind': 'explicit terrain-assisted actual character-rig replay', 'fps': 20,
            'frames': frames, 'duration_seconds': frames/20, 'native_history_used_for_display': False,
            'additional_ik': False, 'additional_retarget': False,
            'camera_mode': 'follow' if args.follow_camera else 'fixed',
            'presentation_positions_sha256': hashlib.sha256(presentation.positions.tobytes()).hexdigest(),
            'rig_asset_sha256': list(presentation.rig_asset_sha256), 'report': report,
            'capture_complete': True}, indent=2)+'\n')
        print(f'COMPLETE {video}', flush=True)
    finally:
        server.stop()


if __name__ == '__main__':
    main()
