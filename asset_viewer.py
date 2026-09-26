"""Local GLB inspection and synthetic rig diagnostics; no token, CSV or Pod."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor/ardy'))
from ardy.skeleton import G1Skeleton34
from ardy.viz.viser_utils import Character
from character_controls import CharacterControls
from live_motion import MotionSession
from retargeting import neutral_source_pose
from studio_server import create_studio_server
from studio_ui import STYLE


class NoInferenceBackend:
    def cancel(self, request_id):
        pass

    def generate(self, request_id, prompt, history):
        raise RuntimeError('Asset viewer has no inference backend; use the Studio for ARDY generation')


def diagnostic_clip(skeleton, frames=150):
    """A deterministic joint sweep, explicitly not model-generated motion."""
    positions, _ = neutral_source_pose(skeleton)
    local = np.tile(np.eye(3), (frames, 34, 1, 1))
    roots = np.tile(positions[0], (frames, 1))
    for frame in range(frames):
        phase = 2 * np.pi * frame / (frames - 1)
        angle = .65 * np.sin(phase)
        c, s = np.cos(angle), np.sin(angle)
        local[frame, skeleton.bone_index['left_shoulder_roll_skel']] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        roots[frame, 0] += .25 * np.sin(phase)
    with torch.inference_mode():
        global_rotations, joint_positions, _ = skeleton.fk(torch.tensor(local, dtype=torch.float64), torch.tensor(roots, dtype=torch.float64))
    return joint_positions.numpy(), global_rotations.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=2341)
    parser.add_argument('--glb', type=Path, help='Open a local GLB on first browser connection')
    parser.add_argument('--characters', type=Path, default=ROOT / '.runtime/characters')
    parser.add_argument('--environment', choices=('studio', 'warehouse', 'none'), default='studio',
                        help='Reflection lighting for inspecting PBR materials')
    args = parser.parse_args()
    torch.set_num_threads(2)
    skeleton = G1Skeleton34()
    positions, rotations = diagnostic_clip(skeleton)
    server = create_studio_server(host='127.0.0.1', port=args.port, label='StageZero · GLB lab')
    server.gui.configure_theme(dark_mode=True, control_layout='collapsible', control_width='large', show_logo=False, show_share_button=False)
    server.scene.set_up_direction('+y')
    server.scene.world_axes.visible = False
    server.scene.configure_environment_map(None if args.environment == 'none' else args.environment)
    server.scene.configure_default_lights(enabled=True, cast_shadow=True)
    server.scene.add_light_ambient('/fill', intensity=.6)
    server.scene.add_box('/floor', dimensions=(20, .05, 20), position=(0, -.026, 0), color=(28, 37, 47))
    fallback = Character('actor', server, skeleton, create_skeleton_mesh=False, create_skinned_mesh=True, mesh_mode='g1_stl', show_foot_contacts=False)
    session = MotionSession(NoInferenceBackend(), positions, rotations)
    session.fps = 25
    controls = CharacterControls(server, session, skeleton, args.characters)
    server.gui.add_html(STYLE)
    server.gui.add_markdown('## GLB character lab\nImport a character, inspect its materials, then test a **synthetic joint sweep**. This page does not call ARDY inference.')
    controls.build_gui(server.gui)
    play_button = server.gui.add_button('Play sweep')
    pause_button = server.gui.add_button('Pause')
    neutral_button = server.gui.add_button('Neutral pose')
    playhead = server.gui.add_markdown('')
    scrub = server.gui.add_slider('Test frame', min=0, max=len(positions)-1, step=1, initial_value=0)

    @scrub.on_update
    def scrubbed(event):
        if event.client is not None:
            with session.lock:
                session.pause()
                session.frame = int(scrub.value)

    @play_button.on_click
    def play_clicked(_):
        session.play()

    @pause_button.on_click
    def pause_clicked(_):
        session.pause()

    @neutral_button.on_click
    def neutral_clicked(_):
        with session.lock:
            session.pause()
            session.frame = 0

    if args.glb is not None:
        if args.glb.stat().st_size > 32 * 1024 * 1024:
            parser.error('GLB exceeds the 32 MiB import limit')
        controls.set_initial_asset(controls.add_file(args.glb.read_bytes(), args.glb.name))

    @server.on_client_connect
    def connected(client):
        client.camera.position = (2.8, 1.8, 3.8)
        client.camera.look_at = (0., .9, 0.)
        client.camera.up_direction = (0., 1., 0.)
        controls.on_client_connect(client)

    print(f'GLB lab (synthetic motion only): http://127.0.0.1:{server.get_port()}', flush=True)
    previous = None
    try:
        while True:
            key = session.tick()
            controls.tick(key)
            if key != previous:
                with session.lock:
                    fallback.set_pose(torch.from_numpy(session.positions[session.frame]), torch.from_numpy(session.rotations[session.frame]))
                previous = key
                playhead.content = f'**Synthetic test** · {session.frame / session.fps:.2f} / {(len(positions)-1) / session.fps:.2f} s'
                scrub.value = session.frame
            play_button.disabled = not session.character_motion_enabled
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    finally:
        session.pause()
        controls.close()
        server.stop()


if __name__ == '__main__':
    main()
