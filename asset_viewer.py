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
from character_geometry import posed_minimum_y
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
    """FK narrow, softly bent stand with an outward-and-return arm sweep.

    This is a lab motion, distinct from the identity-local G1 calibration
    pose used by production retargeting.
    """
    if frames < 2:
        raise ValueError('Diagnostic clip needs at least two frames')
    neutral_positions, _ = neutral_source_pose(skeleton)
    local = np.tile(np.eye(3), (frames, 34, 1, 1))
    roots = np.tile(neutral_positions[0], (frames, 1))

    def pitch_rotation(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def roll_rotation(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    for side in ('left', 'right'):
        hip = neutral_positions[skeleton.bone_index[f'{side}_hip_pitch_skel']]
        hip_roll = neutral_positions[skeleton.bone_index[f'{side}_hip_roll_skel']]
        knee = neutral_positions[skeleton.bone_index[f'{side}_knee_skel']]
        ankle = neutral_positions[skeleton.bone_index[f'{side}_ankle_roll_skel']]

        # The G1 hip motors are separated laterally. Rotate the lower chain
        # inward around its roll pivot until the knee is under the anatomical
        # hip, then restore an upright shin through the knee's local orientation.
        hip_to_roll = hip_roll - hip
        roll_to_knee = knee - hip_roll
        radius = np.linalg.norm(roll_to_knee[:2])
        roll_angle = (np.arctan2(-hip_to_roll[0], (radius**2 - hip_to_roll[0]**2)**.5)
                      - np.arctan2(roll_to_knee[0], -roll_to_knee[1]))
        roll = roll_rotation(roll_angle)
        thigh = hip_to_roll + roll @ roll_to_knee
        shin = ankle - knee

        # The TASM knee pivot sits at the front edge of its skin; a 35 mm G1
        # forward offset gives the visible mesh a gentle ~12-degree bend.
        # Counter-pitch at the ankle preserves level soles.
        knee_forward = .035
        thigh_pitch = np.arctan2(thigh[2], -thigh[1]) - np.arctan2(
            knee_forward, (np.linalg.norm(thigh[1:])**2 - knee_forward**2)**.5)
        shin_pitch = np.arctan2(shin[2], -shin[1]) + np.arctan2(
            knee_forward, (np.linalg.norm(shin[1:])**2 - knee_forward**2)**.5)
        hip_pitch = pitch_rotation(thigh_pitch)
        shin_global = pitch_rotation(shin_pitch)
        local[:, skeleton.bone_index[f'{side}_hip_pitch_skel']] = hip_pitch
        local[:, skeleton.bone_index[f'{side}_hip_roll_skel']] = roll
        local[:, skeleton.bone_index[f'{side}_knee_skel']] = (hip_pitch @ roll).T @ shin_global
        local[:, skeleton.bone_index[f'{side}_ankle_pitch_skel']] = pitch_rotation(-shin_pitch)
        elbow = neutral_positions[skeleton.bone_index[f'{side}_elbow_skel']]
        wrist = neutral_positions[skeleton.bone_index[f'{side}_wrist_yaw_skel']]
        forearm = wrist - elbow
        elbow_pitch = np.arctan2(forearm[2], -forearm[1])
        local[:, skeleton.bone_index[f'{side}_elbow_skel']] = pitch_rotation(elbow_pitch)

    # Ground the actual toe endpoints after the leg correction. Keep the
    # resulting world positions and rotations coupled through the same FK.
    with torch.inference_mode():
        _, stand_positions, _ = skeleton.fk(torch.tensor(local[0], dtype=torch.float64), torch.tensor(roots[0], dtype=torch.float64))
    toe_indices = [skeleton.bone_index[f'{side}_toe_base'] for side in ('left', 'right')]
    roots[:, 1] -= float(stand_positions[toe_indices, 1].min())

    for frame in range(frames):
        phase = 2 * np.pi * frame / (frames - 1)
        angle = .1 + .325 * (1 - np.cos(phase))
        c, s = np.cos(angle), np.sin(angle)
        local[frame, skeleton.bone_index['left_shoulder_roll_skel']] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        local[frame, skeleton.bone_index['right_shoulder_roll_skel']] = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
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
    floor = server.scene.add_box('/floor', dimensions=(20, .05, 20), position=(0, -.025, 0), color=(28, 37, 47))
    fallback = Character('actor', server, skeleton, create_skeleton_mesh=False, create_skinned_mesh=True, mesh_mode='g1_stl', show_foot_contacts=False)
    session = MotionSession(NoInferenceBackend(), positions, rotations)
    session.fps = 25
    controls = CharacterControls(server, session, skeleton, args.characters)
    server.gui.add_html(STYLE)
    server.gui.add_markdown('## GLB character lab\nImport a character, inspect its materials, then test a **synthetic joint sweep**. This page does not call ARDY inference.')
    controls.build_gui(server.gui)
    play_button = server.gui.add_button('Play sweep')
    pause_button = server.gui.add_button('Pause')
    neutral_button = server.gui.add_button('Stand pose')
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
    floor_revision = -1
    try:
        while True:
            key = session.tick()
            controls.tick(key)
            with controls._lock:
                revision, entry = controls.revision, controls.active_entry
            if revision != floor_revision:
                if entry is None:
                    level = 0.
                elif entry.retargeter is None:
                    level = float(entry.asset.bounds[0, 1])
                else:
                    stand = entry.retargeter.retarget(positions[0], rotations[0])
                    level = posed_minimum_y(entry.asset, stand.world_matrices)
                floor.position = (0., level - .025, 0.)
                floor_revision = revision
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
