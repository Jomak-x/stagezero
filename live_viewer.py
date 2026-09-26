"""Milestone 2 viewer; original preview.py remains the unchanged fallback."""
import json
import time
import threading
import numpy as np
import torch
import trimesh
import viser
from PIL import Image
from preview import ROOT, load_recording
from ardy.viz.viser_utils import Character
from live_motion import Backend, MotionSession

def main():
    torch.set_num_threads(2)
    skeleton, positions, rotations = load_recording(ROOT / "assets/recorded_g1.csv")
    server = viser.ViserServer(host="127.0.0.1", port=2335, label="StageZero")
    server.gui.configure_theme(dark_mode=True, control_layout="floating", control_width="medium", show_logo=False, show_share_button=False, brand_color=(72, 202, 183))
    server.scene.set_up_direction("+y")
    server.scene.world_axes.visible = False
    server.scene.configure_environment_map(None)
    server.scene.configure_default_lights(enabled=True, cast_shadow=True)
    server.scene.add_light_ambient("/fill", color=(191, 215, 239), intensity=0.45)
    # A quiet floor and low circular platform give scale without distracting scenery.
    server.scene.add_box("/floor", color=(24, 31, 42), dimensions=(200, 0.1, 200), position=(0, -0.051, 0), cast_shadow=False)
    stage = trimesh.creation.cylinder(radius=2.0, height=0.12, sections=96)
    stage.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, (1, 0, 0)))
    server.scene.add_mesh_simple("/stage", vertices=stage.vertices, faces=stage.faces, color=(68, 83, 99), position=(0, -0.06, 0), flat_shading=False)
    character = Character("actor", server, skeleton, create_skeleton_mesh=False, create_skinned_mesh=True, mesh_mode="g1_stl", show_foot_contacts=False)
    if not character.g1_mesh_rig.mesh_handles:
        raise RuntimeError("Supplied G1 meshes are missing; cannot show the preview")
    for mesh in character.g1_mesh_rig.mesh_handles:
        mesh.color = (202, 219, 222)
    center = positions[:, 0].mean(0).numpy()
    camera_target = (float(center[0]), 0.75, float(center[2]))
    camera_position = (float(center[0]) + 2.6, 1.9, float(center[2]) + 3.4)

    def reset_camera(client):
        client.camera.position = camera_position
        client.camera.look_at = camera_target
        client.camera.up_direction = (0, 1, 0)
        client.camera.fov = np.deg2rad(42)

    @server.on_client_connect
    def connected(client):
        reset_camera(client)

    backend = Backend(ROOT / ".runtime/api-token")
    session = MotionSession(backend, positions.numpy(), rotations.numpy(), ROOT / "review/live-metrics.jsonl")
    server.gui.add_markdown("# StageZero\n**02 / Direct the actor**")
    mode = server.gui.add_dropdown("Motion source", ("Recorded preview", "Live ARDY"), initial_value="Recorded preview")
    badge = server.gui.add_markdown("")
    instruction = server.gui.add_text("Instruction", initial_value="A person waves with their right hand.", multiline=True)
    generate = server.gui.add_button("Generate next 4 seconds", icon=viser.Icon.SPARKLES)
    play = server.gui.add_button("Play / Resume", icon=viser.Icon.PLAYER_PLAY)
    pause = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
    reset = server.gui.add_button("Reset", icon=viser.Icon.ROTATE_2)
    follow = server.gui.add_checkbox("Follow actor", initial_value=True)
    status = server.gui.add_markdown("")
    performance = server.gui.add_markdown("")
    server.gui.add_markdown("Drag: orbit · Right-drag: pan · Scroll: zoom\n\nGenerate holds the pose, then continues. Reset starts a fresh take.")
    server.gui.add_markdown("[ARDY G1 model](https://huggingface.co/nvidia/ARDY-G1-RP-25FPS-Horizon52) · [Recorded source](https://huggingface.co/datasets/bones-studio/seed)")

    @mode.on_update
    def mode_changed(_):
        session.set_mode(mode.value)
        for client in server.get_clients().values():
            reset_camera(client)

    @instruction.on_update
    def instruction_changed(_):
        session.edit_prompt(instruction.value)

    @generate.on_click
    def generate_clicked(_):
        session.submit(instruction.value)

    @play.on_click
    def play_clicked(_):
        session.play()

    @pause.on_click
    def pause_clicked(_):
        session.pause()

    @reset.on_click
    def reset_clicked(_):
        session.reset()
        for client in server.get_clients().values():
            reset_camera(client)

    previous = None
    last_ui = None
    render_thread = None
    previous_root = None

    def acknowledge(client, request_id, submitted):
        # Render completion is an upper bound on visible response, including image return.
        # Only one render request may be in flight; disconnected clients cannot pile up threads.
        try:
            image = client.get_render(height=180, width=320)
            elapsed = time.perf_counter() - submitted
            session.record_ack(request_id, elapsed)
            folder = ROOT / "review/generated"
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(folder / f"{request_id}.jpg")
        except Exception:
            pass  # Receipt latency remains available if a browser disconnects.

    print("StageZero live viewer: http://127.0.0.1:2335", flush=True)
    try:
        while True:
            key = session.tick()
            with session.lock:
                if key != previous:
                    with server.atomic():
                        character.set_pose(torch.from_numpy(session.positions[session.frame]), torch.from_numpy(session.rotations[session.frame]))
                        root = session.positions[session.frame, 0].copy()
                        root[1] = 0
                        if session.kind == "generated":
                            if previous_root is not None and follow.value:
                                delta = root - previous_root
                                for client in server.get_clients().values():
                                    client.camera.position = np.asarray(client.camera.position) + delta
                                    client.camera.look_at = np.asarray(client.camera.look_at) + delta
                            previous_root = root
                        else:
                            previous_root = None
                    previous = key
                    if session.needs_ack and server.get_clients() and (render_thread is None or not render_thread.is_alive()):
                        request_id, submitted = session.needs_ack
                        session.needs_ack = None
                        server.flush()
                        client = next(iter(server.get_clients().values()))
                        render_thread = threading.Thread(target=acknowledge, args=(client, request_id, submitted), daemon=True)
                        render_thread.start()
                live = session.mode == "Live ARDY"
                instruction.disabled = not live
                generate.disabled = not live
                pause.disabled = not (session.playing or session.busy)
                play.disabled = session.playing or (session.kind == "reference" and not session.busy)
                if session.kind == "recorded":
                    label = "**RECORDED PLAYBACK**\nBONES SEED shadow boxing · not AI generation"
                elif session.kind == "generated":
                    label = "**FRESH ARDY GENERATION**\nG1 · 25 fps · complete segments, not streaming"
                else:
                    label = "**LIVE ARDY MODE**\nNo generated motion yet · reference pose only"
                action = "Playing" if session.playing else ("Finished" if session.frame == len(session.positions) - 1 else "Paused")
                text = f"**{action}** · {session.frame / session.fps:.1f} s\n\n{session.status}"
                metric_text = ""
                if session.metrics:
                    metric_text = f"GPU generation: **{session.metrics['generation_seconds']:.2f} s** · received: **{session.metrics['command_to_received_seconds']:.2f} s**"
                    if "command_to_browser_render_ack_seconds" in session.metrics:
                        metric_text += f"\n\nBrowser render acknowledged: **{session.metrics['command_to_browser_render_ack_seconds']:.2f} s**"
                ui = (label, text, metric_text)
                if ui != last_ui:
                    badge.content, status.content, performance.content = ui
                    last_ui = ui
            time.sleep(1 / 120)
    except KeyboardInterrupt:
        session.reset()
        server.stop()


if __name__ == "__main__":
    main()
