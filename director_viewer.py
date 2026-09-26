"""Single-actor directing workflow; previous viewers remain available."""
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
from live_motion import Backend
from directing import DirectorSession

def main():
    torch.set_num_threads(2)
    skeleton, positions, rotations = load_recording(ROOT / "assets/recorded_g1.csv")
    server = viser.ViserServer(host="127.0.0.1", port=2336, label="StageZero")
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
    backend = Backend(ROOT / ".runtime/api-token")
    session = DirectorSession(backend, positions.numpy(), rotations.numpy(), ROOT / "review/live-metrics.jsonl")
    previous_root = None
    center = positions[:, 0].mean(0).numpy()
    camera_target = (float(center[0]), 0.75, float(center[2]))
    camera_position = (float(center[0]) + 4.0, 2.0, float(center[2]) + 3.4)

    def reset_camera(client):
        root = session.positions[session.frame, 0].copy() if session.kind == "generated" else np.zeros(3)
        root[1] = 0
        client.camera.position = np.asarray(camera_position) + root
        client.camera.look_at = np.asarray(camera_target) + root
        client.camera.up_direction = (0, 1, 0)
        client.camera.fov = np.deg2rad(38)

    @server.on_client_connect
    def connected(client):
        reset_camera(client)

    gate_posts = [server.scene.add_box(f"/gate/post{i}", dimensions=(.08, 1.65, .12), color=(83, 113, 131)) for i in range(2)]
    gate_panel = server.scene.add_box("/gate/panel", dimensions=(1.6, 1.25, .06), color=(65, 147, 138), opacity=.5)
    zone_mesh = trimesh.creation.cylinder(radius=1., height=.006, sections=64)
    zone_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, (1, 0, 0)))
    gate_zone = server.scene.add_mesh_simple("/gate/zone", vertices=np.asarray(zone_mesh.vertices, dtype=np.float32), faces=np.asarray(zone_mesh.faces, dtype=np.uint32), color=(53, 156, 141), opacity=.3)
    gate_radius = None
    server.gui.add_markdown("# StageZero\n**Direct · Record · Revise**")
    mode = server.gui.add_dropdown("Motion source", ("Recorded preview", "Live ARDY"), initial_value="Recorded preview")
    badge = server.gui.add_markdown("")
    status = server.gui.add_markdown("")
    instruction = server.gui.add_text("Instruction", initial_value="A person waves with their right hand.", multiline=True)
    generate = server.gui.add_button("Generate from playhead", icon=viser.Icon.SPARKLES)
    play = server.gui.add_button("Play / Resume", icon=viser.Icon.PLAYER_PLAY)
    pause = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
    reset = server.gui.add_button("Rewind", icon=viser.Icon.ROTATE_2)
    new_take = server.gui.add_button("New take")
    takes = server.gui.add_dropdown("Take", ("No takes yet",), initial_value="No takes yet")
    scrub = server.gui.add_slider("Playhead (s)", min=0., max=1., step=.04, initial_value=0.)
    action_label = server.gui.add_markdown("")
    gate_status = server.gui.add_markdown("")
    with server.gui.add_folder("Save / Open", expand_by_default=False):
        project_name = server.gui.add_text("Project name", initial_value="My performance")
        save = server.gui.add_button("Save project + download")
        saved = server.gui.add_dropdown("Saved on mini", ("No saved projects",))
        load_saved = server.gui.add_button("Open selected project")
        upload = server.gui.add_upload_button("Open project file", mime_type=".npz")
        clear_project = server.gui.add_button("New project (backs up current)")
        file_status = server.gui.add_markdown("")
    follow = server.gui.add_checkbox("Follow actor", initial_value=True)

    @follow.on_update
    def follow_changed(event):
        nonlocal previous_root
        if event.client is None or not follow.value:
            return
        with session.lock:
            if session.kind != "generated":
                return
            root = session.positions[session.frame, 0].copy()
            root[1] = 0
            if previous_root is None:
                for client in server.get_clients().values():
                    reset_camera(client)
            else:
                delta = root - previous_root
                for client in server.get_clients().values():
                    client.camera.position = np.asarray(client.camera.position) + delta
            previous_root = root

    with server.gui.add_folder("Performance", expand_by_default=False):
        performance = server.gui.add_markdown("")
    server.gui.add_markdown("Drag: orbit · Right-drag: pan · Scroll: zoom\n\nGenerate at the end to extend. Scrub back and generate to create an alternate ending. Rewind preserves your take; New take starts fresh.")
    server.gui.add_markdown("[ARDY G1 model](https://huggingface.co/nvidia/ARDY-G1-RP-25FPS-Horizon52) · [Recorded source](https://huggingface.co/datasets/bones-studio/seed)")

    @mode.on_update
    def mode_changed(event):
        nonlocal previous_root
        if event.client is None:
            return
        session.set_mode(mode.value)
        previous_root = None
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
        nonlocal previous_root
        session.reset()
        previous_root = None
        for client in server.get_clients().values():
            reset_camera(client)

    project_folder = ROOT / ".runtime/projects"
    project_folder.mkdir(parents=True, exist_ok=True)
    take_map = {}
    saved_map = {}

    def refresh_saved():
        saved_map.clear()
        saved_map.update({p.name: p for p in sorted(project_folder.glob("*.stagezero.npz"), key=lambda p: p.stat().st_mtime, reverse=True)})
        saved.options = tuple(saved_map) or ("No saved projects",)
        if saved.value not in saved.options:
            saved.value = saved.options[0]

    refresh_saved()

    @new_take.on_click
    def begin_take(_):
        nonlocal previous_root
        session.new_take()
        previous_root = None
        for client in server.get_clients().values():
            reset_camera(client)

    @takes.on_update
    def select_take(event):
        if event.client is not None and takes.value in take_map:
            session.select_take(take_map[takes.value])

    @scrub.on_update
    def seek(event):
        if event.client is not None:
            session.seek(round(scrub.value * session.fps))

    @save.on_click
    def save_file(event):
        try:
            path, data = session.save_project(project_folder, project_name.value)
            refresh_saved()
            saved.value = path.name
            if event.client is not None:
                event.client.send_file_download(path.name, data)
        except Exception as exc:
            session.project_status = f"Save failed: {exc}"

    def open_data(data):
        nonlocal previous_root
        try:
            session.load_project(data)
            previous_root = None
            mode.value = "Live ARDY"
            for client in server.get_clients().values():
                reset_camera(client)
        except Exception as exc:
            session.project_status = f"Open failed; current takes preserved: {exc}"

    @load_saved.on_click
    def open_saved(_):
        path = saved_map.get(saved.value)
        if path:
            try:
                open_data(path.read_bytes())
            except OSError as exc:
                session.project_status = f"Open failed: {exc}"

    @upload.on_upload
    def open_uploaded(_):
        open_data(upload.value.content)

    @clear_project.on_click
    def start_project(_):
        nonlocal previous_root
        try:
            session.new_project(project_folder)
            previous_root = None
            mode.value = 'Live ARDY'
            refresh_saved()
            for client in server.get_clients().values():
                reset_camera(client)
        except Exception as exc:
            session.project_status = f'Backup failed; project retained: {exc}'

    previous = None
    last_ui = None
    render_thread = None

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

    print("StageZero directing viewer: http://127.0.0.1:2336", flush=True)
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
                            if follow.value:
                                if previous_root is not None:
                                    delta = root - previous_root
                                    for client in server.get_clients().values():
                                        client.camera.position = np.asarray(client.camera.position) + delta
                                        # Viser moves look_at by the same offset in its position setter.
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
                take_map = {f"{i + 1}. {t.name} · {len(t.positions) / 25:.1f}s": t.id for i, t in enumerate(session.takes.values())}
                options = tuple(take_map) or ("No takes yet",)
                if tuple(takes.options) != options:
                    takes.options = options
                selection = next((name for name, tid in take_map.items() if tid == session.active_take), options[0])
                if takes.value != selection:
                    takes.value = selection
                takes.disabled = not live or not session.takes
                new_take.disabled = not live
                save.disabled = not session.takes
                scrub.max = max(.04, (len(session.positions) - 1) / session.fps)
                scrub.step = 1 / session.fps
                scrub.value = session.frame / session.fps
                scrub.disabled = session.kind == "reference"
                action_label.content = "**Actor: G1** · " + (session.current_action() or "Choose an instruction")
                file_status.content = session.project_status
                gate = session.scene['gate']
                gate_is_open = session.gate_open()
                for i, post in enumerate(gate_posts):
                    post.position = (gate['position'][0] + (-.84 if i == 0 else .84), .825, gate['position'][2])
                    post.visible = gate['enabled']
                gate_panel.position = (gate['position'][0], 2.4 if gate_is_open else .725, gate['position'][2])
                gate_panel.visible = gate['enabled']
                gate_zone.position = (gate['position'][0], .004, gate['position'][2])
                if gate_radius != gate['radius']:
                    gate_zone.vertices = np.asarray(zone_mesh.vertices * np.array([gate['radius'], 1., gate['radius']]), dtype=np.float32)
                    gate_radius = gate['radius']
                gate_zone.visible = gate['enabled']
                gate_status.content = ("Gate: **open**" if gate_is_open else "Gate: closed · opens inside the teal area") if gate['enabled'] else ""
                instruction.disabled = not live
                generate.disabled = not live
                pause.disabled = not (session.playing or session.busy)
                play.disabled = session.playing or (session.kind == "reference" and not session.busy)
                if session.kind == "recorded":
                    label = "**Recorded preview** · not AI generation"
                elif session.kind == "generated":
                    label = "**Stored ARDY motion** · 25 fps · not streaming"
                else:
                    label = "**Live ARDY** · ready for a new take"
                action = "Generating" if session.busy else ("Playing" if session.playing else ("Finished" if session.frame == len(session.positions) - 1 else "Paused"))
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
