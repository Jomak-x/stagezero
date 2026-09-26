"""StageZero motion studio: direct, review, edit, and navigate stored performances."""
import argparse
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
from object_directing import ObjectDirectorSession as DirectorSession
from object_scene import ObjectSceneLayer
from object_controls import add_object_controls
from studio_server import create_studio_server
from studio_camera import StudioCamera
from studio_timeline import StudioTimeline
from studio_ui import StudioUI, section


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=2336)
    parser.add_argument('--project', type=str, help='Open a saved project at startup')
    parser.add_argument('--objects', type=str, help='Load a generated object scene JSON')
    args = parser.parse_args()
    torch.set_num_threads(2)
    skeleton, positions, rotations = load_recording(ROOT / 'assets/recorded_g1.csv')
    server = create_studio_server(host='127.0.0.1', port=args.port, label='StageZero Studio', enable_camera_keyboard_controls=False)
    server.gui.configure_theme(dark_mode=True, control_layout='collapsible', control_width='large', show_logo=False, show_share_button=False, brand_color=(126, 224, 195))
    server.scene.set_up_direction('+y')
    server.scene.world_axes.visible = False
    server.scene.configure_environment_map(None)
    server.scene.configure_default_lights(enabled=True, cast_shadow=True)
    server.scene.add_light_ambient('/fill', color=(191, 215, 239), intensity=.6)
    server.scene.add_box('/floor', color=(20, 28, 38), dimensions=(200, .1, 200), position=(0, -.07, 0), cast_shadow=False)
    grid = server.scene.add_grid('/ground-grid', plane='xz', width=200, height=200, cell_size=.5, section_size=2., cell_color=(44, 57, 70), section_color=(68, 87, 100), position=(0, .008, 0), fade_distance=30., fade_strength=2., shadow_opacity=0.)
    stage = trimesh.creation.cylinder(radius=2., height=.035, sections=96)
    stage.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, (1,0,0)))
    platform = server.scene.add_mesh_simple('/stage', vertices=stage.vertices, faces=stage.faces, color=(37, 52, 65), position=(0,-.0175,0), flat_shading=False)
    character = Character('actor', server, skeleton, create_skeleton_mesh=False, create_skinned_mesh=True, mesh_mode='g1_stl', show_foot_contacts=False)
    if not character.g1_mesh_rig.mesh_handles:
        raise RuntimeError('Supplied G1 meshes are missing; cannot show the preview')
    for mesh in character.g1_mesh_rig.mesh_handles: mesh.color = (206, 226, 233)
    backend = Backend(ROOT / '.runtime/api-token')
    session = DirectorSession(backend, positions.numpy(), rotations.numpy(), ROOT / 'review/live-metrics.jsonl')
    if args.project:
        from pathlib import Path
        session.load_project(Path(args.project).read_bytes())
    if args.objects:
        session.load_objects(args.objects)
    object_layer = ObjectSceneLayer(server)
    previous_objects = None
    def actor_root():
        with session.lock:
            return session.positions[session.frame, 0].copy()
    camera = StudioCamera(server, actor_root)
    @server.on_client_connect
    def connected(client):
        client.camera.near = .05
        client.camera.far = 250.
        camera.reset(client)

    gate_posts = [server.scene.add_box(f'/gate/post{i}', dimensions=(.055,1.65,.08), color=(83,113,131)) for i in range(2)]
    gate_panel = server.scene.add_box('/gate/panel', dimensions=(1.6,1.25,.04), color=(65,147,138), opacity=.28)
    zone_mesh = trimesh.creation.cylinder(radius=1., height=.006, sections=64)
    zone_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2,(1,0,0)))
    gate_zone = server.scene.add_mesh_simple('/gate/zone', vertices=np.asarray(zone_mesh.vertices,dtype=np.float32), faces=np.asarray(zone_mesh.faces,dtype=np.uint32), color=(53,156,141), opacity=.22)
    gizmo = server.scene.add_transform_controls('/gate-edit', scale=.7, active_axes=(True,False,True), disable_rotations=True, translation_limits=((-100,100),(0,0),(-100,100)), visible=False)
    gate_ui = {}
    def edit_gate(position=None):
        with session.lock:
            gate = session.scene['gate']
            gate.update(enabled=gate_ui['enabled'].value, radius=gate_ui['radius'].value,
                        position=[gate_ui['x'].value,0.,gate_ui['z'].value] if position is None else [float(position[0]),0.,float(position[2])])
            for take in session.takes.values():
                take.events = [e for e in take.events if e['type'] != 'gate_open']
                session._record_gate_events(take)
            session.project_revision += 1
            session.project_status = 'Scene updated · save project to keep changes'

    def scene_controls(gui):
        add_object_controls(gui, session)
        section(gui, 'Stage & interaction', 'Move the gate in the viewport or enter its floor coordinates.')
        gate_ui['grid'] = gui.add_checkbox('Show floor grid', initial_value=True)
        gate_ui['stage'] = gui.add_checkbox('Show platform', initial_value=True)
        section(gui, 'Activation gate', 'The teal area triggers a recorded gate event when the actor enters it.')
        gate = session.scene['gate']
        gate_ui['enabled'] = gui.add_checkbox('Enable gate', initial_value=gate['enabled'])
        gate_ui['edit'] = gui.add_checkbox('Show move handle', initial_value=False)
        gate_ui['x'] = gui.add_number('Position X · m', initial_value=float(gate['position'][0]), min=-100.,max=100.,step=.1)
        gate_ui['z'] = gui.add_number('Position Z · m', initial_value=float(gate['position'][2]), min=-100.,max=100.,step=.1)
        gate_ui['radius'] = gui.add_slider('Trigger radius', min=.1,max=3.,step=.05,initial_value=gate['radius'])
        gate_ui['status'] = gui.add_markdown('')
        gui.add_html('<div class="sz-note">Drag the red or blue handle to move along the floor. This gate is an interaction trigger; it does not constrain generated motion.</div>')
        for key in ('enabled','x','z','radius'):
            @gate_ui[key].on_update
            def changed(e):
                if e.client is not None: edit_gate()
        @gate_ui['grid'].on_update
        def grid_changed(_): grid.visible = gate_ui['grid'].value
        @gate_ui['stage'].on_update
        def stage_changed(_): platform.visible = gate_ui['stage'].value

    @gizmo.on_update
    def move_gate(_):
        edit_gate(gizmo.position)

    ui = StudioUI(server, session, camera, ROOT / '.runtime/projects', scene_controls)
    timeline = StudioTimeline(server, session)

    @server.scene.on_keyboard_event('keydown')
    def transport_key(event):
        if event.event_type != 'keydown' or event.ctrl_key or event.meta_key or event.alt_key:
            return
        with session.lock:
            if session.busy or session.kind not in ('recorded', 'generated'):
                return
            if event.key == ' ':
                session.pause() if session.playing else session.play()
            elif event.key in ('ArrowLeft', 'ArrowRight'):
                step = (round(session.fps) if event.shift_key else 1) * (-1 if event.key == 'ArrowLeft' else 1)
                session.seek(session.frame + step)
            elif event.key == 'Home':
                session.seek(0)
            elif event.key == 'End':
                session.seek(len(session.positions) - 1)
    previous = None
    last_scene = None
    render_thread = None
    last_ui = 0.

    def acknowledge(client, request_id, submitted):
        try:
            image = client.get_render(height=180,width=320)
            session.record_ack(request_id,time.perf_counter()-submitted)
            folder = ROOT / 'review/generated'
            folder.mkdir(parents=True,exist_ok=True)
            Image.fromarray(image).save(folder / f'{request_id}.jpg')
        except Exception:
            pass

    print(f'StageZero studio: http://127.0.0.1:{server.get_port()}',flush=True)
    try:
        while True:
            key = session.tick()
            pose_changed = key != previous
            with session.lock:
                if key != previous:
                    with server.atomic():
                        character.set_pose(torch.from_numpy(session.positions[session.frame]),torch.from_numpy(session.rotations[session.frame]))
                        root = session.positions[session.frame,0].copy()
                        if previous is None or key[0] != previous[0] or not session.playing or abs(key[1]-previous[1]) > 5:
                            camera.rebase(root)
                        camera.update(root)
                    previous = key
                if session.needs_ack and server.get_clients() and (render_thread is None or not render_thread.is_alive()):
                    request_id,submitted = session.needs_ack
                    session.needs_ack = None
                    server.flush()
                    render_thread = threading.Thread(target=acknowledge,args=(next(iter(server.get_clients().values())),request_id,submitted),daemon=True)
                    render_thread.start()
                object_key = (key, session.project_revision)
                if object_key != previous_objects:
                    object_layer.update(session.scene.get('objects', []), session.object_states())
                    previous_objects = object_key
                gate = session.scene['gate']
                gate_open = session.gate_open()
                scene_key = (tuple(gate['position']),gate['radius'],gate['enabled'],gate_open,gate_ui['edit'].value)
                if scene_key != last_scene:
                    for i,post in enumerate(gate_posts):
                        post.position = (gate['position'][0]+(-.84 if i==0 else .84),.825,gate['position'][2])
                        post.visible = gate['enabled']
                    gate_panel.position = (gate['position'][0],2.4 if gate_open else .725,gate['position'][2])
                    gate_panel.visible = gate['enabled']
                    gate_zone.position = (gate['position'][0],.004,gate['position'][2])
                    gate_zone.vertices = np.asarray(zone_mesh.vertices*np.array([gate['radius'],1.,gate['radius']]),dtype=np.float32)
                    gate_zone.visible = gate['enabled']
                    gizmo.visible = gate['enabled'] and gate_ui['edit'].value
                    gizmo.position = (gate['position'][0],0.,gate['position'][2])
                    gate_ui['enabled'].value = gate['enabled']
                    gate_ui['x'].value,gate_ui['z'].value = gate['position'][0],gate['position'][2]
                    gate_ui['radius'].value = gate['radius']
                    gate_ui['status'].content = '**Gate open** · actor entered the activation area' if gate_open else '**Gate ready** · waiting for actor' if gate['enabled'] else 'Gate disabled'
                    last_scene = scene_key
            if pose_changed or time.monotonic()-last_ui >= .1:
                with server.atomic():
                    ui.update()
                    timeline.update()
                last_ui = time.monotonic()
            time.sleep(1/60)
    except KeyboardInterrupt:
        session.reset()
        server.stop()


if __name__ == '__main__':
    main()
