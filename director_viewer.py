"""StageZero motion studio: direct, review, edit, and navigate stored performances."""
import argparse
from pathlib import Path
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
from scene_ground import studio_surface_visibility
from object_controls import add_object_controls
from studio_server import create_studio_server
from studio_camera import StudioCamera
from studio_camera_protocol import CameraStudioController
from studio_timeline import StudioTimeline
from studio_ui import StudioUI, section
from character_controls import CharacterControls
from realtime_client import RealtimeClient
from studio_core_session import CoreStudioSession
from studio_core_controls import CoreStudioControls
from studio_core_renderer import StudioCoreRenderer


MAX_STARTUP_GLB_BYTES = 32 * 1024 * 1024


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=2336)
    parser.add_argument('--project', type=str, help='Open a saved project at startup')
    parser.add_argument('--objects', type=str, help='Load a generated object scene JSON')
    parser.add_argument('--characters', type=Path, default=ROOT / '.runtime/characters', help='Private imported GLB library')
    parser.add_argument('--glb', type=Path, help='Open a local GLB on first browser connection')
    parser.add_argument('--environment', choices=('studio', 'warehouse', 'none'), default='studio',
                        help='Reflection lighting for PBR character materials')
    parser.add_argument('--reference-only', action='store_true',
                        help='Start from a static G1 reference without the private recorded CSV')
    parser.add_argument('--recording', type=Path, default=ROOT / 'assets/recorded_g1.csv',
                        help='Private recorded G1 motion CSV')
    parser.add_argument('--token-path', type=Path, default=ROOT / '.runtime/api-token',
                        help='Private Live ARDY bearer token file')
    parser.add_argument('--backend-url', default='http://127.0.0.1:8765',
                        help='Live ARDY backend URL')
    parser.add_argument('--core-backend-url', default='http://127.0.0.1:8769',
                        help='Existing Core scene-direction service; never starts a worker')
    parser.add_argument('--core-token-path', type=Path,
                        help='Core token file (defaults to --token-path); absent token permits replay only')
    return parser


def load_startup_glb(parser, controls, path):
    if path is None:
        return
    try:
        # Bound the read even if a local file changes after startup begins.
        with path.open('rb') as source:
            data = source.read(MAX_STARTUP_GLB_BYTES + 1)
        if len(data) > MAX_STARTUP_GLB_BYTES:
            raise ValueError('GLB exceeds the 32 MiB import limit')
        asset_id = controls.add_file(data, path.name)
    except (OSError, ValueError) as exc:
        parser.error(f'Cannot load startup GLB: {exc}')
    controls.set_initial_asset(asset_id)


def should_update_fallback_pose(pose_changed, fallback_needed, previous_fallback_needed):
    """Refresh G1 on a new pose or when a paused tab needs it again."""
    return fallback_needed and (pose_changed or not previous_fallback_needed)



class UnconfiguredMotionBackend:
    """Explicit offline viewer backend; it never invents motion or contacts a service."""
    def cancel(self, request_id):
        pass

    def generate(self, request_id, prompt, history):
        raise RuntimeError('G1 service is not configured. Supply --token-path for generation; Core example playback is available offline.')


def create_motion_backend(args):
    if args.reference_only and not args.token_path.is_file():
        return UnconfiguredMotionBackend()
    return Backend(args.token_path, args.backend_url)


def main():
    parser = build_parser()
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.reference_only:
        from ardy.skeleton import G1Skeleton34
        from retargeting import neutral_source_pose
        skeleton = G1Skeleton34()
        pose, orientation = neutral_source_pose(skeleton)
        positions = torch.from_numpy(np.stack([pose, pose]).astype(np.float32))
        rotations = torch.from_numpy(np.stack([orientation, orientation]).astype(np.float32))
    else:
        skeleton, positions, rotations = load_recording(args.recording)
    server = create_studio_server(host='127.0.0.1', port=args.port, label='StageZero Studio', enable_camera_keyboard_controls=False)
    server.gui.configure_theme(dark_mode=True, control_layout='collapsible', control_width='large', show_logo=False, show_share_button=False, brand_color=(126, 224, 195))
    server.scene.set_up_direction('+y')
    server.scene.world_axes.visible = False
    server.scene.configure_environment_map(None if args.environment == 'none' else args.environment)
    server.scene.configure_default_lights(enabled=True, cast_shadow=True)
    server.scene.add_light_ambient('/fill', color=(191, 215, 239), intensity=.6)
    floor = server.scene.add_box('/floor', color=(20, 28, 38), dimensions=(200, .1, 200), position=(0, -.07, 0), cast_shadow=False)
    grid = server.scene.add_grid('/ground-grid', plane='xz', width=200, height=200, cell_size=.5, section_size=2., cell_color=(44, 57, 70), section_color=(68, 87, 100), position=(0, .008, 0), fade_distance=30., fade_strength=2., shadow_opacity=0.)
    stage = trimesh.creation.cylinder(radius=2., height=.035, sections=96)
    stage.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, (1,0,0)))
    platform = server.scene.add_mesh_simple('/stage', vertices=stage.vertices, faces=stage.faces, color=(37, 52, 65), position=(0,-.0175,0), flat_shading=False)
    actor_group = server.scene.add_frame('/actor', show_axes=False)
    character = Character('actor', server, skeleton, create_skeleton_mesh=False, create_skinned_mesh=True, mesh_mode='g1_stl', show_foot_contacts=False)
    if not character.g1_mesh_rig.mesh_handles:
        raise RuntimeError('Supplied G1 meshes are missing; cannot show the preview')
    for mesh in character.g1_mesh_rig.mesh_handles: mesh.color = (206, 226, 233)
    backend = create_motion_backend(args)
    session = DirectorSession(backend, positions.numpy(), rotations.numpy(), ROOT / 'review/live-metrics.jsonl')
    if args.reference_only:
        session.set_mode('Live ARDY')
    characters = CharacterControls(server, session, skeleton, args.characters)
    load_startup_glb(parser, characters, args.glb)
    if args.project:
        session.load_project(Path(args.project).read_bytes())
    if args.objects:
        session.load_objects(args.objects)
    object_layer = ObjectSceneLayer(server)
    previous_objects = None
    token_path = args.core_token_path or args.token_path
    core_client = (RealtimeClient(args.core_backend_url, token_path.read_text())
                   if token_path.is_file() and token_path.read_text().strip() else None)
    core = CoreStudioSession(core_client)
    core_renderer = StudioCoreRenderer(server, name_prefix="/core-cast")
    core_ui = None
    core_requested = False
    last_core_clip = None
    last_main_scene = session.scene_document()
    last_main_scene_revision = session.project_revision
    core_document = core.scene_document
    core_document_epoch = None

    def activate_core(active):
        nonlocal core_requested
        with session.lock:
            if active and session.busy:
                raise ValueError('Finish or cancel the current take generation before switching modes.')
            core_requested = bool(active)
            if active:
                session.pause()
                session.set_character_motion_enabled(False)
                session.status = 'Scene direction is active; your G1 takes remain stored.'
            else:
                entry = characters.active_entry
                session.set_character_motion_enabled(entry is None or entry.retargeter is not None)
        actor_group.visible = not active
        core_renderer.set_visible(active)
        camera.rebase(actor_root())

    def build_core_controls(gui):
        nonlocal core_ui
        core_ui = CoreStudioControls(gui, core, session, on_active=activate_core,
                                    project_folder=ROOT / '.runtime/core-projects')
    def actor_root():
        if core_requested and core.snapshot()['total_frames']:
            return core_renderer.actor_root()
        return characters.actor_root()
    camera = StudioCamera(server, actor_root)
    camera_studio = CameraStudioController(server, session, camera)
    @server.on_client_connect
    def connected(client):
        client.camera.near = .05
        client.camera.far = 250.
        camera.reset(client)
        characters.on_client_connect(client)

    gate_posts = [server.scene.add_box(f'/gate/post{i}', dimensions=(.055,1.65,.08), color=(83,113,131)) for i in range(2)]
    gate_panel = server.scene.add_box('/gate/panel', dimensions=(1.6,1.25,.04), color=(65,147,138), opacity=.28)
    zone_mesh = trimesh.creation.cylinder(radius=1., height=.006, sections=64)
    zone_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2,(1,0,0)))
    gate_zone = server.scene.add_mesh_simple('/gate/zone', vertices=np.asarray(zone_mesh.vertices,dtype=np.float32), faces=np.asarray(zone_mesh.faces,dtype=np.uint32), color=(53,156,141), opacity=.22)
    gizmo = server.scene.add_transform_controls('/gate-edit', scale=.7, active_axes=(True,False,True), disable_rotations=True, translation_limits=((-100,100),(0,0),(-100,100)), visible=False)
    gate_ui = {}
    surface_state = None
    def update_studio_surfaces():
        nonlocal surface_state
        visibility = studio_surface_visibility(
            (core_document if core_requested else session.scene).get('objects', []),
            show_grid=bool(gate_ui['grid'].value),
            show_platform=bool(gate_ui['stage'].value),
        )
        if visibility == surface_state:
            return
        surface_state = visibility
        floor.visible, grid.visible, platform.visible = visibility
        gate_ui['grid'].disabled = gate_ui['stage'].disabled = not floor.visible
        gate_ui['surface_note'].content = ('Scene ground is active; studio grid and platform are hidden to prevent overlap.'
                                           if not floor.visible else '')

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
        gate_ui['surface_note'] = gui.add_markdown('')
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
        def grid_changed(_): update_studio_surfaces()
        @gate_ui['stage'].on_update
        def stage_changed(_): update_studio_surfaces()

    @gizmo.on_update
    def move_gate(_):
        edit_gate(gizmo.position)

    ui = StudioUI(server, session, camera, ROOT / '.runtime/projects', scene_controls,
                  characters.build_gui, core_session=core, core_controls=build_core_controls)
    timeline = StudioTimeline(server, session, command_uuid=ui.timeline_command._impl.uuid,
                              core_session=core)

    @server.scene.on_keyboard_event('keydown')
    def transport_key(event):
        if event.event_type != 'keydown' or event.ctrl_key or event.meta_key or event.alt_key:
            return
        if core_requested:
            state = core.snapshot()
            if state['total_frames']:
                if event.key == ' ':
                    core.pause() if state['playing'] else core.play()
                elif event.key in ('ArrowLeft', 'ArrowRight'):
                    step = (20 if event.shift_key else 1) * (-1 if event.key == 'ArrowLeft' else 1)
                    core.seek(max(0, min(state['total_frames']-1, state['frame']+step)))
                elif event.key == 'Home': core.seek(0)
                elif event.key == 'End': core.seek(state['total_frames']-1)
            return
        with session.lock:
            if not session.character_motion_enabled or session.busy or session.kind not in ('recorded', 'generated'):
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
    previous_character = -1
    previous_fallback_needed = False
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
            characters.tick(key)
            if core_requested:
                session.set_character_motion_enabled(False)
                session.pause()
            if session.project_revision != last_main_scene_revision:
                current_main_scene = session.scene_document()
                if current_main_scene != last_main_scene:
                    core.update_scene(current_main_scene)
                    last_main_scene = current_main_scene
                last_main_scene_revision = session.project_revision
            core_state = core.tick()
            if core_state['epoch'] != core_document_epoch:
                core_document = core.scene_document
                core_document_epoch = core_state['epoch']
            core_key = (core_state.get('epoch'), core_state['revision'], core_state['total_frames'])
            if core_requested and core_state['total_frames']:
                if core_key != last_core_clip:
                    core_renderer.set_clip(core.timeline_clip())
                    last_core_clip = core_key
                core_renderer.tick(core_state['frame'])
            core_renderer.set_visible(core_requested and bool(core_state['total_frames']))
            actor_group.visible = not core_requested
            rendered_root = actor_root()
            if core_requested and core_state['total_frames']:
                camera.update(rendered_root)
            character_changed = characters.revision != previous_character
            pose_changed = key != previous or character_changed
            fallback_needed = characters.renderer.fallback_pose_needed
            fallback_pose_changed = should_update_fallback_pose(
                pose_changed, fallback_needed, previous_fallback_needed)
            with session.lock:
                if pose_changed or fallback_pose_changed:
                    with server.atomic():
                        if fallback_pose_changed:
                            character.set_pose(torch.from_numpy(session.positions[session.frame]),torch.from_numpy(session.rotations[session.frame]))
                        if pose_changed:
                            root = rendered_root
                            if character_changed or previous is None or key[0] != previous[0] or not session.playing or abs(key[1]-previous[1]) > 5:
                                camera.rebase(root)
                            camera.update(root)
                    if pose_changed:
                        previous = key
                        previous_character = characters.revision
                previous_fallback_needed = fallback_needed
                if not core_requested:
                    camera_studio.update()
                if session.needs_ack and server.get_clients() and (render_thread is None or not render_thread.is_alive()):
                    request_id,submitted = session.needs_ack
                    session.needs_ack = None
                    server.flush()
                    render_thread = threading.Thread(target=acknowledge,args=(next(iter(server.get_clients().values())),request_id,submitted),daemon=True)
                    render_thread.start()
                object_key = (key, session.project_revision, core_requested,
                              core_key, core_state['frame'] if core_requested else 0)
                if object_key != previous_objects:
                    if core_requested:
                        doc = core_document
                        objects = doc.get('objects', [])
                        # Native navigation uses the saved authored geometry.
                        # G1 proximity/pickup effects remain in the G1 mode; do
                        # not move props underneath a committed Core route.
                        states = [{'id': o['id'], 'position': o['position'],
                                   'color': o['color'], 'active': False} for o in objects]
                        object_layer.update(objects, {'objects': states,
                            'effects': doc.get('effects', []), 'assets': doc.get('assets', []),
                            'lighting': doc.get('lighting', 'neutral'),
                            'seconds': core_state['frame']/20.})
                    else:
                        object_layer.update(session.scene.get('objects', []), session.object_states())
                    update_studio_surfaces()
                    previous_objects = object_key
                gate = session.scene['gate']
                gate_open = session.gate_open()
                scene_key = (core_requested, tuple(gate['position']),gate['radius'],gate['enabled'],gate_open,gate_ui['edit'].value)
                if scene_key != last_scene:
                    for i,post in enumerate(gate_posts):
                        post.position = (gate['position'][0]+(-.84 if i==0 else .84),.825,gate['position'][2])
                        post.visible = gate['enabled'] and not core_requested
                    gate_panel.position = (gate['position'][0],2.4 if gate_open else .725,gate['position'][2])
                    gate_panel.visible = gate['enabled'] and not core_requested
                    gate_zone.position = (gate['position'][0],.004,gate['position'][2])
                    gate_zone.vertices = np.asarray(zone_mesh.vertices*np.array([gate['radius'],1.,gate['radius']]),dtype=np.float32)
                    gate_zone.visible = gate['enabled'] and not core_requested
                    gizmo.visible = gate['enabled'] and gate_ui['edit'].value and not core_requested
                    gizmo.position = (gate['position'][0],0.,gate['position'][2])
                    gate_ui['enabled'].value = gate['enabled']
                    gate_ui['x'].value,gate_ui['z'].value = gate['position'][0],gate['position'][2]
                    gate_ui['radius'].value = gate['radius']
                    gate_ui['status'].content = '**Gate open** · actor entered the activation area' if gate_open else '**Gate ready** · waiting for actor' if gate['enabled'] else 'Gate disabled'
                    last_scene = scene_key
            ui_due = time.monotonic()-last_ui >= .1
            if pose_changed or ui_due:
                with server.atomic():
                    if ui_due:
                        ui.update()
                        core_ui.tick()
                    timeline.update()
                if ui_due:
                    last_ui = time.monotonic()
            time.sleep(1/60)
    except KeyboardInterrupt:
        session.reset()
        core.close()
        core_renderer.remove()
        characters.close()
        server.stop()


if __name__ == '__main__':
    main()
