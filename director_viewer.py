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
from voice_directing import VoiceDirecting
from dialogue_directing import DialogueDirector
from native_pair_session import NativePairSession
from native_pair_renderer import NativePairRenderer
from native_pair_controls import NativePairControls
from native_pair_playback import NativePairPlaybackController  # register local playback protocol
from cast_performance_session import CastPerformanceSession
from cast_performance_renderer import CastPerformanceRenderer
from studio_cast_runtime import NativePlaybackRouter, decode_native_project, set_cast_camera_view


MAX_STARTUP_GLB_BYTES = 32 * 1024 * 1024


def native_core_has_pending_work(snapshot):
    """A playing phase can still have queued or in-flight native work."""
    return (snapshot.get('inflight_request_id') is not None or
            bool(snapshot.get('queued_stages')))


def native_cast_camera_view(clip, placement, cast_roots, *, aspect=16/9):
    """Fit the whole placed performance and standing cast within a 42° view."""
    roots = np.asarray(cast_roots, dtype=float)
    points = [roots + [-.55, -1.1, -.55], roots + [.55, 1.3, .55]]
    if clip is not None:
        angle = np.deg2rad(placement['yaw_degrees'])
        rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                             [-np.sin(angle), 0., np.cos(angle)]])
        world = clip.joints.reshape(-1, 3) @ rotation.T
        points.append(world + [placement['x'], 0., placement['z']])
    points = np.concatenate(points)
    low, high = points.min(axis=0)-.25, points.max(axis=0)+.25
    center = (low+high)/2
    corners = np.array([[x, y, z] for x in (low[0], high[0])
                        for y in (low[1], high[1]) for z in (low[2], high[2])]) - center
    direction = np.array([.95, .48, .8]); direction /= np.linalg.norm(direction)
    right = np.cross([0., 1., 0.], direction); right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    fov = np.deg2rad(42.)
    aspect = float(aspect) if aspect and np.isfinite(aspect) and aspect > 0 else 16/9
    depth = corners @ direction
    distance = max(3., float(np.max(depth + np.abs(corners @ right)/(np.tan(fov/2)*aspect))),
                   float(np.max(depth + np.abs(corners @ up)/np.tan(fov/2)))) * 1.1
    return center + distance*direction, center, fov


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
    parser.add_argument('--native-project', type=Path, help='Open an exact saved native cast performance at startup')
    parser.add_argument('--core-project', type=Path, help='Open an exact Core scene-direction archive for paused replay')
    parser.add_argument('--native-pair-config', type=Path, help='Private native pair configuration; defaults to STAGEZERO_NATIVE_PAIR_CONFIG or .runtime/prompt-native-provider.json')
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
    if args.core_project and (args.project or args.native_project):
        parser.error('--core-project cannot be combined with another startup project')
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
    server.scene.configure_environment_map(None if args.environment == 'none' else args.environment,
                                           background=False, environment_intensity=.15)
    server.scene.configure_default_lights(enabled=False)
    server.scene.add_light_directional('/studio-key', color=(225, 234, 245), intensity=1.6,
                                       position=(3, 6, 4), cast_shadow=True)
    server.scene.add_light_ambient('/fill', color=(191, 215, 239), intensity=.2)
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
    from native_pair_config import resolve_native_pair_provider
    pair_resolution = resolve_native_pair_provider(args.native_pair_config, repo_root=ROOT)
    pair_provider = pair_resolution.provider
    paired = NativePairSession(pair_provider)
    cast = CastPerformanceSession()
    from prompt_scene_plan import ScenePromptPlanner
    cast_planner = ScenePromptPlanner()
    core.update_scene(session.scene_document())
    paired.update_scene(session.scene_document())
    core_renderer = StudioCoreRenderer(server, name_prefix="/core-cast")
    # The opt-in display owns separate rig17 transforms. Never pass those
    # transforms through the ordinary Core retargeter or native history.
    terrain_renderer = None
    paired_renderer = NativePairRenderer(server)
    cast_renderer = CastPerformanceRenderer(server)
    native_render_lock = threading.RLock()
    core_ui = None
    pair_ui = None
    cast_ui = None
    prompt_cast_folder = None
    direction_ui = None
    direction_mode = None
    pair_folder = cast_folder = None
    scene_sync_error = None
    core_requested = False
    paired_requested = False
    cast_requested = False
    native_exporting = False
    last_core_clip = None
    last_core_camera_stream = None
    last_cast_clip = None
    last_main_scene = session.scene_document()
    last_main_scene_revision = session.project_revision
    core_document = core.scene_document
    core_document_epoch = None
    paired_document = paired.scene_document
    paired_document_revision = None
    cast_document = session.scene_document()
    cast_document_revision = None
    native_router = NativePlaybackRouter(server,
        {'paired': (paired, paired_renderer), 'cast': (cast, cast_renderer)}, lock=native_render_lock)
    native_playback = native_router.controller

    def require_native_idle():
        if native_exporting or paired.snapshot()['capturing'] or cast.snapshot()['capturing']:
            raise ValueError('Wait for playback export to finish.')
        if (session.busy or paired.snapshot()['busy'] or cast.snapshot()['busy'] or
                native_core_has_pending_work(core.snapshot())):
            raise ValueError('Finish or cancel current generation before changing modes or opening a project.')

    def activate_mode(mode):
        nonlocal core_requested, paired_requested, cast_requested
        with session.lock, native_render_lock:
            current = 'cast' if cast_requested else 'paired' if paired_requested else 'core' if core_requested else None
            if current == mode:
                return
            require_native_idle()
            if paired_requested:
                paired.deactivate()
            if cast_requested:
                cast.deactivate()
            if core_requested:
                core.deactivate()
            if mode == 'paired':
                paired.activate()
            elif mode == 'cast':
                cast.activate()
            core_requested, paired_requested, cast_requested = mode == 'core', mode == 'paired', mode == 'cast'
            native_router.select(mode if mode in ('paired', 'cast') else None)
            if mode is not None:
                session.pause()
                session.set_character_motion_enabled(False)
                session.status = 'Scene direction is active; your other motion takes remain stored.'
            else:
                entry = characters.active_entry
                session.set_character_motion_enabled(entry is None or entry.retargeter is not None)
            actor_group.visible = mode is None
            terrain_display = core_requested and core.snapshot().get('terrain_aware', False)
            core_renderer.set_visible(core_requested and not terrain_display)
            if terrain_renderer is not None:
                terrain_renderer.set_visible(terrain_display)
            paired_renderer.set_visible(paired_requested)
            cast_renderer.set_visible(cast_requested)
            camera.rebase(actor_root())

    def activate_core(active):
        if active or core_requested:
            activate_mode('core' if active else None)

    def activate_paired(active):
        if active or paired_requested:
            activate_mode('paired' if active else None)

    def activate_cast(active):
        if active or cast_requested:
            activate_mode('cast' if active else None)

    def activate_story():
        with session.lock, native_render_lock:
            require_native_idle()
            activate_mode(None)

    def render_native_capture_frame(frame, state):
        native_playback.update(native_router.get_state(), enabled=paired_requested or cast_requested)
        # Called under native_render_lock after the exact actor pose is set.
        # Do not acquire the main session lock here: live updates acquire that
        # lock before native_render_lock. The paired scene is immutable during
        # its capture lease.
        camera.update(paired_renderer.actor_root())
        doc = paired.scene_document
        objects = doc.get('objects', [])
        states = [{'id': o['id'], 'position': o['position'],
                   'color': o['color'], 'active': False} for o in objects]
        object_layer.update(objects, {'objects': states,
            'effects': doc.get('effects', []), 'assets': doc.get('assets', []),
            'lighting': doc.get('lighting', 'neutral'),
            'seconds': frame / float(state['fps'])})

    def export_native_performance(client, *, composed=False):
        nonlocal previous_objects, native_exporting
        from native_pair_capture import capture_pair
        motion, renderer = (cast, cast_renderer) if composed else (paired, paired_renderer)
        with session.lock, native_render_lock:
            require_native_idle()
            if not motion.snapshot()['active'] or not motion.snapshot()['total_frames']:
                raise ValueError('Open or generate a complete performance before exporting.')
            direction_marks.hide()
            native_router.refresh()
            native_exporting = True
        try:
            return capture_pair(motion, renderer, client,
                ROOT / '.runtime' / ('cast-videos' if composed else 'native-pair-videos') / str(time.time_ns()),
                flush=server.flush, render_lock=native_render_lock,
                render_frame=render_cast_capture_frame if composed else render_native_capture_frame,
                archive_name='scene.cast.stagezero.npz' if composed else 'scene.native-pair.stagezero.npz')
        finally:
            with native_render_lock:
                native_exporting = False
                previous_objects = None

    def export_native_pair(client):
        return export_native_performance(client)

    def render_cast_capture_frame(frame, state):
        # The cast scene is immutable under its capture lease. Never acquire
        # the main session lock while holding the capture render lock.
        camera.update(cast_renderer.actor_root())
        doc = cast.scene_document
        objects = doc.get('objects', [])
        object_layer.update(objects, {'objects': [{'id': obj['id'], 'position': obj['position'],
            'color': obj['color'], 'active': False} for obj in objects],
            'effects': doc.get('effects', []), 'assets': doc.get('assets', []),
            'lighting': doc.get('lighting', 'neutral'), 'seconds': frame / state['fps']})

    def frame_prompt_cast(client):
        if client is None:
            return
        from prompt_scene_camera import prompt_scene_camera_view
        with session.lock, native_render_lock, cast._lock:
            if native_exporting or cast.snapshot()['capturing']:
                raise ValueError('Wait for playback export to finish.')
            clip, state = cast.timeline_clip(), cast.snapshot()
            if clip is None:
                return
            native_router.refresh()
            position, center, fov = prompt_scene_camera_view(clip, cast.scene_document,
                frame=state['frame'], aspect=getattr(client.camera, 'aspect', 16/9))
            camera._manual(client)
            set_cast_camera_view(client, position, center, fov)

    def cast_generation_readiness(source="generate"):
        if core_client is None:
            return "Configure the Core motion connection before generating performers."
        if source == "generate" and pair_provider is None:
            return pair_resolution.status
        return None

    def generate_prompt_cast(prompt, seed, client, *, actor_count=None):
        from prompt_scene_builder import PromptSceneBuilder
        with session.lock, native_render_lock:
            require_native_idle()
            readiness = cast_generation_readiness()
            if readiness:
                raise ValueError(readiness)
            planner = cast_planner if actor_count is None else ScenePromptPlanner(expected_actor_count=actor_count)
            builder = PromptSceneBuilder(prompt, planner, pair_provider, core_client,
                                         ROOT / '.runtime/prompt-scenes/generations', seed=seed)
            activate_cast(True)
            cast.build_performance(builder, session.scene_document(), request={'prompt': prompt, 'seed': seed})
            direction_marks.hide()
        return cast

    def generate_scene_cast(prompt, seconds, client, *, actor_count):
        if seconds is not None:
            raise ValueError("Choose Auto length for a multi-person scene; describe action timing in the prompt.")
        seed = int(cast_ui.seed.value) if cast_ui is not None else 42
        generate_prompt_cast(prompt, seed, client, actor_count=actor_count)
        if cast_ui is not None:
            cast_ui.prompt.value = prompt
        return True

    def frame_native_cast(client):
        if client is None:
            return
        with session.lock, native_render_lock:
            state = paired.snapshot()
            if native_exporting or state.get('capturing'):
                raise RuntimeError('Playback export is running; wait before framing the cast')
            camera._manual(client)
            paired_renderer.sync_cast(state)
            clip = paired.timeline_clip()
            if state['total_frames']:
                paired_renderer.set_clip(clip)
                paired_renderer.tick(state['frame'])
            roots = np.array([paired_renderer.actor_root(a['id']) for a in state['cast']])
            position, center, fov = native_cast_camera_view(
                clip, state['placement'], roots, aspect=getattr(client.camera, 'aspect', 16/9))
            client.camera.position = position
            client.camera.look_at = center
            client.camera.fov = fov

    def build_native_context():
        from native_pair_context import build_studio_context
        return paired.build_context(lambda clip, scene_document, cancelled:
            build_studio_context(clip, core_client, scene_document, paired.snapshot()['placement'],
                                 ROOT / '.runtime/native-pair-context', cancelled=cancelled))

    from paired_direction_preview import PairedDirectionPreview
    direction_marks = PairedDirectionPreview(server, lambda request: direction_ui.set_marks(request))

    def preview_paired_direction(request, client):
        from paired_direction import preview_direction
        result = preview_direction(request, session.scene_document())
        direction_marks.show(result['request'], result['plan'])
        if client is not None:
            points = np.array([[p['x'], .8, p['z']] for p in [*result['request']['starts'], result['request']['meeting']]])
            position, center, fov = native_cast_camera_view(None, {'x':0.,'z':0.,'yaw_degrees':0.}, points,
                                                         aspect=getattr(client.camera, 'aspect', 16/9))
            camera._manual(client)
            client.camera.position = position
            client.camera.look_at = center
            client.camera.fov = fov
        return result

    def generate_paired_direction(request, client):
        from paired_direction import PairedSceneBuilder, validate_request
        request = validate_request(request)
        with session.lock, native_render_lock:
            require_native_idle()
            if request['source'] == 'generate':
                readiness = cast_generation_readiness()
                if readiness:
                    raise ValueError(readiness)
            request = preview_paired_direction(request, client)['request']
            activate_paired(True)
            paired.select_pair(*request['actor_ids'])
            builder = PairedSceneBuilder(request, pair_provider, core_client, ROOT / '.runtime/paired-scenes')
            paired.build_performance(builder, session.scene_document(), request=request)
            direction_marks.hide()
        return paired

    def open_native_project(data):
        kind, decoded = decode_native_project(data)
        with session.lock, native_render_lock:
            require_native_idle()
            motion = cast if kind == 'cast' else paired
            scene = decoded[2] if kind == 'cast' else decoded[3]
            # Validate the Scene tab representation before changing either take.
            session._document(scene)
            if motion.snapshot()['total_frames']:
                suffix = 'cast' if kind == 'cast' else 'native-pair'
                backup = ROOT / '.runtime' / (suffix + '-projects') / ('before-open-' + str(time.time_ns()) + '.' + suffix + '.stagezero.npz')
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.write_bytes(motion.save())
            motion.load(data)
            session.load_scene_document(scene)
            activate_mode(kind)
            native_router.refresh()
            direction_marks.hide()
            if kind == 'paired' and direction_ui is not None:
                request = decoded[0].metadata.get('direction_request')
                if request:
                    direction_ui.restore_request(request)
        # Controls callbacks acquire their lock before invoking this action.
        # Notify outside the main/render transaction to keep one lock order.
        if kind == 'cast' and cast_ui is not None:
            cast_ui.mark_loaded()

    def open_g1_project(data):
        with session.lock, native_render_lock:
            require_native_idle()
            session.load_project(data)
            if cast_requested:
                activate_cast(False)
            if paired_requested:
                activate_paired(False)
            if core_requested:
                core.deactivate()
                activate_core(False)
            direction_marks.hide()

    def build_core_controls(gui):
        nonlocal core_ui, pair_ui, cast_ui, direction_ui, direction_mode, pair_folder, cast_folder, prompt_cast_folder
        from paired_direction_controls import PairedDirectionControls
        from studio_cast_controls import StudioCastControls
        direction_mode = gui.add_dropdown('Direct', ('One character', 'Two characters', 'AI cast · 1–3 people'), initial_value='One character')
        with gui.add_folder('AI cast · 1–3 people', expand_by_default=True) as prompt_cast_folder:
            cast_ui = StudioCastControls(gui, cast, on_generate=generate_prompt_cast,
                on_frame=frame_prompt_cast, on_export=lambda client: export_native_performance(client, composed=True),
                on_open=open_native_project, provider_available=pair_provider is not None and core_client is not None,
                output_root=ROOT / '.runtime/cast-projects', generation_readiness=cast_generation_readiness)
        prompt_cast_folder.visible = False
        with gui.add_folder('Two-person scene', expand_by_default=True) as pair_folder:
            direction_ui = PairedDirectionControls(gui, paired, on_generate=generate_paired_direction,
                on_preview=preview_paired_direction, on_frame_cast=frame_native_cast,
                on_active=activate_paired, scene_provider=session.scene_document,
                on_export=lambda client: pair_ui.start_capture(client), on_edit=direction_marks.hide,
                generation_readiness=cast_generation_readiness)
        with gui.add_folder('Cast, playback and files', expand_by_default=False) as cast_folder:
            pair_ui = NativePairControls(gui, paired, session, on_active=activate_paired,
                project_folder=ROOT / '.runtime/native-pair-projects', on_capture=export_native_pair,
                on_frame_cast=frame_native_cast, on_build_context=build_native_context, on_open=open_native_project)
        pair_folder.visible = cast_folder.visible = False
        with gui.add_folder('Advanced scene motion', expand_by_default=False):
            core_ui = CoreStudioControls(gui, core, session, on_active=activate_core,
                                        project_folder=ROOT / '.runtime/core-projects')
        @direction_mode.on_update
        def change_direction_mode(event):
            wanted = {'One character': None, 'Two characters': 'paired', 'AI cast · 1–3 people': 'cast'}[direction_mode.value]
            # Core remains the advanced one-character mode.
            if wanted is None and core_requested:
                return
            try:
                activate_mode(wanted)
                direction_marks.hide()
            except (ValueError, RuntimeError) as exc:
                session.project_status = str(exc)
                direction_mode.value = 'AI cast · 1–3 people' if cast_requested else 'Two characters' if paired_requested else 'One character'
    def actor_root():
        if cast_requested:
            return cast_renderer.actor_root()
        if paired_requested:
            return paired_renderer.actor_root()
        if core_requested and core.snapshot()['total_frames']:
            if core.snapshot().get('terrain_aware', False):
                with core._lock:
                    presentation = core.terrain_presentation()
                    if presentation is not None:
                        frame = min(core.snapshot()['frame'], presentation.frames - 1)
                        return tuple(float(value) for value in presentation.positions[0, frame, 0])
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
        if cast_requested and not native_exporting and not cast.snapshot()['capturing']:
            frame_prompt_cast(client)

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
            (cast_document if cast_requested else paired_document if paired_requested else core_document if core_requested
             else session.scene).get('objects', []),
            show_grid=bool(gate_ui['grid'].value),
            show_platform=bool(gate_ui['stage'].value),
        )
        if core_requested and core.snapshot().get('terrain_aware', False):
            # Terrain takes use their authored support, including narrow
            # courtyards. A generic stage/grid would overlap it or hide gaps.
            visibility = (False, False, False)
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
                  characters.build_gui, core_session=core, paired_session=paired, cast_session=cast,
                  core_controls=build_core_controls, on_native_open=open_native_project, on_g1_open=open_g1_project,
                  on_story_activate=activate_story, on_generate_cast=generate_scene_cast)
    timeline = StudioTimeline(server, session, command_uuid=ui.timeline_command._impl.uuid,
                              core_session=core, paired_session=paired, cast_session=cast)

    if args.native_project:
        open_native_project(args.native_project.read_bytes())
    if args.core_project:
        core.load_project(args.core_project.read_bytes())
        session.load_scene_document(core.scene_document)
        core.pause()
        core.seek(0)
        activate_core(True)

    def activate_voice_motion():
        activate_story()

    voice = VoiceDirecting(server, session, workflow=ui.story_controls.workflow,
                           on_single_action_submitted=ui.sync_voice_action,
                           on_story_submitted=ui.story_controls.register_voice_job,
                           on_motion_activate=activate_voice_motion)

    dialogue = DialogueDirector(session, server, enabled=lambda: not (core_requested or paired_requested or cast_requested))
    server.on_client_connect(dialogue.connected)
    for client in server.get_clients().values():
        dialogue.connected(client)

    @server.scene.on_keyboard_event('keydown')
    def transport_key(event):
        if event.event_type != 'keydown' or event.ctrl_key or event.meta_key or event.alt_key:
            return
        if core_requested or paired_requested or cast_requested:
            motion = cast if cast_requested else paired if paired_requested else core
            state = motion.snapshot()
            if native_exporting or state.get('capturing') or state.get('busy'):
                return
            if state['total_frames']:
                if event.key == ' ':
                    motion.pause() if state['playing'] else motion.play()
                elif event.key in ('ArrowLeft', 'ArrowRight'):
                    step = (int(state.get('fps', 20)) if event.shift_key else 1) * (-1 if event.key == 'ArrowLeft' else 1)
                    motion.seek(max(0, min(state['total_frames']-1, state['frame']+step)))
                elif event.key == 'Home': motion.seek(0)
                elif event.key == 'End': motion.seek(state['total_frames']-1)
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
            if core_requested or paired_requested or cast_requested:
                session.set_character_motion_enabled(False)
                session.pause()
            pose_changed = key != previous or characters.revision != previous_character
            # Main-session lock always precedes the render lock; capture never
            # acquires the main-session lock while holding the render lock.
            # Keep the native clock, scene snapshot and optional rig17 display
            # on one committed revision while a terrain append publishes.
            with session.lock, native_render_lock, core._lock:
                if not native_exporting and not paired.snapshot().get('capturing') and not cast.snapshot()['capturing']:
                    if session.project_revision != last_main_scene_revision:
                        current_main_scene = session.scene_document()
                        if current_main_scene != last_main_scene:
                            core.update_scene(current_main_scene)
                            try:
                                paired.update_scene(current_main_scene)
                                scene_sync_error = None
                            except ValueError as exc:
                                scene_sync_error = str(exc)
                                session.project_status = 'Background changed; retained paired take uses its saved scene. Generate a new scene to apply these changes. ' + str(exc)
                            if (cast_requested and cast.snapshot()['total_frames'] and
                                    current_main_scene != session._document(cast.scene_document)):
                                session.project_status = 'Scene changed. The retained AI cast take uses its saved scene; generate a new performance to apply the Scene tab changes.'
                            last_main_scene = current_main_scene
                        last_main_scene_revision = session.project_revision
                    core_state = core.tick()
                    paired_state = paired.tick()
                    # Hold the publication lock through render and document
                    # selection so a completed build cannot mix old/new scenes.
                    with cast._lock:
                        cast_state = cast.tick()
                        native_router.refresh()
                        cast_state = cast.snapshot()
                        if cast_state['revision'] != cast_document_revision:
                            cast_document = cast.scene_document if cast_state['total_frames'] else session.scene_document()
                            cast_document_revision = cast_state['revision']
                        cast_clip = cast.timeline_clip()
                        if cast_requested and cast_clip is not None and cast_clip is not last_cast_clip:
                            last_cast_clip = cast_clip
                            for client in server.get_clients().values():
                                frame_prompt_cast(client)
                    if cast_requested and not cast_state['total_frames']:
                        cast_document = session.scene_document()
                    document_key = (core_state['epoch'], core_state.get('terrain_aware', False),
                                    core_state.get('terrain_display_revision', 0))
                    if document_key != core_document_epoch:
                        core_document = core.scene_document
                        core_document_epoch = document_key
                    if paired_state['revision'] != paired_document_revision:
                        paired_document = paired.scene_document
                        paired_document_revision = paired_state['revision']
                    terrain_display = core_state.get('terrain_aware', False)
                    core_key = (core_state.get('epoch'), core_state['revision'], core_state['total_frames'],
                                terrain_display, core_state.get('terrain_display_revision', 0))
                    paired_key = (paired_state.get('epoch'), paired_state['revision'], paired_state['total_frames'])
                    if core_requested and core_state['total_frames']:
                        if terrain_display:
                            presentation = core.terrain_presentation()
                            if presentation is not None:
                                if terrain_renderer is None:
                                    from terrain_assisted_renderer import TerrainAssistedRenderer
                                    terrain_renderer = TerrainAssistedRenderer(server, name_prefix="/terrain-core-cast")
                                if core_key != last_core_clip:
                                    terrain_renderer.set_presentation(presentation, actor_ids=presentation.actor_ids)
                                    last_core_clip = core_key
                                terrain_renderer.tick(core_state['frame'])
                        else:
                            if core_key != last_core_clip:
                                core_renderer.set_clip(core.timeline_clip())
                                last_core_clip = core_key
                            core_renderer.tick(core_state['frame'])
                    camera_stream = (core_state.get('epoch'), terrain_display,
                                     core_state.get('terrain_display_revision', 0) if terrain_display else None)
                    if core_requested and camera_stream != last_core_camera_stream:
                        camera.rebase(actor_root())
                        last_core_camera_stream = camera_stream
                    core_renderer.set_visible(core_requested and not terrain_display and bool(core_state['total_frames']))
                    if terrain_renderer is not None:
                        terrain_renderer.set_visible(core_requested and terrain_display and bool(core_state['total_frames']))
                    paired_renderer.set_visible(paired_requested)
                    cast_renderer.set_visible(cast_requested and bool(cast_state['total_frames']))
                    actor_group.visible = not (core_requested or paired_requested or cast_requested)
                    rendered_root = actor_root()
                    if ((core_requested and core_state['total_frames']) or
                            (paired_requested and paired_state['total_frames']) or
                            (cast_requested and cast_state['total_frames'])):
                        camera.update(rendered_root)
                    character_changed = characters.revision != previous_character
                    pose_changed = key != previous or character_changed
                    fallback_needed = characters.renderer.fallback_pose_needed
                    fallback_pose_changed = should_update_fallback_pose(
                        pose_changed, fallback_needed, previous_fallback_needed)
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
                    if not (core_requested or paired_requested or cast_requested):
                        camera_studio.update()
                    if session.needs_ack and server.get_clients() and (render_thread is None or not render_thread.is_alive()):
                        request_id,submitted = session.needs_ack
                        session.needs_ack = None
                        server.flush()
                        render_thread = threading.Thread(target=acknowledge,args=(next(iter(server.get_clients().values())),request_id,submitted),daemon=True)
                        render_thread.start()
                    object_key = (key, session.project_revision, core_requested, paired_requested, cast_requested,
                                  cast_state['revision'], cast_state['frame'] if cast_requested else 0,
                                  core_key, paired_key,
                                  core_state['frame'] if core_requested else
                                  paired_state['frame'] if paired_requested else 0)
                    if object_key != previous_objects:
                        if core_requested or paired_requested or cast_requested:
                            doc = cast_document if cast_requested else paired_document if paired_requested else core_document
                            display_frame = cast_state['frame'] if cast_requested else paired_state['frame'] if paired_requested else core_state['frame']
                            objects = doc.get('objects', [])
                            # Native and paired archives retain a scene snapshot for
                            # playback. InterGen did not condition its generation on
                            # these objects; the UI labels it a backdrop.
                            states = [{'id': o['id'], 'position': o['position'],
                                       'color': o['color'], 'active': False} for o in objects]
                            if core_requested and core_state.get('scene_reactions_enabled'):
                                from core_scene_reactions import object_states as core_object_states
                                states = core_object_states(doc, core.timeline_clip(), int(display_frame),
                                    enabled=True, start_frame=core_state.get('scene_reactions_start_frame', 0),
                                    terrain=core_state.get('terrain_navigation_enabled', False),
                                    terrain_start_frame=core_state.get('terrain_navigation_start_frame', 0))
                            object_layer.update(objects, {'objects': states,
                                'effects': doc.get('effects', []), 'assets': doc.get('assets', []),
                                'lighting': doc.get('lighting', 'neutral'),
                                'seconds': display_frame / float(cast_state['fps'] if cast_requested else paired_state['fps'] if paired_requested else core_state['fps'])})
                        else:
                            object_layer.update(session.scene.get('objects', []), session.object_states())
                        update_studio_surfaces()
                        previous_objects = object_key
                    gate = session.scene['gate']
                    gate_open = session.gate_open()
                    scene_key = (core_requested, paired_requested, cast_requested, tuple(gate['position']),gate['radius'],gate['enabled'],gate_open,gate_ui['edit'].value)
                    if scene_key != last_scene:
                        for i,post in enumerate(gate_posts):
                            post.position = (gate['position'][0]+(-.84 if i==0 else .84),.825,gate['position'][2])
                            post.visible = gate['enabled'] and not (core_requested or paired_requested or cast_requested)
                        gate_panel.position = (gate['position'][0],2.4 if gate_open else .725,gate['position'][2])
                        gate_panel.visible = gate['enabled'] and not (core_requested or paired_requested or cast_requested)
                        gate_zone.position = (gate['position'][0],.004,gate['position'][2])
                        gate_zone.vertices = np.asarray(zone_mesh.vertices*np.array([gate['radius'],1.,gate['radius']]),dtype=np.float32)
                        gate_zone.visible = gate['enabled'] and not (core_requested or paired_requested or cast_requested)
                        gizmo.visible = gate['enabled'] and gate_ui['edit'].value and not (core_requested or paired_requested or cast_requested)
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
                        voice.update()
                        dialogue.update()
                        pair_ui.tick()
                        cast_ui.tick()
                        prompt_cast_folder.visible = cast_requested
                        direction_ui.tick()
                        pair_folder.visible = cast_folder.visible = paired_requested
                        expected_mode = 'AI cast · 1–3 people' if cast_requested else 'Two characters' if paired_requested else 'One character'
                        if direction_mode.value != expected_mode:
                            direction_mode.value = expected_mode
                    timeline.update()
                if ui_due:
                    last_ui = time.monotonic()
            time.sleep(1/60)
    except KeyboardInterrupt:
        voice.close()
        dialogue.close()
        ui.story_controls.close()
        session.reset()
        core.close()
        paired.close()
        cast.close()
        core_renderer.remove()
        if terrain_renderer is not None:
            terrain_renderer.remove()
        paired_renderer.remove()
        cast_renderer.remove()
        characters.close()
        server.stop()


if __name__ == '__main__':
    main()
