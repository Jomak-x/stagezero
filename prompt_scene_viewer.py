"""Prompt-first cast research studio; existing main UI remains available unchanged."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import threading
import time
import numpy as np

# Register custom client messages before an early reconnect can cache Viser types.
from native_pair_playback import NativePairPlaybackController
from studio_camera_protocol import CameraStudioController  # protocol registration
from character_actor import TexturedSkinnedMeshProps  # protocol registration
from studio_server import create_studio_server
from object_scene import ObjectSceneLayer
from paired_scene import scene_copy
from scene_ground import studio_surface_visibility
from native_pair_provider import NativePairProvider
from realtime_client import RealtimeClient
from cast_performance_session import CastPerformanceSession
from cast_performance_renderer import CastPerformanceRenderer
from prompt_scene_plan import ScenePromptPlanner
from prompt_scene_builder import PromptSceneBuilder
from prompt_scene_controls import PromptSceneControls
from native_pair_capture import capture_pair

ROOT = Path(__file__).resolve().parent


class PromptSceneActions:
    """Serialize scene changes with capture acquisition and live rendering."""
    def __init__(self, cast, renderer, backgrounds, current_scene, *, render_lock,
                 draw_scene, build, capture):
        self.cast, self.renderer = cast, renderer
        self.backgrounds = backgrounds
        self.current_scene = scene_copy(current_scene)
        self.render_lock = render_lock
        self.draw_scene, self.build, self.capture = draw_scene, build, capture
        self.exporting = False
        self._hidden_clip = None

    def background_document(self):
        with self.render_lock:
            return scene_copy(self.current_scene)

    def _require_idle(self):
        state = self.cast.snapshot()
        if self.exporting or state['capturing']:
            raise ValueError('Wait for video export to finish.')
        if state['busy']:
            raise ValueError('Finish or cancel generation before changing the scene.')
        return state

    def generate(self, prompt, seed, client):
        with self.render_lock:
            self._require_idle()
            builder = self.build(prompt, seed)
            self.cast.build_performance(builder, self.current_scene, request={'prompt': prompt, 'seed': seed})

    def background(self, label):
        with self.render_lock:
            self._require_idle()
            if label not in self.backgrounds:
                raise ValueError('Choose an available background.')
            # A rejected pause must not switch the scene beneath a capture.
            self.cast.pause()
            self.current_scene = scene_copy(self.backgrounds[label])
            self._hidden_clip = self.cast.timeline_clip()
            self.renderer.set_visible(False)
            self.draw_scene()

    def open_project(self, data):
        with self.render_lock:
            self._require_idle()
            self.cast.load(data)
            self.current_scene = self.cast.scene_document
            self._hidden_clip = None
            self.renderer.set_visible(False)
            self.draw_scene()

    def play_saved(self):
        with self.render_lock:
            self._require_idle()
            self.cast.play()
            self.current_scene = self.cast.scene_document
            self._hidden_clip = None
            self.renderer.set_visible(True)
            self.draw_scene()

    def export(self, client):
        with self.render_lock:
            state = self._require_idle()
            if not state['total_frames']:
                raise ValueError('Open or generate a complete performance before exporting.')
            self.exporting = True
            try:
                self.current_scene = self.cast.scene_document
                self._hidden_clip = None
                self.renderer.set_visible(True)
                self.draw_scene()
            except Exception:
                self.exporting = False
                raise
        try:
            # The reservation above closes the gap before capture() acquires
            # its session lease, without holding the render lock for encoding.
            return self.capture(client)
        finally:
            with self.render_lock:
                self.exporting = False

    def seek(self, frame):
        with self.render_lock:
            state = self.cast.snapshot()
            if (state['total_frames'] and not state['busy'] and not state['capturing']
                    and not self.exporting):
                self.cast.seek(max(0, min(int(frame), state['total_frames'] - 1)))

    def display_new_clip(self, clip, state):
        """Do not undo a background picked after this take finished building."""
        with self.render_lock:
            visible = clip is not self._hidden_clip
            if visible:
                self.current_scene = self.cast.scene_document
            self.renderer.sync_cast(state)
            self.renderer.set_clip(clip)
            self.renderer.set_visible(visible)
            return visible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=2382)
    parser.add_argument('--background', action='append', required=True, help='Label=/absolute/scene.json')
    parser.add_argument('--core-url', default='http://127.0.0.1:8769')
    parser.add_argument('--token-path', type=Path, required=True)
    parser.add_argument('--native-pair-config', type=Path, required=True)
    parser.add_argument('--project', type=Path)
    parser.add_argument('--output-root', type=Path, default=ROOT/'.runtime/prompt-scenes')
    args = parser.parse_args()
    backgrounds = {}
    for item in args.background:
        label, separator, path = item.partition('=')
        if not separator or not label.strip(): parser.error('Each background must be Label=/path/to/scene.json')
        backgrounds[label] = scene_copy(json.loads(Path(path).read_text()))
    core = RealtimeClient(args.core_url, args.token_path.read_text())
    provider = NativePairProvider.from_config(args.native_pair_config)
    planner = ScenePromptPlanner()
    cast = CastPerformanceSession()
    cast.activate()
    current_scene = next(iter(backgrounds.values()))
    if args.project:
        cast.load(args.project.read_bytes())
        current_scene = cast.scene_document
    server = create_studio_server(host='127.0.0.1', port=args.port, label='StageZero · Prompt scenes', enable_camera_keyboard_controls=False)
    server.gui.configure_theme(dark_mode=True, control_layout='collapsible', control_width='large', show_logo=False, show_share_button=False, brand_color=(126,224,195))
    server.scene.set_up_direction('+y')
    server.scene.world_axes.visible = False
    server.scene.configure_default_lights(enabled=True, cast_shadow=True)
    server.scene.add_light_ambient('/fill', color=(191,215,239), intensity=.6)
    floor = server.scene.add_box('/floor', color=(25,33,43), dimensions=(200,.1,200), position=(0,-.07,0), cast_shadow=False)
    layer = ObjectSceneLayer(server)
    renderer = CastPerformanceRenderer(server)
    playback = NativePairPlaybackController(server, get_state=lambda: dict(cast.snapshot(), enabled=True))
    renderer.local_playback = playback
    render_lock = threading.RLock()
    last_clip = None
    last_layout = None
    last_frame = None
    last_ui = 0.

    def draw_scene(seconds=0.):
        current_scene = actions.current_scene
        objects = current_scene.get('objects', [])
        layer.update(objects, {'objects':[{'id':obj['id'],'position':obj['position'],'color':obj['color'],'active':False} for obj in objects],
            'assets':current_scene.get('assets',[]), 'effects':current_scene.get('effects',[]),
            'lighting':current_scene.get('lighting','neutral'), 'seconds':seconds})
        floor.visible = studio_surface_visibility(objects, show_grid=False, show_platform=False)[0]

    def frame_everyone(client):
        if client is None: return
        with render_lock, cast._lock:
            clip = cast.timeline_clip()
            state = cast.snapshot()
            if clip is None: return
            if state['capturing'] or actions.exporting: raise ValueError('Wait for video export to finish.')
            from prompt_scene_camera import prompt_scene_camera_view
            position, center, fov = prompt_scene_camera_view(clip, actions.current_scene,
                frame=state['frame'], aspect=getattr(client.camera,'aspect',16/9))
            client.camera.position, client.camera.look_at, client.camera.fov = position, center, fov

    actions = PromptSceneActions(cast, renderer, backgrounds, current_scene,
        render_lock=render_lock, draw_scene=draw_scene,
        build=lambda prompt, seed: PromptSceneBuilder(prompt, planner, provider, core, args.output_root/'generations', seed=seed),
        capture=lambda client: capture_pair(cast, renderer, client, args.output_root/'videos'/str(time.time_ns()),
            flush=server.flush, render_lock=render_lock,
            render_frame=lambda frame,state: draw_scene(frame/state['fps']),
            archive_name='scene.cast.stagezero.npz'))

    controls = PromptSceneControls(server.gui, cast, backgrounds=backgrounds, on_generate=actions.generate,
        on_background=actions.background, on_frame=frame_everyone, on_export=actions.export,
        on_open=actions.open_project, output_root=args.output_root/'projects', on_play=actions.play_saved,
        get_background=actions.background_document)
    server.timeline.disable_constraints()
    server.timeline.set_visible(False)
    @server.timeline.on_frame_change
    def seek(frame):
        actions.seek(frame)
    @server.on_client_connect
    def connected(client):
        with render_lock:
            client.camera.near, client.camera.far = .05, 250.
            client.camera.position, client.camera.look_at = (6.,4.,6.), (0.,1.,0.)
            if not actions.exporting and not cast.snapshot()['capturing']:
                frame_everyone(client)
    draw_scene()
    print(f'Prompt scene studio: http://127.0.0.1:{args.port}', flush=True)
    try:
        while True:
            # Build completion publishes under the session lock. Read the
            # snapshot and clip together so a new shorter/different cast cannot
            # be paired with an old frame or old cast list.
            with render_lock, cast._lock:
                state = cast.tick()
                if not state['capturing'] and not actions.exporting:
                    clip = cast.timeline_clip()
                    if clip is not None and clip is not last_clip:
                        visible = actions.display_new_clip(clip, state)
                        last_clip = clip
                        draw_scene(state['frame']/state['fps'])
                        if visible:
                            for client in server.get_clients().values(): frame_everyone(client)
                    if clip is not None: renderer.tick(state['frame'])
                    playback.update(state, enabled=clip is not None)
                    draw_scene(state['frame']/state['fps'])
            if time.monotonic()-last_ui >= .1:
                controls.tick()
                with render_lock, cast._lock:
                    state = cast.snapshot()
                    clip = cast.timeline_clip()
                    if clip is not last_layout:
                        server.timeline.clear_prompts()
                        if clip is not None:
                            for index,segment in enumerate(clip.metadata.get('segments', [])):
                                server.timeline.add_prompt(segment['label'],segment['start_frame'],segment['end_frame_exclusive'],color=(72,202,183),uuid=f'cast-{index}')
                            server.timeline.set_fps(30.)
                            server.timeline.set_zoom_settings(default_num_frames_zoom=clip.frames,max_frames_zoom=clip.frames)
                            server.timeline.set_frame_range(0,clip.frames-1)
                        server.timeline.set_visible(clip is not None)
                        last_layout = clip
                    if clip is not None and state['frame'] != last_frame:
                        server.timeline.set_current_frame(state['frame'])
                        last_frame = state['frame']
                last_ui = time.monotonic()
            time.sleep(1/60)
    except KeyboardInterrupt:
        cast.close()
        renderer.remove()
        server.stop()


if __name__ == '__main__': main()
