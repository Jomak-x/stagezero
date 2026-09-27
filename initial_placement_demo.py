"""Interactive Neon staging: live plans, real character meshes, editable marks.

This previews initial placement only. Exported plans feed trial_prompt_scene.py;
this viewer does not claim to generate new Core/InterGen motion.
"""
from __future__ import annotations

import argparse
from html import escape
import json
import math
import os
import queue
from pathlib import Path
import shlex
import threading
import time
from unittest.mock import patch

import numpy as np

from experiments.native_pair_rig import NativeRigActor, NativeRigAsset
from initial_placement_session import InitialPlacementSession
from object_generation import GatewayGenerator
from prompt_scene_builder import place_initial_pose
from prompt_scene_plan import ScenePromptPlanner
from studio_server import create_studio_server

ROOT = Path(__file__).resolve().parent
COLORS = [(62, 218, 219), (255, 184, 82), (191, 147, 255)]
SCENE = {'version': 2, 'name': 'Open staging studio', 'objects': [], 'effects': [], 'lighting': 'neutral'}
PROMPT = ('Three people named Alice, Bob and Carol stand on a stage. Alice is on the left facing Bob, '
          'Bob is on the right facing Alice, and Carol stands behind them facing the audience (+Z). '
          'Choose their exact starting positions with comfortable space between them. Each waves hello in turn.')


class PlacementDemo:
    def __init__(self, server, session, output):
        self.server, self.session, self.output = server, session, output
        self.lock = threading.RLock()
        self.busy = False
        self.syncing = False
        self.selected = None
        self.display = {}
        self.events = []
        self.asset = NativeRigAsset(ROOT/'assets/paired/Xbot.glb')
        with np.load(ROOT/'review/prompt-scenes/solo-city-capture/scene.cast.stagezero.npz', allow_pickle=False) as source:
            self.pose, _, _ = place_initial_pose(source['joints'][0, 0], {'x': 0., 'z': 0., 'yaw_degrees': 0.})
        server.scene.set_up_direction('+y')
        server.scene.world_axes.visible = False
        server.scene.configure_environment_map('studio', background=False, environment_intensity=.35)
        server.scene.configure_default_lights(enabled=False)
        server.scene.add_light_directional('/key', color=(230, 238, 255), intensity=2., position=(3, 7, 5), cast_shadow=True)
        server.scene.add_light_ambient('/fill', color=(190, 211, 240), intensity=.6)
        server.scene.add_box('/floor', color=(30, 40, 56), dimensions=(70, .1, 70), position=(0, -.06, 0))
        server.scene.add_grid('/grid', plane='xz', width=60, height=60, cell_size=1., section_size=5.,
            cell_color=(67, 80, 97), section_color=(95, 115, 136), position=(0, .001, 0), fade_distance=20.)
        server.scene.add_label('/audience', 'AUDIENCE  ·  +Z', position=(0, .04, 5.))
        gui = server.gui
        gui.configure_theme(dark_mode=True, control_layout='floating', control_width='medium',
                            show_logo=False, show_share_button=False, brand_color=(62, 218, 219))
        gui.add_markdown('## Direct the starting scene\nLet Neon place your cast. Then select a person and drag their arrows or turn the ring.')
        self.prompt = gui.add_text('Scene direction', initial_value=PROMPT, multiline=True)
        self.generate = gui.add_button('Place with Neon', color='teal')
        self.status = gui.add_html('<p>Ready for a live Neon plan.</p>')
        self.actors = gui.add_dropdown('Selected character', ('No cast yet',), initial_value='No cast yet')
        self.mode = gui.add_dropdown('Handle', ('Move', 'Turn'), initial_value='Move')
        self.x = gui.add_number('X · metres', initial_value=0., min=-24., max=24., step=.1)
        self.z = gui.add_number('Z · metres', initial_value=0., min=-24., max=24., step=.1)
        self.yaw = gui.add_slider('Facing · degrees', min=-180., max=180., step=1., initial_value=0.)
        self.summary = gui.add_html('')
        self.reset = gui.add_button('Restore AI placement', color='gray')
        self.frame = gui.add_button('Frame everyone', color='gray')
        self.export = gui.add_button('Export edited plan', color='gray')
        self.capture = gui.add_button('Save current view', color='gray')
        gui.add_markdown('Initial-placement preview using the studio character meshes. Export the plan to generate a performance with the same marks. No new motion is generated here.')
        self.gizmo = server.scene.add_transform_controls('/edit-start', scale=1.2,
            active_axes=(True, False, True), disable_rotations=True,
            translation_limits=((-24,24),(0,0),(-24,24)), depth_test=False, visible=False)
        self.generate.on_click(lambda event: self.generate_plan())
        self.frame.on_click(lambda event: self.frame_clients())
        self.reset.on_click(lambda event: self.restore())
        self.export.on_click(self.export_plan)
        self.capture.on_click(self.save_view)
        self.actors.on_update(self.select_changed)
        self.mode.on_update(self.mode_changed)
        for handle in (self.x, self.z, self.yaw):
            handle.on_update(self.fields_changed)
        self.gizmo.on_update(self.dragged)
        server.on_client_connect(lambda client: self.camera(client))
        self.sync()

    def enable(self):
        ready = self.selected is not None
        for h in (self.generate, self.prompt):
            h.disabled = self.busy
        for h in (self.actors, self.mode, self.x, self.z, self.yaw, self.reset, self.export):
            h.disabled = self.busy or not ready
        self.gizmo.visible = ready and not self.busy

    def camera(self, client):
        state = self.session.snapshot()
        points = list((state.get('placement') or {}).get('starts', {}).values())
        x = sum(p['x'] for p in points)/len(points) if points else 0.
        z = sum(p['z'] for p in points)/len(points) if points else 0.
        span = max([abs(p[a]-(x if a == 'x' else z)) for p in points for a in ('x','z')] or [2.])
        distance = max(8., span*3.)
        client.camera.up_direction = (0., 1., 0.)
        client.camera.position = (x+distance*.5, 5., z+distance)
        client.camera.look_at = (x, .7, z)
        client.camera.fov = math.radians(45.)

    def frame_clients(self):
        for client in self.server.get_clients().values():
            self.camera(client)

    def persist(self, action):
        state = self.session.snapshot()
        self.events.append({'action': action, 'time': time.time(), 'placement': state.get('placement')})
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output/'state.json').write_text(json.dumps(state, indent=2)+'\n')
        (self.output/'events.json').write_text(json.dumps(self.events, indent=2)+'\n')
        (self.output/'edited-plan.json').write_text(json.dumps(self.session.export_plan(), indent=2)+'\n')
        (self.output/'scene.json').write_text(json.dumps(SCENE, indent=2)+'\n')

    def sync(self):
        state = self.session.snapshot()
        if not state.get('plan'):
            self.enable(); return
        actors = state['plan']['actors']; starts = state['placement']['starts']
        ids = [actor['id'] for actor in actors]
        self.syncing = True
        try:
            for aid in list(self.display):
                if aid not in ids:
                    self.display[aid]['body'].remove()
                    self.display[aid]['group'].remove()
                    del self.display[aid]
            labels = {f"{actor['name']} · {actor['id']}": actor['id'] for actor in actors}
            self.actor_labels = labels
            if self.selected not in ids:
                self.selected = ids[0]
            for i, actor in enumerate(actors):
                aid = actor['id']; point = starts[aid]; color = COLORS[i]
                if aid not in self.display:
                    group = self.server.scene.add_frame('/cast/'+aid, show_axes=False)
                    body = NativeRigActor(self.server, '/cast/'+aid+'/body', self.asset, color)
                    body.set_pose(self.pose)
                    for mesh in body.handles:
                        mesh.on_click(lambda event, actor_id=aid: self.select(actor_id))
                    label = self.server.scene.add_label('/cast/'+aid+'/name', actor['name'], position=(0, 2.1, 0))
                    arrow = np.array([[[0,.03,0],[0,.03,1.]], [[0,.03,1.],[-.18,.03,.73]], [[0,.03,1.],[.18,.03,.73]]],dtype=np.float32)
                    self.server.scene.add_line_segments('/cast/'+aid+'/facing', points=arrow, colors=color, line_width=5.)
                    self.display[aid] = {'body':body,'group':group,'label':label}
                node = self.display[aid]
                node['label'].text = actor['name'] + (' · selected' if aid == self.selected else '')
                half = math.radians(point['yaw_degrees'])/2
                node['group'].position = (point['x'],0.,point['z'])
                node['group'].wxyz = (math.cos(half),0.,math.sin(half),0.)
            self.actors.options = tuple(labels)
            self.actors.value = next(label for label, aid in labels.items() if aid == self.selected)
            point = starts[self.selected]
            self.x.value,self.z.value,self.yaw.value = point['x'],point['z'],point['yaw_degrees']
            self.gizmo.position = (point['x'],0.,point['z'])
            half = math.radians(point['yaw_degrees'])/2
            self.gizmo.wxyz = (math.cos(half),0.,math.sin(half),0.) if self.mode.value == 'Turn' else (1.,0.,0.,0.)
            rows = ''.join(f"<div style='margin:8px 0'><b>{escape(actor['name'])}</b> · X {starts[actor['id']]['x']:.1f}, Z {starts[actor['id']]['z']:.1f} · {starts[actor['id']]['yaw_degrees']:.0f}°</div>" for actor in actors)
            self.summary.content = rows
            self.enable()
        finally:
            self.syncing = False

    def select(self, actor_id):
        with self.lock:
            if self.busy: return
            self.selected = actor_id; self.sync()

    def select_changed(self, event):
        if self.syncing or event.client is None: return
        self.select(self.actor_labels[self.actors.value])

    def mode_changed(self, event):
        if self.syncing or event.client is None: return
        with self.lock:
            turn = self.mode.value == 'Turn'
            # PivotControls uses the two active axes to choose the rotation
            # plane: XZ produces the Y-axis facing ring.
            self.gizmo.active_axes = (True,False,True)
            self.gizmo.disable_axes = self.gizmo.disable_sliders = turn
            self.gizmo.disable_rotations = not turn
            self.sync()

    def edit(self, action, **fields):
        with self.lock:
            if self.busy or self.selected is None: return
            try:
                self.session.edit_actor(self.selected, **fields)
                self.persist(action)
                self.status.content = '<p>Manual placement saved. Export to keep these marks.</p>'
            except ValueError as exc:
                self.status.content = '<p>'+escape(str(exc))+'</p>'
            self.sync()

    def fields_changed(self, event):
        if self.syncing or event.client is None: return
        self.edit('manual fields', x=self.x.value, z=self.z.value, yaw_degrees=self.yaw.value)

    def dragged(self, event):
        if self.syncing or event.client is None: return
        if self.mode.value == 'Turn':
            w,x,y,z = self.gizmo.wxyz
            yaw = math.degrees(math.atan2(2*(w*y+x*z), 1-2*(y*y+z*z)))
            self.edit('manual turn handle', yaw_degrees=round(yaw,1))
        else:
            self.edit('manual drag handle', x=round(float(self.gizmo.position[0]),2), z=round(float(self.gizmo.position[2]),2))

    def restore(self):
        with self.lock:
            if self.busy or self.selected is None: return
            self.session.reset(); self.persist('restore AI'); self.sync()
            self.status.content = '<p>Restored the original AI placement.</p>'

    def generate_plan(self):
        with self.lock:
            if self.busy: return
            prompt = self.prompt.value
            self.busy = True; self.enable()
            self.status.content = '<p>Neon is choosing the cast, starting marks and facing directions…</p>'
        def run():
            try:
                self.session.plan(prompt)
                with self.lock:
                    self.persist('live Neon plan')
                    self.status.content = '<p>Live Neon plan ready. Select a person and drag the handle.</p>'
            except Exception as exc:
                with self.lock:
                    self.status.content = '<p>Planning failed: '+escape(str(exc))+'</p>'
            finally:
                with self.lock:
                    self.busy = False; self.sync(); self.frame_clients()
        threading.Thread(target=run, daemon=True).start()

    def export_plan(self, event):
        with self.lock:
            if self.busy or self.selected is None: return
            data = (json.dumps(self.session.export_plan(),indent=2)+'\n').encode()
            if event.client: event.client.send_file_download('character-start-plan.json',data)
            self.persist('export plan')
            self.status.content = '<p>Exported the validated plan with your edited starting marks.</p>'

    def save_view(self, event):
        if event.client is None: return
        from PIL import Image
        try:
            result = queue.Queue(maxsize=1)
            def render():
                try:
                    result.put((event.client.get_render(width=1280, height=720), None))
                except Exception as exc:
                    result.put((None, exc))
            threading.Thread(target=render, daemon=True).start()
            pixels, error = result.get(timeout=30.)
            if error is not None: raise error
            self.output.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(self.output/'interactive-view.png')
            self.status.content = '<p>Saved the current 3D view.</p>'
        except Exception as exc:
            self.status.content = '<p>View capture failed: '+escape(str(exc))+'</p>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=24995)
    parser.add_argument('--client-build', type=Path)
    parser.add_argument('--gateway-env', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'.runtime/interactive-placement')
    parser.add_argument('--restore-state', type=Path, help='Restore a saved local editor state')
    args = parser.parse_args()
    if args.client_build: os.environ['STAGEZERO_CLIENT_BUILD'] = str(args.client_build.resolve())
    if args.gateway_env:
        for line in args.gateway_env.read_text().splitlines():
            parts = shlex.split(line, comments=True)
            if len(parts)==1 and '=' in parts[0]:
                key,value=parts[0].split('=',1)
                if key in ('NEON_AI_GATEWAY_BASE_URL','NEON_AI_GATEWAY_TOKEN'): os.environ[key]=value
    gateway = GatewayGenerator.from_env(stage='layout'); gateway.model='gpt-6-astra'
    session = InitialPlacementSession(SCENE, planner=ScenePromptPlanner(gateway))
    if args.restore_state:
        previous = json.loads(args.restore_state.read_text())
        session.set_plan(previous['ai_plan'])
        for aid in previous['changed_actor_ids']:
            session.edit_actor(aid, **previous['placement']['starts'][aid])
    # create_studio_server serves our already-built client. A fresh Viser pip
    # install otherwise tries to build a second, unused upstream browser app.
    build = Path(os.environ.get('STAGEZERO_CLIENT_BUILD', ROOT/'studio_client/build'))
    if not (build/'index.html').is_file():
        raise RuntimeError('Build studio_client first, or pass --client-build')
    with patch('viser._client_autobuild.ensure_client_is_built'):
        server = create_studio_server(host='127.0.0.1',port=args.port,label='Neon · Character staging')
    if server.get_port()!=args.port:
        server.stop(); raise RuntimeError('Dedicated demo port is occupied')
    PlacementDemo(server, session, args.output)
    print(f'Interactive staging: http://127.0.0.1:{args.port}',flush=True)
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == '__main__': main()
