"""Live Core direction and exact project playback on the shared Core27 rig."""
from __future__ import annotations
import argparse
from html import escape
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor/ardy'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8769')
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--port', type=int, default=2350)
    parser.add_argument('--project', type=Path)
    parser.add_argument('--scene', type=Path, help='Scene plan JSON for geometry display')
    parser.add_argument('--actors', type=int, choices=(1, 2), default=1)
    parser.add_argument('--output', type=Path, default=ROOT / '.runtime/realtime-projects')
    args = parser.parse_args()
    import torch
    import viser
    from ardy.skeleton import CoreSkeleton27
    from ardy.viz.viser_utils import Character
    from realtime_client import RealtimeClient, request_body
    from realtime_director import RealtimeDirector

    torch.set_num_threads(2)
    client = RealtimeClient(args.url, args.token_file.read_text())
    director = (RealtimeDirector.load_project(args.project.read_bytes()) if args.project
                else RealtimeDirector(tuple(f'actor_{i}' for i in range(args.actors))))
    server = viser.ViserServer(host='127.0.0.1', port=args.port, label='StageZero realtime motion')
    if server.get_port() != args.port:
        server.stop()
        raise OSError('Requested viewer port is occupied')
    server.gui.configure_theme(dark_mode=True, show_logo=False, show_share_button=False)
    server.scene.set_up_direction('+y')
    server.scene.world_axes.visible = False
    server.scene.add_box('/floor', dimensions=(60, .1, 60), position=(0, -.1, 0), color=(26, 34, 46))
    skeleton = CoreSkeleton27()
    characters = []
    for i in range(2):
        actor_id = f'actor_{i}'
        character = Character('realtime_' + actor_id, server, skeleton, create_skeleton_mesh=False,
                              create_skinned_mesh=True, mesh_mode='core_skin', show_foot_contacts=False, dark_mode=True)
        character.skinned_mesh.color = ((104,217,231), (244,177,109))[i]
        character.skinned_mesh.visible = False
        characters.append(character)
    scene_handles = []
    current_scene = director.project_metadata.get('scene')
    current_placements = director.project_metadata.get('initial_placements')
    if current_scene is not None:
        from experiments.review_scene_interactions import _draw_scene
        scene_handles = _draw_scene(server, current_scene, None)
    if args.scene:
        import json
        from experiments.review_scene_interactions import _draw_scene
        plan = json.loads(args.scene.read_text())
        current_scene = plan.get('scene', plan)
        scene_handles = _draw_scene(server, current_scene, plan.get('affordances'))
    server.gui.add_markdown('# StageZero realtime\nNative Core motion · 20 fps · buffered two-second generation windows')
    prompt = server.gui.add_text('Direction', initial_value='A person walks forward naturally and looks around.')
    frames = server.gui.add_dropdown('Duration', ('2 seconds', '6 seconds', '12 seconds'), initial_value='6 seconds')
    generate = server.gui.add_button('Generate / redirect')
    play = server.gui.add_button('Play')
    pause = server.gui.add_button('Pause')
    retry = server.gui.add_button('Retry failed generation')
    save = server.gui.add_button('Save exact project')
    upload = server.gui.add_upload_button('Open exact project', mime_type='.npz')
    status = server.gui.add_markdown('Ready.')
    notice = server.gui.add_markdown('')
    stop = threading.Event()
    scene_active = threading.Event()
    free_inference = threading.Event()
    from scene_beats import SCENARIOS, build_scene
    preset = server.gui.add_dropdown('Scene', SCENARIOS, initial_value=SCENARIOS[0])
    research = server.gui.add_checkbox('Enable research paired motion', initial_value=False)
    run_preset = server.gui.add_button('Run complete scene')
    layout = server.gui.add_button('Load scene layout')
    nav_actor = server.gui.add_dropdown('Navigate actor', ('Actor 1', 'Actor 2'), initial_value='Actor 1')
    nav_target = server.gui.add_text('Object name or ID', initial_value='Gate')
    nav_verb = server.gui.add_dropdown('Navigate action', ('go_through', 'approach'), initial_value='go_through')
    navigate = server.gui.add_button('Navigate to object')

    @layout.on_click
    def load_layout(_):
        nonlocal director, current_scene, current_placements, scene_handles
        try:
            if scene_active.is_set():
                raise ValueError('Wait for the coordinated scene to complete.')
            plan = build_scene(preset.value)
            if director.total_frames:
                args.output.mkdir(parents=True, exist_ok=True)
                (args.output / f'before-layout-{time.time_ns()}.npz').write_bytes(director.save_project())
            free_inference.clear()
            director.pause()
            director = RealtimeDirector(plan['actor_ids'], project_metadata={
                'scene': plan['scene'], 'initial_placements': plan['initial_placements']})
            current_scene, current_placements = plan['scene'], plan['initial_placements']
            from experiments.review_scene_interactions import _draw_scene
            for handle in scene_handles:
                handle.remove()
            scene_handles = _draw_scene(server, current_scene, None)
            for character in characters:
                character.skinned_mesh.visible = False
            nav_target.value = current_scene['objects'][0]['name'] if current_scene['objects'] else ''
            notice.content = 'Scene geometry loaded. Choose an object and navigation action.'
        except Exception as exc:
            notice.content = escape(str(exc))

    @navigate.on_click
    def navigate_object(_):
        try:
            if scene_active.is_set() or director.snapshot()['contact_open']:
                raise ValueError('Wait for the coordinated contact scene to finish.')
            if current_scene is None:
                raise ValueError('Load a scene layout first.')
            actor_index = 0 if nav_actor.value == 'Actor 1' else 1
            if actor_index >= len(director.actor_ids):
                raise ValueError('This scene has one actor.')
            matches = [o for o in current_scene['objects']
                       if nav_target.value.casefold().strip() in (o['id'].casefold(), o['name'].casefold())]
            if len(matches) != 1:
                raise ValueError('Choose an exact, unambiguous object name or ID from the scene.')
            from realtime_navigation import plan_navigation
            stages, route = plan_navigation(current_scene, director.actor_ids,
                actor_id=director.actor_ids[actor_index], target_id=matches[0]['id'],
                verb=nav_verb.value, last_clip=director.timeline_clip(),
                initial_placements=current_placements)
            first = stages[0]
            director.interrupt(first.prompt, kind=first.kind, frames=first.frames,
                source=first.source, actor_prompts=first.actor_prompts, metadata=first.metadata)
            if len(stages) > 1:
                director.queue_sequence(stages[1:])
            free_inference.set()
            director.play()
            notice.content = 'Following a route through the declared scene geometry.'
        except Exception as exc:
            notice.content = escape(str(exc))

    @server.on_client_connect
    def connected(connection):
        connection.camera.position = (4., 2.8, 5.)
        connection.camera.look_at = (0., 1., 0.)
        connection.camera.up_direction = (0., 1., 0.)

    @generate.on_click
    def direction(_):
        try:
            if scene_active.is_set():
                raise ValueError('A coordinated scene is running. Wait for its safe release and completion before free direction.')
            if director.snapshot()['contact_open']:
                raise ValueError('This partial contact scene cannot resume without its cached pair. Replay it or rerun the complete scene.')
            count = {'2 seconds':40, '6 seconds':120, '12 seconds':240}[frames.value]
            if director.total_frames:
                director.interrupt(prompt.value, frames=count)
            else:
                director.submit_instruction(prompt.value, frames=count, metadata=(
                    {'initial_placements': current_placements} if current_placements else {}))
            free_inference.set()
            director.play()
            notice.content = ''
        except Exception as exc:
            notice.content = escape(str(exc))

    @play.on_click
    def start(_):
        if scene_active.is_set():
            return
        if director.total_frames and director.snapshot()['frame'] == director.total_frames - 1:
            director.seek(0)
        director.play()

    @pause.on_click
    def paused(_):
        if not scene_active.is_set():
            director.pause()

    @retry.on_click
    def retried(_):
        if not free_inference.is_set():
            notice.content = 'Use Run complete scene to retry a coordinated scene.'
            return
        director.retry()
        director.play()

    @save.on_click
    def saved(event):
        try:
            args.output.mkdir(parents=True, exist_ok=True)
            path = args.output / f'realtime-{time.time_ns()}.stagezero.npz'
            if current_scene is not None:
                director.project_metadata.update(scene=current_scene, initial_placements=current_placements)
            data = director.save_project()
            path.write_bytes(data)
            if event.client is not None:
                event.client.send_file_download(path.name, data)
            notice.content = 'Saved exact project: ' + str(path)
        except Exception as exc:
            notice.content = escape(str(exc))

    @upload.on_upload
    def opened(_):
        nonlocal director, scene_handles, current_scene, current_placements
        try:
            if scene_active.is_set():
                raise ValueError('Wait for the coordinated scene to complete before loading a project')
            restored = RealtimeDirector.load_project(upload.value.content)
            free_inference.clear()
            previous = director
            previous.pause()
            director = restored
            for handle in scene_handles:
                handle.remove()
            scene_handles = []
            current_scene = restored.project_metadata.get('scene')
            current_placements = restored.project_metadata.get('initial_placements')
            if current_scene is not None:
                from experiments.review_scene_interactions import _draw_scene
                scene_handles = _draw_scene(server, current_scene, None)
            for i, character in enumerate(characters):
                character.skinned_mesh.visible = i < len(director.actor_ids)
            notice.content = 'Loaded exact saved motion. Press Play or add a new direction.'
        except Exception as exc:
            notice.content = 'Could not open project: ' + escape(str(exc))

    @run_preset.on_click
    def run_scene_preset(_):
        nonlocal director, scene_handles, current_scene, current_placements
        if scene_active.is_set():
            return
        plan = build_scene(preset.value)
        if plan['provenance']['research_pair_model'] and not research.value:
            notice.content = 'This paired scene uses InterGen research motion. Enable the research option to run it.'
            return
        try:
            if director.total_frames:
                args.output.mkdir(parents=True, exist_ok=True)
                (args.output / f'before-scene-{time.time_ns()}.npz').write_bytes(director.save_project())
        except Exception as exc:
            notice.content = 'Could not back up the current timeline: ' + escape(str(exc))
            return
        scene_active.set()
        free_inference.clear()
        director.pause()
        try:
            from experiments.review_scene_interactions import _draw_scene
            for handle in scene_handles:
                handle.remove()
            current_scene = plan['scene']
            current_placements = plan['initial_placements']
            scene_handles = _draw_scene(server, current_scene, None)
        except Exception as exc:
            scene_active.clear()
            notice.content = 'Scene setup failed: ' + escape(str(exc))
            return
        def expose(new_director):
            nonlocal director
            director = new_director
            for i, character in enumerate(characters):
                character.skinned_mesh.visible = i < len(director.actor_ids)
        def generate_scene():
            from experiments.generate_scene_showcase import run_scene
            try:
                report = run_scene(plan, client, args.output / (plan['name'] + '-' + str(time.time_ns())),
                                   on_director=expose, cancelled=stop.is_set)
                notice.content = 'Complete scene generated, saved, and measured: ' + report['status']
            except Exception as exc:
                director.pause()
                notice.content = 'Scene requires attention: ' + escape(str(exc))
            finally:
                scene_active.clear()
        threading.Thread(target=generate_scene, daemon=True, name='scene-runner').start()

    def infer():
        while not stop.is_set():
            if scene_active.is_set() or not free_inference.is_set():
                stop.wait(.03)
                continue
            for request_id in director.take_cancellations():
                try:
                    client.cancel(request_id)
                except Exception:
                    pass
            current = director
            request = current.claim_request()
            if request is None:
                stop.wait(.03)
                continue
            try:
                clips = client.wait(request_body(request),
                    cancelled=lambda: stop.is_set() or scene_active.is_set() or not free_inference.is_set() or director is not current or current.snapshot()['inflight_request_id'] != request.request_id)
                current.complete(request.request_id, clips[0])
            except Exception as exc:
                current.fail(request.request_id, exc)
                print(f'Generation failed: {type(exc).__name__}: {exc}', flush=True)
    threading.Thread(target=infer, daemon=True, name='realtime-client').start()
    last = None
    try:
        while True:
            active = director
            snapshot = active.snapshot() if scene_active.is_set() else active.tick()
            marker = (id(active), snapshot['revision'], snapshot['frame'])
            if snapshot['total_frames'] and marker != last:
                positions, rotations = active.frame_pose(snapshot['frame'])
                with server.atomic():
                    for i, character in enumerate(characters[:len(active.actor_ids)]):
                        character.skinned_mesh.visible = True
                        character.set_pose(torch.from_numpy(positions[i]), torch.from_numpy(rotations[i]))
                last = marker
            status.content = (f"**{escape(snapshot['status'])}**\n\n"
                              f"{snapshot['frame']/20:.2f}s / {snapshot['total_frames']/20:.2f}s · "
                              f"buffer {snapshot['buffer_frames']/20:.2f}s · {snapshot['mode']} profile")
            time.sleep(.025)
    finally:
        stop.set()
        server.stop()


if __name__ == '__main__':
    main()
