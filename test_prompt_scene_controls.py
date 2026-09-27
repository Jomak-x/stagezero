import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from viser import GuiEvent, UploadedFile
from prompt_scene_controls import PromptSceneControls
from test_paired_direction_controls import Gui as BaseGui, Handle


class Upload(Handle):
    def on_upload(self, callback): self.callbacks['upload'] = callback

    def upload(self, content):
        self.value = UploadedFile(name='performance.cast.stagezero.npz', content=content)
        event = GuiEvent(client=None, client_id=None, target=self)
        self.callbacks['upload'](event)
        return event


class Gui(BaseGui):
    def add_html(self, content): return self._new('HTML', content=content)
    def add_upload_button(self, label, **kwargs):
        item = Upload(); self.handles[label] = item; return item


class UpdatingDropdown(Handle):
    """Viser invokes on_update synchronously for programmatic value changes."""
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == 'value' and 'update' in self.callbacks:
            self.callbacks['update'](SimpleNamespace(client=None))


class UpdatingGui(Gui):
    def add_dropdown(self, label, options, initial_value=None):
        item = UpdatingDropdown(options=tuple(options), value=initial_value)
        self.handles[label] = item
        return item


class Session:
    def __init__(self):
        self.state = dict(busy=False,capturing=False,total_frames=0,status='Ready',error=None)
        self.clip = None
    def snapshot(self): return dict(self.state)
    def timeline_clip(self): return self.clip
    def play(self): pass
    def pause(self): pass
    def cancel(self): pass
    def retry(self): pass


class PromptControlsTests(unittest.TestCase):
    def setUp(self):
        self.gui,self.session,self.calls = Gui(),Session(),[]
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.controls = PromptSceneControls(self.gui,self.session,backgrounds={'City':{}},
            on_generate=lambda *args:self.calls.append(args),on_background=lambda x:None,
            on_frame=lambda x:None,on_export=lambda x:None,on_open=lambda data:self.calls.append(data),
            output_root=self.folder.name)

    def test_whole_prompt_and_seed_forwarded_without_fixed_duration_or_cast_fields(self):
        self.controls.prompt.value = 'Three people greet each other in turn.'
        self.controls.seed.value = 17
        self.controls.generate.click('browser')
        self.assertEqual(self.calls,[('Three people greet each other in turn.',17,'browser')])
        self.assertNotIn('Seconds',self.gui.handles)
        self.assertNotIn('First person X',self.gui.handles)

    def test_busy_and_export_hold_input_and_preserve_cancel(self):
        self.session.state.update(busy=True,total_frames=30)
        self.controls.tick()
        self.assertTrue(self.controls.generate.disabled)
        self.assertTrue(self.controls.background.disabled)
        self.assertTrue(self.controls.play.disabled)
        self.assertFalse(self.controls.cancel.disabled)
        self.session.state.update(busy=False,capturing=True)
        self.controls.tick()
        self.assertTrue(self.controls.cancel.disabled)
        self.assertTrue(self.controls.open.disabled)

    def test_plan_status_cannot_inject_html(self):
        self.session.state['status'] = '<script>bad</script>'
        self.session.clip = SimpleNamespace(actor_ids=('actor_1',),frames=60,fps=30,
            metadata={'plan':{'title':'<img onerror=bad>', 'beats':[
                {'actor_ids':['actor_1'],'prompt':'<script>bad</script>','seconds':2}], 'warnings':[]}})
        self.controls.tick()
        self.assertNotIn('<script>',self.controls.status.content)
        self.assertNotIn('<img',self.controls.plan.content)
        self.assertIn('&lt;script&gt;',self.controls.plan.content)

    def test_upload_uses_installed_viser_event_target_value_content(self):
        event = self.controls.open.upload(b'archive')
        self.assertIsInstance(event, GuiEvent)
        self.assertIsInstance(event.target.value, UploadedFile)
        self.assertFalse(hasattr(event, 'file'))
        self.assertEqual(self.calls,[b'archive'])

    def test_initial_open_and_play_labels_follow_authoritative_saved_background(self):
        city, market = {'name': 'City'}, {'name': 'Market'}
        current = [market]
        changes = []
        controls = PromptSceneControls(UpdatingGui(), self.session,
            backgrounds={'City': city, 'Market': market},
            get_background=lambda: current[0],
            on_generate=lambda *args: None,
            on_background=lambda label: changes.append(label),
            on_frame=lambda client: None, on_export=lambda client: None,
            on_open=lambda data: current.__setitem__(0, market),
            on_play=lambda: current.__setitem__(0, market), output_root=self.folder.name)
        self.assertEqual(controls.background.value, 'Market')
        self.assertEqual(changes, [])
        current[0] = city
        controls.tick()
        self.assertEqual(controls.background.value, 'City')
        controls.open.upload(b'project')
        self.assertEqual(controls.background.value, 'Market')
        current[0] = city
        controls.tick()
        controls._notice = 'Exported the complete performance.'
        controls.play.click()
        self.assertEqual(controls.background.value, 'Market')
        self.assertNotIn('Exported', controls.status.content)
        self.assertEqual(changes, [])  # Programmatic sync never changes scenes.

    def test_unknown_project_background_gets_accurate_label_and_failed_change_rolls_back(self):
        custom = {'name': 'Imported courtyard'}
        self.controls.get_background = lambda: custom
        self.controls.on_background = Mock(side_effect=ValueError('Background unavailable'))
        self.controls.tick()
        self.assertEqual(self.controls.background.value, 'Saved background · Imported courtyard')
        self.assertIn(self.controls.background.value, self.controls.background.options)
        self.controls.background.edit('City')
        self.assertEqual(self.controls.background.value, 'Saved background · Imported courtyard')
        self.assertIn('Background unavailable', self.controls.status.content)

    def test_background_and_retry_clear_previous_export_notice(self):
        self.controls._notice = 'Exported the complete performance.'
        self.controls.background.edit('City')
        self.assertNotIn('Exported', self.controls.status.content)
        self.controls._notice = 'Previous export failed.'
        self.controls.retry.click()
        self.assertNotIn('export', self.controls.status.content)

    def test_queued_disabled_actions_cannot_generate_or_open_during_busy_state(self):
        self.session.state['busy'] = True
        self.controls.generate.click()
        self.controls.open.upload(b'archive')
        self.controls.on_play = Mock()
        self.controls.play.click()
        self.controls.on_play.assert_not_called()
        self.assertEqual(self.calls, [])
        self.assertIn('Finish or cancel generation', self.controls.status.content)

    def test_export_reservation_blocks_queued_changes_before_capture_thread_starts(self):
        self.session.state['total_frames'] = 30
        video = Path(self.folder.name) / 'playback.mp4'
        video.write_bytes(b'video')
        self.controls.on_export = Mock(return_value=video)
        self.controls.on_play = Mock()
        self.controls.on_background = Mock()
        with patch('prompt_scene_controls.threading.Thread') as worker:
            self.controls.export.click()
            self.controls.export.click()
            self.assertEqual(worker.call_count, 1)
            self.assertTrue(self.controls._exporting)
            self.assertIn('Exporting', self.controls.status.content)
            self.controls.play.click()
            self.controls.background.edit('City')
            self.controls.generate.click()
            self.controls.open.upload(b'archive')
            self.controls.on_play.assert_not_called()
            self.controls.on_background.assert_not_called()
            self.assertEqual(self.calls, [])
            worker.call_args.kwargs['target']()
        self.assertFalse(self.controls._exporting)
        self.assertIn('Exported', self.controls.status.content)
        self.controls.play.click()
        self.controls.on_play.assert_called_once()
        self.assertNotIn('Exported', self.controls.status.content)


class PromptSceneActionTests(unittest.TestCase):
    def setUp(self):
        from cast_performance_session import CastPerformanceSession
        from paired_scene import EMPTY_SCENE
        from prompt_scene_viewer import PromptSceneActions
        from test_cast_performance import performance
        self.cast = CastPerformanceSession()
        self.addCleanup(self.cast.close)
        self.clip = performance(1)
        self.city = dict(EMPTY_SCENE, name='City')
        self.market = dict(EMPTY_SCENE, name='Market')
        self.cast.load_performance(self.clip, self.market)
        self.renderer = SimpleNamespace(set_visible=Mock(), sync_cast=Mock(), set_clip=Mock())
        self.draw = Mock()
        self.build = Mock(return_value=lambda scene, **kwargs: self.clip)
        self.capture = Mock(return_value=Path('playback.mp4'))
        self.actions = PromptSceneActions(self.cast, self.renderer,
            {'City': self.city, 'Market': self.market}, self.market,
            render_lock=threading.RLock(), draw_scene=self.draw,
            build=self.build, capture=self.capture)

    def test_play_and_open_restore_the_saved_scene_and_invalid_open_preserves_current_scene(self):
        project = self.cast.save()
        self.actions.background('City')
        self.assertEqual(self.actions.background_document(), self.city)
        self.assertEqual(self.cast.scene_document, self.market)
        with self.assertRaises(ValueError):
            self.actions.open_project(b'invalid')
        self.assertEqual(self.actions.background_document(), self.city)
        self.assertIs(self.cast.timeline_clip(), self.clip)
        self.actions.play_saved()
        self.assertEqual(self.actions.background_document(), self.market)
        self.actions.background('City')
        self.actions.open_project(project)
        self.assertEqual(self.actions.background_document(), self.market)
        self.assertFalse(self.cast.snapshot()['playing'])

    def test_background_rejection_does_not_change_scene_or_mesh_visibility(self):
        self.cast.begin_capture()
        with self.assertRaisesRegex(ValueError, 'export'):
            self.actions.background('City')
        self.assertEqual(self.actions.background_document(), self.market)
        self.renderer.set_visible.assert_not_called()
        self.draw.assert_not_called()

    def test_export_reserves_state_before_session_capture_lease_and_restores_scene(self):
        entered, release = threading.Event(), threading.Event()
        def capture(client):
            entered.set()
            if not release.wait(2.):
                raise AssertionError('Export was not released')
            return Path('playback.mp4')
        self.actions.capture = capture
        self.actions.background('City')
        self.cast.seek(3)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.actions.export, None)
            self.assertTrue(entered.wait(1.))
            self.assertFalse(self.cast.snapshot()['capturing'])
            self.assertEqual(self.actions.background_document(), self.market)
            try:
                for action in (lambda: self.actions.background('City'),
                        lambda: self.actions.generate('A scene', 1, None),
                        lambda: self.actions.open_project(b'archive'), self.actions.play_saved):
                    with self.assertRaisesRegex(ValueError, 'export'):
                        action()
                self.actions.seek(0)
                self.assertEqual(self.cast.snapshot()['frame'], 3)
                self.build.assert_not_called()
            finally:
                release.set()
            self.assertEqual(future.result(1.), Path('playback.mp4'))
        self.assertFalse(self.actions.exporting)

    def test_export_setup_failure_releases_action_reservation(self):
        self.renderer.set_visible.side_effect = RuntimeError('Renderer unavailable')
        with self.assertRaisesRegex(RuntimeError, 'Renderer unavailable'):
            self.actions.export(None)
        self.assertFalse(self.actions.exporting)
        self.capture.assert_not_called()

    def test_generation_retains_selected_background_and_blocks_open_until_complete(self):
        entered, release = threading.Event(), threading.Event()
        scenes = []
        def builder(scene, **kwargs):
            scenes.append(scene)
            entered.set()
            release.wait(2.)
            return self.clip
        self.build.return_value = builder
        self.actions.background('City')
        self.actions.generate('Walk to the city', 17, None)
        self.assertTrue(entered.wait(1.))
        try:
            with self.assertRaisesRegex(ValueError, 'generation'):
                self.actions.open_project(self.cast.save())
            with self.assertRaisesRegex(ValueError, 'generation'):
                self.actions.background('Market')
            self.assertEqual(self.actions.background_document(), self.city)
        finally:
            release.set()
            self.cast._thread.join(2.)
        self.assertEqual(scenes, [self.city])
        self.build.assert_called_once_with('Walk to the city', 17)

    def test_late_render_of_completed_take_does_not_undo_a_new_background_choice(self):
        # A worker has published, but the render loop has not observed it yet.
        self.actions.background('City')
        visible = self.actions.display_new_clip(self.clip, self.cast.snapshot())
        self.assertFalse(visible)
        self.assertEqual(self.actions.background_document(), self.city)
        self.renderer.set_visible.assert_called_with(False)
        # A genuinely newer completed performance restores its own saved scene.
        from test_cast_performance import performance
        newer = performance(2)
        self.cast.load_performance(newer, self.market)
        self.assertTrue(self.actions.display_new_clip(newer, self.cast.snapshot()))
        self.assertEqual(self.actions.background_document(), self.market)


if __name__ == '__main__': unittest.main()
