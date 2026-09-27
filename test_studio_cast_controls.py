"""UI contract checks for the main Studio cast form; no server or model needed."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from viser import GuiEvent, UploadedFile

from studio_cast_controls import StudioCastControls
from test_paired_direction_controls import Gui as BaseGui, Handle


class Upload(Handle):
    def on_upload(self, callback):
        self.callbacks['upload'] = callback

    def upload(self, content):
        self.value = UploadedFile(name='performance.cast.stagezero.npz', content=content)
        event = GuiEvent(client=None, client_id=None, target=self)
        self.callbacks['upload'](event)
        return event


class Gui(BaseGui):
    def add_html(self, content):
        return self._new('HTML', content=content)

    def add_upload_button(self, label, **_kwargs):
        result = Upload()
        self.handles[label] = result
        return result


class Session:
    def __init__(self):
        self.state = dict(active=True, busy=False, capturing=False, total_frames=0,
                          status='Ready', failure=None, progress=None)
        self.clip = None
        self.cancel = Mock()
        self.retry = Mock()
        self.save = Mock(return_value=b'complete archive')

    def snapshot(self):
        return dict(self.state)

    def timeline_clip(self):
        return self.clip


def clip(prompt='Saved direction'):
    return SimpleNamespace(actor_ids=('actor_1', 'actor_2'), frames=90, fps=30,
        metadata={'plan': {'title': 'A meeting', 'prompt': prompt,
                           'beats': [{'actor_ids': ['actor_1', 'actor_2'],
                                      'prompt': 'They meet.', 'seconds': 3}],
                           'warnings': []}})


class StudioCastControlsTests(unittest.TestCase):
    def setUp(self):
        self.gui = Gui()
        self.session = Session()
        self.available = [True]
        self.generate = Mock()
        self.frame = Mock()
        self.export = Mock()
        self.open = Mock()
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.controls = StudioCastControls(self.gui, self.session,
            on_generate=self.generate, on_frame=self.frame, on_export=self.export,
            on_open=self.open, provider_available=lambda: self.available[0],
            output_root=self.folder.name)

    def test_prompt_seed_and_client_forwarded_without_manual_cast_or_duration(self):
        self.controls.prompt.value = 'Three people greet each other in turn.'
        self.controls.seed.value = 17
        self.controls.generate.click('browser')
        self.generate.assert_called_once_with('Three people greet each other in turn.', 17, 'browser')
        self.assertNotIn('Background', self.gui.handles)
        self.assertNotIn('Seconds', self.gui.handles)
        self.assertNotIn('Play', self.gui.handles)
        self.assertNotIn('Pause', self.gui.handles)

    def test_busy_capture_and_provider_gate_queued_events_but_allow_cancel(self):
        self.session.state.update(busy=True, total_frames=90)
        self.controls.tick()
        self.assertTrue(self.controls.generate.disabled)
        self.assertTrue(self.controls.open.disabled)
        self.assertFalse(self.controls.cancel.disabled)
        self.controls.generate.click()
        self.controls.open.upload(b'archive')
        self.generate.assert_not_called()
        self.open.assert_not_called()
        self.controls.cancel.click()
        self.session.cancel.assert_called_once()
        self.session.state.update(busy=False, capturing=True)
        self.controls.tick()
        self.assertTrue(self.controls.cancel.disabled)
        self.assertTrue(self.controls.save.disabled)
        self.session.state.update(capturing=False)
        self.available[0] = False
        self.controls.tick()
        self.assertTrue(self.controls.generate.disabled)
        self.assertIn('both motion providers', self.controls.status.content)
        self.controls.generate.click()
        self.generate.assert_not_called()

    def test_plan_and_progress_are_escaped_and_prompt_edits_survive_new_clip(self):
        self.session.state.update(total_frames=90, status='<script>alert(1)</script>')
        self.session.clip = clip('<img src=x>')
        self.session.clip.metadata['plan']['title'] = '<b>unsafe</b>'
        self.session.clip.metadata['plan']['beats'][0]['prompt'] = '<script>unsafe</script>'
        self.controls.tick()
        self.assertEqual(self.controls.prompt.value, '<img src=x>')
        self.assertNotIn('<script>', self.controls.status.content)
        self.assertNotIn('<script>', self.controls.plan.content)
        self.assertIn('&lt;script&gt;', self.controls.plan.content)
        self.controls.prompt.value = 'My next idea'
        self.session.clip = clip('New generated prompt')
        self.controls.tick()
        self.assertEqual(self.controls.prompt.value, 'My next idea')
        self.session.state.update(busy=True, progress={'phase': 'planning'})
        self.controls.tick()
        self.assertIn('Planning', self.controls.status.content)

    def test_upload_reads_real_viser_target_and_syncs_archive_prompt(self):
        old = clip('Old prompt')
        self.session.clip = old
        self.session.state['total_frames'] = 90
        self.controls.tick()
        self.controls.prompt.value = 'Unsaved draft'
        new = clip('Archived prompt')
        self.open.side_effect = lambda data: setattr(self.session, 'clip', new)
        event = self.controls.open.upload(b'archive')
        self.assertIsInstance(event, GuiEvent)
        self.assertIsInstance(event.target.value, UploadedFile)
        self.open.assert_called_once_with(b'archive')
        self.assertEqual(self.controls.prompt.value, 'Archived prompt')
        self.controls.prompt.value = 'Draft after opening'
        self.controls.tick()
        self.assertEqual(self.controls.prompt.value, 'Draft after opening')

    def test_retry_save_frame_and_export_reservation(self):
        self.session.state.update(total_frames=90, failure='Old failure')
        self.session.clip = clip()
        self.controls.tick()
        self.controls.retry.click()
        self.session.retry.assert_called_once()
        client = Mock()
        self.controls.save.click(client)
        saved = list(Path(self.folder.name).glob('*.cast.stagezero.npz'))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].read_bytes(), b'complete archive')
        client.send_file_download.assert_called_once()
        self.controls.frame.click('browser')
        self.frame.assert_called_once_with('browser')
        video = Path(self.folder.name) / 'video.mp4'
        video.write_bytes(b'video')
        self.export.return_value = video
        with patch('studio_cast_controls.threading.Thread') as worker:
            self.controls.export.click()
            self.controls.export.click()
            self.assertEqual(worker.call_count, 1)
            self.assertTrue(self.controls._exporting)
            self.controls.open.upload(b'archive')
            self.controls.generate.click()
            self.open.assert_not_called()
            self.generate.assert_not_called()
            worker.call_args.kwargs['target']()
        self.assertFalse(self.controls._exporting)
        self.assertIn('Exported', self.controls.status.content)
        self.export.assert_called_once_with(None)

    def test_failed_open_retains_clip_and_draft(self):
        self.session.clip = clip('Existing')
        self.session.state['total_frames'] = 90
        self.controls.tick()
        self.controls.prompt.value = 'Unsubmitted draft'
        self.open.side_effect = ValueError('Invalid archive')
        self.controls.open.upload(b'bad')
        self.assertEqual(self.controls.prompt.value, 'Unsubmitted draft')
        self.assertIn('Invalid archive', self.controls.status.content)
        self.assertIsNotNone(self.session.timeline_clip())

    def test_toolbar_load_refreshes_archived_prompt_without_callbacks(self):
        self.session.clip = clip('Earlier prompt')
        self.session.state['total_frames'] = 90
        self.controls.tick()
        self.controls.prompt.value = 'Unsaved draft'
        self.session.clip = clip('Toolbar archive prompt')
        self.controls.mark_loaded()
        self.assertEqual(self.controls.prompt.value, 'Toolbar archive prompt')
        self.assertIn('A meeting', self.controls.plan.content)
        self.generate.assert_not_called()
        self.open.assert_not_called()
        self.export.assert_not_called()


if __name__ == '__main__':
    unittest.main()
