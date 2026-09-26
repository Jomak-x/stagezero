"""Sidebar state and transport tests with counted Viser-style handles."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock

import numpy as np

from directing import DirectorSession
from studio_ui import StudioUI, CREATE, EXTEND, REPLACE, AUTO, SET_DURATION, TARGET_TOTAL
from takes import Take
from test_live_motion import ControlledBackend, wait_until


class Handle:
    def __init__(self, **values):
        object.__setattr__(self, 'writes', [])
        object.__setattr__(self, 'callbacks', {})
        for key, value in values.items():
            object.__setattr__(self, key, value)

    def __setattr__(self, name, value):
        if name not in ('writes', 'callbacks'):
            self.writes.append((name, value))
        object.__setattr__(self, name, value)

    def on_click(self, callback):
        self.callbacks['click'] = callback
        return callback

    def on_update(self, callback):
        self.callbacks['update'] = callback
        return callback

    def on_upload(self, callback):
        self.callbacks['upload'] = callback
        return callback

    def click(self, value=None):
        if value is not None:
            self.value = value
        self.callbacks['click'](SimpleNamespace(client=object()))

    def edit(self, value):
        self.value = value
        self.callbacks['update'](SimpleNamespace(client=object()))


class Gui:
    def __init__(self):
        self.handles = []

    def _handle(self, **values):
        values.setdefault('visible', True)
        handle = Handle(**values)
        self.handles.append(handle)
        return handle

    def add_html(self, content):
        return self._handle(content=content)

    def add_markdown(self, content):
        return self._handle(content=content)

    def add_button_group(self, label, options):
        return self._handle(label=label, options=options, value=options[0], disabled=False)

    def add_text(self, label, initial_value='', **kwargs):
        return self._handle(label=label, value=initial_value, disabled=False)

    def add_button(self, label, **kwargs):
        return self._handle(label=label, disabled=False, visible=kwargs.get('visible', True))

    def add_dropdown(self, label, options, initial_value=None):
        return self._handle(label=label, options=options, value=initial_value or options[0], disabled=False)

    def add_checkbox(self, label, initial_value=False):
        return self._handle(label=label, value=initial_value, disabled=False)

    def add_upload_button(self, label, **kwargs):
        return self._handle(label=label, value=None, disabled=False)

    def add_tab_group(self):
        return self

    def add_tab(self, label):
        return self

    def add_folder(self, label, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class StudioUITests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.backend = ControlledBackend()
        self.session = DirectorSession(
            self.backend,
            np.zeros((120, 34, 3), dtype=np.float32),
            np.tile(np.eye(3, dtype=np.float32), (120, 34, 1, 1)),
        )
        self.gui = Gui()
        camera = SimpleNamespace(build_gui=lambda gui: None)
        self.ui = StudioUI(SimpleNamespace(gui=self.gui), self.session, camera,
                           Path(self.temp.name), lambda gui: None)

    def tearDown(self):
        self.backend.release.set()
        self.temp.cleanup()

    def test_idle_updates_publish_nothing_and_seek_draft_is_stable(self):
        for handle in self.gui.handles:
            handle.writes.clear()
        for _ in range(100):
            self.ui.update()
        self.assertEqual(sum(len(handle.writes) for handle in self.gui.handles), 0)

        self.ui.seek_time.value = '0.25'
        self.session.seek(30)
        for handle in self.gui.handles:
            handle.writes.clear()
        self.ui.update()
        self.assertEqual(self.ui.seek_time.value, '0.25')
        self.assertEqual(self.ui.seek_time.writes, [])
        self.assertIn('0.50', self.ui.playhead.content)
        self.assertIn('/ 2.00 s', self.ui.playhead.content)
        self.assertIn('Paused', self.ui.status.content)
        self.assertLessEqual(sum(len(handle.writes) for handle in self.gui.handles), 3)

    def test_go_and_frame_steps_clamp_to_last_sample(self):
        self.ui.seek_time.value = '1.98'
        self.ui.seek_go.click()
        self.assertEqual(self.session.frame, 119)
        self.ui.update()
        self.assertIn('Finished', self.ui.status.content)
        self.ui.frames.click('−1 frame')
        self.assertEqual(self.session.frame, 118)
        self.ui.frames.click('+1 frame')
        self.assertEqual(self.session.frame, 119)
        self.ui.transport.click('Start')
        self.assertEqual(self.session.frame, 0)
        self.ui.seek_time.value = '2.00'  # Clip duration selects its final sample.
        self.ui.seek_go.click()
        self.assertEqual(self.session.frame, 119)
        self.ui.seek_time.value = '2.01'
        self.ui.seek_go.click()
        self.assertEqual(self.session.frame, 119)
        self.assertIn('2.00 s', self.session.status)

    def test_create_switches_to_live_and_requires_valid_prompt(self):
        self.assertEqual(self.ui.edit_action.value, CREATE)
        self.assertFalse(self.ui.generate.disabled)
        self.assertIn('4.16 seconds', self.ui.duration_preview.content)
        self.ui.prompt.edit('x' * 501)
        self.ui.update()
        self.assertTrue(self.ui.generate.disabled)
        self.assertIn('501/500', self.ui.prompt_count.content)
        self.ui.generate.click()
        self.assertFalse(self.session.busy)

        self.ui.prompt.edit('Wave gently')
        self.ui.update()
        self.assertFalse(self.ui.generate.disabled)
        self.ui.generate.click()
        self.assertEqual(self.session.mode, 'Live ARDY')
        self.assertTrue(self.session.busy)
        self.ui.update()
        self.assertIn('Generating', self.ui.status.content)
        self.assertTrue(self.ui.prompt.disabled)
        self.assertTrue(self.ui.cancel.visible)
        self.assertTrue(self.backend.started.wait(1))
        self.ui.cancel.click()
        self.assertFalse(self.session.busy)
        self.backend.release.set()
        wait_until(lambda: not self.session.busy)
        self.assertEqual(self.session.kind, 'reference')

    def _seed_take(self, length=100):
        positions = np.zeros((length, 34, 3), dtype=np.float32)
        rotations = np.tile(np.eye(3, dtype=np.float32), (length, 34, 1, 1))
        motion = np.zeros((length, 414), dtype=np.float32)
        self.session.set_mode('Live ARDY')
        take = Take('a', 'First', positions, rotations, motion)
        self.session.takes = {take.id: take}
        self.session.select_take(take.id)
        self.ui.update()
        return take

    def test_extend_uses_selected_take_and_target_total_not_playhead(self):
        self._seed_take()
        self.session.seek(25)
        self.ui.edit_action.edit(EXTEND)
        self.ui.duration_mode.edit(TARGET_TOTAL)
        self.ui.duration_seconds.edit('9.00')
        self.assertIn('4.00 seconds → 9.00 seconds total', self.ui.duration_preview.content)
        self.session.submit = Mock()
        self.ui.generate.click()
        self.session.submit.assert_called_once_with(
            'A person waves with their right hand.', seconds=5.0,
            edit_mode='extend', at_frame=None)
        self.assertEqual(self.session.frame, 25)

    def test_replace_uses_explicit_time_and_shows_original_preservation(self):
        self._seed_take()
        self.ui.edit_action.edit(REPLACE)
        self.ui.replace_time.edit('1.00')
        self.ui.duration_mode.edit(SET_DURATION)
        self.ui.duration_seconds.edit('2.00')
        self.assertIn('Keep first 1.00 seconds', self.ui.duration_preview.content)
        self.assertIn('3.00 second take', self.ui.duration_preview.content)
        self.assertIn('Original stays available', self.ui.duration_preview.content)
        self.session.submit = Mock()
        self.ui.generate.click()
        self.session.submit.assert_called_once_with(
            'A person waves with their right hand.', seconds=2.0,
            edit_mode='replace', at_frame=25)

        self.ui.replace_time.edit('4.00')
        self.assertTrue(self.ui.generate.disabled)
        self.assertIn('Enter a time', self.ui.duration_preview.content)
        self.ui.replace_time.edit('0.00')
        self.assertFalse(self.ui.generate.disabled)
        self.ui.replace_time.edit('0.08')
        self.assertTrue(self.ui.generate.disabled)
        self.assertIn('0.16 seconds or later', self.ui.duration_preview.content)

    def test_prepare_edit_controls_and_invalid_total(self):
        self._seed_take()
        self.assertNotIn('no inference', self.ui.status.content)
        self.assertNotIn('Recorded generated take', self.ui.status.content)
        self.session.seek(50)
        self.ui.prepare_replace.click()
        self.ui.update()
        self.assertIn('Ready to change this ending', self.ui.status.content)
        self.assertEqual(self.ui.edit_action.value, REPLACE)
        self.assertEqual(self.ui.replace_time.value, '2.00')
        self.ui.prepare_extend.click()
        self.assertEqual(self.ui.edit_action.value, EXTEND)
        self.ui.duration_mode.edit(TARGET_TOTAL)
        self.ui.duration_seconds.edit('3.00')
        self.assertTrue(self.ui.generate.disabled)
        self.assertIn('must add 0.16–30 seconds', self.ui.duration_preview.content)
        self.ui.duration_mode.edit(AUTO)
        self.assertFalse(self.ui.duration_seconds.visible)
        self.assertFalse(self.ui.generate.disabled)

    def test_target_total_minimum_quantizes_exactly(self):
        self._seed_take(25)
        self.ui.edit_action.edit(EXTEND)
        self.ui.duration_mode.edit(TARGET_TOTAL)
        self.ui.duration_seconds.edit('1.16')
        self.assertFalse(self.ui.generate.disabled)
        self.assertIn('1.16 seconds total', self.ui.duration_preview.content)
        self.session.submit = Mock()
        self.ui.generate.click()
        self.session.submit.assert_called_once_with(
            'A person waves with their right hand.', seconds=0.16,
            edit_mode='extend', at_frame=None)

    def test_auto_discloses_prompt_duration_cap(self):
        self.ui.prompt.edit('Wave for 60 seconds')
        self.assertIn('capped at 30 s', self.ui.duration_preview.content)
        self.assertIn('30.00 seconds', self.ui.duration_preview.content)

    def test_reference_and_recorded_states_have_distinct_duration(self):
        self.ui.mode.edit('Live ARDY')
        self.ui.update()
        self.assertIn('Reference pose', self.ui.status.content)
        self.assertIn('/ 0.00 s', self.ui.playhead.content)
        self.assertTrue(self.ui.seek_go.disabled)
        self.ui.mode.edit('Recorded preview')
        self.session.seek(119)
        self.ui.update()
        self.assertIn('Finished', self.ui.status.content)
        self.assertIn('/ 2.00 s', self.ui.playhead.content)

    def test_pending_take_choice_is_not_overwritten_before_callback(self):
        positions = np.zeros((4, 34, 3), dtype=np.float32)
        rotations = np.tile(np.eye(3, dtype=np.float32), (4, 34, 1, 1))
        motion = np.zeros((4, 414), dtype=np.float32)
        self.session.set_mode('Live ARDY')
        self.session.takes = {
            take_id: Take(take_id, name, positions.copy(), rotations.copy(), motion.copy())
            for take_id, name in (('a', 'First'), ('b', 'Second'))
        }
        self.session.select_take('a')
        self.ui.update()
        second = next(label for label, take_id in self.ui.take_map.items() if take_id == 'b')

        # The browser changes its local selection before Viser dispatches the
        # callback. A viewer update must leave that pending choice alone.
        self.ui.takes.value = second
        self.ui.takes.writes.clear()
        self.ui.update()
        self.assertEqual(self.ui.takes.value, second)
        self.assertFalse(any(name == 'value' for name, _ in self.ui.takes.writes))

        captured = threading.Event()

        class SignalingMap(dict):
            def get(self, key, default=None):
                value = super().get(key, default)
                captured.set()
                return value

        self.ui.take_map = SignalingMap(self.ui.take_map)
        with self.session.lock:
            thread = threading.Thread(
                target=self.ui.takes.callbacks['update'],
                args=(SimpleNamespace(client=object()),),
            )
            thread.start()
            self.assertTrue(captured.wait(1))
            # This replaces the option map while the callback waits on the
            # session lock; it must still act on the captured ID.
            self.ui.update()
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.session.active_take, 'b')
        self.ui.update()
        self.assertEqual(self.ui.takes.value, second)


if __name__ == '__main__':
    unittest.main()
