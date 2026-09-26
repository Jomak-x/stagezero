"""The visible action editor must operate on the motion that projects save."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from directing import DirectorSession
from studio_ui import CREATE, EDIT, EXTEND, SET_DURATION, StudioUI
from takes import decode_project, encode_project
from test_live_motion import ControlledBackend, wait_until
from test_studio_ui import Gui


class ActionEditorUITests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.backend = ControlledBackend()
        self.backend.release.set()
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

    def make_actions(self, count=3):
        self.session.set_mode('Live ARDY')
        for index in range(count):
            self.session.submit(f'Action {index + 1}', seconds=4.16,
                                edit_mode='new' if index == 0 else 'extend')
            wait_until(lambda: not self.session.busy)
            self.session.pause()
        self.ui.update()
        return self.session.takes[self.session.active_take]

    def select_action(self, index):
        labels = tuple(self.ui.segment_map)
        self.assertGreater(len(labels), index)
        self.ui.segments.edit(labels[index])

    def test_selecting_action_moves_playhead_and_loads_its_direction(self):
        take = self.make_actions()
        self.assertEqual(self.ui.segments.value, 'Select an action')
        self.select_action(1)
        self.assertEqual(self.session.frame, take.segments[1]['start'])
        self.assertEqual(self.ui.prompt.value, 'Action 2')
        self.assertEqual(self.ui.edit_action.value, EDIT)
        self.assertTrue(self.ui.delete_action.visible)
        self.assertIn('Other actions stay', self.ui.duration_preview.content)

    def test_replacing_middle_action_keeps_the_surrounding_actions_and_saves(self):
        original = self.make_actions()
        self.select_action(1)
        self.ui.prompt.edit('A new middle action')
        self.ui.duration_mode.edit(SET_DURATION)
        self.ui.duration_seconds.edit('2.00')
        self.assertIn('10.32s total', self.ui.duration_preview.content)
        with mock.patch.object(self.session, 'submit', wraps=self.session.submit) as submit:
            self.ui.generate.click()
            submit.assert_called_once_with('A new middle action', seconds=2.0,
                                           edit_mode='action', at_frame=104)
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        edited = self.session.takes[self.session.active_take]
        self.assertEqual(edited.id, original.id)
        self.assertEqual([s['prompt'] for s in edited.segments],
                         ['Action 1', 'A new middle action', 'Action 3'])
        self.assertEqual([s['start'] for s in edited.segments], [0, 104, 154])
        self.assertEqual(len(edited.positions), 258)
        saved = encode_project(self.session.takes, edited.id, self.session.frame,
                               self.session.scene)
        loaded, active, _, _ = decode_project(saved)
        self.assertEqual(active, edited.id)
        self.assertEqual([s['prompt'] for s in loaded[active].segments],
                         ['Action 1', 'A new middle action', 'Action 3'])

    def test_new_take_choice_clears_active_selection_but_retains_saved_takes(self):
        original = self.make_actions(1)
        self.ui.edit_action.edit(CREATE)
        self.assertIsNone(self.session.active_take)
        self.assertIn(original.id, self.session.takes)
        self.assertEqual(self.ui.edit_action.value, CREATE)
        self.assertEqual(self.ui.takes.value, 'New take · not generated yet')
        self.assertIn('first action', self.ui.generate.label.lower())
        existing = next(label for label, identifier in self.ui.take_map.items()
                        if identifier == original.id)
        self.ui.takes.edit(existing)
        self.ui.update()
        self.assertEqual(self.session.active_take, original.id)
        self.assertEqual(self.ui.edit_action.value, EXTEND)

    def test_delete_action_and_undo_restore_visible_timeline_and_saved_motion(self):
        original = self.make_actions()
        self.select_action(1)
        self.ui.delete_action.click()
        edited = self.session.takes[original.id]
        self.assertEqual([s['prompt'] for s in edited.segments],
                         ['Action 1', 'Action 3'])
        self.assertEqual(len(edited.positions), 208)
        self.assertTrue(self.ui.undo.visible)
        self.assertEqual(len(self.ui.segment_map), 2)
        saved = encode_project(self.session.takes, edited.id, self.session.frame,
                               self.session.scene)
        loaded, _, _, _ = decode_project(saved)
        self.assertEqual([s['prompt'] for s in loaded[edited.id].segments],
                         ['Action 1', 'Action 3'])
        self.ui.undo.click()
        restored = self.session.takes[original.id]
        self.assertEqual(len(restored.positions), 312)
        self.assertEqual([s['prompt'] for s in restored.segments],
                         ['Action 1', 'Action 2', 'Action 3'])
        self.assertEqual(len(self.ui.segment_map), 3)

    def test_delete_take_and_undo_change_the_visible_take_selection(self):
        first = self.make_actions(1)
        self.session.submit('Another take', seconds=4.16, edit_mode='new')
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        second_id = self.session.active_take
        self.ui.update()
        self.ui.delete_take.click()
        self.assertNotIn(second_id, self.session.takes)
        self.assertIn(first.id, self.session.takes)
        self.assertEqual(self.session.active_take, first.id)
        self.assertTrue(self.ui.undo.visible)
        self.ui.undo.click()
        self.assertIn(second_id, self.session.takes)
        self.assertEqual(self.session.active_take, second_id)
        self.assertEqual(len(self.ui.take_map), 2)

    def test_busy_session_ignores_edit_and_delete_commands(self):
        take = self.make_actions(2)
        self.select_action(0)
        self.session.busy = True
        self.ui.update()
        self.assertTrue(self.ui.generate.disabled)
        self.assertTrue(self.ui.delete_action.disabled)
        self.assertTrue(self.ui.delete_take.disabled)
        self.ui.delete_action.click()
        self.ui.delete_take.click()
        self.ui.generate.click()
        self.assertEqual(len(self.session.takes), 1)
        self.assertIs(self.session.takes[take.id], take)
        self.session.busy = False
        self.ui.update()


if __name__ == '__main__':
    unittest.main()
