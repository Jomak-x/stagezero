"""Takes controls edit the motion that the project serializer stores."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from directing import DirectorSession
from studio_ui import StudioUI
from takes import decode_project, encode_project
from test_live_motion import ControlledBackend
from test_studio_ui import Gui
from test_take_sequencing import make_take


class ActionEditorUITests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        upload = mock.patch('studio_ui.install_upload_snapshots')
        upload.start()
        self.addCleanup(upload.stop)
        backend = ControlledBackend()
        backend.release.set()
        self.session = DirectorSession(
            backend, np.zeros((120, 34, 3), dtype=np.float32),
            np.tile(np.eye(3, dtype=np.float32), (120, 34, 1, 1)))
        self.gui = Gui()
        camera = SimpleNamespace(build_gui=lambda gui: None)
        self.ui = StudioUI(SimpleNamespace(gui=self.gui), self.session, camera,
                           Path(self.temp.name), lambda gui: None)
        self.session.set_mode('Live ARDY')
        self.take = make_take('one', 2, 3, .2, frames=12)
        self.take.segments = [dict(start=i * 4, end=(i + 1) * 4, prompt=prompt)
                              for i, prompt in enumerate(('walk', 'wave', 'dance'))]
        self.session.takes = {self.take.id: self.take}
        self.session.select_take(self.take.id)
        self.ui.update()

    def select_action(self, index):
        label = tuple(self.ui.segment_map)[index]
        self.ui.segments.edit(label)
        self.assertEqual(self.ui._selected_sequence, (self.session.takes[self.take.id], index))

    def test_selected_action_opens_current_editor_at_its_start(self):
        self.select_action(1)
        self.assertFalse(self.ui.edit_selected_action.disabled)
        self.ui.edit_selected_action.click()
        self.assertEqual(self.ui.action_edit, (self.take.id, 1, 'replace'))
        self.assertEqual(self.ui.action_prompt.value, 'wave')
        self.assertEqual(self.session.frame, 4)

    def test_delete_and_undo_update_visible_timeline_and_saved_motion(self):
        self.select_action(1)
        self.assertFalse(self.ui.delete_selected_action.disabled)
        self.ui.delete_selected_action.click()
        edited = self.session.takes[self.take.id]
        self.assertEqual([segment['prompt'] for segment in edited.segments], ['walk', 'dance'])
        self.assertEqual(len(edited.positions), 8)
        self.assertEqual(len(self.ui.segment_map), 2)
        saved = encode_project(self.session.takes, edited.id, self.session.frame,
                               self.session.scene, self.session.cameras)
        loaded, active, _, _ = decode_project(saved)
        self.assertEqual([segment['prompt'] for segment in loaded[active].segments], ['walk', 'dance'])
        self.assertFalse(self.ui.undo_sequence_edit.disabled)
        self.ui.undo_sequence_edit.click()
        self.assertIs(self.session.takes[self.take.id], self.take)
        self.assertEqual(len(self.ui.segment_map), 3)

    def test_busy_state_disables_delete_and_append(self):
        self.select_action(1)
        self.session.busy = True
        self.ui.update()
        self.assertTrue(self.ui.delete_selected_action.disabled)
        self.assertTrue(self.ui.append_saved.disabled)
        self.ui.delete_selected_action.click()
        self.assertIs(self.session.takes[self.take.id], self.take)
        self.session.busy = False


if __name__ == '__main__':
    unittest.main()
