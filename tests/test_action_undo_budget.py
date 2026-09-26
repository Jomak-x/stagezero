"""Undoing an action edit must keep the whole project within its motion budget."""

import unittest
from unittest.mock import patch

import numpy as np

from directing import DirectorSession
from takes import decode_project, encode_project
from test_live_motion import ControlledBackend, wait_until


class ActionUndoBudgetTests(unittest.TestCase):
    def setUp(self):
        backend = ControlledBackend()
        backend.release.set()
        positions = np.zeros((20, 34, 3), dtype=np.float32)
        rotations = np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1))
        self.session = DirectorSession(backend, positions, rotations)
        self.session.set_mode('Live ARDY')

    def _finish_generation(self):
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        return self.session.takes[self.session.active_take]

    def _shorten_second_action(self):
        self.session.submit('first')
        self._finish_generation()
        self.session.submit('second', seconds=2, edit_mode='extend')
        original = self._finish_generation()
        self.assertEqual(len(original.positions), 154)
        self.session.seek(120)
        self.assertTrue(self.session.submit_action_edit('short', 1, 'replace', seconds=.16))
        edited = self._finish_generation()
        self.assertEqual(len(edited.positions), 108)
        self.assertTrue(self.session.can_undo_action_edit)
        return original, edited

    def _roundtrip(self):
        archive = encode_project(self.session.takes, self.session.active_take,
                                 self.session.frame, self.session.scene)
        restored, active, frame, _ = decode_project(archive)
        self.assertEqual(active, self.session.active_take)
        self.assertEqual(frame, self.session.frame)
        return restored

    def test_over_budget_undo_preserves_state_and_can_retry_after_removal(self):
        with patch('directing.MAX_TOTAL_FRAMES', 400), patch('takes.MAX_TOTAL_FRAMES', 400):
            original, edited = self._shorten_second_action()
            self.session.seek(104)
            self.session.submit('branch')
            child = self._finish_generation()
            self.assertEqual((child.parent, child.branch_frame), (edited.id, 104))
            self.session.submit('extra', seconds=2, edit_mode='new')
            extra = self._finish_generation()
            self.assertEqual(sum(len(t.positions) for t in self.session.takes.values()), 367)

            self.session.select_take(child.id)
            self.session.seek(110)
            before = (self.session.active_take, self.session.frame,
                      self.session.project_revision, self.session.action_edit_revision,
                      self.session.clip_revision)

            self.assertFalse(self.session.undo_action_edit())
            self.assertIn('budget', self.session.status.lower())
            self.assertTrue(self.session.can_undo_action_edit)
            self.assertEqual((self.session.active_take, self.session.frame,
                              self.session.project_revision, self.session.action_edit_revision,
                              self.session.clip_revision), before)
            self.assertIs(self.session.takes[original.id], edited)
            self.assertIs(self.session.takes[child.id], child)
            self.assertIs(self.session.takes[extra.id], extra)
            self.assertEqual((child.parent, child.branch_frame), (edited.id, 104))
            restored = self._roundtrip()
            self.assertEqual((restored[child.id].parent, restored[child.id].branch_frame),
                             (edited.id, 104))

            self.session.select_take(extra.id)
            self.assertTrue(self.session.remove_active_take())
            revision = self.session.project_revision
            self.assertTrue(self.session.undo_action_edit())
            self.assertIs(self.session.takes[original.id], original)
            self.assertEqual((self.session.active_take, self.session.frame), (original.id, 120))
            self.assertEqual(self.session.project_revision, revision + 1)
            self.assertFalse(self.session.can_undo_action_edit)
            self.assertEqual((child.parent, child.branch_frame), (None, None))
            self.assertEqual(sum(len(t.positions) for t in self.session.takes.values()), 363)
            self._roundtrip()

    def test_exact_limit_undo_subtracts_edited_take_when_another_is_active(self):
        with patch('directing.MAX_TOTAL_FRAMES', 204), patch('takes.MAX_TOTAL_FRAMES', 204):
            original, edited = self._shorten_second_action()
            self.session.submit('other', seconds=2, edit_mode='new')
            other = self._finish_generation()
            self.session.seek(17)
            self.assertEqual(self.session.active_take, other.id)
            self.assertEqual(sum(len(t.positions) for t in self.session.takes.values()), 158)
            revision = self.session.project_revision

            self.assertTrue(self.session.undo_action_edit())
            self.assertIs(self.session.takes[original.id], original)
            self.assertIs(self.session.takes[other.id], other)
            self.assertIsNot(self.session.takes[original.id], edited)
            self.assertEqual((self.session.active_take, self.session.frame), (original.id, 120))
            self.assertEqual(self.session.project_revision, revision + 1)
            self.assertFalse(self.session.can_undo_action_edit)
            self.assertEqual(sum(len(t.positions) for t in self.session.takes.values()), 204)
            restored = self._roundtrip()
            self.assertEqual(len(restored[original.id].positions), 154)
            self.assertEqual(len(restored[other.id].positions), 50)


if __name__ == '__main__':
    unittest.main()
