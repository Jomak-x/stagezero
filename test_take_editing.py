"""Editing alters stored motion transactionally and keeps future generation coherent."""
import tempfile
import unittest
from unittest import mock
import numpy as np

from directing import DirectorSession
from live_motion import MODEL
from take_editing import pose_of_take
from take_sequencing import motion_statistics
from takes import decode_project, encode_project, validate_take
from test_live_motion import ControlledBackend, wait_until
from test_take_sequencing import make_take


class PhysicalBackend(ControlledBackend):
    def generate(self, request_id, prompt, history):
        self.histories.append(history)
        self.started.set()
        self.release.wait(2)
        if self.error:
            raise RuntimeError('Backend unavailable')
        take = make_take('generated', -7, 4, -.7, frames=104)
        return {k: getattr(take, k) for k in ('positions', 'rotations', 'motion')} | {
            'metadata': dict(request_id=request_id, model=MODEL, fps=25, generation_seconds=.1)}


class TakeEditingTests(unittest.TestCase):
    def setUp(self):
        self.backend = PhysicalBackend()
        self.backend.release.set()
        reference = make_take('reference', 0, 0, 0)
        self.s = DirectorSession(self.backend, reference.positions, reference.rotations)
        self.s.set_mode('Live ARDY')
        self.original = make_take('one', 2, -3, .2, frames=12)
        self.original.segments = [dict(start=i*4, end=(i+1)*4, prompt=p)
                                  for i, p in enumerate(('walk', 'wave', 'dance'))]
        self.s.takes = {'one': self.original}
        self.s.select_take('one')
        self.s.scene['gate']['enabled'] = False

    def roundtrip(self):
        return decode_project(encode_project(self.s.takes, self.s.active_take, self.s.frame, self.s.scene))

    def assertFeaturesMatch(self, take):
        mean, scale = motion_statistics()
        f = take.motion * scale + mean
        np.testing.assert_allclose(f[:, :3], take.positions[:, 0], atol=2e-6)
        decoded = f[:, 5:104].reshape(-1, 33, 3).copy()
        decoded[..., 0] += f[:, None, 0]
        decoded[..., 2] += f[:, None, 2]
        np.testing.assert_allclose(decoded, take.positions[:, 1:], atol=3e-6)
        columns = f[:, 104:308].reshape(-1, 34, 6)
        np.testing.assert_allclose(columns[..., :3], take.rotations[..., 0], atol=2e-6)
        np.testing.assert_allclose(columns[..., 3:], take.rotations[..., 1], atol=2e-6)

    def test_middle_delete_joins_sequence_and_undo_is_exact(self):
        edited = self.s.delete_action(1)
        self.assertEqual([(s['start'], s['end'], s['prompt']) for s in edited.segments],
                         [(0, 4, 'walk'), (4, 8, 'dance')])
        np.testing.assert_array_equal(edited.positions[:4], self.original.positions[:4])
        np.testing.assert_allclose(edited.positions[4, 0], edited.positions[3, 0], atol=1e-6)
        self.assertFeaturesMatch(edited)
        self.roundtrip()
        self.assertTrue(self.s.can_undo_edit)
        self.assertTrue(self.s.undo_edit())
        np.testing.assert_array_equal(self.s.positions, self.original.positions)
        self.assertEqual(self.s.takes['one'].segments, self.original.segments)
        self.assertFalse(self.s.can_undo_edit)
        self.assertFalse(self.s.undo_edit())

    def test_delete_first_keeps_original_start_and_last_keeps_prefix(self):
        edited = self.s.delete_action(0)
        np.testing.assert_allclose(edited.positions[0, 0], self.original.positions[0, 0], atol=1e-6)
        self.assertEqual([s['prompt'] for s in edited.segments], ['wave', 'dance'])
        self.s.undo_edit()
        edited = self.s.delete_action(2)
        np.testing.assert_array_equal(edited.motion, self.original.motion[:8])
        self.assertEqual(len(edited.positions), 8)

    def test_delete_take_repairs_children_and_undo_restores_provenance(self):
        child = make_take('child', 0, 0, 0)
        child.parent, child.branch_frame = 'one', 8
        self.s.takes['child'] = child
        self.s.delete_take()
        self.assertEqual(self.s.active_take, 'child')
        self.assertIsNone(self.s.takes['child'].parent)
        self.assertIsNone(self.s.takes['child'].branch_frame)
        self.roundtrip()
        self.s.undo_edit()
        self.assertEqual(self.s.active_take, 'one')
        self.assertEqual(self.s.takes['child'].parent, 'one')
        self.assertEqual(self.s.takes['child'].branch_frame, 8)

    def test_deleting_only_action_returns_blank_take_and_undo_restores_it(self):
        self.original.segments = [dict(start=0, end=12, prompt='walk')]
        self.s.delete_action(0)
        self.assertEqual(self.s.takes, {})
        self.assertIsNone(self.s.active_take)
        self.assertEqual(self.s.kind, 'reference')
        self.assertTrue(self.s.undo_edit())
        self.assertEqual(self.s.active_take, 'one')
        np.testing.assert_array_equal(self.s.motion, self.original.motion)

    def test_placement_keeps_motion_features_and_save_continuation_consistent(self):
        self.s.seek(7)
        edited = self.s.set_start_pose(-5., 6., 90.)
        self.assertEqual(self.s.frame, 0)
        self.assertFalse(self.s.playing)
        np.testing.assert_allclose(self.s.get_start_pose(), (-5., 6., 90.), atol=2e-6)
        self.assertFeaturesMatch(edited)
        np.testing.assert_array_equal(edited.positions[..., 1], self.original.positions[..., 1])
        loaded, active, _, _ = self.roundtrip()
        np.testing.assert_array_equal(loaded[active].motion, edited.motion)
        self.s.submit('walk', edit_mode='extend', seconds=1)
        wait_until(lambda: not self.s.busy)
        np.testing.assert_array_equal(self.backend.histories[-1], edited.motion)

    def test_blank_placement_preview_new_generation_and_draft_save(self):
        self.s.new_take()
        self.s.set_start_pose(7, -9, 135)
        np.testing.assert_allclose(self.s.positions[0, 0, [0, 2]], (7, -9), atol=1e-6)
        self.assertIsNone(self.s.motion)
        with tempfile.TemporaryDirectory() as folder:
            _, content = self.s.save_project(folder, 'draft')
        takes, active, frame, scene = decode_project(content)
        self.assertEqual(active, 'one')
        self.assertEqual(frame, 0)
        self.assertEqual(scene['actor_start'], [7., -9., 135.])
        self.s.new_take()
        np.testing.assert_allclose(self.s.positions[0, 0, [0, 2]], (7, -9), atol=1e-6)
        self.s.submit('walk', edit_mode='new', seconds=1)
        wait_until(lambda: not self.s.busy)
        np.testing.assert_allclose(self.s.get_start_pose(), (7, -9, 135), atol=2e-5)
        self.assertFeaturesMatch(self.s.takes[self.s.active_take])
        self.assertTrue(self.s.undo_edit())
        self.assertIsNone(self.s.active_take)
        self.assertEqual(len(self.s.takes), 1)
        np.testing.assert_allclose(self.s.positions[0, 0, [0, 2]], (7, -9), atol=1e-6)

    def test_replace_middle_action_preserves_following_actions_and_normalized_history(self):
        with mock.patch('directing.MAX_TAKES', 1):
            self.s.submit('new movement', edit_mode='action', at_frame=4, seconds=1)
            wait_until(lambda: not self.s.busy)
        edited = self.s.takes['one']
        self.assertEqual(len(self.s.takes), 1)
        self.assertEqual([s['prompt'] for s in edited.segments], ['walk', 'new movement', 'dance'])
        self.assertEqual([(s['start'], s['end']) for s in edited.segments], [(0, 4), (4, 29), (29, 33)])
        np.testing.assert_array_equal(edited.motion[:4], self.original.motion[:4])
        np.testing.assert_array_equal(self.backend.histories[-1], self.original.motion[:4])
        np.testing.assert_allclose(edited.positions[29, 0], edited.positions[28, 0], atol=1e-6)
        self.assertFeaturesMatch(edited)
        self.roundtrip()
        self.assertTrue(self.s.undo_edit())
        np.testing.assert_array_equal(self.s.motion, self.original.motion)

    def test_replace_first_action_keeps_start_and_generation_error_keeps_original(self):
        start = self.s.get_start_pose()
        self.s.submit('new movement', edit_mode='action', at_frame=0, seconds=1)
        wait_until(lambda: not self.s.busy)
        np.testing.assert_allclose(self.s.get_start_pose(), start, atol=2e-5)
        self.assertEqual([s['prompt'] for s in self.s.takes['one'].segments], ['new movement', 'wave', 'dance'])
        self.s.undo_edit()
        self.backend.error = True
        self.s.submit('bad backend', edit_mode='action', at_frame=4, seconds=1)
        wait_until(lambda: not self.s.busy)
        np.testing.assert_array_equal(self.s.motion, self.original.motion)
        self.assertIn('Generation failed', self.s.status)

    def test_busy_and_invalid_edits_preserve_project(self):
        self.s.busy = True
        for edit in (lambda: self.s.delete_take(), lambda: self.s.delete_action(0), lambda: self.s.set_start_pose(1, 1, 0)):
            with self.assertRaisesRegex(ValueError, 'Wait for generation'):
                edit()
        self.s.busy = False
        for index in (-1, 3, True, None):
            with self.assertRaises(ValueError):
                self.s.delete_action(index)
        for pose in ((float('nan'), 0, 0), (0, 101, 0), (0, 0, float('inf'))):
            with self.assertRaises(ValueError):
                self.s.set_start_pose(*pose)
        self.assertIs(self.s.takes['one'], self.original)
        self.assertEqual(self.s.project_revision, 0)


    def test_drag_updates_coalesce_and_load_clears_undo(self):
        original_pose = self.s.get_start_pose()
        self.s.set_start_pose(1, 2, 30)
        self.s.set_start_pose(2, 3, 45)
        self.s.set_start_pose(3, 4, 60)
        self.assertTrue(self.s.undo_edit())
        np.testing.assert_allclose(self.s.get_start_pose(), original_pose, atol=1e-6)
        self.s.set_start_pose(4, 5, 90)
        content = encode_project(self.s.takes, self.s.active_take, 0, self.s.scene)
        self.s.load_project(content)
        self.assertFalse(self.s.can_undo_edit)

    def test_edit_after_short_first_segment_does_not_require_history(self):
        self.original.segments = [dict(start=0, end=1, prompt='short'), dict(start=1, end=12, prompt='walk')]
        self.s.submit('new', edit_mode='action', at_frame=1, seconds=1)
        wait_until(lambda: not self.s.busy)
        self.assertIsNone(self.backend.histories[-1])
        self.assertEqual(self.s.takes['one'].segments[1]['start'], 1)
        self.assertEqual(self.s.takes['one'].segments[1]['prompt'], 'new')
        self.roundtrip()

    def test_undo_cannot_erase_unrelated_later_changes(self):
        self.s.delete_action(1)
        self.assertTrue(self.s.can_undo_edit)
        self.s.rename_active_take('My new name')
        self.assertFalse(self.s.can_undo_edit)
        self.assertFalse(self.s.undo_edit())
        self.assertEqual(self.s.takes['one'].name, 'My new name')


if __name__ == '__main__':
    unittest.main()
