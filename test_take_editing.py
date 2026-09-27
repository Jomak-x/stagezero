"""Stored action splices keep scene metadata and remain undoable."""
import unittest

import numpy as np

from directing import DirectorSession
from take_editing import delete_stored_action, splice_action
from takes import decode_project, encode_project, validate_take
from test_live_motion import ControlledBackend
from test_take_sequencing import make_take


class TakeEditingTests(unittest.TestCase):
    def setUp(self):
        backend = ControlledBackend()
        backend.release.set()
        self.session = DirectorSession(
            backend, np.zeros((20, 34, 3), dtype=np.float32),
            np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1)))
        self.session.set_mode('Live ARDY')
        self.original = make_take('one', 2, -3, .2, frames=12)
        self.original.segments = [dict(start=i * 4, end=(i + 1) * 4, prompt=prompt)
                                  for i, prompt in enumerate(('walk', 'wave', 'dance'))]
        self.session.takes = {'one': self.original}
        self.session.select_take('one')
        self.session.scene['gate']['enabled'] = False

    def _add_media(self):
        first = self.session.add_camera([0, 2, 4], [1, 0, 0, 0], 1.0)
        second = self.session.add_camera([1, 3, 5], [1, 0, 0, 0], 1.2)
        self.original.camera_cuts = [
            dict(id='cut-a', frame=0, camera_id=first['id']),
            dict(id='cut-b', frame=8, camera_id=second['id']),
        ]
        self.original.audio_assets = {'middle': b'middle audio', 'last': b'last audio'}
        self.original.dialogue = [
            dict(text='Middle', voice_id='voice', audio_id='middle', start_frame=4, end_frame=8),
            dict(text='Last', voice_id='voice', audio_id='last', start_frame=8, end_frame=12),
        ]
        validate_take(self.original)
        return first, second

    def test_delete_middle_retains_suffix_camera_audio_and_undo(self):
        _, second = self._add_media()
        child = make_take('child', 0, 0, 0)
        child.parent, child.branch_frame = 'one', 8
        self.session.takes[child.id] = child
        revision = self.session.project_revision
        edited = self.session.delete_action(1)
        self.assertEqual([(s['start'], s['end'], s['prompt']) for s in edited.segments],
                         [(0, 4, 'walk'), (4, 8, 'dance')])
        np.testing.assert_array_equal(edited.motion[:4], self.original.motion[:4])
        np.testing.assert_allclose(edited.positions[4, 0], edited.positions[3, 0], atol=1e-6)
        self.assertEqual([(cut['frame'], cut['camera_id']) for cut in edited.camera_cuts],
                         [(0, self.original.camera_cuts[0]['camera_id']), (4, second['id'])])
        self.assertEqual([(cue['text'], cue['start_frame'], cue['end_frame'])
                          for cue in edited.dialogue], [('Last', 4, 8)])
        self.assertEqual(edited.audio_assets, {'last': b'last audio'})
        self.assertIsNone(child.parent)
        self.assertEqual(self.session.project_revision, revision + 1)
        loaded, active, _, _, cameras = decode_project(
            encode_project(self.session.takes, edited.id, self.session.frame,
                           self.session.scene, self.session.cameras), include_cameras=True)
        self.assertEqual(loaded[active].dialogue, edited.dialogue)
        self.assertEqual(loaded[active].camera_cuts, edited.camera_cuts)
        self.assertEqual(len(cameras), 2)
        self.assertTrue(self.session.undo_action_edit())
        self.assertIs(self.session.takes['one'], self.original)
        self.assertEqual(child.parent, 'one')
        self.assertEqual(child.branch_frame, 8)

    def test_delete_first_keeps_old_start_and_last_keeps_prefix(self):
        first = delete_stored_action(self.session, 0)
        np.testing.assert_allclose(first.positions[0, 0], self.original.positions[0, 0], atol=1e-6)
        self.assertEqual([s['prompt'] for s in first.segments], ['wave', 'dance'])
        self.assertTrue(self.session.undo_action_edit())
        last = delete_stored_action(self.session, 2)
        np.testing.assert_array_equal(last.motion, self.original.motion[:8])

    def test_splice_replacement_keeps_surrounding_actions_and_media(self):
        self._add_media()
        replacement = make_take('replacement', -8, 4, -.7, frames=8)
        edited = splice_action(self.original, 1, replacement)
        self.assertEqual([s['prompt'] for s in edited.segments],
                         ['walk', 'walk', 'wave', 'dance'])
        self.assertEqual([(s['start'], s['end']) for s in edited.segments],
                         [(0, 4), (4, 8), (8, 12), (12, 16)])
        self.assertEqual(edited.dialogue[-1]['start_frame'], 12)
        self.assertEqual(edited.audio_assets, {'last': b'last audio'})
        validate_take(edited)

    def test_deleting_only_action_uses_take_removal_undo(self):
        self.original.segments = [dict(start=0, end=12, prompt='walk')]
        self.assertIsNone(delete_stored_action(self.session, 0))
        self.assertEqual(self.session.takes, {})
        self.assertTrue(self.session.can_undo_take_removal)
        self.assertTrue(self.session.undo_remove_take())
        self.assertIs(self.session.takes['one'], self.original)

    def test_invalid_or_story_edits_do_not_change_project(self):
        revision = self.session.project_revision
        for index in (-1, 3, True, None):
            with self.assertRaises(ValueError):
                delete_stored_action(self.session, index)
        self.original.segments[1]['beat_id'] = 'story-wave'
        with self.assertRaisesRegex(ValueError, 'story editor'):
            delete_stored_action(self.session, 1)
        self.session.busy = True
        with self.assertRaisesRegex(ValueError, 'Wait for generation'):
            delete_stored_action(self.session, 0)
        self.session.busy = False
        self.assertIs(self.session.takes['one'], self.original)
        self.assertEqual(self.session.project_revision, revision)


if __name__ == '__main__':
    unittest.main()
