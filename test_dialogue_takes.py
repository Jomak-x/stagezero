"""Dialogue stays attached to saved single-actor motion through timeline edits."""
import io
import json
import unittest
from unittest.mock import patch

import numpy as np

from directing import DirectorSession
from take_editing import extend_take_hold
from takes import decode_project, encode_project, merge_dialogue, slice_dialogue, validate_take
from test_live_motion import ControlledBackend, wait_until


class DialogueTakeTests(unittest.TestCase):
    def setUp(self):
        self.backend = ControlledBackend()
        self.backend.release.set()
        self.session = DirectorSession(self.backend, np.zeros((20, 34, 3), dtype=np.float32),
                                       np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1)))
        self.session.set_mode('Live ARDY')
        self.session.submit('Walk', seconds=.8)
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        self.take = self.session.takes[self.session.active_take]
        self.take.dialogue = [dict(text='Hello.', voice_id='voice-f', audio_id='a',
                                   character_id='actor', line_id='line-1', start_frame=2,
                                   end_frame=14)]
        self.take.audio_assets = {'a': b'fake encoded audio'}

    def archive(self):
        return encode_project(self.session.takes, self.session.active_take, 0,
                              self.session.scene, self.session.cameras)

    def test_version_two_archive_retains_cameras_and_audio(self):
        camera = self.session.add_camera([0, 2, 4], [1, 0, 0, 0], 1.0)
        self.session.add_camera_cut(self.take.id, 0, camera['id'])
        restored, active, _, _, cameras = decode_project(self.archive(), include_cameras=True)
        self.assertEqual(cameras[0]['id'], camera['id'])
        self.assertEqual(restored[active].camera_cuts, self.take.camera_cuts)
        self.assertEqual(restored[active].dialogue, self.take.dialogue)
        self.assertEqual(restored[active].audio_assets, self.take.audio_assets)

    def test_old_archive_defaults_to_empty_dialogue(self):
        with np.load(io.BytesIO(self.archive()), allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files if '_audio_' not in key}
        manifest = json.loads(str(arrays['manifest']))
        manifest['takes'][0].pop('dialogue')
        manifest['takes'][0].pop('audio_ids')
        arrays['manifest'] = np.array(json.dumps(manifest))
        output = io.BytesIO()
        np.savez_compressed(output, **arrays)
        restored, active, _, _ = decode_project(output.getvalue())
        self.assertEqual(restored[active].dialogue, [])
        self.assertEqual(restored[active].audio_assets, {})

    def test_trim_rebases_cue_and_drops_unreferenced_asset(self):
        self.take.dialogue.append(dict(text='Later.', voice_id='voice-m', audio_id='b',
                                       start_frame=15, end_frame=20))
        self.take.audio_assets['b'] = b'later'
        self.session.seek(7)
        trimmed = self.session.trim_after_playhead()
        self.assertEqual((trimmed.dialogue[0]['start_frame'], trimmed.dialogue[0]['end_frame']), (2, 8))
        self.assertEqual(trimmed.audio_assets, {'a': b'fake encoded audio'})
        self.assertEqual(self.take.dialogue[0]['end_frame'], 14)

    def test_audio_offset_and_collision_safe_append(self):
        clipped, assets = slice_dialogue(self.take, 5, 20)
        self.assertEqual((clipped[0]['start_frame'], clipped[0]['end_frame']), (0, 9))
        self.assertAlmostEqual(clipped[0]['audio_offset_seconds'], 3 / 25)
        self.assertEqual(assets, self.take.audio_assets)
        other = self.session.duplicate_active_take()
        other.audio_assets['a'] = b'different encoded audio'
        merged, merged_assets = merge_dialogue(((self.take, 0), (other, len(self.take.positions))))
        self.assertNotEqual(merged[0]['audio_id'], merged[1]['audio_id'])
        self.assertEqual(len(merged_assets), 2)
        self.assertEqual(merged[1]['start_frame'], len(self.take.positions) + 2)

    def test_action_regeneration_invalidates_affected_cues_and_undo_restores(self):
        self.take.segments = [dict(start=0, end=10, prompt='Walk'),
                              dict(start=10, end=20, prompt='Turn')]
        self.take.dialogue = [dict(text='Before', voice_id='v', audio_id='a', start_frame=1, end_frame=3),
                              dict(text='After', voice_id='v', audio_id='b', start_frame=11, end_frame=16)]
        self.take.audio_assets = {'a': b'first', 'b': b'second'}
        self.assertTrue(self.session.submit_action_edit('Wave', 1, 'replace'))
        wait_until(lambda: not self.session.busy)
        edited = self.session.takes[self.take.id]
        self.assertEqual([cue['text'] for cue in edited.dialogue], ['Before'])
        self.assertEqual(edited.audio_assets, {'a': b'first'})
        self.assertTrue(self.session.undo_action_edit())
        self.assertEqual(len(self.session.takes[self.take.id].dialogue), 2)

    def test_stationary_tail_preserves_motion_and_cameras(self):
        camera = self.session.add_camera([0, 2, 4], [1, 0, 0, 0], 1.0)
        self.session.add_camera_cut(self.take.id, 0, camera['id'])
        previous = self.take
        original_length = len(previous.positions)
        extended = extend_take_hold(self.session, previous.id, original_length + 15)
        self.assertEqual(len(extended.positions), original_length + 15)
        np.testing.assert_array_equal(extended.positions[:original_length], previous.positions)
        np.testing.assert_array_equal(extended.positions[-1], previous.positions[-1])
        np.testing.assert_array_equal(extended.rotations[-1], previous.rotations[-1])
        np.testing.assert_array_equal(extended.motion[-1], previous.motion[-1])
        self.assertEqual(extended.camera_cuts, previous.camera_cuts)
        self.assertEqual(extended.dialogue, previous.dialogue)
        validate_take(extended)
        restored, _, _, _, _ = decode_project(self.archive(), include_cameras=True)
        self.assertEqual(restored[previous.id].audio_assets, previous.audio_assets)

    def test_voice_line_and_tail_commit_atomically(self):
        previous = self.take
        revision = self.session.project_revision
        cue = dict(text='Second line', voice_id='voice-m', audio_id='new',
                   start_frame=18, end_frame=35)
        assets = {**previous.audio_assets, 'new': b'new audio'}
        with self.assertRaisesRegex(ValueError, 'timeline'):
            extend_take_hold(self.session, previous.id, 30,
                             dialogue=previous.dialogue + [cue], audio_assets=assets)
        self.assertIs(self.session.takes[previous.id], previous)
        self.assertEqual(self.session.project_revision, revision)
        with patch('take_editing.MAX_PROJECT_AUDIO_BYTES', len(b'fake encoded audio')):
            with self.assertRaisesRegex(ValueError, 'Project audio'):
                extend_take_hold(self.session, previous.id, 35,
                                 dialogue=previous.dialogue + [cue], audio_assets=assets)
        self.assertIs(self.session.takes[previous.id], previous)
        self.assertEqual(self.session.project_revision, revision)
        committed = extend_take_hold(self.session, previous.id, 35,
                                     dialogue=previous.dialogue + [cue], audio_assets=assets)
        self.assertEqual(len(committed.positions), 35)
        self.assertEqual(committed.dialogue[-1], cue)
        self.assertEqual(committed.audio_assets['new'], b'new audio')
        self.assertEqual(self.session.project_revision, revision + 1)
        self.assertEqual(self.session.positions.shape[0], 35)


if __name__ == '__main__':
    unittest.main()
