"""Stored take sequencing preserves physical features and later continuation."""
import unittest
from unittest import mock

import numpy as np

from directing import DirectorSession
from take_editing import append_saved_take
from take_sequencing import align_take, motion_statistics
from takes import Take, decode_project, encode_project, validate_take
from test_live_motion import ControlledBackend, wait_until


def make_take(take_id, x, z, angle, frames=8):
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=np.float32)
    roots = np.zeros((frames, 3), dtype=np.float32)
    roots[:, 0], roots[:, 1], roots[:, 2] = x + np.arange(frames) * .04, .8, z
    rest = np.zeros((34, 3), dtype=np.float32)
    rest[:, 1] = np.linspace(0, .7, 34)
    rest[1, 0], rest[8, 0] = .1, -.1
    positions = rest @ rotation.T + roots[:, None]
    rotations = np.tile(rotation, (frames, 34, 1, 1))
    features = np.zeros((frames, 414), dtype=np.float32)
    features[:, :3] = roots
    features[:, 3:5] = (c, s)
    local = positions[:, 1:].copy()
    local[..., 0] -= roots[:, None, 0]
    local[..., 2] -= roots[:, None, 2]
    features[:, 5:104] = local.reshape(frames, 99)
    features[:, 104:308] = np.concatenate((rotations[..., 0], rotations[..., 1]), axis=-1).reshape(frames, 204)
    features[:, 308:410] = np.tile((1., 0., 0.), 34)
    features[:, 410:414] = (1., 0., 1., 0.)
    mean, scale = motion_statistics()
    motion = (features - mean) / scale
    return Take(take_id, take_id, positions, rotations, motion,
                segments=[dict(start=0, end=4, prompt='walk'), dict(start=4, end=frames, prompt='wave')])


class TakeSequencingTests(unittest.TestCase):
    def setUp(self):
        self.backend = ControlledBackend()
        self.backend.release.set()
        self.session = DirectorSession(self.backend, np.zeros((20, 34, 3), dtype=np.float32),
                                       np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1)))
        self.session.set_mode('Live ARDY')
        self.target = make_take('first', 2, 3, np.pi / 2)
        self.source = make_take('second', -4, -5, 0)
        self.session.takes = {t.id: t for t in (self.target, self.source)}
        self.session.select_take(self.target.id)
        self.session.scene['gate']['enabled'] = False

    def test_alignment_preserves_normalized_motion_and_body_features(self):
        positions, rotations, motion = align_take(self.source, self.target)
        mean, scale = motion_statistics()
        features = motion * scale + mean
        np.testing.assert_allclose(positions[0, 0], self.target.positions[-1, 0], atol=1e-6)
        np.testing.assert_allclose(features[:, :3], positions[:, 0], atol=1e-6)
        np.testing.assert_allclose(features[:, 3:5], np.tile((0., 1.), (8, 1)), atol=1e-6)
        decoded_rotations = features[:, 104:308].reshape(8, 34, 6)
        np.testing.assert_allclose(decoded_rotations[..., :3], rotations[..., 0], atol=1e-6)
        np.testing.assert_allclose(decoded_rotations[..., 3:], rotations[..., 1], atol=1e-6)
        decoded_positions = features[:, 5:104].reshape(8, 33, 3).copy()
        decoded_positions[..., 0] += features[:, None, 0]
        decoded_positions[..., 2] += features[:, None, 2]
        np.testing.assert_allclose(decoded_positions, positions[:, 1:], atol=1e-6)
        np.testing.assert_allclose(features[:, 308:410].reshape(8, 34, 3),
                                   np.tile((0., 0., -1.), (8, 34, 1)), atol=1e-6)
        np.testing.assert_allclose(features[:, 410:414], np.tile((1., 0., 1., 0.), (8, 1)), atol=1e-6)
        np.testing.assert_array_equal(positions[..., 1], self.source.positions[..., 1])

    def test_append_at_end_preserves_source_prefix_and_saves(self):
        source_arrays = [getattr(self.source, key).copy() for key in ('positions', 'rotations', 'motion')]
        revision, clip_revision = self.session.project_revision, self.session.clip_revision
        self.session.seek(2)  # Append does not overwrite from the playhead.
        combined = self.session.append_take(self.source.id)
        self.assertEqual(combined.id, self.target.id)
        self.assertEqual(len(combined.positions), 16)
        self.assertEqual(self.session.frame, 8)
        self.assertFalse(self.session.playing)
        self.assertEqual(self.session.project_revision, revision + 1)
        self.assertGreater(self.session.clip_revision, clip_revision)
        self.assertEqual([(s['start'], s['end']) for s in combined.segments], [(0, 4), (4, 8), (8, 12), (12, 16)])
        for key, source_array in zip(('positions', 'rotations', 'motion'), source_arrays):
            np.testing.assert_array_equal(getattr(combined, key)[:8], getattr(self.target, key))
            np.testing.assert_array_equal(getattr(self.source, key), source_array)
        self.assertEqual(self.backend.histories, [])
        loaded, active, frame, _ = decode_project(encode_project(self.session.takes, combined.id, 8, self.session.scene))
        np.testing.assert_array_equal(loaded[active].motion, combined.motion)
        self.assertEqual(frame, 8)

    def test_later_generation_receives_aligned_append_history(self):
        combined = append_saved_take(self.session, self.source.id)
        self.session.submit('walk forward', edit_mode='extend')
        wait_until(lambda: not self.session.busy)
        np.testing.assert_array_equal(self.backend.histories[-1], combined.motion)

    def test_append_preserves_camera_cuts_and_conflicting_dialogue_audio(self):
        first_camera = self.session.add_camera([0, 2, 4], [1, 0, 0, 0], 1.0)
        second_camera = self.session.add_camera([1, 3, 5], [1, 0, 0, 0], 1.2)
        self.target.camera_cuts = [dict(id='target-cut', frame=0, camera_id=first_camera['id'])]
        self.source.camera_cuts = [dict(id='source-cut', frame=0, camera_id=second_camera['id'])]
        self.target.audio_assets = {'line': b'first audio'}
        self.source.audio_assets = {'line': b'second audio'}
        self.target.dialogue = [dict(text='First', voice_id='voice', audio_id='line',
                                     start_frame=0, end_frame=4)]
        self.source.dialogue = [dict(text='Second', voice_id='voice', audio_id='line',
                                     start_frame=0, end_frame=4)]
        combined = append_saved_take(self.session, self.source.id)
        self.assertEqual([(cut['frame'], cut['camera_id']) for cut in combined.camera_cuts],
                         [(0, first_camera['id']), (8, second_camera['id'])])
        self.assertEqual([(cue['text'], cue['start_frame'], cue['end_frame'])
                          for cue in combined.dialogue], [('First', 0, 4), ('Second', 8, 12)])
        self.assertNotEqual(combined.dialogue[0]['audio_id'], combined.dialogue[1]['audio_id'])
        self.assertEqual(set(combined.audio_assets.values()), {b'first audio', b'second audio'})
        validate_take(combined)
        loaded, active, _, _, cameras = decode_project(
            encode_project(self.session.takes, combined.id, self.session.frame,
                           self.session.scene, self.session.cameras), include_cameras=True)
        self.assertEqual(loaded[active].dialogue, combined.dialogue)
        self.assertEqual(loaded[active].camera_cuts, combined.camera_cuts)
        self.assertEqual(len(cameras), 2)
        self.assertTrue(self.session.undo_action_edit())
        self.assertIs(self.session.takes[self.target.id], self.target)

    def test_events_are_recomputed_for_new_world_path(self):
        self.source.events = [dict(type='gate_open', frame=1)]
        self.session.scene['gate'] = dict(position=[50., 0., 50.], radius=.1, enabled=True)
        combined = append_saved_take(self.session, self.source.id)
        self.assertEqual(combined.events, [])  # Old source gate was somewhere else.
        self.session.takes[self.target.id] = self.target
        self.session.select_take(self.target.id)
        point = self.target.positions[-1, 0].copy()
        point[2] -= .16
        self.session.scene['gate'] = dict(position=point.tolist(), radius=.01, enabled=True)
        combined = append_saved_take(self.session, self.source.id)
        self.assertEqual(combined.events, [dict(type='gate_open', frame=12)])

    def test_invalid_requests_leave_project_unchanged(self):
        revision = self.session.project_revision
        for source_id in ('missing', self.target.id, None):
            with self.assertRaises(ValueError):
                append_saved_take(self.session, source_id)
        self.session.busy = True
        with self.assertRaisesRegex(ValueError, 'Wait for generation'):
            append_saved_take(self.session, self.source.id)
        self.assertTrue(self.session.busy)
        self.session.busy = False
        with mock.patch('take_editing.MAX_FRAMES', 15):
            with self.assertRaisesRegex(ValueError, 'budget'):
                append_saved_take(self.session, self.source.id)
        with mock.patch('take_editing.MAX_TOTAL_FRAMES', 23):
            with self.assertRaisesRegex(ValueError, 'budget'):
                append_saved_take(self.session, self.source.id)
        self.assertIs(self.session.takes[self.target.id], self.target)
        self.assertEqual(self.session.project_revision, revision)
        self.session.set_mode('Recorded preview')
        with self.assertRaisesRegex(ValueError, 'Select the take'):
            append_saved_take(self.session, self.source.id)

    def test_full_take_count_still_allows_append(self):
        with mock.patch('directing.MAX_TAKES', 2):
            self.assertEqual(len(append_saved_take(self.session, self.source.id).positions), 16)


if __name__ == '__main__':
    unittest.main()
