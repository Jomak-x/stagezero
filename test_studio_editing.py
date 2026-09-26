"""Editing controls operate on stored motion without invoking the backend."""
import unittest
from unittest import mock

import numpy as np

from directing import DirectorSession
from takes import decode_project, encode_project
from test_live_motion import ControlledBackend, wait_until


class StudioEditingTests(unittest.TestCase):
    def setUp(self):
        self.backend = ControlledBackend()
        self.backend.release.set()
        self.session = DirectorSession(
            self.backend,
            np.zeros((20, 34, 3), dtype=np.float32),
            np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1)),
        )

    def generate(self, prompt='first'):
        self.session.set_mode('Live ARDY')
        self.session.submit(prompt)
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        return self.session.takes[self.session.active_take]

    def test_recorded_seek_speed_and_loop(self):
        self.session.seek(8)
        self.assertEqual(self.session.frame, 8)
        with mock.patch('directing.time.perf_counter', return_value=100.0):
            self.session.set_playback_speed(2)
            self.session.set_loop(True)
            self.session.play()
        with mock.patch('directing.time.perf_counter', return_value=100.1):
            self.session.tick()
        self.assertIn(self.session.frame, (19, 0))  # 8 + 12 frames reaches the loop boundary.
        self.assertTrue(self.session.playing)
        with mock.patch('directing.time.perf_counter', return_value=100.15):
            self.session.set_playback_speed(.5)
        paused_frame = self.session.frame
        with mock.patch('directing.time.perf_counter', return_value=100.25):
            self.session.tick()
        self.assertIn(self.session.frame - paused_frame, (2, 3))
        with self.assertRaises(ValueError):
            self.session.set_playback_speed(0)

    def test_trim_duplicate_rename_and_roundtrip(self):
        original = self.generate()
        self.session.seek(40)
        trimmed = self.session.trim_after_playhead()
        self.assertIsNotNone(trimmed)
        self.assertEqual(len(original.positions), 104)
        self.assertEqual(len(trimmed.positions), 41)
        self.assertEqual(trimmed.parent, original.id)
        self.assertEqual(trimmed.branch_frame, 40)
        self.assertEqual(trimmed.segments[0]['end'], 41)
        self.assertEqual(self.session.frame, 40)
        np.testing.assert_array_equal(trimmed.motion, original.motion[:41])

        self.assertTrue(self.session.rename_active_take('Short performance'))
        duplicated = self.session.duplicate_active_take()
        self.assertEqual(duplicated.name, 'Short performance copy')
        self.assertEqual(duplicated.parent, trimmed.id)
        self.assertEqual(len(duplicated.positions), 41)
        duplicated.positions[0, 0, 0] += 5
        self.assertNotEqual(duplicated.positions[0, 0, 0], trimmed.positions[0, 0, 0])
        data = encode_project(self.session.takes, duplicated.id, self.session.frame, self.session.scene)
        loaded, active, frame, _ = decode_project(data)
        self.assertEqual(active, duplicated.id)
        self.assertEqual(frame, 40)
        self.assertEqual(loaded[trimmed.id].name, 'Short performance')
        self.assertEqual(len(loaded[original.id].positions), 104)

    def test_trim_rejects_too_early_and_end_without_changing_source(self):
        original = self.generate()
        revision = self.session.project_revision
        self.session.seek(2)
        self.assertIsNone(self.session.trim_after_playhead())
        self.session.seek(103)
        self.assertIsNone(self.session.trim_after_playhead())
        self.assertEqual(set(self.session.takes), {original.id})
        self.assertEqual(self.session.project_revision, revision)

    def test_generated_playback_uses_selected_speed(self):
        self.session.set_playback_speed(.5)
        self.generate()
        self.session.seek(10)
        with mock.patch('directing.time.perf_counter', return_value=200.0):
            self.session.play()
        with mock.patch('directing.time.perf_counter', return_value=200.2):
            self.session.tick()
        self.assertIn(self.session.frame, (12, 13))
        self.session.set_loop(False)
        self.session.seek(102)
        with mock.patch('directing.time.perf_counter', return_value=300.0):
            self.session.play()
        with mock.patch('directing.time.perf_counter', return_value=301.0):
            self.session.tick()
        self.assertEqual(self.session.frame, 103)
        self.assertFalse(self.session.playing)

    def test_renamed_take_does_not_collide_with_next_generated_name(self):
        first = self.generate()
        self.assertTrue(self.session.rename_active_take('Take 2'))
        self.session.new_take()
        self.generate('second')
        self.assertEqual(first.name, 'Take 2')
        self.assertEqual(len({t.name for t in self.session.takes.values()}), 2)

    def test_editing_a_stored_take_requires_live_mode(self):
        first = self.generate()
        self.session.set_mode('Recorded preview')
        self.assertIsNone(self.session.duplicate_active_take())
        self.assertIsNone(self.session.trim_after_playhead())
        self.assertFalse(self.session.rename_active_take('Hidden change'))
        self.assertFalse(self.session.next_action())
        self.assertEqual(set(self.session.takes), {first.id})
        self.assertEqual(self.session.kind, 'recorded')

    def test_action_jumps_follow_segment_starts(self):
        original = self.generate()
        self.session.seek(103)
        self.session.submit('second')
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        self.assertEqual(self.session.active_take, original.id)
        self.session.seek(120)
        self.assertTrue(self.session.previous_action())
        self.assertEqual(self.session.frame, 104)
        self.assertTrue(self.session.previous_action())
        self.assertEqual(self.session.frame, 0)
        self.assertTrue(self.session.next_action())
        self.assertEqual(self.session.frame, 104)
        self.assertTrue(self.session.next_action())
        self.assertEqual(self.session.frame, 104)


if __name__ == '__main__':
    unittest.main()
