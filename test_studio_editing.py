"""Editing controls operate on stored motion without invoking the backend."""
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from directing import DirectorSession
from takes import decode_project, encode_project
from test_live_motion import ControlledBackend, result, wait_until
from motion_quality import JOINT_INDEX


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

    def test_scene_beat_ids_survive_editing_first_movement_and_suffix(self):
        take = self.generate()
        take.segments = [
            dict(start=0, end=52, prompt='Fall to the floor', beat_id='fall'),
            dict(start=52, end=104, prompt='Wave', beat_id='wave'),
        ]
        from test_story_action_execution import standing

        def falling_result(request_id, prompt, history):
            output = result(request_id)
            output['positions'] = standing(104)
            output['positions'][20:, JOINT_INDEX['pelvis_skel'], 1] = .2
            for side in ('left', 'right'):
                output['positions'][20:, JOINT_INDEX[f'{side}_shoulder_pitch_skel'], 1] = .2
                output['positions'][20:, JOINT_INDEX[f'{side}_shoulder_pitch_skel'], 2] = .5
            return output

        self.backend.generate = falling_result
        self.assertTrue(self.session.submit_action_edit(
            'Fall carefully', 0, 'replace', seconds=2.08))
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        edited = self.session.takes[take.id]
        self.assertIsNot(edited, take)
        self.assertEqual([segment.get('beat_id') for segment in edited.segments],
                         ['fall', 'wave'])
        self.assertEqual([segment['prompt'] for segment in edited.segments],
                         ['Fall carefully', 'Wave'])

    def test_auto_get_up_edit_extends_until_upright_and_keeps_undo(self):
        take = self.generate()
        take.segments = [dict(start=0, end=52,
                              prompt='Get up from lying on the floor', beat_id='rise'),
                         dict(start=52, end=104, prompt='Dance', beat_id='dance')]
        calls = []

        def rising_result(request_id, prompt, history):
            calls.append((prompt, history))
            output = result(request_id)
            output['positions'][:] = 0
            # The first short chunk stays prone. The continued chunk rises
            # after 70 frames, leaving ten stable upright frames by frame 80.
            if len(calls) == 2:
                standing = output['positions'][70:]
                standing[:, JOINT_INDEX['pelvis_skel'], 1] = 1.0
                for side in ('left', 'right'):
                    standing[:, JOINT_INDEX[f'{side}_hip_yaw_skel'], 1] = .8
                    standing[:, JOINT_INDEX[f'{side}_knee_skel'], 1] = .4
                    standing[:, JOINT_INDEX[f'{side}_shoulder_pitch_skel'], 1] = 1.5
            return output

        self.backend.generate = rising_result
        self.assertTrue(self.session.submit_action_edit(
            'Get up from lying on the floor', 0, 'replace',
            automatic_timing=True))
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        edited = self.session.takes[take.id]
        self.assertIsNot(edited, take)
        self.assertEqual(edited.segments[0]['end'], 132)
        self.assertEqual(edited.segments[1]['start'], 132)
        self.assertEqual(edited.segments[0]['beat_id'], 'rise')
        self.assertEqual(edited.segments[1]['beat_id'], 'dance')
        self.assertEqual(calls[0][0], 'A person pushes up from lying on the floor and stands upright.')
        self.assertEqual(len(calls), 3)
        self.assertTrue(self.session.can_undo_action_edit)
        self.assertTrue(self.session.undo_action_edit())
        self.assertIs(self.session.takes[take.id], take)

    def test_fixed_get_up_edit_keeps_original_when_terminal_pose_is_prone(self):
        take = self.generate()
        take.segments[0]['beat_id'] = 'rise'
        revision = self.session.action_edit_revision
        self.assertTrue(self.session.submit_action_edit(
            'Get up from lying on the floor', 0, 'replace',
            seconds=2.0, automatic_timing=False))
        wait_until(lambda: not self.session.busy)
        self.assertIs(self.session.takes[take.id], take)
        self.assertEqual(self.session.action_edit_revision, revision)
        self.assertIn('did not finish upright', self.session.status)

    def test_action_edit_retries_quality_rejection_only(self):
        take = self.generate()
        attempts = []

        def quality_then_success(request_id, prompt, history):
            attempts.append(request_id)
            if len(attempts) < 3:
                raise RuntimeError('Motion quality rejected 3 attempts: '
                                   'motion discontinuity. No motion committed.')
            return result(request_id)

        self.backend.generate = quality_then_success
        self.assertTrue(self.session.submit_action_edit('Wave again', 0, 'replace'))
        wait_until(lambda: not self.session.busy)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(len(set(attempts)), 3)
        self.assertEqual(self.session.takes[take.id].segments[0]['prompt'], 'Wave again')

        updated = self.session.takes[take.id]
        attempts.clear()

        def unavailable(request_id, prompt, history):
            attempts.append(request_id)
            raise RuntimeError('Backend unavailable')

        self.backend.generate = unavailable
        self.assertTrue(self.session.submit_action_edit('Wave once more', 0, 'replace'))
        wait_until(lambda: not self.session.busy)
        self.assertEqual(len(attempts), 1)
        self.assertIs(self.session.takes[take.id], updated)

    def test_edit_preserves_suffix_timing_mode_after_archive_reload(self):
        from test_story_jobs import StoryQueueTests

        for suffix_mode, edit_auto, expected_length in (
            ('fixed', True, 79), ('auto', False, 50), (None, True, 79),
        ):
            with self.subTest(suffix_mode=suffix_mode, edit_auto=edit_auto):
                self.session.new_take()
                take = self.generate()
                take.segments = [
                    dict(start=0, end=25, prompt='A person falls.', beat_id='fall'),
                    dict(start=25, end=104, prompt='A person gets up.', beat_id='rise'),
                ]
                if suffix_mode is not None:
                    take.segments[1]['timing_mode'] = suffix_mode
                data = encode_project(self.session.takes, take.id, 0, self.session.scene)
                self.session.load_project(data)

                def standing_result(request_id, prompt, history):
                    output = result(request_id)
                    output['positions'] = StoryQueueTests.standing_positions()
                    return output

                self.backend.generate = standing_result
                self.assertTrue(self.session.submit_action_edit(
                    'A person waves.', 0, 'replace', automatic_timing=edit_auto))
                wait_until(lambda: not self.session.busy)
                edited = self.session.takes[take.id]
                self.assertIn('Undo is available', self.session.status)
                suffix = edited.segments[1]
                self.assertEqual(suffix['end'] - suffix['start'], expected_length)
                self.assertEqual(suffix['timing_mode'], suffix_mode or 'fixed')
                archived = encode_project(self.session.takes, take.id, 0, self.session.scene)
                restored, _, _, _ = decode_project(archived)
                self.assertEqual(restored[take.id].segments, edited.segments)

    def test_archive_rejects_invalid_segment_timing_mode(self):
        take = self.generate()
        take.segments[0]['timing_mode'] = 'arbitrary'
        with self.assertRaisesRegex(ValueError, 'Invalid segment timing mode'):
            encode_project(self.session.takes, take.id, 0, self.session.scene)

    def test_finite_action_edit_keeps_complete_motion_or_preserves_original(self):
        with np.load(Path(__file__).parent / 'tests/fixtures/scene_action_completion.npz', allow_pickle=False) as data:
            captured_fall = data['fall'].copy()
        for automatic in (True, False):
            with self.subTest(automatic=automatic):
                self.session.new_take()
                take = self.generate()
                take.segments = [dict(start=0, end=30, prompt='A person waves.', beat_id='first'),
                                 dict(start=30, end=104, prompt='A person waves.', beat_id='second')]
                calls = []

                def falling_result(request_id, prompt, history):
                    calls.append((prompt, history))
                    output = result(request_id)
                    output['positions'] = captured_fall.copy()
                    return output

                self.backend.generate = falling_result
                self.assertTrue(self.session.submit_action_edit(
                    'A person falls to the ground.', 0, 'replace', automatic_timing=automatic))
                wait_until(lambda: not self.session.busy)
                if automatic:
                    edited = self.session.takes[take.id]
                    self.assertIsNot(edited, take)
                    self.assertEqual(edited.segments[0]['end'], 88)
                    self.assertEqual(len(edited.positions), 162)
                    self.assertEqual(len(calls), 2)
                    np.testing.assert_array_equal(edited.positions[:88], captured_fall[:88])
                    self.assertTrue(self.session.undo_action_edit())
                    self.assertIs(self.session.takes[take.id], take)
                else:
                    self.assertIs(self.session.takes[take.id], take)
                    self.assertEqual(len(calls), 3)
                    self.assertIn('did not complete', self.session.status)

    def test_fixed_action_edit_retries_semantics_before_committing(self):
        from test_story_action_execution import standing

        take = self.generate()
        take.segments = [dict(start=0, end=25, prompt='A person waves.'),
                         dict(start=25, end=45, prompt='A person waves.'),
                         dict(start=45, end=104, prompt='A person waves.')]
        calls = []

        def semantic_result(request_id, prompt, history):
            calls.append((request_id, prompt, history.copy()))
            output = result(request_id)
            output['positions'] = standing(104)
            output['motion'][:] = len(calls) + 10
            if len(calls) > 1:
                output['positions'][:, JOINT_INDEX['pelvis_skel'], 1] = .2
                for side in ('left', 'right'):
                    output['positions'][:, JOINT_INDEX[f'{side}_shoulder_pitch_skel'], 1] = .2
                    output['positions'][:, JOINT_INDEX[f'{side}_shoulder_pitch_skel'], 2] = .5
            return output

        self.backend.generate = semantic_result
        self.assertTrue(self.session.submit_action_edit('A person falls.', 1, 'replace'))
        wait_until(lambda: not self.session.busy)
        edited = self.session.takes[take.id]
        self.assertIsNot(edited, take)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len({call[0] for call in calls}), 3)
        np.testing.assert_array_equal(calls[0][2], calls[1][2])
        np.testing.assert_array_equal(edited.motion[:25], take.motion[:25])
        np.testing.assert_array_equal(edited.motion[25:45], np.full((20, 414), 12))
        self.assertEqual(len(edited.positions), 104)

    def test_backflip_edit_can_succeed_on_fifth_quality_attempt(self):
        with np.load(Path(__file__).parent / 'tests/fixtures/scene_action_completion.npz', allow_pickle=False) as data:
            captured_flip = data['backflip'].copy()
        take = self.generate()
        calls = []

        def retrying_flip(request_id, prompt, history):
            calls.append(request_id)
            if len(calls) < 5:
                raise RuntimeError('Motion quality rejected 3 attempts: motion discontinuity. No motion committed.')
            output = result(request_id)
            output['positions'] = captured_flip.copy()
            return output

        self.backend.generate = retrying_flip
        self.assertTrue(self.session.submit_action_edit(
            'A person performs a backflip.', 0, 'replace', automatic_timing=True))
        wait_until(lambda: not self.session.busy)
        edited = self.session.takes[take.id]
        self.assertIsNot(edited, take)
        self.assertEqual(len(calls), 5)
        self.assertEqual(len(set(calls)), 5)
        self.assertEqual(len(edited.positions), 83)
        np.testing.assert_array_equal(edited.positions, captured_flip[:83])

    def test_dance_edit_retries_frozen_committed_window_without_changing_duration(self):
        from test_story_jobs import active_dance_positions

        take = self.generate()
        take.segments = [dict(start=0, end=25, prompt='A person waves.'),
                         dict(start=25, end=50, prompt='A person waves.'),
                         dict(start=50, end=104, prompt='A person waves.')]
        calls = []

        def dance_result(request_id, prompt, history):
            calls.append((request_id, history.copy()))
            output = result(request_id)
            output['positions'] = active_dance_positions()
            output['motion'][:] = 10 + len(calls)
            if len(calls) == 1:
                output['positions'][:25] = output['positions'][0]
            return output

        self.backend.generate = dance_result
        self.assertTrue(self.session.submit_action_edit('A person dances.', 1, 'replace'))
        wait_until(lambda: not self.session.busy)
        edited = self.session.takes[take.id]
        self.assertIsNot(edited, take)
        self.assertEqual(len(edited.positions), 104)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len({call[0] for call in calls}), 3)
        np.testing.assert_array_equal(calls[0][1], calls[1][1])
        np.testing.assert_array_equal(edited.motion[:25], take.motion[:25])
        np.testing.assert_array_equal(edited.motion[25:50], np.full((25, 414), 12))

    def test_auto_early_completion_acknowledges_last_generated_request(self):
        from test_story_action_execution import standing

        self.session.set_mode('Live ARDY')
        self.session.submit('A person waves.', seconds=8.32)
        wait_until(lambda: not self.session.busy)
        requests = []

        def upright_result(request_id, prompt, history):
            requests.append(request_id)
            output = result(request_id)
            output['positions'] = standing(104)
            return output

        self.backend.generate = upright_result
        self.assertTrue(self.session.submit_action_edit(
            'A person stands upright.', 0, 'replace', automatic_timing=True))
        wait_until(lambda: not self.session.busy)
        self.assertEqual(len(self.session.positions), 25)
        self.assertEqual(len(requests), 1)
        request_id = self.session.needs_ack[0]
        self.assertEqual(request_id, requests[-1])
        self.assertEqual(request_id, self.session.metrics['request_id'])
        self.session.record_ack(request_id, .25)
        self.assertEqual(self.session.metrics['command_to_browser_render_ack_seconds'], .25)


if __name__ == '__main__':
    unittest.main()
