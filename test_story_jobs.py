"""Offline queue checks for complete scene generation and cancellation."""

import threading
import time
import unittest
from pathlib import Path

import numpy as np

from story_jobs import StoryJobQueue, _recovered_upright
from motion_quality import JOINT_INDEX, ROOT, SHOULDERS
from story_recovery import (FLOOR_RECOVERY_PROMPT, quality_failure_reasons,
                            recovery_completion_frame, recovery_prompt)
from story_planning import fit_story_duration


def active_dance_positions():
    """Synthetic articulated activity for transport tests, not model motion."""
    from test_story_action_execution import standing
    poses = standing(104)
    poses[:, :, 0] += (.06 * np.sin(np.arange(104) * .35))[:, None]
    poses[:, ROOT, 0] = 0
    return poses


def plan(seconds=8.32):
    source = {'version': 1, 'title': 'Walk and wave', 'prompt': 'Walk, then wave.',
              'beats': [{'id': 'beat-1', 'prompt': 'Walk forward.', 'seconds': 4.16},
                        {'id': 'beat-2', 'prompt': 'Wave a hand.', 'seconds': 4.16}],
              'warnings': []}
    return fit_story_duration(source, seconds)


def until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('Timed out waiting for background story work')


class FakeBackend:
    def __init__(self, url='pod-one'):
        self.url = url
        self.calls = []
        self.cancelled = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def generate(self, request_id, prompt, history):
        self.calls.append((request_id, prompt, None if history is None else history.copy()))
        self.started.set()
        self.release.wait(3)
        number = len(self.calls)
        positions = np.zeros((104, 34, 3), dtype=np.float32)
        if 'dances energetically' in prompt:
            positions = active_dance_positions()
        rotations = np.tile(np.eye(3, dtype=np.float32), (104, 34, 1, 1))
        motion = np.full((104, 414), number, dtype=np.float32)
        return {'positions': positions, 'rotations': rotations, 'motion': motion,
                'metadata': {'request_id': request_id, 'model': 'ARDY-G1-RP-25FPS-Horizon52',
                             'fps': 25, 'generation_seconds': .01}}

    def cancel(self, request_id):
        self.cancelled.append(request_id)


class StoryQueueTests(unittest.TestCase):
    def test_dance_retry_checks_only_committed_window_and_preserves_history(self):
        class DanceBackend(FakeBackend):
            def generate(self, request_id, prompt, history):
                output = super().generate(request_id, prompt, history)
                if len(self.calls) == 2:
                    # Activity later in a discarded tail must not hide a frozen
                    # requested one-second window.
                    output['positions'][:25] = output['positions'][0]
                return output

        source = {'version': 1, 'title': 'Dance retry', 'prompt': 'Wave, then dance, then wave.',
                  'beats': [dict(id='beat-1', prompt='A person waves.', seconds=1),
                            dict(id='beat-2', prompt='A person dances.', seconds=1),
                            dict(id='beat-3', prompt='A person waves.', seconds=1)], 'warnings': []}
        backend = DanceBackend()
        queue = StoryJobQueue([backend])
        try:
            identifier = queue.submit(source)
            until(lambda: queue.snapshot(identifier)['status'] in ('completed', 'failed'))
            self.assertEqual(queue.snapshot(identifier)['status'], 'completed')
            self.assertEqual(len(backend.calls), 4)
            self.assertEqual(len({call[0] for call in backend.calls}), 4)
            np.testing.assert_array_equal(backend.calls[1][2], backend.calls[2][2])
            take = queue.result(identifier)
            self.assertEqual(len(take.motion), 75)
            np.testing.assert_array_equal(take.motion[25:50], np.full((25, 414), 3))
        finally:
            queue.close()

    def test_frozen_dance_exhausts_bounded_retries_without_suffix(self):
        class FrozenBackend(FakeBackend):
            def generate(self, request_id, prompt, history):
                output = super().generate(request_id, prompt, history)
                output['positions'][:] = output['positions'][0]
                return output

        source = {'version': 1, 'title': 'Dance freeze', 'prompt': 'Dance, then wave.',
                  'beats': [dict(id='beat-1', prompt='A person dances.', seconds=1),
                            dict(id='beat-2', prompt='A person waves.', seconds=1)], 'warnings': []}
        backend = FrozenBackend()
        queue = StoryJobQueue([backend])
        try:
            identifier = queue.submit(source)
            until(lambda: queue.snapshot(identifier)['status'] == 'failed')
            self.assertEqual(len(backend.calls), 3)
            self.assertEqual(queue.snapshot(identifier)['progress']['completed_frames'], 0)
            self.assertFalse(queue.snapshot(identifier)['result_available'])
        finally:
            queue.close()

    def test_real_finite_actions_complete_in_auto_and_reject_fixed_truncation(self):
        with np.load(Path(__file__).parent / 'tests/fixtures/scene_action_completion.npz', allow_pickle=False) as data:
            fixtures = {key: data[key].copy() for key in ('fall', 'backflip')}
        for kind, prompt, planned, endpoint in (
            ('fall', 'A person falls to the ground.', 30, 88),
            ('backflip', 'A person performs a backflip.', 38, 83),
        ):
            for automatic, explicit in ((True, False), (False, False), (True, True)):
                with self.subTest(kind=kind, automatic=automatic, explicit=explicit):
                    class CapturedBackend(FakeBackend):
                        def generate(self, request_id, motion_prompt, history):
                            result = super().generate(request_id, motion_prompt, history)
                            result['positions'] = fixtures[kind].copy()
                            return result

                    backend = CapturedBackend()
                    queue = StoryJobQueue([backend])
                    source = {'version': 1, 'title': 'Finite action',
                              'prompt': prompt.rstrip('.') + (f' for {planned / 25} seconds' if explicit else '') + ', then wave.',
                              'beats': [dict(id='beat-1', prompt=prompt, seconds=planned / 25),
                                        dict(id='beat-2', prompt='A person waves.', seconds=1)], 'warnings': []}
                    try:
                        identifier = queue.submit(source, automatic=automatic)
                        until(lambda: queue.snapshot(identifier)['status'] in ('completed', 'failed'))
                        snapshot = queue.snapshot(identifier)
                        if automatic and not explicit:
                            self.assertEqual(snapshot['status'], 'completed', snapshot['error'])
                            take = queue.result(identifier)
                            self.assertEqual(take.segments[0]['end'], endpoint)
                            self.assertEqual(len(take.positions), endpoint + 25)
                            np.testing.assert_array_equal(take.positions[:endpoint], fixtures[kind][:endpoint])
                            self.assertEqual(snapshot['progress']['completed_chunks'], snapshot['progress']['total_chunks'])
                        else:
                            self.assertEqual(snapshot['status'], 'failed')
                            self.assertIn('did not complete', snapshot['error'])
                            self.assertEqual(len(backend.calls), 5 if kind == 'backflip' else 3)
                            self.assertFalse(snapshot['result_available'])
                    finally:
                        queue.close()

    def test_auto_travel_accumulates_distance_across_chunks(self):
        from test_story_action_execution import standing

        class TravelBackend(FakeBackend):
            def generate(self, request_id, prompt, history):
                result = super().generate(request_id, prompt, history)
                result['positions'] = standing(104)
                result['positions'][:, :, 2] += np.linspace(
                    (len(self.calls) - 1) * 10, len(self.calls) * 10, 104)[:, None]
                return result

        source = {'version': 1, 'title': 'Twenty meters', 'prompt': 'Run 20 meters, then wave.',
                  'beats': [dict(id='beat-1', prompt='A person runs 20 meters.', seconds=1),
                            dict(id='beat-2', prompt='A person waves.', seconds=1)], 'warnings': []}
        backend = TravelBackend()
        queue = StoryJobQueue([backend])
        try:
            identifier = queue.submit(source, automatic=True)
            until(lambda: queue.snapshot(identifier)['status'] in ('completed', 'failed'))
            self.assertEqual(queue.snapshot(identifier)['status'], 'completed')
            take = queue.result(identifier)
            self.assertEqual(take.segments[0]['end'], 208)
            self.assertEqual(len(take.positions), 233)
            self.assertEqual(len(backend.calls), 3)
            np.testing.assert_array_equal(backend.calls[1][2], np.ones((52, 414)))
        finally:
            queue.close()

    def test_fixed_semantic_retry_uses_unchanged_history_and_discards_rejected_motion(self):
        from test_story_action_execution import standing

        class SemanticBackend(FakeBackend):
            def generate(self, request_id, prompt, history):
                output = super().generate(request_id, prompt, history)
                output['positions'] = standing(104)
                if len(self.calls) >= 3:
                    output['positions'][:, ROOT, 1] = .2
                    output['positions'][:, SHOULDERS, 1] = .2
                    output['positions'][:, SHOULDERS, 2] = .5
                return output

        source = {'version': 1, 'title': 'Retry action', 'prompt': 'Wave, then fall, then wave.',
                  'beats': [dict(id='beat-1', prompt='A person waves.', seconds=1),
                            dict(id='beat-2', prompt='A person falls.', seconds=.8),
                            dict(id='beat-3', prompt='A person waves.', seconds=1)], 'warnings': []}
        backend = SemanticBackend()
        queue = StoryJobQueue([backend])
        try:
            identifier = queue.submit(source)
            until(lambda: queue.snapshot(identifier)['status'] in ('completed', 'failed'))
            self.assertEqual(queue.snapshot(identifier)['status'], 'completed')
            self.assertEqual(len(backend.calls), 4)
            self.assertEqual(len({call[0] for call in backend.calls}), 4)
            np.testing.assert_array_equal(backend.calls[1][2], backend.calls[2][2])
            take = queue.result(identifier)
            self.assertEqual(len(take.motion), 70)
            np.testing.assert_array_equal(take.motion[25:45], np.full((20, 414), 3))
            self.assertEqual(queue.snapshot(identifier)['progress']['completed_chunks'], 3)
        finally:
            queue.close()

    def test_unfinished_finite_action_stops_at_30_seconds_before_suffix(self):
        from test_story_action_execution import standing

        class NeverFallingBackend(FakeBackend):
            def generate(self, request_id, prompt, history):
                result = super().generate(request_id, prompt, history)
                result['positions'] = standing(104)
                return result

        backend = NeverFallingBackend()
        queue = StoryJobQueue([backend])
        source = {'version': 1, 'title': 'Bounded fall', 'prompt': 'Fall, then wave.',
                  'beats': [dict(id='beat-1', prompt='A person falls.', seconds=1),
                            dict(id='beat-2', prompt='A person waves.', seconds=1)], 'warnings': []}
        try:
            identifier = queue.submit(source, automatic=True)
            until(lambda: queue.snapshot(identifier)['status'] == 'failed')
            snapshot = queue.snapshot(identifier)
            self.assertEqual(snapshot['progress']['completed_frames'], 750)
            self.assertEqual(snapshot['progress']['total_frames'], 775)
            self.assertEqual(len(backend.calls), 24)
            self.assertEqual(snapshot['attempt'], 3)
            self.assertTrue(all('falls' in call[1] for call in backend.calls))
            self.assertFalse(snapshot['result_available'])
        finally:
            queue.close()

    @staticmethod
    def recovery_plan(seconds=2):
        return {'version': 1, 'title': 'Get up and dance', 'prompt': 'Get up and dance.',
                'beats': [dict(id='beat-1', prompt='A person gets up from the floor.', seconds=seconds),
                          dict(id='beat-2', prompt='A person dances.', seconds=1)], 'warnings': []}

    @staticmethod
    def standing_positions():
        p = np.zeros((104, 34, 3), dtype=np.float32)
        p[:, ROOT, 1] = .9
        p[:, SHOULDERS, 1] = 1.3
        for side in ('left', 'right'):
            p[:, JOINT_INDEX[f'{side}_hip_yaw_skel'], 1] = .9
            p[:, JOINT_INDEX[f'{side}_knee_skel'], 1] = .45
        return p

    def test_upright_proxy_rejects_prone_and_crouched_endings_and_scales(self):
        upright = self.standing_positions()
        self.assertTrue(_recovered_upright([upright]))
        self.assertTrue(_recovered_upright([upright * 2 + [5, 3, -2]]))
        prone = upright.copy()
        prone[:, :, [1, 2]] = prone[:, :, [2, 1]]
        self.assertFalse(_recovered_upright([prone]))
        crouched = upright.copy()
        crouched[:, ROOT, 1] = .4
        crouched[:, SHOULDERS, 1] = .8
        self.assertFalse(_recovered_upright([crouched]))
        # One upright endpoint is insufficient; the ending must be sustained.
        prone[-1] = upright[-1]
        self.assertFalse(_recovered_upright([prone]))
        self.assertFalse(_recovered_upright([np.zeros_like(upright)]))

    def recovery_backend(self, upright_on_call):
        positions = self.standing_positions()

        class RecoveringBackend(FakeBackend):
            def generate(self, request_id, prompt, history):
                result = super().generate(request_id, prompt, history)
                if len(self.calls) >= upright_on_call and 'dances energetically' not in prompt:
                    result['positions'] = positions.copy()
                return result

        return RecoveringBackend()

    def test_auto_continues_recovery_with_history_before_dancing(self):
        backend = self.recovery_backend(upright_on_call=2)
        queue = StoryJobQueue([backend])
        try:
            identifier = queue.submit(self.recovery_plan(), automatic=True)
            until(lambda: queue.snapshot(identifier)['status'] == 'completed')
            self.assertEqual([call[1] for call in backend.calls],
                             [FLOOR_RECOVERY_PROMPT] * 2 + ['A person dances energetically with rhythmic footwork and swinging arms.'])
            np.testing.assert_array_equal(backend.calls[1][2], np.ones((48, 414)))
            take = queue.result(identifier)
            self.assertEqual(len(take.motion), 125)
            self.assertEqual(take.segments[0]['end'], 100)
            self.assertEqual(take.segments[0]['recovery_adjustment_frames'], 50)
            self.assertEqual(take.segments[1]['start'], 100)
            snapshot = queue.snapshot(identifier)
            self.assertEqual(snapshot['progress']['total_frames'], 125)
            self.assertEqual(snapshot['progress']['total_chunks'], 3)
            self.assertEqual(snapshot['progress']['fraction'], 1)
        finally:
            queue.close()

    def test_recovery_never_advances_prone_or_exposes_a_partial_take(self):
        for automatic, calls in ((False, 3), (True, 9)):
            backend = self.recovery_backend(upright_on_call=100)
            queue = StoryJobQueue([backend])
            try:
                identifier = queue.submit(self.recovery_plan(), automatic=automatic)
                until(lambda: queue.snapshot(identifier)['status'] == 'failed')
                self.assertEqual(len(backend.calls), calls)
                self.assertTrue(all(call[1] == FLOOR_RECOVERY_PROMPT for call in backend.calls))
                self.assertIn('did not finish getting upright', queue.snapshot(identifier)['error'])
                self.assertFalse(queue.snapshot(identifier)['result_available'])
            finally:
                queue.close()

    def test_recovery_respects_prose_timing_and_scene_budget(self):
        for prose in (True, False):
            backend = self.recovery_backend(upright_on_call=100)
            queue = StoryJobQueue([backend])
            source = self.recovery_plan()
            if prose:
                source['prompt'] = 'Get up in two seconds and dance.'
            else:
                source['beats'][1]['seconds'] = 30
                for i, seconds in enumerate((30, 30, 28), 3):
                    source['beats'].append(dict(id=f'beat-{i}', prompt='A person dances.', seconds=seconds))
            try:
                identifier = queue.submit(source, automatic=True)
                until(lambda: queue.snapshot(identifier)['status'] == 'failed')
                self.assertEqual(len(backend.calls), 3)
                self.assertEqual(queue.snapshot(identifier)['attempt'], 1 if prose else 3)
                self.assertLessEqual(queue.snapshot(identifier)['progress']['total_frames'], 3000)
            finally:
                queue.close()

    def test_unrelated_run_timing_does_not_disable_auto_recovery(self):
        backend = self.recovery_backend(upright_on_call=3)
        queue = StoryJobQueue([backend])
        source = self.recovery_plan()
        source['prompt'] = 'Run for five seconds, then get up and dance.'
        source['beats'].insert(0, dict(id='beat-1', prompt='A person runs.', seconds=1))
        source['beats'][1]['id'] = 'beat-2'
        source['beats'][2]['id'] = 'beat-3'
        try:
            identifier = queue.submit(source, automatic=True)
            until(lambda: queue.snapshot(identifier)['status'] == 'completed')
            take = queue.result(identifier)
            self.assertEqual([segment['end'] - segment['start'] for segment in take.segments],
                             [25, 100, 25])
            self.assertEqual(len(backend.calls), 4)
            self.assertEqual(queue.snapshot(identifier)['actual_seconds'], 6)
        finally:
            queue.close()

    def test_cancel_during_recovery_extension_stops_later_actions(self):
        class BlockingRecovery(FakeBackend):
            def generate(self, request_id, prompt, history):
                if len(self.calls) == 1:
                    self.started.clear()
                    self.release.clear()
                return super().generate(request_id, prompt, history)

        backend = BlockingRecovery()
        queue = StoryJobQueue([backend])
        try:
            identifier = queue.submit(self.recovery_plan(), automatic=True)
            until(lambda: len(backend.calls) == 2)
            self.assertTrue(queue.cancel(identifier))
            backend.release.set()
            until(lambda: queue._jobs[identifier].backend is None)
            self.assertEqual(len(backend.calls), 2)
            self.assertEqual(queue.snapshot(identifier)['status'], 'cancelled')
            self.assertFalse(queue.snapshot(identifier)['result_available'])
        finally:
            backend.release.set()
            queue.close()

    def test_auto_finishes_recovery_once_upright_instead_of_repeating_get_up(self):
        for automatic, frames, calls in ((True, 50, 2), (False, 150, 3)):
            backend = self.recovery_backend(upright_on_call=1)
            queue = StoryJobQueue([backend])
            try:
                identifier = queue.submit(self.recovery_plan(seconds=6), automatic=automatic)
                until(lambda: queue.snapshot(identifier)['status'] == 'completed')
                take = queue.result(identifier)
                self.assertEqual(take.segments[0]['end'], frames)
                self.assertEqual(len(backend.calls), calls)
                self.assertEqual(queue.snapshot(identifier)['progress']['total_chunks'], calls)
                self.assertEqual(queue.snapshot(identifier)['actual_seconds'], (frames + 25) / 25)
                self.assertEqual(queue.snapshot(identifier)['progress']['fraction'], 1)
            finally:
                queue.close()

    def test_recovery_helpers_preserve_seated_or_negative_intent(self):
        seated = 'A person stands up from a chair.'
        self.assertEqual(recovery_prompt(seated, context='Fall then stand up from a chair.'), seated)
        negative = 'A person does not get up from the floor.'
        self.assertEqual(recovery_prompt(negative), negative)
        self.assertEqual(recovery_completion_frame(self.standing_positions()), 50)
        positions = self.standing_positions()
        positions[:60] = 0
        self.assertEqual(recovery_completion_frame(positions), 70)

    def test_auto_recovery_cannot_exceed_30_seconds(self):
        backend = self.recovery_backend(upright_on_call=100)
        queue = StoryJobQueue([backend])
        try:
            identifier = queue.submit(self.recovery_plan(seconds=29), automatic=True)
            until(lambda: queue.snapshot(identifier)['status'] == 'failed')
            self.assertEqual(queue.snapshot(identifier)['progress']['completed_frames'], 750)
            self.assertEqual(queue.snapshot(identifier)['progress']['total_frames'], 775)
            self.assertEqual(len(backend.calls), 24)
            self.assertEqual(queue.snapshot(identifier)['attempt'], 3)
        finally:
            queue.close()

    def test_quality_rejection_retries_fresh_ids_with_multiple_reasons(self):
        class QualityBackend(FakeBackend):
            def generate(self, request_id, prompt, history):
                if len(self.calls) < 2:
                    self.calls.append((request_id, prompt, history))
                    raise RuntimeError('RuntimeError: Generated candidates failed motion-quality checks '
                                       '(horizon_seam, intra_clip_jump); no motion committed. Retry the instruction.')
                return super().generate(request_id, prompt, history)

        quality = QualityBackend()
        queue = StoryJobQueue([quality])
        try:
            identifier = queue.submit(plan())
            until(lambda: queue.snapshot(identifier)['status'] == 'completed')
            self.assertEqual(len({call[0] for call in quality.calls}), len(quality.calls))
            self.assertEqual([call[1] for call in quality.calls[:3]], ['Walk forward.'] * 3)
        finally:
            queue.close()

    def test_quality_retry_recognizer_rejects_unknown_or_unstructured_errors(self):
        prefix = 'Generated candidates failed motion-quality checks'
        self.assertEqual(quality_failure_reasons(prefix), ())
        self.assertEqual(quality_failure_reasons(prefix + ' (horizon_seam, intra_clip_jump)'),
                         ('horizon_seam', 'intra_clip_jump'))
        for suffix in (' (unknown)', ' (intra_clip_jump, auth)', ' (intra_clip_jump,)'):
            self.assertIsNone(quality_failure_reasons(prefix + suffix))
        self.assertIsNone(quality_failure_reasons('Backend HTTP 401'))
        self.assertIsNone(quality_failure_reasons('Extra text: ' + prefix))
        self.assertEqual(quality_failure_reasons(
            'RuntimeError: Motion quality rejected 3 attempts: floor penetration, '
            'motion discontinuity. No motion committed.'),
            ('floor_penetration', 'intra_clip_jump'))
        for message in ('Motion quality rejected 3 attempts: auth. No motion committed.',
                        'Motion quality rejected 99 attempts: invalid motion. No motion committed.',
                        'Extra text: Motion quality rejected 1 attempt: invalid motion. No motion committed.'):
            self.assertIsNone(quality_failure_reasons(message))

        class AuthBackend(FakeBackend):
            def generate(self, request_id, prompt, history):
                self.calls.append((request_id, prompt, history))
                raise RuntimeError('Backend HTTP 401')

        auth = AuthBackend()
        queue = StoryJobQueue([auth])
        try:
            identifier = queue.submit(plan())
            until(lambda: queue.snapshot(identifier)['status'] == 'failed')
            self.assertEqual(len(auth.calls), 1)
        finally:
            queue.close()

    def test_current_backend_rejection_retries_same_accepted_history(self):
        class CurrentBackend(FakeBackend):
            def generate(self, request_id, prompt, history):
                if 1 <= len(self.calls) <= 2:
                    self.calls.append((request_id, prompt, history.copy()))
                    raise RuntimeError('Motion quality rejected 3 attempts: motion discontinuity. No motion committed.')
                return super().generate(request_id, prompt, history)

        backend = CurrentBackend()
        queue = StoryJobQueue([backend])
        try:
            identifier = queue.submit(plan())
            until(lambda: queue.snapshot(identifier)['status'] == 'completed')
            self.assertEqual(len(backend.calls), 4)
            self.assertEqual(len({call[0] for call in backend.calls}), 4)
            for call in backend.calls[1:]:
                np.testing.assert_array_equal(call[2], np.ones((52, 414)))
        finally:
            queue.close()

    def test_scene_segments_persist_explicit_recovery_timing(self):
        for automatic, prompt, mode, length in (
            (True, 'Get up in six seconds and dance.', 'fixed', 150),
            (False, 'Get up and dance.', 'fixed', 150),
            (True, 'Get up and dance.', 'auto', 50),
        ):
            with self.subTest(automatic=automatic, prompt=prompt):
                backend = self.recovery_backend(upright_on_call=1)
                queue = StoryJobQueue([backend])
                source = self.recovery_plan(seconds=6)
                source['prompt'] = prompt
                try:
                    identifier = queue.submit(source, automatic=automatic)
                    until(lambda: queue.snapshot(identifier)['status'] == 'completed')
                    segment = queue.result(identifier).segments[0]
                    self.assertEqual(segment['timing_mode'], mode)
                    self.assertEqual(segment['end'], length)
                finally:
                    queue.close()

    def test_cancel_during_quality_retry_stops_further_backend_calls(self):
        class RetryBackend(FakeBackend):
            def generate(self, request_id, prompt, history):
                self.calls.append((request_id, prompt, history))
                self.started.set()
                self.release.wait(3)
                raise RuntimeError('RuntimeError: Generated candidates failed motion-quality checks '
                                   '(intra_clip_jump); no motion committed. Retry the instruction.')

        backend = RetryBackend()
        backend.release.clear()
        queue = StoryJobQueue([backend])
        try:
            identifier = queue.submit(plan())
            self.assertTrue(backend.started.wait(1))
            self.assertTrue(queue.cancel(identifier))
            backend.release.set()
            until(lambda: queue.snapshot(identifier)['status'] == 'cancelled')
            self.assertEqual(len(backend.calls), 1)
            self.assertFalse(queue.snapshot(identifier)['result_available'])
        finally:
            queue.close()

    def test_exhausted_quality_retries_explain_failure(self):
        class RejectingBackend(FakeBackend):
            def generate(self, request_id, prompt, history):
                self.calls.append((request_id, prompt, history))
                raise RuntimeError('Generated candidates failed motion-quality checks '
                                   '(intra_clip_jump); no motion committed. Retry the instruction.')

        backend = RejectingBackend()
        queue = StoryJobQueue([backend])
        try:
            identifier = queue.submit(plan())
            until(lambda: queue.snapshot(identifier)['status'] == 'failed')
            self.assertEqual(len(backend.calls), 3)
            self.assertIn('motion discontinuity', queue.snapshot(identifier)['error'])
            self.assertIn('after 3 attempts', queue.snapshot(identifier)['error'])
        finally:
            queue.close()

    def test_backflip_quality_retries_use_five_attempt_budget_per_scene(self):
        class RejectingBackend(FakeBackend):
            def generate(self, request_id, prompt, history):
                self.calls.append((request_id, prompt, history))
                raise RuntimeError('Motion quality rejected 3 attempts: motion discontinuity. No motion committed.')

        backend = RejectingBackend()
        queue = StoryJobQueue([backend])
        source = {'version': 1, 'title': 'Flip', 'prompt': 'Do a backflip.',
                  'beats': [dict(id='beat-1', prompt='A person performs a backflip.', seconds=3)],
                  'warnings': []}
        try:
            identifier = queue.submit(source, automatic=True)
            until(lambda: queue.snapshot(identifier)['status'] == 'failed')
            self.assertEqual(len(backend.calls), 15)
            self.assertEqual(len({call[0] for call in backend.calls}), 15)
            self.assertEqual(queue.snapshot(identifier)['attempt'], 3)
            self.assertIn('after 5 attempts', queue.snapshot(identifier)['error'])
            self.assertEqual(queue.snapshot(identifier)['progress']['completed_frames'], 0)
        finally:
            queue.close()

    def test_exact_minute_with_varied_travel_actions_has_continuous_history(self):
        backend = FakeBackend()
        queue = StoryJobQueue([backend])
        try:
            source = {
                'version': 1, 'title': 'Long plaza walk',
                'prompt': 'Walk across the plaza, then walk back, then wave several times.',
                'beats': [
                    {'id': 'beat-1', 'prompt': 'Walk across the plaza.', 'seconds': 25},
                    {'id': 'beat-2', 'prompt': 'Walk back to the entrance.', 'seconds': 29},
                    {'id': 'beat-3', 'prompt': 'Wave several times.', 'seconds': 6},
                ], 'warnings': [],
            }
            identifier = queue.submit(fit_story_duration(source, 60))
            until(lambda: queue.snapshot(identifier)['status'] == 'completed')
            take = queue.result(identifier)
            self.assertEqual(len(take.motion), 1500)
            self.assertEqual(list(dict.fromkeys(item['prompt'] for item in take.segments)),
                             ['Walk across the plaza.', 'Walk back to the entrance.',
                              'Wave several times.'])
            self.assertEqual([item['end'] - item['start'] for item in take.segments],
                             [625, 725, 150])
            self.assertGreater(len(backend.calls), len(source['beats']))
            self.assertEqual(take.segments[-1]['end'], 1500)
            self.assertEqual(queue.snapshot(identifier)['progress']['fraction'], 1)
            self.assertIsNone(backend.calls[0][2])
            self.assertTrue(all(call[2] is not None for call in backend.calls[1:]))
            self.assertTrue(all(len(call[2]) <= 52 and len(call[2]) % 4 == 0
                                for call in backend.calls[1:]))
            self.assertTrue(queue.release(identifier))
            with self.assertRaises(RuntimeError):
                queue.result(identifier)
        finally:
            queue.close()

    def test_cancelled_active_job_never_exposes_partial_take(self):
        backend = FakeBackend()
        backend.release.clear()
        queue = StoryJobQueue([backend])
        try:
            first = queue.submit(plan())
            self.assertTrue(backend.started.wait(1))
            second = queue.submit(plan())
            self.assertEqual(queue.snapshot(second)['status'], 'queued')
            self.assertTrue(queue.cancel(first))
            backend.release.set()
            until(lambda: queue.snapshot(second)['status'] == 'completed')
            self.assertEqual(queue.snapshot(first)['status'], 'cancelled')
            self.assertFalse(queue.snapshot(first)['result_available'])
            self.assertTrue(backend.cancelled)
            with self.assertRaises(RuntimeError):
                queue.result(first)
        finally:
            queue.close()


if __name__ == '__main__':
    unittest.main()
