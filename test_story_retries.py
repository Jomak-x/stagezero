"""CPU regressions for private, bounded whole-scene Auto retries."""
from pathlib import Path
import threading
import unittest

import numpy as np

from story_jobs import StoryJobQueue
from test_story_jobs import FakeBackend, plan, until


QUALITY_ERROR = 'Motion quality rejected 3 attempts: motion discontinuity. No motion committed.'


class RetryBoundaryQueue(StoryJobQueue):
    def __init__(self, backend, *, block_restart=False):
        self.restart_snapshots = []
        self.restart_started = threading.Event()
        self.restart_release = threading.Event()
        if not block_restart:
            self.restart_release.set()
        super().__init__([backend])

    def _generate(self, job, backend):
        if job.attempt > 1:
            self.restart_snapshots.append(self.snapshot(job.id))
            self.restart_started.set()
            self.restart_release.wait(3)
        return super()._generate(job, backend)


class HistorySensitiveBackend(FakeBackend):
    """Only the first scene's accepted fall makes its later action fail."""
    def __init__(self):
        super().__init__()
        self.scene_attempt = 0
        with np.load(Path(__file__).parent / 'tests/fixtures/scene_action_completion.npz', allow_pickle=False) as data:
            self.fall = data['fall'].copy()

    def generate(self, request_id, prompt, history):
        if history is None:
            self.scene_attempt += 1
        output = super().generate(request_id, prompt, history)
        if 'falls' in prompt:
            output['positions'] = self.fall.copy()
        elif self.scene_attempt == 1:
            raise RuntimeError(QUALITY_ERROR)
        return output


def fall_and_wave():
    return {'version': 1, 'title': 'Fresh scene history', 'prompt': 'Fall, then wave.',
            'beats': [dict(id='beat-1', prompt='A person falls to the ground.', seconds=1.2),
                      dict(id='beat-2', prompt='A person waves.', seconds=1)], 'warnings': []}


class StorySceneRetryTests(unittest.TestCase):
    def test_second_scene_attempt_resets_history_counters_and_adjusted_totals(self):
        backend = HistorySensitiveBackend()
        queue = RetryBoundaryQueue(backend)
        try:
            identifier = queue.submit(fall_and_wave(), automatic=True)
            until(lambda: queue.snapshot(identifier)['status'] == 'completed')
            snapshot = queue.snapshot(identifier)
            self.assertEqual((snapshot['attempt'], snapshot['max_attempts']), (2, 3))
            self.assertEqual(len(backend.calls), 6)
            self.assertEqual(len({call[0] for call in backend.calls}), 6)
            self.assertIsNone(backend.calls[0][2])
            self.assertIsNone(backend.calls[4][2])
            for call in backend.calls[1:4]:
                np.testing.assert_array_equal(call[2], backend.calls[1][2])
            restarted = queue.restart_snapshots[0]
            self.assertEqual(restarted['status'], 'running')
            self.assertEqual(restarted['attempt'], 2)
            self.assertFalse(restarted['result_available'])
            self.assertEqual(restarted['progress'], dict(
                completed_beats=0, total_beats=2, completed_chunks=0, total_chunks=2,
                completed_frames=0, total_frames=55, current_beat=None, fraction=0))
            take = queue.result(identifier)
            self.assertEqual(len(take.motion), 113)
            np.testing.assert_array_equal(take.motion[:88], np.full((88, 414), 5))
            self.assertEqual(snapshot['progress']['completed_frames'], 113)
            self.assertEqual(snapshot['progress']['completed_chunks'], 2)
            self.assertEqual(snapshot['progress']['fraction'], 1)
        finally:
            queue.restart_release.set()
            queue.close()

    def test_cancel_at_restart_boundary_never_publishes_none_or_calls_backend_again(self):
        backend = HistorySensitiveBackend()
        queue = RetryBoundaryQueue(backend, block_restart=True)
        try:
            identifier = queue.submit(fall_and_wave(), automatic=True)
            self.assertTrue(queue.restart_started.wait(2))
            self.assertTrue(queue.cancel(identifier))
            queue.restart_release.set()
            until(lambda: queue._jobs[identifier].backend is None)
            snapshot = queue.snapshot(identifier)
            self.assertEqual(snapshot['status'], 'cancelled')
            self.assertEqual(snapshot['attempt'], 2)
            self.assertFalse(snapshot['result_available'])
            self.assertEqual(len(backend.calls), 4)
            with self.assertRaises(RuntimeError):
                queue.result(identifier)
        finally:
            queue.restart_release.set()
            queue.close()

    def test_auto_quality_failure_exhausts_exactly_three_private_scene_attempts(self):
        class RejectingBackend(FakeBackend):
            def generate(self, request_id, prompt, history):
                self.calls.append((request_id, prompt, history))
                raise RuntimeError(QUALITY_ERROR)

        backend = RejectingBackend()
        queue = StoryJobQueue([backend])
        try:
            identifier = queue.submit(plan(), automatic=True)
            until(lambda: queue.snapshot(identifier)['status'] == 'failed')
            snapshot = queue.snapshot(identifier)
            self.assertEqual((snapshot['attempt'], snapshot['max_attempts']), (3, 3))
            self.assertIn('Scene attempt 3/3', snapshot['error'])
            self.assertEqual(len(backend.calls), 9)
            self.assertEqual(len({call[0] for call in backend.calls}), 9)
            self.assertFalse(snapshot['result_available'])
        finally:
            queue.close()

    def test_fixed_or_transport_failures_do_not_restart_scene(self):
        for automatic, message, calls in ((False, QUALITY_ERROR, 3),
                                           (True, 'Backend HTTP 401', 1),
                                           (True, 'Pod connection unavailable', 1)):
            with self.subTest(automatic=automatic, message=message):
                class RejectingBackend(FakeBackend):
                    def generate(self, request_id, prompt, history):
                        self.calls.append((request_id, prompt, history))
                        raise RuntimeError(message)

                backend = RejectingBackend()
                queue = StoryJobQueue([backend])
                try:
                    identifier = queue.submit(plan(), automatic=automatic)
                    until(lambda: queue.snapshot(identifier)['status'] == 'failed')
                    self.assertEqual(queue.snapshot(identifier)['attempt'], 1)
                    self.assertEqual(len(backend.calls), calls)
                finally:
                    queue.close()

    def test_explicit_action_completion_failure_stops_without_whole_scene_retry(self):
        source = {'version': 1, 'title': 'Timed rise', 'prompt': 'Get up in two seconds.',
                  'beats': [dict(id='beat-1', prompt='A person gets up.', seconds=2)], 'warnings': []}
        backend = FakeBackend()
        queue = StoryJobQueue([backend])
        try:
            identifier = queue.submit(source, automatic=True)
            until(lambda: queue.snapshot(identifier)['status'] == 'failed')
            self.assertEqual(queue.snapshot(identifier)['attempt'], 1)
            self.assertEqual(len(backend.calls), 3)
        finally:
            queue.close()

    def test_prose_fixed_scene_does_not_restart_even_when_submitted_as_auto(self):
        class RejectingBackend(FakeBackend):
            def generate(self, request_id, prompt, history):
                self.calls.append((request_id, prompt, history))
                raise RuntimeError(QUALITY_ERROR)

        source = plan()
        source['prompt'] = 'An 8.32 second scene: walk, then wave.'
        backend = RejectingBackend()
        queue = StoryJobQueue([backend])
        try:
            identifier = queue.submit(source, automatic=True)
            until(lambda: queue.snapshot(identifier)['status'] == 'failed')
            snapshot = queue.snapshot(identifier)
            self.assertEqual((snapshot['attempt'], snapshot['max_attempts']), (1, 1))
            self.assertEqual(len(backend.calls), 3)
        finally:
            queue.close()

    def test_none_result_is_never_marked_completed(self):
        class EmptyQueue(StoryJobQueue):
            def _generate(self, job, backend):
                return None

        queue = EmptyQueue([FakeBackend()])
        try:
            identifier = queue.submit(plan(), automatic=True)
            until(lambda: queue.snapshot(identifier)['status'] == 'failed')
            self.assertEqual(queue.snapshot(identifier)['attempt'], 1)
            self.assertFalse(queue.snapshot(identifier)['result_available'])
        finally:
            queue.close()


if __name__ == '__main__':
    unittest.main()
