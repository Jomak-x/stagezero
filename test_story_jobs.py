"""Offline queue checks for complete scene generation and cancellation."""

import threading
import time
import unittest

import numpy as np

from story_jobs import StoryJobQueue
from story_planning import fit_story_duration


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
        rotations = np.tile(np.eye(3, dtype=np.float32), (104, 34, 1, 1))
        motion = np.full((104, 414), number, dtype=np.float32)
        return {'positions': positions, 'rotations': rotations, 'motion': motion,
                'metadata': {'request_id': request_id, 'model': 'ARDY-G1-RP-25FPS-Horizon52',
                             'fps': 25, 'generation_seconds': .01}}

    def cancel(self, request_id):
        self.cancelled.append(request_id)


class StoryQueueTests(unittest.TestCase):
    def test_quality_rejection_retries_fresh_ids_but_transport_error_does_not(self):
        class QualityBackend(FakeBackend):
            def generate(self, request_id, prompt, history):
                if len(self.calls) < 2:
                    self.calls.append((request_id, prompt, history))
                    raise RuntimeError('RuntimeError: Generated candidates failed motion-quality checks '
                                       '(intra_clip_jump); no motion committed. Retry the instruction.')
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

    def test_exact_minute_has_editable_actions_and_continuous_history(self):
        backend = FakeBackend()
        queue = StoryJobQueue([backend])
        try:
            identifier = queue.submit(plan(60))
            until(lambda: queue.snapshot(identifier)['status'] == 'completed')
            take = queue.result(identifier)
            self.assertEqual(len(take.motion), 1500)
            self.assertEqual(list(dict.fromkeys(item['prompt'] for item in take.segments)),
                             ['Walk forward.', 'Wave a hand.'])
            self.assertTrue(all(item['end'] - item['start'] <= 150 for item in take.segments))
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
