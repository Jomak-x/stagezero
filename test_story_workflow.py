"""Offline integration checks for publishing generated scenes into the editor."""

import copy
import threading
import unittest
from unittest.mock import Mock

import numpy as np

from directing import DirectorSession
from story_jobs import StoryJobQueue
from story_workflow import SerializedBackend, StoryWorkflow
from takes import Take, decode_project, encode_project
from test_story_jobs import FakeBackend, plan, until


class Planner:
    def __init__(self):
        self.calls = []

    def plan(self, prompt, context=None, seconds=None):
        self.calls.append((prompt, copy.deepcopy(context), seconds))
        result = plan(seconds)
        result['prompt'] = prompt
        return result


class BlockingPlanner(Planner):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def plan(self, prompt, context=None, seconds=None):
        self.started.set()
        self.release.wait(3)
        return super().plan(prompt, context, seconds)


def session():
    recorded = np.zeros((20, 34, 3), dtype=np.float32)
    rotations = np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1))
    result = DirectorSession(Mock(), recorded, rotations)
    result.set_mode('Live ARDY')
    return result


class StoryWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.session = session()
        self.backend = FakeBackend()
        self.queue = StoryJobQueue([self.backend])
        self.planner = Planner()
        self.workflow = StoryWorkflow(self.session, self.planner, self.queue)

    def tearDown(self):
        self.workflow.close()

    def completed(self, seconds=8.32):
        identifier = self.workflow.submit('  Walk, then wave.  ', seconds=seconds)
        until(lambda: self.workflow.snapshot(identifier)['status'] == 'completed')
        return identifier

    def test_load_adds_editable_take_and_round_trips_archive(self):
        identifier = self.completed(60)
        self.assertEqual(self.planner.calls[0][0], 'Walk, then wave.')
        self.assertEqual(self.planner.calls[0][2], 60)
        self.assertTrue(self.workflow.load(identifier))
        take = self.session.takes[self.session.active_take]
        self.assertEqual(len(take.motion), 1500)
        self.assertEqual(list(dict.fromkeys(segment['prompt'] for segment in take.segments)),
                         ['Walk forward.', 'Wave a hand.'])
        self.assertEqual(self.workflow.snapshot(identifier)['take_id'], take.id)
        self.assertTrue(self.workflow.load(identifier))
        self.assertEqual(len(self.session.takes), 1)
        archive = encode_project(self.session.takes, take.id, 0, self.session.scene)
        decoded, active, _, _ = decode_project(archive)
        self.assertEqual(decoded[active].segments, take.segments)

    def test_manual_load_accepts_second_result_after_first_changes_revision(self):
        first = self.completed()
        second = self.completed()
        self.assertTrue(self.workflow.load(first, automatic=True))
        self.assertFalse(self.workflow.load(second, automatic=True))
        self.assertTrue(self.workflow.load(second))
        self.assertEqual(len(self.session.takes), 2)

    def test_changed_scene_or_replaced_project_rejects_result(self):
        first = self.completed()
        self.session.scene['gate']['position'][0] = 2
        self.assertFalse(self.workflow.load(first, automatic=True))
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.workflow.load(first)
        second = self.completed()
        self.session.scene = copy.deepcopy(self.session.scene)
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.workflow.load(second)
        self.assertFalse(self.session.takes)

    def test_cancel_during_planning_ignores_late_response(self):
        blocking = BlockingPlanner()
        workflow = StoryWorkflow(self.session, blocking, self.queue)
        try:
            identifier = workflow.submit('Walk, then wave.')
            self.assertTrue(blocking.started.wait(1))
            self.assertTrue(workflow.cancel(identifier))
            blocking.release.set()
            until(lambda: workflow.snapshot(identifier)['status'] == 'cancelled')
            self.assertIsNone(workflow.snapshot(identifier)['plan'])
            self.assertFalse(self.session.takes)
        finally:
            workflow.close()

    def test_serialized_backend_cancellation_prevents_waiting_call(self):
        backend = FakeBackend()
        backend.release.clear()
        wrapper = SerializedBackend(backend)
        results = []

        def call(identifier):
            try:
                wrapper.generate(identifier, 'Walk.', None)
            except RuntimeError:
                results.append(identifier)

        first = threading.Thread(target=call, args=('first',))
        second = threading.Thread(target=call, args=('second',))
        first.start()
        self.assertTrue(backend.started.wait(1))
        second.start()
        wrapper.cancel('second')
        backend.release.set()
        first.join(1)
        second.join(1)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(results, ['second'])


if __name__ == '__main__':
    unittest.main()
