"""Atomic cast publication, late cancellation, exact transport and capture."""
import threading
import unittest

import numpy as np

from cast_performance_session import CastPerformanceSession
from paired_scene import EMPTY_SCENE
from test_cast_performance import performance


class CastPerformanceSessionTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.session = CastPerformanceSession(clock=lambda: self.now)
        self.addCleanup(self.session.close)

    def finish(self):
        self.session._thread.join(2.)
        self.assertFalse(self.session._thread.is_alive())

    def test_default_is_inactive_empty_and_complete_build_publishes_plan_names(self):
        self.assertFalse(self.session.snapshot()['active'])
        self.assertEqual(self.session.snapshot()['cast'], [])
        clip = performance(1)
        scene = dict(EMPTY_SCENE, name='Requested stage')
        self.session.build_performance(lambda document, **kwargs: clip, scene, request={'prompt': 'Solo scene'})
        self.finish()
        state = self.session.snapshot()
        self.assertTrue(state['active'])
        self.assertTrue(state['playing'])
        self.assertFalse(state['busy'])
        self.assertIs(self.session.timeline_clip(), clip)
        self.assertEqual(state['actor_ids'], ['performer_1'])
        self.assertEqual(state['selected_pair'], [])
        self.assertEqual(state['cast'][0]['name'], 'Performer 1')
        self.assertEqual(self.session.scene_document, scene)

    def test_failure_preserves_previous_take_scene_and_cast_then_retry_is_atomic(self):
        original = performance(1)
        self.session.load_performance(original)
        original_scene = self.session.scene_document
        replacement = performance(3, offset=8.)
        entered, release = threading.Event(), threading.Event()
        calls = []
        def builder(scene, *, cancelled, on_progress):
            calls.append(scene)
            on_progress({'stage': 'building tracks'})
            entered.set()
            release.wait(2.)
            if len(calls) == 1:
                raise RuntimeError('service token-secret unavailable')
            scene['name'] = 'Builder mutated its isolated copy'
            return replacement
        builder.token = 'token-secret'
        desired_scene = dict(EMPTY_SCENE, name='New stage')
        self.session.build_performance(builder, desired_scene, request={'prompt': 'Three actors'})
        self.assertTrue(entered.wait(1.))
        self.assertIs(self.session.timeline_clip(), original)
        self.assertEqual(self.session.snapshot()['progress']['stage'], 'building tracks')
        self.assertEqual(len(self.session.snapshot()['cast']), 1)
        release.set()
        self.finish()
        self.assertIs(self.session.timeline_clip(), original)
        self.assertEqual(self.session.scene_document, original_scene)
        self.assertNotIn('token-secret', self.session.snapshot()['failure'])
        self.session.retry()
        self.finish()
        self.assertIs(self.session.timeline_clip(), replacement)
        self.assertEqual(self.session.scene_document, desired_scene)
        self.assertEqual(len(self.session.snapshot()['cast']), 3)

    def test_cancelled_late_result_and_progress_cannot_replace_loaded_take(self):
        original = performance(2)
        replacement = performance(3)
        self.session.load_performance(original)
        entered, release = threading.Event(), threading.Event()
        cancelled_result = []
        def builder(scene, *, cancelled, on_progress):
            entered.set()
            release.wait(2.)
            cancelled_result.append(cancelled())
            on_progress({'stage': 'late'})
            return replacement
        self.session.build_performance(builder)
        self.assertTrue(entered.wait(1.))
        self.session.cancel()
        with self.assertRaisesRegex(RuntimeError, 'running or cancelling'):
            self.session.build_performance(builder)
        release.set()
        self.finish()
        self.assertEqual(cancelled_result, [True])
        self.assertIs(self.session.timeline_clip(), original)
        self.assertIsNone(self.session.snapshot()['progress'])
        self.assertIsNone(self.session.snapshot()['failure'])

    def test_bad_builder_result_does_not_publish_any_partial_track_set(self):
        original = performance()
        self.session.load_performance(original)
        self.session.build_performance(lambda scene, **kwargs: {'joints': original.joints})
        self.finish()
        self.assertIs(self.session.timeline_clip(), original)
        self.assertIn('complete CastPerformance', self.session.snapshot()['failure'])

    def test_transport_revisions_change_for_commands_not_natural_ticks(self):
        self.session.load_performance(performance(frames=90))
        state = self.session.play()
        revision = state['transport_revision']
        self.now += 1.
        state = self.session.tick()
        self.assertEqual(state['frame'], 30)
        self.assertEqual(state['transport_revision'], revision)
        for operation in (lambda: self.session.seek(40), self.session.pause, self.session.play, self.session.restart):
            state = operation()
            self.assertGreater(state['transport_revision'], revision)
            revision = state['transport_revision']
        self.now += 10.
        state = self.session.tick()
        self.assertEqual(state['frame'], 89)
        self.assertFalse(state['playing'])
        self.assertEqual(state['transport_revision'], revision)

    def test_capture_lease_freezes_time_and_rejects_take_and_scene_mutation(self):
        clip = performance()
        self.session.load_performance(clip)
        self.session.seek(3)
        state = self.session.begin_capture()
        self.assertTrue(state['capturing'])
        for mutation in (self.session.play, self.session.pause, self.session.deactivate,
                lambda: self.session.seek(1), lambda: self.session.load_performance(clip),
                lambda: self.session.update_scene(EMPTY_SCENE),
                lambda: self.session.build_performance(lambda scene, **kwargs: clip)):
            with self.assertRaisesRegex(RuntimeError, 'export is running'):
                mutation()
        revision = state['transport_revision']
        state = self.session.capture_seek(6)
        self.assertGreater(state['transport_revision'], revision)
        self.now += 100.
        self.assertEqual(self.session.tick()['frame'], 6)
        self.assertEqual(self.session.end_capture(3)['frame'], 3)
        self.assertFalse(self.session.snapshot()['capturing'])

    def test_session_save_load_restores_exact_tracks_cast_scene_and_playhead(self):
        clip = performance(3)
        scene = dict(EMPTY_SCENE, name='Portable scene')
        self.session.load_performance(clip, scene)
        self.session.seek(6)
        other = CastPerformanceSession()
        self.addCleanup(other.close)
        state = other.load(self.session.save())
        self.assertEqual(state['frame'], 6)
        self.assertFalse(state['playing'])
        self.assertEqual(state['cast'], self.session.snapshot()['cast'])
        np.testing.assert_array_equal(other.timeline_clip().joints, clip.joints)
        self.assertEqual(other.scene_document, scene)
        before = other.timeline_clip()
        with self.assertRaises(ValueError):
            other.load(b'bad archive')
        self.assertIs(other.timeline_clip(), before)

    def test_capture_frame_access_covers_every_track_for_each_cast_size(self):
        for count in (1, 2, 3):
            with self.subTest(count=count):
                clip = performance(count)
                self.session.load_performance(clip)
                self.session.begin_capture()
                for frame in range(clip.frames):
                    state = self.session.capture_seek(frame)
                    self.assertEqual(state['actor_ids'], list(clip.actor_ids))
                    np.testing.assert_array_equal(self.session.frame_pose(), clip.joints[frame])
                    self.assertEqual(self.session.frame_pose().shape, (count, 22, 3))
                self.session.end_capture()


if __name__ == '__main__':
    unittest.main()
