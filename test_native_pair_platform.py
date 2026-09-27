"""Exact native archive/state/render regressions, not animation-quality proofs."""
import io
import json
import threading
import unittest
from types import SimpleNamespace

import numpy as np

from native_pair_clip import NativePairClip, load_source
from native_pair_session import NativePairSession, REVIEWED_HANDSHAKE
from native_pair_renderer import NativePairRenderer
from test_native_pair_rig import Scene


class NativePairPlatformTests(unittest.TestCase):
    def session(self, provider=None, clock=None):
        result = NativePairSession(provider, **({'clock': clock} if clock else {}))
        self.addCleanup(result.close)
        result.load_reviewed_handshake()
        return result

    def test_original_exact_joint_and_feature_bytes_survive_project_roundtrip(self):
        session = self.session()
        third = session.add_actor('Third performer')
        session.select_pair(third, 'actor_2')
        session.seek(97)
        archive = session.save()
        restored = self.session()
        restored.load(archive)
        with np.load(REVIEWED_HANDSHAKE, allow_pickle=False) as source:
            self.assertEqual(source['joints'].tobytes(), restored.timeline_clip().joints.tobytes())
            self.assertEqual(source['features'].tobytes(), restored.timeline_clip().features.tobytes())
        self.assertEqual(restored.snapshot()['selected_pair'], [third, 'actor_2'])
        self.assertEqual(restored.snapshot()['frame'], 97)
        self.assertEqual(restored.snapshot()['fps'], 30)
        self.assertEqual(restored.snapshot()['cast'], session.snapshot()['cast'])

    def test_playback_advances_native_frames_and_stops_at_end(self):
        now = [10.]
        session = self.session(clock=lambda: now[0])
        session.play()
        now[0] = 11.
        self.assertEqual(session.tick()['frame'], 30)
        session.seek(90)
        now[0] = 11.5
        self.assertEqual(session.tick()['frame'], 105)
        now[0] = 20.
        self.assertEqual(session.tick()['frame'], 209)
        self.assertFalse(session.snapshot()['playing'])

    def test_cancelled_completion_cannot_replace_existing_clip(self):
        release, started = threading.Event(), threading.Event()
        clip = load_source(REVIEWED_HANDSHAKE)
        class Provider:
            def generate(self, **request):
                started.set()
                release.wait(2)
                return NativePairClip(clip.joints + 10, clip.features, clip.metadata)
        session = self.session(Provider())
        original = session.timeline_clip()
        session.generate('A new pair', 42)
        self.assertTrue(started.wait(1))
        session.cancel()
        release.set()
        session._thread.join(2)
        self.assertIs(session.timeline_clip(), original)
        self.assertFalse(session.snapshot()['busy'])
        self.assertIsNone(session.snapshot()['failure'])

    def test_failure_preserves_clip_then_retry_succeeds(self):
        clip = load_source(REVIEWED_HANDSHAKE)
        class Provider:
            token = 'private-token'
            attempts = 0
            def generate(self, **request):
                self.attempts += 1
                if self.attempts == 1:
                    raise RuntimeError('failed private-token')
                return clip
        session = self.session(Provider())
        original = session.timeline_clip()
        session.generate('Two people greet', 42)
        session._thread.join(2)
        self.assertIs(session.timeline_clip(), original)
        self.assertEqual(session.snapshot()['failure'], 'failed [redacted]')
        session.retry()
        session._thread.join(2)
        self.assertIs(session.timeline_clip(), clip)
        self.assertIsNone(session.snapshot()['failure'])

    def test_pair_reselection_invalidates_pending_generation(self):
        release, started = threading.Event(), threading.Event()
        clip = load_source(REVIEWED_HANDSHAKE)
        class Provider:
            def generate(self, **request):
                started.set(); release.wait(2)
                return clip
        session = self.session(Provider())
        original = session.timeline_clip()
        third = session.add_actor('Third')
        session.generate('Greeting', 42)
        self.assertTrue(started.wait(1))
        session.select_pair('actor_1', third)
        release.set(); session._thread.join(2)
        self.assertIs(session.timeline_clip(), original)
        self.assertEqual(session.snapshot()['selected_pair'], ['actor_1', third])

    def test_no_core_or_nonfinite_archive_can_replace_good_clip(self):
        session = self.session()
        original = session.timeline_clip()
        for shape, fps in (((10, 2, 27, 3), 20), ((10, 2, 22, 3), 20)):
            stream = io.BytesIO()
            np.savez_compressed(stream, joints=np.zeros(shape, np.float32), metadata=json.dumps({'fps': fps}))
            with self.assertRaises(ValueError):
                session.load_source(stream.getvalue())
            self.assertIs(session.timeline_clip(), original)
        with self.assertRaises(ValueError):
            NativePairClip(np.full((4, 2, 22, 3), np.nan, np.float32))
        with self.assertRaises(ValueError):
            original.joints.flags.writeable = True

    def test_raw_source_not_smoothed_substitute(self):
        original = load_source(REVIEWED_HANDSHAKE)
        stream = io.BytesIO()
        np.savez_compressed(stream, joints=original.joints, smoothed_joints=original.joints+20,
                            features=original.features, metadata=json.dumps({'fps': 30}))
        np.testing.assert_array_equal(load_source(stream.getvalue()).joints, original.joints)

    def test_renderer_preserves_selected_pair_and_parks_unselected_cast(self):
        session = self.session()
        third = session.add_actor('Observer')
        renderer = NativePairRenderer(SimpleNamespace(scene=Scene()))
        self.addCleanup(renderer.remove)
        renderer.sync_cast(session.snapshot())
        renderer.set_clip(session.timeline_clip())
        renderer.tick(80)
        for i, identifier in enumerate(session.snapshot()['selected_pair']):
            np.testing.assert_array_equal(renderer.actors[identifier].joint_positions,
                                          session.timeline_clip().joints[80, i])
        observer = renderer.actors[third].joint_positions.copy()
        renderer.tick(100)
        np.testing.assert_array_equal(renderer.actors[third].joint_positions, observer)
        session.select_pair('actor_1', third)
        renderer.sync_cast(session.snapshot())
        renderer.tick(80)
        np.testing.assert_array_equal(renderer.actors[third].joint_positions, session.timeline_clip().joints[80, 1])
        old = renderer.actors['actor_2'].joint_positions.copy()
        renderer.tick(100)
        np.testing.assert_array_equal(renderer.actors['actor_2'].joint_positions, old)
        self.assertGreater(np.linalg.norm(renderer.actor_root('actor_2') - renderer.actor_root()), 1.5)

    @staticmethod
    def context_fixture(base):
        return NativePairClip(base.joints, metadata=dict(base.metadata, model='ARDY Core + InterGen',
            segments=[{'label': 'Structural test composition', 'start_frame': 0,
                       'end_frame_exclusive': base.frames, 'frames': base.frames}]))

    def test_context_failure_preserves_source_and_retry_commits_complete_result(self):
        session = self.session()
        original = session.timeline_clip()
        expected_bytes = original.joints.tobytes(), original.features.tobytes()
        attempts = []
        def builder(base, scene, *, cancelled):
            attempts.append(base)
            if len(attempts) == 1:
                base.metadata['prompt'] = 'Builder mutated its own metadata'
                raise RuntimeError('Context service failed')
            return self.context_fixture(base)
        session.build_context(builder)
        session._thread.join(2)
        self.assertIs(session.timeline_clip(), original)
        self.assertNotEqual(original.metadata['prompt'], 'Builder mutated its own metadata')
        self.assertIn('Context service failed', session.snapshot()['failure'])
        session.retry()
        session._thread.join(2)
        self.assertTrue(session.snapshot()['composed'])
        self.assertIsNone(session.snapshot()['failure'])
        self.assertEqual(session.original_pair_clip().joints.tobytes(), expected_bytes[0])
        self.assertEqual(session.original_pair_clip().features.tobytes(), expected_bytes[1])
        self.assertEqual(original.joints.tobytes(), expected_bytes[0])
        with self.assertRaisesRegex(ValueError, 'already composed'):
            session.build_context(builder)

    def test_context_cancel_load_and_pair_switch_ignore_late_result(self):
        for action in ('cancel', 'load', 'pair'):
            with self.subTest(action=action):
                session = self.session()
                original = session.timeline_clip()
                third = session.add_actor('Third')
                started, release = threading.Event(), threading.Event()
                def builder(base, scene, *, cancelled):
                    started.set(); release.wait(2)
                    return self.context_fixture(base)
                session.build_context(builder)
                self.assertTrue(started.wait(1))
                self.assertEqual(session.snapshot()['pending_operation'], 'context')
                with self.assertRaises(RuntimeError):
                    session.set_placement(x=2.)
                with self.assertRaises(RuntimeError):
                    session.set_hand_pose('fists')
                with self.assertRaises(RuntimeError):
                    session.generate('Other job', 42)
                if action == 'cancel':
                    session.cancel()
                elif action == 'load':
                    session.load_reviewed_handshake()
                else:
                    session.select_pair('actor_1', third)
                retained = session.timeline_clip()
                release.set(); session._thread.join(2)
                self.assertIs(session.timeline_clip(), retained)
                self.assertFalse(session.snapshot()['composed'])
                self.assertFalse(session.snapshot()['busy'])
                self.assertIsNone(session.snapshot()['failure'])
                self.assertEqual(retained.joints.tobytes(), original.joints.tobytes())

    def test_composed_segments_are_disclosed_and_archive_preserves_them(self):
        session = self.session()
        source = session.timeline_clip()
        metadata = dict(source.metadata, model='ARDY Core + InterGen', segments=[
            {'label': 'Core approach with display resampling', 'start_frame': 0, 'end_frame_exclusive': 60,
             'frames': 60, 'source': 'ardy_core', 'kind': 'approach'},
            {'label': 'Authored transition', 'start_frame': 60, 'end_frame_exclusive': 90,
             'frames': 30, 'source': 'authored_transition', 'kind': 'transition'},
            {'label': 'Native paired segment', 'start_frame': 90, 'end_frame_exclusive': 210,
             'frames': 120, 'source': 'intergen', 'kind': 'paired_action'}])
        # Structural fixture only, not a claim these source frames were generated by Core.
        clip = NativePairClip(source.joints, metadata=metadata)
        session.load_source(clip)
        state = session.snapshot()
        self.assertTrue(state['composed'])
        self.assertEqual(state['source'], 'ARDY Core + InterGen')
        self.assertEqual([s['start'] for s in state['segments']], [0, 60, 90])
        self.assertEqual([s['source'] for s in state['segments']], ['ardy_core', 'authored_transition', 'intergen'])
        session.play()
        self.assertIn('composed motion', session.snapshot()['status'])
        restored = self.session()
        restored.load(session.save())
        self.assertEqual(restored.timeline_clip().metadata['segments'], metadata['segments'])
        self.assertEqual(restored.timeline_clip().joints.tobytes(), source.joints.tobytes())
        self.assertIsNone(restored.timeline_clip().features)

    def test_performance_build_progress_and_atomic_commit(self):
        session = self.session()
        original = session.timeline_clip()
        started, release = threading.Event(), threading.Event()
        result_clip = self.context_fixture(original)
        def builder(scene, *, cancelled, on_progress):
            on_progress({'stage': 'routing', 'completed': 1, 'total': 3})
            started.set(); release.wait(2)
            self.assertFalse(cancelled())
            return {'clip': result_clip, 'placement': {'x': 2., 'z': 0., 'yaw_degrees': 30.}}
        session.build_performance(builder, request={'action': 'handshake'})
        self.assertTrue(started.wait(1))
        self.assertIs(session.timeline_clip(), original)
        self.assertEqual(session.snapshot()['pending_operation'], 'performance')
        self.assertEqual(session.snapshot()['progress']['stage'], 'routing')
        for edit in (lambda: session.add_actor('Blocked'),
                     lambda: session.rename_actor('actor_1', 'Blocked'),
                     lambda: session.set_placement(x=4.)):
            with self.assertRaises(RuntimeError):
                edit()
        release.set(); session._thread.join(2)
        self.assertIs(session.timeline_clip(), result_clip)
        self.assertEqual(session.snapshot()['placement']['x'], 2.)
        self.assertIsNone(session.snapshot()['progress'])

    def test_failed_performance_retry_then_scene_change_invalidates_retry(self):
        session = self.session()
        original = session.timeline_clip()
        attempts = []
        def builder(scene, *, cancelled, on_progress):
            attempts.append(scene)
            if len(attempts) == 1:
                raise RuntimeError('temporary service failure')
            return {'clip': original, 'placement': {'x': 0., 'z': 0., 'yaw_degrees': 0.}}
        session.build_performance(builder, request={'seed': 42})
        session._thread.join(2)
        self.assertIs(session.timeline_clip(), original)
        self.assertIn('temporary', session.snapshot()['failure'])
        session.retry(); session._thread.join(2)
        self.assertEqual(attempts[0], attempts[1])
        scene = session.scene_document
        scene['name'] = 'Edited scene'
        session.update_scene(scene)
        with self.assertRaisesRegex(ValueError, 'No generation request'):
            session.retry()

    def test_cancelled_performance_late_output_retains_complete_previous_state(self):
        session = self.session()
        original = session.timeline_clip()
        started, release = threading.Event(), threading.Event()
        def builder(scene, *, cancelled, on_progress):
            started.set(); release.wait(2)
            self.assertTrue(cancelled())
            on_progress('late progress')
            return {'clip': self.context_fixture(original),
                    'placement': {'x': 50., 'z': 0., 'yaw_degrees': 0.}}
        session.build_performance(builder)
        self.assertTrue(started.wait(1))
        session.cancel(); release.set(); session._thread.join(2)
        self.assertIs(session.timeline_clip(), original)
        self.assertEqual(session.snapshot()['placement']['x'], 0.)
        self.assertIsNone(session.snapshot()['progress'])
        self.assertFalse(session.snapshot()['busy'])

    def test_scene_and_placement_collision_reject_before_commit(self):
        from scene_objects import make_object
        session = self.session()
        original = session.timeline_clip()
        scene = session.scene_document
        blocker = make_object('crate', 0)
        blocker['position'] = original.joints[0, 0, 0].tolist()
        scene['objects'].append(blocker)
        with self.assertRaisesRegex(ValueError, 'scene solid'):
            session.update_scene(scene)
        self.assertEqual(session.scene_document['objects'], [])
        blocker['position'][0] += 10.
        session.update_scene(scene)
        with self.assertRaisesRegex(ValueError, 'scene solid'):
            session.set_placement(x=10.)
        self.assertEqual(session.snapshot()['placement']['x'], 0.)
        shifted = NativePairClip(original.joints + [10., 0., 0.], metadata=original.metadata)
        with self.assertRaisesRegex(ValueError, 'scene solid'):
            session.load_source(shifted)
        self.assertIs(session.timeline_clip(), original)

    def test_generated_and_imported_collision_reject_before_commit(self):
        from native_pair_clip import encode_project
        from scene_objects import make_object
        original = load_source(REVIEWED_HANDSHAKE)
        shifted = NativePairClip(original.joints + [10., 0., 0.], metadata=original.metadata)
        class Provider:
            def generate(self, **request):
                return shifted
        session = self.session(Provider())
        retained = session.timeline_clip()
        scene = session.scene_document
        blocker = make_object('crate', 0)
        blocker['position'] = shifted.joints[0, 0, 0].tolist()
        scene['objects'].append(blocker)
        session.update_scene(scene)
        session.generate('Greeting', 42)
        session._thread.join(2)
        self.assertIs(session.timeline_clip(), retained)
        self.assertIn('scene solid', session.snapshot()['failure'])
        state = session.snapshot()
        archive = encode_project(shifted, state['cast'], state['selected_pair'], scene)
        with self.assertRaisesRegex(ValueError, 'scene solid'):
            session.load(archive)
        self.assertIs(session.timeline_clip(), retained)

    def test_replacing_source_discards_context_retry_and_original(self):
        session = self.session()
        original = session.timeline_clip()
        def builder(base, scene, *, cancelled):
            return self.context_fixture(base)
        session.build_context(builder); session._thread.join(2)
        self.assertIs(session.original_pair_clip(), original)
        session.load_reviewed_handshake()
        self.assertIsNone(session.original_pair_clip())
        with self.assertRaisesRegex(ValueError, 'No generation request'):
            session.retry()

    def test_malformed_segment_provenance_is_rejected_before_replacing_motion(self):
        session = self.session()
        clip = session.timeline_clip()
        valid = {'label': 'Complete source', 'start_frame': 0, 'end_frame_exclusive': 210, 'frames': 210}
        bad_segments = [[], [dict(valid, start_frame=1)], [dict(valid, end_frame_exclusive=209, frames=209)],
                        [dict(valid, frames=209)], [dict(valid, start_frame=False)],
                        [dict(valid, label='')], [dict(valid, source='x' * 81)],
                        [dict(valid, invented='claim')], [valid, valid]]
        for segments in bad_segments:
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                session.load_source(NativePairClip(clip.joints, metadata={'segments': segments}))
            self.assertIs(session.timeline_clip(), clip)
        legacy = NativePairClip(clip.joints, metadata={'segments': [valid]})
        self.assertEqual(legacy.segments[0]['source'], 'mixed')
        self.assertEqual(legacy.segments[0]['kind'], 'mixed_segment')
        self.assertFalse(session.snapshot()['composed'])
        self.assertEqual(session.snapshot()['segments'][0]['source'], 'InterGen')

    def test_explicit_fists_persist_and_do_not_modify_generated_pose_arrays(self):
        session = self.session()
        original = session.timeline_clip()
        session.set_hand_pose('fists')
        self.assertEqual(session.snapshot()['hand_pose'], 'fists')
        self.assertEqual(original.joints.tobytes(), session.timeline_clip().joints.tobytes())
        self.assertEqual(original.features.tobytes(), session.timeline_clip().features.tobytes())
        renderer = NativePairRenderer(SimpleNamespace(scene=Scene()))
        self.addCleanup(renderer.remove)
        renderer.sync_cast(session.snapshot())
        renderer.set_clip(session.timeline_clip())
        renderer.tick(80)
        for index, identifier in enumerate(session.snapshot()['selected_pair']):
            np.testing.assert_array_equal(renderer.actors[identifier].joint_positions, original.joints[80,index])
            self.assertEqual(renderer.actors[identifier].provenance['authored_hand_pose']['name'], 'fist')
        restored = self.session()
        restored.load(session.save())
        self.assertEqual(restored.snapshot()['hand_pose'], 'fists')
        session.set_hand_pose('relaxed')
        self.assertEqual(session.snapshot()['hand_pose'], 'relaxed')
        with self.assertRaises(ValueError):
            session.set_hand_pose('automatic')

    def test_new_cast_displays_static_reviewed_stances_before_loading_motion(self):
        session = NativePairSession()
        self.addCleanup(session.close)
        session.add_actor('Third')
        renderer = NativePairRenderer(SimpleNamespace(scene=Scene()))
        self.addCleanup(renderer.remove)
        renderer.sync_cast(session.snapshot())
        renderer.set_visible(True)
        original = load_source(REVIEWED_HANDSHAKE).joints[0]
        roots = []
        for index, actor in enumerate(session.snapshot()['cast']):
            display = renderer.actors[actor['id']]
            self.assertTrue(display.visible)
            pose = display.joint_positions.copy()
            np.testing.assert_allclose(pose-pose[0], original[index % 2]-original[index % 2, 0], atol=1e-6)
            renderer.tick(100)
            np.testing.assert_array_equal(display.joint_positions, pose)
            roots.append(renderer.actor_root(actor['id']))
        self.assertGreater(np.linalg.norm(roots[0]-roots[1]), 1.79)
        self.assertGreater(np.linalg.norm(roots[0]-roots[2]), 1.79)
        np.testing.assert_allclose(renderer.actor_root(), np.mean(roots, axis=0))
        self.assertEqual(session.snapshot()['total_frames'], 0)

    def test_shared_placement_keeps_source_and_pair_distances_exact(self):
        session = self.session()
        source = session.timeline_clip().joints.copy()
        session.set_placement(x=3., z=-2., yaw_degrees=90.)
        renderer = NativePairRenderer(SimpleNamespace(scene=Scene()))
        self.addCleanup(renderer.remove)
        renderer.sync_cast(session.snapshot())
        renderer.set_clip(session.timeline_clip())
        renderer.tick(80)
        world = np.stack([renderer.actors[i].joint_positions for i in session.snapshot()['selected_pair']])
        rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        np.testing.assert_allclose(world, source[80] @ rotation.T + [3., 0., -2.], atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(world[0]-world[1], axis=-1),
                                   np.linalg.norm(source[80,0]-source[80,1], axis=-1), atol=1e-6)
        np.testing.assert_array_equal(source, session.timeline_clip().joints)
        restored = self.session()
        restored.load(session.save())
        self.assertEqual(restored.snapshot()['placement'], session.snapshot()['placement'])
        with self.assertRaises(ValueError):
            session.set_placement(x=float('nan'))

    def test_capture_lease_freezes_transport_and_mutation(self):
        session = self.session()
        session.seek(42)
        state = session.begin_capture()
        self.assertTrue(state['capturing'])
        for mutate in (session.play, session.pause, session.deactivate,
                       lambda: session.seek(1), lambda: session.add_actor('Unexpected'),
                       lambda: session.set_placement(x=1), session.load_reviewed_handshake):
            with self.assertRaises(RuntimeError):
                mutate()
        session.capture_seek(80)
        self.assertEqual(session.tick(10000.)['frame'], 80)
        self.assertTrue(session.save())
        session.end_capture(42)
        self.assertEqual(session.snapshot()['frame'], 42)
        self.assertFalse(session.snapshot()['capturing'])
        session.play()

    def test_cast_invariants(self):
        session = self.session()
        with self.assertRaises(ValueError):
            session.select_pair('actor_1', 'actor_1')
        with self.assertRaises(ValueError):
            session.remove_actor('actor_1')
        third = session.add_actor('Guest')
        session.rename_actor(third, 'Rival')
        session.remove_actor(third)
        for i in range(6):
            session.add_actor(f'Guest {i}')
        with self.assertRaises(ValueError):
            session.add_actor('Too many')


if __name__ == '__main__':
    unittest.main()
