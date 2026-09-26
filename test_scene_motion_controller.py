"""Controller regressions for scene-aware motion and take preservation."""

import copy
import math
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

from object_directing import ObjectDirectorSession
from cinematic_adventure import make_temple
from scene_composition import make_preset
from scene_interaction_geometry import SceneInteractionGeometry
from scene_objects import make_object
from scene_motion import WALK_SPEED, plan_scene_motion
from test_live_motion import result, wait_until


class RecordingBackend:
    """Backend that records the actual prompt/history crossing the worker boundary."""

    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.finished = threading.Event()

    def cancel(self, _request_id):
        pass  # Cancellation must work even when the backend ignores it.

    def generate(self, request_id, prompt, history):
        self.calls.append((prompt, None if history is None else history.copy()))
        self.started.set()
        self.release.wait(2)
        output = result(request_id, marker=len(self.calls))
        self.finished.set()
        return output


class SceneMotionControllerTests(unittest.TestCase):
    def setUp(self):
        self.backend = RecordingBackend()
        self.session = ObjectDirectorSession(
            self.backend,
            np.zeros((20, 34, 3), dtype=np.float32),
            np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1)),
        )
        self.session.set_mode('Live ARDY')

    def generate(self, prompt='walk', **kwargs):
        self.session.submit(prompt, **kwargs)
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        return self.session.takes[self.session.active_take]

    def fake_scene_motion(self, corrected=17, recommended_seconds=None):
        """A deterministic scene adapter exposing processed data flow."""
        starts = []
        priors = []

        def plan(scene, prompt, start):
            starts.append((copy.deepcopy(scene), prompt, start.copy()))
            if prompt == 'unsupported':
                raise ValueError('No reachable walk surface for that target')
            return types.SimpleNamespace(backend_prompt='walk toward the landing',
                                         recommended_seconds=recommended_seconds)

        def apply(output, _plan, prior_positions):
            priors.append(prior_positions.copy())
            output['positions'] = output['positions'].copy()
            output['motion'] = output['motion'].copy()
            output['positions'][:, :, 0] = prior_positions[:, 0] + 1
            output['motion'][:] = corrected + len(priors)
            return output

        module = types.SimpleNamespace(plan_scene_motion=plan, apply_scene_motion=apply)
        return patch.dict('sys.modules', {'scene_motion': module}), starts, priors

    def test_corrected_motion_is_next_chunk_history_and_pose_progresses(self):
        shim, starts, priors = self.fake_scene_motion()
        with shim:
            take = self.generate('walk to the landing', seconds=8)
        self.assertEqual(len(self.backend.calls), 2)
        self.assertEqual([call[0] for call in self.backend.calls],
                         ['walk toward the landing'] * 2)
        self.assertIsNone(self.backend.calls[0][1])
        np.testing.assert_array_equal(self.backend.calls[1][1],
                                      np.full((52, 414), 18, dtype=np.float32))
        self.assertEqual(len(priors), 2)
        np.testing.assert_array_equal(priors[0], np.zeros((34, 3), dtype=np.float32))
        np.testing.assert_array_equal(priors[1][:, 0], np.ones(34, dtype=np.float32))
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(take.positions), 200)
        np.testing.assert_array_equal(take.positions[:104, :, 0], 1)
        np.testing.assert_array_equal(take.positions[104:, :, 0], 2)
        self.assertEqual(take.segments[0]['end'], 200)

    def test_final_chunk_is_trimmed_before_scene_processing_and_metrics(self):
        processed_lengths = []

        def process(output, _plan, prior_positions):
            length = len(output['positions'])
            processed_lengths.append(length)
            if length != (104 if len(processed_lengths) == 1 else 21):
                raise ValueError('Scene processor received frames beyond the retained clip')
            output['positions'][:, 0, 0] = len(processed_lengths)
            output['metadata']['processed_frames'] = length
            output['metadata']['scene_motion'] = f'{length} frames grounded'
            return output

        module = types.SimpleNamespace(
            plan_scene_motion=lambda *_: types.SimpleNamespace(backend_prompt='walk'),
            apply_scene_motion=process,
        )
        with patch.dict('sys.modules', {'scene_motion': module}):
            take = self.generate('walk', seconds=5)
        self.assertEqual(processed_lengths, [104, 21])
        self.assertEqual(len(take.positions), 125)
        np.testing.assert_array_equal(take.positions[:104, 0, 0], 1)
        np.testing.assert_array_equal(take.positions[104:, 0, 0], 2)
        self.assertEqual(self.session.metrics['processed_frames'], 21)
        self.assertIn('21 frames grounded', self.session.status)

    def test_scene_duration_hint_only_extends_text_length_auto_estimate(self):
        shim, _starts, _priors = self.fake_scene_motion(recommended_seconds=8)
        with shim:
            auto = self.generate('walk', edit_mode='new')
            self.assertEqual(len(auto.positions), 200)
            self.assertEqual(self.session.duration_label, 'Auto · scene route')
            explicit = self.generate('walk', seconds=5, edit_mode='new')
            self.assertEqual(len(explicit.positions), 125)
            self.assertEqual(self.session.duration_label, 'Specified duration')
            in_prompt = self.generate('walk for 6 seconds', edit_mode='new')
            self.assertEqual(len(in_prompt.positions), 150)
            self.assertEqual(self.session.duration_label, 'Auto · duration stated in prompt')
        self.assertEqual([len(t.positions) for t in (auto, explicit, in_prompt)],
                         [200, 125, 150])

    def test_scene_duration_hint_is_capped_and_checked_against_take_budget(self):
        shim, _starts, _priors = self.fake_scene_motion(recommended_seconds=45)
        with shim:
            capped = self.generate('walk', edit_mode='new')
            self.assertEqual(len(capped.positions), 750)
            self.assertIn('capped at 30 s', self.session.duration_label)
            calls = len(self.backend.calls)
            with patch('directing.MAX_FRAMES', 900):
                self.session.submit('walk', edit_mode='extend')
            self.assertFalse(self.session.busy)
            self.assertEqual(len(self.backend.calls), calls)
            self.assertIs(self.session.takes[capped.id], capped)
            self.assertIn('Motion budget reached', self.session.status)

    def test_branch_and_extend_keep_exact_prefix_and_use_last_pose(self):
        # The first take is ordinary backend output. Scene correction begins at
        # the requested edit point and cannot rewrite saved frames before it.
        with patch.dict('sys.modules', {'scene_motion': types.SimpleNamespace(plan_scene_motion=lambda *_: None)}):
            original = self.generate('plain walk')
        original_arrays = {key: getattr(original, key).copy()
                           for key in ('positions', 'rotations', 'motion')}
        self.session.seek(40)
        shim, starts, _priors = self.fake_scene_motion()
        with shim:
            branch = self.generate('toward landing')
            extension = self.generate('toward landing', edit_mode='extend')
        self.assertNotEqual(branch.id, original.id)
        self.assertEqual(branch.parent, original.id)
        self.assertEqual(branch.branch_frame, 40)
        self.assertIs(self.session.takes[original.id], original)
        for key, values in original_arrays.items():
            np.testing.assert_array_equal(getattr(branch, key)[:41], values[:41])
            np.testing.assert_array_equal(getattr(original, key), values)
        self.assertEqual(starts[0][2].shape, (34, 3))
        np.testing.assert_array_equal(starts[0][2], original.positions[40])
        self.assertEqual(extension.id, branch.id)
        for key in ('positions', 'rotations', 'motion'):
            np.testing.assert_array_equal(getattr(extension, key)[:len(branch.positions)],
                                          getattr(branch, key))
        np.testing.assert_array_equal(starts[1][2], branch.positions[-1])

    def test_unsupported_target_rejects_without_changing_saved_take(self):
        shim, _starts, _priors = self.fake_scene_motion()
        with shim:
            original = self.generate()
            original_motion = original.motion.copy()
            calls = len(self.backend.calls)
            self.session.submit('unsupported', edit_mode='extend')
        self.assertFalse(self.session.busy)
        self.assertEqual(len(self.backend.calls), calls)
        self.assertIs(self.session.takes[original.id], original)
        np.testing.assert_array_equal(original.motion, original_motion)
        self.assertEqual(self.session.active_take, original.id)
        self.assertIn('No reachable walk surface', self.session.status)

    def test_rejected_new_instruction_cancels_older_inflight_result(self):
        shim, _starts, _priors = self.fake_scene_motion()
        with shim:
            original = self.generate()
            self.backend.started.clear()
            self.backend.finished.clear()
            self.backend.release.clear()
            self.session.submit('toward landing', edit_mode='extend')
            self.assertTrue(self.backend.started.wait(1))
            self.session.submit('unsupported', edit_mode='extend')
            self.assertFalse(self.session.busy)
            self.backend.release.set()
            self.assertTrue(self.backend.finished.wait(1))
            time.sleep(.05)
        self.assertIs(self.session.takes[original.id], original)
        self.assertEqual(len(self.session.takes), 1)
        self.assertEqual(len(original.positions), 104)
        self.assertIn('No reachable walk surface', self.session.status)

    def test_scene_change_during_generation_discards_stale_result(self):
        self._assert_geometry_change_cancels(lambda session: session.set_scene(make_preset('Designed apartment')))

    def test_object_change_during_generation_discards_stale_result(self):
        self._assert_geometry_change_cancels(lambda session: session.set_objects([make_object('chair', 0)]))

    def _assert_geometry_change_cancels(self, change):
        shim, _starts, _priors = self.fake_scene_motion()
        with shim:
            original = self.generate()
            original_arrays = {key: getattr(original, key).copy()
                               for key in ('positions', 'rotations', 'motion')}
            self.backend.started.clear()
            self.backend.finished.clear()
            self.backend.release.clear()
            self.session.submit('toward landing', edit_mode='extend')
            self.assertTrue(self.backend.started.wait(1))
            change(self.session)
            self.assertFalse(self.session.busy)
            self.backend.release.set()
            self.assertTrue(self.backend.finished.wait(1))
            # The backend completion event fires before the controller has
            # processed its result, so allow that worker handoff to finish.
            time.sleep(.05)
        self.assertEqual(len(self.session.takes), 1)
        self.assertIs(self.session.takes[original.id], original)
        self.assertEqual(self.session.active_take, original.id)
        for key, values in original_arrays.items():
            np.testing.assert_array_equal(getattr(original, key), values)
        self.assertIn('Scene changed', self.session.status)

    def test_real_temple_stair_ascent_stays_on_rendered_route_across_chunks(self):
        pose = np.zeros((34, 3), dtype=np.float32)
        pose[:, 1] = .95
        pose[1], pose[8] = [-.1, .9, 0], [.1, .9, 0]
        for hip, knee, ankle, toe, foot, x in ((3, 4, 5, 6, 7, -.1),
                                                (10, 11, 12, 13, 14, .1)):
            pose[hip], pose[knee] = [x, .84, 0], [x, .47, -.03]
            pose[ankle], pose[toe], pose[foot] = ([x, .1, 0], [x, .02, -.07],
                                                   [x, 0, -.14])

        class PoseBackend(RecordingBackend):
            def generate(self, request_id, prompt, history):
                self.calls.append((prompt, None if history is None else history.copy()))
                output = result(request_id, marker=0)
                output['positions'][:] = pose
                return output

        scene = make_temple()
        backend = PoseBackend()
        session = ObjectDirectorSession(
            backend, np.repeat(pose[None], 20, axis=0),
            np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1)),
        )
        session.set_mode('Live ARDY')
        session.set_scene(scene)
        route = plan_scene_motion(scene, 'walk up ceremonial stairs', pose).route
        route_length = np.linalg.norm(np.diff(route[:, [0, 2]], axis=0), axis=1).sum()
        session.submit('walk up ceremonial stairs')
        wait_until(lambda: not session.busy)
        self.assertEqual(session.kind, 'generated', session.status)
        take = session.takes[session.active_take]
        self.assertGreaterEqual(len(take.positions), math.ceil(route_length / WALK_SPEED * 25))
        self.assertEqual(len(backend.calls), math.ceil(len(take.positions) / 104))
        self.assertGreaterEqual(len(backend.calls), 3)
        np.testing.assert_array_equal(backend.calls[1][1], take.motion[52:104])
        np.testing.assert_array_equal(backend.calls[2][1], take.motion[156:208])
        self.assertTrue(all('alternating steps' in prompt for prompt, _ in backend.calls))

        geometry = SceneInteractionGeometry.from_scene(scene)
        roots = take.positions[:, 0]
        supports = [geometry.support_height(p[0], p[2], p[1] - .7,
                                            max_step_up=.6, max_drop=2)
                    for p in roots]
        self.assertTrue(all(height is not None for height in supports))
        clearances = roots[:, 1] - np.asarray(supports)
        self.assertGreater(clearances.min(), .2)
        self.assertLess(clearances.max(), 1.05)
        lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(route[:, [0, 2]], axis=0), axis=1))]
        expected_progress = min(WALK_SPEED * len(roots) / 25, lengths[-1])
        expected_xz = [np.interp(expected_progress, lengths, route[:, axis])
                       for axis in (0, 2)]
        np.testing.assert_allclose(roots[-1, [0, 2]], expected_xz, atol=.08)
        self.assertGreater(supports[-1], supports[0] + .5)


if __name__ == '__main__':
    unittest.main()
