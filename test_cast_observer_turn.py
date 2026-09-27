"""Observer request contracts and real geometry gates; no model execution."""
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cast_observer_turn import generate_observer_turn, ObserverTurnRejected, ObserverTurnIneligible
from experiments.native_pair_rig import NativeRigAsset
from native_pair_transition import CORE_TO_NATIVE, shared_place_pair
from realtime_backend import validate_job
from test_native_pair_rig import fixture_glb


class MotionClient:
    """Known motion independent of requested target schedules."""
    def __init__(self, pose, *, settle_on_second=False, drift=0., keep_heading=False, moving_tail=False):
        self.pose = pose
        self.requests, self.outputs = [], []
        self.settle_on_second, self.drift = settle_on_second, drift
        self.keep_heading, self.moving_tail = keep_heading, moving_tail

    def wait(self, request, cancelled):
        validate_job(request)
        self.requests.append(request)
        if self.keep_heading:
            angles = np.zeros(40)
        elif len(self.requests) == 2:
            # 90-degree finite-duration turn followed by a true stationary tail.
            a = np.clip(np.arange(40)/27, 0, 1)
            angles = (a*a*(3-2*a))*math.pi/2
        else:
            angles = np.zeros(40) if len(self.requests) == 1 else np.full(40, math.pi/2)
        native = np.stack([shared_place_pair(np.repeat(self.pose[None, None], 2, axis=1), yaw=x)[0, 0]
                           for x in angles])
        native[..., 0] += self.drift
        if self.moving_tail or (self.settle_on_second and len(self.requests) == 1):
            native[-10:, 20, 2] += np.where(np.arange(10) % 2, .04, 0.)
        core = np.zeros((1, 40, 27, 3)); core[0][:, CORE_TO_NATIVE] = native
        result = SimpleNamespace(actor_ids=tuple(request['actor_ids']), positions=core,
                                 native_features=np.full((1, 40, 330), len(self.requests), dtype=float),
                                 fps=20, frames=40)
        self.outputs.append(result)
        return [result]


class ObserverTurnTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name)/'rig.glb'; path.write_bytes(fixture_glb()[0])
        self.pose = NativeRigAsset(path).rest.copy()
        # Anatomically relaxed wrists; lengths remain those of the fixture.
        for shoulder, elbow, wrist in ((16, 18, 20), (17, 19, 21)):
            self.pose[elbow] = self.pose[shoulder]+[0, -.28, 0]
            self.pose[wrist] = self.pose[elbow]+[0, -.28, 0]
        self.left = np.repeat(self.pose[None], 4, axis=0)

    def generate(self, client, **kwargs):
        return generate_observer_turn(self.left, client, actor_id='released', target_xz=[3, 0],
                                      duration_frames=kwargs.pop('duration_frames', 180), seed=9, **kwargs)

    def test_exact_initial_placement_no_fake_history_and_full_provenance(self):
        client = MotionClient(self.pose)
        saved = self.left.copy()
        track, report = self.generate(client)
        self.assertEqual(track.shape, (180, 22, 3))
        self.assertEqual(len(client.requests), 2)
        request = client.requests[0]
        self.assertNotIn('history', request)
        self.assertEqual(request['actor_ids'], ['released'])
        placement = request['initial_placements']['released']
        np.testing.assert_array_equal(placement['position_xz'], self.left[-1, 0, [0, 2]])
        self.assertAlmostEqual(placement['yaw'], 0.)
        goals = request['root_targets']['released']
        self.assertEqual(goals[0]['heading'], placement['yaw'])
        self.assertAlmostEqual(goals[-1]['heading'], 0.)
        self.assertAlmostEqual(client.requests[1]['root_targets']['released'][-1]['heading'], math.pi/2)
        self.assertFalse(report['warmup_played'])
        self.assertEqual([w['used_in_performance'] for w in report['windows']], [False, True])
        self.assertTrue(all(g['position_xz'] == placement['position_xz'] for g in goals))
        self.assertTrue(report['mechanical_gate_passed'])
        self.assertFalse(report['animation_accepted'])
        self.assertTrue(report['generated_motion']['settled_tail_passed'])
        spans = report['spans']
        self.assertEqual([s['source'] for s in spans], ['authored_transition', 'ardy_core', 'stationary_hold'])
        self.assertEqual([s['start_frame'] for s in spans], [0]+[s['end_frame_exclusive'] for s in spans[:-1]])
        self.assertEqual(spans[-1]['end_frame_exclusive'], 180)
        np.testing.assert_array_equal(track[-30:], np.repeat(track[-1:], 30, axis=0))
        np.testing.assert_array_equal(saved, self.left)
        self.assertLess(report['generated_motion']['final_heading_error_degrees'], 1e-8)

    def test_actual_native330_history_only_on_second_window(self):
        client = MotionClient(self.pose)
        # The mock's hand discontinuity between horizons is intentionally invalid;
        # contract checks still apply on the rejected candidate.
        try:
            self.generate(client)
        except ObserverTurnRejected:
            pass
        self.assertEqual(len(client.requests), 2)
        second = client.requests[1]
        self.assertNotIn('initial_placements', second)
        self.assertEqual(second['stage_kind'], 'continuation')
        np.testing.assert_array_equal(second['history']['native_features'], client.outputs[0].native_features)

    def test_short_interval_skips_without_model_call(self):
        client = MotionClient(self.pose)
        with self.assertRaises(ObserverTurnIneligible):
            self.generate(client, duration_frames=149)
        self.assertFalse(client.requests)

    def test_unattained_heading_rejected_after_two_horizons_with_evidence(self):
        client = MotionClient(self.pose, keep_heading=True)
        with self.assertRaises(ObserverTurnRejected) as caught:
            self.generate(client)
        self.assertEqual(len(client.requests), 2)
        self.assertIsNotNone(caught.exception.candidate_joints)
        self.assertIn('does not face', str(caught.exception))
        self.assertFalse(caught.exception.report['mechanical_gate_passed'])

    def test_spatial_failure_does_not_spend_second_horizon(self):
        client = MotionClient(self.pose, drift=.31)
        with self.assertRaises(ObserverTurnRejected) as caught:
            self.generate(client)
        self.assertEqual(len(client.requests), 1)
        self.assertIn('root_excursion', str(caught.exception))
        self.assertEqual(caught.exception.candidate_joints.shape, (60, 22, 3))

    def test_moving_feet_cannot_be_hidden_by_static_padding(self):
        client = MotionClient(self.pose)
        original = client.wait
        def sliding(request, cancelled):
            result = original(request, cancelled)
            if len(client.requests) >= 2:
                result[0].positions[0, -10:, CORE_TO_NATIVE[10], 2] += np.where(np.arange(10) % 2, .04, 0.)
            return result
        client.wait = sliding
        with self.assertRaises(ObserverTurnRejected) as caught:
            self.generate(client)
        self.assertEqual(len(client.requests), 3)
        self.assertIn('tail_foot', str(caught.exception))
        self.assertFalse(caught.exception.report['generated_motion']['settled_tail_passed'])
        self.assertEqual([s['source'] for s in caught.exception.report['spans']], ['ardy_core'])

    def test_nonzero_released_placement_and_wrapped_shortest_heading_ramp(self):
        pair = np.repeat(self.left[:, None], 2, axis=1)
        self.left = shared_place_pair(pair, yaw=math.radians(170), translation=[1, 0, -2])[:, 0]
        root = self.left[-1, 0, [0, 2]]
        desired = math.radians(-170)
        target = root+3*np.array([math.sin(desired), math.cos(desired)])
        client = MotionClient(self.left[-1], keep_heading=True)
        generate_observer_turn(self.left, client, actor_id='released', target_xz=target,
                               duration_frames=180, seed=9)
        request = client.requests[0]
        np.testing.assert_array_equal(request['initial_placements']['released']['position_xz'], root)
        self.assertAlmostEqual(request['initial_placements']['released']['yaw'], math.radians(170))
        self.assertTrue(all(g['heading'] == request['initial_placements']['released']['yaw'] for g in request['root_targets']['released']))
        headings = np.unwrap([g['heading'] for g in client.requests[1]['root_targets']['released']])
        self.assertAlmostEqual(math.degrees(headings[-1]-headings[0]), 20.)
        self.assertTrue(np.all(np.diff(headings) >= 0))

    def test_transport_failure_preserves_typed_diagnostics(self):
        client = SimpleNamespace(wait=lambda *args, **kwargs: (_ for _ in ()).throw(OSError('transport failed')))
        with self.assertRaises(OSError) as caught:
            self.generate(client)
        self.assertIsNone(caught.exception.observer_candidate)
        self.assertEqual(caught.exception.observer_report['rejection_reasons'], ['transport failed'])

    def test_bridge_failures_try_only_original_strict_lengths(self):
        client = MotionClient(self.pose)
        with patch('cast_observer_turn.authored_direction_bridge', side_effect=ValueError('unsafe boundary')) as bridge:
            with self.assertRaises(ObserverTurnRejected) as caught:
                self.generate(client)
        self.assertEqual([call.kwargs['frames'] for call in bridge.call_args_list], [21, 24, 30])
        self.assertTrue(all(set(call.kwargs) == {'left_fps', 'right_fps', 'frames'} for call in bridge.call_args_list))
        self.assertEqual(len(caught.exception.report['bridge_attempts']), 3)


    def test_unsettled_warmup_stops_before_second_call(self):
        client = MotionClient(self.pose, settle_on_second=True)
        with self.assertRaisesRegex(ObserverTurnRejected, 'Neutral warmup rejected'):
            self.generate(client)
        self.assertEqual(len(client.requests), 1)

    def test_raised_warmup_start_is_not_played(self):
        client = MotionClient(self.pose)
        original = client.wait
        def raised(request, cancelled):
            result = original(request, cancelled)
            if len(client.requests) == 1:
                result[0].positions[0, :10, CORE_TO_NATIVE[20], 1] += .8
            return result
        client.wait = raised
        track, report = self.generate(client)
        self.assertLess((track[:, [20,21],1]-track[:,:1,1]).max(), .35)
        self.assertEqual(report['spans'][1]['source_frames'], 40)
        np.testing.assert_array_equal(client.requests[1]['history']['native_features'], client.outputs[0].native_features)

    def test_entry_arm_raise_rejected_despite_settled_tail(self):
        client = MotionClient(self.pose)
        original = client.wait
        def raised(request, cancelled):
            result = original(request, cancelled)
            if len(client.requests) == 2:
                # Rotate the whole arm up without changing source lengths.
                core = result[0].positions[0]
                for elbow, wrist, shoulder in ((18,20,16),(19,21,17)):
                    shoulder_position = core[0, CORE_TO_NATIVE[shoulder]].copy()
                    core[0, CORE_TO_NATIVE[elbow]] = shoulder_position+[0,.28,0]
                    core[0, CORE_TO_NATIVE[wrist]] = shoulder_position+[0,.56,0]
            return result
        client.wait = raised
        with self.assertRaisesRegex(ObserverTurnRejected, 'entry wrist rise'):
            self.generate(client)

    def test_cancellation_and_wrong_schema_never_publish(self):
        client = MotionClient(self.pose)
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            self.generate(client, cancelled=lambda: True)
        self.assertFalse(client.requests)
        original = client.wait
        def wrong(request, cancelled):
            result = original(request, cancelled); result[0].fps = 30
            return result
        client.wait = wrong
        with self.assertRaisesRegex(ObserverTurnRejected, 'incompatible'):
            self.generate(client)


if __name__ == '__main__':
    unittest.main()
