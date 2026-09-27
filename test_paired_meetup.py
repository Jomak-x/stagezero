"""CPU tests of planning, transport contracts, rejection and native preservation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from experiments.native_pair_rig import NativeRigAsset
from native_pair_clip import NativePairClip, load_source
from native_pair_transition import CORE_TO_NATIVE, shared_place_pair
from scene_objects import make_object
from realtime_backend import validate_job
from paired_meetup import plan_meetup, build_meetup, _minimum_separation, _route_separation
from test_native_pair_rig import fixture_glb

SCENE = {'version': 2, 'name': 'Stage', 'objects': [], 'effects': [], 'lighting': 'neutral'}
IDS = ('alice', 'bob')
STARTS = [{'x': -1., 'z': -2.}, {'x': 1., 'z': -4.}]
MEETING = {'x': 0., 'z': 0., 'yaw_degrees': 0.}


class Client:
    def __init__(self, pair, drift=0.):
        self.pair, self.drift, self.requests = pair, drift, []
    def wait(self, request, cancelled):
        self.requests.append(request)
        p = np.zeros((2, 40, 27, 3), dtype=float)
        for i, aid in enumerate(request['actor_ids']):
            p[i][:, CORE_TO_NATIVE] = self.pair[0, i]
            goals = request['root_targets'][aid]
            xz = np.stack([np.interp(np.arange(40), [g['frame'] for g in goals],
                                    [g['position_xz'][axis] for g in goals]) for axis in range(2)], axis=-1)
            for axis, coordinate in enumerate((0, 2)):
                p[i, :, :, coordinate] += (xz[:, axis]-self.pair[0, i, 0, coordinate])[:, None]
        p[..., 0] += self.drift
        return [SimpleNamespace(positions=p, rotations=np.broadcast_to(np.eye(3), (2, 40, 27, 3, 3)),
                                native_features=np.full((2, 40, 330), len(self.requests)),
                                actor_ids=tuple(request['actor_ids']), fps=20, frames=40)]


class MeetupTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory(); self.addCleanup(self.folder.cleanup)
        path = Path(self.folder.name)/'fixture.glb'; path.write_bytes(fixture_glb()[0])
        pose = NativeRigAsset(path).rest
        poses = np.repeat(np.stack([pose+[-1, 0, 0], pose+[1, 0, 0]])[None], 8, axis=0)
        self.pair = NativePairClip(poses, metadata={'model': 'InterGen', 'render_hand_pose': 'fists'})
    def plan(self, **kwargs):
        return plan_meetup(self.pair, SCENE, actor_ids=IDS, starts=STARTS, meeting=MEETING, **kwargs)
    def client(self):
        plan = self.plan(); p = plan['placement']
        return Client(shared_place_pair(self.pair.joints, translation=[p['x'], 0, p['z']]))
    def build(self, client, **kwargs):
        return build_meetup(self.pair, client, SCENE, actor_ids=IDS, starts=STARTS, meeting=MEETING, **kwargs)
    def test_plan_is_json_and_equal_arrival_with_distinct_starts(self):
        plan = self.plan(); json.dumps(plan, allow_nan=False)
        self.assertNotEqual(plan['routes'][0]['distance_m'], plan['routes'][1]['distance_m'])
        self.assertLessEqual(plan['approach_seconds'], 20)
        self.assertLessEqual(plan['total_frames'], 1000)
        for i, aid in enumerate(IDS):
            np.testing.assert_allclose(plan['horizons'][-2]['root_targets'][aid][-1]['position_xz'], plan['routes'][i]['points'][-1])
            np.testing.assert_allclose(plan['horizons'][-1]['root_targets'][aid][-1]['position_xz'], plan['routes'][i]['points'][-1])
    def test_meeting_anchors_root_midpoint_and_rigid_yaw(self):
        p = plan_meetup(self.pair, SCENE, actor_ids=IDS, starts=STARTS,
                        meeting={'x': 3, 'z': 2, 'yaw_degrees': 75})['placement']
        world = shared_place_pair(self.pair.joints, yaw=np.deg2rad(p['yaw_degrees']), translation=[p['x'], 0, p['z']])
        np.testing.assert_allclose(world[0, :, 0].mean(0)[[0, 2]], [3, 2])
    def test_real_window_contract_exact_source_and_hand_pose(self):
        client = self.client(); progress = []; archive = []
        result = self.build(client, on_progress=progress.append, on_core_chunk=lambda *args: archive.append(args))
        self.assertEqual(result['clip'].metadata['render_hand_pose'], 'fists')
        np.testing.assert_array_equal(result['clip'].joints[-self.pair.frames:], self.pair.joints)
        np.testing.assert_array_equal(result['joints'][-self.pair.frames:], client.pair)
        self.assertEqual(len(archive), len(client.requests))
        self.assertEqual(progress[-1]['phase'], 'complete')
        self.assertIn('initial_placements', client.requests[0]); self.assertNotIn('history', client.requests[0])
        for i, request in enumerate(client.requests[1:], 1):
            self.assertNotIn('initial_placements', request)
            np.testing.assert_array_equal(request['history']['native_features'], np.full((2, 40, 330), i))
            self.assertEqual(set(request['actor_prompts']), set(IDS))
        for request in client.requests:
            validate_job(request)
    def test_missed_route_archived_and_rejected_without_next_job(self):
        client = self.client(); client.drift = 1.; archive = []
        with self.assertRaisesRegex(ValueError, 'missed the planned route'):
            self.build(client, on_core_chunk=lambda *args: archive.append(args))
        self.assertEqual(len(client.requests), 1); self.assertEqual(len(archive), 1)
    def test_cancel_before_core(self):
        client = self.client()
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            self.build(client, cancelled=lambda: True)
        self.assertEqual(client.requests, [])
    def test_invalid_inputs_and_budget_rejected(self):
        for starts in ([{'x': float('nan'), 'z': 0}, STARTS[1]], [{'x': 30, 'z': 0}, STARTS[1]], [STARTS[0]]):
            with self.assertRaises(ValueError):
                plan_meetup(self.pair, SCENE, actor_ids=IDS, starts=starts, meeting=MEETING)
        with self.assertRaises(ValueError): self.plan(max_seconds=4)
        with self.assertRaises(ValueError):
            plan_meetup(self.pair, SCENE, actor_ids=('a', 'a'), starts=STARTS, meeting=MEETING)
    def test_continuous_crossing_detected_between_samples(self):
        self.assertAlmostEqual(_minimum_separation(np.array([[[-1, 0], [1, 0]], [[1, 0], [-1, 0]]])), 0.)
        starts = [{'x': 2, 'z': 0}, {'x': -2, 'z': 0}]
        plan = plan_meetup(self.pair, SCENE, actor_ids=IDS, starts=starts, meeting=MEETING)
        self.assertTrue(plan['approach_rerouted'])
        self.assertGreaterEqual(_route_separation(plan['routes']), .55)
        self.assertEqual([r['actor_id'] for r in plan['routes']], list(IDS))
        self.assertEqual([r['points'][0] for r in plan['routes']], [[2., 0.], [-2., 0.]])
        np.testing.assert_allclose([r['points'][-1] for r in plan['routes']], [[-1., 0.], [1., 0.]])
        self.assertEqual(plan, plan_meetup(self.pair, SCENE, actor_ids=IDS, starts=starts, meeting=MEETING))

    def test_default_route_does_not_enter_fallback(self):
        with patch('paired_meetup._alternate_routes', side_effect=AssertionError('Default route changed')):
            plan = self.plan()
        self.assertFalse(plan['approach_rerouted'])
        self.assertEqual([len(route['points']) for route in plan['routes']], [2, 2])

    def test_actual_rotated_sparring_routes_preserve_endpoints(self):
        root = Path(__file__).parent
        pair = load_source(root/'review/two-character/platform-integration/block37/block37.native.npz')
        scene = json.loads((root/'review/scene-integration/live-city.json').read_text())
        plan = plan_meetup(pair, scene, actor_ids=IDS,
                           starts=[{'x': -3, 'z': -1}, {'x': 3, 'z': 1}],
                           meeting={'x': 0, 'z': 0, 'yaw_degrees': 90})
        self.assertTrue(plan['approach_rerouted'])
        self.assertEqual(plan['approach_seconds'], 10)
        self.assertGreaterEqual(plan['minimum_planned_root_separation_m'], .55)
        placement = plan['placement']
        world = shared_place_pair(pair.joints, yaw=np.pi/2,
                                  translation=[placement['x'], 0., placement['z']])
        for i, route in enumerate(plan['routes']):
            np.testing.assert_allclose(route['points'][-1], world[0, i, 0, [0, 2]])
            self.assertTrue(route['ground']['continuous_support_verified'])

    def test_crossing_remains_rejected_if_detours_exceed_budget(self):
        with self.assertRaisesRegex(ValueError, 'no bounded clear detour'):
            plan_meetup(self.pair, SCENE, actor_ids=IDS, speed_mps=.2, max_seconds=12,
                        starts=[{'x': 1, 'z': 0}, {'x': -1, 'z': 0}], meeting=MEETING)

    def test_crossing_remains_rejected_if_all_detours_leave_floor(self):
        # Native placement preflight succeeds; direct routes fit. Every detour
        # must go through the same ground validator and cannot be accepted.
        from paired_meetup import validate_ground_path
        checked = []
        def ground(scene, points):
            checked.append(points)
            if any(abs(point[1]) > .3 for point in points):
                raise ValueError('Detour leaves continuous authored floor')
            return validate_ground_path(scene, points)
        with patch('paired_meetup.validate_ground_path', side_effect=ground):
            with self.assertRaisesRegex(ValueError, 'no bounded clear detour'):
                plan_meetup(self.pair, SCENE, actor_ids=IDS,
                            starts=[{'x': 2, 'z': 0}, {'x': -2, 'z': 0}], meeting=MEETING)
        self.assertGreater(len(checked), 2)
    def test_obstacle_detour_and_floor_preflight(self):
        blocker = make_object('crate', 0)
        blocker.update(position=[1., .5, -2.], size=[.7, 1., .7])
        scene = dict(SCENE, objects=[blocker])
        plan = plan_meetup(self.pair, scene, actor_ids=IDS, starts=STARTS, meeting=MEETING)
        self.assertGreater(len(plan['routes'][1]['points']), 2)
        floor = make_object('platform', 1)
        floor.update(position=[0., -.05, 0.], size=[6., .1, 6.])
        with self.assertRaisesRegex(ValueError, 'floor'):
            plan_meetup(self.pair, dict(SCENE, objects=[floor]), actor_ids=IDS, starts=STARTS, meeting=MEETING)

    def test_foreign_actor_output_rejected_without_continuing(self):
        client = self.client(); original = client.wait
        def wrong(request, cancelled):
            output = original(request, cancelled)
            output[0].actor_ids = ('intruder', 'bob')
            return output
        client.wait = wrong
        with self.assertRaisesRegex(ValueError, 'incompatible actors'):
            self.build(client)
        self.assertEqual(len(client.requests), 1)

    def test_entry_rejects_vertical_drop_before_bridge_despite_exact_xz(self):
        shifted = self.pair.joints.copy()
        shifted[:, 0, :, 1] -= .9097
        native = NativePairClip(shifted, metadata=self.pair.metadata)
        before = native.joints.copy()
        client = self.client()
        with patch('paired_meetup.authored_direction_bridge') as bridge:
            with self.assertRaisesRegex(ValueError, r'root-height gap for alice is 0.91 m, exceeding 0.30 m'):
                build_meetup(native, client, SCENE, actor_ids=IDS, starts=STARTS, meeting=MEETING)
            bridge.assert_not_called()
        np.testing.assert_array_equal(native.joints, before)
        self.assertEqual(len(client.requests), len(self.plan()['horizons']))
        np.testing.assert_allclose(client.requests[-1]['root_targets']['alice'][-1]['position_xz'],
                                   before[0, 0, 0, [0, 2]])

    def test_accepted_entry_reports_signed_height_gaps_and_limit(self):
        shifted = self.pair.joints.copy()
        shifted[:, 0, :, 1] -= .29
        native = NativePairClip(shifted, metadata=self.pair.metadata)
        result = build_meetup(native, self.client(), SCENE, actor_ids=IDS, starts=STARTS, meeting=MEETING)
        np.testing.assert_allclose(result['metadata']['entry_root_height_gaps_m'], [-.29, 0.], atol=1e-12)
        self.assertEqual(result['metadata']['max_entry_root_height_gap_m'], .30)
        self.assertEqual(result['plan']['limits']['max_entry_root_height_gap_m'], .30)
        np.testing.assert_array_equal(result['clip'].joints[-native.frames:], native.joints)

    def test_failed_entry_gate_never_publishes(self):
        with patch('paired_meetup.authored_direction_bridge', return_value=(np.zeros((21,2,22,3)),
                   {'mechanical_gate_passed': False, 'rejection_reasons': ['too fast']})):
            with self.assertRaisesRegex(ValueError, 'transition rejected'):
                self.build(self.client())


if __name__ == '__main__': unittest.main()
