"""CPU-only stage scheduling, provenance, navigation and failure contracts."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from group_scene_sequence import build_group_sequence, _routes
from native_pair_transition import core27_to_native22
from paired_scene import EMPTY_SCENE
from scene_objects import make_object
from realtime_backend import validate_job
from test_independent_group_motion import _core_pose


def fixture(count=3, seconds=(2, 2), travel=False):
    ids = [f'actor_{index+1}' for index in range(count)]
    plan = {'prompt': 'People meet, dance together, then all backflip and land.',
            'actors': [{'id': aid, 'name': aid, 'start': None} for aid in ids], 'beats': []}
    for index, duration in enumerate(seconds):
        prompt = 'dance energetically' if index == 0 else 'perform one backflip and land'
        plan['beats'].append({'id': f'beat-{index+1}', 'actor_ids': ids[:1], 'prompt': prompt,
                             'seconds': duration, 'concurrent_solos': {aid: prompt for aid in ids[1:]}})
    starts = {aid: {'x': index*3., 'z': -1.3 if travel else 0., 'yaw_degrees': 90. if travel else 0.}
              for index, aid in enumerate(ids)}
    targets = {aid: dict(value, z=0., yaw_degrees=0.) for aid, value in starts.items()}
    return plan, starts, targets


class Client:
    def __init__(self, *, malformed=False, discontinuous=False, miss_route=False):
        self.requests, self.tokens, self.poses = [], {}, {}
        self.malformed, self.discontinuous, self.miss_route = malformed, discontinuous, miss_route
        self.outputs = []

    def wait(self, request, cancelled):
        validate_job(request)  # The real HTTP contract, without any model execution.
        aid = request['actor_ids'][0]
        self.requests.append(copy.deepcopy(request))
        initial = request.get('initial_placements', {}).get(aid)
        if initial:
            pose = _core_pose(initial['position_xz'][0])
            pose[:, 2] += initial['position_xz'][1]
        else:
            carry = 40 if 'root_targets' in request else 4
            expected = np.full((1, carry, 330), self.tokens[aid])
            np.testing.assert_array_equal(request['history']['native_features'], expected)
            pose = self.poses[aid].copy()
        positions = np.repeat(pose[None], 40, axis=0)
        if 'root_targets' in request and not self.miss_route:
            samples = request['root_targets'][aid]
            for axis, coordinate in ((0, 0), (2, 1)):
                roots = np.interp(np.arange(40), [t['frame'] for t in samples],
                                  [t['position_xz'][coordinate] for t in samples])
                positions[:, :, axis] += roots[:, None]-pose[0, axis]
        if self.discontinuous and not initial:
            positions[:, :, 0] += 1.
        token = float(len(self.requests))
        native = np.full((1, 40, 330), token)
        if self.malformed:
            positions[0, 0, 0] = np.nan
        self.tokens[aid], self.poses[aid] = token, positions[-1].copy()
        self.outputs.append(positions.copy())
        return [SimpleNamespace(actor_ids=(aid,), fps=20, frames=40,
                                positions=positions[None], native_features=native)]


class GroupSceneSequenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def build(self, client=None, *, fresh_action_stages=False, **kwargs):
        plan, starts, targets = fixture(**kwargs)
        result = build_group_sequence(client or Client(), EMPTY_SCENE, plan, starts, targets,
                                       seed=42, output_root=self.root, fresh_action_stages=fresh_action_stages)
        return result

    def manifest(self):
        return json.loads(next(self.root.glob('*/manifest.json')).read_text())

    def test_all_actor_private_history_crosses_action_boundaries_and_no_trim(self):
        client = Client()
        result = self.build(client, seconds=(3, 3))
        self.assertEqual(result['performance'].frames, 240)
        self.assertEqual(len(client.requests), 12)
        for aid in result['performance'].actor_ids:
            requests = [request for request in client.requests if request['actor_ids'] == [aid]]
            self.assertEqual([request['prompt'] for request in requests],
                             ['dance energetically']*2+['perform one backflip and land']*2)
            self.assertIn('initial_placements', requests[0])
            self.assertTrue(all('history' in request for request in requests[1:]))
        manifest = self.manifest()
        self.assertEqual(len(manifest['sources']), 12)
        self.assertEqual([(stage['start_frame'], stage['end_frame_exclusive']) for stage in manifest['stages']],
                         [(0, 120), (120, 240)])
        self.assertEqual(manifest['stages'][1]['actor_prompts'],
                         {aid: 'perform one backflip and land' for aid in result['performance'].actor_ids})
        for index in range(3):
            np.testing.assert_allclose(result['performance'].joints[-1, index],
                                       core27_to_native22(client.outputs[7+2*index][-1]))
        self.assertTrue(result['performance'].metadata['action_coverage']['final_stage_retained_in_full'])
        self.assertEqual(result['performance'].metadata['action_coverage']['observed'], 'unverified')
        self.assertFalse(result['performance'].metadata['animation_accepted'])

    def test_responsive_action_history_uses_actual_private_suffix_and_archives_full_source(self):
        class FrameTagged(Client):
            def __init__(self):
                super().__init__()
                self.previous = {}
            def wait(self, request, cancelled):
                aid = request['actor_ids'][0]
                if 'history' in request:
                    carry = 40 if 'root_targets' in request else 4
                    np.testing.assert_array_equal(request['history']['native_features'],
                                                  self.previous[aid][:, -carry:])
                    # Parent fixture checks actor identity with its scalar tag.
                    parent_request = copy.deepcopy(request)
                    parent_request['history']['native_features'] = np.full((1, carry, 330), self.tokens[aid]).tolist()
                else:
                    parent_request = request
                returned = super().wait(parent_request, cancelled)
                self.requests[-1] = copy.deepcopy(request)
                native = returned[0].native_features + np.arange(40)[None, :, None]/1000
                returned[0].native_features = native
                self.previous[aid] = native.copy()
                return returned
        client = FrameTagged()
        result = self.build(client, travel=True, seconds=(4, 4))
        manifest = self.manifest()
        self.assertEqual(manifest['history_policy']['action_frames'], 4)
        for request, source in zip(client.requests, manifest['sources']):
            expected = 0 if 'initial_placements' in request else 40 if 'root_targets' in request else 4
            self.assertEqual(source['conditioning_history_frames'], expected)
            with np.load(source['path']) as archive:
                self.assertEqual(archive['native_features'].shape, (1, 40, 330))
        self.assertEqual(result['performance'].frames, 360)
        self.assertEqual(result['performance'].metadata['continuity']['history_policy']['action_frames'], 4)
        self.assertTrue(all(stage['authored_frames'] == 0 for stage in manifest['stages']))

    def test_real_route_targets_then_actions_and_shared_arrival(self):
        client = Client()
        result = self.build(client, travel=True)
        self.assertEqual(result['performance'].frames, 240)
        manifest = self.manifest()
        self.assertEqual([stage['kind'] for stage in manifest['stages']],
                         ['approach', 'independent_motion', 'independent_motion'])
        self.assertEqual(manifest['approach']['arrival_seconds'], 2)
        self.assertEqual(manifest['approach']['approach_seconds'], 4)
        self.assertTrue(all(value < 1e-8 for value in manifest['stages'][0]['arrival_errors_m'].values()))
        approach_requests = [request for request in client.requests if 'root_targets' in request]
        self.assertEqual(len(approach_requests), 6)
        self.assertAlmostEqual(approach_requests[0]['root_targets']['actor_1'][0]['heading'], np.pi/2)
        self.assertEqual(len(manifest['seams']), 9)
        for request in client.requests[6:]:
            self.assertNotIn('root_targets', request)
            self.assertIn('history', request)
        for aid in result['performance'].actor_ids:
            last = manifest['approach']['horizons'][-1]['root_targets'][aid][-1]['position_xz']
            self.assertEqual(last, [manifest['targets'][aid]['x'], 0.])

    def test_one_actor_already_at_marks_needs_no_approach(self):
        result = self.build(count=1)
        self.assertEqual(result['performance'].frames, 120)
        self.assertTrue(result['approach']['already_at_targets'])
        self.assertEqual(len(result['performance'].metadata['segments']), 2)

    def test_yaw_only_approach_preserves_start_then_requests_target_heading(self):
        plan, starts, targets = fixture(count=1)
        starts['actor_1']['yaw_degrees'] = 90.
        route = _routes(EMPTY_SCENE, ('actor_1',), starts, targets)
        self.assertFalse(route['already_at_targets'])
        self.assertTrue(route['turn_only'])
        self.assertEqual(route['approach_seconds'], 2)
        points = route['horizons'][0]['root_targets']['actor_1']
        self.assertAlmostEqual(points[0]['heading'], np.pi/2)
        self.assertAlmostEqual(points[-1]['heading'], 0.)
        self.assertTrue(all(point['position_xz'] == [0., 0.] for point in points))
        client = Client()
        result = build_group_sequence(client, EMPTY_SCENE, plan, starts, targets,
                                       seed=42, output_root=self.root)
        self.assertEqual(result['performance'].frames, 180)
        self.assertAlmostEqual(client.requests[0]['initial_placements']['actor_1']['yaw'], np.pi/2)

    def test_equivalent_yaw_needs_no_approach(self):
        plan, starts, targets = fixture(count=1)
        starts['actor_1']['yaw_degrees'] = 360.
        self.assertTrue(_routes(EMPTY_SCENE, ('actor_1',), starts, targets)['already_at_targets'])

    def test_fresh_action_stages_reset_only_entry_and_add_shared_disclosed_bridges(self):
        client = Client()
        result = self.build(client, seconds=(4, 4), fresh_action_stages=True)
        self.assertEqual(result['performance'].frames, 261)
        manifest = self.manifest()
        for aid in result['performance'].actor_ids:
            requests = [request for request in client.requests if request['actor_ids'] == [aid]]
            self.assertEqual(['initial_placements' in request for request in requests], [True, False, True, False])
        self.assertEqual(len(manifest['transitions']), 1)
        transition = manifest['transitions'][0]
        self.assertEqual((transition['start_frame'], transition['end_frame_exclusive']), (120, 141))
        self.assertEqual(set(transition['actor_reports']), set(result['performance'].actor_ids))
        self.assertTrue(all(report['mechanical_gate_passed'] for report in transition['actor_reports'].values()))
        self.assertEqual([segment['source'] for segment in result['performance'].metadata['segments']],
                         ['ardy_core', 'authored_transition', 'ardy_core'])
        self.assertEqual(manifest['stages'][1]['start_frame'], 141)
        self.assertTrue(manifest['action_coverage']['final_stage_retained_in_full'])
        for stage in manifest['stages']:
            for aid, source in stage['actor_sources'].items():
                self.assertEqual(source['source_frames_trimmed'], 0)
                index = result['performance'].actor_ids.index(aid)
                track = np.load(source['candidate_path'])['joints']
                np.testing.assert_array_equal(result['performance'].joints[stage['start_frame']:stage['end_frame_exclusive'], index], track)
        self.assertTrue(result['performance'].metadata['continuity']['authored_transitions'])

    def test_fresh_actions_after_approach_have_two_disclosed_transitions(self):
        client = Client()
        result = self.build(client, count=1, travel=True, fresh_action_stages=True)
        self.assertEqual(result['performance'].frames, 282)
        manifest = self.manifest()
        self.assertEqual(len(manifest['transitions']), 2)
        self.assertEqual(['initial_placements' in request for request in client.requests], [True, False, True, True])
        self.assertEqual([(stage['start_frame'], stage['end_frame_exclusive']) for stage in manifest['stages']],
                         [(0, 120), (141, 201), (222, 282)])
        self.assertTrue(all(Path(transition['actor_candidate_paths']['actor_1']).exists()
                            for transition in manifest['transitions']))

    def test_fresh_stage_uses_actual_prior_root_and_heading(self):
        from paired_meetup import _heading
        class Drift(Client):
            def wait(self, request, cancelled):
                returned = super().wait(request, cancelled)
                positions = returned[0].positions[0]
                positions[:, :, 0] += np.linspace(0, .08, 40)[:, None]
                self.poses[request['actor_ids'][0]] = positions[-1].copy()
                return returned
        client = Drift()
        result = self.build(client, count=1, fresh_action_stages=True)
        first = np.load(result['manifest'].parent/'stage-00-actor_1.npz')['joints']
        initial = client.requests[1]['initial_placements']['actor_1']
        np.testing.assert_allclose(initial['position_xz'], first[-1, 0, [0, 2]])
        self.assertAlmostEqual(initial['yaw'], _heading(first[-1]))

    def test_failed_fresh_bridge_keeps_full_generated_candidates(self):
        with patch('group_scene_sequence.authored_direction_bridge', side_effect=ValueError('mechanical bridge rejection')):
            with self.assertRaisesRegex(ValueError, 'mechanical bridge rejection'):
                self.build(count=1, fresh_action_stages=True)
        manifest = self.manifest()
        self.assertEqual(manifest['status'], 'rejected')
        self.assertEqual(len(manifest['sources']), 2)
        self.assertTrue(Path(manifest['stages'][-1]['candidate_path']).exists())

    def test_shared_bridge_retries_every_actor_at_same_longer_duration(self):
        from native_pair_transition import authored_direction_bridge
        calls = []
        def measured(left, right, **kwargs):
            calls.append((float(left[-1, 0, 0, 0]), kwargs['frames']))
            bridge, report = authored_direction_bridge(left, right, **kwargs)
            if kwargs['frames'] == 21 and left[-1, 0, 0, 0] > 2.:
                report = dict(report, mechanical_gate_passed=False,
                              rejection_reasons=['authored bridge exceeds maximum joint speed'])
            return bridge, report
        with patch('group_scene_sequence.authored_direction_bridge', side_effect=measured):
            result = self.build(fresh_action_stages=True, seconds=(4, 4))
        manifest = self.manifest()
        transition = manifest['transitions'][0]
        self.assertEqual([duration for _, duration in calls], [21]*3+[24]*3)
        self.assertEqual([attempt['frames'] for attempt in transition['attempts']], [21, 24])
        self.assertEqual(transition['frames'], 24)
        self.assertEqual(result['performance'].frames, 264)
        self.assertEqual(manifest['frames'], 264)
        self.assertEqual(manifest['stages'][1]['start_frame'], 144)
        self.assertEqual(manifest['stages'][1]['end_frame_exclusive'], 264)
        for attempt in transition['attempts']:
            for path in attempt['actor_candidate_paths'].values():
                self.assertEqual(len(np.load(path)['joints']), attempt['frames'])
        for stage in manifest['stages']:
            for index, aid in enumerate(result['performance'].actor_ids):
                track = np.load(stage['actor_sources'][aid]['candidate_path'])['joints']
                np.testing.assert_array_equal(result['performance'].joints[stage['start_frame']:stage['end_frame_exclusive'], index], track)

    def test_shared_bridge_cannot_consume_source_or_future_transition_budget(self):
        from native_pair_transition import authored_direction_bridge
        calls = []
        def measured(left, right, **kwargs):
            calls.append(kwargs['frames'])
            bridge, report = authored_direction_bridge(left, right, **kwargs)
            return bridge, dict(report, mechanical_gate_passed=False, rejection_reasons=['too fast'])
        # Three2s source stages + two minimum21-frame bridges =222. Only
        # two spare frames remain, so a24-frame first bridge must not be tried.
        with patch('group_scene_sequence.MAX_FRAMES', 224):
            with patch('group_scene_sequence.authored_direction_bridge', side_effect=measured):
                with self.assertRaisesRegex(ValueError, 'total-frame budgets'):
                    self.build(count=1, seconds=(2, 2, 2), fresh_action_stages=True)
        self.assertEqual(calls, [21])
        self.assertEqual(self.manifest()['frames'], 222)
        self.assertEqual(self.manifest()['status'], 'rejected')

    def test_fresh_bridge_budget_rejects_before_compute(self):
        client = Client()
        with patch('group_scene_sequence.MAX_FRAMES', 250):
            with self.assertRaisesRegex(ValueError, 'cannot be trimmed'):
                self.build(client, seconds=(4, 4), fresh_action_stages=True)
        self.assertFalse(client.requests)
        self.assertEqual(self.manifest()['frames'], 261)

    def test_cancel_after_return_preserves_sources_and_never_returns_performance(self):
        client = Client()
        plan, starts, targets = fixture()
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            build_group_sequence(client, EMPTY_SCENE, plan, starts, targets, seed=42,
                                 output_root=self.root, cancelled=lambda: bool(client.requests))
        manifest = self.manifest()
        self.assertEqual(manifest['status'], 'cancelled')
        self.assertEqual(len(manifest['sources']), 1)

    def test_bad_core_return_archived_before_rejection(self):
        with self.assertRaisesRegex(ValueError, 'Core returned incompatible'):
            self.build(Client(malformed=True))
        manifest = self.manifest()
        self.assertEqual(manifest['status'], 'rejected')
        with np.load(manifest['sources'][0]['path'], allow_pickle=False) as source:
            self.assertTrue(np.isnan(source['positions']).any())

    def test_all_multiple_returned_candidates_archived(self):
        class Multiple(Client):
            def wait(self, request, cancelled):
                returned = super().wait(request, cancelled)
                return returned*2
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            self.build(Multiple())
        self.assertEqual(len(self.manifest()['sources']), 2)

    def test_geometry_rejection_retains_stage_candidate(self):
        with patch('group_scene_sequence._geometry', side_effect=ValueError('body-proxy overlap')):
            with self.assertRaisesRegex(ValueError, 'body-proxy overlap'):
                self.build()
        manifest = self.manifest()
        self.assertTrue(Path(manifest['stages'][0]['candidate_path']).exists())
        self.assertEqual(len(manifest['sources']), 3)

    def test_root_clear_but_body_overlap_is_rejected(self):
        from native_pair_transition import CORE27_NAMES
        class OverlappingHands(Client):
            def wait(self, request, cancelled):
                returned = super().wait(request, cancelled)
                side = 'RightHand' if request['actor_ids'] == ['actor_1'] else 'LeftHand'
                joint = CORE27_NAMES.index(side)
                returned[0].positions[:, :, joint] = [1.5, 1.1, 0.]
                return returned
        with self.assertRaisesRegex(ValueError, 'body-proxy overlap'):
            self.build(OverlappingHands(), count=2)
        manifest = self.manifest()
        self.assertTrue(Path(manifest['stages'][0]['candidate_path']).exists())
        self.assertTrue(manifest['failure_diagnostics']['body_clearance'])

    def test_seam_rejected_without_replacing_generated_action(self):
        with self.assertRaisesRegex(ValueError, 'continuation seam'):
            self.build(Client(discontinuous=True), count=1)
        manifest = self.manifest()
        self.assertEqual(len(manifest['sources']), 2)
        self.assertEqual(manifest['seams'][0]['mean_joint_jump_m'], 1.)

    def test_frame_overflow_rejects_before_compute_without_trimming_ending(self):
        client = Client()
        with self.assertRaisesRegex(ValueError, 'cannot be trimmed'):
            self.build(client, travel=True, seconds=(10, 10, 10))
        self.assertFalse(client.requests)
        self.assertGreater(self.manifest()['frames'], 1000)

    def test_missing_actor_and_mixed_pairs_rejected_before_compute(self):
        for change in ('missing', 'pair'):
            plan, starts, targets = fixture()
            if change == 'missing':
                plan['beats'][1]['concurrent_solos'].pop('actor_3')
            else:
                plan['beats'][1]['actor_ids'] = ['actor_1', 'actor_2']
            with self.assertRaisesRegex(ValueError, 'every actor exactly one'):
                build_group_sequence(Client(), EMPTY_SCENE, plan, starts, targets,
                                     seed=1, output_root=self.root)

    def test_crossing_synchronous_approach_rejected(self):
        plan, starts, targets = fixture(count=2)
        targets = {'actor_1': starts['actor_2'], 'actor_2': starts['actor_1']}
        with self.assertRaisesRegex(ValueError, 'cross too closely'):
            build_group_sequence(Client(), EMPTY_SCENE, plan, starts, targets,
                                 seed=1, output_root=self.root)
        self.assertFalse(self.manifest()['sources'])

    def test_static_obstacle_routed_without_scene_mutation_and_blocked_slot_rejected(self):
        plan, starts, targets = fixture(count=1)
        starts['actor_1'].update(x=-2, z=0)
        targets['actor_1'].update(x=2, z=0)
        blocker = make_object('crate', 0)
        blocker.update(position=[0., .5, 0.], size=[.7, 1., .7])
        scene = dict(EMPTY_SCENE, objects=[blocker])
        before = copy.deepcopy(scene)
        route = _routes(scene, ('actor_1',), starts, targets)
        self.assertGreater(len(route['routes'][0]['points']), 2)
        self.assertEqual(scene, before)
        targets['actor_1']['x'] = 0
        with self.assertRaisesRegex(ValueError, 'overlaps scene geometry'):
            _routes(scene, ('actor_1',), starts, targets)

    def test_unsupported_ground_rejected(self):
        plan, starts, targets = fixture(count=1)
        floor = make_object('platform', 0)
        floor.update(position=[0., -.05, 0.], size=[6., .1, 6.])
        targets['actor_1']['z'] = 5.
        with self.assertRaisesRegex(ValueError, 'floor'):
            _routes(dict(EMPTY_SCENE, objects=[floor]), ('actor_1',), starts, targets)

    def test_model_cannot_ignore_navigation_targets(self):
        with self.assertRaisesRegex(ValueError, 'missed the planned group route'):
            self.build(Client(miss_route=True), travel=True)
        self.assertTrue(self.manifest()['sources'])


if __name__ == '__main__':
    unittest.main()
