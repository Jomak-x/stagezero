"""CPU contracts only: injected test doubles are never motion quality evidence."""
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import numpy as np

from experiments.native_pair_rig import NativeRigAsset
from native_pair_clip import NativePairClip
from native_pair_transition import CORE_TO_NATIVE, shared_place_pair
from prompt_scene_builder import PromptSceneBuilder, align_pair, _bridge, check_cast_geometry, pair_source_prompt, _scene_with_idle_roots, select_initial_staging, select_later_meeting, select_native_wait_pose, generate_later_approach
from paired_meetup import plan_meetup
from experiments.trial_prompt_scene import ReplayPairProvider
from realtime_backend import validate_job
from test_native_pair_rig import fixture_glb

SCENE = {'version': 2, 'name': 'Stage', 'objects': [], 'effects': [], 'lighting': 'neutral'}


class CoreDouble:
    def __init__(self, pose):
        self.pose, self.requests = pose, []

    def wait(self, request, cancelled):
        validate_job(request)
        self.requests.append(request)
        n = len(request['actor_ids'])
        positions = np.zeros((n, 40, 27, 3))
        for i, aid in enumerate(request['actor_ids']):
            pose = self.pose.copy()
            placement = request.get('initial_placements', {}).get(aid)
            if placement:
                pose[:, [0, 2]] += np.asarray(placement['position_xz'])-pose[0, [0, 2]]
            else:
                # Recorded test history carries its actual previous XZ origin.
                pose[:, [0, 2]] += np.asarray(request['history']['native_features'])[i, -1, :2]-pose[0, [0, 2]]
            positions[i][:, CORE_TO_NATIVE] = pose
            if aid in request.get('root_targets', {}):
                targets = request['root_targets'][aid]
                for coordinate, axis in enumerate((0, 2)):
                    desired = np.interp(np.arange(40), [goal['frame'] for goal in targets],
                                        [goal['position_xz'][coordinate] for goal in targets])
                    positions[i, :, :, axis] += (desired-positions[i, :, 0, axis])[:, None]
        features = np.zeros((n, 40, 330))
        features[:, :, :2] = positions[:, :, 0][:, :, [0, 2]]
        return [SimpleNamespace(actor_ids=tuple(request['actor_ids']), positions=positions,
            native_features=features, rotations=np.broadcast_to(np.eye(3), (n, 40, 27, 3, 3)), frames=40, fps=20)]


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rig = self.root/'fixture.glb'; rig.write_bytes(fixture_glb()[0])
        self.pose = NativeRigAsset(rig).rest
        self.client = CoreDouble(self.pose)
        self.staging = {'starts': {f'actor_{i+1}': {'x': i*3., 'z': 0.} for i in range(3)},
                        'meeting': {'x': 1.5, 'z': 0., 'yaw_degrees': 0.}, 'target_id': None}
        self.placement = patch('prompt_scene_plan.auto_place', return_value=self.staging)
        self.placement.start(); self.addCleanup(self.placement.stop)

    def build(self, count=1, beats=None, provider=None, **kwargs):
        plan = {'title': 'Test contract', 'actors': [{'id': f'actor_{i+1}', 'name': str(i)} for i in range(count)],
                'beats': beats or [{'id': 'beat-1', 'actor_ids': ['actor_1'], 'prompt': 'Wave.', 'seconds': 4}]}
        planner = SimpleNamespace(plan=lambda *args, **kw: plan)
        return PromptSceneBuilder('Test contract', planner, provider, self.client, self.root/'sources')(SCENE, **kwargs)

    def test_solo_real_history_contract_complete_windows_archive_and_duration(self):
        result = self.build()
        self.assertEqual(result.joints.shape, (120, 1, 22, 3))
        self.assertEqual(len(self.client.requests), 2)
        self.assertIn('initial_placements', self.client.requests[0])
        self.assertNotIn('initial_placements', self.client.requests[1])
        self.assertEqual(np.asarray(self.client.requests[1]['history']['native_features']).shape, (1, 40, 330))
        metadata = result.metadata
        self.assertEqual(metadata['segments'][0]['source'], 'ardy_core')
        self.assertFalse(metadata['animation_accepted'])
        manifest = json.loads(Path(metadata['source_manifest']).read_text())
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(len(manifest['sources']), 2)
        with np.load(manifest['sources'][0]['path']) as saved:
            self.assertEqual(saved['native_features'].shape, (1, 40, 330))

    def test_three_cast_solo_changes_hold_persistent_poses_and_reset_history(self):
        beats = [{'id': f'beat-{i+1}', 'actor_ids': [f'actor_{i+1}'], 'prompt': 'Stand.', 'seconds': 2} for i in range(3)]
        result = self.build(3, beats)
        self.assertEqual(result.joints.shape[1:], (3, 22, 3))
        self.assertTrue(all(len(r['actor_ids']) <= 2 for r in self.client.requests))
        self.assertEqual(self.client.requests[0]['actor_ids'], ['actor_2', 'actor_3'])
        for request in self.client.requests[1:]:
            self.assertIn('initial_placements', request)
            self.assertNotIn('history', request)
        for activity in result.metadata['segment_activity']:
            for aid in activity['held_actor_ids']:
                chunk = result.joints[activity['start_frame']:activity['end_frame_exclusive'], result.actor_ids.index(aid)]
                np.testing.assert_array_equal(chunk, np.repeat(chunk[:1], len(chunk), axis=0))
        np.testing.assert_array_equal(result.joints[0, 0], result.joints[-1, 0])

    def test_pair_alignment_is_shared_rigid_and_preserves_every_frame(self):
        source = np.repeat(np.stack([self.pose+[-1, 0, 0], self.pose+[1, 0, 0]])[None], 60, axis=0)
        source[:, 0, 20, 2] += np.linspace(0, .2, 60)
        prior = shared_place_pair(source[:2], yaw=.8, translation=[3, 0, -2])
        aligned, report = align_pair(source, prior)
        np.testing.assert_allclose(aligned[:, :, 0], shared_place_pair(source, yaw=.8, translation=[3, 0, -2])[:, :, 0])
        np.testing.assert_allclose(np.linalg.norm(source[:, 0]-source[:, 1], axis=-1), np.linalg.norm(aligned[:, 0]-aligned[:, 1], axis=-1))
        np.testing.assert_array_equal(aligned[..., 1], source[..., 1])
        self.assertEqual(len(aligned), len(source))

    def test_global_cast_names_become_two_person_source_roles(self):
        actors = [{'id': 'actor_1', 'name': 'Person 1'}, {'id': 'actor_2', 'name': 'Person 2'}, {'id': 'actor_3', 'name': 'Person 3'}]
        result = pair_source_prompt('Person 2 helps Person 3 stand up; actor_2 steps back.', ['actor_2', 'actor_3'], actors)
        self.assertEqual(result, 'the first person helps the second person stand up; the first person steps back.')

    def test_full_pair_source_retained_after_later_shared_placement(self):
        pair_joints = np.repeat(np.stack([self.pose, self.pose+[3, 0, 0]])[None], 30, axis=0)
        pair = NativePairClip(pair_joints, np.zeros((30, 2, 262)), {'model': 'InterGen'})
        provider = SimpleNamespace(generate=lambda *args, **kwargs: pair, last_raw_archive=None)
        beats = [{'id': 'beat-1', 'actor_ids': ['actor_1'], 'prompt': 'Stand.', 'seconds': 2},
                 {'id': 'beat-2', 'actor_ids': ['actor_1', 'actor_2'], 'prompt': 'Face each other.', 'seconds': 1}]
        result = self.build(2, beats, provider)
        np.testing.assert_allclose(result.joints[-30:], pair_joints, atol=1e-12)
        self.assertEqual(result.metadata['segments'][-1]['source'], 'intergen')
        self.assertTrue(any(s['source'] == 'intergen' for s in result.metadata['sources']))

    def test_first_pair_uses_real_meetup_and_third_track_holds_generated_pose(self):
        pair_joints = np.repeat(np.stack([self.pose, self.pose+[3, 0, 0]])[None], 30, axis=0)
        pair = NativePairClip(pair_joints, np.zeros((30, 2, 262)), {'model': 'InterGen'})
        provider = SimpleNamespace(generate=lambda *args, **kwargs: pair, last_raw_archive=None)
        beats = [{'id': 'beat-1', 'actor_ids': ['actor_1', 'actor_2'], 'prompt': 'Face each other.', 'seconds': 1}]
        result = self.build(3, beats, provider)
        np.testing.assert_allclose(result.joints[-30:, :2], pair_joints, atol=1e-12)
        np.testing.assert_array_equal(result.joints[:, 2], np.repeat(result.joints[:1, 2], result.frames, axis=0))
        self.assertEqual([s['source'] for s in result.metadata['segments']], ['ardy_core', 'authored_transition', 'intergen'])
        self.assertTrue(all(len(r['actor_ids']) <= 2 for r in self.client.requests))

    def test_second_pair_prefetch_overlaps_first_core_approach_and_captures_each_raw_archive(self):
        second_started, release_second = threading.Event(), threading.Event()
        self.addCleanup(release_second.set)
        pose = self.pose
        calls, raw_archives, observed = [], [], []
        class Provider:
            last_raw_archive = None
            running = False
            def generate(inner, prompt, seed, frames, cancelled):
                self.assertFalse(inner.running, 'InterGen calls must stay serialized')
                inner.running = True
                calls.append(prompt)
                if len(calls) == 2:
                    second_started.set()
                    self.assertTrue(release_second.wait(2), 'Core approach never overlapped the second source')
                joints = np.repeat(np.stack([pose, pose+[3, 0, 0]])[None], frames, axis=0)
                pair = NativePairClip(joints, np.full((frames, 2, 262), seed, dtype=float), {'model': 'InterGen', 'prompt': prompt})
                output = io.BytesIO()
                np.savez_compressed(output, joints=pair.joints, features=pair.features, metadata=np.array(json.dumps(pair.metadata)))
                inner.last_raw_archive = output.getvalue()
                raw_archives.append(inner.last_raw_archive)
                inner.running = False
                return pair
        provider = Provider()
        original = self.client.wait
        def core(request, cancelled):
            # Skip the third actor's seed request; inspect the first actual approach.
            if len(request['actor_ids']) == 2 and not observed:
                self.assertTrue(second_started.wait(2), 'Second source was not prefetched')
                observed.append(provider.running)
                release_second.set()
            return original(request, cancelled)
        self.client.wait = core
        beats = [{'id': 'beat-1', 'actor_ids': ['actor_1', 'actor_2'], 'prompt': 'Person 1 faces Person 2.', 'seconds': 1},
                 {'id': 'beat-2', 'actor_ids': ['actor_1', 'actor_2'], 'prompt': 'Person 1 faces Person 2.', 'seconds': 1}]
        result = self.build(2, beats, provider)
        self.assertEqual(observed, [True])
        self.assertFalse(provider.running)
        self.assertEqual(calls, ['the first person faces the second person.']*2)
        sources = [source for source in result.metadata['sources'] if source['source'] == 'intergen']
        self.assertEqual(len(sources), 2)
        for source, raw in zip(sources, raw_archives):
            self.assertEqual(Path(source['path']).read_bytes(), raw)
            self.assertTrue(Path(source['path']).with_suffix('.json').is_file())
        self.assertEqual(sources[1]['source_role_map'], [
            {'source_actor_index': 0, 'actor_id': 'actor_1'}, {'source_actor_index': 1, 'actor_id': 'actor_2'}])

    def test_cancel_joins_active_prefetch_and_cancels_queued_sources(self):
        started, stopped, external = threading.Event(), threading.Event(), threading.Event()
        calls = []
        class Provider:
            def generate(inner, prompt, seed, frames, cancelled):
                calls.append(prompt); started.set()
                try:
                    for _ in range(200):
                        if cancelled():
                            raise RuntimeError('Provider cancelled')
                        external.wait(.01)
                    raise AssertionError('Prefetch did not receive cancellation')
                finally:
                    stopped.set()
        beats = [{'id': f'beat-{i+1}', 'actor_ids': ['actor_1', 'actor_2'], 'prompt': 'Face each other.', 'seconds': 1} for i in range(3)]
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            self.build(2, beats, Provider(), cancelled=started.is_set)
        self.assertTrue(stopped.is_set(), 'Builder returned with a hidden source job still active')
        self.assertEqual(len(calls), 1)
        manifest = json.loads(next((self.root/'sources').glob('*/manifest.json')).read_text())
        self.assertEqual(manifest['status'], 'cancelled')

    def test_completed_orphan_pair_archived_and_manifested_on_core_failure(self):
        completed = threading.Event()
        pose = self.pose
        class Provider:
            last_raw_archive = None
            def generate(inner, prompt, seed, frames, cancelled):
                pair = NativePairClip(np.repeat(np.stack([pose, pose+[3, 0, 0]])[None], frames, axis=0), metadata={'model': 'InterGen'})
                completed.set()
                return pair
        def core(request, cancelled):
            self.assertTrue(completed.wait(2))
            raise ValueError('Core intentionally rejected')
        self.client.wait = core
        beats = [{'id': 'beat-1', 'actor_ids': ['actor_1', 'actor_2'], 'prompt': 'Face each other.', 'seconds': 1},
                 {'id': 'beat-2', 'actor_ids': ['actor_3'], 'prompt': 'Stand.', 'seconds': 2}]
        with self.assertRaisesRegex(ValueError, 'Core intentionally rejected'):
            self.build(3, beats, Provider())
        manifest = json.loads(next((self.root/'sources').glob('*/manifest.json')).read_text())
        self.assertEqual(manifest['status'], 'rejected')
        source = next(s for s in manifest['sources'] if s['source'] == 'intergen')
        self.assertFalse(source['used_in_performance'])
        self.assertTrue(Path(source['path']).is_file())

    def test_core_failure_cancels_and_joins_active_pair_without_masking_original_error(self):
        started, stopped = threading.Event(), threading.Event()
        class Provider:
            def generate(inner, prompt, seed, frames, cancelled):
                started.set()
                try:
                    for _ in range(200):
                        if cancelled():
                            raise RuntimeError('Secondary pair cancellation')
                        stopped.wait(.005)
                    raise AssertionError('Local failure stop was not propagated')
                finally:
                    stopped.set()
        def core(request, cancelled):
            self.assertTrue(started.wait(2))
            raise ValueError('Original Core rejection')
        self.client.wait = core
        beats = [{'id': 'beat-1', 'actor_ids': ['actor_1', 'actor_2'], 'prompt': 'Face each other.', 'seconds': 1},
                 {'id': 'beat-2', 'actor_ids': ['actor_3'], 'prompt': 'Stand.', 'seconds': 2}]
        with self.assertRaisesRegex(ValueError, 'Original Core rejection'):
            self.build(3, beats, Provider())
        self.assertTrue(stopped.is_set())
        manifest = json.loads(next((self.root/'sources').glob('*/manifest.json')).read_text())
        self.assertEqual(manifest['error'], 'Original Core rejection')

    def test_inactive_future_participant_uses_own_native_anatomy_and_stable_role(self):
        first = NativePairClip(np.repeat(np.stack([self.pose, self.pose+[3, 0, 0]])[None], 30, axis=0), metadata={'model': 'InterGen'})
        incoming = NativePairClip(np.repeat(np.stack([self.pose, self.pose*1.25+[3, 0, 0]])[None], 30, axis=0), metadata={'model': 'InterGen'})
        pairs = iter((first, incoming))
        provider = SimpleNamespace(generate=lambda *a, **kw: next(pairs), last_raw_archive=None)
        beats = [{'id': 'beat-1', 'actor_ids': ['actor_1', 'actor_2'], 'prompt': 'Face each other.', 'seconds': 1},
                 {'id': 'beat-2', 'actor_ids': ['actor_2', 'actor_3'], 'prompt': 'Face each other.', 'seconds': 1}]
        result = self.build(3, beats, provider)
        self.assertTrue(all(len(request['actor_ids']) == 2 for request in self.client.requests), 'Native idle seed must not start a Core stand-in job')
        expected = incoming.joints[0, 1].copy()
        expected[:, 0] += 6-expected[0, 0]
        expected[:, 2] -= expected[0, 2]
        np.testing.assert_array_equal(result.joints[0, 2], expected)
        manifest = json.loads(Path(result.metadata['source_manifest']).read_text())
        idle = manifest['idle_initializations'][0]
        self.assertEqual((idle['source'], idle['source_beat_index'], idle['source_actor_index']), ('intergen', 1, 1))
        np.testing.assert_allclose(result.joints[-30:, 2]-result.joints[-30:, 2, :1], incoming.joints[:, 1]-incoming.joints[:, 1, :1])

    def test_distant_later_pair_runs_real_core_travel_not_long_authored_bridge(self):
        pair = NativePairClip(np.repeat(np.stack([self.pose, self.pose+[1, 0, 0]])[None], 30, axis=0), metadata={'model': 'InterGen'})
        provider = SimpleNamespace(generate=lambda *a, **kw: pair, last_raw_archive=None)
        beats = [{'id': 'beat-1', 'actor_ids': ['actor_1'], 'prompt': 'Stand.', 'seconds': 2},
                 {'id': 'beat-2', 'actor_ids': ['actor_1', 'actor_2'], 'prompt': 'Face each other.', 'seconds': 1}]
        result = self.build(2, beats, provider)
        travel = [request for request in self.client.requests if len(request['actor_ids']) == 2]
        self.assertGreater(len(travel), 0)
        self.assertNotIn('history', travel[0])
        self.assertEqual(travel[0]['initial_placements']['actor_1']['position_xz'], [0., 0.])
        self.assertEqual(travel[0]['initial_placements']['actor_2']['position_xz'], [3., 0.])
        manifest = json.loads(Path(result.metadata['source_manifest']).read_text())
        self.assertEqual(manifest['beats'][-1]['navigation'], 'real Core approach from current poses')
        self.assertGreater(max(manifest['beats'][-1]['root_gaps_m']), .30)
        alignment = manifest['beats'][-1]['shared_placement']
        expected = shared_place_pair(pair.joints, yaw=alignment['yaw_radians'], translation=alignment['translation'])
        np.testing.assert_allclose(result.joints[-30:], expected, atol=1e-12)

    def test_idle_root_proxy_reroutes_approach_without_editing_authored_scene(self):
        idle = self.pose.copy(); idle[:, [0, 2]] -= idle[0, [0, 2]]
        planning, proxies = _scene_with_idle_roots(SCENE, {'idle': idle}, ['a', 'b'])
        self.assertEqual(SCENE['objects'], [])
        self.assertEqual(proxies[0]['root_xz'], [0., 0.])
        pair = NativePairClip(np.repeat(np.stack([self.pose+[2, 0, 0], self.pose+[2, 0, 3]])[None], 30, axis=0), metadata={'model': 'InterGen'})
        plan = plan_meetup(pair, planning, actor_ids=['a', 'b'], starts=[{'x': -2., 'z': 0.}, {'x': -2., 'z': 3.}],
                           meeting={'x': 2., 'z': 1.5}, entry_policy='continuous', speed_mps=.85)
        self.assertGreater(len(plan['routes'][0]['points']), 2)
        self.assertTrue(any(abs(point[1]) > .5 for point in plan['routes'][0]['points'][1:-1]))

    def test_automatic_start_candidates_preserve_explicit_start_and_archive_rejection(self):
        pair = NativePairClip(np.repeat(np.stack([self.pose, self.pose+[3, 0, 0]])[None], 30, axis=0), metadata={'model': 'InterGen'})
        plan = {'actors': [{'id': 'actor_1', 'start': {'x': 0., 'z': 0.}}, {'id': 'actor_2', 'start': None}]}
        placement = {key: value for key, value in self.staging.items()}
        placement['starts'] = {aid: dict(self.staging['starts'][aid]) for aid in ('actor_1', 'actor_2')}
        calls, reports = [], []
        def route(*args, **kwargs):
            calls.append(kwargs['starts'])
            if kwargs['starts'] == [placement['starts']['actor_1'], placement['starts']['actor_2']]:
                raise ValueError('Original approach is obstructed')
            return plan_meetup(*args, **kwargs)
        with patch('prompt_scene_builder.plan_meetup', side_effect=route):
            selected, poses, scene, proxies = select_initial_staging(pair, SCENE, plan, placement, {}, ['actor_1', 'actor_2'], candidate_reports=reports)
        self.assertEqual(selected['starts']['actor_1'], {'x': 0., 'z': 0.})
        self.assertNotEqual(selected['starts']['actor_2'], placement['starts']['actor_2'])
        self.assertEqual(next(record for record in reports if record['candidate'] == 'original')['rejection'], 'Original approach is obstructed')
        self.assertTrue(any(record.get('selected') and record['accepted'] for record in reports))
        self.assertLessEqual(len(reports), 19)
        self.assertEqual(placement['starts']['actor_2'], {'x': 3., 'z': 0.})

    def test_all_explicit_starts_are_not_relocated_after_rejection(self):
        pair = NativePairClip(np.repeat(np.stack([self.pose, self.pose+[3, 0, 0]])[None], 30, axis=0), metadata={'model': 'InterGen'})
        plan = {'actors': [{'id': aid, 'start': dict(self.staging['starts'][aid])} for aid in ('actor_1', 'actor_2')]}
        placement = dict(self.staging, starts={aid: dict(self.staging['starts'][aid]) for aid in ('actor_1', 'actor_2')})
        reports = []
        with patch('prompt_scene_builder.plan_meetup', side_effect=ValueError('Explicit route blocked')):
            with self.assertRaisesRegex(ValueError, 'Explicit route blocked'):
                select_initial_staging(pair, SCENE, plan, placement, {}, ['actor_1', 'actor_2'], candidate_reports=reports)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]['starts'], placement['starts'])

    def test_automatic_staging_prefers_short_source_roles_over_swapped_generic_marks(self):
        pair = NativePairClip(np.repeat(np.stack([self.pose, self.pose+[3, 0, 0]])[None], 30, axis=0), metadata={'model': 'InterGen'})
        plan = {'actors': [{'id': aid, 'start': None} for aid in ('actor_1', 'actor_2')]}
        placement = dict(self.staging, starts={'actor_1': {'x': 3., 'z': 0.}, 'actor_2': {'x': 0., 'z': 0.}})
        reports = []
        selected, _, _, _ = select_initial_staging(pair, SCENE, plan, placement, {}, ['actor_1', 'actor_2'], candidate_reports=reports)
        self.assertLess(selected['starts']['actor_1']['x'], selected['starts']['actor_2']['x'])
        chosen = next(record for record in reports if record.get('selected'))
        self.assertNotEqual(chosen['candidate'], 'original')
        self.assertEqual((chosen['approach_seconds'], chosen['total_route_distance_m']),
                         min((record['approach_seconds'], record['total_route_distance_m']) for record in reports if record['accepted']))

    def test_native_replay_keeps_exact_archive_and_refuses_changed_prompt_or_duration(self):
        path = self.root/'pair-00.npz'
        joints = np.repeat(np.stack([self.pose, self.pose+[3, 0, 0]])[None], 30, axis=0)
        np.savez_compressed(path, joints=joints, features=np.zeros((30, 2, 262)), metadata=np.array(json.dumps({'model': 'InterGen', 'prompt': 'Two people greet.'})))
        manifest = self.root/'manifest.json'
        manifest.write_text(json.dumps({'seed': 42, 'sources': [{'source': 'intergen', 'path': str(path), 'source_prompt': 'Two people greet.'}]}))
        provider = ReplayPairProvider(manifest)
        result = provider.generate('Two people greet.', 42, 30, lambda: False)
        np.testing.assert_array_equal(result.joints, joints)
        self.assertEqual(provider.last_raw_archive, path.read_bytes())
        for prompt, seed, frames in [('Changed action.', 42, 30), ('Two people greet.', 42, 60), ('Two people greet.', 43, 30)]:
            with self.assertRaises(ValueError):
                provider.generate(prompt, seed, frames, lambda: False)

    def test_later_meeting_moves_active_pair_away_from_unchanged_idle_pose(self):
        pair = NativePairClip(np.repeat(np.stack([self.pose, self.pose+[0, 0, 1]])[None], 30, axis=0), metadata={'model': 'InterGen'})
        original = pair.joints.copy()
        poses = {'actor_1': self.pose.copy(), 'actor_2': self.pose+[0, 0, 2], 'actor_3': self.pose+[0, 0, -2]}
        pose_copies = {aid: pose.copy() for aid, pose in poses.items()}
        prior = np.repeat(np.stack([poses['actor_2'], poses['actor_3']])[None], 2, axis=0)
        plan = {'actors': [{'id': aid} for aid in poses], 'meeting': None}
        reports = []
        result = select_later_meeting(pair, SCENE, plan, self.staging, poses, ['actor_2', 'actor_3'], prior, candidate_reports=reports)
        self.assertFalse(reports[0]['accepted'])
        self.assertIn('rejection', reports[0])
        chosen = next(record for record in reports if record.get('selected'))
        self.assertGreater(np.linalg.norm(chosen['meeting_offset_xz']), 0.)
        self.assertLessEqual(len(reports), 25)
        self.assertEqual((chosen['approach_seconds'], chosen['total_route_distance_m']),
                         min((record['approach_seconds'], record['total_route_distance_m']) for record in reports if record['accepted']))
        self.assertIsNotNone(result['route_plan'])
        for aid in poses:
            np.testing.assert_array_equal(poses[aid], pose_copies[aid])
        np.testing.assert_array_equal(pair.joints, original)
        alignment = result['alignment']
        expected = shared_place_pair(original, yaw=alignment['yaw_radians'], translation=alignment['translation'])
        np.testing.assert_allclose(result['placed'], expected, atol=1e-12)

    def test_later_explicit_meeting_is_never_shifted_away_from_idle_collision(self):
        pair = NativePairClip(np.repeat(np.stack([self.pose, self.pose+[0, 0, 1]])[None], 30, axis=0), metadata={'model': 'InterGen'})
        poses = {'actor_1': self.pose.copy(), 'actor_2': self.pose+[0, 0, 2], 'actor_3': self.pose+[0, 0, -2]}
        prior = np.repeat(np.stack([poses['actor_2'], poses['actor_3']])[None], 2, axis=0)
        plan = {'actors': [{'id': aid} for aid in poses], 'meeting': {'x': 0., 'z': 0.}}
        reports = []
        with self.assertRaisesRegex(ValueError, 'No bounded later meeting'):
            select_later_meeting(pair, SCENE, plan, dict(self.staging, meeting={'x': 0., 'z': 0.}), poses,
                                  ['actor_2', 'actor_3'], prior, candidate_reports=reports)
        self.assertEqual(len(reports), 1)
        self.assertEqual([reports[0]['meeting'][axis] for axis in ('x', 'z')], [0., 0.])
        self.assertTrue(reports[0]['explicit_meeting_preserved'])

    def waiting_source(self):
        neutral = self.pose.copy()
        neutral[20] = neutral[0]+[-.18, -.15, 0.]
        neutral[21] = neutral[0]+[.18, -.15, 0.]
        source = np.repeat(np.stack([neutral, neutral+[3, 0, 0]])[None], 20, axis=0)
        source[:6, 1, 9, 2] += .5
        source[:6, 1, [20, 21], 1] += .5
        source[:6, 1, [20, 21], 2] += .5
        return source

    def test_waiting_selection_uses_exact_better_native_frame_with_lower_hands(self):
        source = self.waiting_source()
        original = source.copy()
        pose, report = select_native_wait_pose(source, 1)
        self.assertEqual(report['source_frame'], 6)
        self.assertTrue(report['changed_from_first_frame'])
        self.assertFalse(report['joint_pose_modified'])
        self.assertLess(report['selected_metrics']['torso_tilt_degrees'], report['baseline_metrics']['torso_tilt_degrees'])
        self.assertLess(report['selected_metrics']['mean_hand_height_above_hips_m'], report['baseline_metrics']['mean_hand_height_above_hips_m'])
        np.testing.assert_array_equal(pose, source[6, 1])
        np.testing.assert_array_equal(source, original)

    def test_waiting_selection_does_not_choose_low_hands_during_a_step(self):
        source = self.waiting_source()
        for frame in range(6, len(source)):
            source[frame, 1, [7, 8, 10, 11], 0] += (frame-5)*.04
        pose, report = select_native_wait_pose(source, 1)
        self.assertEqual(report['eligible_frame_count'], 0)
        self.assertEqual(report['source_frame'], 0)
        np.testing.assert_array_equal(pose, source[0, 1])

    def test_waiting_selection_keeps_first_pose_without_meaningful_improvement(self):
        source = self.waiting_source()[6:].copy()
        source[8, 1, 20, 0] += .001
        source[8, 1, 21, 0] -= .001
        pose, report = select_native_wait_pose(source, 1)
        self.assertGreater(report['eligible_frame_count'], 0)
        self.assertEqual(report['source_frame'], 0)
        np.testing.assert_array_equal(pose, source[0, 1])

    def test_actual_idle_collision_is_archived_and_retried_with_fresh_core_history(self):
        first = NativePairClip(np.repeat(np.stack([self.pose, self.pose+[3, 0, 0]])[None], 30, axis=0), metadata={'model': 'InterGen'})
        later = NativePairClip(np.repeat(np.stack([self.pose, self.pose+[1, 0, 0]])[None], 30, axis=0), metadata={'model': 'InterGen'})
        sources = iter((first, later))
        provider = SimpleNamespace(generate=lambda *args, **kwargs: next(sources), last_raw_archive=None)
        original = self.client.wait
        rejected_requests = []
        def generate(request, cancelled):
            returned = original(request, cancelled)
            if request['actor_ids'] == ['actor_2', 'actor_3'] and not rejected_requests:
                rejected_requests.append(request)
                # One actual emitted elbow sphere overlaps the stationary actor1
                # proxy while the Core root still follows its requested route.
                returned[0].positions[0, 5, CORE_TO_NATIVE[18]] = [0., 1., 0.]
            return returned
        self.client.wait = generate
        beats = [{'id': 'beat-1', 'actor_ids': ['actor_1', 'actor_2'], 'prompt': 'Face each other.', 'seconds': 1},
                 {'id': 'beat-2', 'actor_ids': ['actor_2', 'actor_3'], 'prompt': 'Face each other.', 'seconds': 1}]
        result = self.build(3, beats, provider)
        manifest = json.loads(Path(result.metadata['source_manifest']).read_text())
        attempts = manifest['core_approach_attempts']['beat-2']
        self.assertEqual([item['status'] for item in attempts], ['rejected_idle_collision', 'accepted'])
        self.assertEqual([item['idle_route_padding_m'] for item in attempts], [.25, .40])
        self.assertGreater(attempts[0]['idle_collisions'][0]['worst_penetration_m'], 0.)
        self.assertEqual(attempts[0]['idle_collisions'][0]['context']['stage'], 'core_window')
        self.assertEqual(len(attempts[0]['source_core_archives']), 1)
        self.assertTrue(Path(attempts[0]['source_core_archives'][0]).is_file())
        later_requests = [request for request in self.client.requests if request['actor_ids'] == ['actor_2', 'actor_3']]
        self.assertIn('initial_placements', later_requests[0])
        self.assertIn('initial_placements', later_requests[1])
        self.assertNotIn('history', later_requests[1])
        self.assertTrue(all(request['actor_ids'] == ['actor_2', 'actor_3'] for request in later_requests))

    def test_idle_retry_limit_and_noncollision_failures_do_not_retry(self):
        selected = {'planning_scene': SCENE, 'starts': [{'x': 0, 'z': 0}, {'x': 2, 'z': 0}],
                    'meeting': {'x': 1, 'z': 0, 'yaw_degrees': 0}}
        prior = np.repeat(np.stack([self.pose, self.pose+[2, 0, 0]])[None], 2, axis=0)
        pair = NativePairClip(np.repeat(prior, 2, axis=0), metadata={'model': 'InterGen'})
        for error in (ValueError('Mechanical transition rejected'), RuntimeError('Service authentication failed')):
            attempts = []
            with patch('prompt_scene_builder.build_meetup', side_effect=error) as generate:
                with self.assertRaises(type(error)):
                    generate_later_approach(pair, self.client, SCENE, {}, {}, {}, ['a', 'b'], prior, selected,
                                            seed=42, attempt_reports=attempts)
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(len(attempts), 1)
        error = ValueError('Measured scene-idle-a collision')
        error.idle_collision_report = [{'object_id': 'scene-idle-a', 'collision_frames': 1}]
        attempts = []
        with patch('prompt_scene_builder.build_meetup', side_effect=error) as generate, patch('prompt_scene_builder.select_later_meeting', return_value=selected):
            with self.assertRaisesRegex(ValueError, 'Measured scene-idle-a'):
                generate_later_approach(pair, self.client, SCENE, {}, {}, {}, ['a', 'b'], prior, selected,
                                        seed=42, attempt_reports=attempts)
        self.assertEqual(generate.call_count, 3)
        self.assertEqual([item['idle_route_padding_m'] for item in attempts], [.25, .40, .55])

    def test_idle_collision_checked_even_during_intended_pair_contact(self):
        cast = np.repeat(np.stack([self.pose, self.pose+[.2, 0, 0], self.pose+[.3, 0, 0]])[None], 10, axis=0)
        with self.assertRaisesRegex(ValueError, 'actor_1 and actor_3'):
            check_cast_geometry(cast, SCENE, ('actor_1', 'actor_2', 'actor_3'), contact_pair=('actor_1', 'actor_2'))
        check_cast_geometry(cast[:, :2], SCENE, ('actor_1', 'actor_2'), contact_pair=('actor_1', 'actor_2'))

    def test_large_height_change_rejected_without_fabricated_transition(self):
        source = np.repeat(self.pose[None, None], 4, axis=0)
        target = source.copy(); target[..., 1] -= .8
        with self.assertRaisesRegex(ValueError, 'root-height gap'):
            _bridge(source, target)

    def test_budget_rejected_before_gpu_and_no_truncation(self):
        beats = [{'id': f'beat-{i}', 'actor_ids': ['actor_1'], 'prompt': 'Stand.', 'seconds': 10} for i in range(4)]
        with self.assertRaisesRegex(ValueError, '1000-frame'):
            self.build(beats=beats)
        self.assertEqual(self.client.requests, [])

    def test_bad_core_archived_before_rejection(self):
        original = self.client.wait
        def malformed(*args, **kwargs):
            output = original(*args, **kwargs)
            output[0].native_features = np.zeros((1, 40, 262))
            return output
        self.client.wait = malformed
        with self.assertRaisesRegex(ValueError, 'native history'):
            self.build()
        manifest = json.loads(next((self.root/'sources').glob('*/manifest.json')).read_text())
        self.assertEqual(manifest['status'], 'rejected')
        self.assertEqual(len(manifest['sources']), 1)
        self.assertTrue(Path(manifest['sources'][0]['path']).is_file())

    def test_cancellation_no_source_work(self):
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            self.build(cancelled=lambda: True)
        self.assertEqual(self.client.requests, [])


if __name__ == '__main__':
    unittest.main()
