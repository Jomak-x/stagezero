"""CPU-only contracts for the bounded concurrent prompt route."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cast_performance import CastPerformance
from prompt_group_adapter import independent_starts, build_independent_solos, overlay_concurrent_third
from prompt_scene_builder import PromptSceneBuilder
from prompt_scene_plan import validate_plan


SCENE = {'version': 2, 'name': 'Stage', 'objects': [], 'effects': [], 'lighting': 'neutral'}


def plan(*, pair=False, count=3, seconds=4):
    primary = ['actor_1', 'actor_2'] if pair else ['actor_1']
    others = {'actor_3': 'Actor 3 waves.'} if pair else {
        f'actor_{i}': f'Actor {i} waves.' for i in range(2, count + 1)}
    return {'version': 1, 'title': 'Concurrent motion', 'prompt': 'Perform concurrently.',
            'actor_count': count,
            'actors': [{'id': f'actor_{i}', 'name': f'Person {i}', 'start': None}
                       for i in range(1, count + 1)],
            'meeting': None, 'beats': [{'id': 'beat-1', 'actor_ids': primary,
                'prompt': 'The pair shakes hands.' if pair else 'Actor 1 waves.',
            'seconds': seconds, 'concurrent_solos': others}], 'warnings': []}


def sequence_plan(*, meeting=True):
    raw = plan()
    raw['beats'].append({'id': 'beat-2', 'actor_ids': ['actor_1'],
                         'prompt': 'Actor 1 backflips.', 'seconds': 4,
                         'concurrent_solos': {'actor_2': 'Actor 2 backflips.',
                                              'actor_3': 'Actor 3 backflips.'}})
    if meeting:
        raw['meeting'] = {'x': 0, 'z': 0}
    return validate_plan(raw, SCENE)


def clip(count=3, *, action_start=60, action_frames=120):
    frames = action_start + action_frames
    joints = np.zeros((frames, count, 22, 3), dtype=np.float64)
    for index in range(count):
        joints[:, index, :, 0] = index * 3.0
    ids = [f'actor_{i}' for i in range(1, count + 1)]
    metadata = {'fps': 30, 'frames': frames, 'plan': plan(pair=True),
                'segments': [{'source': 'ardy_core', 'kind': 'approach', 'label': 'Approach',
                              'start_frame': 0, 'end_frame_exclusive': action_start,
                              'frames': action_start},
                             {'source': 'intergen', 'kind': 'paired_action', 'label': 'Pair action',
                              'start_frame': action_start, 'end_frame_exclusive': frames,
                              'frames': action_frames}],
                'segment_activity': [
                    {'start_frame': 0, 'end_frame_exclusive': action_start,
                     'active_actor_ids': ids[:2], 'held_actor_ids': ids[2:],
                     'contact_actor_ids': []},
                    {'start_frame': action_start, 'end_frame_exclusive': frames,
                     'active_actor_ids': ids[:2], 'held_actor_ids': ids[2:],
                     'contact_actor_ids': ids[:2]}]}
    return CastPerformance(ids, joints, metadata=metadata)


class SchemaTests(unittest.TestCase):
    def test_legacy_canonicalization_is_identical_without_optional_field(self):
        raw = plan(count=2)
        raw['beats'][0].pop('concurrent_solos')
        raw['beats'][0]['actor_ids'] = ['actor_1', 'actor_2']
        clean = validate_plan(raw)
        self.assertEqual(clean['beats'][0], raw['beats'][0])

    def test_three_solos_and_pair_plus_third_share_one_duration(self):
        solos = validate_plan(plan())
        self.assertEqual(solos['beats'][0]['concurrent_solos'],
                         {'actor_2': 'Actor 2 waves.', 'actor_3': 'Actor 3 waves.'})
        self.assertEqual(solos['beats'][0]['seconds'], 4)
        self.assertEqual(validate_plan(plan(pair=True))['beats'][0]['seconds'], 4)

    def test_malformed_ids_prompts_duration_contact_and_timing_rejected(self):
        cases = []
        unknown = plan(); unknown['beats'][0]['concurrent_solos'] = {'actor_4': 'Wave.'}; cases.append(unknown)
        overlap = plan(); overlap['beats'][0]['concurrent_solos']['actor_1'] = 'Wave.'; cases.append(overlap)
        missing = plan(); missing['beats'][0]['concurrent_solos'].pop('actor_3'); cases.append(missing)
        blank = plan(); blank['beats'][0]['concurrent_solos']['actor_2'] = ' '; cases.append(blank)
        null = plan(); null['beats'][0]['concurrent_solos'] = None; cases.append(null)
        short = plan(pair=True, seconds=1); cases.append(short)
        long = plan(pair=True, seconds=7.1); cases.append(long)
        contact = plan(); contact['prompt'] = 'All three hug together.'; cases.append(contact)
        primary_contact = plan(); primary_contact['beats'][0]['prompt'] = 'All three hug together.'; cases.append(primary_contact)
        ordered = plan(); ordered['beats'].append({'id': 'beat-2', 'actor_ids': ['actor_1'],
                                                   'prompt': 'Wave.', 'seconds': 2}); cases.append(ordered)
        wrong_pair = plan(pair=True); wrong_pair['beats'][0]['actor_ids'] = ['actor_2', 'actor_3']
        wrong_pair['beats'][0]['concurrent_solos'] = {'actor_1': 'Wave.'}; cases.append(wrong_pair)
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_plan(raw)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_starts_bind_explicit_coordinates_and_enforce_two_metres(self):
        raw = validate_plan(plan())
        raw['actors'][0]['start'] = {'x': 4., 'z': -3.}
        starts = independent_starts(raw, SCENE)
        self.assertEqual((starts['actor_1']['x'], starts['actor_1']['z']), (4., -3.))
        for first in starts.values():
            for second in starts.values():
                if first is not second:
                    self.assertGreaterEqual(np.hypot(first['x']-second['x'], first['z']-second['z']), 2)
        raw['actors'][1]['start'] = {'x': 5., 'z': -3.}
        with self.assertRaisesRegex(ValueError, '2 metres'):
            independent_starts(raw, SCENE)

    def test_independent_route_passes_all_prompts_and_names_to_group_generator(self):
        raw = validate_plan(plan())
        raw['actors'][0]['start_yaw_degrees'] = 90.
        observed = {}
        generated_clip = CastPerformance(['actor_1', 'actor_2', 'actor_3'],
                                         np.zeros((120, 3, 22, 3), dtype=np.float64),
                                         metadata={'fps': 30, 'frames': 120, 'warnings': ['Core source warning']})
        def generate(_client, _scene, **kwargs):
            observed.update(kwargs)
            return {'performance': generated_clip, 'manifest': self.root/'group.json'}
        with patch('independent_group_motion.generate_independent_tracks', side_effect=generate):
            result, _ = build_independent_solos(None, SCENE, raw, seed=7, output_root=self.root)
        self.assertEqual([actor['prompt'] for actor in observed['actors']],
                         ['Actor 1 waves.', 'Actor 2 waves.', 'Actor 3 waves.'])
        self.assertEqual(observed['actors'][0]['yaw_degrees'], 90.)
        self.assertEqual(observed['seconds'], 4)
        self.assertEqual(result.metadata['concurrency_status'], 'accepted_by_geometry_gates')
        self.assertEqual(result.metadata['warnings'], ['Core source warning'])
        self.assertEqual(result.metadata['plan']['beats'][0]['concurrent_solos']['actor_3'], 'Actor 3 waves.')

    def test_sequence_mode_off_rejects_before_placement_or_core(self):
        cases = (sequence_plan(), sequence_plan(meeting=False),
                 validate_plan(dict(plan(), meeting={'x': 0, 'z': 0}), SCENE))
        for raw in cases:
            for mode in ('', 'invalid'):
                with self.subTest(beats=len(raw['beats']), meeting=raw['meeting'], mode=mode), \
                     patch.dict(os.environ, {'STAGEZERO_GROUP_SEQUENCE_MODE': mode}), \
                     patch('group_scene_placement.group_sequence_placement') as placement, \
                     patch('group_scene_sequence.build_group_sequence') as sequence, \
                     patch('independent_group_motion.generate_independent_tracks') as legacy_core:
                    with self.assertRaisesRegex(ValueError, 'experimental Studio'):
                        build_independent_solos(None, SCENE, raw, seed=7, output_root=self.root)
                    placement.assert_not_called()
                    sequence.assert_not_called()
                    legacy_core.assert_not_called()

    def test_fresh_and_continuous_modes_forward_yaw_and_stage_policy(self):
        raw = sequence_plan()
        raw['actors'][0]['start_yaw_degrees'] = 90.
        source = CastPerformance(['actor_1', 'actor_2', 'actor_3'],
                                 np.zeros((120, 3, 22, 3), dtype=np.float64),
                                 metadata={'fps': 30, 'frames': 120, 'warnings': ['Source warning']})
        for mode, fresh in ((' FRESH ', True), ('continuous', False)):
            observed = {}
            def generate(_client, _scene, _plan, starts, targets, **kwargs):
                observed.update(starts=starts, targets=targets, **kwargs)
                return {'performance': source, 'manifest': self.root/'sequence.json'}
            with self.subTest(mode=mode), \
                 patch.dict(os.environ, {'STAGEZERO_GROUP_SEQUENCE_MODE': mode}), \
                 patch('group_scene_sequence.build_group_sequence', side_effect=generate) as sequence, \
                 patch('independent_group_motion.generate_independent_tracks') as legacy_core:
                result, _ = build_independent_solos(None, SCENE, raw, seed=7, output_root=self.root)
                sequence.assert_called_once()
                legacy_core.assert_not_called()
            self.assertEqual(observed['fresh_action_stages'], fresh)
            self.assertEqual(observed['starts']['actor_1']['yaw_degrees'], 90.)
            self.assertEqual(result.metadata['placement']['starts']['actor_1']['yaw_degrees'], 90.)
            self.assertEqual(result.metadata['concurrency_kind'], 'independent_group_sequence')
            self.assertIn('Source warning', result.metadata['warnings'])

    def test_single_beat_without_meeting_stays_on_legacy_route(self):
        raw = validate_plan(plan())
        source = CastPerformance(['actor_1', 'actor_2', 'actor_3'],
                                 np.zeros((120, 3, 22, 3), dtype=np.float64),
                                 metadata={'fps': 30, 'frames': 120, 'warnings': []})
        for mode in ('', 'fresh'):
            with self.subTest(mode=mode), \
                 patch.dict(os.environ, {'STAGEZERO_GROUP_SEQUENCE_MODE': mode}), \
                 patch('independent_group_motion.generate_independent_tracks',
                       return_value={'performance': source}) as legacy_core, \
                 patch('group_scene_placement.group_sequence_placement') as placement, \
                 patch('group_scene_sequence.build_group_sequence') as sequence:
                result, _ = build_independent_solos(None, SCENE, raw, seed=7, output_root=self.root)
                legacy_core.assert_called_once()
                placement.assert_not_called()
                sequence.assert_not_called()
                self.assertEqual(result.metadata['concurrency_kind'], 'independent_solos')

    def test_one_beat_with_meeting_uses_opted_in_sequence_route(self):
        raw = validate_plan(dict(plan(), meeting={'x': 0, 'z': 0}), SCENE)
        source = CastPerformance(['actor_1', 'actor_2', 'actor_3'],
                                 np.zeros((120, 3, 22, 3), dtype=np.float64),
                                 metadata={'fps': 30, 'frames': 120, 'warnings': []})
        with patch.dict(os.environ, {'STAGEZERO_GROUP_SEQUENCE_MODE': 'continuous'}), \
             patch('group_scene_sequence.build_group_sequence',
                   return_value={'performance': source}) as sequence, \
             patch('independent_group_motion.generate_independent_tracks') as legacy_core:
            result, _ = build_independent_solos(None, SCENE, raw, seed=7, output_root=self.root)
        sequence.assert_called_once()
        self.assertFalse(sequence.call_args.kwargs['fresh_action_stages'])
        legacy_core.assert_not_called()
        self.assertEqual(result.metadata['concurrency_kind'], 'independent_group_sequence')

    def test_builder_dispatches_concurrent_solos_without_legacy_staging(self):
        raw = validate_plan(plan())
        expected = CastPerformance(['actor_1', 'actor_2', 'actor_3'],
                                   np.zeros((120, 3, 22, 3), dtype=np.float64),
                                   metadata={'concurrency_status': 'accepted_by_geometry_gates'})
        planner = SimpleNamespace(plan=lambda *_args, **_kwargs: raw)
        builder = PromptSceneBuilder(raw['prompt'], planner, None, object(), self.root, seed=3)
        with patch('prompt_group_adapter.build_independent_solos',
                   return_value=(expected, {'manifest': self.root/'group.json'})) as routed, \
             patch('prompt_scene_plan.auto_place') as legacy_staging:
            actual = builder(SCENE)
        np.testing.assert_array_equal(actual.joints, expected.joints)
        routed.assert_called_once()
        legacy_staging.assert_not_called()
        manifest = next(self.root.glob('scene-*/manifest.json'))
        self.assertEqual(json.loads(manifest.read_text())['concurrency_status'], 'accepted_by_geometry_gates')
        self.assertIn('wall_seconds', actual.metadata)
        self.assertEqual(actual.metadata['stage_timings'][-1]['stage'], 'independent_concurrent_core')

    def test_pair_overlay_uses_resolved_action_frames_and_reports_fallback(self):
        raw = validate_plan(plan(pair=True))
        baseline = clip(action_start=60, action_frames=120)
        baseline_metadata = baseline.metadata
        baseline_metadata['warnings'] = ['Native pair warning']
        baseline = CastPerformance(baseline.actor_ids, baseline.joints, metadata=baseline_metadata)
        observed = {}
        def overlay(_client, source, **kwargs):
            observed.update(kwargs)
            return {'performance': baseline, 'project_bytes': source,
                    'manifest': self.root/'overlay.json', 'fallback': True,
                    'reason': 'geometry rejection'}
        with patch('independent_group_motion.overlay_third_track', side_effect=overlay):
            result, generated = overlay_concurrent_third(None, baseline, SCENE, raw,
                                                          seed=3, output_root=self.root)
        self.assertEqual((observed['start_frame'], observed['end_frame']), (60, 180))
        self.assertEqual(observed['prompt'], 'Actor 3 waves.')
        self.assertTrue(generated['fallback'])
        self.assertIn('not achieved', result.metadata['concurrency_fallback'])
        self.assertIn('Native pair warning', result.metadata['warnings'])
        self.assertIn(result.metadata['concurrency_fallback'], result.metadata['plan']['warnings'])
        self.assertEqual(result.metadata['concurrency_status'], 'fallback_original_pair_only')
        np.testing.assert_array_equal(result.joints, baseline.joints)

    def test_pair_adapter_rejects_eight_second_interval_at_its_own_boundary(self):
        raw = plan(pair=True, seconds=8)
        baseline = clip(action_start=60, action_frames=240)
        with self.assertRaisesRegex(ValueError, '2–7 second'):
            overlay_concurrent_third(None, baseline, SCENE, raw, seed=3, output_root=self.root)


if __name__ == '__main__':
    unittest.main()
