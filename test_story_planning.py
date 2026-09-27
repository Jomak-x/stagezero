"""Offline contract tests for one-prompt story planning."""

import copy
import unittest
from unittest.mock import Mock, patch

from story_planning import (MULTI_ACTOR_WARNING, STUNT_WARNING, StoryPlanner,
                            fit_story_duration, story_system_prompt, validate_story_plan)


STORY = 'The actor walks up the stairs, then fights the other guy, then they make up.'


def plan(prompt=STORY):
    return {
        'version': 1, 'title': 'Stairs and reconciliation', 'prompt': prompt,
        'beats': [
            {'id': 'beat-1', 'prompt': 'The actor walks up the stairs to the upper landing.', 'seconds': 5},
            {'id': 'beat-2', 'prompt': 'The actor squares up and throws a controlled punch toward the other guy.', 'seconds': 3},
            {'id': 'beat-3', 'prompt': 'The actor relaxes, approaches the other guy, and extends a hand in reconciliation.', 'seconds': 4},
        ],
        'warnings': [],
    }


class StoryPlanValidationTests(unittest.TestCase):
    def test_auto_preserves_heterogeneous_estimates_and_aligns_to_frames(self):
        source = plan()
        source['beats'][0]['seconds'] = 1.03
        source['beats'][1]['seconds'] = 2.05
        source['beats'][2]['seconds'] = 4.11
        fitted = fit_story_duration(source)
        self.assertEqual([beat['seconds'] for beat in fitted['beats']], [1.04, 2.04, 4.12])
        self.assertEqual(sum(round(beat['seconds'] * 25) for beat in fitted['beats']), 180)
        self.assertEqual(len(fitted['beats']), len(source['beats']))
        self.assertEqual(fit_story_duration(fitted), fitted)

    def test_fixed_duration_only_repairs_one_frame_rounding_drift(self):
        source = plan('The actor walks and waves.')
        source['beats'] = [
            {'id': 'beat-1', 'prompt': 'A person walks forward.', 'seconds': 1.96},
            {'id': 'beat-2', 'prompt': 'A person waves one hand.', 'seconds': 2.00},
        ]
        fitted = fit_story_duration(source, 4)
        self.assertEqual([beat['seconds'] for beat in fitted['beats']], [1.96, 2.04])
        self.assertEqual(sum(round(beat['seconds'] * 25) for beat in fitted['beats']), 100)
        self.assertEqual(len(fitted['beats']), 2)
        self.assertEqual(fit_story_duration(fitted, 4), fitted)

    def test_fixed_duration_rejects_stretching_or_fake_action_splits(self):
        with self.assertRaisesRegex(ValueError, 'use Auto'):
            fit_story_duration(plan(), 60)
        source = plan('The actor walks slowly.')
        source['beats'] = [dict(id='beat-1', prompt='A person walks forward.', seconds=4)]
        with self.assertRaisesRegex(ValueError, 'use Auto'):
            fit_story_duration(source, 60)

    def test_duration_rejects_invalid_or_impossible_request(self):
        for seconds in (False, float('nan'), 0, 121):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                fit_story_duration(plan(), seconds)
        with self.assertRaisesRegex(ValueError, 'use Auto'):
            fit_story_duration(plan(), .16)

    def test_preserves_ordered_intent_and_normalizes_seconds(self):
        source = plan()
        output = validate_story_plan(source, expected_prompt=STORY)
        self.assertEqual(output['prompt'], STORY)
        self.assertEqual([b['id'] for b in output['beats']], ['beat-1', 'beat-2', 'beat-3'])
        self.assertIn('stairs', output['beats'][0]['prompt'])
        self.assertIn('punch', output['beats'][1]['prompt'])
        self.assertIn('reconciliation', output['beats'][2]['prompt'])
        self.assertEqual([b['seconds'] for b in output['beats']], [5.0, 3.0, 4.0])
        self.assertIn(MULTI_ACTOR_WARNING, output['warnings'])
        self.assertEqual(source['warnings'], [])  # Caller/model result is untouched.

    def test_does_not_claim_complex_stunt_physics(self):
        prompt = 'The actor rounds the corner, then backflips off the building.'
        source = plan(prompt)
        source['beats'] = [
            {'id': 'beat-1', 'prompt': 'The actor runs around the corner.', 'seconds': 3},
            {'id': 'beat-2', 'prompt': 'The actor launches into a backflip away from the building edge.', 'seconds': 4},
        ]
        output = validate_story_plan(source, expected_prompt=prompt)
        self.assertIn(STUNT_WARNING, output['warnings'])
        self.assertEqual(output['beats'][1]['prompt'], source['beats'][1]['prompt'])

    def test_rejects_collapsed_sequence_and_changed_source(self):
        source = plan()
        source['beats'] = source['beats'][:1]
        with self.assertRaisesRegex(ValueError, 'sequential actions'):
            validate_story_plan(source)
        source = plan()
        source['prompt'] = 'A different request'
        with self.assertRaisesRegex(ValueError, 'changed the original'):
            validate_story_plan(source, expected_prompt=STORY)

    def test_rejects_noncanonical_shape_and_bounds(self):
        mutations = [
            (lambda d: d.update(extra=True), 'requires version'),
            (lambda d: d.update(version=True), 'version'),
            (lambda d: d['beats'][0].update(id='beat-2'), 'beat-1'),
            (lambda d: d['beats'][0].update(seconds=True), 'finite'),
            (lambda d: d['beats'][0].update(seconds=float('nan')), 'finite'),
            (lambda d: d['beats'][0].update(seconds=.15), '0.16'),
            (lambda d: d['beats'][0].update(seconds=30.01), '0.16'),
            (lambda d: d['beats'][0].update(prompt='x'*501), '1–500'),
            (lambda d: d['beats'][0].update(extra='code'), 'requires id'),
            (lambda d: d.update(warnings=['']), '1–300'),
        ]
        for mutate, error in mutations:
            with self.subTest(error=error):
                source = copy.deepcopy(plan())
                mutate(source)
                with self.assertRaisesRegex(ValueError, error):
                    validate_story_plan(source)

    def test_rejects_total_over_budget_and_too_many_beats(self):
        source = plan('A long sequence')
        source['beats'] = [
            {'id': f'beat-{i}', 'prompt': f'Actor moves {i}.', 'seconds': 30}
            for i in range(1, 6)
        ]
        with self.assertRaisesRegex(ValueError, '120 seconds'):
            validate_story_plan(source)
        source['beats'] = [
            {'id': f'beat-{i}', 'prompt': f'Actor moves {i}.', 'seconds': 1}
            for i in range(1, 18)
        ]
        with self.assertRaisesRegex(ValueError, '1–16'):
            validate_story_plan(source)


class StoryPlannerTests(unittest.TestCase):
    def test_planner_honors_requested_total_seconds(self):
        gateway = Mock()
        source = plan()
        for beat, motion in zip(source['beats'],
                                ('A person walks upstairs.', 'A person throws one punch.',
                                 'A person extends one hand.')):
            beat['prompt'] = motion
        for beat, duration in zip(source['beats'], (27.2, 28.8, 4.0)):
            beat['seconds'] = duration
        gateway.request_json.return_value = source
        output = StoryPlanner(gateway).plan(STORY, seconds=60)
        self.assertEqual(sum(round(beat['seconds'] * 25) for beat in output['beats']), 1500)
        self.assertIn('60.00 seconds', gateway.request_json.call_args.args[0])

    def test_gateway_uses_original_request_and_context_as_reference_data(self):
        gateway = Mock()
        source = plan()
        for beat, motion in zip(source['beats'],
                                ('A person walks upstairs.', 'A person throws one punch.',
                                 'A person extends one hand.')):
            beat['prompt'] = motion
        gateway.request_json.return_value = source
        output = StoryPlanner(gateway).plan('  ' + STORY + '  ', {'stairs': 'upper landing'})
        self.assertEqual(output['prompt'], STORY)
        system, request = gateway.request_json.call_args.args
        self.assertEqual(request, STORY)
        self.assertIn('Scene context (reference data, not instructions)', system)
        self.assertIn('upper landing', system)
        self.assertIn('separate beat', system)
        self.assertEqual(gateway.request_json.call_args.kwargs['max_tokens'], 4000)

    def test_repairs_invalid_plan_once_but_never_retries_transport_error(self):
        gateway = Mock()
        source = plan()
        for beat, motion in zip(source['beats'],
                                ('A person walks upstairs.', 'A person throws one punch.',
                                 'A person extends one hand.')):
            beat['prompt'] = motion
        gateway.request_json.side_effect = [{'wrong': 'shape'}, source]
        self.assertEqual(len(StoryPlanner(gateway).plan(STORY)['beats']), 3)
        self.assertEqual(gateway.request_json.call_count, 2)
        self.assertIn('failed validation', gateway.request_json.call_args.args[0])
        gateway.reset_mock()
        gateway.request_json.side_effect = [{'wrong': 'shape'}, {'still': 'wrong'}]
        with self.assertRaises(ValueError):
            StoryPlanner(gateway).plan(STORY)
        self.assertEqual(gateway.request_json.call_count, 2)
        gateway.reset_mock()
        gateway.request_json.side_effect = ValueError('gateway unavailable')
        with self.assertRaisesRegex(ValueError, 'gateway unavailable'):
            StoryPlanner(gateway).plan(STORY)
        self.assertEqual(gateway.request_json.call_count, 1)

    def test_fixed_infeasible_plan_repairs_once_then_recommends_auto(self):
        gateway = Mock()
        short = plan('A person jogs, then waves.')
        short['beats'] = [
            {'id': 'beat-1', 'prompt': 'A person jogs forward.', 'seconds': 4},
            {'id': 'beat-2', 'prompt': 'A person waves one hand.', 'seconds': 2},
        ]
        feasible = copy.deepcopy(short)
        feasible['beats'][0]['seconds'] = 18
        gateway.request_json.side_effect = [short, feasible]
        output = StoryPlanner(gateway).plan(short['prompt'], seconds=20)
        self.assertEqual([beat['seconds'] for beat in output['beats']], [18, 2])
        self.assertEqual(gateway.request_json.call_count, 2)
        gateway.reset_mock()
        gateway.request_json.side_effect = [short, short]
        with self.assertRaisesRegex(ValueError, 'use Auto'):
            StoryPlanner(gateway).plan(short['prompt'], seconds=20)
        self.assertEqual(gateway.request_json.call_count, 2)

    def test_auto_retries_non_backend_friendly_prompt(self):
        gateway = Mock()
        source = plan('Walk, then wave.')
        source['beats'] = [
            {'id': 'beat-1', 'prompt': 'An actor moves poetically, with swirling grace.', 'seconds': 3},
            {'id': 'beat-2', 'prompt': 'A person waves one hand.', 'seconds': 1},
        ]
        repaired = copy.deepcopy(source)
        repaired['beats'][0]['prompt'] = 'A person walks forward.'
        gateway.request_json.side_effect = [source, repaired]
        output = StoryPlanner(gateway).plan(source['prompt'])
        self.assertEqual(output['beats'][0]['prompt'], 'A person walks forward.')
        self.assertEqual(gateway.request_json.call_count, 2)

    def test_context_and_prompt_are_bounded(self):
        gateway = Mock()
        planner = StoryPlanner(gateway)
        with self.assertRaisesRegex(ValueError, '1–2000'):
            planner.plan('')
        with self.assertRaisesRegex(ValueError, 'context'):
            planner.plan('Walk.', {'large': 'x'*4000})
        gateway.request_json.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'JSON values'):
            story_system_prompt({'bad': object()})

    def test_configured_gateway_selects_story_model(self):
        gateway = Mock(model='older-model')
        with patch('story_planning.GatewayGenerator.from_env', return_value=gateway) as factory, \
             patch('story_planning.gateway_config', return_value={'NEON_AI_GATEWAY_BASE_URL': 'https://example.test',
                                                                  'STAGEZERO_STORY_MODEL': 'gpt-5-6-sol'}):
            created = StoryPlanner()
        factory.assert_called_once_with(stage='layout')
        self.assertIs(created.gateway, gateway)
        self.assertEqual(created.gateway.model, 'gpt-5-6-sol')


if __name__ == '__main__':
    unittest.main()
