"""CPU-only semantic gateway and real geometry staging contracts."""
import math
import unittest
from unittest.mock import Mock, patch

from prompt_scene_plan import ScenePromptPlanner, validate_plan, auto_place, THREE_CONTACT_WARNING
from scene_objects import make_object

SCENE = {'version': 2, 'name': 'Studio', 'objects': [], 'effects': [], 'lighting': 'neutral'}


def document(count=2):
    ids = [f'actor_{i + 1}' for i in range(count)]
    groups = [ids[:2]] if count < 3 else [ids[:2], ids[1:]]
    return {'version': 1, 'title': 'Greeting', 'prompt': 'People greet each other.',
            'actor_count': count, 'actors': [{'id': a, 'name': a, 'start': None} for a in ids],
            'meeting': None, 'beats': [{'id': f'beat-{i + 1}', 'actor_ids': group,
            'prompt': 'Two people greet each other.' if len(group) == 2 else 'A person waves.',
            'seconds': 3.1} for i, group in enumerate(groups)], 'warnings': []}


class PlanTests(unittest.TestCase):
    def test_one_two_three_and_duration_quantization(self):
        for count in (1, 2, 3):
            raw = document(count)
            clean = validate_plan(raw, SCENE, raw['prompt'])
            self.assertEqual(clean['actor_count'], count)
            self.assertEqual(clean['beats'][0]['seconds'], 4 if count == 1 else 3.1)
            clean['actors'][0]['name'] = 'Changed'
            self.assertEqual(raw['actors'][0]['name'], 'actor_1')

    def test_rejects_unknown_ids_missing_actors_and_three_body_beat(self):
        for ids in (['actor_1', 'actor_1'], ['actor_1', 'invented'], ['actor_1', 'actor_2', 'actor_3'], [True]):
            raw = document(3)
            raw['beats'][0]['actor_ids'] = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validate_plan(raw)
        raw = document(3)
        raw['beats'] = raw['beats'][:1]
        with self.assertRaisesRegex(ValueError, 'Every actor'):
            validate_plan(raw)

    def test_bounds_and_no_extra_fields(self):
        for key, value in (('actor_count', True), ('version', True), ('actor_count', 4), ('beats', []), ('warnings', 'none')):
            raw = document(); raw[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_plan(raw)
        raw = document(); raw['script'] = 'execute'
        with self.assertRaises(ValueError):
            validate_plan(raw)
        for seconds in (True, float('nan'), float('inf'), .5, 7.1):
            raw = document(); raw['beats'][0]['seconds'] = seconds
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                validate_plan(raw)
        raw = document(1)
        raw['beats'] = [dict(raw['beats'][0], id=f'beat-{i+1}', seconds=8) for i in range(4)]
        with self.assertRaisesRegex(ValueError, '30 seconds'):
            validate_plan(raw)

    def test_stable_ids_order_and_exact_request(self):
        raw = document(); raw['actors'].reverse()
        with self.assertRaisesRegex(ValueError, 'stable IDs'):
            validate_plan(raw)
        with self.assertRaisesRegex(ValueError, 'original request'):
            validate_plan(document(), expected_prompt='Changed')
        raw = document(); raw['beats'][0]['id'] = 'beat-2'
        with self.assertRaisesRegex(ValueError, 'sequential'):
            validate_plan(raw)

    def test_coordinate_bounds_and_actual_target(self):
        for start in ({'x': 25, 'z': 0}, {'x': float('nan'), 'z': 0}, {'x': 0}, {'x': 0, 'z': True}):
            raw = document(); raw['actors'][0]['start'] = start
            with self.subTest(start=start), self.assertRaises(ValueError):
                validate_plan(raw)
        raw = document(); raw['meeting'] = {'target_id': 'imaginary'}
        with self.assertRaisesRegex(ValueError, 'actual scene'):
            validate_plan(raw, SCENE)

    def test_start_yaw_is_independent_of_position_and_backward_compatible(self):
        raw = document(3)
        raw['actors'][0]['start'] = {'x': -4, 'z': 0}
        raw['actors'][0]['start_yaw_degrees'] = 90
        raw['actors'][1]['start_yaw_degrees'] = 0
        raw['actors'][2]['start_yaw_degrees'] = None
        clean = validate_plan(raw)
        self.assertEqual(clean['actors'][0]['start'], {'x': -4., 'z': 0.})
        self.assertEqual(clean['actors'][0]['start_yaw_degrees'], 90.)
        self.assertIsNone(clean['actors'][1]['start'])
        self.assertEqual(clean['actors'][1]['start_yaw_degrees'], 0.)
        self.assertIsNone(clean['actors'][2]['start_yaw_degrees'])
        self.assertNotIn('start_yaw_degrees', validate_plan(document(1))['actors'][0])

    def test_start_yaw_rejects_invalid_values_and_nested_fields(self):
        for yaw in (True, '90', float('nan'), float('inf'), 180.1, -180.1):
            raw = document(1)
            raw['actors'][0]['start_yaw_degrees'] = yaw
            with self.subTest(yaw=yaw), self.assertRaisesRegex(ValueError, 'start yaw'):
                validate_plan(raw)
        raw = document(1)
        raw['actors'][0]['start'] = {'x': 0, 'z': 0, 'yaw_degrees': 90}
        with self.assertRaisesRegex(ValueError, 'x and z only'):
            validate_plan(raw)

    def test_three_contact_warning_discloses_serial_adaptation(self):
        raw = document(3); raw['prompt'] = 'All three hug together.'
        self.assertIn(THREE_CONTACT_WARNING, validate_plan(raw)['warnings'])
        raw = document(2); raw['prompt'] = 'All three hug together.'
        with self.assertRaisesRegex(ValueError, 'all three'):
            validate_plan(raw)


class GatewayTests(unittest.TestCase):
    def gateway(self, response):
        return Mock(model='test-model', request_json=Mock(return_value=response))

    def test_raw_plan_timing_and_model_are_retained(self):
        raw = document(1); gateway = self.gateway(raw)
        result = ScenePromptPlanner(gateway, clock=iter([10., 12.5]).__next__).plan(raw['prompt'], SCENE)
        self.assertEqual(result['planning_seconds'], 2.5)
        self.assertEqual(result['planning_attempts'], 1)
        self.assertFalse(result['planner_cache_hit'])
        self.assertEqual(result['raw_plan']['beats'][0]['seconds'], 3.1)
        self.assertEqual(result['beats'][0]['seconds'], 4)
        self.assertEqual(validate_plan(result)['actor_count'], 1)
        self.assertIn('EVERY major action and the ending', gateway.request_json.call_args.args[0])

    def test_one_repair_without_silent_fallback(self):
        raw = document(); gateway = self.gateway(raw)
        gateway.request_json.side_effect = [{'wrong': 1}, raw]
        planner = ScenePromptPlanner(gateway)
        result = planner.plan(raw['prompt'], SCENE)
        self.assertEqual(result['planning_attempts'], 2)
        self.assertEqual(len(planner.raw_plans), 2)
        gateway.request_json.side_effect = [{'wrong': 1}, {'still_wrong': 2}]
        with self.assertRaises(ValueError):
            planner.plan(raw['prompt'], dict(SCENE, lighting='warm'))
        self.assertEqual(gateway.request_json.call_count, 4)

    def test_transport_failure_not_retried(self):
        gateway = self.gateway(None); gateway.request_json.side_effect = ValueError('Connection failed')
        with self.assertRaisesRegex(ValueError, 'Connection failed'):
            ScenePromptPlanner(gateway).plan('Wave.', SCENE)
        self.assertEqual(gateway.request_json.call_count, 1)

    def test_cancel_before_and_after_call(self):
        raw = document(); gateway = self.gateway(raw)
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            ScenePromptPlanner(gateway).plan(raw['prompt'], SCENE, cancelled=lambda: True)
        gateway.request_json.assert_not_called()
        checks = iter([False, False, True])
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            ScenePromptPlanner(gateway).plan(raw['prompt'], SCENE, cancelled=lambda: next(checks))
        self.assertEqual(gateway.request_json.call_count, 1)

    def test_injected_audit_fields_are_not_gateway_intent(self):
        raw = document(); raw['planning_seconds'] = 0
        gateway = self.gateway(raw)
        with self.assertRaises(ValueError):
            ScenePromptPlanner(gateway).plan(raw['prompt'], SCENE)
        self.assertEqual(gateway.request_json.call_count, 2)

    def test_model_override(self):
        raw = document(); gateway = self.gateway(raw)
        result = ScenePromptPlanner(gateway, model='chosen-model').plan(raw['prompt'], SCENE)
        self.assertEqual(result['planner_model'], 'chosen-model')

    def test_gateway_requests_per_actor_world_facing_with_scene_sizes(self):
        raw = document(2)
        raw['actors'][0]['start'] = {'x': -3, 'z': 0}
        raw['actors'][0]['start_yaw_degrees'] = 90
        raw['actors'][1]['start_yaw_degrees'] = -90
        crate = make_object('crate', 0)
        gateway = self.gateway(raw)
        result = ScenePromptPlanner(gateway).plan(raw['prompt'], dict(SCENE, objects=[crate]))
        self.assertEqual([actor['start_yaw_degrees'] for actor in result['actors']], [90., -90.])
        instruction = gateway.request_json.call_args.args[0]
        self.assertIn('initial world position and facing independently', instruction)
        self.assertIn('0 faces +Z, 90 faces +X', instruction)
        self.assertIn('direction may be supplied even when start is null', instruction)
        self.assertIn('"size":', instruction)


class CacheTests(unittest.TestCase):
    def setUp(self):
        def answer(_system, prompt, **_kwargs):
            raw = document()
            raw['prompt'] = prompt
            return raw
        self.gateway = Mock(model='model-a', request_json=Mock(side_effect=answer))
        self.planner = ScenePromptPlanner(self.gateway)

    def test_cache_skips_gateway_and_defensively_copies_both_views(self):
        first = self.planner.plan('Greet.', SCENE)
        self.assertFalse(first['planner_cache_hit'])
        first['actors'][0]['name'] = 'caller mutation'
        first['raw_plan']['actors'][0]['name'] = 'raw caller mutation'
        hit = self.planner.plan('Greet.', SCENE)
        self.assertTrue(hit['planner_cache_hit'])
        self.assertEqual(hit['planning_seconds'], 0.)
        self.assertEqual(hit['actors'][0]['name'], 'actor_1')
        self.assertEqual(hit['raw_plan']['actors'][0]['name'], 'actor_1')
        self.assertEqual(self.planner.raw_plans, [hit['raw_plan']])
        hit['actors'][0]['name'] = 'cached result mutation'
        self.planner.raw_plans[0]['actors'][0]['name'] = 'audit mutation'
        next_hit = self.planner.plan('Greet.', SCENE)
        self.assertEqual(next_hit['actors'][0]['name'], 'actor_1')
        self.assertEqual(next_hit['raw_plan']['actors'][0]['name'], 'actor_1')
        self.assertEqual(self.gateway.request_json.call_count, 1)

    def test_exact_prompt_full_scene_and_model_each_invalidate(self):
        self.planner.plan('Greet.', SCENE)
        self.planner.plan(' Greet. ', SCENE)
        self.planner.plan('Wave.', SCENE)
        self.planner.plan('Greet.', dict(SCENE, lighting='warm'))
        crate = make_object('crate', 0)
        self.planner.plan('Greet.', dict(SCENE, objects=[crate]))
        crate['size'][0] += .1
        self.planner.plan('Greet.', dict(SCENE, objects=[crate]))
        self.gateway.model = 'model-b'
        self.planner.plan('Greet.', SCENE)
        self.assertEqual(self.gateway.request_json.call_count, 7)
        self.assertTrue(self.planner.plan('Greet.', SCENE)['planner_cache_hit'])

    def test_cache_is_per_instance(self):
        self.planner.plan('Greet.', SCENE)
        other = ScenePromptPlanner(self.gateway)
        self.assertFalse(other.plan('Greet.', SCENE)['planner_cache_hit'])
        self.assertEqual(self.gateway.request_json.call_count, 2)

    def test_lru_keeps_recent_entries_with_sixteen_entry_limit(self):
        for index in range(16):
            self.planner.plan(f'Greeting {index}.', SCENE)
        self.assertTrue(self.planner.plan('Greeting 0.', SCENE)['planner_cache_hit'])
        self.planner.plan('Greeting 16.', SCENE)
        self.assertTrue(self.planner.plan('Greeting 0.', SCENE)['planner_cache_hit'])
        self.assertFalse(self.planner.plan('Greeting 1.', SCENE)['planner_cache_hit'])
        self.assertEqual(self.gateway.request_json.call_count, 18)

    def test_failed_plans_are_not_cached(self):
        raw = document(); raw['prompt'] = 'Greet.'
        self.gateway.request_json.side_effect = [{'wrong': True}, {'wrong': True}, raw]
        with self.assertRaises(ValueError):
            self.planner.plan('Greet.', SCENE)
        self.assertFalse(self.planner.plan('Greet.', SCENE)['planner_cache_hit'])
        self.assertTrue(self.planner.plan('Greet.', SCENE)['planner_cache_hit'])
        self.assertEqual(self.gateway.request_json.call_count, 3)

    def test_cancellation_is_checked_before_cached_return(self):
        self.planner.plan('Greet.', SCENE)
        checks = iter([False, True])
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            self.planner.plan('Greet.', SCENE, cancelled=lambda: next(checks))
        self.assertEqual(self.gateway.request_json.call_count, 1)
        self.assertTrue(self.planner.plan('Greet.', SCENE)['planner_cache_hit'])

    def test_cancelled_success_is_not_cached(self):
        checks = iter([False, False, False, True])
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            self.planner.plan('Greet.', SCENE, cancelled=lambda: next(checks))
        self.assertFalse(self.planner.plan('Greet.', SCENE)['planner_cache_hit'])
        self.assertEqual(self.gateway.request_json.call_count, 2)

    def test_initial_approach_instruction_preserves_later_movement(self):
        raw = document()
        raw['prompt'] = 'They walk in, shake hands, then walk away together.'
        raw['beats'] = [dict(raw['beats'][0], prompt='Two people shake hands.'),
                        dict(raw['beats'][0], id='beat-2', prompt='Two people walk away together.')]
        self.gateway.request_json.side_effect = None
        self.gateway.request_json.return_value = raw
        result = self.planner.plan(raw['prompt'], SCENE)
        self.assertEqual([beat['prompt'] for beat in result['beats']],
                         ['Two people shake hands.', 'Two people walk away together.'])
        instruction = self.gateway.request_json.call_args.args[0]
        self.assertIn('Core automatically generates travel', instruction)
        self.assertIn('before the first interaction', instruction)
        self.assertIn('Later movement between interaction', instruction)


    def test_paired_fall_contract_keeps_roles_and_every_requested_action(self):
        raw = document()
        raw['prompt'] = ('They dance; actor 1 trips and falls, actor 2 helps actor 1 stand, '
                         'then they hug and laugh together.')
        actions = [
            ('Two people dance together.', 4),
            ('Person 1 trips and falls while person 2 watches and prepares to help.', 3),
            ('Person 2 helps person 1 stand up.', 6),
            ('The two people hug and laugh together.', 5),
        ]
        raw['beats'] = [{'id': f'beat-{i+1}', 'actor_ids': ['actor_1', 'actor_2'],
                         'prompt': action, 'seconds': seconds}
                        for i, (action, seconds) in enumerate(actions)]
        self.gateway.request_json.side_effect = None
        self.gateway.request_json.return_value = raw
        result = self.planner.plan(raw['prompt'], SCENE)
        self.assertEqual(result['beats'], raw['beats'])
        self.assertEqual(result['raw_plan']['prompt'], raw['prompt'])
        instruction = self.gateway.request_json.call_args.args[0]
        self.assertIn('context-dependent fall or recovery', instruction)
        self.assertIn('Include both actor IDs', instruction)
        self.assertIn('partner watching or preparing to help', instruction)
        self.assertIn('Do not replace the fall with a catch', instruction)
        self.assertIn('falls with no involved partner remain solo beats', instruction)


class StagingTests(unittest.TestCase):
    def test_all_cast_counts_have_separated_starts_and_routes(self):
        for count in (1, 2, 3):
            placed = auto_place(document(count), SCENE)
            self.assertEqual(len(placed['starts']), count)
            points = [(v['x'], v['z']) for v in placed['starts'].values()]
            for i, a in enumerate(points):
                for b in points[i+1:]:
                    self.assertGreaterEqual(math.dist(a, b), 1.5)
            self.assertEqual(set(placed['starts']), set(placed['routes']))
            self.assertTrue(all(-180 <= start['yaw_degrees'] <= 180 for start in placed['starts'].values()))
            self.assertFalse(placed['physical_contact_verified'])

    def test_explicit_and_yaw_only_directions_survive_staging_for_three_actors(self):
        raw = document(3)
        raw['meeting'] = {'x': 0, 'z': 0}
        raw['actors'][0].update(start={'x': -4, 'z': 0}, start_yaw_degrees=0)
        raw['actors'][1].update(start={'x': 4, 'z': 0}, start_yaw_degrees=-90)
        raw['actors'][2]['start_yaw_degrees'] = 180
        placed = auto_place(raw, SCENE)
        self.assertEqual(placed['starts']['actor_1'], {'x': -4., 'z': 0., 'yaw_degrees': 0.})
        self.assertEqual(placed['starts']['actor_2'], {'x': 4., 'z': 0., 'yaw_degrees': -90.})
        self.assertEqual(placed['starts']['actor_3']['yaw_degrees'], 180.)
        self.assertEqual(placed['starts']['actor_3']['x'], 0.)

    def test_missing_direction_follows_first_route_then_meeting(self):
        raw = document(1)
        raw['actors'][0]['start'] = {'x': -4, 'z': -3}
        raw['meeting'] = {'x': 0, 'z': 0}
        def detour(start, end, _obstacles, _radius):
            return [start, (start[0], end[1]), end]
        with patch('interaction_planner._path', side_effect=detour):
            placed = auto_place(raw, SCENE)
        route = placed['routes']['actor_1']
        start = placed['starts']['actor_1']
        first = next(point for point in route[1:] if math.dist(point, [start['x'], start['z']]) > 1e-9)
        expected = math.degrees(math.atan2(first[0]-start['x'], first[1]-start['z']))
        self.assertAlmostEqual(start['yaw_degrees'], expected)
        self.assertEqual(start['yaw_degrees'], 0.)
        raw['actors'][0]['start'] = {'x': 0, 'z': 0}
        self.assertEqual(auto_place(raw, SCENE)['starts']['actor_1']['yaw_degrees'], 0.)

    def test_obstacle_is_avoided_and_target_resolved_beside_it(self):
        crate = make_object('crate', 0)
        scene = dict(SCENE, objects=[crate])
        raw = document(3); raw['meeting'] = {'target_id': crate['id']}
        placed = auto_place(raw, scene)
        self.assertEqual(placed['target_id'], crate['id'])
        self.assertNotEqual([placed['meeting']['x'], placed['meeting']['z']], [crate['position'][0], crate['position'][2]])

    def test_attached_scene_target_resolves_to_existing_owner(self):
        crate = make_object('crate', 0)
        target = {'id': 'greeting-point', 'name': 'Greeting point', 'object_id': crate['id'],
                  'kind': 'landing', 'local_position': [0., 0., 0.]}
        scene = dict(SCENE, version=3, objects=[crate], assets=[], targets=[target])
        raw = document(3); raw['meeting'] = {'target_id': target['id']}
        self.assertEqual(auto_place(raw, scene)['target_id'], target['id'])

    def test_explicit_invalid_start_or_meeting_is_not_silently_moved(self):
        crate = make_object('crate', 0); scene = dict(SCENE, objects=[crate])
        point = {'x': crate['position'][0], 'z': crate['position'][2]}
        raw = document(); raw['actors'][0]['start'] = point
        with self.assertRaisesRegex(ValueError, 'Desired start'):
            auto_place(raw, scene)
        raw = document(); raw['meeting'] = point
        with self.assertRaisesRegex(ValueError, 'No bounded'):
            auto_place(raw, scene)

    def test_explicit_clear_anchors_preserved(self):
        raw = document(); raw['actors'][0]['start'] = {'x': -4, 'z': -3}
        raw['meeting'] = {'x': 0, 'z': 0}
        result = auto_place(raw, SCENE)
        self.assertEqual((result['starts']['actor_1']['x'], result['starts']['actor_1']['z']), (-4., -3.))
        self.assertEqual(result['meeting']['x'], 0)

    def test_authored_floor_bounds_all_staging(self):
        floor = make_object('platform', 0)
        floor.update(position=[0., -.05, 0.], size=[12., .1, 12.])
        result = auto_place(document(3), dict(SCENE, objects=[floor]))
        for point in result['starts'].values():
            self.assertLessEqual(max(abs(point['x']), abs(point['z'])), 5.6)
        floor['position'] = [40., -.05, 40.]
        with self.assertRaisesRegex(ValueError, 'No bounded'):
            auto_place(document(3), dict(SCENE, objects=[floor]))

    def test_cancel_and_too_close_requested_starts(self):
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            auto_place(document(), SCENE, cancelled=lambda: True)
        raw = document()
        for actor in raw['actors']:
            actor['start'] = {'x': 0, 'z': 0}
        with self.assertRaisesRegex(ValueError, '1.5 metres'):
            auto_place(raw, SCENE)


if __name__ == '__main__':
    unittest.main()
