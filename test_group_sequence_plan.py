"""CPU-only contracts for ordered group solos and semantic plan review."""

from copy import deepcopy
import json
import unittest
from unittest.mock import Mock

from prompt_scene_plan import ScenePromptPlanner, validate_plan
from test_prompt_scene_plan import SCENE


REQUEST = ('3 people are meeting each other and they start a dance party '
           'togehter and at the end all of them do a backflip')


def sequence(count=3, prompt=REQUEST):
    ids = [f'actor_{n}' for n in range(1, count + 1)]

    def beat(number, action, seconds):
        return {'id': f'beat-{number}', 'actor_ids': [ids[0]],
                'prompt': f'A person {action}.', 'seconds': seconds,
                'concurrent_solos': {actor: f'A person {action}.' for actor in ids[1:]}}

    return {'version': 1, 'title': 'Group celebration', 'prompt': prompt,
            'actor_count': count,
            'actors': [{'id': actor, 'name': actor, 'start': None,
                        'start_yaw_degrees': 0 if index == 0 else None}
                       for index, actor in enumerate(ids)],
            'meeting': {'x': 0, 'z': 0},
            'beats': [beat(1, 'dances energetically', 6),
                      beat(2, 'performs a backflip and lands', 4)],
            'warnings': []}


class GroupSequenceSchemaTests(unittest.TestCase):
    def test_meeting_and_ordered_all_cast_solos_are_valid(self):
        raw = sequence()
        clean = validate_plan(raw, SCENE, expected_prompt=REQUEST)
        self.assertEqual(clean['meeting'], {'x': 0., 'z': 0.})
        self.assertEqual([beat['seconds'] for beat in clean['beats']], [6, 4])
        self.assertEqual(clean['actors'][0]['start_yaw_degrees'], 0.)
        self.assertEqual(clean['actors'][1]['start_yaw_degrees'], None)
        self.assertEqual([set(beat['actor_ids']) | set(beat['concurrent_solos'])
                          for beat in clean['beats']], [set(['actor_1', 'actor_2', 'actor_3'])] * 2)

    def test_four_stages_and_thirty_second_bound(self):
        raw = sequence()
        raw['beats'] = [dict(deepcopy(raw['beats'][0]), id=f'beat-{n}', seconds=6)
                        for n in range(1, 5)]
        self.assertEqual(len(validate_plan(raw)['beats']), 4)
        raw['beats'][0]['seconds'] = 8
        raw['beats'][1]['seconds'] = 8
        raw['beats'][-1]['seconds'] = 10
        with self.assertRaisesRegex(ValueError, '30 seconds'):
            validate_plan(raw)

    def test_every_stage_requires_complete_disjoint_cast(self):
        raw = sequence()
        raw['beats'][1]['concurrent_solos'].pop('actor_3')
        with self.assertRaisesRegex(ValueError, 'cover the cast'):
            validate_plan(raw)
        raw = sequence()
        raw['beats'][1]['concurrent_solos']['actor_1'] = 'A person flips.'
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            validate_plan(raw)
        raw = sequence()
        raw['beats'][1]['concurrent_solos']['actor_4'] = 'A person flips.'
        with self.assertRaisesRegex(ValueError, 'known actors'):
            validate_plan(raw)

    def test_mixed_pair_or_serial_stage_is_rejected(self):
        raw = sequence()
        raw['beats'][1] = {'id': 'beat-2', 'actor_ids': ['actor_1', 'actor_2'],
                           'prompt': 'Two people wave.', 'seconds': 3,
                           'concurrent_solos': {'actor_3': 'A person flips.'}}
        with self.assertRaisesRegex(ValueError, 'all performers in solo actions'):
            validate_plan(raw)
        raw = sequence()
        raw['beats'][1].pop('concurrent_solos')
        with self.assertRaisesRegex(ValueError, 'all performers in solo actions'):
            validate_plan(raw)

    def test_single_pair_plus_third_remains_valid(self):
        raw = sequence()
        raw['meeting'] = {'x': 0, 'z': 0}
        raw['beats'] = [{'id': 'beat-1', 'actor_ids': ['actor_1', 'actor_2'],
                         'prompt': 'Two people shake hands.', 'seconds': 3,
                         'concurrent_solos': {'actor_3': 'A person waves.'}}]
        self.assertEqual(validate_plan(raw)['beats'][0]['actor_ids'], ['actor_1', 'actor_2'])
        raw = sequence()
        raw['beats'] = raw['beats'][:1]
        self.assertEqual(validate_plan(raw)['meeting'], {'x':0.,'z':0.})


class GroupSequenceReviewTests(unittest.TestCase):
    def test_too_close_generated_starts_are_repaired_before_motion(self):
        bad=sequence(prompt='Three people wave together.')
        bad['meeting']=None;bad['beats']=bad['beats'][:1]
        for i,actor in enumerate(bad['actors']):actor['start']={'x':i*1.4,'z':0.}
        good=deepcopy(bad)
        for i,actor in enumerate(good['actors']):actor['start']={'x':i*2.6,'z':0.}
        gateway=Mock(model='test',request_json=Mock(side_effect=[bad,good]))
        result=ScenePromptPlanner(gateway,expected_actor_count=3).plan(bad['prompt'],SCENE)
        self.assertEqual(result['planning_attempts'],2)
        self.assertEqual(result['actors'][1]['start']['x'],2.6)

    def test_exact_request_keeps_dance_and_final_flip_for_every_actor(self):
        raw = sequence()
        gateway = Mock(model='test-model', request_json=Mock(side_effect=[
            raw, {'preserves_request': True, 'reason': 'All requested stages are covered.'}]))
        result = ScenePromptPlanner(gateway, expected_actor_count=3).plan(REQUEST, SCENE)
        self.assertEqual(result['planning_attempts'], 1)
        self.assertEqual(gateway.request_json.call_count, 2)
        self.assertEqual(result['beats'][-1]['concurrent_solos']['actor_3'],
                         'A person performs a backflip and lands.')
        planner_instruction = gateway.request_json.call_args_list[0].args[0]
        self.assertIn('Never collapse dance and backflip', planner_instruction)
        review_instruction, payload = gateway.request_json.call_args_list[1].args
        self.assertIn('single compound beat', review_instruction)
        self.assertEqual(json.loads(payload)['original_request'], REQUEST)
        self.assertEqual(json.loads(payload)['proposed_plan']['beats'], result['beats'])

    def test_collapsed_beat_gets_one_repair_then_rejected_or_fixed(self):
        request = 'All three people dance together, then all of them backflip.'
        collapsed = sequence(prompt=request)
        collapsed['meeting'] = None
        collapsed['beats'] = [dict(collapsed['beats'][0],
                                   prompt='A person dances and then performs a backflip.',
                                   concurrent_solos={actor: 'A person dances and then performs a backflip.'
                                                     for actor in ('actor_2', 'actor_3')})]
        fixed = sequence(prompt=request)
        fixed['meeting'] = None
        gateway = Mock(model='test-model', request_json=Mock(side_effect=[
            collapsed, {'preserves_request': False,
                        'reason': 'Dance and final backflip were collapsed into one beat.'},
            fixed, {'preserves_request': True, 'reason': 'Separate ordered actions for everyone.'}]))
        result = ScenePromptPlanner(gateway, expected_actor_count=3).plan(request, SCENE)
        self.assertEqual(result['planning_attempts'], 2)
        self.assertEqual(len(result['beats']), 2)
        self.assertIn('omitted or reordered requested action', gateway.request_json.call_args_list[2].args[0])

    def test_split_cast_and_missing_ending_fail_review(self):
        split = sequence()
        split['beats'] = [
            {'id': 'beat-1', 'actor_ids': ['actor_1'], 'prompt': 'A person dances.', 'seconds': 4},
            {'id': 'beat-2', 'actor_ids': ['actor_2'], 'prompt': 'A person dances.', 'seconds': 4},
            {'id': 'beat-3', 'actor_ids': ['actor_3'], 'prompt': 'A person dances.', 'seconds': 4},
        ]
        gateway = Mock(model='test-model', request_json=Mock(side_effect=[
            split, {'preserves_request': False, 'reason': 'Only one actor dances per beat; no final flips.'},
            split, {'preserves_request': False, 'reason': 'Final backflip is missing.'}]))
        with self.assertRaisesRegex(ValueError, 'requested action'):
            ScenePromptPlanner(gateway, expected_actor_count=3).plan(REQUEST, SCENE)
        self.assertEqual(gateway.request_json.call_count, 4)

    def test_explicit_gather_requires_meeting_anchor(self):
        missing = sequence()
        missing['meeting'] = None
        gateway = Mock(model='test-model', request_json=Mock(return_value=missing))
        with self.assertRaisesRegex(ValueError, 'meeting anchor'):
            ScenePromptPlanner(gateway, expected_actor_count=3).plan(REQUEST, SCENE)
        self.assertEqual(gateway.request_json.call_count, 2)

    def test_review_covers_two_actor_explicit_group_sequence(self):
        request = 'Both dancers dance, then they both backflip.'
        raw = sequence(2, request)
        gateway = Mock(model='test-model', request_json=Mock(side_effect=[
            raw, {'preserves_request': True, 'reason': 'Both actors have both stages.'}]))
        self.assertEqual(ScenePromptPlanner(gateway, expected_actor_count=2).plan(request, SCENE)['actor_count'], 2)
        self.assertEqual(gateway.request_json.call_count, 2)


if __name__ == '__main__':
    unittest.main()
