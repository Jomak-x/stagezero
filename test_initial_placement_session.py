"""The live staging editor preserves a validated, exportable AI plan."""
from copy import deepcopy
import json
import unittest
from unittest.mock import Mock

from initial_placement_session import InitialPlacementSession
from prompt_scene_plan import validate_plan, auto_place
from scene_objects import make_object


SCENE = {'version': 2, 'name': 'Test floor', 'objects': [], 'effects': [], 'lighting': 'neutral'}


def plan(count=2):
    ids = [f'actor_{i + 1}' for i in range(count)]
    beats = [{'id': f'beat-{i + 1}', 'actor_ids': [aid], 'prompt': 'Wave hello.', 'seconds': 2}
             for i, aid in enumerate(ids)]
    return {'version': 1, 'title': 'Waves', 'prompt': 'Everyone waves.',
            'actor_count': count,
            'actors': [{'id': aid, 'name': aid, 'start': None} for aid in ids],
            'meeting': None, 'beats': beats, 'warnings': []}


class InitialPlacementSessionTests(unittest.TestCase):
    def test_plan_edit_export_reset_and_detached_snapshot(self):
        ai = plan(2)
        ai['actors'][0]['start'] = {'x': -3, 'z': 0}
        ai['actors'][0]['start_yaw_degrees'] = 90
        ai['actors'][1]['start'] = {'x': 3, 'z': 0}
        ai['actors'][1]['start_yaw_degrees'] = -90
        ai['planner_model'] = 'gpt-6-astra'
        ai['raw_plan'] = deepcopy({key: value for key, value in ai.items() if key != 'planner_model'})
        ai['planning_seconds'] = 2.5
        ai['planning_attempts'] = 1
        ai['planner_cache_hit'] = False
        planner = Mock()
        planner.plan.return_value = ai
        session = InitialPlacementSession(SCENE, planner=planner)
        first = session.plan(ai['prompt'])
        planner.plan.assert_called_once_with(ai['prompt'], SCENE)
        original = deepcopy(first['placement'])
        self.assertEqual(first['ai_plan']['planner_model'], 'gpt-6-astra')
        self.assertEqual(first['changed_actor_ids'], [])

        edited = session.edit_actor('actor_1', x=-4.0, z=0.5, yaw_degrees=45)
        self.assertEqual(edited['changed_actor_ids'], ['actor_1'])
        self.assertEqual(edited['placement']['starts']['actor_1'],
                         {'x': -4.0, 'z': 0.5, 'yaw_degrees': 45.0})
        self.assertEqual(edited['placement']['starts']['actor_2'], original['starts']['actor_2'])
        exported = session.export_plan()
        self.assertEqual(validate_plan(exported, SCENE), exported)
        self.assertEqual(auto_place(exported, SCENE)['starts'], edited['placement']['starts'])
        self.assertEqual(exported['actors'][0]['start'], {'x': -4., 'z': .5})
        self.assertEqual(exported['actors'][0]['start_yaw_degrees'], 45.)
        json.dumps(exported, allow_nan=False)

        edited['placement']['starts']['actor_1']['x'] = 999
        edited['plan']['actors'][0]['name'] = 'tampered'
        self.assertEqual(session.snapshot()['placement']['starts']['actor_1']['x'], -4.)
        restored = session.reset()
        self.assertEqual(restored['placement'], original)
        self.assertEqual(restored['changed_actor_ids'], [])
        self.assertEqual(restored['plan']['actors'][0]['name'], 'actor_1')

    def test_invalid_edits_leave_everything_unchanged(self):
        crate = make_object('crate', 0)
        crate['position'] = [10, .5, 10]
        scene = dict(SCENE, objects=[crate])
        session = InitialPlacementSession(scene)
        session.set_plan(plan(2))
        before = session.snapshot()
        for kwargs in ({'x': 25}, {'z': float('nan')}, {'yaw_degrees': 181},
                       {'x': True}, {'x': 10, 'z': 10}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                session.edit_actor('actor_1', **kwargs)
            self.assertEqual(session.snapshot(), before)
        other = before['placement']['starts']['actor_2']
        with self.assertRaisesRegex(ValueError, 'at least 1.5 metres apart'):
            session.edit_actor('actor_1', x=other['x'], z=other['z'])
        self.assertEqual(session.snapshot(), before)
        with self.assertRaisesRegex(ValueError, 'Unknown cast actor'):
            session.edit_actor('actor_9', x=0)
        self.assertEqual(session.snapshot(), before)

        invalid = plan(2)
        invalid['actors'][0]['start'] = {'x': 30, 'z': 0}
        with self.assertRaises(ValueError):
            session.set_plan(invalid)
        self.assertEqual(session.snapshot(), before)

    def test_one_to_three_actor_plans_are_supported(self):
        for count in (1, 2, 3):
            with self.subTest(count=count):
                session = InitialPlacementSession(SCENE)
                result = session.set_plan(plan(count))
                self.assertEqual(len(result['placement']['starts']), count)
                self.assertEqual(len(session.export_plan()['actors']), count)


if __name__ == '__main__':
    unittest.main()
