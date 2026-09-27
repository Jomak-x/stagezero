from copy import deepcopy
from unittest import TestCase
from unittest.mock import Mock
from test_prompt_scene_plan import document, SCENE
from prompt_scene_plan import ScenePromptPlanner

class ExpectedActorCountTests(TestCase):
    def test_repair_preserves_requested_count(self):
        valid=document()
        wrong=deepcopy(valid);wrong['actor_count']=1;wrong['actors']=wrong['actors'][:1];wrong['beats']=[dict(wrong['beats'][0],actor_ids=['actor_1'],seconds=4)]
        gateway=Mock(model='test',request_json=Mock(side_effect=[wrong,valid]))
        result=ScenePromptPlanner(gateway,expected_actor_count=valid['actor_count']).plan(valid['prompt'],SCENE)
        self.assertEqual(result['actor_count'],valid['actor_count'])
        self.assertEqual(gateway.request_json.call_count,2)
        self.assertIn('preserve every performer',gateway.request_json.call_args.args[0])

    def test_repeated_count_loss_is_rejected(self):
        raw=document()
        gateway=Mock(model='test',request_json=Mock(return_value=raw))
        with self.assertRaisesRegex(ValueError,'Requested 3 performers'):
            ScenePromptPlanner(gateway,expected_actor_count=3).plan(raw['prompt'],SCENE)

    def test_invalid_expected_count(self):
        for value in (0,4,True,'2'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                ScenePromptPlanner(expected_actor_count=value)
