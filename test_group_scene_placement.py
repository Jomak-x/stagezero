import unittest
from copy import deepcopy
from group_scene_placement import group_sequence_placement

class GroupPlacementTests(unittest.TestCase):
    def setUp(self):
        self.scene={'version':2,'name':'empty','objects':[],'effects':[],'lighting':'neutral'}
        self.plan={'actor_count':3,'actors':[{'id':f'actor_{i}','start':None,'start_yaw_degrees':None} for i in range(1,4)],'meeting':{'x':0.,'z':0.,'target_id':None}}
    def test_all_three_have_real_separated_routes(self):
        p=group_sequence_placement(self.plan,self.scene)
        for aid in p['starts']:
            self.assertNotEqual(p['starts'][aid]['z'],p['targets'][aid]['z'])
            self.assertGreaterEqual(len(p['routes'][aid]),2)
        self.assertEqual(len({v['x'] for v in p['targets'].values()}),3)
    def test_explicit_initial_location_and_heading_preserved(self):
        self.plan['actors'][0].update(start={'x':-3.,'z':-4.},start_yaw_degrees=90.)
        p=group_sequence_placement(self.plan,self.scene)
        self.assertEqual(p['starts']['actor_1'],{'x':-3.,'z':-4.,'yaw_degrees':90.})
    def test_cancellation(self):
        with self.assertRaises(RuntimeError):group_sequence_placement(self.plan,self.scene,cancelled=lambda:True)
    def test_city_formation_keeps_room_for_action(self):
        import json
        from pathlib import Path
        scene=json.loads(Path('review/prompt-scenes/backgrounds/city.json').read_text())
        self.plan['meeting']={'x':0.,'z':-12.}
        result=group_sequence_placement(self.plan,scene)
        self.assertEqual(result['action_obstacle_margin_m'],2.)
        self.assertTrue(all(abs(p['x'])<.01 for p in result['targets'].values()))
    def test_cancelled_explicit_inplace_scene_does_no_work(self):
        self.plan['meeting']=None
        for i,actor in enumerate(self.plan['actors']):actor['start']={'x':i*3.,'z':0.}
        with self.assertRaises(RuntimeError):
            group_sequence_placement(self.plan,self.scene,cancelled=lambda:True)
    def test_alternative_formation_after_crossing_routes(self):
        from unittest.mock import patch
        from group_scene_sequence import _routes
        calls=[]
        def check(scene,ids,starts,targets):
            calls.append(targets)
            if len(calls)==1:raise ValueError('Routes cross')
            return _routes(scene,ids,starts,targets)
        with patch('group_scene_sequence._routes',side_effect=check):
            result=group_sequence_placement(self.plan,self.scene)
        self.assertEqual(len(calls),2)
        self.assertTrue(all(abs(p['x'])<.01 for p in result['targets'].values()))
    def test_inplace_preserves_pr34_heading(self):
        self.plan['meeting']=None;self.plan['actors'][1]['start_yaw_degrees']=135.
        p=group_sequence_placement(self.plan,self.scene)
        self.assertEqual(p['starts']['actor_2']['yaw_degrees'],135.)
        self.assertEqual(p['starts'],p['targets'])
if __name__=='__main__':unittest.main()
