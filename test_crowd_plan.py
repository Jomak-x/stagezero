import unittest
from crowd_plan import validate_crowd_intent,plan_crowd
class CrowdPlanTests(unittest.TestCase):
    def setUp(self):
        self.raw=dict(title='Crossing',speed_min_mps=.8,speed_max_mps=1.5,diagonal_fraction=.3,group_fraction=.2,side_weights=dict(north=1,east=2,south=1,west=2),style_weights=dict(casual=3,brisk=2,relaxed=1))
    def test_execution_config_one_call(self):
        class Gateway:
            model='test';calls=0
            def request_json(g,*a,**k):g.calls+=1;return self.raw
        g=Gateway();p=plan_crowd('Cross',g);self.assertEqual(g.calls,1);self.assertEqual(p['config']['count'],64);self.assertAlmostEqual(sum(p['config']['side_weights'].values()),1);self.assertNotIn('title',p['config'])
    def test_invalid_values(self):
        for k,v in [('speed_min_mps',float('nan')),('group_fraction',True),('extra',1),('side_weights',dict(north=-1,east=1,south=1,west=1))]:
            with self.subTest(k=k),self.assertRaises(ValueError):validate_crowd_intent(dict(self.raw,**{k:v}))
