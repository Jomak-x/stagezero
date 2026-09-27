"""Meaningful geometry, swept collision and saved task continuity gates."""
import copy,json,math,unittest
from pathlib import Path
from crowd_demo_layout import make_layout
from crowd_demo_navigation import DemoConfig,Planner,swept_distance,measure,validate,phase
class GeometryTests(unittest.TestCase):
    def test_shop_doors_furniture_and_glazing(self):
        city=make_layout('city')
        self.assertTrue(city.valid(-29,-16.85));self.assertTrue(city.valid(-17,-16.85))
        self.assertFalse(city.valid(-31,-16.85));self.assertFalse(city.valid(-17,-20.1));self.assertFalse(city.valid(-29,-22.7))
        self.assertTrue(city.valid(-29,-21.25));self.assertTrue(city.valid(-19,-21))
    def test_city_limits_roads_to_authored_crossings(self):
        city=make_layout('city')
        for x in [-6.5,7]:
            for z in [-11,0,11]:self.assertTrue(city.valid(x,z))
        for x,z in [(-25,0),(23,9),(0,-40),(-57.5,-59.5)]:self.assertFalse(city.valid(x,z))
        for x,z in [(-25,-14),(23,14),(-61,-49)]:self.assertTrue(city.valid(x,z))
    def test_station_uses_level_route(self):
        layout=make_layout('station')
        for z in [-28,-20,-10,0,10]:self.assertTrue(layout.valid(2,z))
        self.assertFalse(layout.valid(41,-2));self.assertFalse(layout.valid(22,4))
    def test_crossing_scale_and_obstacles(self):
        layout=make_layout('crossing');self.assertTrue(layout.valid(0,0));self.assertFalse(layout.valid(43*.58,42*.58));self.assertFalse(layout.valid(23.5*.58,25.3*.58))
    def test_swept_headon_detected(self):
        self.assertAlmostEqual(swept_distance((-1,0),(1,0),(1,0),(-1,0)),0)
        self.assertAlmostEqual(swept_distance((-1,0),(1,0),(-1,1),(1,1)),1)
    def test_signal_phase_matches_viewer(self):
        for t,expected in [(0,'red'),(9.9,'red'),(10,'walk'),(45.9,'walk'),(46,'clearance'),(53.9,'clearance'),(54,'red'),(64,'walk'),(108,'red')]:
            self.assertEqual(phase(t),expected)
    def test_counts_supported(self):
        for count in [64,100,112,128]:self.assertEqual(DemoConfig(count=count).count,count)
        with self.assertRaises(ValueError):DemoConfig(count=129)
class SavedDemoTests(unittest.TestCase):
    def test_saved_scene_quality(self):
        files=list(Path('review/crowd-demos').glob('*-112.json'))
        if not files:self.skipTest('Saved demos not built; run experiments/build_crowd_demos.py')
        for path in files:
            with self.subTest(path=path):
                d=json.loads(path.read_text());m=validate(d)
                self.assertEqual(len(d['frames']),1201);self.assertEqual(len(d['agents']),112)
                self.assertEqual([a['id'] for a in d['agents']],list(range(112)))
                self.assertTrue(all(a['schedule'] and a['schedule'][0]['start']==0 and a['schedule'][-1]['end']==120 for a in d['agents']))
                self.assertTrue(all(len(f['people'])==112 for f in d['frames']))
                self.assertGreater(m['min_mobile_actor_travel_m'],10)
                if d['environment']=='city':
                    self.assertEqual(m['unpainted_road_actor_samples'],0);self.assertEqual(m['flashmob_actor_count'],24);self.assertEqual(m['outsider_formation_samples'],0);self.assertLessEqual(m['max_dance_slot_root_offset_m'],.42)
                    self.assertGreater(m['actors_completing_street_crossing'],0)
                    event=next(e for e in d['events'] if e['kind']=='flashmob')
                    for actor,slot in zip(event['actor_ids'],event['slot_xz']):
                        self.assertLess(math.dist(d['frames'][390]['people'][actor][:2],slot),.025)
                        self.assertLess(math.dist(d['frames'][650]['people'][actor][:2],slot),.025)
                        self.assertGreater(math.dist(d['frames'][-1]['people'][actor][:2],slot),2)
                    self.assertGreaterEqual(m['interior_visitors'],4);self.assertEqual(m['stationary_work_actor_ids'],[63]);self.assertEqual(m['min_actor_travel_m'],0)
                if d['environment']=='crossing':
                    self.assertGreaterEqual(m['peak_green_central_actors'],40);self.assertEqual(m['actors_reaching_opposite_corner'],112);self.assertEqual(m['road_actor_samples_during_wait'],0)
    def test_validator_rejects_corrupt_collision(self):
        f=Path('review/crowd-demos/crossing-112.json')
        if not f.exists():self.skipTest('saved crossing absent')
        d=json.loads(f.read_text());d['frames'][10]['people'][1][:2]=d['frames'][10]['people'][0][:2]
        with self.assertRaises(ValueError):validate(d)
if __name__=='__main__':unittest.main()
