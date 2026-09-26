"""Contact geometry must agree with the actual rendered asset transforms."""

import unittest
import copy

from cinematic_adventure import make_temple
from scene_interaction_geometry import SceneInteractionGeometry
from scene_objects import make_object


class SceneInteractionGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = make_temple()
        cls.geometry = SceneInteractionGeometry.from_scene(cls.scene)

    def test_temple_stair_route_follows_five_rendered_treads(self):
        for name in ('Ceremonial stairs', 'Far ceremonial stairs'):
            route = next(route for route in self.geometry.stair_routes
                         if route['name'] == name)
            steps = route['steps']
            self.assertEqual(len(steps), 5)
            self.assertTrue(all(steps[i+1][1] > steps[i][1] and
                                steps[i+1][2] < steps[i][2]
                                for i in range(4)))
            # Each route point is supported by rendered triangles, even after
            # the asset bounds normalization used by the viewer.
            for x, y, z in steps:
                self.assertAlmostEqual(self.geometry.support_height(x, z, y, .03, .03), y, places=5)

    def test_side_stairs_follow_object_yaw(self):
        side = [r for r in self.geometry.stair_routes if r['name'] == 'Side stone stair']
        self.assertEqual(len(side), 2)
        for route in side:
            steps = route['steps']
            self.assertTrue(all(abs(p[2] - steps[0][2]) < 1e-6 for p in steps))
            self.assertTrue(all(steps[i+1][1] > steps[i][1] for i in range(4)))

    def test_support_respects_height_and_bridge_gap(self):
        # The starting court and bridge are at different heights. The bridge
        # cannot pull a ground-level character through the ravine.
        self.assertIsNotNone(self.geometry.support_height(0, 0, 0))
        self.assertIsNone(self.geometry.support_height(0, -14, 0, .45, .6))
        bridge_y = self.geometry.support_height(0, -14, 3, .45, .6)
        self.assertIsNotNone(bridge_y)
        self.assertAlmostEqual(bridge_y, 3.0, delta=.2)

    def test_objects_are_available_for_avoidance(self):
        door = next(o for o in self.scene['objects'] if o['id'] == 'temple-shrine-door')
        self.assertTrue(self.geometry.obstacle_at(*door['position']))
        self.assertFalse(self.geometry.obstacle_at(*door['position'], ignore_object_ids=(door['id'],)))

    def test_prop_only_scene_uses_stage_but_no_phantom_ball_top(self):
        ball = make_object('ball', 0, (2, .15, 0))
        chair = make_object('chair', 0, (4, .45, 0))
        geometry = SceneInteractionGeometry.from_scene({'objects': [ball, chair], 'assets': []})
        self.assertEqual(geometry.support_height(0, 0, 0, .1, .1), 0)
        self.assertIsNone(geometry.support_height(2, 0, .30, .05, .05))
        self.assertAlmostEqual(geometry.support_height(4, 0, .4725, .05, .05), .4725)

    def test_old_overlapped_stair_tread_is_not_a_walkable_hidden_floor(self):
        old_scene = copy.deepcopy(self.scene)
        next(o for o in old_scene['objects'] if o['name'] == 'Ceremonial stairs')['position'][2] = -3.7
        geometry = SceneInteractionGeometry.from_scene(old_scene)
        # At this point the fourth tread sits inside the raised terrace core.
        # Its top is below a solid slab, so a foot cannot stand on it.
        self.assertIsNone(geometry.support_height(0, -3.8, 1.27, .50, .60))


if __name__ == '__main__':
    unittest.main()
