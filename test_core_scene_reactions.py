import copy
import unittest
from types import SimpleNamespace
import numpy as np
from core_scene_reactions import object_states, evaluated_scene, check_reactive_geometry
from scene_objects import make_object
from interaction_scene import scene_objects, passage_for


def clip(xzs):
    p = np.zeros((1, len(xzs), 27, 3), dtype=np.float32)
    p[..., 1] = 1.
    p[0, :, :, 0] = np.asarray(xzs)[:, 0, None]
    p[0, :, :, 2] = np.asarray(xzs)[:, 1, None]
    return SimpleNamespace(positions=p, frames=len(xzs), fps=20, actor_ids=('actor_1',))


class ReactionsTests(unittest.TestCase):
    def setUp(self):
        self.door = make_object('door', 0)
        self.door['size'] = [1.6, 2., .12]
        self.scene = {'version': 2, 'name': 'Door', 'objects': [self.door], 'effects': [], 'lighting': 'neutral'}

    def test_opt_in_easing_seek_and_no_future_trigger(self):
        motion = clip([(0, -2)]*5 + [(0, -.7)]*20)
        original = copy.deepcopy(self.scene)
        self.assertFalse(object_states(self.scene, motion, 24)[0]['active'])
        self.assertFalse(object_states(self.scene, motion, 4, enabled=True)[0]['active'])
        first = object_states(self.scene, motion, 5, enabled=True)[0]
        self.assertEqual(first['position'][1], 1.)
        middle = object_states(self.scene, motion, 13, enabled=True)[0]
        self.assertAlmostEqual(middle['opening_fraction'], .5)
        self.assertAlmostEqual(middle['position'][1], 2.)
        final = object_states(self.scene, motion, 24, enabled=True)[0]
        self.assertEqual(final['position'][1], 3.)
        self.assertEqual(first, object_states(self.scene, motion, 5, enabled=True)[0])
        self.assertEqual(self.scene, original)
        np.testing.assert_array_equal(motion.positions[0, 0, 0], [0,1,-2])

    def test_start_frame_does_not_change_old_motion(self):
        motion = clip([(0, -.7)]*5 + [(0,-3)]*20)
        self.assertFalse(object_states(self.scene, motion, 24, enabled=True, start_frame=5)[0]['active'])

    def test_full_history_holds_open_and_collision_matches_frame(self):
        history = clip([(0,-.7)]*20 + [(0,-3)]*50)
        crossing = clip([(0,0)]*2)
        check_reactive_geometry(crossing, self.scene, history, enabled=True)
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            check_reactive_geometry(crossing, self.scene, enabled=True)
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            check_reactive_geometry(crossing, self.scene, history, enabled=False)

    def test_touch_and_elevated_doors_stay_static(self):
        self.door['interaction']['trigger'] = 'touch'
        motion = clip([(0,0)]*25)
        self.assertFalse(object_states(self.scene, motion, enabled=True)[0]['active'])
        self.door['interaction']['trigger'] = 'proximity'
        self.door['position'][1] = 4
        self.assertFalse(object_states(self.scene, motion, enabled=True)[0]['active'])

    def test_rotated_door_activation_and_passage_match_render(self):
        self.door['yaw'] = 90
        self.door['interaction']['radius'] = .2
        motion = clip([(.22, 0)]*25)
        doc = evaluated_scene(self.scene, motion, enabled=True)
        passage = passage_for(scene_objects(doc)[0], None, actor_height_m=1.65)
        self.assertEqual(passage.yaw_degrees, 90)
        self.assertEqual(doc['objects'][0]['position'][1], 3.)
        self.assertFalse(object_states(self.scene, clip([(.7,0)]*25), enabled=True)[0]['active'])

    def test_options_reject_ambiguous_values(self):
        for enabled, start in [('false',0), (True,True), (True,-1)]:
            with self.assertRaises(ValueError):
                object_states(self.scene, None, enabled=enabled, start_frame=start)
            with self.assertRaises(ValueError):
                check_reactive_geometry(clip([(0,-3)]), self.scene, enabled=enabled, start_frame=start)

    def test_lamp_and_actor_on_another_floor(self):
        lamp = make_object('lamp', 1)
        scene = {**self.scene, 'objects':[lamp]}
        motion=clip([(0,-.5)]*3)
        self.assertTrue(object_states(scene,motion,enabled=True)[0]['active'])
        lamp['position'][1] += 3
        self.assertFalse(object_states(scene,motion,enabled=True)[0]['active'])
        lamp['position'][1] -= 3
        motion.positions[...,1]=4
        self.assertFalse(object_states(scene,motion,enabled=True)[0]['active'])


if __name__ == '__main__':
    unittest.main()
