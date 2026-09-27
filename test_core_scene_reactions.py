import copy
import unittest
from types import SimpleNamespace
import numpy as np
from core_scene_reactions import object_states, evaluated_scene, check_reactive_geometry
from scene_objects import make_object
from interaction_scene import scene_objects, passage_for


def clip(xzs, heights=None):
    p = np.zeros((1, len(xzs), 27, 3), dtype=np.float32)
    p[..., 1] = 1.
    if heights is not None:
        p[0, :, :, 1] = np.asarray(heights)[:, None]
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

    def test_elevated_gate_uses_observed_support_relative_root(self):
        self.door['position'][1] = 7.1  # Original base/floor is 6.1 m.
        self.door['interaction']['radius'] = .2
        motion = clip([(0, -2)] * 3 + [(0, -.2)] * 3 + [(0, -.2)] * 20,
                      [7.1] * 3 + [1.] * 3 + [7.1] * 20)
        original_scene = copy.deepcopy(self.scene)
        original_motion = motion.positions.copy()
        self.assertFalse(object_states(self.scene, motion, 5, enabled=True,
                                       terrain=True)[0]['active'])
        self.assertFalse(object_states(self.scene, motion, 6, enabled=True,
                                       terrain=False)[0]['active'])
        first = object_states(self.scene, motion, 6, enabled=True, terrain=True)[0]
        self.assertEqual(first['trigger_frame'], 6)
        self.assertEqual(first['position'][1], 7.1)
        self.assertEqual(first['opening_fraction'], 0.)
        opened = object_states(self.scene, motion, 22, enabled=True, terrain=True)[0]
        self.assertEqual(opened['trigger_frame'], 6)
        self.assertEqual(opened['opening_fraction'], 1.)
        self.assertAlmostEqual(opened['position'][1], 9.1)
        evaluated = evaluated_scene(self.scene, motion, 22, enabled=True, terrain=True)
        self.assertAlmostEqual(evaluated['objects'][0]['position'][1], 9.1)
        self.assertEqual(self.scene, original_scene)
        np.testing.assert_array_equal(motion.positions, original_motion)
        self.assertEqual(opened, object_states(self.scene, motion, 22,
                                               enabled=True, terrain=True)[0])

    def test_elevated_gate_replay_start_frame_and_height_bounds(self):
        self.door['position'][1] = 7.1
        motion = clip([(0, -.7)] * 3 + [(0, -3)] * 3 + [(0, -.7)] * 20,
                      [7.1] * 26)
        self.assertFalse(object_states(self.scene, motion, 5, enabled=True,
                                       terrain=True, start_frame=3)[0]['active'])
        state = object_states(self.scene, motion, 6, enabled=True,
                              terrain=True, start_frame=3)[0]
        self.assertEqual(state['trigger_frame'], 6)
        copied = clip([(0, -.7)] * 3 + [(0, -3)] * 3 + [(0, -.7)] * 20,
                      [7.1] * 26)
        self.assertEqual(state, object_states(self.scene, copied, 6,
                                              enabled=True, terrain=True,
                                              start_frame=3)[0])
        for bad_height in (1., 6.49, 6.7, 7.5, 7.71, 9.):
            wrong_floor = clip([(0, -.7)] * 25, [bad_height] * 25)
            self.assertFalse(object_states(self.scene, wrong_floor,
                                           enabled=True, terrain=True)[0]['active'])

    def test_terrain_activation_start_ignores_old_elevated_prefix_only(self):
        elevated = copy.deepcopy(self.door)
        elevated['id'] = 'elevated-door'
        elevated['position'][1] = 7.1
        scene = {**self.scene, 'objects': [self.door, elevated]}
        motion = clip([(0, -.2)] * 9, [1.] * 3 + [7.1] * 6)
        old = object_states(scene, motion, 5, enabled=True, terrain=True,
                            start_frame=0, terrain_start_frame=6)
        self.assertEqual(old[0]['trigger_frame'], 0)
        self.assertFalse(old[1]['active'])
        current = object_states(scene, motion, 8, enabled=True, terrain=True,
                                start_frame=0, terrain_start_frame=6)
        self.assertEqual(current[0]['trigger_frame'], 0)
        self.assertEqual(current[1]['trigger_frame'], 6)
        later = object_states(scene, motion, 8, enabled=True, terrain=True,
                              start_frame=7, terrain_start_frame=6)
        self.assertEqual(later[0]['trigger_frame'], None)
        self.assertEqual(later[1]['trigger_frame'], 7)

    def test_terrain_collision_requires_aware_checker_and_evaluates_lift(self):
        self.door['position'][1] = 7.1
        motion = clip([(0, -.7)] * 18, [7.1] * 18)
        with self.assertRaisesRegex(ValueError, 'support-aware'):
            check_reactive_geometry(motion, self.scene, enabled=True, terrain=True)
        observed = []

        def checker(positions, skeleton, scene, affordances):
            self.assertEqual(skeleton, 'core27')
            self.assertEqual(positions.shape[0], 1)
            observed.append(scene['objects'][0]['position'][1])
            return {'total_collision_frames': 0}

        check_reactive_geometry(motion, self.scene, enabled=True, terrain=True,
                                collision_checker=checker)
        self.assertEqual(len(observed), motion.frames)
        self.assertAlmostEqual(observed[0], 7.1)
        self.assertAlmostEqual(observed[-1], 9.1)

    def test_elevated_gate_history_opens_only_from_committed_roots(self):
        self.door['position'][1] = 7.1
        history = clip([(0, -.7)] * 20 + [(0, -3)] * 5, [7.1] * 25)
        proposal = clip([(0, 0)] * 2, [7.1] * 2)
        positions = []

        def checker(_positions, _skeleton, scene, _affordances):
            positions.append(scene['objects'][0]['position'][1])
            return {'total_collision_frames': 0}

        check_reactive_geometry(proposal, self.scene, history, enabled=True,
                                terrain=True, collision_checker=checker)
        self.assertEqual(positions, [9.1, 9.1])
        self.assertFalse(object_states(self.scene, history, 0, enabled=True,
                                       terrain=True, start_frame=20)[0]['active'])
        positions.clear()
        check_reactive_geometry(proposal, self.scene, history, enabled=True,
                                terrain=True, start_frame=27,
                                collision_checker=checker)
        self.assertEqual(positions, [7.1, 7.1])

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
        for enabled, start, terrain, terrain_start in [
            ('false',0,False,0), (True,True,False,0), (True,-1,False,0),
            (True,0,'yes',0), (True,0,True,True), (True,0,True,-1)]:
            with self.assertRaises(ValueError):
                object_states(self.scene, None, enabled=enabled,
                              start_frame=start, terrain=terrain,
                              terrain_start_frame=terrain_start)
            with self.assertRaises(ValueError):
                check_reactive_geometry(clip([(0,-3)]), self.scene,
                                        enabled=enabled, start_frame=start,
                                        terrain=terrain,
                                        terrain_start_frame=terrain_start)

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
