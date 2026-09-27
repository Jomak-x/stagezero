"""Ground contact and feature-history regressions for scene-aware locomotion."""
import unittest
from pathlib import Path
import numpy as np

from cinematic_adventure import make_temple
from scene_motion import (FEET, PARENTS, WALK_SPEED, FPS, plan_scene_motion,
                          apply_scene_motion, encode_scene_motion)
from take_sequencing import motion_statistics


def clip():
    # Checked-in, generated G1 poses provide real bone offsets and rotations.
    with np.load(Path(__file__).parent / 'assets/motion-goals.npz') as data:
        p = data['overhead_positions'][0].copy()
        r = data['overhead_rotations'][0].copy()
    p[:, [0, 2]] -= p[0, [0, 2]]
    p[:, 1] -= np.min(p[list(FEET), 1])
    return dict(positions=np.repeat(p[None], 104, axis=0),
                rotations=np.repeat(r[None], 104, axis=0),
                motion=np.zeros((104, 414), dtype=np.float32), metadata={})


class SceneMotionTests(unittest.TestCase):
    def test_stair_command_requires_geometry_and_reachable_entry(self):
        p = clip()['positions'][0]
        with self.assertRaisesRegex(ValueError, 'no reachable stair'):
            plan_scene_motion({}, 'walk upstairs', p)
        with self.assertRaisesRegex(ValueError, 'cannot be reached safely'):
            plan_scene_motion(make_temple(), 'walk downstairs', p)
        self.assertIsNone(plan_scene_motion(make_temple(), 'wave hello', p))
        ordinary = plan_scene_motion({}, 'walk three steps forward', p)
        self.assertIsNone(ordinary.route)
        self.assertEqual(ordinary.backend_prompt, 'walk three steps forward')

    def test_actual_temple_contacts_speed_and_corrected_history(self):
        source = clip()
        original = source['positions'].copy()
        plan = plan_scene_motion(make_temple(), 'walk up the stairs', source['positions'][0])
        self.assertEqual(plan.stair_name, 'Ceremonial stairs')
        self.assertGreater(plan.recommended_seconds, 8)
        previous = source['positions'][0].copy()
        for chunk in range(4):
            prior_copy = previous.copy()
            out = apply_scene_motion(source, plan, prior_positions=previous)
            np.testing.assert_array_equal(previous, prior_copy)
            p, r = out['positions'], out['rotations']
            roots = np.vstack((previous[0], p[:, 0]))
            self.assertLessEqual(np.max(np.linalg.norm(np.diff(roots[:, [0, 2]], axis=0), axis=1)), WALK_SPEED / FPS + 1e-5)
            np.testing.assert_allclose(r @ r.swapaxes(-1, -2), np.broadcast_to(np.eye(3), r.shape), atol=1e-5)
            # Every corrected bone must reconstruct from its parent's global
            # rotation and the original immutable G1 rest offset.
            for j, parent in enumerate(PARENTS):
                if parent < 0:
                    continue
                offset = np.einsum('tji,tj->ti', source['rotations'][:, parent], source['positions'][:, j] - source['positions'][:, parent])
                rebuilt = p[:, parent] + np.einsum('tij,tj->ti', r[:, parent], offset)
                np.testing.assert_allclose(rebuilt, p[:, j], atol=2e-5)
            grounded = 0
            for frame in p:
                clearance = []
                for j in FEET:
                    q = frame[j]
                    h = plan.geometry.support_height(q[0], q[2], q[1], max_step_up=.6, max_drop=1)
                    self.assertIsNotNone(h)
                    self.assertGreaterEqual(q[1] - h, -.002)
                    clearance.append(q[1] - h)
                grounded += min(clearance) < .055
            self.assertGreater(grounded, 90)
            mean, scale = motion_statistics()
            features = out['motion'] * scale + mean
            np.testing.assert_allclose(features[:, :3], p[:, 0], atol=1e-5)
            decoded = features[:, 5:104].reshape(104, 33, 3).copy()
            decoded[..., 0] += p[:, None, 0, 0]
            decoded[..., 2] += p[:, None, 0, 2]
            np.testing.assert_allclose(decoded, p[:, 1:], atol=1e-5)
            np.testing.assert_allclose(features[:, 104:308].reshape(104, 34, 6)[..., :3], r[..., 0], atol=1e-5)
            np.testing.assert_allclose(features[:-1, 308:410].reshape(103, 34, 3), np.diff(p, axis=0)*FPS, atol=2e-5)
            previous = p[-1].copy()
        self.assertTrue(out['metadata']['scene_interaction']['route_complete'])
        self.assertGreater(previous[0, 1], 2.5)
        np.testing.assert_array_equal(source['positions'], original)

    def test_descent_from_landing_has_no_approach_backtrack_and_reaches_ground(self):
        source = clip()
        scene = make_temple()
        up = plan_scene_motion(scene, 'walk upstairs', source['positions'][0])
        for _ in range(4):
            out = apply_scene_motion(source, up)
        start = out['positions'][-1].copy()
        down = plan_scene_motion(scene, 'walk downstairs', start)
        self.assertGreater(down.route[1, 2], down.route[0, 2])
        self.assertLess(down.route[-1, 1], .1)
        prior = start
        for _ in range(3):
            out = apply_scene_motion(source, down, prior_positions=prior)
            for frame in out['positions']:
                clearances = []
                for j in FEET:
                    q = frame[j]
                    h = down.geometry.support_height(q[0], q[2], q[1], max_step_up=.6, max_drop=1)
                    self.assertIsNotNone(h)
                    self.assertGreaterEqual(q[1] - h, -.002)
                    clearances.append(q[1] - h)
                self.assertLess(min(clearances), .055)
            prior = out['positions'][-1]
        self.assertTrue(out['metadata']['scene_interaction']['route_complete'])
        self.assertLess(prior[0, 1], 1.)
        self.assertGreater(prior[0, 2], -1.)

    def test_running_keeps_generated_gait_except_explicit_stair_routes(self):
        pose = clip()['positions'][0]
        for prompt in ('run forward', 'jog around the court'):
            self.assertIsNone(plan_scene_motion(make_temple(), prompt, pose))
        stairs = plan_scene_motion(make_temple(), 'run upstairs', pose)
        self.assertIsNotNone(stairs.route)
        self.assertIn('walks slowly', stairs.backend_prompt)

    def test_generic_walk_stops_before_wall_and_preserves_floor_contact(self):
        from scene_objects import make_object
        wall = make_object('wall', 0, (0, .8, .8))
        wall['size'] = [2., 1.6, .2]
        scene = dict(objects=[wall], assets=[])
        source = clip()
        source['positions'][..., 2] += np.arange(104)[:, None] * .04
        plan = plan_scene_motion(scene, 'walk forward', source['positions'][0])
        self.assertIsNone(plan.route)
        out = apply_scene_motion(source, plan)
        p = out['positions']
        self.assertTrue(out['metadata']['scene_interaction']['blocked'])
        self.assertEqual(out['metadata']['scene_motion'], 'Stopped before obstacle')
        self.assertGreater(p[-1, 0, 2], .3)
        self.assertLess(p[:, 0, 2].max(), .55)
        np.testing.assert_allclose(p[-20:, 0, [0, 2]], np.broadcast_to(p[-1, 0, [0, 2]], (20, 2)), atol=1e-6)
        self.assertGreaterEqual(p[:, list(FEET), 1].min(), -.002)
        self.assertTrue(np.all(p[:, list(FEET), 1].min(axis=1) < .055))

    def test_far_stairs_approach_is_partial_without_teleport(self):
        source = clip()
        source['positions'][..., 2] += 5
        plan = plan_scene_motion(make_temple(), 'walk upstairs', source['positions'][0])
        out = apply_scene_motion(source, plan)
        self.assertLess(np.linalg.norm(out['positions'][-1, 0, [0, 2]] - source['positions'][0, 0, [0, 2]]), 1.46)
        self.assertFalse(out['metadata']['scene_interaction']['route_complete'])
        self.assertIn('Approaching', out['metadata']['scene_motion'])


if __name__ == '__main__':
    unittest.main()
