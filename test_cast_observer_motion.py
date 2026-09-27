"""Observer mechanics/provenance checks; these do not establish visual quality."""
import unittest

import numpy as np

from cast_observer_motion import LOWER_BODY, PARENTS, refine_observers


def pose():
    # Standing person, +Z forward, with a released right greeting arm.
    p = np.array([[0, .95, 0], [.1, .9, 0], [-.1, .9, 0], [0, 1.08, 0],
                  [.1, .48, .01], [-.1, .48, .01], [0, 1.23, 0],
                  [.1, .08, 0], [-.1, .08, 0], [0, 1.40, 0],
                  [.1, .03, .15], [-.1, .03, .15], [0, 1.53, 0],
                  [.09, 1.43, 0], [-.09, 1.43, 0], [0, 1.72, 0],
                  [.18, 1.42, 0], [-.18, 1.42, 0], [.22, 1.13, .04],
                  [-.20, 1.2, .21], [.23, .87, .08], [-.21, 1.08, .45]])
    return p


def fixture(frames=210):
    p = np.repeat(np.stack((pose(), pose()+[2, 0, 0], pose()+[-2, 0, 1]))[None], frames, axis=0)
    activities = [{'start_frame': 0, 'end_frame_exclusive': 120, 'active_actor_ids': ['a', 'b']},
                  {'start_frame': 120, 'end_frame_exclusive': frames, 'active_actor_ids': ['b', 'c']}]
    # Active frames intentionally have motion so accidental edits are detected.
    p[:120, :2, :, 2] += np.sin(np.arange(120)[:, None, None]*.07)*.03
    p[120:, 1:, :, 2] += np.sin(np.arange(frames-120)[:, None, None]*.07)*.03
    return p, ['a', 'b', 'c'], activities


class ObserverMotionTests(unittest.TestCase):
    def test_active_lower_body_and_reentry_samples_exact(self):
        p, ids, activities = fixture()
        original = p.copy()
        out, report = refine_observers(p, ids, activities)
        np.testing.assert_array_equal(p, original)
        np.testing.assert_array_equal(out[:120, :2], p[:120, :2])
        np.testing.assert_array_equal(out[120:, 1:], p[120:, 1:])
        np.testing.assert_array_equal(out[:, :, LOWER_BODY], p[:, :, LOWER_BODY])
        np.testing.assert_array_equal(out[:2, 2], p[:2, 2])
        np.testing.assert_array_equal(out[118:120, 2], p[118:120, 2])
        np.testing.assert_array_equal(out[120:122, 0], p[120:122, 0])
        self.assertGreater(report['applied_actor_frames'], 100)
        self.assertFalse(report['model_generated'])
        self.assertEqual(report['visual_acceptance'], 'unverified')
        self.assertFalse(np.shares_memory(p, out))

    def test_released_arm_relaxes_without_changing_lengths(self):
        p, ids, activities = fixture()
        out, report = refine_observers(p, ids, activities)
        self.assertLess(out[-1, 0, 21, 1], p[-1, 0, 21, 1]-.15)
        before = np.linalg.norm(p[:, :, 1:]-p[:, :, PARENTS[1:]], axis=-1)
        after = np.linalg.norm(out[:, :, 1:]-out[:, :, PARENTS[1:]], axis=-1)
        np.testing.assert_allclose(before, after, atol=1e-12, rtol=0)
        self.assertLess(report['maximum_bone_length_error_m'], 1e-12)
        # Settled observers continue small motion instead of a second freeze.
        self.assertGreater(np.linalg.norm(np.diff(out[-35:, 0, 9], axis=0), axis=1).sum(), .005)

    def test_segmentation_invariant_and_seed_deterministic(self):
        p, ids, activities = fixture()
        split = [dict(activities[0], end_frame_exclusive=60),
                 dict(activities[0], start_frame=60), activities[1]]
        a, report_a = refine_observers(p, ids, activities, seed=9)
        b, report_b = refine_observers(p, ids, split, seed=9)
        c, _ = refine_observers(p, ids, activities, seed=10)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(report_a, report_b)
        self.assertGreater(abs(a-c).max(), .001)

    def test_upright_correction_preserves_feet_and_boundary_speed(self):
        p, ids, activities = fixture()
        upper = [j for j in range(22) if j not in LOWER_BODY]
        angle = np.radians(18)
        matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        p[:120, 2, upper] = (p[:120, 2, upper]-p[:120, 2, 0, None]) @ matrix.T+p[:120, 2, 0, None]
        out, _ = refine_observers(p, ids, activities)
        def tilt(v):
            return np.degrees(np.arctan2(np.linalg.norm(v[[0, 2]]), v[1]))
        self.assertLess(tilt(out[60, 2, 9]-out[60, 2, 0]), tilt(p[60, 2, 9]-p[60, 2, 0])-5)
        self.assertLess(np.linalg.norm(out[2, 2]-out[1, 2], axis=-1).max()*30, .01)
        self.assertLess(np.linalg.norm(out[118, 2]-out[117, 2], axis=-1).max()*30, .02)

    def test_existing_inactive_motion_is_not_overwritten(self):
        p, ids, activities = fixture()
        p[:120, 2, 20, 0] += np.arange(120)*.001
        out, report = refine_observers(p, ids, activities)
        np.testing.assert_array_equal(out[:120, 2], p[:120, 2])
        self.assertIn('already contains motion', report['spans'][-1]['skipped'])

    def test_single_actor_no_inactive_motion(self):
        p = np.repeat(pose()[None, None], 20, axis=0)
        out, report = refine_observers(p, ['a'], [{'start_frame': 0, 'end_frame_exclusive': 20, 'active_actor_ids': ['a']}])
        np.testing.assert_array_equal(out, p)
        self.assertEqual(report['spans'], [])

    def test_float32_preserves_type_and_bounds(self):
        p, ids, activities = fixture()
        out, report = refine_observers(p.astype(np.float32), ids, activities)
        self.assertEqual(out.dtype, np.float32)
        self.assertLess(report['maximum_bone_length_error_m'], 1e-6)
        self.assertLess(report['maximum_joint_correction_m'], .8)

    def test_invalid_inputs_rejected(self):
        p, ids, activities = fixture()
        cases = [(p, ids, activities, {'fps': 0}), (p, ids, activities, {'seed': True}),
                 (p, ['a', 'a', 'c'], activities, {}), (p, ids, activities[:1], {}),
                 (p, ids, [dict(activities[0], start_frame=1), activities[1]], {}),
                 (p, ids, [dict(activities[0], active_actor_ids=['missing']), activities[1]], {}),
                 (p.astype(int), ids, activities, {}), (p*np.nan, ids, activities, {})]
        for data, actor_ids, spans, kwargs in cases:
            with self.subTest(kwargs=kwargs, actors=actor_ids):
                with self.assertRaises(ValueError):
                    refine_observers(data, actor_ids, spans, **kwargs)


if __name__ == '__main__':
    unittest.main()
