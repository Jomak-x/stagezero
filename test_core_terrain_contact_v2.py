"""Safety contracts for the offline Core terrain contact experiment."""

import unittest

import numpy as np

from experiments.core_terrain_contact_v2 import NEUTRAL, _fk, adapt_clip


class Plane:
    def __init__(self, height):
        self.height = height

    def support_height(self, x, z, current_y, max_step_up=.4, max_drop=.4):
        return self.height if current_y-max_drop <= self.height <= current_y+max_step_up else None


class CoreTerrainContactTests(unittest.TestCase):
    def setUp(self):
        self.rotations = np.broadcast_to(np.eye(3), (5, 27, 3, 3)).copy()
        root = np.array([0., -NEUTRAL[26, 1], 0.])
        self.positions = np.repeat(_fk(root, self.rotations[0])[None], 5, axis=0)

    def test_level_contact_keeps_native_motion_exact(self):
        original_p = self.positions.copy()
        original_r = self.rotations.copy()
        positions, rotations, report = adapt_clip(self.positions, self.rotations, Plane(0.),
                                                  reference_sole_y=np.zeros(5))
        self.assertTrue(report['accepted'], report['rejection_reasons'])
        np.testing.assert_array_equal(positions, original_p)
        np.testing.assert_array_equal(rotations, original_r)
        np.testing.assert_array_equal(self.positions, original_p)
        np.testing.assert_array_equal(self.rotations, original_r)

    def test_large_terrain_change_is_rejected_with_declared_budgets(self):
        _, _, report = adapt_clip(self.positions, self.rotations, Plane(.3),
                                  reference_sole_y=np.full(5, .1))
        self.assertFalse(report['accepted'])
        self.assertTrue(report['rejection_reasons'])
        self.assertEqual(report['budgets_m']['foot_relative_root'], .12)
        self.assertEqual(report['budgets_m']['root_y'], .08)
        self.assertEqual(report['budgets_m']['root_xz'], .06)


if __name__ == '__main__':
    unittest.main()
