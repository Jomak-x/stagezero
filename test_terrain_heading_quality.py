"""Bounded startup-heading quality checks; contact and native streams stay separate."""
import unittest

import numpy as np

from terrain_assisted_rig import _boundary_heading


class BoundaryHeadingQualityTests(unittest.TestCase):
    def setUp(self):
        self.runs = [[(0, None, None)], [(0, None, None)]]

    def peaks(self, yaw, inherited=0.):
        window = np.r_[inherited, inherited, yaw[:14]]
        return np.array([abs(np.diff(window)).max()*20,
                         abs(np.diff(window, n=2)).max()*400])

    def test_short_startup_turn_fades_target_and_preserves_join(self):
        desired = np.radians([0., 3., 10., 13., 13., 10., 7., 4., 1., -1., -2., -2.5,
                              -3., -3.5, -4., -4.])
        saved = desired.copy()
        blended, used = _boundary_heading(desired, 0., 20., [(4, 16)], self.runs)
        self.assertTrue(used)
        self.assertEqual(blended[0], 0.)
        np.testing.assert_array_equal(blended[11:], desired[11:])
        np.testing.assert_array_equal(desired, saved)
        self.assertTrue((self.peaks(blended) < self.peaks(desired)).all())

    def test_fade_cannot_push_startup_wobble_into_a_worse_join(self):
        desired = np.radians([0., .2, .5, .3, -1.3, -4.4, -8.1, -11., -12., -10.7,
                              -7., -2., 3., 7., 8., 8.])
        blended, used = _boundary_heading(desired, 0., 20., [(4, 16)], self.runs)
        self.assertFalse(used)
        np.testing.assert_array_equal(blended, desired)

    def test_long_pivot_immediate_walk_and_contact_turn_keep_existing_curve(self):
        desired = np.linspace(.2, 1.8, 60)
        phase = np.linspace(0., 1., 12)
        expected = desired.copy()
        expected[:12] += -.2*(1-phase*phase*(3-2*phase))
        for bouts, runs in [([(40, 60)], self.runs), ([(0, 60)], self.runs),
                            ([], self.runs), ([(8, 60)], [[(0,), (5,)], [(0,)]])]:
            with self.subTest(bouts=bouts, runs=runs):
                blended, used = _boundary_heading(desired, 0., 20., bouts, runs)
                self.assertFalse(used)
                np.testing.assert_array_equal(blended, expected)

    def test_boundary_wrap_uses_equivalent_nearest_heading(self):
        desired = np.radians(179.+np.array([0., 3., 10., 13., 13., 10., 7., 4., 1.,
                                            -1., -2., -2.5, -3., -3.5, -4., -4.]))
        blended, used = _boundary_heading(desired, np.radians(-181.), 20., [(4, 16)], self.runs)
        self.assertTrue(used)
        self.assertAlmostEqual(blended[0], np.radians(179.))
        self.assertLess(abs(np.diff(blended)).max(), np.radians(3.))
        equivalent, equivalent_used = _boundary_heading(
            desired+2*np.pi, np.radians(179.), 20., [(4, 16)], self.runs)
        self.assertEqual(equivalent_used, used)
        np.testing.assert_allclose(equivalent-2*np.pi, blended, atol=1e-12, rtol=0.)


if __name__ == '__main__':
    unittest.main()
