"""Counterexamples for geometric scene interaction measurements."""

import unittest

import numpy as np

from interaction_metrics import (
    CORE27_JOINT_NAMES, G1_JOINT_NAMES, continuity, floor_motion,
    gate_traversal, goal_endpoint, hand_contact, joint_index,
    object_contact, pair_separation,
)


def clip(skeleton="core27", frames=5):
    joints = 27 if skeleton == "core27" else 34
    p = np.zeros((frames, joints, 3), dtype=float)
    p[..., 1] = 1.0
    return p


class InteractionMetricTests(unittest.TestCase):
    def test_named_official_joint_orders(self):
        self.assertEqual(len(CORE27_JOINT_NAMES), 27)
        self.assertEqual(len(G1_JOINT_NAMES), 34)
        self.assertEqual(joint_index("core27", "LeftHandEnd"), 17)
        self.assertEqual(joint_index("g1", "right_hand_roll_skel"), 33)

    def test_rotated_gate_crossing_inside_and_around(self):
        normal = np.array([1.0, 1.0]) / np.sqrt(2)
        tangent = np.array([-normal[1], normal[0]])
        through = clip()
        around = clip()
        for i, offset in enumerate(np.linspace(-1, 1, len(through))):
            through[i][:, (0, 2)] = offset * normal
            around[i][:, (0, 2)] = offset * normal + 2.0 * tangent
        kwargs = dict(skeleton="core27", center=(0, 1, 0), normal_xz=(1, 1),
                      opening_width_m=1.0, body_radius_m=.15)
        self.assertTrue(gate_traversal(through, **kwargs)["traversed_proxy"])
        self.assertFalse(gate_traversal(around, **kwargs)["traversed_proxy"])
        self.assertEqual(gate_traversal(around, **kwargs)["crossing_count"], 1)
        # Ending beyond the gate is insufficient when the path crossed outside.
        self.assertGreater(around[-1, 0, 0], through[-1, 0, 0] - 2)

    def test_gate_body_clearance_and_direction(self):
        p = clip()
        p[:, :, 2] = np.linspace(-1, 1, len(p))[:, None]
        p[:, joint_index("core27", "LeftShoulder"), 0] = .48
        kwargs = dict(skeleton="core27", center=(0, 1, 0), normal_xz=(0, 1),
                      opening_width_m=1.0, body_radius_m=.1)
        self.assertFalse(gate_traversal(p, **kwargs)["traversed_proxy"])
        self.assertFalse(gate_traversal(p[::-1], **kwargs)["traversed_proxy"])
        self.assertTrue(gate_traversal(p[::-1], direction="either", **kwargs)["crossing_count"])
        p[:, :, 2] = np.array([-1, -.5, 0, 0, -.2])[:, None]
        self.assertFalse(gate_traversal(p, **kwargs)["traversed_proxy"])

    def test_goal_endpoint_and_pair_overlap(self):
        a = clip("core27")
        b = clip("g1")
        a[:, 0, 0] = np.linspace(0, 1, len(a))
        b[:, 0, 0] = a[:, 0, 0] + .1
        goal = goal_endpoint(a, skeleton="core27", target_xz=(1, 0))
        self.assertTrue(goal["within_tolerance"])
        pair = pair_separation(a, b, skeleton_a="core27", skeleton_b="g1")
        self.assertEqual(pair["root_disc_overlap_proxy_frames"], len(a))
        self.assertAlmostEqual(pair["min_root_separation_xz_m"], .1)

    def test_handoff_timing_gap_is_not_contact(self):
        a = clip("core27", 6)
        b = clip("core27", 6)
        ia = joint_index("core27", "RightHandEnd")
        ib = joint_index("core27", "LeftHandEnd")
        a[:, ia, :] = [2, 1, 0]
        b[:, ib, :] = [2, 1, 0]
        a[:3, ia, :] = [0, 1, 0]
        b[3:, ib, :] = [0, 1, 0]
        # Each actor reaches the object, but at different times.
        self.assertTrue(object_contact(a, skeleton="core27", object_xyz=(0, 1, 0),
                                       minimum_duration_s=.12)["contact_proxy"])
        self.assertTrue(object_contact(b, skeleton="core27", object_xyz=(0, 1, 0),
                                       hand="left", minimum_duration_s=.12)["contact_proxy"])
        contact = hand_contact(a, b, skeleton_a="core27", skeleton_b="core27",
                               minimum_duration_s=.12)
        self.assertFalse(contact["contact_proxy"])
        self.assertEqual(contact["near_frames"], 0)

    def test_simultaneous_contact_needs_contiguous_frames(self):
        a = clip("g1", 6)
        b = clip("g1", 6)
        ia = joint_index("g1", "right_hand_roll_skel")
        ib = joint_index("g1", "left_hand_roll_skel")
        a[:, ia, 0] = 1
        b[:, ib, 0] = 3
        b[[0, 2, 4], ib, 0] = 1
        kwargs = dict(skeleton_a="g1", skeleton_b="g1", fps=25,
                      minimum_duration_s=.08)
        sparse = hand_contact(a, b, **kwargs)
        self.assertEqual(sparse["near_frames"], 3)
        self.assertFalse(sparse["contact_proxy"])
        b[1, ib, 0] = 1
        self.assertTrue(hand_contact(a, b, **kwargs)["contact_proxy"])

    def test_floor_and_seam_metrics(self):
        p = clip("g1", 5)
        left_toe = joint_index("g1", "left_toe_base")
        right_toe = joint_index("g1", "right_toe_base")
        p[:, [left_toe, right_toe], 1] = .02
        p[:, left_toe, 0] = np.linspace(0, .2, len(p))
        floor = floor_motion(p, skeleton="g1", fps=25)
        self.assertEqual(floor["toe_penetration_max_depth_m"], 0)
        self.assertGreater(floor["low_toe_slide_mean_mps"], 0)
        p[3:, :, 0] += .3
        seam = continuity(p, skeleton="g1", fps=25, horizon_boundaries=(3,))
        self.assertAlmostEqual(seam["horizon_seams"][0]["root_step_m"], .3)

    def test_bad_shapes_rejected(self):
        with self.assertRaises(ValueError):
            gate_traversal(np.zeros((2, 26, 3)), skeleton="core27",
                           center=(0, 0, 0), normal_xz=(0, 1), opening_width_m=1)
        with self.assertRaises(ValueError):
            pair_separation(clip(frames=2), clip(frames=3),
                            skeleton_a="core27", skeleton_b="core27")
        with self.assertRaises(ValueError):
            continuity(clip(), skeleton="core27", horizon_boundaries=(5,))


if __name__ == "__main__":
    unittest.main()
