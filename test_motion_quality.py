"""Synthetic checks for G1 clip measurements; no model or GPU required."""

import ast
from pathlib import Path
import unittest

import numpy as np

from motion_quality import G1_JOINT_NAMES, JOINT_INDEX, analyze_motion


def clip(frames=4):
    positions = np.zeros((frames, 34, 3), dtype=np.float64)
    positions[:, JOINT_INDEX["pelvis_skel"], 1] = 1.0
    for side in ("left", "right"):
        positions[:, JOINT_INDEX[f"{side}_shoulder_pitch_skel"], 1] = 1.4
        positions[:, JOINT_INDEX[f"{side}_hand_roll_skel"], 1] = 1.0
    rotations = np.tile(np.eye(3), (frames, 34, 1, 1))
    return positions, rotations


class MotionQualityTests(unittest.TestCase):
    def test_joint_order_matches_upstream_g1_definition(self):
        source = Path(__file__).parent / "vendor/ardy/ardy/skeleton/definitions.py"
        tree = ast.parse(source.read_text())
        g1 = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1Skeleton34")
        assignment = next(node for node in g1.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "bone_order_names_with_parents"
                                  for target in node.targets))
        names = tuple(name for name, _parent in ast.literal_eval(assignment.value))
        self.assertEqual(G1_JOINT_NAMES, names)

    def test_root_path_speed_and_squat_cue(self):
        p, r = clip()
        p[:, 0, 0] = [0.0, .1, .2, .1]
        p[:, 0, 1] = [1.0, .8, .6, 1.0]
        result = analyze_motion(p, r, fps=10)
        self.assertAlmostEqual(result["root_path_length_m"], .3)
        self.assertAlmostEqual(result["root_net_displacement_m"], .1)
        self.assertAlmostEqual(result["root_mean_speed_mps"], 1.0)
        self.assertAlmostEqual(result["root_drop_from_start_m"], .4)
        self.assertAlmostEqual(result["root_vertical_range_m"], .4)
        self.assertEqual(result["invalid_rotation_count"], 0)

    def test_floor_and_sliding_are_toe_based_proxies(self):
        p, _ = clip()
        left = JOINT_INDEX["left_toe_base"]
        right = JOINT_INDEX["right_toe_base"]
        p[:, left, 0] = [0.0, .1, .2, .3]
        p[:, right, 1] = [0.0, 0.0, -.03, -.03]
        result = analyze_motion(p, fps=10)
        self.assertEqual(result["floor_penetration_frames"], 2)
        self.assertAlmostEqual(result["floor_penetration_max_depth_m"], .03)
        self.assertEqual(result["foot_contact_proxy_samples"], 5)
        self.assertAlmostEqual(result["foot_sliding_proxy_peak_mps"], 1.0)
        # A raised toe produces no presumed contact even when moving horizontally.
        p[:, left, 1] = 1.0
        p[:, right, 1] = 1.0
        self.assertIsNone(analyze_motion(p, fps=10)["foot_sliding_proxy_mean_mps"])

    def test_hand_height_and_overhead_fraction(self):
        p, _ = clip()
        left_hand = JOINT_INDEX["left_hand_roll_skel"]
        right_hand = JOINT_INDEX["right_hand_roll_skel"]
        p[:, left_hand, 1] = [1.5, 1.6, 1.0, 1.0]
        p[:, right_hand, 1] = [1.0, 1.0, 1.8, 1.0]
        result = analyze_motion(p, overhead_margin=.05)
        self.assertAlmostEqual(result["left_hand_overhead_fraction"], .5)
        self.assertAlmostEqual(result["right_hand_overhead_fraction"], .25)
        self.assertAlmostEqual(result["either_hand_overhead_fraction"], .75)
        self.assertAlmostEqual(result["right_hand_above_shoulder_max_m"], .4)

    def test_boundary_position_and_velocity_discontinuity(self):
        prior, prior_r = clip(2)
        p, r = clip(2)
        prior[:, 0, 0] = [0.0, 1.0]
        p[:, 0, 0] = [3.0, 6.0]
        r[0, 0] = np.diag([-1, -1, 1])
        result = analyze_motion(p, r, prior_positions=prior, prior_rotations=prior_r, fps=1)
        self.assertAlmostEqual(result["boundary_root_position_jump_m"], 2.0)
        self.assertAlmostEqual(result["boundary_entry_velocity_jump_mps"], 1.0)
        self.assertAlmostEqual(result["boundary_exit_velocity_jump_mps"], 1.0)
        self.assertAlmostEqual(result["boundary_root_rotation_jump_deg"], 180.0)
        self.assertFalse(analyze_motion(p)["boundary_available"])

    def test_invalid_values_are_reported_without_nan_metrics(self):
        p, r = clip()
        r[0, 0] *= 2
        r[1, 0, 0, 0] = np.nan
        result = analyze_motion(p, r)
        self.assertFalse(result["rotations_finite"])
        self.assertEqual(result["invalid_rotation_count"], 2)
        self.assertTrue(result["quality_metrics_available"])
        p[0, 0, 0] = np.inf
        result = analyze_motion(p, r)
        self.assertFalse(result["quality_metrics_available"])
        self.assertEqual(result["nonfinite_position_count"], 1)
        self.assertNotIn("root_mean_speed_mps", result)

    def test_shape_and_rate_validation(self):
        p, r = clip()
        with self.assertRaises(ValueError):
            analyze_motion(p[:, :33])
        with self.assertRaises(ValueError):
            analyze_motion(p, r[:, :33])
        with self.assertRaises(ValueError):
            analyze_motion(p, fps=0)
        with self.assertRaises(ValueError):
            analyze_motion(p, prior_rotations=r)


if __name__ == "__main__":
    unittest.main()
