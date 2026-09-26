"""CPU verification for G1 pose-goal alignment and native condition masks."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor" / "ardy"))

from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep
from ardy.skeleton.definitions import G1Skeleton34
from motion_action_goals import _heading, build_action_conditions, load_action_goal_asset, prepare_goal_pose


class MotionActionGoalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skeleton = G1Skeleton34(load=False)
        cls.metadata, cls.arrays = load_action_goal_asset()

    def _history(self, action: str, frames: int, *, yaw: float = 0.0, dx: float = 0.0, dz: float = 0.0):
        a = self.arrays
        root = a[f"{action}_anchor_root"]
        right = a[f"{action}_anchor_right_hip"]
        left = a[f"{action}_anchor_left_hip"]
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=np.float32)
        history = np.zeros((frames, 34, 3), dtype=np.float32)
        for index, position in ((0, root), (8, right), (1, left)):
            history[:, index] = (position - root) @ rotation.T + root + np.array((dx, 0, dz), dtype=np.float32)
        return history

    def test_asset_has_generated_provenance_and_finite_poses(self):
        self.assertEqual(self.metadata["goal_kind"], "generated_reference_keyframes")
        for action in ("overhead", "squat"):
            info = self.metadata["actions"][action]
            self.assertEqual(info["source_seed"], 33)
            self.assertEqual(len(info["keyframes_in_104_frame_generation"]), 3)
            self.assertTrue(np.isfinite(self.arrays[f"{action}_rotations"]).all())

    def test_unrotated_reference_is_unchanged(self):
        for action in ("overhead", "squat"):
            with self.subTest(action=action):
                p, r, meta = prepare_goal_pose(action, self._history(action, 4), self.skeleton)
                np.testing.assert_allclose(p, self.arrays[f"{action}_positions"], atol=2e-6)
                np.testing.assert_allclose(r, self.arrays[f"{action}_rotations"], atol=2e-6)
                self.assertAlmostEqual(meta["world_yaw_rotation_rad"], 0.0, places=5)
                self.assertEqual(meta["mode"], "reference_assisted_native_ardy_conditioning")

    def test_90_degree_yaw_and_translation_rotate_entire_pose(self):
        yaw, dx, dz = np.pi / 2, 3.0, -2.0
        rotation = np.array(((0, 0, 1), (0, 1, 0), (-1, 0, 0)), dtype=np.float32)
        for action in ("overhead", "squat"):
            with self.subTest(action=action):
                history = self._history(action, 12, yaw=yaw, dx=dx, dz=dz)
                p, r, meta = prepare_goal_pose(action, history, self.skeleton)
                anchor = self.arrays[f"{action}_anchor_root"]
                expected_p = (self.arrays[f"{action}_positions"] - anchor) @ rotation.T + anchor
                expected_p[..., 0] += dx
                expected_p[..., 2] += dz
                expected_r = rotation @ self.arrays[f"{action}_rotations"]
                np.testing.assert_allclose(p, expected_p, atol=2e-6)
                np.testing.assert_allclose(r, expected_r, atol=2e-6)
                self.assertAlmostEqual(meta["world_yaw_rotation_rad"], yaw, places=5)
                self.assertAlmostEqual(float(p[0, 0, 0] - expected_p[0, 0, 0]), 0.0, places=5)
                source_heading = _heading(self.arrays[f"{action}_positions"][0], 8, 1)
                aligned_heading = _heading(p[0], 8, 1)
                heading_shift = np.arctan2(np.sin(aligned_heading - source_heading),
                                           np.cos(aligned_heading - source_heading))
                self.assertAlmostEqual(float(heading_shift), yaw, places=5)

    def test_first_horizon_requires_no_history(self):
        self.assertEqual(build_action_conditions(None, "overhead", None, generated_offset=0), (None, None, None))

    def test_invalid_action_and_skeleton_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported action"):
            prepare_goal_pose("wave", self._history("overhead", 4), self.skeleton)
        with self.assertRaisesRegex(ValueError, "G1"):
            prepare_goal_pose("overhead", self._history("overhead", 4), SimpleNamespace(name="other", nbjoints=34))
        with self.assertRaisesRegex(ValueError, "offset 52"):
            build_action_conditions(None, "overhead", None, generated_offset=104)

    def test_official_cpu_constraint_masks_are_native_and_do_not_touch_history(self):
        for action, history_len, expected_channels in (("overhead", 4, 105), ("squat", 12, 312)):
            with self.subTest(action=action):
                positions = self._history(action, history_len, yaw=np.pi / 2, dx=2, dz=-1)
                representation = ArdyMotionRep(self.skeleton, fps=25)
                representation.normalize = lambda tensor: tensor
                representation.inverse = lambda motion, is_normalized: {
                    "posed_joints": torch.from_numpy(positions).unsqueeze(0),
                }
                model = SimpleNamespace(
                    skeleton=self.skeleton, gen_horizon_len=52, motion_rep=representation,
                )
                normalized_history = torch.zeros((1, history_len, 414), dtype=torch.float32)
                observed, mask, meta = build_action_conditions(
                    model, action, normalized_history, generated_offset=52, device="cpu",
                )
                self.assertEqual(tuple(observed.shape), (1, history_len + 52, 414))
                self.assertEqual(int(mask.count_nonzero()), expected_channels)
                self.assertEqual(int(mask[:, :history_len].count_nonzero()), 0)
                self.assertEqual(meta["conditioned_channels"], expected_channels)
                self.assertEqual(meta["window_keyframes"], [history_len + n - 52 for n in meta["reference_keyframes"]])


if __name__ == "__main__":
    unittest.main()
