"""The local character lab presents an anatomically readable FK diagnostic."""

import unittest

import numpy as np

from asset_viewer import diagnostic_clip
from retargeting import neutral_source_pose


class AssetDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from asset_viewer import G1Skeleton34

        cls.skeleton = G1Skeleton34()
        cls.positions, cls.rotations = diagnostic_clip(cls.skeleton)

    def test_stand_pose_has_narrow_forward_bent_legs_and_level_grounded_feet(self):
        positions = self.positions[0]
        skeleton = self.skeleton
        for side in ('left', 'right'):
            hip = positions[skeleton.bone_index[f'{side}_hip_pitch_skel']]
            knee = positions[skeleton.bone_index[f'{side}_knee_skel']]
            ankle = positions[skeleton.bone_index[f'{side}_ankle_roll_skel']]
            thigh, shin = knee - hip, ankle - knee
            thigh_pitch = np.degrees(np.arctan2(thigh[2], -thigh[1]))
            shin_pitch = np.degrees(np.arctan2(shin[2], -shin[1]))
            with self.subTest(side=side):
                # The joint chain has a 52 mm outward hip motor offset. Its
                # rotations must bring both knee and ankle beneath the hip.
                self.assertLess(abs(knee[0] - hip[0]), .006)
                self.assertLess(abs(ankle[0] - hip[0]), .006)
                self.assertLess(abs(ankle[0] - knee[0]), .003)
                self.assertGreater(thigh_pitch - shin_pitch, 11)
                self.assertLess(thigh_pitch - shin_pitch, 13)
                self.assertGreater(knee[2] - hip[2], .03)
                self.assertGreater(knee[2] - ankle[2], .03)
                self.assertLess(abs(ankle[2] - hip[2]), .008)
                self.assertLess(thigh[1], -.3)
                self.assertLess(shin[1], -.3)
                toe = positions[skeleton.bone_index[f'{side}_toe_base']]
                self.assertAlmostEqual(float(toe[1]), 0, places=6)
                np.testing.assert_allclose(
                    self.rotations[0, skeleton.bone_index[f'{side}_ankle_roll_skel']],
                    np.eye(3), atol=1e-9,
                )

    def test_both_arms_only_sweep_outward_and_return(self):
        skeleton = self.skeleton
        self.assertEqual(self.positions.shape, (150, 34, 3))
        np.testing.assert_allclose(self.positions[0], self.positions[-1], atol=1e-9)
        np.testing.assert_allclose(self.rotations[0], self.rotations[-1], atol=1e-9)
        for side, sign in (('left', 1), ('right', -1)):
            shoulder = skeleton.bone_index[f'{side}_shoulder_pitch_skel']
            elbow = skeleton.bone_index[f'{side}_elbow_skel']
            wrist = skeleton.bone_index[f'{side}_wrist_yaw_skel']
            for endpoint in (elbow, wrist):
                distance = sign * (self.positions[:, endpoint, 0] - self.positions[:, shoulder, 0])
                baseline = distance[0]
                with self.subTest(side=side, endpoint=endpoint):
                    self.assertGreater(baseline, 0)
                    self.assertGreater(float(np.max(distance)), baseline + .04)
                    self.assertGreaterEqual(float(np.min(distance)), baseline - 1e-6)
            elbow_position = self.positions[0, elbow]
            wrist_position = self.positions[0, wrist]
            self.assertLess(float(wrist_position[1] - elbow_position[1]), -.15)
            self.assertLess(abs(float(wrist_position[2] - elbow_position[2])), .02)

    def test_global_rotations_and_positions_are_one_fk_pose(self):
        neutral = self.skeleton.neutral_joints.numpy()
        parents = self.skeleton.joint_parents
        for joint in range(1, 34):
            parent = int(parents[joint])
            offset = neutral[joint] - neutral[parent]
            predicted = self.positions[:, parent] + self.rotations[:, parent] @ offset
            np.testing.assert_allclose(self.positions[:, joint], predicted, atol=1e-9)

    def test_diagnostic_does_not_change_production_calibration(self):
        positions, rotations = neutral_source_pose(self.skeleton)
        neutral = self.skeleton.neutral_joints.numpy()
        np.testing.assert_allclose(positions - positions[0], neutral - neutral[0], atol=1e-10)
        np.testing.assert_allclose(rotations, np.tile(np.eye(3), (34, 1, 1)), atol=1e-10)
        self.assertGreater(abs(positions[self.skeleton.bone_index['left_knee_skel'], 2] -
                               positions[self.skeleton.bone_index['left_hip_pitch_skel'], 2]), .04)


if __name__ == '__main__':
    unittest.main()
