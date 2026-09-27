"""Source selection contracts; numerical support checks are not animation acceptance."""
import json
from pathlib import Path
import unittest

import numpy as np

from native_wait_pose import select_native_wait_pose


class NativeWaitPoseTests(unittest.TestCase):
    def source(self, frames=30):
        pose = np.zeros((22, 3), dtype=np.float32)
        pose[0] = [0, 1., 0]
        pose[9] = [0, 1.5, 0]
        for hip, knee, ankle, toe, x in ((1, 4, 7, 10, -.12), (2, 5, 8, 11, .12)):
            pose[hip] = [x, .95, 0]
            pose[knee] = [x, .50, .015]
            pose[ankle] = [x, .07, 0]
            pose[toe] = [x, .015, .15]
        pose[20], pose[21] = [-.18, .85, 0], [.18, .85, 0]
        return np.repeat(np.stack([pose, pose+[2, 0, 0]])[None], frames, axis=0).astype(np.float32)

    def test_better_supported_pose_can_have_high_hands(self):
        source = self.source()
        source[:10, 1, [8, 11], 1] += .075
        source[10:, 1, [20, 21], 1] += .7
        before = source.copy()
        pose, report = select_native_wait_pose(source, 1)
        self.assertGreaterEqual(report['source_frame'], 10)
        self.assertGreater(report['selected_metrics']['mean_hand_height_above_hips_m'], .08)
        self.assertLess(report['selected_metrics']['ankle_height_asymmetry_m'], .001)
        np.testing.assert_array_equal(pose, source[report['source_frame'], 1])
        np.testing.assert_array_equal(source, before)
        self.assertEqual(pose.dtype, source.dtype)
        pose[:] = 0
        np.testing.assert_array_equal(source, before)

    def test_toe_contact_does_not_hide_raised_ankle(self):
        source = self.source()
        source[:10, 1, 8, 1] += .065
        _, report = select_native_wait_pose(source, 1)
        self.assertGreaterEqual(report['source_frame'], 10)
        self.assertEqual(report['baseline_metrics']['toe_height_asymmetry_m'], 0)
        self.assertGreater(report['baseline_metrics']['ankle_height_asymmetry_m'], .06)

    def test_strong_lean_and_crouch_are_ineligible(self):
        source = self.source()
        source[:10, 1, 9, 2] += .6
        source[10:20, 1, 0, 1] = .2
        _, report = select_native_wait_pose(source, 1)
        self.assertGreaterEqual(report['source_frame'], 20)
        self.assertEqual(report['ineligible_frame_counts']['strong_torso_lean'], 10)
        self.assertEqual(report['ineligible_frame_counts']['crouched_or_degenerate_legs'], 10)

    def test_continuously_moving_feet_do_not_become_idle_candidates(self):
        source = self.source()
        source[:, 1, [7, 8, 10, 11], 0] += np.arange(len(source))[:, None]*.04
        pose, report = select_native_wait_pose(source, 1)
        self.assertEqual(report['eligible_frame_count'], 0)
        self.assertEqual(report['source_frame'], 0)
        self.assertIn('fallback', report['selection_reason'])
        self.assertIn('feet_moving_in_source', report['selected_metrics']['failed_gates'])
        np.testing.assert_array_equal(pose, source[0, 1])

    def test_actor_above_partner_is_not_normalized_to_its_own_floor(self):
        source = self.source()
        source[:, 1, :, 1] += .25
        _, report = select_native_wait_pose(source, 1)
        self.assertEqual(report['eligible_frame_count'], 0)
        self.assertIn('support_high_above_native_floor_proxy', report['selected_metrics']['failed_gates'])
        self.assertFalse(report['physical_contact_verified'])

    def test_whole_pair_suspension_never_claims_grounding(self):
        source = self.source()
        source[..., 1] += 1.
        pose, report = select_native_wait_pose(source, 1)
        self.assertFalse(report['support_reference']['scene_floor_known'])
        self.assertFalse(report['support_reference']['rendered_sole_contact_known'])
        self.assertFalse(report['physical_contact_verified'])
        self.assertGreater(pose[10, 1], 1.)
        json.dumps(report, allow_nan=False)

    def test_suitable_first_pose_is_stable_under_tiny_changes(self):
        source = self.source()
        source[12:, 1, 9, 2] = .0001
        _, report = select_native_wait_pose(source, 1)
        self.assertEqual(report['source_frame'], 0)

    def test_degenerate_legs_and_invalid_input_are_rejected_honestly(self):
        source = self.source()
        source[:, 1, [1, 4, 7, 2, 5, 8]] = source[:, 1, 0:1]
        _, report = select_native_wait_pose(source, 1)
        self.assertEqual(report['eligible_frame_count'], 0)
        for bad, actor in [(source[:3], 1), (source, 2), (np.full_like(source, np.nan), 0)]:
            with self.assertRaises(ValueError):
                select_native_wait_pose(bad, actor)

    def test_recorded_source_reduces_support_asymmetry_despite_raised_hands(self):
        path = Path(__file__).parent/'review/main-cast-ui/generation/pair-01.npz'
        with np.load(path) as data:
            source = data['joints']
        pose, report = select_native_wait_pose(source, 1)
        before, after = report['baseline_metrics'], report['selected_metrics']
        self.assertLess(after['ankle_height_asymmetry_m'], before['ankle_height_asymmetry_m']/2)
        self.assertLess(after['toe_height_asymmetry_m'], before['toe_height_asymmetry_m']/2)
        self.assertLess(after['torso_tilt_degrees'], before['torso_tilt_degrees'])
        self.assertGreater(after['mean_hand_height_above_hips_m'], .08)
        np.testing.assert_array_equal(pose, source[report['source_frame'], 1])

    def test_final_market_sources_remain_exact_and_report_support_limits(self):
        root = Path(__file__).parent/'review/interaction-quality/final'
        files = sorted(root.glob('market*/sources/*/pair-01.npz'))
        self.assertEqual(len(files), 3)
        for path in files:
            with self.subTest(path=path), np.load(path) as data:
                source = data['joints']
                pose, report = select_native_wait_pose(source, 1)
                np.testing.assert_array_equal(pose, source[report['source_frame'], 1])
                self.assertFalse(report['physical_contact_verified'])
                if 's42' in str(path):
                    self.assertGreater(max(report['selected_metrics']['toe_heights_native_m']), .04)
                    self.assertFalse(report['support_reference']['scene_floor_known'])


if __name__ == '__main__':
    unittest.main()
