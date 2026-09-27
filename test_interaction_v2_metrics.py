"""Regression checks for grounded diagnostics on retained PR26 motion archives."""

import unittest

import numpy as np

from experiments.review_interaction_v2 import LEGACY, ROOT, compare_archives, ground_height, measure


class InteractionV2MetricsTest(unittest.TestCase):
    def test_ground_height_uses_scene_slab_and_reports_unsupported_points(self):
        import json

        scene = json.loads((ROOT / "review/prompt-scenes/backgrounds/market.json").read_text())
        floor = ground_height(scene, np.array([[0., 0.], [50., 50.]]))
        self.assertTrue(np.isfinite(floor[0]))
        self.assertAlmostEqual(float(floor[0]), -.01, places=5)
        self.assertTrue(np.isnan(floor[1]))

    def test_retained_overlap_incidents_and_held_spans_are_visible(self):
        industrial = measure(LEGACY / "final/industrial-spar-s45-fixed-plan/scene.cast.stagezero.npz")
        market = measure(LEGACY / "final/market-three-s49-fixed-plan/scene.cast.stagezero.npz")
        self.assertEqual(8, sum(item["overlap_frames"] for item in industrial["body_overlap_proxy"]
                                if item["window"] == "unintended"))
        self.assertEqual(10, sum(item["overlap_frames"] for item in market["body_overlap_proxy"]
                                 if item["window"] == "unintended"))
        self.assertEqual(2, len(market["intended_hand_contact_proxy"]))
        self.assertEqual(1, len(market["pair_handoffs"]))
        self.assertTrue(any(actor["held_spans"] for actor in market["performers"]))
        self.assertTrue(all(foot["scene_support_coverage_frames"] == market["frames"]
                            for actor in market["performers"] for foot in actor["feet"]))

    def test_raw_unrefined_composition_preserves_active_pair_frames(self):
        final = LEGACY / "final/market-three-s48-fixed-plan"
        raw = next((final / "sources").glob("scene-*/unrefined-cast.npz"))
        comparison = compare_archives(raw, final / "scene.cast.stagezero.npz")
        self.assertTrue(comparison["before_is_raw_unrefined_joints"])
        self.assertEqual(0, comparison["root_change"]["changed_root_frames_over_1mm"])
        self.assertTrue(all(item["active_intergen_display_frames_equal"]
                            for item in comparison["paired_display_checks"]))


if __name__ == "__main__":
    unittest.main()
