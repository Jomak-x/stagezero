"""CPU-only synthetic policy checks; no model or Pod needed."""

import unittest

import numpy as np

from motion_policy import recognize_action, select_candidate, validate_generation_options
from motion_quality import HANDS, ROOT, SHOULDERS, TOES


def candidate(frames=104):
    positions = np.zeros((frames, 34, 3), dtype=np.float64)
    positions[:, ROOT, 1] = 1.0
    positions[:, SHOULDERS, 1] = 1.4
    positions[:, HANDS, 1] = 1.0
    rotations = np.tile(np.eye(3), (frames, 34, 1, 1))
    return {"positions": positions, "rotations": rotations}


class MotionPolicyTests(unittest.TestCase):
    def test_options_resolve_profiles_and_reject_malformed_values(self):
        self.assertEqual(validate_generation_options(None), {
            "profile": "legacy", "history_frames": 52, "carry_frames": 52,
            "cfg_weight": (2.0, 2.0), "seed": None, "candidates": 1, "pose_goal": None,
        })
        self.assertEqual(validate_generation_options({"profile": "responsive", "seed": 0, "candidates": 3})["carry_frames"], 4)
        self.assertEqual(validate_generation_options({"profile": "expressive"})["cfg_weight"], (4.0, 2.0))
        for options in ({"unknown": 1}, {"profile": "fast"}, {"profile": True},
                        {"seed": True}, {"seed": -1}, {"seed": 2**32}, {"seed": float("nan")},
                        {"candidates": 0}, {"candidates": 4}, {"candidates": 1.0},
                        {"candidates": False}, {"candidates": float("inf")},
                        {"pose_goal": None}, {"pose_goal": 1}, {"pose_goal": "true"}, []):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_generation_options(options)
        self.assertFalse(validate_generation_options({"pose_goal": False})["pose_goal"])

    def test_recognizer_is_narrow_and_negation_aware(self):
        self.assertEqual(recognize_action("A person raises both arms overhead."), "overhead")
        self.assertEqual(recognize_action("Lift both hands above the head."), "overhead")
        self.assertEqual(recognize_action("A person waves with their right hand."), "wave")
        self.assertEqual(recognize_action("A person does a squat."), "squat")
        self.assertEqual(recognize_action("A person stops and stands still."), "stop")
        for prompt in ("Do not raise both arms overhead.", "Don’t raise both arms overhead.",
                       "Can’t do a squat.", "Never wave the right hand.",
                       "Walk without squatting.", "A person walks forward.", "Raise one arm overhead.",
                       "Wave the right hand then squat.",
                       "Raise both arms overhead and wave the right hand.",
                       "Squat, then stop and stand still."):
            with self.subTest(prompt=prompt):
                self.assertIsNone(recognize_action(prompt))

    def test_overhead_requires_five_consecutive_frames_and_both_hands(self):
        scattered = candidate()
        scattered["positions"][np.ix_([0, 2, 4, 6, 8], HANDS, [1])] = 1.7
        result = select_candidate("Raise both arms overhead", [scattered])
        self.assertFalse(result["assessments"][0]["proxy"]["met"])
        self.assertEqual(result["assessments"][0]["proxy"]["longest_continuous_frames"], 1)
        sustained = candidate()
        sustained["positions"][10:15, HANDS, 1] = 1.7
        result = select_candidate("Raise both arms overhead", [sustained])
        self.assertTrue(result["assessments"][0]["proxy"]["met"])

    def test_safe_candidate_beats_unsafe_high_scoring_gesture(self):
        safe = candidate()
        safe["positions"][10:16, HANDS[1], 1] = 1.7
        unsafe = candidate()
        unsafe["positions"][:, HANDS[1], 1] = 1.8
        unsafe["positions"][:, TOES[0], 1] = -.06
        original = safe["positions"].copy()
        result = select_candidate("Wave the right hand", [unsafe, safe])
        self.assertEqual(result["chosen_index"], 1)
        self.assertIn("floor_penetration", result["assessments"][0]["reasons"])
        np.testing.assert_array_equal(safe["positions"], original)

    def test_squat_uses_prior_last_four_and_stop_uses_final_25(self):
        prior = candidate(8)["positions"]
        prior[-4:, ROOT, 1] = [1.0, 1.0, 1.0, 1.0]
        squat = candidate()
        squat["positions"][40:50, ROOT, 1] = .8
        result = select_candidate("Do a squat", [squat], prior_positions=prior)
        self.assertEqual(result["chosen_index"], 0)
        self.assertTrue(result["assessments"][0]["proxy"]["met"])
        self.assertAlmostEqual(result["assessments"][0]["proxy"]["drop_m"], .2)
        self.assertIsNone(select_candidate("Do a squat", [squat])["assessments"][0]["proxy"]["met"])

        moving = candidate()
        moving["positions"][-25:, ROOT, 0] = np.arange(25) * .01
        still = candidate()
        result = select_candidate("Stop and stand still", [moving, still])
        self.assertEqual(result["chosen_index"], 1)
        self.assertFalse(result["assessments"][0]["proxy"]["met"])
        self.assertTrue(result["assessments"][1]["proxy"]["met"])

    def test_seams_rotations_and_nonfinite_arrays_fail_closed(self):
        prior = candidate(4)["positions"]
        discontinuous = candidate()
        discontinuous["positions"][:, :, 0] += .2
        invalid_rotation = candidate()
        invalid_rotation["rotations"][0, 0] *= 2
        nonfinite = candidate()
        nonfinite["positions"][0, 0, 0] = np.nan
        result = select_candidate("Walk forward", [discontinuous, invalid_rotation, nonfinite], prior_positions=prior)
        self.assertIsNone(result["chosen_index"])
        self.assertIn("history_seam", result["assessments"][0]["reasons"])
        self.assertIn("invalid_rotations", result["assessments"][1]["reasons"])
        self.assertIn("nonfinite_positions", result["assessments"][2]["reasons"])
        horizon_jump = candidate()
        horizon_jump["positions"][52:, :, 0] += .2
        result = select_candidate("Walk forward", [horizon_jump])
        self.assertIn("horizon_seam", result["assessments"][0]["reasons"])
        mid_clip_jump = candidate()
        mid_clip_jump["positions"][73:, :, 0] += .2
        result = select_candidate("Walk forward", [mid_clip_jump])
        self.assertIsNone(result["chosen_index"])
        self.assertIn("intra_clip_jump", result["assessments"][0]["reasons"])


if __name__ == "__main__":
    unittest.main()
