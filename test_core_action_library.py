"""Contract checks for honest, reproducible Core action records."""

import unittest

from core_action_library import MODEL, PROMPTS, SAMPLE_FRAMES, make_action_candidate, record_action_outcome


def targets():
    return [{"frame": frame, "position_xz": [0., frame / 40], "heading": 0.}
            for frame in SAMPLE_FRAMES]


class ActionLibraryTests(unittest.TestCase):
    def test_every_requested_action_has_an_unobserved_reproducible_candidate(self):
        self.assertEqual(set(PROMPTS), {"walk", "stop", "turn", "meet", "guard",
                                        "lunge", "dodge", "hit_reaction", "retreat", "depart"})
        for action in PROMPTS:
            candidate = make_action_candidate(action, "actor_1", seed=301,
                                              root_targets=targets(), conditions={"layout": "city"})
            self.assertEqual(candidate["model"], MODEL)
            self.assertEqual(candidate["seed"], 301)
            self.assertEqual(candidate["native_constraints"]["root_targets"], targets())
            self.assertIsNone(candidate["output"])
            self.assertIsNone(candidate["measurement"])
            self.assertEqual(candidate["status"], "planned_unobserved")

    def test_result_requires_output_and_measured_evidence(self):
        candidate = make_action_candidate("lunge", "actor_1", seed=1, root_targets=targets())
        with self.assertRaisesRegex(ValueError, "output reference"):
            record_action_outcome(candidate, output={}, measurement={"visual_review": "failed"},
                                  status="observed_fail")
        with self.assertRaisesRegex(ValueError, "measurements"):
            record_action_outcome(candidate, output={"path": "take.npz"}, measurement={},
                                  status="observed_pass")
        with self.assertRaisesRegex(ValueError, "positive geometry"):
            record_action_outcome(candidate, output={"path": "take.npz"},
                                  measurement={"geometry_checks_pass": True}, status="observed_pass")
        passed = record_action_outcome(candidate, output={"path": "good.npz"},
                                       measurement={"geometry_checks_pass": True,
                                                    "visual_review_pass": True,
                                                    "instruction_followed": True},
                                       status="observed_pass")
        self.assertEqual(passed["status"], "observed_pass")
        failed = record_action_outcome(candidate, output={"path": "take.npz"},
                                       measurement={"visual_review": "failed"}, status="observed_fail")
        self.assertEqual(failed["status"], "observed_fail")
        self.assertEqual(candidate["status"], "planned_unobserved")

    def test_invalid_native_constraints_rejected(self):
        bad = targets()
        bad[-1] = {"frame": 39, "position_xz": [0., 0.], "heading": 4.}
        with self.assertRaisesRegex(ValueError, "outside native bounds"):
            make_action_candidate("walk", "actor_1", seed=1, root_targets=bad)
        with self.assertRaisesRegex(ValueError, "uint32"):
            make_action_candidate("walk", "actor_1", seed=-1, root_targets=targets())
        with self.assertRaisesRegex(ValueError, "finite JSON"):
            make_action_candidate("walk", "actor_1", seed=1, root_targets=targets(),
                                  conditions={"bad": float("nan")})


if __name__ == "__main__":
    unittest.main()
