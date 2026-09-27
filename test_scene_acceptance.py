"""Synthetic, CPU-only checks for the saved-scene acceptance report."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from motion_quality import JOINT_INDEX, ROOT, SHOULDERS
from scene_acceptance import analyze_archive, analyze_take, main
from takes import Take, encode_project


def standing(frames=25):
    positions = np.zeros((frames, 34, 3), dtype=np.float32)
    positions[:, ROOT, 1] = .9
    positions[:, SHOULDERS, 1] = 1.3
    for side in ("left", "right"):
        positions[:, JOINT_INDEX[f"{side}_hip_yaw_skel"], 1] = .9
        positions[:, JOINT_INDEX[f"{side}_knee_skel"], 1] = .45
    return positions


def take(positions, prompts):
    assert sum(frames for _, frames in prompts) == len(positions)
    rotations = np.tile(np.eye(3, dtype=np.float32), (len(positions), 34, 1, 1))
    motion = np.zeros((len(positions), 414), dtype=np.float32)
    cursor, segments = 0, []
    for prompt, frames in prompts:
        segments.append({"start": cursor, "end": cursor + frames, "prompt": prompt})
        cursor += frames
    return Take("synthetic", "Synthetic scene", positions, rotations, motion, segments)


def checks(report):
    return {check["name"]: check for segment in report["segments"] for check in segment["checks"]}


class SceneAcceptanceTests(unittest.TestCase):
    def test_coverage_and_continuity_detect_teleport_without_travel_label(self):
        positions = standing(24)
        positions[12:, :, 0] += 1.0
        report = analyze_take(take(positions, [("A person stands still.", 12),
                                               ("A person stands still.", 12)]))
        self.assertEqual(report["segment_count"], 2)
        self.assertTrue(report["structural_checks"][0]["passed"])
        self.assertFalse(checks(report)["root_seam_continuity"]["passed"])
        self.assertFalse(report["passed"])

    def test_requested_distance_uses_actual_planar_net_displacement(self):
        positions = standing(100)
        positions[:, :, 0] += np.linspace(0, 5, len(positions))[:, None]
        report = analyze_take(take(positions, [("A person sprints forward for 20 meters.", 100)]))
        self.assertAlmostEqual(report["segments"][0]["metrics"]["root_net_displacement_m"], 5.0)
        self.assertFalse(checks(report)["travel_displacement"]["passed"])
        positions[:, :, 0] += np.linspace(0, 11, len(positions))[:, None]
        self.assertTrue(checks(analyze_take(take(positions, [("A person sprints forward for 20 meters.", 100)])))
                        ["travel_displacement"]["passed"])

    def test_stationary_dance_fails_activity_without_claiming_style_classification(self):
        positions = standing(25)
        positions[:, JOINT_INDEX["left_hand_roll_skel"], 0] = np.linspace(0, .5, 25)
        report = analyze_take(take(positions, [("A person dances in place.", 25)]))
        self.assertEqual(report["segments"][0]["checks"][0]["name"], "sustained_motion_activity")
        self.assertFalse(report["segments"][0]["checks"][0]["passed"])
        self.assertGreater(report["segments"][0]["metrics"]
                           ["pose_mean_joint_speed_body_lengths_per_second"], 0)
        self.assertIn("Geometric proxies only", report["interpretation"])

    def test_backflip_requires_sustained_inversion_and_upright_landing(self):
        positions = standing(25)
        positions[5:8, SHOULDERS, 1] = .5
        good = analyze_take(take(positions, [("A person performs a backflip.", 25)]))
        self.assertTrue(checks(good)["backflip_torso_inversion"]["passed"])
        self.assertTrue(checks(good)["backflip_terminal_upright"]["passed"])
        brief = positions.copy()
        brief[6:8, SHOULDERS, 1] = 1.3
        self.assertFalse(checks(analyze_take(take(brief, [("A person performs a backflip.", 25)])))
                         ["backflip_torso_inversion"]["passed"])
        bad_landing = positions.copy()
        bad_landing[-10:, SHOULDERS, 1] = .5
        self.assertFalse(checks(analyze_take(take(bad_landing, [("A person performs a backflip.", 25)])))
                         ["backflip_terminal_upright"]["passed"])

    def test_fall_recovery_and_stand_check_terminal_pose(self):
        upright = standing(25)
        low = upright.copy()
        low[-5:, ROOT, 1] = .2
        low[-5:, SHOULDERS, 1] = .3
        self.assertTrue(checks(analyze_take(take(low, [("A person falls to the ground.", 25)])))
                        ["fall_terminal_low_pose"]["passed"])
        self.assertFalse(checks(analyze_take(take(upright, [("A person falls to the ground.", 25)])))
                         ["fall_terminal_low_pose"]["passed"])
        prone = upright.copy()
        prone[-10:, SHOULDERS, 1] = .9
        for prompt, name in (("A person gets up from the ground.", "recovery_terminal_upright"),
                             ("A person stands upright.", "standing_terminal_upright")):
            self.assertFalse(checks(analyze_take(take(prone, [(prompt, 25)])))[name]["passed"])
            self.assertTrue(checks(analyze_take(take(upright, [(prompt, 25)])))[name]["passed"])

    def test_archive_cli_writes_json_and_failure_exit_code(self):
        positions = standing(25)
        source = take(positions, [("A person performs a backflip.", 25)])
        archive = encode_project({source.id: source}, source.id, 0, {})
        report = analyze_archive(archive)
        self.assertEqual(report["selected_take_id"], source.id)
        self.assertFalse(report["take"]["passed"])
        with tempfile.TemporaryDirectory() as directory:
            source_path, report_path = Path(directory) / "scene.stagezero.npz", Path(directory) / "report.json"
            source_path.write_bytes(archive)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                status = main([str(source_path), "--output", str(report_path)])
            self.assertEqual(status, 1)
            self.assertEqual(json.loads(output.getvalue()), json.loads(report_path.read_text()))


if __name__ == "__main__":
    unittest.main()
