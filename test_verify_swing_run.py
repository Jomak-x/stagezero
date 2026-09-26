"""Independent checks that the run auditor catches geometric corruption."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.verify_swing_run import (action_sequence, audit, box_depths,
                                          segment_clearance_samples, sphere_box_penetration,
                                          validate_command_states)


ROOT = Path(__file__).resolve().parent


class VerifySwingRunTests(unittest.TestCase):
    def test_sequence_requires_airborne_and_attached_state(self):
        commands = [{"id": i + 1, "action": action} for i, action in enumerate(
            ("swing", "left", "right", "carry", "left", "land", "kiss"))]
        sequence = action_sequence(commands)
        states = {c["id"]: {"frame": (c["id"] - 1) * 100, "phase": "swing",
                            "status": "applied", "carry_amount": 0.,
                            "flight_amount": 1., "grounded": False} for c in commands}
        states[4]["phase"] = "pickup"
        states[5]["carry_amount"] = 1.
        states[6].update(phase="landing", carry_amount=1.)
        states[7].update(phase="kiss", grounded=True, flight_amount=0.)
        events = {("mj_attached", 350): {"event": "mj_attached", "frame": 350},
                  ("landing_contact", 550): {"event": "landing_contact", "frame": 550}}
        self.assertTrue(validate_command_states(sequence, states, events))
        states[5]["carry_amount"] = 0.
        self.assertFalse(validate_command_states(sequence, states, events))

    def test_sphere_box_penetration_and_sequence(self):
        box = {"min": [0, 0, 0], "max": [1, 1, 1]}
        self.assertEqual(sphere_box_penetration([3, .5, .5], .1, box), 0)
        self.assertAlmostEqual(sphere_box_penetration([1.05, .5, .5], .1, box), .05)
        self.assertFalse(action_sequence([{"action": x, "id": i} for i, x in enumerate(
            ["swing", "left", "right", "carry", "land", "kiss"])])["complete"])
        self.assertTrue(action_sequence([{"action": x, "id": i} for i, x in enumerate(
            ["swing", "left", "right", "carry", "left", "land", "kiss"])])["complete"])

    def test_full_chain_catches_arm_obstacle_and_web_crossing(self):
        with np.load(ROOT / "assets/swing-motion/swing.npz", allow_pickle=False) as archive:
            p = archive["positions"][0].astype(float)
        actors = np.stack([p, p + [3, 0, 0]])
        points, radii, segments = segment_clearance_samples(actors)
        right_hand = p[10]
        box = {"min": (right_hand - .18).tolist(), "max": (right_hand + .18).tolist()}
        depth = box_depths(points, radii, box)
        self.assertGreater(float(depth[segments == 10].max()), .15)
        self.assertLess(sphere_box_penetration(p[0], .13, box), .001)
        web = np.linspace(right_hand + [0, 0, -1], right_hand + [0, 0, 1], 21)
        self.assertGreater(float(box_depths(web, np.full(21, .025), box).max()), .15)

    def test_audit_uses_logged_poses_to_catch_bone_drift(self):
        with np.load(ROOT / "assets/swing-motion/swing.npz", allow_pickle=False) as archive:
            p = archive["positions"][0].astype(float)
            r = archive["rotations"][0].astype(float)
        positions = np.stack([p, p + [2, 0, 0]])
        rotations = np.stack([r, r])
        roof = {"id": "roof", "building_id": "none", "position": [0, float(np.mean(positions[0, [22, 26], 1])), 0],
                "radius": 100}
        scene = {"buildings": [], "roofs": [roof]}
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            with (run / "frames.jsonl").open("w") as log:
                for frame in range(3):
                    current = positions.copy()
                    if frame == 2:
                        current[0, 10, 0] += .05
                    state = {"frame": frame + 1, "time": (frame + 1) / 60,
                             "phase": "swing", "actors": [
                                 {"positions": current[i].tolist(), "rotations": rotations[i].tolist()}
                                 for i in range(2)], "metrics": {"pose": {}}}
                    log.write(json.dumps(state) + "\n")
            result = audit(run, scene, require_video=False)
            self.assertEqual(result["frames"], 3)
            self.assertTrue(result["gates"]["fixed_60hz_frames"])
            self.assertTrue(result["gates"]["finite_poses"])
            self.assertFalse(result["gates"]["bone_drift_under_1cm"])
            self.assertGreater(result["bone_length_drift_m"]["max"], .01)


if __name__ == "__main__":
    unittest.main()
