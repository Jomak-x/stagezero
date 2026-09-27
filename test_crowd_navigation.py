"""Behavioral checks for the reproducible crossing trial."""

import json
import unittest

import numpy as np

from crowd_navigation import (
    CROSS, WAIT, CrowdConfig, simulate_crowd, traffic_phase,
)


class CrowdNavigationTests(unittest.TestCase):
    def test_phase_cycle_and_curb_wait(self):
        cfg = CrowdConfig(count=16, duration_s=20)
        self.assertEqual(traffic_phase(0, cfg), ("red", 0))
        self.assertEqual(traffic_phase(15, cfg), ("walk", 0))
        self.assertEqual(traffic_phase(53, cfg), ("clearance", 0))
        self.assertEqual(traffic_phase(60, cfg), ("red", 1))
        result = simulate_crowd(cfg)
        sample = result.trajectories
        self.assertTrue(np.all(sample[: 15 * cfg.sample_hz, :, 4] == WAIT))
        self.assertTrue(np.any(sample[17 * cfg.sample_hz, :, 4] == CROSS))
        self.assertTrue(np.isfinite(sample).all())
        self.assertTrue(np.all(sample[:, :, :2] <= cfg.world_half_extent_m))

    def test_seed_repeats_routes_and_motion(self):
        cfg = CrowdConfig(count=32, seed=42, duration_s=25)
        first, second = simulate_crowd(cfg), simulate_crowd(cfg)
        np.testing.assert_array_equal(first.trajectories, second.trajectories)
        self.assertEqual(first.agents, second.agents)
        changed = simulate_crowd(CrowdConfig(count=32, seed=43, duration_s=25))
        self.assertFalse(np.array_equal(first.trajectories, changed.trajectories))

    def test_groups_share_routes_and_starts(self):
        result = simulate_crowd(CrowdConfig(count=64, group_fraction=0.5, duration_s=2))
        grouped = {}
        for agent in result.agents:
            if agent["group_id"] is not None:
                grouped.setdefault(agent["group_id"], []).append(agent)
        self.assertTrue(grouped)
        for members in grouped.values():
            self.assertIn(len(members), (2, 3))
            self.assertEqual(len({m["route"] for m in members}), 1)
            self.assertEqual(len({m["launch_delay_s"] for m in members}), 1)
            points = np.array([m["start_xz"] for m in members])
            gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
            self.assertLessEqual(float(gaps.max()), 1.91)
            self.assertGreaterEqual(float(gaps[gaps > 0].min()), 0.78)

    def test_full_plan_scale_has_measured_constraints(self):
        result = simulate_crowd(CrowdConfig(count=100, seed=42, duration_s=120))
        metrics = result.metrics
        self.assertGreaterEqual(metrics["crossings_completed"], 190)
        self.assertEqual(metrics["deadlock_agents"], 0)
        self.assertLessEqual(metrics["overlapping_pair_steps"], 2)
        self.assertGreater(metrics["minimum_disc_clearance_m"], -0.02)
        self.assertGreater(metrics["preprojection_overlap_pair_steps"], 0)
        self.assertGreater(metrics["projection_total_m"], 0)
        self.assertEqual(metrics["root_boundary_violations"], 0)
        self.assertEqual(metrics["red_all_inside_agent_steps"], 0)
        self.assertGreater(metrics["acceleration_max_mps2"], 10)
        self.assertEqual(result.trajectories.shape, (1801, 100, 6))

    def test_export_contract_and_validation(self):
        result = simulate_crowd(CrowdConfig(count=16, duration_s=2))
        exported = result.to_dict()
        self.assertEqual(exported["schema"], "stagezero.crowd.v1")
        self.assertEqual(exported["person_columns"],
                         ["x", "z", "heading", "speed", "state", "distance"])
        self.assertEqual(len(exported["frames"][0]["people"]), 16)
        json.dumps(exported, allow_nan=False)
        with self.assertRaises(ValueError):
            CrowdConfig(sample_hz=17)
        with self.assertRaises(ValueError):
            simulate_crowd({"count": 16, "unknown": 1})


if __name__ == "__main__":
    unittest.main()
