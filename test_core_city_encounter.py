"""CPU-only city encounter planning and measurement contracts."""

import unittest
from types import SimpleNamespace
import json
import math
from pathlib import Path

import numpy as np

from core_action_library import SAMPLE_FRAMES
from core_city_encounter import build_city_encounter, measure_city_encounter
from realtime_backend import validate_job
from realtime_client import request_body
from realtime_clip import CanonicalClip
from realtime_director import RealtimeDirector
from scene_composition import make_preset
from scene_objects import make_object


class CityEncounterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.city = make_preset("City boulevard", 0)

    def _planned_geometry_clip(self, report):
        """Idealized track used only to isolate measurement verdicts."""
        frames = report["planned_frames"]
        positions = np.zeros((2, frames, 27, 3), np.float32)
        ids = ("actor_1", "actor_2")
        for actor_index, aid in enumerate(ids):
            previous = report["initial_placements"][aid]["position_xz"]
            for stage_index, targets in enumerate(report["planned_stage_root_targets"]):
                samples = targets[aid]
                sample_frames = [0] + [item["frame"] for item in samples]
                sample_points = [previous] + [item["position_xz"] for item in samples]
                for frame in range(40):
                    segment = next((j for j in range(1, len(sample_frames))
                                    if frame <= sample_frames[j]), len(sample_frames) - 1)
                    fraction = ((frame - sample_frames[segment - 1]) /
                                max(1, sample_frames[segment] - sample_frames[segment - 1]))
                    point = (np.asarray(sample_points[segment - 1]) * (1 - fraction) +
                             np.asarray(sample_points[segment]) * fraction)
                    positions[actor_index, stage_index * 40 + frame, :, 0] = point[0]
                    positions[actor_index, stage_index * 40 + frame, :, 2] = point[1]
                previous = sample_points[-1]
        rotations = np.broadcast_to(np.eye(3, dtype=np.float32),
                                    (2, frames, 27, 3, 3)).copy()
        for actor_index in range(2):
            meeting = report["planned_meeting_frame"]
            toward = (positions[1 - actor_index, meeting, 0] -
                      positions[actor_index, meeting, 0])
            angle = math.atan2(float(toward[0]), float(toward[2]))
            rotations[actor_index, :, 0] = np.array([
                [math.cos(angle), 0, math.sin(angle)], [0, 1, 0],
                [-math.sin(angle), 0, math.cos(angle)]], dtype=np.float32)
        return positions, rotations, np.zeros((2, frames, 330), np.float32)

    def test_measured_geometry_rejects_floating_actor_and_interior_teleport(self):
        _, _, report = build_city_encounter(self.city)
        positions, rotations, features = self._planned_geometry_clip(report)

        def measure(sample):
            clip = CanonicalClip(sample, rotations, 20, ("actor_1", "actor_2"),
                                 "ardy_core", native_features=features)
            return measure_city_encounter(clip, self.city, report)

        baseline = measure(positions)
        self.assertTrue(baseline["geometry_checks_pass"])
        self.assertTrue(baseline["actors"]["actor_1"]["vertical_support_proxy_pass"])
        self.assertTrue(all(item["route_target_proxy_pass"]
                            for item in baseline["action_measurements"]))

        floating = positions.copy()
        floating[0, :, :, 1] += 4.
        floating_result = measure(floating)
        self.assertFalse(floating_result["geometry_checks_pass"])
        self.assertFalse(floating_result["actors"]["actor_1"]["vertical_support_proxy_pass"])
        self.assertGreater(floating_result["actors"]["actor_1"]["floor_motion"][
            "toe_min_height_above_floor_m"], 3.9)

        mostly_airborne = positions.copy()
        heights = np.zeros(report["planned_frames"], np.float32)
        heights[20:60] = np.linspace(0, 4, 40, dtype=np.float32)
        heights[60:460] = 4.
        heights[460:500] = np.linspace(4, 0, 40, dtype=np.float32)
        mostly_airborne[0, :, :, 1] += heights[:, None]
        airborne_result = measure(mostly_airborne)
        self.assertFalse(airborne_result["geometry_checks_pass"])
        self.assertFalse(airborne_result["actors"]["actor_1"]["vertical_support_proxy_pass"])
        self.assertGreater(airborne_result["actors"]["actor_1"][
            "longest_unsupported_run_frames"], 400)

        teleported = positions.copy()
        teleported[0, 15, :, 0] += 1.
        teleport_result = measure(teleported)
        self.assertFalse(teleport_result["geometry_checks_pass"])
        self.assertFalse(teleport_result["actors"]["actor_1"]["root_continuity_proxy_pass"])
        self.assertFalse(all(item["route_target_proxy_pass"]
                             for item in teleport_result["action_measurements"]))

    def test_default_city_builds_full_contactless_story_with_valid_native_targets(self):
        placements, stages, report = build_city_encounter(self.city, seed=101)
        self.assertEqual(set(placements), {"actor_1", "actor_2"})
        self.assertEqual(report["status"], "planned_unobserved")
        self.assertEqual(report["contact_mode"], "staged_no_contact")
        self.assertTrue(report["planned_path_ground_support"]["actor_1"]["continuous_support_verified"])
        self.assertGreaterEqual(report["planned_min_root_separation_m"], .9)
        self.assertEqual(report["planned_frames"], len(stages) * 40)
        self.assertEqual(stages[0].metadata["initial_placements"], placements)
        self.assertEqual(stages[-1].metadata["city_encounter"]["actions"]["actor_2"], "depart")
        self.assertTrue(any(stage.metadata["city_encounter"]["actions"]["actor_1"] == "lunge"
                            for stage in stages))
        director = RealtimeDirector(("actor_1", "actor_2"))
        director.queue_sequence(stages)
        for index, stage in enumerate(stages):
            self.assertEqual(stage.metadata["seed"], 101 + index)
            self.assertEqual(set(stage.metadata["root_targets"]), {"actor_1", "actor_2"})
            for aid in ("actor_1", "actor_2"):
                self.assertTrue(set(SAMPLE_FRAMES) <= {goal["frame"] for goal in stage.metadata["root_targets"][aid]})
                self.assertEqual(stage.metadata["action_candidates"][aid]["status"], "planned_unobserved")
            body = request_body(SimpleNamespace(request_id=f"test-{index}", stage_kind=stage.kind,
                frames=40, source=stage.source, prompt=stage.prompt, actor_ids=("actor_1", "actor_2"),
                actor_prompts=stage.actor_prompts, metadata=stage.metadata, history=None))
            self.assertEqual(validate_job(body)["seed"], 101 + index)

    def test_routes_and_timing_change_the_plan_without_claiming_motion_success(self):
        reports = {}
        for route in ("direct", "west", "east"):
            for timing in ("measured", "brisk"):
                _, stages, report = build_city_encounter(self.city, route_variant=route,
                                                          timing_variant=timing)
                reports[(route, timing)] = report
                self.assertEqual(report["observed_output"], None)
                self.assertEqual(report["planned_windows"], len(stages))
        self.assertLess(reports[("west", "measured")]["meeting_positions_xz"]["actor_1"][0],
                        reports[("direct", "measured")]["meeting_positions_xz"]["actor_1"][0])
        self.assertGreater(reports[("east", "measured")]["meeting_positions_xz"]["actor_1"][0],
                           reports[("direct", "measured")]["meeting_positions_xz"]["actor_1"][0])
        self.assertGreater(reports[("direct", "measured")]["planned_frames"],
                           reports[("direct", "brisk")]["planned_frames"])

    def test_generated_city_scene_variants_use_clear_street_anchors(self):
        for name in ("generated-art-deco-city", "generated-rainy-city"):
            with self.subTest(scene=name):
                scene = json.loads((Path(__file__).parent / "examples/scenes" / f"{name}.json").read_text())
                placements, stages, report = build_city_encounter(scene)
                self.assertGreaterEqual(report["planned_min_root_separation_m"], .9)
                self.assertGreater(len(stages), 5)
                self.assertEqual(placements["actor_1"]["position_xz"], [-1.2, 0.])

    def test_single_street_obstacle_forces_longer_route(self):
        _, _, clear = build_city_encounter(self.city)
        blocked = make_preset("City boulevard", 0)
        crate = make_object("crate", len(blocked["objects"]))
        crate.update(id="street-crate", position=[-.8, .5, -1.1], size=[.5, 1., .5])
        blocked["objects"].append(crate)
        _, _, detour = build_city_encounter(blocked)
        self.assertGreater(detour["planned_route_lengths_m"]["actor_1"],
                           clear["planned_route_lengths_m"]["actor_1"] + .2)
        self.assertGreaterEqual(detour["planned_min_root_separation_m"], .9)

    def test_rejects_overlap_obstacle_and_unsupported_ground(self):
        with self.assertRaisesRegex(ValueError, "separation"):
            build_city_encounter(self.city, placements={
                "actor_1": {"position_xz": [0, 0]}, "actor_2": {"position_xz": [0, 1]}})
        blocked = make_preset("City boulevard", 0)
        crate = make_object("crate", len(blocked["objects"]))
        crate.update(id="blocking-crate", position=[-1.2, .5, 0.], size=[1., 1., 1.])
        blocked["objects"].append(crate)
        with self.assertRaisesRegex(ValueError, "overlaps scene geometry"):
            build_city_encounter(blocked)
        small_floor = make_object("platform", 0)
        small_floor.update(position=[0, -.1, 0], size=[6., .2, 6.])
        bare = {"version": 2, "name": "Small floor", "objects": [small_floor],
                "effects": [], "lighting": "neutral"}
        with self.assertRaises(ValueError):
            build_city_encounter(bare, placements={
                "actor_1": {"position_xz": [-3.2, 0.]},
                "actor_2": {"position_xz": [3.2, 0.]}})
        no_floor = {"version": 2, "name": "No floor", "objects": [],
                    "effects": [], "lighting": "neutral"}
        with self.assertRaisesRegex(ValueError, "requires authored, continuous ground"):
            build_city_encounter(no_floor, placements={
                "actor_1": {"position_xz": [-1.2, 0.]},
                "actor_2": {"position_xz": [1.2, -6.]}})

    def test_measurement_waits_for_complete_clip_and_never_claims_contact(self):
        _, _, report = build_city_encounter(self.city)
        self.assertEqual(measure_city_encounter(None, self.city, report)["status"], "not_verified")
        frames = report["planned_frames"]
        positions = np.zeros((2, frames, 27, 3), np.float32)
        rotations = np.broadcast_to(np.eye(3, dtype=np.float32),
                                    (2, frames, 27, 3, 3)).copy()
        partial = CanonicalClip(positions[:, :40], rotations[:, :40], 20,
                                ("actor_1", "actor_2"), "ardy_core")
        self.assertEqual(measure_city_encounter(partial, self.city, report)["status"], "not_verified")
        complete = CanonicalClip(positions, rotations, 20,
                                 ("actor_1", "actor_2"), "ardy_core")
        result = measure_city_encounter(complete, self.city, report)
        self.assertEqual(result["status"], "measured")
        self.assertFalse(result["geometry_checks_pass"])
        self.assertFalse(result["physical_contact_verified"])
        self.assertIn("continuity", result["actors"]["actor_1"])
        self.assertIn("actor_1", result["mutual_facing_at_meeting"])
        self.assertIn("horizon_seam_max_root_step_m", result["actors"]["actor_1"])
        self.assertGreater(result["pair_separation"]["root_disc_overlap_proxy_frames"], 0)


if __name__ == "__main__":
    unittest.main()
