"""CPU checks for geometry-grounded real-time root conditioning."""

from __future__ import annotations

import math
import unittest

import numpy as np

from realtime_backend import validate_job
from realtime_client import request_body
from realtime_clip import CanonicalClip
from realtime_director import RealtimeDirector
from realtime_navigation import plan_navigation
from scene_objects import make_object


def scene(*objects):
    return {"version": 2, "name": "Navigation test", "objects": list(objects),
            "effects": [], "lighting": "neutral"}


def arch(z=2.4):
    item = make_object("arch", 0)
    item.update(id="gate", name="North gate", position=[0, 1.5, z], size=[3., 3., .4])
    return item


def placement():
    return {"walker": {"position_xz": [0., -2.], "yaw": 0.},
            "witness": {"position_xz": [2., -1.], "yaw": math.pi}}


def existing_clip():
    positions = np.zeros((2, 40, 27, 3), np.float32)
    positions[0, :, :, 0] = 0
    positions[0, :, :, 2] = -2
    positions[1, :, :, 0] = 2
    positions[1, :, :, 2] = -1
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32),
                                (2, 40, 27, 3, 3)).copy()
    return CanonicalClip(positions, rotations, 20, ("walker", "witness"), "ardy_core")


class NavigationTests(unittest.TestCase):
    def test_gate_route_dense_local_targets_and_stationary_other_actor(self):
        stages, route = plan_navigation(
            scene(arch()), ("walker", "witness"), actor_id="walker",
            target_id="gate", verb="go_through", initial_placements=placement())
        self.assertGreater(len(stages), 2)
        self.assertLessEqual(len(stages), 15)
        self.assertEqual(route["target_id"], "gate")
        self.assertFalse(route["assumptions"]["motion_following_verified"])
        self.assertEqual(route["schedule"]["frames"], len(stages) * 40)
        first = stages[0]
        self.assertEqual(first.metadata["initial_placements"], placement())
        self.assertEqual(set(first.metadata["root_targets"]), {"walker", "witness"})
        for stage in stages:
            self.assertEqual(stage.kind, "approach")
            self.assertEqual(stage.frames, 40)
            goals = stage.metadata["root_targets"]
            self.assertTrue(set((7, 15, 23, 31, 39)) <= {x["frame"] for x in goals["walker"]})
            self.assertTrue(set((7, 15, 23, 31, 39)) <= {x["frame"] for x in goals["witness"]})
            self.assertTrue(all(x["position_xz"] == [2., -1.] for x in goals["witness"]))
            self.assertTrue(all(abs(x["heading"] - math.pi) < 1e-5 for x in goals["witness"]))
        route_z = [x["position_xz"][1] for stage in stages for x in stage.metadata["root_targets"]["walker"]]
        self.assertGreater(route_z[-1], 2.4)
        self.assertTrue(all(b >= a - 1e-6 for a, b in zip(route_z, route_z[1:])))
        self.assertTrue(all(x["position_xz"] == stages[-1].metadata["root_targets"]["walker"][0]["position_xz"]
                            for x in stages[-1].metadata["root_targets"]["walker"]))
        self.assertTrue(all(abs(x["heading"]) < 1e-5
                            for x in stages[-1].metadata["root_targets"]["walker"]))

    def test_every_navigation_horizon_passes_backend_validation_at_pi(self):
        starts = placement()  # Stationary witness has exact +pi heading.
        stages, _ = plan_navigation(scene(arch()), ("walker", "witness"),
                                    actor_id="walker", target_id="gate",
                                    verb="go_through", initial_placements=starts)
        director = RealtimeDirector(("walker", "witness"),
                                    target_buffer_frames=600, max_buffer_frames=600)
        director.queue_sequence(stages)
        for stage in stages:
            request = director.claim_request()
            self.assertIsNotNone(request)
            body = request_body(request)
            validated = validate_job(body)
            self.assertEqual(validated["actor_ids"], ("walker", "witness"))
            stationary = stage.metadata["root_targets"]["witness"]
            self.assertTrue(all(-math.pi <= target["heading"] <= math.pi
                                for target in stationary))
            self.assertTrue(all(target["heading"] == math.pi for target in stationary))
            positions = np.zeros((2, 40, 27, 3), np.float32)
            rotations = np.broadcast_to(np.eye(3, dtype=np.float32),
                                        (2, 40, 27, 3, 3)).copy()
            native = np.zeros((2, 40, 330), np.float32)
            director.complete(request.request_id,
                              CanonicalClip(positions, rotations, 20,
                                            ("walker", "witness"), "ardy_core",
                                            native_features=native))

    def test_last_clip_stable_actor_order_and_no_replacement_poses(self):
        stages, _ = plan_navigation(scene(arch()), ("walker", "witness"),
                                    actor_id="walker", target_id="gate",
                                    verb="go_through", last_clip=existing_clip())
        self.assertNotIn("initial_placements", stages[0].metadata)
        self.assertNotIn("positions", stages[0].metadata)
        self.assertEqual(stages[0].metadata["root_targets"]["witness"][-1]["position_xz"], [2., -1.])
        with self.assertRaisesRegex(ValueError, "stable actor IDs"):
            plan_navigation(scene(arch()), ("witness", "walker"), actor_id="walker",
                            target_id="gate", verb="go_through", last_clip=existing_clip())

    def test_route_detours_around_held_actor_without_mutating_scene(self):
        source_scene = scene(arch())
        original_objects = list(source_scene["objects"])
        starts = placement()
        starts["witness"]["position_xz"] = [0., 0.]
        stages, route = plan_navigation(source_scene, ("walker", "witness"),
                                        actor_id="walker", target_id="gate",
                                        verb="go_through", initial_placements=starts)
        self.assertEqual(source_scene["objects"], original_objects)
        self.assertGreaterEqual(route["schedule"]["sampled_route_clearance_m"], .64)
        self.assertTrue(any(abs(goal["position_xz"][0]) > .64
                            for stage in stages
                            for goal in stage.metadata["root_targets"]["walker"]))
        self.assertTrue(all(goal["position_xz"] == [0., 0.]
                            for stage in stages
                            for goal in stage.metadata["root_targets"]["witness"]))

    def test_rejects_blocked_and_unknown_targets(self):
        crate = make_object("crate", 1)
        crate["position"] = [0., .4, 2.4]
        with self.assertRaisesRegex(ValueError, "blocked"):
            plan_navigation(scene(arch(), crate), ("walker", "witness"),
                            actor_id="walker", target_id="gate", verb="go_through",
                            initial_placements=placement())
        with self.assertRaisesRegex(ValueError, "Unknown target_id"):
            plan_navigation(scene(arch()), ("walker", "witness"),
                            actor_id="walker", target_id="missing", verb="go_through",
                            initial_placements=placement())

    def test_route_plus_hold_must_fit_thirty_seconds(self):
        with self.assertRaisesRegex(ValueError, "30-second"):
            plan_navigation(scene(arch(z=16.5)), ("walker", "witness"),
                            actor_id="walker", target_id="gate", verb="go_through",
                            initial_placements=placement())

    def test_single_actor_approaches_solid_object_and_holds(self):
        crate = make_object("crate", 0)
        crate.update(id="console-box", name="Console", position=[0., .4, 1.5])
        stages, route = plan_navigation(scene(crate), ("walker",),
                                        actor_id="walker", target_id="console-box",
                                        verb="approach",
                                        initial_placements={"walker": {"position_xz": [0., -2.]}})
        self.assertEqual(route["verb"], "approach")
        self.assertEqual(len(stages[-1].metadata["root_targets"]), 1)
        terminal = stages[-1].metadata["root_targets"]["walker"]
        self.assertEqual(len({tuple(item["position_xz"]) for item in terminal}), 1)
        self.assertLess(terminal[-1]["position_xz"][1], 1.5)

    def test_authored_floor_boundary_arch_rejects_and_inset_arch_passes(self):
        floor = make_object("platform", 0)
        floor.update(position=[0, -.1, 0], size=[7, .2, 6])
        gate = arch(z=-2.6)
        starts = {"walker": {"position_xz": [0, 0]}}
        with self.assertRaisesRegex(ValueError, "authored floor"):
            plan_navigation(scene(floor, gate), ("walker",), actor_id="walker", target_id="gate",
                            verb="go_through", initial_placements=starts)
        gate["position"][2] = -1.5
        _, route = plan_navigation(scene(floor, gate), ("walker",), actor_id="walker", target_id="gate",
                                  verb="go_through", initial_placements=starts)
        self.assertTrue(route["ground_support"]["continuous_support_verified"])

    def test_floor_union_checks_continuous_segments_not_only_sample_points(self):
        from realtime_navigation import validate_ground_path
        left = make_object("platform", 0)
        left.update(position=[-3, -.1, 0], size=[6, .2, 6])
        right = make_object("platform", 1)
        right.update(position=[3, -.1, 0], size=[6, .2, 6])
        result = validate_ground_path(scene(left, right), [(-2, 0), (2, 0)])
        self.assertEqual(len(result["floor_ids"]), 2)
        right["position"][0] = 3.01
        with self.assertRaisesRegex(ValueError, "gap"):
            validate_ground_path(scene(left, right), [(-2, 0), (2, 0)])
        with self.assertRaisesRegex(ValueError, "authored floor"):
            validate_ground_path(scene(left), [(-.1, 0)])

    def test_rotated_floor_coverage_and_height_steps(self):
        from realtime_navigation import validate_ground_path
        floor = make_object("platform", 0)
        floor.update(kind="custom", asset="floor", position=[5, -.1, 0], size=[4, .2, 12], yaw=90)
        doc = scene(floor)
        doc.update(version=3, assets=[{"id": "floor"}])
        self.assertTrue(validate_ground_path(doc, [[0, 0], [8, 0]])["continuous_support_verified"])
        with self.assertRaisesRegex(ValueError, "authored floor"):
            validate_ground_path(doc, [[5, 0], [5, 2]])
        floor.update(kind="platform", position=[-3, -.1, 0], size=[6, .2, 6], yaw=0)
        high = make_object("platform", 1)
        high.update(position=[3, -.02, 0], size=[6, .2, 6])
        with self.assertRaisesRegex(ValueError, "different floor elevations"):
            validate_ground_path(scene(floor, high), [(-2, 0), (2, 0)])

    def test_actual_city_support_remains_available_and_invalid_paths_reject(self):
        import json
        from pathlib import Path
        from realtime_navigation import validate_ground_path
        from studio_interaction_scene import recommend_placements
        doc = json.loads((Path(__file__).parent / "review/scene-integration/live-city.json").read_text())
        point = recommend_placements(doc, 1)[0]["position_xz"]
        _, route = plan_navigation(doc, ("walker",), actor_id="walker", target_id="city-2",
                                  verb="approach", initial_placements={"walker": {"position_xz": point}})
        self.assertTrue(route["ground_support"]["continuous_support_verified"])
        self.assertFalse(validate_ground_path(scene(), [[0, 0], [100, 100]])["authored_floor"])
        for points in ([], [[float("nan"), 0]], [["bad", 0]], [[0, 0, 0]]):
            with self.assertRaises(ValueError):
                validate_ground_path(doc, points)

    def test_invalid_placements_and_unsupported_verb_fail_before_generation(self):
        with self.assertRaisesRegex(ValueError, "all stable actor IDs"):
            plan_navigation(scene(arch()), ("walker", "witness"), actor_id="walker",
                            target_id="gate", verb="go_through",
                            initial_placements={"walker": placement()["walker"]})
        with self.assertRaisesRegex(ValueError, "approach or go_through"):
            plan_navigation(scene(arch()), ("walker", "witness"), actor_id="walker",
                            target_id="gate", verb="jump", initial_placements=placement())
        bad = placement()
        bad["walker"]["position_xz"] = [math.nan, 0]
        with self.assertRaisesRegex(ValueError, "finite"):
            plan_navigation(scene(arch()), ("walker", "witness"), actor_id="walker",
                            target_id="gate", verb="go_through", initial_placements=bad)


if __name__ == "__main__":
    unittest.main()
