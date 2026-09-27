"""Current-main scene geometry and navigation integration, without inference."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from interaction_planner import plan_action, _obstacles, _inside
from interaction_scene import scene_objects, is_walkable_ground
from interaction_scene_collision import scene_collision
from realtime_navigation import plan_navigation
from scene_objects import make_object, evaluate_objects
from studio_interaction_scene import (adapt_studio_scene, resolve_target,
                                      recommend_placements, measure_scene_motion)

ROOT = Path(__file__).resolve().parent


def scene(*objects):
    return {"version": 2, "name": "Test", "objects": list(objects), "effects": [], "lighting": "neutral"}


def gate():
    obj = make_object("arch", 0)
    obj.update(position=[0, 1.5, 2], size=[3, 3, .4], name="Gate")
    return obj


def city():
    return json.loads((ROOT / "review/scene-integration/live-city.json").read_text())


class StudioSceneAdapterTests(unittest.TestCase):
    def test_actual_fifty_object_city_keeps_geometry_and_excludes_floor_blockers(self):
        doc = city()
        before = copy.deepcopy(doc)
        adapted = adapt_studio_scene(doc)
        self.assertEqual(doc, before)
        self.assertEqual(len(adapted["scene"]["objects"]), 50)
        floors = [o for o in scene_objects(adapted["scene"]) if is_walkable_ground(o)]
        self.assertEqual({o.id for o in floors}, {"city-0", "city-1"})
        self.assertFalse({o.id for o in floors} & {o["id"] for o in adapted["objects"]})
        placements = recommend_placements(adapted["scene"], 2)
        obstacles = _obstacles(scene_objects(adapted["scene"]), None, 1.65, .4)
        for placement in placements:
            self.assertFalse(any(_inside(placement["position_xz"], o, .4) for o in obstacles))
        self.assertGreaterEqual(np.linalg.norm(np.subtract(placements[0]["position_xz"], placements[1]["position_xz"])), 1.5)
        # The original authored floor at the origin used to reject this route.
        target = next(o for o in adapted["objects"] if o["id"] == "city-2")
        route = plan_action({"actor_id": "a", "verb": "approach", "target_id": target["id"]},
                            adapted["scene"], [*placements[0]["position_xz"][:1], 0, placements[0]["position_xz"][1]])
        self.assertGreater(len(route["waypoints"]), 1)

    def test_rotated_floor_footprint_and_elevated_slab_are_not_confused(self):
        doc = city()
        doc["objects"] = [doc["objects"][0]]
        obj = doc["objects"][0]
        obj.update(position=[8, -.025, 0], size=[4, .05, 12], yaw=90)
        adapted = adapt_studio_scene(doc)
        points = recommend_placements(adapted["scene"], 2)
        for item in points:
            x, z = item["position_xz"]
            self.assertLessEqual(abs(z), 1.6)
            self.assertLessEqual(abs(x - 8), 5.6)
        obj["position"][1] = .3
        self.assertFalse(is_walkable_ground(scene_objects(doc)[0]))
        poses = np.full((1, 27, 3), [8., .3, 0.])
        self.assertEqual(scene_collision(poses, "core27", doc)["total_collision_frames"], 1)

    def test_gate_route_and_blocked_opening(self):
        arch = gate()
        adapted = adapt_studio_scene(scene(arch))
        self.assertIn("go_through", adapted["objects"][0]["actions"])
        stages, _ = plan_navigation(adapted["scene"], ("a",), actor_id="a", target_id=arch["id"],
            verb="go_through", initial_placements={"a": {"position_xz": [0, -1], "yaw": 0}})
        self.assertTrue(stages)
        crate = make_object("crate", 0, [0, .4, 2])
        with self.assertRaisesRegex(ValueError, "blocked"):
            plan_action({"actor_id": "a", "verb": "go_through", "target_id": arch["id"]}, scene(arch, crate), [0, 0, -1])

    def test_closed_door_and_custom_named_holes_reject(self):
        door = make_object("door", 0)
        closed = adapt_studio_scene(scene(door))
        self.assertEqual(closed["objects"][0]["actions"], ["approach"])
        with self.assertRaisesRegex(ValueError, "closed"):
            plan_action({"actor_id": "a", "verb": "go_through", "target_id": door["id"]}, closed["scene"], [0, 0, -2])
        doc = city()
        doc["objects"] = [doc["objects"][2]]
        doc["objects"][0]["name"] = "Open gate"
        adapted = adapt_studio_scene(doc)
        self.assertEqual(adapted["objects"][0]["actions"], ["approach"])
        with self.assertRaisesRegex(ValueError, "verified"):
            plan_action({"actor_id": "a", "verb": "go_through", "target_id": doc["objects"][0]["id"]}, doc, [0, 0, -2])

    def test_real_evaluated_open_door_has_geometry_backed_passage(self):
        door = make_object("door", 0)
        poses = np.zeros((1, 34, 3)); poses[:, :, 2] = -.1
        states = evaluate_objects([door], poses, 0)
        self.assertTrue(states[0]["active"])
        adapted = adapt_studio_scene(scene(door), object_states=states)
        self.assertIn("go_through", adapted["objects"][0]["actions"])
        result = plan_action({"actor_id": "a", "verb": "go_through", "target_id": door["id"]}, adapted["scene"], [0, 0, -2])
        self.assertEqual(result["geometry"]["source"], "raised_procedural_door")
        fake = copy.deepcopy(states); fake[0]["position"] = door["position"]
        with self.assertRaisesRegex(ValueError, "rendered"):
            adapt_studio_scene(scene(door), object_states=fake)

    def test_exact_ids_and_ambiguous_names(self):
        a, b = gate(), gate(); b["id"] = "gate-b"
        adapted = adapt_studio_scene(scene(a, b))
        self.assertEqual(resolve_target(adapted, "gate-b"), "gate-b")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            resolve_target(adapted, "gate")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            resolve_target(adapted, "some gate")

    def test_sixty_four_object_scene_holds_second_actor_without_exceeding_budget(self):
        objects = [gate()]
        for i in range(63):
            objects.append(make_object("crate", i, [10 + i % 9, .4, -10 + i // 9]))
        adapted = adapt_studio_scene(scene(*objects))
        stages, _ = plan_navigation(adapted["scene"], ("a", "b"), actor_id="a", target_id="arch-0",
            verb="go_through", initial_placements={"a": {"position_xz": [0, -1]}, "b": {"position_xz": [-2, -1]}})
        self.assertTrue(stages)
        self.assertEqual(len(adapted["scene"]["objects"]), 64)

    def test_core_route_coordinate_bounds_and_g1_measurement(self):
        crate = make_object("crate", 0, [27, .4, 0])
        with self.assertRaisesRegex(ValueError, "25 m"):
            plan_navigation(scene(crate), ("a",), actor_id="a", target_id=crate["id"], verb="approach",
                            initial_placements={"a": {"position_xz": [24, 0]}})
        result = measure_scene_motion(np.zeros((2, 34, 3)), scene())
        self.assertEqual(result["collision"]["skeleton"], "g1")
        self.assertEqual(result["collision"]["total_collision_frames"], 0)


if __name__ == "__main__":
    unittest.main()
