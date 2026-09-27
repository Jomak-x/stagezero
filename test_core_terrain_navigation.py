"""CPU contracts for explicit terrain commands; no claim about generated gait."""
import copy
import math
import unittest

import numpy as np

from core_spatial_commands import parse_commands, plan_command
from realtime_backend import validate_job
from realtime_client import request_body
from realtime_clip import CanonicalClip
from realtime_director import RealtimeDirector
from scene_composition import make_preset
from studio_interaction_scene import adapt_studio_scene


def temple(angle=0., offset=(0., 0., 0.)):
    scene = make_preset("Jungle temple")
    theta = math.radians(angle)
    rotation = np.array(((math.cos(theta), 0, math.sin(theta)), (0, 1, 0),
                         (-math.sin(theta), 0, math.cos(theta))))
    origin = np.asarray(offset)
    for obj in scene["objects"]:
        obj["position"] = (np.asarray(obj["position"]) @ rotation.T + origin).tolist()
        obj["yaw"] = obj.get("yaw", 0.) + angle
    return scene, lambda p: np.asarray(p) @ rotation.T + origin


def adapted(scene, evaluated=None):
    result = adapt_studio_scene(scene if evaluated is None else evaluated)
    result["original_scene"] = scene
    return result


def native_root(xyz, yaw=math.pi):
    positions = np.zeros((1, 1, 27, 3), np.float32)
    positions[0, 0, :, :] = xyz
    c, s = math.cos(yaw), math.sin(yaw)
    matrix = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), np.float32)
    rotations = np.broadcast_to(matrix, (1, 1, 27, 3, 3)).copy()
    return CanonicalClip(positions, rotations, 20, ("actor_1",), "ardy_core", {},
                         np.zeros((1, 1, 330), np.float32))


class TerrainCommandTests(unittest.TestCase):
    def test_comma_in_exact_object_name_is_not_an_action_separator(self):
        scene, _ = temple()
        crate = next(obj for obj in scene["objects"] if obj["name"] == "Mossy rubble")
        crate["name"] = "Large, blue crate"
        actions = parse_commands("approach Large, blue crate, walk 1m forward", adapted(scene))
        self.assertEqual([action["verb"] for action in actions], ["approach", "move"])
        self.assertEqual(actions[0]["target_id"], crate["id"])
        crate["name"] = "Large, open crate"
        actions = parse_commands("approach Large, open crate, walk 1m forward", adapted(scene))
        self.assertEqual([action["verb"] for action in actions], ["approach", "move"])
        self.assertEqual(actions[0]["target_id"], crate["id"])

    def test_four_action_temple_cpu_plan_from_committed_endpoints(self):
        scene, _ = temple()
        actions = parse_commands(
            "walk up the ceremonial stairs, cross the bridge, open temple gate, and enter",
            adapted(scene))
        history = None
        for index, action in enumerate(actions):
            evaluated = copy.deepcopy(scene)
            if index == 3:
                door = next(obj for obj in evaluated["objects"] if obj["kind"] == "door")
                door["position"][1] += door["size"][1]
            stages, route = plan_command(action, adapted(scene, evaluated),
                                          ("actor_1",), "actor_1", history,
                                          {"actor_1": {"position_xz": [0., 0.], "yaw": math.pi}})
            self.assertTrue(stages)
            self.assertEqual(route["terrain_navigation_version"], 1)
            end = route["waypoints"][-1]
            history = native_root([end["position_xz"][0], end["support_y"]+.95,
                                   end["position_xz"][1]])
        self.assertEqual(route["verb"], "go_through")
        self.assertEqual({row["role"] for row in route["waypoints"]} &
                         {"entry", "center", "exit"}, {"entry", "center", "exit"})

    def test_natural_four_actions_and_ambiguous_alias(self):
        scene, _ = temple()
        actions = parse_commands("walk up the ceremonial stairs, cross the bridge, open temple gate, and enter",
                                 adapted(scene))
        self.assertEqual([item["verb"] for item in actions],
                         ["ascend", "cross", "open", "go_through"])
        self.assertEqual(actions[-1]["target_id"], actions[-2]["target_id"])
        self.assertEqual(parse_commands("open temple gate", adapted(scene))[0]["verb"], "open")
        with self.assertRaisesRegex(ValueError, "qualifier"):
            parse_commands("open nonexistent moon gate", adapted(scene))
        with self.assertRaisesRegex(ValueError, "qualifier"):
            parse_commands("cross moon bridge", adapted(scene))
        with self.assertRaisesRegex(ValueError, "preceding open"):
            parse_commands("enter", adapted(scene))
        with self.assertRaisesRegex(ValueError, "at most 4"):
            parse_commands("walk up stairs, cross bridge, open gate, enter, walk 1m forward",
                           adapted(scene))
        duplicate = copy.deepcopy(scene)
        bridge = next(obj for obj in duplicate["objects"] if obj["name"] == "Suspended ravine bridge")
        bridge["position"][0] = -2.
        other = copy.deepcopy(bridge)
        other["id"] += "-other"
        other["position"][0] = 2.
        duplicate["objects"].append(other)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            parse_commands("cross bridge", adapted(duplicate))
        second_gate = copy.deepcopy(scene)
        door = next(obj for obj in second_gate["objects"] if obj["kind"] == "door")
        other_door = copy.deepcopy(door)
        other_door["id"] += "-other"
        other_door["position"][0] += 5.
        second_gate["objects"].append(other_door)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            parse_commands("open gate", adapted(second_gate))

    def test_explicit_session_terrain_continuation_only(self):
        scene, _ = temple()
        first = parse_commands("walk 1m forward", adapted(scene))[0]
        self.assertNotIn("terrain", first)
        continuing = adapted(scene)
        continuing["terrain_active"] = True
        action = parse_commands("walk 1m forward", continuing)[0]
        self.assertTrue(action["terrain"])
        _, route = plan_command(action, continuing, ("actor_1",), "actor_1", None,
                                {"actor_1": {"position_xz": [0., 0.], "yaw": 0.}})
        self.assertEqual(route["terrain_navigation_version"], 1)

    def test_stairs_native_schema_and_world_translation(self):
        for angle, offset in ((0., (0., 0., 0.)), (37., (4., 6., -3.))):
            with self.subTest(angle=angle):
                scene, transform = temple(angle, offset)
                action = parse_commands("walk up the ceremonial stairs", adapted(scene))[0]
                start = transform([0., 0., 0.])
                stages, route = plan_command(action, adapted(scene), ("actor_1",), "actor_1", None,
                    {"actor_1": {"position_xz": [float(start[0]), float(start[2])],
                                 "yaw": math.atan2(-math.sin(math.radians(angle)), -math.cos(math.radians(angle)))}})
                self.assertEqual(route["terrain_navigation_version"], 1)
                self.assertGreater(route["waypoints"][-1]["support_y"]-route["waypoints"][0]["support_y"], 1.5)
                self.assertTrue(all(stage.metadata["terrain_navigation_version"] == 1 for stage in stages))
                self.assertTrue(all(len(stage.metadata["root_targets"]["actor_1"]) <= 24 for stage in stages))
                self.assertTrue(all("root_height" in target for stage in stages
                                    for target in stage.metadata["root_targets"]["actor_1"]))
                self.assertTrue(all("heading" not in target for stage in stages
                                    if stage.metadata["terrain_navigation"]["stage"] > 0
                                    for target in stage.metadata["root_targets"]["actor_1"]))
                self.assertAlmostEqual(stages[0].metadata["coordinate_frames_y"]["actor_1"], offset[1], delta=.04)
                director = RealtimeDirector(("actor_1",))
                director.queue_sequence(stages)
                body = request_body(director.claim_request())
                validate_job(body)
                self.assertIn("coordinate_frames_y", body)
                self.assertIn("root_height", body["root_targets"]["actor_1"][0])

    def test_bridge_route_and_missing_bridge_fail_clearly(self):
        scene, _ = temple()
        bridge = parse_commands("cross bridge", adapted(scene))[0]
        stages, route = plan_command(bridge, adapted(scene), ("actor_1",), "actor_1", None,
                                     {"actor_1": {"position_xz": [0., 0.], "yaw": math.pi}})
        self.assertEqual(route["target_id"], next(obj["id"] for obj in scene["objects"]
                                                   if obj["name"] == "Suspended ravine bridge"))
        self.assertGreater(route["waypoints"][-1]["support_y"], 3.)
        self.assertGreater(len(stages), 1)
        missing = copy.deepcopy(scene)
        missing["objects"] = [obj for obj in missing["objects"]
                              if obj["name"] != "Suspended ravine bridge"]
        missing["targets"] = [target for target in missing["targets"]
                              if target["object_id"] != bridge["target_id"]]
        with self.assertRaisesRegex(ValueError, "Unknown terrain target"):
            parse_commands("cross bridge", adapted(missing))
        gate = parse_commands("open gate", adapted(missing))[0]
        with self.assertRaises(ValueError):
            plan_command(gate, adapted(missing), ("actor_1",), "actor_1", None,
                         {"actor_1": {"position_xz": [0., 0.], "yaw": math.pi}})

    def test_elevated_gate_waits_and_enter_requires_observed_lift(self):
        scene, _ = temple()
        actor = native_root([0., 5.09348, -18.62])
        open_action = parse_commands("open gate", adapted(scene))[0]
        stages, route = plan_command(open_action, adapted(scene), ("actor_1",), "actor_1", actor, None)
        self.assertEqual(route["schedule"]["door_hold_frames"], 40)
        self.assertEqual(stages[-1].prompt, "A person stands upright and relaxed.")
        with self.assertRaisesRegex(ValueError, "not fully open"):
            plan_command({"verb": "go_through", "target_id": open_action["target_id"]},
                         adapted(scene), ("actor_1",), "actor_1", actor, None)
        opened = copy.deepcopy(scene)
        door = next(obj for obj in opened["objects"] if obj["id"] == open_action["target_id"])
        door["position"][1] += door["size"][1]
        entry_actor = native_root([0., 6.36457, -21.79])
        repeated_stages, repeated = plan_command(open_action, adapted(scene, opened),
            ("actor_1",), "actor_1", entry_actor, None)
        self.assertTrue(repeated_stages)
        self.assertEqual(repeated["schedule"]["door_hold_frames"], 40)
        enter = parse_commands("enter gate", adapted(scene, opened))[0]
        _, passage = plan_command(enter, adapted(scene, opened), ("actor_1",),
                                  "actor_1", entry_actor, None)
        self.assertEqual({row["role"] for row in passage["waypoints"]} &
                         {"entry", "center", "exit"}, {"entry", "center", "exit"})
        self.assertAlmostEqual(passage["geometry"]["floor_y_m"], 5.6)


if __name__ == "__main__":
    unittest.main()
