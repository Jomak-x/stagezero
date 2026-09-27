"""CPU checks for changed industrial geometry and committed route continuation."""
from __future__ import annotations

import copy
import math
import unittest

import numpy as np

from core_spatial_commands import parse_commands, plan_command
from realtime_backend import validate_job
from realtime_clip import CanonicalClip
from realtime_client import request_body
from realtime_director import RealtimeDirector
from scene_composition import validate_scene
from studio_interaction_scene import adapt_studio_scene
from switchback_traversal import industrial_switchback_scene


ACTOR = "actor_1"


def _clip(root, yaw=-.6):
    positions = np.zeros((1, 1, 27, 3), np.float32)
    positions[:] = root
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), np.float32)
    rotations = np.broadcast_to(rotation, (1, 1, 27, 3, 3)).copy()
    return CanonicalClip(positions, rotations, 20, (ACTOR,), "ardy_core", {},
                         np.zeros((1, 1, 330), np.float32))


def _adapted(authored, evaluated=None, *, heading=None):
    result = adapt_studio_scene(authored if evaluated is None else evaluated)
    result["original_scene"] = authored
    result["terrain_active"] = True
    if heading is not None:
        result["terrain_planning_heading"] = heading
    return result


def _changed_switchback():
    """Rotate, raise, translate, and widen a real deck without renaming it."""
    scene = industrial_switchback_scene()
    gantry = next(obj for obj in scene["objects"] if obj["id"] == "yard-gantry")
    gantry["size"][2] = 2.8
    angle = math.radians(37)
    rotation = np.array(((math.cos(angle), 0, math.sin(angle)),
                         (0, 1, 0), (-math.sin(angle), 0, math.cos(angle))))
    offset = np.array((3., 4., -2.))
    for obj in scene["objects"]:
        obj["position"] = (np.asarray(obj["position"]) @ rotation.T + offset).tolist()
        if obj["kind"] in ("custom", "door"):
            obj["yaw"] = obj.get("yaw", 0.) + 37.
    return validate_scene(scene), lambda point: np.asarray(point) @ rotation.T + offset


class FollowupTerrainNavigationTests(unittest.TestCase):
    def test_changed_industrial_stairs_bridge_gate_and_subsequent_relative_routes(self):
        scene, transform = _changed_switchback()
        start = transform((0., 0., 2.))
        placement = {ACTOR: {"position_xz": [float(start[0]), float(start[2])],
                             "yaw": math.pi}}
        committed = None
        routes = []
        for command in ("walk up foundry approach stairs", "cross service gantry bridge",
                        "open workshop door", "enter workshop door",
                        "walk down loading exit steps"):
            evaluated = scene
            if command == "enter workshop door":
                evaluated = copy.deepcopy(scene)
                door = next(obj for obj in evaluated["objects"] if obj["kind"] == "door")
                door["position"][1] += door["size"][1]
            current = _adapted(scene, evaluated)
            action = parse_commands(command, current)[0]
            stages, route = plan_command(action, current, (ACTOR,), ACTOR, committed,
                                         placement)
            self.assertTrue(stages)
            self.assertEqual(route["terrain_navigation_version"], 1)
            if committed is not None:
                np.testing.assert_allclose(route["support_xyz"][0],
                                           committed.positions[0, -1, 0] - [0., .95, 0.],
                                           atol=2e-4)
            last = route["waypoints"][-1]
            committed = _clip((last["position_xz"][0], last["support_y"]+.95,
                               last["position_xz"][1]))
            routes.append(route)

        self.assertEqual([route["verb"] for route in routes],
                         ["ascend", "cross", "open", "go_through", "descend"])
        self.assertEqual([route["target_id"] for route in routes],
                         ["yard-approach-stairs", "yard-gantry", "yard-workshop-door",
                          "yard-workshop-door", "yard-exit-stairs"])
        self.assertAlmostEqual(routes[0]["waypoints"][-1]["support_y"], 4.2, places=5)
        self.assertAlmostEqual(routes[-1]["waypoints"][-1]["support_y"], 4., places=5)
        bridge_center = next(point for point in routes[1]["waypoints"]
                             if point["role"] == "center")
        np.testing.assert_allclose(bridge_center["position_xz"],
                                   transform((6., .2, -5.))[[0, 2]], atol=2e-4)
        np.testing.assert_allclose(np.unique(np.asarray(routes[-1]["support_xyz"])[:, 1]),
                                   [4., 4.04, 4.08, 4.12, 4.16, 4.2], atol=1e-5)
        end = routes[-1]["waypoints"][-1]
        np.testing.assert_allclose(end["position_xz"], transform((10.5, 0., -11.375))[[0, 2]],
                                   atol=2e-4)
        self.assertEqual({point["role"] for point in routes[3]["waypoints"]} &
                         {"entry", "center", "exit"}, {"entry", "center", "exit"})

        # Native yaw deliberately disagrees with the displayed route heading.
        # Each new command must start at the previous committed endpoint while
        # interpreting the direction from the accepted display heading.
        heading = math.atan2(-math.sin(math.radians(37)),
                             -math.cos(math.radians(37)))
        current = _adapted(scene, heading=heading)
        for direction, local_end in (("forward", (10.5, 0., -11.875)),
                                     ("right", (10., 0., -11.875))):
            action = parse_commands(f"walk 0.5m {direction}", current)[0]
            previous = committed.positions.copy()
            _, route = plan_command(action, current, (ACTOR,), ACTOR, committed, None)
            np.testing.assert_allclose(route["support_xyz"][0],
                                       previous[0, -1, 0] - [0., .95, 0.], atol=2e-4)
            np.testing.assert_allclose(route["waypoints"][-1]["position_xz"],
                                       transform(local_end)[[0, 2]], atol=2e-4)
            np.testing.assert_array_equal(committed.positions, previous)
            last = route["waypoints"][-1]
            committed = _clip((last["position_xz"][0], last["support_y"]+.95,
                               last["position_xz"][1]))

    def test_turn_targets_wrap_both_sides_of_pi(self):
        scene = validate_scene(industrial_switchback_scene())
        for heading, direction, offset in ((math.pi-.04, "right", math.pi/2),
                                           (-math.pi+.04, "left", -math.pi/2)):
            with self.subTest(heading=heading):
                current = _adapted(scene, heading=heading)
                action = parse_commands(f"walk 0.5m {direction}", current)[0]
                stages, route = plan_command(action, current, (ACTOR,), ACTOR,
                                             _clip((0., .95, 2.)), None)
                self.assertEqual(route["schedule"]["initial_turn_frames"], 40)
                targets = stages[0].metadata["root_targets"][ACTOR]
                angles = np.array([target["heading"] for target in targets])
                self.assertTrue(np.all(np.abs(angles) <= math.pi))
                expected = heading + offset
                self.assertAlmostEqual(math.atan2(math.sin(angles[-1]-expected),
                                                   math.cos(angles[-1]-expected)), 0., places=5)
                unwrapped = np.unwrap(angles)
                self.assertTrue(np.all(np.sign(np.diff(unwrapped)) == np.sign(offset)))
                self.assertLess(abs(unwrapped[-1]-unwrapped[0]), math.pi)
                np.testing.assert_allclose(route["waypoints"][-1]["position_xz"],
                                           [.5*math.sin(expected),
                                            2.+.5*math.cos(expected)], atol=2e-4)

    def test_exact_pi_turn_targets_stay_within_core_schema(self):
        scene = validate_scene(industrial_switchback_scene())
        current = _adapted(scene, heading=math.pi/2)
        action = parse_commands("walk 0.5m right", current)[0]
        stages, _ = plan_command(action, current, (ACTOR,), ACTOR,
                                 _clip((0., .95, 2.)), None)
        self.assertEqual(stages[0].metadata["root_targets"][ACTOR][-1]["heading"],
                         math.pi)
        director = RealtimeDirector((ACTOR,))
        director.queue_sequence(stages)
        validate_job(request_body(director.claim_request()))


if __name__ == "__main__":
    unittest.main()
