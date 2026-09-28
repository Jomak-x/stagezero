"""Committed mid-flight routes can finish the selected rendered staircase."""
from __future__ import annotations

import math
import unittest

import numpy as np

from core_spatial_commands import parse_commands, plan_command
from realtime_clip import CanonicalClip
from scene_composition import validate_scene
from scene_interaction_geometry import SceneInteractionGeometry
from studio_interaction_scene import adapt_studio_scene
from switchback_traversal import industrial_switchback_scene


ACTOR = "actor_1"


def _adapted(scene):
    result = adapt_studio_scene(scene)
    result.update(original_scene=scene, terrain_active=True)
    return result


def _committed(route):
    end = route["waypoints"][-1]
    positions = np.zeros((1, 1, 27, 3), np.float32)
    positions[:] = (end["position_xz"][0], end["support_y"]+.95,
                    end["position_xz"][1])
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32),
                                (1, 1, 27, 3, 3)).copy()
    return CanonicalClip(positions, rotations, 20, (ACTOR,), "ardy_core", {},
                         np.zeros((1, 1, 330), np.float32))


def _rotated_scene():
    angle = math.radians(37.)
    rotation = np.array(((math.cos(angle), 0., math.sin(angle)), (0., 1., 0.),
                         (-math.sin(angle), 0., math.cos(angle))))
    offset = np.array((3., 4., -2.))
    scene = industrial_switchback_scene()
    for obj in scene["objects"]:
        obj["position"] = (np.asarray(obj["position"]) @ rotation.T + offset).tolist()
        if obj["kind"] in ("custom", "door"):
            obj["yaw"] = obj.get("yaw", 0.)+37.
    return validate_scene(scene), lambda point: np.asarray(point) @ rotation.T + offset


class PartialStairTests(unittest.TestCase):
    def test_only_selected_tread_support_qualifies_for_partial_start(self):
        scene = validate_scene(industrial_switchback_scene())
        geometry = SceneInteractionGeometry.from_scene(scene)
        # The upper deck has the same height as the last approach tread, but
        # it is not part of the selected flight.
        self.assertAlmostEqual(geometry.support_height(0., -3., .2), .2)
        self.assertAlmostEqual(geometry.stair_routes[0]["steps"][-1][1], .2)
        adapted = _adapted(scene)
        with self.assertRaisesRegex(ValueError, "destination side"):
            plan_command(parse_commands("walk up foundry approach stairs", adapted)[0],
                         adapted, (ACTOR,), ACTOR, None,
                         {ACTOR: {"position_xz": [0., -3.], "yaw": math.pi}})

    def test_relative_walk_then_finish_both_flights(self):
        scene = validate_scene(industrial_switchback_scene())
        adapted = _adapted(scene)
        cases = (
            ((0., 2.), "walk 3.75m forward", (0., -1.75), .16,
             "walk up foundry approach stairs", (0., -2.25), .20),
            ((10.5, -7.7), "walk 2.925m forward", (10.5, -10.625), .04,
             "walk down loading exit steps", (10.5, -11.375), 0.),
        )
        for start, move, mid, mid_y, stairs, end, end_y in cases:
            with self.subTest(stairs=stairs):
                placement = {ACTOR: {"position_xz": list(start), "yaw": math.pi}}
                _, first = plan_command(parse_commands(move, adapted)[0], adapted,
                                        (ACTOR,), ACTOR, None, placement)
                np.testing.assert_allclose(first["waypoints"][-1]["position_xz"], mid,
                                           atol=1e-5)
                self.assertAlmostEqual(first["waypoints"][-1]["support_y"], mid_y)
                committed = _committed(first)
                stages, second = plan_command(parse_commands(stairs, adapted)[0], adapted,
                                              (ACTOR,), ACTOR, committed, None)
                self.assertTrue(stages)
                np.testing.assert_allclose(second["support_xyz"][0],
                                           [mid[0], mid_y, mid[1]], atol=2e-5)
                np.testing.assert_allclose(second["waypoints"][-1]["position_xz"], end,
                                           atol=1e-5)
                self.assertAlmostEqual(second["waypoints"][-1]["support_y"], end_y)

    def test_partial_routes_follow_rotated_translated_treads(self):
        scene, transform = _rotated_scene()
        adapted = _adapted(scene)
        yaw = math.atan2(-math.sin(math.radians(37)),
                         -math.cos(math.radians(37)))
        cases = (
            ((0., 0., 2.), "walk 3.75m forward", (0., .16, -1.75),
             "walk up foundry approach stairs", (0., .20, -2.25)),
            ((10.5, .2, -7.7), "walk 2.925m forward", (10.5, .04, -10.625),
             "walk down loading exit steps", (10.5, 0., -11.375)),
        )
        for start, move, middle, stairs, end in cases:
            with self.subTest(stairs=stairs):
                world_start = transform(start)
                placement = {ACTOR: {"position_xz": [float(world_start[0]),
                                                      float(world_start[2])],
                                     "yaw": yaw}}
                _, first = plan_command(parse_commands(move, adapted)[0], adapted,
                                        (ACTOR,), ACTOR, None, placement)
                np.testing.assert_allclose(first["support_xyz"][-1], transform(middle),
                                           atol=2e-5)
                _, second = plan_command(parse_commands(stairs, adapted)[0], adapted,
                                         (ACTOR,), ACTOR, _committed(first), None)
                np.testing.assert_allclose(second["support_xyz"][-1], transform(end),
                                           atol=2e-5)

    def test_second_and_penultimate_treads_in_both_directions(self):
        scene = validate_scene(industrial_switchback_scene())
        adapted = _adapted(scene)
        cases = (
            ((0., -.75), "walk up foundry approach stairs", (0., -2.25)),
            ((0., -1.75), "walk down foundry approach stairs", (0., -.25)),
            ((10.5, -9.125), "walk down loading exit steps", (10.5, -11.375)),
            ((10.5, -10.625), "walk up loading exit steps", (10.5, -8.375)),
        )
        for start, command, goal in cases:
            with self.subTest(start=start, command=command):
                placement = {ACTOR: {"position_xz": list(start), "yaw": math.pi}}
                _, route = plan_command(parse_commands(command, adapted)[0], adapted,
                                        (ACTOR,), ACTOR, None, placement)
                np.testing.assert_allclose(route["waypoints"][-1]["position_xz"], goal,
                                           atol=1e-5)

    def test_terminal_tread_and_wrong_side_still_reject(self):
        scene = validate_scene(industrial_switchback_scene())
        adapted = _adapted(scene)
        cases = (
            ((0., -2.25), "walk up foundry approach stairs"),
            ((10.5, -11.375), "walk down loading exit steps"),
            ((0., -3.), "walk up foundry approach stairs"),
            ((0., .5), "walk down foundry approach stairs"),
            ((10.5, -7.7), "walk up loading exit steps"),
        )
        for start, command in cases:
            with self.subTest(start=start, command=command):
                placement = {ACTOR: {"position_xz": list(start), "yaw": math.pi}}
                with self.assertRaisesRegex(ValueError, "destination side"):
                    plan_command(parse_commands(command, adapted)[0], adapted,
                                 (ACTOR,), ACTOR, None, placement)


if __name__ == "__main__":
    unittest.main()
