"""CPU integration tests: real session/director contract with mock native clips."""
import math
import threading
import time
import unittest

import numpy as np

from core_spatial_commands import parse_commands, plan_command, measure_completion
from realtime_backend import validate_job
from realtime_clip import CanonicalClip
from realtime_navigation import plan_navigation
from scene_objects import make_object
from studio_core_session import CoreStudioSession, EMPTY_SCENE
from studio_interaction_scene import adapt_studio_scene
from test_studio_core_session import wait_for


def scene(*objects):
    return {**EMPTY_SCENE, "objects": list(objects)}


def native(roots, yaw=0.):
    roots = np.asarray(roots, np.float32)
    p = np.zeros((1, len(roots), 27, 3), np.float32)
    p[..., 1] = 1.
    p[0, :, :, 0] = roots[:, 0, None]
    p[0, :, :, 2] = roots[:, 1, None]
    matrix = np.asarray([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]], np.float32)
    r = np.broadcast_to(matrix, (1, len(roots), 27, 3, 3)).copy()
    return CanonicalClip(p, r, 20, ("actor_1",), "ardy_core", {}, np.zeros((1, len(roots), 330), np.float32))


class TargetClient:
    def __init__(self, *, drift=.1, yaw_drift=.2, stationary=False):
        self.calls = []
        self.root = None
        self.drift = drift
        self.yaw_drift = yaw_drift
        self.stationary = stationary
        self.errors = 0
        self.release = threading.Event()
        self.release.set()
        self.started = threading.Event()

    def wait(self, body, *, cancelled):
        validate_job(body)
        self.calls.append(body)
        self.started.set()
        while not self.release.wait(.005):
            if cancelled():
                raise RuntimeError("cancelled")
        if self.errors:
            self.errors -= 1
            raise RuntimeError("test transport failure")
        start = self.root if self.root is not None else body["initial_placements"]["actor_1"]["position_xz"]
        goals = body.get("root_targets", {}).get("actor_1", [])
        if not goals or self.stationary:
            roots = np.tile(start, (40, 1))
            yaw = 0.
        else:
            frames = [-1] + [item["frame"] for item in goals]
            points = [start] + [[item["position_xz"][0] + self.drift, item["position_xz"][1]] for item in goals]
            roots = np.column_stack([np.interp(np.arange(40), frames, np.asarray(points)[:, i]) for i in (0, 1)])
            yaw = goals[-1]["heading"] + self.yaw_drift
        self.root = roots[-1].copy()
        return [native(roots, yaw)]


def finish(session):
    def advance():
        state = session.snapshot()
        if state["total_frames"]:
            session.seek(state["total_frames"]-1)
        return state["spatial_commands"]["status"] != "running"
    wait_for(advance, timeout=5.)
    return session.snapshot()["spatial_commands"]


class SpatialParserTests(unittest.TestCase):
    def test_exact_language_names_bounds_and_unsupported_clauses(self):
        box = make_object("crate", 0)
        box.update(id="box-id", name="Console")
        adapted = adapt_studio_scene(scene(box))
        result = parse_commands("walk two metres forward then walk one metre left then approach box-id", adapted)
        self.assertEqual([a["verb"] for a in result], ["move", "move", "approach"])
        self.assertEqual(parse_commands("go to the Console", adapted)[0]["target_id"], "box-id")
        for text in ("walk two metres forward then climb stairs", "open the door", "walk 2m forward and dance", "walk 99m left", "walk 0m right", ";".join(["walk 1m left"]*5), "walk 1m left then approach missing"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_commands(text, adapted)
        named = {**box, "name": "Open Gate"}
        self.assertEqual(parse_commands("approach Open Gate", adapt_studio_scene(scene(named)))[0]["target_id"], "box-id")
        other = {**box, "id": "second"}
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            parse_commands("approach Console", adapt_studio_scene(scene(box, other)))

    def test_relative_direction_rotates_with_actual_heading_and_turn_walk_prompt(self):
        adapted = adapt_studio_scene(scene())
        for yaw, expected in ((0., [-2., 0.]), (math.pi/2, [0., 2.])):
            _, route = plan_command({"verb": "move", "direction": "left", "distance_m": 2}, adapted, ("actor_1",), "actor_1", native([[0, 0]], yaw), None)
            np.testing.assert_allclose(route["waypoints"][-1]["position_xz"], expected, atol=1e-5)
        stages, _ = plan_command({"verb": "move", "direction": "back", "distance_m": 2}, adapted, ("actor_1",), "actor_1", native([[0, 0]]), None)
        self.assertIn("turns", stages[0].prompt)
        self.assertEqual(stages[1].prompt, "A person walks forward naturally.")
        self.assertEqual(stages[-1].prompt, "A person stands upright and relaxed.")
        self.assertAlmostEqual(abs(stages[0].metadata["root_targets"]["actor_1"][-1]["heading"]), math.pi, places=5)

    def test_point_route_avoids_solids_and_rejects_floor_exit_without_synthetic_target(self):
        box = make_object("crate", 0)
        box.update(position=[0, .4, 1], size=[.5, .8, .5])
        doc = scene(box)
        _, route = plan_navigation(doc, ("actor_1",), actor_id="actor_1", verb="move", target_xz=[0, 2], initial_placements={"actor_1": {"position_xz": [0, 0]}})
        self.assertTrue(any(abs(w["position_xz"][0]) > .5 for w in route["waypoints"]))
        self.assertEqual(len(doc["objects"]), 1)
        floor = make_object("platform", 1)
        floor.update(position=[0, -.1, 0], size=[6, .2, 6])
        with self.assertRaisesRegex(ValueError, "authored floor"):
            plan_navigation(scene(floor), ("actor_1",), actor_id="actor_1", verb="move", target_xz=[0, 4], initial_placements={"actor_1": {"position_xz": [0, 0]}})

    def test_crossing_requires_measured_passage_interior_not_just_exit_arrival(self):
        arch = make_object("arch", 0)
        arch.update(id="arch", position=[0, 1.5, 2], size=[3, 3, .4])
        _, route = plan_navigation(scene(arch), ("actor_1",), actor_id="actor_1", target_id="arch", verb="go_through", initial_placements={"actor_1": {"position_xz": [0, 0]}})
        end = route["waypoints"][-1]["position_xz"]
        good = measure_completion(route, native([[0, 0], [0, 2], end]), 0)
        self.assertTrue(good["completed"])
        teleported = measure_completion(route, native([end]), 0)
        self.assertFalse(teleported["crossing_verified"])
        bad = measure_completion(route, native([[0, 0], [3, 1], [3, 3], end]), 0)
        self.assertTrue(bad["arrival_verified"])
        self.assertFalse(bad["crossing_verified"])


class SpatialSessionTests(unittest.TestCase):
    def setUp(self):
        self.client = TargetClient()
        self.session = CoreStudioSession(self.client)
        self.addCleanup(self.session.close)
        self.addCleanup(self.client.release.set)
        self.session.start(1, scene(), {"actor_1": {"position_xz": [0, 0], "yaw": 0}})

    def test_next_leg_uses_measured_terminal_root_heading_and_archive_provenance(self):
        self.session.spatial_commands("actor_1", "walk one metre forward then walk one metre left")
        report = finish(self.session)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["completed_actions"], 2)
        first, second = report["legs"]
        actual = first["measurement"]["terminal_xz"]
        np.testing.assert_allclose(second["route"]["waypoints"][0]["position_xz"], actual, atol=1e-5)
        expected = [actual[0]-math.cos(.2), actual[1]+math.sin(.2)]
        np.testing.assert_allclose(second["route"]["waypoints"][-1]["position_xz"], expected, atol=1e-5)
        original = self.session.timeline_clip()
        with CoreStudioSession() as restored:
            restored.load(self.session.save())
            self.assertEqual(restored.snapshot()["spatial_commands"], report)
            np.testing.assert_array_equal(original.positions, restored.timeline_clip().positions)
            np.testing.assert_array_equal(original.native_features, restored.timeline_clip().native_features)

    def test_false_arrival_stops_next_leg_and_preserves_generated_motion(self):
        self.client.stationary = True
        self.session.spatial_commands("actor_1", "walk 1m forward then walk 1m left")
        report = finish(self.session)
        self.assertEqual(report["status"], "arrival_failed")
        self.assertEqual(report["completed_actions"], 0)
        self.assertEqual(len(report["legs"]), 1)
        self.assertGreater(self.session.snapshot()["total_frames"], 0)
        self.assertEqual(self.session.snapshot()["queued_stages"], 0)

    def test_short_relative_move_cannot_succeed_without_moving(self):
        self.client.stationary = True
        self.session.spatial_commands("actor_1", "walk 0.25m forward then walk 1m left")
        report = finish(self.session)
        self.assertEqual(report["status"], "arrival_failed")
        self.assertEqual(report["completed_actions"], 0)
        self.assertEqual(len(report["legs"]), 1)
        self.assertEqual(report["legs"][0]["measurement"]["arrival_tolerance_m"], .125)

    def test_malformed_archived_details_do_not_break_playback(self):
        self.session.spatial_commands("actor_1", "walk 1m forward")
        finish(self.session)
        metadata = self.session._director.project_metadata["studio_core"]
        original = self.session.timeline_clip().positions.copy()
        for detail in (None, 7, {"invalid": True}):
            metadata["last_spatial_commands"]["detail"] = detail
            with CoreStudioSession() as restored:
                restored.load(self.session.save())
                self.assertIsNone(restored.snapshot()["spatial_commands"])
                np.testing.assert_array_equal(original, restored.timeline_clip().positions)
                restored.tick()

    def test_invalid_later_clause_preserves_pending_work_and_ordinary_redirect_cancels(self):
        self.client.release.clear()
        self.session.spatial_commands("actor_1", "walk 1m forward then walk 1m left")
        self.assertTrue(self.client.started.wait(1.))
        before = self.session.snapshot()
        with self.assertRaises(ValueError):
            self.session.spatial_commands("actor_1", "walk 1m forward then approach missing")
        self.assertEqual(self.session.snapshot()["epoch"], before["epoch"])
        self.assertEqual(self.session.snapshot()["queued_stages"], before["queued_stages"])
        with self.assertRaisesRegex(ValueError, "Wait"):
            self.session.direct({"actor_1": "Wave"}, queued=True)
        self.session.direct({"actor_1": "Wave"})
        self.assertEqual(self.session.snapshot()["spatial_commands"]["status"], "cancelled")
        self.client.release.set()
        wait_for(lambda: self.session.snapshot()["total_frames"] == 40)
        self.assertEqual(self.client.calls[-1]["actor_prompts"], {"actor_1": "Wave"})

    def test_failure_retry_and_scene_change_keep_committed_prefix(self):
        self.client.errors = 1
        self.session.spatial_commands("actor_1", "walk 1m forward then walk 1m left")
        wait_for(lambda: self.session.snapshot()["spatial_commands"]["status"] == "generation_failed")
        self.assertTrue(self.session.retry())
        report = finish(self.session)
        self.assertEqual(report["status"], "completed")
        prefix = self.session.timeline_clip().positions.copy()
        self.client.release.clear()
        self.client.started.clear()
        self.session.spatial_commands("actor_1", "walk 1m forward")
        self.session.seek(self.session.snapshot()["total_frames"]-1)
        self.assertTrue(self.client.started.wait(1.))
        self.session.update_scene({**scene(), "name": "Edited"})
        self.client.release.set()
        self.assertEqual(self.session.snapshot()["spatial_commands"]["status"], "cancelled")
        np.testing.assert_array_equal(prefix, self.session.timeline_clip().positions)

    def test_arch_in_real_sequence_is_measured_before_advancing(self):
        self.client.drift = self.client.yaw_drift = 0.
        arch = make_object("arch", 0)
        arch.update(id="arch", name="Arch", position=[0, 1.5, 2], size=[3, 3, .4])
        self.session.update_scene(scene(arch))
        self.session.spatial_commands("actor_1", "go through arch then walk 1m forward")
        report = finish(self.session)
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["legs"][0]["measurement"]["crossing_verified"])

    def test_closed_door_is_deferred_until_actual_approach_opens_it(self):
        self.client.drift = self.client.yaw_drift = 0.
        door = make_object("door", 0)
        door.update(id="door", name="Door", position=[0, 1.2, 2], size=[2, 2.4, .2])
        door["interaction"] = {"trigger": "proximity", "action": "open", "radius": .8}
        self.session.update_scene(scene(door))
        with self.assertRaisesRegex(ValueError, "closed"):
            self.session.spatial_commands("actor_1", "go through door")
        self.assertFalse(self.session.scene_reactions_enabled)
        self.session.spatial_commands("actor_1", "open door then go through door")
        report = finish(self.session)
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["legs"][0]["measurement"]["automatic_door_open_verified"])
        self.assertTrue(report["legs"][1]["measurement"]["crossing_verified"])
        with CoreStudioSession() as restored:
            restored.load(self.session.save())
            self.assertTrue(restored.scene_reactions_enabled)
            self.assertEqual(restored.snapshot()["scene_reactions_start_frame"], 0)

    def test_later_geometry_failure_preserves_completed_prefix(self):
        self.client.drift = self.client.yaw_drift = 0.
        door = make_object("door", 0)
        door.update(id="door", name="Door", position=[0, 1.2, 3], size=[2, 2.4, .2])
        door["interaction"] = {"trigger": "touch", "action": "open", "radius": .1}
        self.session.update_scene(scene(door))
        self.session.spatial_commands("actor_1", "walk 1m forward then go through door")
        report = finish(self.session)
        self.assertEqual(report["status"], "planning_failed")
        self.assertEqual(report["completed_actions"], 1)
        self.assertEqual(len(report["legs"]), 1)
        self.assertIn("closed", report["detail"])
        self.assertEqual(self.session.snapshot()["queued_stages"], 0)

    def test_unfinished_archive_is_provenance_only_and_does_not_resume(self):
        self.session.spatial_commands("actor_1", "walk 4m forward then walk 1m left")
        wait_for(lambda: self.session.snapshot()["total_frames"] == 80)
        with CoreStudioSession(self.client) as restored:
            restored.load(self.session.save())
            self.assertEqual(restored.snapshot()["spatial_commands"]["status"], "cancelled")
            self.assertEqual(restored.snapshot()["queued_stages"], 0)
            self.assertFalse(restored.retry())
            self.assertIsNone(restored._thread)


if __name__ == "__main__":
    unittest.main()
