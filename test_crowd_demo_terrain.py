"""Regression checks for the exact authored station-box proxy and route claim."""
import unittest

import numpy as np

from experiments.crowd_demo_terrain import evaluate, station_scene
from experiments.crowd_demo_kiosk_motion import preflight, station_kiosk_scene
from core_spatial_commands import parse_commands, plan_command
from scene_interaction_geometry import SceneInteractionGeometry
from studio_interaction_scene import adapt_studio_scene


class CrowdDemoTerrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = station_scene()
        cls.geometry = SceneInteractionGeometry.from_scene(cls.scene)
        cls.result = evaluate()

    def test_station_boxes_preserve_source_dimensions_and_support(self):
        stairs = next(obj for obj in self.scene["objects"] if obj["id"] == "east-stairs")
        terrace = next(obj for obj in self.scene["objects"] if obj["id"] == "east-terrace")
        self.assertEqual(stairs["position"], [41., .75, -1.65])
        self.assertEqual(stairs["size"], [15., 1.5, 3.])
        self.assertEqual(terrace["position"], [41., .75, -11.])
        self.assertEqual(terrace["size"], [15., 1.5, 15.])
        for step in range(6):
            y = .25 + .25 * step
            z = -.4 - .5 * step
            self.assertAlmostEqual(self.geometry.support_height(41., z, y, .01, .01), y)
        self.assertIsNone(self.geometry.support_height(41., -3.325, 1.5, .25, .35))
        self.assertFalse(self.geometry.stair_routes)
        self.assertNotIn("__studio_floor__", {s.object_id for s in self.geometry.walkable_surfaces})

    def test_opt_in_is_explicit_for_a_location_command(self):
        ordinary = adapt_studio_scene(self.scene)
        self.assertNotIn("terrain", parse_commands("walk 1m forward", ordinary)[0])
        ordinary["terrain_active"] = True
        self.assertTrue(parse_commands("walk 1m forward", ordinary)[0]["terrain"])

    def test_station_stair_command_rejects_before_native_generation(self):
        adapted = adapt_studio_scene(self.scene)
        adapted["terrain_active"] = True
        action = parse_commands("walk up east station stairs", adapted)[0]
        self.assertEqual(action["target_id"], "east-stairs")
        with self.assertRaisesRegex(ValueError, "no unambiguous rendered stair flight"):
            plan_command(action, adapted, ("actor_1",), "actor_1", None,
                         {"actor_1": {"position_xz": [41., 1.], "yaw": 3.14159}})

    def test_geometry_plan_avoids_straight_line_collision(self):
        baseline = self.result["baseline"]
        terrain = self.result["terrain"]
        self.assertGreater(baseline["collision_or_support_failure_count"], 0)
        self.assertEqual(terrain["waypoint_failures"], [])
        self.assertEqual(terrain["swept_failures"], [])
        self.assertGreater(terrain["path_length_m"], 18.)
        route = np.asarray(terrain["waypoints_xyz"])
        np.testing.assert_allclose(route[0], [32., 0., -8.])
        np.testing.assert_allclose(route[-1], [50., 0., -8.])
        self.assertTrue(np.any(route[:, 2] > -.5))  # Reroutes around terrace frontage.
        self.assertEqual(self.result["stairs"]["recognized_stair_flights"], 0)
        self.assertFalse(self.result["provenance"]["native_motion_generated"])


class KioskMotionPreflightTests(unittest.TestCase):
    def test_authored_kiosk_parts_and_bounded_core_route(self):
        scene = station_kiosk_scene()
        kiosk = next(obj for obj in scene["objects"] if obj["id"] == "koma-kiosk")
        asset = next(asset for asset in scene["assets"] if asset["id"] == kiosk["asset"])
        self.assertEqual(len(asset["parts"]), 4)
        self.assertAlmostEqual(kiosk["position"][0], -22. + np.sin(.25) * .2125)
        self.assertAlmostEqual(kiosk["yaw"], np.degrees(.25))
        _, evidence = preflight()
        self.assertGreater(evidence["straight_blocked_count"], 0)
        self.assertEqual(evidence["planned_horizons"], 6)
        route = np.asarray(evidence["route"]["support_xyz"])
        self.assertTrue(np.any(route[:, 0] > -19.))
        self.assertLessEqual(np.abs(route).max(), 25.)



if __name__ == "__main__":
    unittest.main()
