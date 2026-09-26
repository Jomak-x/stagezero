"""Checks that the live adapter reproduces the captured generated geometry."""

import hashlib
import unittest

import numpy as np

from swing_scene import SOURCE_PATH, SOURCE_SHA256, load_swing_scene


class SwingSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = load_swing_scene()

    def test_snapshot_and_original_mesh_topology(self):
        self.assertEqual(hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(), SOURCE_SHA256)
        self.assertEqual(len(self.scene["buildings"]), 7)
        self.assertEqual(sum(len(mesh["indices"]) // 3 for mesh in self.scene["meshes"]), 10548)
        for mesh in self.scene["meshes"]:
            self.assertEqual(len(mesh["indices"]) % 3, 0)
            self.assertLess(max(mesh["indices"]), len(mesh["positions"]))

    def test_collision_bounds_contain_rendered_buildings(self):
        for building in self.scene["buildings"]:
            points = np.concatenate([
                np.asarray(mesh["positions"])
                for mesh in self.scene["meshes"] if mesh["object_id"] == building["id"]
            ])
            np.testing.assert_allclose(points.min(axis=0), building["min"], atol=1e-6)
            np.testing.assert_allclose(points.max(axis=0), building["max"], atol=1e-6)

    def test_pickup_and_destination_are_on_generated_roofs(self):
        roofs = {roof["id"]: roof for roof in self.scene["roofs"]}
        buildings = {building["id"]: building for building in self.scene["buildings"]}
        pickup = roofs[self.scene["pickup_roof_id"]]
        destination = roofs[self.scene["landing_roof_id"]]
        self.assertEqual(pickup["building_id"], "city-4")
        self.assertEqual(destination["building_id"], "city-10")
        self.assertGreater(pickup["position"][1], 7)
        self.assertGreater(destination["position"][1], 9)
        self.assertLess(destination["position"][2], pickup["position"][2] - 10)
        np.testing.assert_allclose(self.scene["mj_spawn"],
                                   [pickup["position"][0], pickup["position"][1] + .96,
                                    pickup["position"][2]])
        for roof in self.scene["roofs"]:
            bounds = buildings[roof["building_id"]]
            for axis in (0, 2):
                self.assertGreaterEqual(roof["position"][axis], bounds["min"][axis])
                self.assertLessEqual(roof["position"][axis], bounds["max"][axis])
            self.assertLessEqual(roof["position"][1], bounds["max"][1] + 1e-6)
        for anchor in self.scene["anchors"]:
            roof = roofs[f"roof-{anchor['building_id']}"]
            building = buildings[anchor["building_id"]]
            positions = np.concatenate([
                np.asarray(mesh["positions"])
                for mesh in self.scene["meshes"] if mesh["object_id"] == anchor["building_id"]
            ])
            self.assertTrue(np.any(np.all(positions == anchor["position"], axis=1)))
            self.assertGreater(anchor["position"][1], roof["position"][1] - 1.4)
            self.assertLessEqual(anchor["position"][1], building["max"][1] + 1e-6)
            street_x = building["max"][0] if anchor["position"][0] < 0 else building["min"][0]
            planar = np.hypot(anchor["position"][0] - street_x,
                              anchor["position"][2] - building["max"][2])
            self.assertGreater(anchor["terminal_allowance_m"], planar)


if __name__ == "__main__":
    unittest.main()
