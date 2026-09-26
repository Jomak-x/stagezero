"""Structural and composition checks for procedural environment presets."""

import itertools
import math
import unittest

from scene_environments import make_city, make_residential, make_room


def _expanded_bounds(part):
    repeat = part.get("repeat", {"count": [1, 1, 1], "step": [0, 0, 0]})
    position = part["position"]
    size = part["size"]
    # Only the picture uses part rotation; its rotated box is checked exactly.
    angle = math.radians(part.get("rotation", [0, 0, 0])[2])
    extent = [(abs(math.cos(angle))*size[0] + abs(math.sin(angle))*size[1])/2,
              (abs(math.sin(angle))*size[0] + abs(math.cos(angle))*size[1])/2,
              size[2]/2]
    for index in itertools.product(*(range(n) for n in repeat["count"])):
        center = [position[axis] + index[axis]*repeat["step"][axis] for axis in range(3)]
        yield [(center[axis]-extent[axis], center[axis]+extent[axis]) for axis in range(3)]


class SceneEnvironmentTests(unittest.TestCase):
    def test_presets_are_deterministic_and_vary_by_seed(self):
        for factory in (make_city, make_residential, make_room):
            with self.subTest(factory=factory.__name__):
                self.assertEqual(factory(0), factory(0))
                self.assertNotEqual(factory(0), factory(1))
                self.assertEqual(factory(0)["version"], 3)

    def test_assets_fit_geometry_limits_and_are_referenced(self):
        for factory in (make_city, make_residential, make_room):
            with self.subTest(factory=factory.__name__):
                scene = factory(7)
                assets = scene["assets"]
                self.assertLessEqual(len(assets), 16)
                self.assertLessEqual(len(scene["objects"]), 64)
                ids = [asset["id"] for asset in assets]
                self.assertEqual(len(ids), len(set(ids)))
                expanded_total = 0
                for asset in assets:
                    self.assertLessEqual(len(asset["parts"]), 64)
                    expanded = 0
                    for part in asset["parts"]:
                        for bounds in _expanded_bounds(part):
                            expanded += 1
                            for low, high in bounds:
                                self.assertGreaterEqual(low, -.500001, (asset["id"], part))
                                self.assertLessEqual(high, .500001, (asset["id"], part))
                    self.assertLessEqual(expanded, 512, asset["id"])
                    expanded_total += expanded
                self.assertLessEqual(expanded_total, 12000)
                shape_counts = {asset["id"]: sum(1 for part in asset["parts"]
                                                  for _ in _expanded_bounds(part)) for asset in assets}
                self.assertLessEqual(sum(shape_counts[obj["asset"]] for obj in scene["objects"]), 12000)
                self.assertTrue(all(obj["asset"] in ids for obj in scene["objects"]))

    def test_city_has_open_performance_area_and_skyline(self):
        scene = make_city(2)
        objects = scene["objects"]
        buildings = [obj for obj in objects if obj["asset"] in
                     {"masonry", "office", "brownstone", "storefront"}]
        self.assertGreaterEqual(len(buildings), 7)
        self.assertGreaterEqual(len([obj for obj in buildings if obj["position"][2] < -7]), 5)
        for obj in buildings:
            # The character and nearest 2 m ground plane are free of solid structures.
            inner_x = max(abs(obj["position"][0]) - obj["size"][0]/2, 0)
            inner_z = max(abs(obj["position"][2]) - obj["size"][2]/2, 0)
            self.assertGreater(math.hypot(inner_x, inner_z), 2)

    def test_city_reads_as_continuous_streetfront_blocks(self):
        for seed in (0, 1, 2, 99):
            with self.subTest(seed=seed):
                scene = make_city(seed)
                self.assertEqual(len(scene["objects"]), 50)
                fronts = [obj for obj in scene["objects"] if obj["name"] == "Streetfront building"]
                skyline = [obj for obj in scene["objects"] if obj["name"] == "Distant skyline building"]
                self.assertEqual(len(fronts), 18)
                self.assertEqual(len(skyline), 6)
                for side in (-1, 1):
                    row = sorted((obj for obj in fronts if obj["position"][0]*side > 0),
                                 key=lambda obj: obj["position"][2], reverse=True)
                    self.assertEqual(len(row), 9)
                    self.assertTrue(all(obj["yaw"] == (90 if side == -1 else -90)
                                        for obj in row))
                    self.assertTrue(all(abs(a["position"][2]-b["position"][2]) -
                                        (a["size"][0]+b["size"][0])/2 < .2
                                        for a, b in zip(row, row[1:])))
                    self.assertGreater(min(abs(obj["position"][0])-obj["size"][2]/2
                                           for obj in row), 5.4)
                self.assertEqual(len([obj for obj in scene["objects"]
                                      if obj["asset"] == "street-tree"]), 10)
                self.assertEqual(len([obj for obj in scene["objects"]
                                      if obj["asset"] == "street-lamp"]), 10)

    def test_residential_is_a_row_of_street_facing_houses(self):
        for seed in (0, 1, 2, 99):
            with self.subTest(seed=seed):
                scene = make_residential(seed)
                self.assertEqual(len(scene["objects"]), 56)
                houses = [obj for obj in scene["objects"] if obj["asset"].startswith("house-")]
                fences = [obj for obj in scene["objects"] if obj["asset"] == "residential-fence"]
                self.assertEqual(len(houses), 16)
                self.assertEqual(len(fences), 16)
                self.assertEqual(len([obj for obj in scene["objects"]
                                      if obj["asset"] == "mailbox"]), 4)
                for side in (-1, 1):
                    row = sorted((obj for obj in houses if obj["position"][0]*side > 0),
                                 key=lambda obj: obj["position"][2], reverse=True)
                    self.assertEqual(len(row), 8)
                    self.assertTrue(all(obj["yaw"] == (90 if side == -1 else -90)
                                        for obj in row))
                    self.assertEqual([round(obj["position"][2], 1) for obj in row],
                                     [-2.5-4*i for i in range(8)])
                    self.assertGreater(min(abs(obj["position"][0])-obj["size"][2]/2
                                           for obj in row), 6.8)
                self.assertTrue(all(abs(obj["position"][0]) > 4
                                    for obj in houses+fences))

    def test_room_is_architectural_and_furniture_rests_on_floor(self):
        scene = make_room(0)
        names = {obj["asset"] for obj in scene["objects"]}
        self.assertTrue({"rear-wall", "side-wall", "floorboards", "sofa", "coffee-table",
                         "bookshelf", "floor-lamp", "indoor-plant", "wall-art"} <= names)
        for obj in scene["objects"]:
            if obj["asset"] not in {"rear-wall", "side-wall", "wall-art", "woven-rug", "floorboards"}:
                self.assertAlmostEqual(obj["position"][1] - obj["size"][1]/2, 0, places=5)

    def test_preset_documents_pass_scene_validation(self):
        from scene_composition import validate_scene
        for factory in (make_city, make_residential, make_room):
            scene = factory(3)
            self.assertEqual(validate_scene(scene), scene)


if __name__ == "__main__":
    unittest.main()
