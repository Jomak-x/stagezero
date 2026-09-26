"""Composition and budget checks for the two large cinematic action sets."""

import unittest

from asset_geometry import compile_asset, expanded_count, triangle_count
from cinematic_adventure import make_dockyard, make_temple, suggested_targets
from scene_composition import encode_scene, validate_scene


class CinematicAdventureTests(unittest.TestCase):
    def test_sets_are_valid_deterministic_and_portable(self):
        for factory in (make_dockyard, make_temple):
            with self.subTest(factory=factory.__name__):
                scene = factory(7)
                self.assertEqual(scene, factory(7))
                self.assertNotEqual(scene, factory(8))
                self.assertEqual(validate_scene(scene), validate_scene(factory(7)))
                self.assertLess(len(encode_scene(scene)), 1_000_000)
                self.assertEqual(scene["version"], 3)
                self.assertGreaterEqual(len(scene["objects"]), 40)
                self.assertLessEqual(len(scene["objects"]), 64)
                self.assertLessEqual(len(scene["assets"]), 16)
                lookup = {a["id"]: a for a in scene["assets"]}
                custom = [o for o in scene["objects"] if o["kind"] == "custom"]
                self.assertTrue(all(o["asset"] in lookup for o in custom))
                self.assertLessEqual(sum(expanded_count(lookup[o["asset"]]) for o in custom), 12_000)
                self.assertLessEqual(sum(triangle_count(lookup[o["asset"]]) for o in custom), 250_000)
                self.assertTrue(all(-100 <= v <= 100 for o in scene["objects"] for v in o["position"]))
                self.assertTrue(all(.05 <= v <= 60 for o in scene["objects"] for v in o["size"]))

    def test_dockyard_is_a_continuous_chase_route_with_height(self):
        scene = make_dockyard()
        objects = scene["objects"]
        containers = [o for o in objects if o.get("asset", "").startswith("dock-container")]
        self.assertGreaterEqual(len(containers), 20)
        self.assertTrue(any(o["position"][1] > 6 for o in containers))
        self.assertTrue(all(abs(o["position"][0]) - o["size"][0] / 2 > 3.9
                            for o in containers))
        self.assertEqual(sum(o.get("asset") == "dock-gantry" for o in objects), 2)
        self.assertEqual(sum(o.get("asset") == "dock-catwalk" for o in objects), 2)
        self.assertTrue(any(o["kind"] == "console" and o["interaction"]["action"] == "activate"
                            for o in objects))
        self.assertTrue(any(o.get("asset") == "dock-water" and o["position"][0] > 14
                            for o in objects))

    def test_temple_has_separated_terraces_and_real_crossing(self):
        scene = make_temple()
        objects = scene["objects"]
        near, far = [o for o in objects if o["name"] in ("Near jungle ground", "Far jungle ground")]
        near_back = near["position"][2] - near["size"][2] / 2
        far_front = far["position"][2] + far["size"][2] / 2
        self.assertGreater(near_back - far_front, 4)
        bridge = next(o for o in objects if o.get("asset") == "temple-bridge")
        self.assertTrue(far_front < bridge["position"][2] < near_back)
        landings = [o for o in objects if o["name"] in ("Near bridge landing", "Far bridge landing")]
        landing_gap = abs(landings[0]["position"][2] - landings[1]["position"][2]) - landings[0]["size"][2]
        self.assertGreater(bridge["size"][2], landing_gap)
        self.assertGreaterEqual(sum(o.get("asset") == "temple-tree" for o in objects), 10)
        self.assertGreaterEqual(sum(o.get("asset") == "temple-tower" for o in objects), 4)
        self.assertTrue(any(o["kind"] == "door" and o["interaction"]["action"] == "open"
                            for o in objects))

    def test_targets_reference_modeled_props_and_stay_local(self):
        for factory in (make_dockyard, make_temple):
            with self.subTest(factory=factory.__name__):
                scene = factory()
                objects = {o["id"]: o for o in scene["objects"]}
                targets = suggested_targets(scene)
                self.assertGreaterEqual(len(targets), 20)
                self.assertEqual(len({t["id"] for t in targets}), len(targets))
                for target in targets:
                    self.assertEqual(set(target), {"id", "name", "object_id", "kind", "local_position"})
                    self.assertIn(target["object_id"], objects)
                    self.assertIn(target["kind"], {"swing_anchor", "landing", "climb", "vault"})
                    self.assertEqual(len(target["local_position"]), 3)
                    self.assertTrue(all(-.5 <= v <= .5 for v in target["local_position"]))
                self.assertTrue(any(t["kind"] == "swing_anchor" for t in targets))
                self.assertTrue(any(t["kind"] == "landing" for t in targets))

    def test_bridge_deck_door_and_origin_match_rendered_surfaces(self):
        scene = make_temple()
        objects = {o["name"]: o for o in scene["objects"]}
        targets = {t["object_id"]: t for t in suggested_targets(scene) if t["kind"] == "landing"}

        def landing_y(name):
            obj = objects[name]
            return obj["position"][1] + obj["size"][1] * targets[obj["id"]]["local_position"][1]

        self.assertAlmostEqual(landing_y("Starting court"), .02, places=5)
        self.assertAlmostEqual(landing_y("Near bridge landing"),
                               landing_y("Suspended ravine bridge"), places=3)
        self.assertAlmostEqual(landing_y("Far bridge landing"),
                               landing_y("Suspended ravine bridge"), places=3)
        door = objects["Shrine door"]
        self.assertAlmostEqual(door["position"][1] - door["size"][1] / 2,
                               landing_y("High temple dais"), places=5)

    def test_paving_layers_and_harbor_water_are_visible(self):
        temple = make_temple()
        terrace = next(a for a in temple["assets"] if a["id"] == "temple-terrace")
        body, grout, tile = terrace["parts"][:3]
        self.assertLess(body["position"][1] + body["size"][1] / 2,
                        grout["position"][1] + grout["size"][1] / 2)
        self.assertLess(grout["position"][1] + grout["size"][1] / 2,
                        tile["position"][1] - tile["size"][1] / 2)
        self.assertGreater(tile["repeat"]["step"][0], tile["size"][0])
        self.assertGreater(tile["repeat"]["step"][2], tile["size"][2])

        dock = make_dockyard()
        water = next(o for o in dock["objects"] if o.get("asset") == "dock-water")
        asset = next(a for a in dock["assets"] if a["id"] == "dock-water")
        vertices, _, _ = compile_asset(asset)
        lower, upper = vertices[:, 1].min(), vertices[:, 1].max()
        body_top = asset["parts"][0]["position"][1] + asset["parts"][0]["size"][1] / 2
        rendered_top = water["position"][1] + water["size"][1] * (body_top - (lower + upper) / 2) / (upper - lower)
        self.assertGreater(rendered_top, -.02)  # Default stage floor.
        self.assertLess(rendered_top, 0)


if __name__ == "__main__":
    unittest.main()
