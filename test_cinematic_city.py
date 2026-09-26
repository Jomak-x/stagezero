"""Structural checks for the reusable multi-block swing city."""

import unittest

from asset_geometry import compile_asset, expanded_count, triangle_count
from cinematic_city import make_swing_city, suggested_targets
from scene_composition import encode_scene, validate_scene


class SwingCityTests(unittest.TestCase):
    def test_dense_city_is_valid_portable_and_seeded(self):
        for seed in (0, 3, 1_000_000):
            scene = make_swing_city(seed)
            self.assertEqual(scene, validate_scene(scene))
            self.assertEqual(scene, make_swing_city(seed))
            self.assertLess(len(encode_scene(scene)), 1_000_000)
            self.assertLessEqual(len(scene['objects']), 64)
            self.assertLessEqual(len(scene['assets']), 16)
            assets = {asset['id']: asset for asset in scene['assets']}
            self.assertLessEqual(sum(expanded_count(assets[obj['asset']])
                                     for obj in scene['objects']), 12_000)
            self.assertLessEqual(sum(triangle_count(assets[obj['asset']])
                                     for obj in scene['objects']), 250_000)
            buildings = [obj for obj in scene['objects']
                         if obj['id'].startswith('city-building-')]
            self.assertEqual(len(buildings), 32)
            self.assertGreaterEqual(len({obj['size'][1] for obj in buildings}), 4)
            self.assertGreater(max(obj['position'][0] for obj in buildings) -
                               min(obj['position'][0] for obj in buildings), 55)
            self.assertGreater(max(obj['position'][2] for obj in buildings) -
                               min(obj['position'][2] for obj in buildings), 55)
            roads = [obj for obj in scene['objects'] if obj['asset'] == 'swing-road']
            self.assertEqual(len(roads), 8)
            self.assertEqual({obj.get('yaw', 0) for obj in roads}, {0, 90.0})
            crossings = [obj for obj in scene['objects']
                         if obj['asset'] == 'swing-crossing']
            self.assertEqual(len(crossings), 4)
            self.assertGreater(min(obj['position'][1] for obj in crossings),
                               max(obj['position'][1] for obj in roads))

    def test_buildings_have_windows_on_all_four_faces(self):
        scene = make_swing_city()
        for asset in scene['assets']:
            if not asset['id'].startswith('swing-') or asset['id'] in {
                    'swing-ground', 'swing-road', 'swing-plaza', 'swing-lamp',
                    'swing-bridge', 'swing-crossing'}:
                continue
            facade_glass = [part for part in asset['parts']
                            if 'repeat' in part and part['repeat']['count'][1] >= 7
                            and min(part['size'][0], part['size'][2]) <= .02]
            for axis in (0, 2):
                self.assertTrue(any(part['position'][axis] < -.44 for part in facade_glass),
                                asset['id'])
                self.assertTrue(any(part['position'][axis] > .44 for part in facade_glass),
                                asset['id'])

    def test_targets_refer_to_roofs_walls_and_bridges(self):
        scene = make_swing_city(7)
        objects = {obj['id']: obj for obj in scene['objects']}
        assets = {asset['id']: asset for asset in scene['assets']}
        bounds = {}

        def normal_y(asset_id, raw_y):
            if asset_id not in bounds:
                vertices, _, _ = compile_asset(assets[asset_id])
                bounds[asset_id] = (vertices.min(axis=0), vertices.max(axis=0))
            lower, upper = bounds[asset_id]
            return float((raw_y - (lower[1] + upper[1]) / 2) /
                         (upper[1] - lower[1]))

        targets = suggested_targets(scene)
        self.assertEqual(len(targets), 98)
        self.assertEqual(len({target['id'] for target in targets}), len(targets))
        self.assertEqual({target['kind'] for target in targets},
                         {'swing_anchor', 'landing', 'climb', 'vault'})
        for target in targets:
            obj = objects[target['object_id']]
            self.assertEqual(len(target['local_position']), 3)
            self.assertTrue(all(-.5 <= value <= .5 for value in target['local_position']))
            world = [obj['position'][axis] + obj['size'][axis] * value
                     for axis, value in enumerate(target['local_position'])]
            self.assertTrue(all(abs(value) < 100 for value in world))
            if target['kind'] == 'landing':
                self.assertGreater(world[1], 10)
                expected_roof = (obj['position'][1] + obj['size'][1] *
                                 normal_y(obj['asset'], .416))
                self.assertAlmostEqual(world[1], expected_roof, places=5)
            if target['kind'] == 'swing_anchor':
                expected_parapet = (obj['position'][1] + obj['size'][1] *
                                    normal_y(obj['asset'], .44))
                self.assertAlmostEqual(world[1], expected_parapet, places=5)
        for bridge in (obj for obj in objects.values()
                       if obj['asset'] == 'swing-bridge'):
            deck = bridge['position'][1] + bridge['size'][1] * normal_y('swing-bridge', -.19)
            hosts = [obj for obj in objects.values()
                     if obj['id'].startswith('city-building-')
                     and abs(obj['position'][2] - bridge['position'][2]) < .001
                     and abs(abs(obj['position'][0] - bridge['position'][0]) - 4.15) < .001]
            self.assertEqual(len(hosts), 2)
            for host in hosts:
                roof = host['position'][1] + host['size'][1] * normal_y(host['asset'], .416)
                self.assertLess(abs(deck - roof), .15)


if __name__ == '__main__':
    unittest.main()
