"""Regression checks for generated prop geometry defects in the saved city."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from asset_geometry import _outward, _primitive, compile_asset, validate_assets
from asset_quality import assess_assets, refine_assets


CITY = Path(__file__).parent / 'review' / 'demo' / 'fresh-city.json'


class AssetQualityTests(unittest.TestCase):
    def test_saved_city_window_slabs_and_floating_details_are_repaired(self):
        source = json.loads(CITY.read_text())['assets']
        original = copy.deepcopy(source)
        issues = assess_assets(source)
        self.assertEqual(sum(i['code'] == 'repeat_overlap' for i in issues), 4)
        self.assertEqual(sum(i['code'] == 'facade_occlusion' for i in issues), 2)
        self.assertGreaterEqual(sum(i['code'] == 'detached_detail' for i in issues), 3)

        repaired = refine_assets(source)
        self.assertEqual(source, original)  # no caller-owned mutation
        self.assertEqual(repaired, refine_assets(repaired))
        self.assertEqual(assess_assets(repaired), [])
        self.assertEqual(validate_assets(repaired), repaired)

        for asset in repaired:
            if 'Building' not in asset['name']:
                continue
            grid = next((p for p in asset['parts'] if p.get('repeat') and
                         p['repeat']['count'][0] >= 3 and p['repeat']['count'][1] >= 5), None)
            if grid is None:
                continue
            for axis in (0, 1):
                self.assertLess(grid['size'][axis], abs(grid['repeat']['step'][axis]))
            dark = [p for p in asset['parts'] if p is not grid and p['shape'] == 'box'
                    and sum(p['color']) < sum(grid['color']) * .8 and
                    p['size'][2] < .01 and p['size'][0] > .35]
            self.assertTrue(dark)
            self.assertGreater(grid['position'][2] - grid['size'][2] / 2,
                               max(p['position'][2] + p['size'][2] / 2 for p in dark))

    def test_repeated_posts_keep_intentional_overlap(self):
        asset = {'id': 'railing', 'name': 'Railing', 'parts': [
            {'shape': 'box', 'position': [0, 0, 0], 'size': [.8, .04, .04], 'color': [70, 70, 70]},
            {'shape': 'box', 'position': [-.3, -.1, 0], 'size': [.04, .3, .04],
             'color': [80, 80, 80], 'repeat': {'count': [4, 1, 1], 'step': [.2, 0, 0]}},
        ]}
        self.assertFalse(any(i['code'] == 'repeat_overlap' for i in assess_assets([asset])))
        self.assertEqual(refine_assets([asset]), validate_assets([asset]))

    def test_small_coplanar_overlay_is_moved_without_exceeding_bounds(self):
        asset = {'id': 'sign', 'name': 'Wall sign', 'parts': [
            {'shape': 'box', 'position': [0, 0, 0], 'size': [.8, .8, .8], 'color': [100, 100, 100]},
            {'shape': 'box', 'position': [0, 0, .395], 'size': [.2, .1, .01], 'color': [200, 200, 200]},
        ]}
        self.assertIn('coplanar_surface', [i['code'] for i in assess_assets([asset])])
        repaired = refine_assets([asset])
        self.assertNotIn('coplanar_surface', [i['code'] for i in assess_assets(repaired)])
        self.assertGreater(repaired[0]['parts'][1]['position'][2], .395)
        compile_asset(repaired[0])

    def test_floating_detail_reaches_main_mass_without_swallowing_it(self):
        road = {'id': 'road', 'name': 'Road with paint', 'parts': [
            {'shape': 'box', 'position': [0, -.015, 0], 'size': [1, .94, 1], 'color': [50, 50, 50]},
            {'shape': 'box', 'position': [0, .49, 0], 'size': [.02, .012, .15], 'color': [250, 240, 210]},
        ]}
        fixed = refine_assets([road])[0]
        self.assertEqual(fixed['parts'][0], validate_assets([road])[0]['parts'][0])
        self.assertGreater(fixed['parts'][1]['size'][1], road['parts'][1]['size'][1])
        self.assertEqual(assess_assets([fixed]), [])

    def test_compiled_primitive_faces_have_valid_outward_normals(self):
        # Distinguishes recipe occlusion from a winding bug in GLB meshes.
        for shape in ('box', 'sphere', 'cylinder', 'cone'):
            with self.subTest(shape=shape):
                vertices, faces = _primitive(shape)
                tri = vertices[_outward(vertices, faces)]
                normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                self.assertTrue(np.all(np.linalg.norm(normals, axis=1) > 1e-10))
                self.assertTrue(np.all(np.sum(normals * tri.mean(axis=1), axis=1) >= -1e-10))


if __name__ == '__main__':
    unittest.main()
