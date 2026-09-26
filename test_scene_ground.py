"""Regression coverage for stage surfaces beneath authored environments."""

import unittest

from scene_composition import make_preset
from scene_ground import has_authored_ground, studio_surface_visibility
from scene_objects import make_object


class SceneGroundTests(unittest.TestCase):
    def test_scene_floors_replace_studio_surfaces(self):
        for name in (
            'Rooftop swing district', 'Harbor chase', 'Jungle temple',
            'City boulevard', 'Residential neighborhood', 'Market square',
            'Warehouse workshop', 'Designed apartment',
            'Neon research lab', 'Cozy living room', 'Winter plaza',
        ):
            with self.subTest(name=name):
                objects = make_preset(name)['objects']
                self.assertTrue(has_authored_ground(objects))
                self.assertEqual(studio_surface_visibility(objects), (False, False, False))

    def test_object_only_scene_preserves_user_choices(self):
        objects = [make_object('table', 0), make_object('crate', 1)]
        self.assertFalse(has_authored_ground(objects))
        self.assertEqual(studio_surface_visibility(objects), (True, True, True))
        self.assertEqual(studio_surface_visibility(objects, show_grid=False), (True, False, True))
        self.assertEqual(studio_surface_visibility(objects, show_platform=False), (True, True, False))

    def test_surface_removal_restores_choices(self):
        ground = make_preset('Designed apartment')['objects'][0]
        self.assertEqual(studio_surface_visibility([ground], show_grid=False), (False, False, False))
        self.assertEqual(studio_surface_visibility([], show_grid=False), (True, False, True))

    def test_elevated_table_and_small_platform_do_not_hide_floor(self):
        table = make_object('table', 0)
        table['size'] = [7, .5, 6]
        table['position'] = [0, 1.0, 0]
        small = make_object('platform', 1)
        self.assertFalse(has_authored_ground([table, small]))


if __name__ == '__main__':
    unittest.main()
