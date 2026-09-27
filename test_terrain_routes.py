"""Rendered terrain routing only; these tests do not validate generated gait."""
import copy
import math
import unittest

import numpy as np

from scene_interaction_geometry import SceneInteractionGeometry
from scene_navigation import plan_navigation_route


def box_asset(identifier='floor'):
    return {'id': identifier, 'name': identifier, 'parts': [
        {'shape': 'box', 'position': [0, 0, 0], 'size': [1, 1, 1], 'color': [100]*3}]}


def custom(identifier, asset, position, size):
    return dict(id=identifier, name=identifier, kind='custom', asset=asset,
                position=position, size=size)


def staircase_scene():
    stairs = {'id': 'steps', 'name': 'Broad blocks', 'parts': [
        {'shape': 'box', 'position': [0, -.3+.2*i, .3-.2*i], 'size': [1, .2, .25],
         'color': [100]*3} for i in range(4)]}
    return {'assets': [box_asset(), stairs], 'objects': [
        custom('ground', 'floor', [0, -.1, 0], [10, .2, 10]),
        custom('flight', 'steps', [0, .4, 0], [2, .8, 1.7])]}


class TerrainRouteTests(unittest.TestCase):
    def test_route_preserves_each_real_tread_elevation(self):
        geometry = SceneInteractionGeometry.from_scene(staircase_scene())
        self.assertEqual(len(geometry.stair_routes), 1)
        route = plan_navigation_route(geometry, [0, 0, 1.2], [0, .8, -.6], max_step_up=.25)
        np.testing.assert_allclose(np.unique(np.round(route[:, 1], 5)), [0, .2, .4, .6, .8])
        self.assertGreater(len(route), 5)
        for x, y, z in route:
            self.assertAlmostEqual(geometry.support_height(x, z, y, .01, .01), y)

    def test_rotated_translated_terrain_keeps_world_support_and_treads(self):
        original = staircase_scene()
        base = SceneInteractionGeometry.from_scene(original)
        angle = math.radians(37)
        rotation = np.array(((math.cos(angle), 0, math.sin(angle)), (0, 1, 0),
                             (-math.sin(angle), 0, math.cos(angle))))
        offset = np.array((8., 5., -4.))
        scene = copy.deepcopy(original)
        for obj in scene['objects']:
            obj['position'] = (np.array(obj['position']) @ rotation.T + offset).tolist()
            obj['yaw'] = 37
        transformed = SceneInteractionGeometry.from_scene(scene)
        np.testing.assert_allclose(transformed.stair_routes[0]['steps'],
                                   np.array(base.stair_routes[0]['steps']) @ rotation.T + offset)
        start = np.array((0, 0, 1.2)) @ rotation.T + offset
        end = np.array((0, .8, -.6)) @ rotation.T + offset
        route = plan_navigation_route(transformed, start, end, max_step_up=.25)
        np.testing.assert_allclose((route[-1]-offset) @ rotation, [0, .8, -.6], atol=1e-6)
        self.assertNotIn('__studio_floor__', {s.object_id for s in transformed.walkable_surfaces})
        self.assertIsNone(transformed.support_height(50, 50, 0))

    def test_authored_gap_and_unsafe_water_never_gain_an_implicit_floor(self):
        scene = {'assets': [box_asset()], 'objects': [
            custom('near', 'floor', [0, -.1, 3], [10, .2, 4]),
            custom('far', 'floor', [0, -.1, -3], [10, .2, 4])]}
        geometry = SceneInteractionGeometry.from_scene(scene)
        self.assertIsNone(geometry.support_height(0, 0, 0))
        with self.assertRaisesRegex(ValueError, 'No safe walking route'):
            plan_navigation_route(geometry, [0, 0, 2], [0, 0, -2], max_expansions=100)
        scene['objects'] = [custom('water', 'floor', [0, -.1, 0], [2, .2, 2])]
        self.assertIsNone(SceneInteractionGeometry.from_scene(scene).support_height(0, 0, 0))

    def test_bridge_and_ground_are_distinct_support_layers(self):
        scene = {'assets': [box_asset()], 'objects': [
            custom('ground', 'floor', [0, -.1, 0], [10, .2, 10]),
            custom('bridge', 'floor', [0, 3, 0], [2, .2, 5])]}
        geometry = SceneInteractionGeometry.from_scene(scene)
        self.assertAlmostEqual(geometry.support_height(0, 0, 0, .25, .35), 0)
        self.assertAlmostEqual(geometry.support_height(0, 0, 3.1, .25, .35), 3.1)
        with self.assertRaisesRegex(ValueError, 'No safe walking route'):
            plan_navigation_route(geometry, [0, 0, 3], [0, 3.1, 0], max_expansions=100)

    def test_prop_only_scene_retains_the_ordinary_studio_floor(self):
        geometry = SceneInteractionGeometry.from_scene({'objects': []})
        route = plan_navigation_route(geometry, [0, 0, 0], [2, 0, 1])
        np.testing.assert_allclose(route, [[0, 0, 0], [2, 0, 1]])

    def test_closed_and_open_door_geometry_matches_real_aperture(self):
        door = dict(id='door', kind='door', name='Door', size=[2.5, 3, .2], position=[0, 1.5, 0])
        closed = SceneInteractionGeometry.from_scene({'objects': [door]})
        self.assertTrue(closed.obstacle_at(0, 1, 0, radius=.2))
        opened = SceneInteractionGeometry.from_scene({'objects': [dict(door, position=[0, 4.5, 0])]})
        self.assertFalse(opened.obstacle_at(0, 1, 0, radius=.2))
        self.assertFalse(opened.obstacle_at(1.15, 1, 0, radius=.2))
        self.assertTrue(opened.obstacle_at(1.15, 4, 0, radius=.2))

    def test_unrendered_hinge_angle_cannot_create_a_phantom_passage(self):
        door = dict(id='door', kind='door', name='Door', size=[2.5, 3, .2], position=[0, 1.5, 0])
        for angle in (-90, 90, 180, float('nan')):
            geometry = SceneInteractionGeometry.from_scene({'objects': [dict(door, _open_angle=angle)]})
            self.assertTrue(geometry.obstacle_at(0, 1, 0, radius=.2))
            with self.assertRaisesRegex(ValueError, 'no clear walkable approach'):
                plan_navigation_route(geometry, [0, 0, 1], [0, 0, 0])

    def test_invalid_limits_and_unsupported_endpoints_fail_clearly(self):
        geometry = SceneInteractionGeometry.from_scene({'objects': []})
        for kwargs in ({'radius': -1}, {'max_step_up': float('nan')}, {'sample_spacing': 1},
                       {'max_expansions': True}):
            with self.assertRaises(ValueError):
                plan_navigation_route(geometry, [0, 0, 0], [1, 0, 0], **kwargs)
        with self.assertRaisesRegex(ValueError, 'no clear walkable approach'):
            plan_navigation_route(geometry, [0, 5, 0], [1, 0, 0])


class TempleTerrainRouteTests(unittest.TestCase):
    """Exercise the shipped recipe, not a private repaired terrain fixture."""

    def scene(self, angle=0., offset=(0., 0., 0.)):
        from scene_composition import make_preset
        scene = make_preset('Jungle temple')
        angle_radians = math.radians(angle)
        rotation = np.array(((math.cos(angle_radians), 0, math.sin(angle_radians)), (0, 1, 0),
                             (-math.sin(angle_radians), 0, math.cos(angle_radians))))
        offset = np.asarray(offset)
        for obj in scene['objects']:
            obj['position'] = (np.array(obj['position']) @ rotation.T + offset).tolist()
            obj['yaw'] = obj.get('yaw', 0.) + angle
        return scene, lambda p: np.asarray(p) @ rotation.T + offset

    def test_shipped_temple_support_connects_each_flight_bridge_and_gate(self):
        for angle, offset in ((0., (0., 0., 0.)), (37., (4., 6., -3.))):
            with self.subTest(angle=angle, offset=offset):
                scene, transform = self.scene(angle, offset)
                geometry = SceneInteractionGeometry.from_scene(scene)
                flights = {route['name']: route for route in geometry.stair_routes}
                self.assertEqual(len(flights['Ceremonial stairs']['steps']), 9)
                self.assertEqual(len(flights['Bridge approach stairs']['steps']), 9)
                self.assertEqual(len(flights['Far ceremonial stairs']['steps']), 11)
                route = plan_navigation_route(geometry, transform([0, 0, 0]),
                                              transform([0, 5.6, -21.9]),
                                              radius=.18, max_step_up=.25)
                self.assertGreater(len(route), 50)
                self.assertLessEqual(float(np.diff(route[:, 1]).max()), .25)
                self.assertGreater(route[-1, 1]-offset[1], 5.4)
                for x, y, z in route:
                    self.assertAlmostEqual(geometry.support_height(x, z, y, .01, .01), y)
                    self.assertFalse(geometry.obstacle_at(x, y+.9, z, radius=.18))
                self.assertNotIn('__studio_floor__', {s.object_id for s in geometry.walkable_surfaces})

    def test_raised_gate_blocks_center_until_observed_open(self):
        for angle, offset in ((0., (0., 0., 0.)), (37., (4., 6., -3.))):
            with self.subTest(angle=angle, offset=offset):
                scene, transform = self.scene(angle, offset)
                geometry = SceneInteractionGeometry.from_scene(scene)
                start, middle, end = map(transform, ([0, 5.6, -21.9], [0, 5.6, -22.38], [0, 5.6, -23.28]))
                with self.assertRaisesRegex(ValueError, 'no clear walkable approach'):
                    plan_navigation_route(geometry, start, middle, radius=.18, max_step_up=.25)
                # This is the evaluated assembly lift used by the renderer
                # and core_scene_reactions, not a fictitious hinge or exemption.
                door = next(obj for obj in scene['objects'] if obj['kind'] == 'door')
                door['position'][1] += door['size'][1]
                opened = SceneInteractionGeometry.from_scene(scene)
                entry = plan_navigation_route(opened, start, middle, radius=.18, max_step_up=.25)
                exit_route = plan_navigation_route(opened, entry[-1], end, radius=.18, max_step_up=.25)
                self.assertGreater(exit_route[-1, 1]-offset[1], 5.4)
                for point in (entry[-1], exit_route[-1]):
                    self.assertFalse(opened.obstacle_at(point[0], point[1]+.9, point[2], radius=.18))

    def test_missing_bridge_leaves_a_real_ravine_gap(self):
        scene, transform = self.scene(37., (4., 6., -3.))
        scene['objects'] = [obj for obj in scene['objects'] if obj['name'] != 'Suspended ravine bridge']
        geometry = SceneInteractionGeometry.from_scene(scene)
        x, y, z = transform([0., 3., -14.])
        self.assertIsNone(geometry.support_height(x, z, y, .25, .35))


if __name__ == '__main__':
    unittest.main()
