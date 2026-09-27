"""CPU contracts for the reusable, real-geometry temple traversal fixture."""
from __future__ import annotations

import copy
import math
import unittest

import numpy as np

from core_scene_reactions import evaluated_scene, object_states
from core_spatial_commands import parse_commands, plan_command
from realtime_clip import CanonicalClip
from scene_composition import validate_scene
from scene_interaction_geometry import SceneInteractionGeometry
from studio_interaction_scene import adapt_studio_scene
from traversal_kit import (RISE_M, START_Z, STEP_COUNT, TEMPLE_START_ROOT_XYZ,
                           temple_route_metadata, traversable_temple_scene,
                           traversable_observatory_scene)


def _adapted(scene, evaluated=None):
    result = adapt_studio_scene(scene if evaluated is None else evaluated)
    result['original_scene'] = scene
    return result


def _roots(xyz, frames=1):
    positions = np.zeros((1, frames, 27, 3), np.float32)
    positions[0, :, :, :] = xyz
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32),
                                (1, frames, 27, 3, 3)).copy()
    return CanonicalClip(positions, rotations, 20, ('actor_1',), 'ardy_core', {},
                         np.zeros((1, frames, 330), np.float32))


def _transformed_scene(scene, *, angle_degrees, shift):
    result = copy.deepcopy(scene)
    theta = math.radians(angle_degrees)
    c, s = math.cos(theta), math.sin(theta)
    rotation = np.array(((c, 0., s), (0., 1., 0.), (-s, 0., c)))

    def move(point):
        return (rotation @ np.asarray(point, dtype=float) + shift).tolist()

    for obj in result['objects']:
        obj['position'] = move(obj['position'])
        if obj['kind'] in ('custom', 'door'):
            obj['yaw'] = obj.get('yaw', 0.) + angle_degrees
    for field in ('position', 'look_at'):
        result['camera'][field] = move(result['camera'][field])
    return validate_scene(result), move


def _plan_four_actions(scene, text, start_xyz, initial_yaw):
    actions = parse_commands(text, _adapted(scene))
    assert [action['verb'] for action in actions] == ['ascend', 'cross', 'open', 'go_through']
    history, routes = None, []
    for index, action in enumerate(actions):
        evaluated = scene
        if index == 3:
            held = _roots(history.positions[0, -1, 0], frames=20)
            evaluated = evaluated_scene(scene, held, enabled=True, terrain=True)
        stages, route = plan_command(action, _adapted(scene, evaluated),
                                     ('actor_1',), 'actor_1', history,
                                     {'actor_1': {'position_xz': [start_xyz[0], start_xyz[2]],
                                                  'yaw': initial_yaw}})
        assert stages and route['terrain_navigation_version'] == 1
        final = route['waypoints'][-1]
        history = _roots([final['position_xz'][0], final['support_y']+.95,
                          final['position_xz'][1]])
        routes.append(route)
    return routes


class TraversableTempleTests(unittest.TestCase):
    def test_real_scene_schema_support_and_gap(self):
        scene = validate_scene(traversable_temple_scene())
        geometry = SceneInteractionGeometry.from_scene(scene)
        self.assertEqual(len(geometry.stair_routes), 1)
        self.assertEqual(geometry.stair_routes[0]['object_id'], 'temple-stairs')
        self.assertEqual(len(geometry.stair_routes[0]['steps']), STEP_COUNT)
        self.assertEqual(temple_route_metadata()['start_root_xyz'], list(TEMPLE_START_ROOT_XYZ))
        for z, height in ((START_Z, 0.), (-.25, RISE_M), (-.75, 2*RISE_M),
                          (-1.25, 3*RISE_M), (-1.75, 4*RISE_M),
                          (-2.25, 5*RISE_M), (-5., .2), (-7.72, .2),
                          (-9.3, .2), (-10.5, .2)):
            with self.subTest(z=z):
                self.assertAlmostEqual(geometry.support_height(0., z, height,
                    max_step_up=.1, max_drop=.1), height, places=5)
        self.assertIsNone(geometry.support_height(2., -5., 0.,
                                                  max_step_up=.3, max_drop=.5))
        bridge = next(o for o in scene['objects'] if o['id'] == 'temple-bridge')
        self.assertTrue(bridge['name'].endswith('bridge'))

    def test_four_actions_with_observed_automatic_whole_door_lift(self):
        scene = validate_scene(traversable_temple_scene())
        actions = parse_commands(
            'walk up the shallow temple stairs, cross the bridge, open temple gate, and enter',
            _adapted(scene))
        self.assertEqual([action['verb'] for action in actions],
                         ['ascend', 'cross', 'open', 'go_through'])
        history = None
        endpoints = []
        for index, action in enumerate(actions):
            evaluated = scene
            if index == 3:
                held = _roots(endpoints[-1], frames=20)
                states = object_states(scene, held, enabled=True, terrain=True)
                door_state = next(row for row in states if row['id'] == 'temple-gate')
                self.assertAlmostEqual(door_state['opening_fraction'], 1.)
                evaluated = evaluated_scene(scene, held, enabled=True, terrain=True)
                original_door = next(o for o in scene['objects'] if o['id'] == 'temple-gate')
                lifted_door = next(o for o in evaluated['objects'] if o['id'] == 'temple-gate')
                self.assertAlmostEqual(lifted_door['position'][1]-original_door['position'][1],
                                       original_door['size'][1])
            stages, route = plan_command(action, _adapted(scene, evaluated),
                                         ('actor_1',), 'actor_1', history,
                                         {'actor_1': {'position_xz': [0., START_Z],
                                                      'yaw': math.pi}})
            self.assertTrue(stages)
            self.assertEqual(route['terrain_navigation_version'], 1)
            final = route['waypoints'][-1]
            endpoint = [final['position_xz'][0], final['support_y']+.95,
                        final['position_xz'][1]]
            endpoints.append(endpoint)
            history = _roots(endpoint)
        self.assertAlmostEqual(endpoints[-1][1], 1.15)
        self.assertEqual({row['role'] for row in route['waypoints']} &
                         {'entry', 'center', 'exit'}, {'entry', 'center', 'exit'})

    def test_missing_bridge_and_closed_gate_fail(self):
        scene = validate_scene(traversable_temple_scene())
        missing = copy.deepcopy(scene)
        missing['objects'] = [o for o in missing['objects'] if o['id'] != 'temple-bridge']
        with self.assertRaisesRegex(ValueError, 'Unknown terrain target'):
            parse_commands('cross bridge', _adapted(missing))
        action = parse_commands('open temple gate, enter', _adapted(scene))[1]
        approaching = _roots([0., 1.15, -8.72])
        with self.assertRaisesRegex(ValueError, 'not fully open'):
            plan_command(action, _adapted(scene), ('actor_1',), 'actor_1',
                         approaching, None)

    def test_translated_rotated_route_is_repeatable_across_headings(self):
        source = traversable_temple_scene()
        scene, move = _transformed_scene(source, angle_degrees=37.,
                                         shift=np.array((4., 1.4, -3.)))
        start = move(TEMPLE_START_ROOT_XYZ)
        facing_route = math.atan2(math.sin(math.pi + math.radians(37.)),
                                  math.cos(math.pi + math.radians(37.)))
        text = ('walk up the shallow temple stairs, cross the bridge, '
                'open temple gate, and enter')
        routes = _plan_four_actions(scene, text, start, facing_route)
        self.assertAlmostEqual(routes[0]['waypoints'][0]['support_y'], 1.4, places=4)
        self.assertAlmostEqual(routes[-1]['waypoints'][-1]['support_y'], 1.6, places=4)
        expected_exit = move((0., .2, -9.88))
        actual_exit = routes[-1]['waypoints'][-1]['position_xz']
        self.assertLess(np.linalg.norm(np.asarray(actual_exit)-
                                       np.asarray(expected_exit)[[0, 2]]), .03)
        self.assertEqual({row['role'] for row in routes[-1]['waypoints']} &
                         {'entry', 'center', 'exit'}, {'entry', 'center', 'exit'})
        # A different initial heading inserts a native turn, while repeated
        # planning from the same committed state remains deterministic.
        action = parse_commands('walk up the shallow temple stairs', _adapted(scene))[0]
        turned = math.atan2(math.sin(facing_route-math.pi/2),
                            math.cos(facing_route-math.pi/2))
        placement = {'actor_1': {'position_xz': [start[0], start[2]],
                                 'yaw': turned}}
        first_stages, first = plan_command(action, _adapted(scene),
                                           ('actor_1',), 'actor_1', None, placement)
        second_stages, second = plan_command(action, _adapted(scene),
                                             ('actor_1',), 'actor_1', None, placement)
        self.assertEqual(first['waypoints'], second['waypoints'])
        self.assertEqual(first['schedule'], second['schedule'])
        self.assertEqual(len(first_stages), len(second_stages))
        self.assertEqual(first['schedule']['initial_turn_frames'], 40)

    def test_renamed_wider_background_uses_same_generic_actions(self):
        scene = traversable_observatory_scene()
        self.assertFalse(any(obj['id'].startswith('temple-') for obj in scene['objects']))
        self.assertFalse(any(asset['id'].startswith('temple-') for asset in scene['assets']))
        geometry = SceneInteractionGeometry.from_scene(scene)
        self.assertEqual(geometry.stair_routes[0]['object_id'], 'observatory-stairs')
        self.assertAlmostEqual(geometry.support_height(1.1, -5., .2), .2, places=5)
        routes = _plan_four_actions(
            scene,
            'walk up the shallow observatory steps, cross the northern copper walkway, '
            'open observatory door, and enter',
            TEMPLE_START_ROOT_XYZ, math.pi)
        self.assertEqual([route['target_id'] for route in routes],
                         ['observatory-stairs', 'observatory-bridge',
                          'observatory-gate', 'observatory-gate'])
        self.assertAlmostEqual(routes[-1]['waypoints'][-1]['support_y'], .2, places=5)

    def test_wider_background_rejects_ambiguous_and_blocked_crossings(self):
        scene = traversable_observatory_scene()
        duplicate = copy.deepcopy(scene)
        bridge = next(obj for obj in duplicate['objects'] if obj['id'] == 'observatory-bridge')
        extra = copy.deepcopy(bridge)
        extra['id'] = 'second-copper-walkway'
        extra['name'] = 'Second copper walkway'
        extra['position'][0] += 4.
        duplicate['objects'].append(extra)
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            parse_commands('cross walkway', _adapted(duplicate))

        blocked = copy.deepcopy(scene)
        blocked['objects'].append({
            'id': 'observatory-barricade', 'name': 'Barricade', 'kind': 'wall',
            'position': [0., 1.45, -5.], 'size': [2.8, 2.5, .35],
            'color': [110, 102, 91],
            'interaction': {'action': 'none', 'trigger': 'none', 'radius': 0.}})
        blocked = validate_scene(blocked)
        cross = parse_commands('cross walkway', _adapted(blocked))[0]
        approach = _roots([0., 1.15, -2.25])
        with self.assertRaisesRegex(ValueError, 'blocked|disconnected|No safe walking route|no clear walkable approach'):
            plan_command(cross, _adapted(blocked), ('actor_1',), 'actor_1',
                         approach, None)


if __name__ == '__main__':
    unittest.main()
