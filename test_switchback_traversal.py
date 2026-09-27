"""CPU proof of a distinct supported route and broad-tread stair planning."""
from __future__ import annotations

import copy
import math
import unittest

import numpy as np

from core_scene_reactions import evaluated_scene, object_states
from core_spatial_commands import parse_commands, plan_command
from core_terrain_navigation import ACTOR_RADIUS_M
from realtime_clip import CanonicalClip
from scene_composition import validate_scene
from scene_interaction_geometry import SceneInteractionGeometry
from studio_interaction_scene import adapt_studio_scene
from traversal_kit import traversable_temple_scene
from switchback_traversal import (EXIT_TREAD_RUN_M, START_ROOT_XYZ, START_YAW_RAD,
                                  industrial_switchback_scene, switchback_route_metadata)


def _adapted(scene, evaluated=None, *, continuing=False):
    result = adapt_studio_scene(scene if evaluated is None else evaluated)
    result['original_scene'] = scene
    if continuing:
        result['terrain_active'] = True
    return result


def _clip(root_xyz, frames=1):
    positions = np.zeros((1, frames, 27, 3), np.float32)
    positions[0, :, :, :] = root_xyz
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32),
                                (1, frames, 27, 3, 3)).copy()
    return CanonicalClip(positions, rotations, 20, ('actor_1',), 'ardy_core', {},
                         np.zeros((1, frames, 330), np.float32))


class IndustrialSwitchbackTests(unittest.TestCase):
    def test_scene_has_two_real_stair_flights_and_an_open_gap(self):
        scene = validate_scene(industrial_switchback_scene())
        geometry = SceneInteractionGeometry.from_scene(scene)
        routes = {flight['object_id']: flight for flight in geometry.stair_routes}
        self.assertEqual(set(routes), {'yard-approach-stairs', 'yard-exit-stairs'})
        self.assertEqual(len(routes['yard-exit-stairs']['steps']), 5)
        self.assertAlmostEqual(abs(routes['yard-exit-stairs']['steps'][1][2]-
                                   routes['yard-exit-stairs']['steps'][0][2]),
                               EXIT_TREAD_RUN_M)
        for x, z, height in ((0., 2., 0.), (0., -2.25, .2),
                             (1., -4., .2), (6., -5., .2),
                             (10.5, -8.375, .16), (10.5, -11.375, 0.),
                             (10.5, -12., 0.)):
            with self.subTest(x=x, z=z):
                self.assertAlmostEqual(geometry.support_height(x, z, height,
                    max_step_up=.3, max_drop=.5), height, places=5)
        self.assertIsNone(geometry.support_height(6., -2., 0.,
                                                  max_step_up=.3, max_drop=.5))
        self.assertEqual(switchback_route_metadata()['start_root_xyz'],
                         list(START_ROOT_XYZ))

    def test_repeated_submissions_turn_detour_door_and_descent(self):
        scene = validate_scene(industrial_switchback_scene())
        metadata = switchback_route_metadata()
        self.assertEqual(len(metadata['ordered_commands']), 3)
        committed = None
        routes = []
        evaluated = scene
        for submission_index, text in enumerate(metadata['ordered_commands']):
            actions = parse_commands(text, _adapted(scene, evaluated,
                                                    continuing=submission_index > 0))
            for action in actions:
                if action['verb'] == 'go_through':
                    held = _clip(committed.positions[0, -1, 0], frames=20)
                    gate = next(state for state in object_states(
                        scene, held, enabled=True, terrain=True)
                        if state['id'] == 'yard-workshop-door')
                    self.assertAlmostEqual(gate['opening_fraction'], 1.)
                    evaluated = evaluated_scene(scene, held, enabled=True, terrain=True)
                stages, route = plan_command(
                    action, _adapted(scene, evaluated, continuing=submission_index > 0),
                    ('actor_1',), 'actor_1', committed,
                    {'actor_1': {'position_xz': [START_ROOT_XYZ[0], START_ROOT_XYZ[2]],
                                 'yaw': START_YAW_RAD}})
                self.assertTrue(stages)
                self.assertEqual(route['terrain_navigation_version'], 1)
                last = route['waypoints'][-1]
                committed = _clip((last['position_xz'][0], last['support_y']+.95,
                                   last['position_xz'][1]))
                routes.append(route)
        self.assertEqual([route['verb'] for route in routes],
                         ['ascend', 'cross', 'open', 'go_through', 'descend'])
        cross = routes[1]
        self.assertEqual(cross['assumptions']['actor_radius_m'], ACTOR_RADIUS_M)
        self.assertEqual(cross['schedule']['initial_turn_frames'], 40)
        self.assertGreaterEqual(len(cross['support_xyz']), 4)
        # The cargo crate blocks the direct chord; the supported plan bends
        # around it on the broad loading platform before entering the gantry.
        first, bend, middle = (np.asarray(p) for p in cross['support_xyz'][:3])
        a, b = (bend-first)[[0, 2]], (middle-first)[[0, 2]]
        self.assertGreater(abs(a[0]*b[1]-a[1]*b[0]), .5)
        crate = next(obj for obj in scene['objects'] if obj['id'] == 'yard-forklift-crate')
        lower = np.asarray(crate['position'])[[0, 2]] - np.asarray(crate['size'])[[0, 2]]/2
        upper = np.asarray(crate['position'])[[0, 2]] + np.asarray(crate['size'])[[0, 2]]/2
        path = np.asarray(cross['support_xyz'])[:, [0, 2]]
        sample = np.concatenate([start[None, :] +
            np.linspace(0., 1., max(2, math.ceil(np.linalg.norm(end-start)/.01)+1))[:, None]
            *(end-start)[None, :] for start, end in zip(path[:-1], path[1:])])
        outside = np.maximum(np.maximum(lower-sample, sample-upper), 0.)
        self.assertGreaterEqual(float(np.min(np.linalg.norm(outside, axis=1))),
                                ACTOR_RADIUS_M)
        self.assertAlmostEqual(routes[-1]['waypoints'][-1]['support_y'], 0., places=5)
        self.assertEqual(routes[-1]['target_id'], 'yard-exit-stairs')

    def test_missing_ambiguous_and_blocked_target_fail(self):
        scene = validate_scene(industrial_switchback_scene())
        missing = copy.deepcopy(scene)
        missing['objects'] = [obj for obj in missing['objects'] if obj['id'] != 'yard-gantry']
        with self.assertRaisesRegex(ValueError, 'Unknown terrain target'):
            parse_commands('cross bridge', _adapted(missing))

        duplicate = copy.deepcopy(scene)
        gantry = next(obj for obj in duplicate['objects'] if obj['id'] == 'yard-gantry')
        other = copy.deepcopy(gantry)
        other['id'] = 'yard-second-gantry'
        other['name'] = 'Second service bridge'
        other['position'][2] += 5.
        duplicate['objects'].append(other)
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            parse_commands('cross bridge', _adapted(duplicate))

        blocked = copy.deepcopy(scene)
        blocked['objects'].append({
            'id': 'yard-solid-barricade', 'name': 'Gantry barricade',
            'kind': 'wall', 'position': [6., 1.45, -5.],
            'size': [2.4, 2.5, 2.2], 'color': [97, 96, 88],
            'interaction': {'action': 'none', 'trigger': 'none', 'radius': 0.}})
        blocked = validate_scene(blocked)
        action = parse_commands('cross service gantry bridge', _adapted(blocked))[0]
        with self.assertRaisesRegex(ValueError, 'no clear walkable approach|blocked|disconnected|No safe walking route'):
            plan_command(action, _adapted(blocked), ('actor_1',), 'actor_1',
                         _clip((0., 1.15, -2.25)), None)

    def test_descriptive_temple_stairs_ignore_named_entry_landing(self):
        scene = validate_scene(traversable_temple_scene())
        actions = parse_commands('walk up temple stairs then cross temple bridge',
                                 _adapted(scene))
        self.assertEqual(actions[0]['target_alias'], 'stairs')
        self.assertEqual([action['verb'] for action in actions], ['ascend', 'cross'])
        _, ascent = plan_command(actions[0], _adapted(scene), ('actor_1',),
                                 'actor_1', None,
                                 {'actor_1': {'position_xz': [0., .85], 'yaw': math.pi}})
        self.assertEqual(ascent['target_id'], 'temple-stairs')
        last = ascent['waypoints'][-1]
        root = _clip((last['position_xz'][0], last['support_y']+.95,
                      last['position_xz'][1]))
        _, crossing = plan_command(actions[1], _adapted(scene), ('actor_1',),
                                   'actor_1', root, None)
        self.assertEqual(crossing['target_id'], 'temple-bridge')

        # Exact naming never converts a landing into a flight.
        exact = parse_commands('walk up Level stair entry', _adapted(scene))[0]
        self.assertEqual(exact['target_id'], 'temple-entry')
        with self.assertRaisesRegex(ValueError, 'no unambiguous rendered stair flight'):
            plan_command(exact, _adapted(scene), ('actor_1',), 'actor_1', None,
                         {'actor_1': {'position_xz': [0., .85], 'yaw': math.pi}})

    def test_two_actual_stair_flights_still_require_disambiguation(self):
        scene = copy.deepcopy(traversable_temple_scene())
        flight = next(obj for obj in scene['objects'] if obj['id'] == 'temple-stairs')
        flight['position'][0] = -.5
        twin = copy.deepcopy(flight)
        twin['id'] = 'temple-twin-stairs'
        twin['name'] = 'Twin temple stairs'
        twin['position'][0] = .5
        scene['objects'].append(twin)
        scene = validate_scene(scene)
        action = parse_commands('walk up temple stairs', _adapted(scene))[0]
        self.assertEqual(action['target_alias'], 'stairs')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            plan_command(action, _adapted(scene), ('actor_1',), 'actor_1', None,
                         {'actor_1': {'position_xz': [0., .85], 'yaw': math.pi}})


if __name__ == '__main__':
    unittest.main()
