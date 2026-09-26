"""Behavior checks for generated scene object data and pure playback."""

import copy
import math
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from object_scene import ObjectSceneLayer
from scene_objects import KINDS, MAX_OBJECTS, evaluate_objects, make_object, validate_objects


class FakeHandle:
    def __init__(self, shape, color, **geometry):
        self.shape = shape
        self.color = color
        self.geometry = geometry
        self.position = (0, 0, 0)
        self.removed = False

    def remove(self):
        self.removed = True


class FakeScene:
    def __init__(self):
        self.items = {}

    def add_box(self, name, dimensions, color, **kwargs):
        return self._add(name, 'box', color, dimensions=dimensions)

    def add_icosphere(self, name, radius, color, **kwargs):
        return self._add(name, 'sphere', color, radius=radius)

    def add_mesh_simple(self, name, vertices, faces, color, **kwargs):
        return self._add(name, 'mesh', color, vertices=vertices, faces=faces)

    def _add(self, name, shape, color, **geometry):
        item = FakeHandle(shape, color, **geometry)
        self.items[name] = item
        return item


class FakeServer:
    def __init__(self):
        self.scene = FakeScene()


class SceneObjectValidationTests(unittest.TestCase):
    def test_templates_are_canonical_and_rest_on_the_ground(self):
        objects = [make_object(kind, i) for i, kind in enumerate(KINDS)]
        self.assertEqual(validate_objects(objects), objects)
        self.assertEqual(len(objects), 16)
        self.assertEqual(len({item["id"] for item in objects}), 16)
        for item in objects:
            self.assertEqual(item["position"][1], item["size"][1] / 2)

    def test_rejects_untrusted_fields_and_invalid_physical_values(self):
        good = make_object("ball", 0)
        mutations = (
            lambda o: o.update(script="move_character()"),
            lambda o: o["interaction"].update(code="eval()"),
            lambda o: o["interaction"].update(action="open"),
            lambda o: o["interaction"].update(trigger="proximity"),
            lambda o: o["interaction"].update(radius=1.0),
            lambda o: o.update(position=[0, math.nan, 0]),
            lambda o: o.update(position=[101, 0, 0]),
            lambda o: o.update(size=[-1, 1, 1]),
            lambda o: o.update(size=[0.3, 0.4, 0.3]),
            lambda o: o.update(color=[256, 2, 3]),
            lambda o: o.update(id="not an id"),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                candidate = copy.deepcopy(good)
                mutate(candidate)
                with self.assertRaises(ValueError):
                    validate_objects([candidate])
        with self.assertRaises(ValueError):
            validate_objects([good, copy.deepcopy(good)])
        with self.assertRaises(ValueError):
            validate_objects([make_object("lamp", i) for i in range(MAX_OBJECTS + 1)])

    def test_validation_returns_detached_canonical_values(self):
        original = make_object("lamp", 2)
        clean = validate_objects([original])[0]
        original["position"][0] = 50
        original["interaction"]["radius"] = 3
        self.assertEqual(clean["position"][0], 0)
        self.assertEqual(clean["interaction"]["radius"], 0.7)

    def test_static_props_and_large_environment_sizes(self):
        self.assertEqual(MAX_OBJECTS, 64)
        objects = [make_object('wall', i) for i in range(MAX_OBJECTS)]
        objects[0]['size'] = [12, 3, .2]
        self.assertEqual(len(validate_objects(objects)), MAX_OBJECTS)
        for kind in ('table', 'sofa', 'crate', 'barrel', 'pillar', 'wall', 'arch',
                     'plant', 'tree', 'rock', 'platform'):
            obj = make_object(kind, 0)
            self.assertEqual(obj['interaction'], {'action': 'none', 'trigger': 'none', 'radius': 0.0})
            for field, value in (('action', 'activate'), ('trigger', 'proximity'), ('radius', .5)):
                bad = copy.deepcopy(obj)
                bad['interaction'][field] = value
                with self.subTest(kind=kind, field=field), self.assertRaises(ValueError):
                    validate_objects([bad])
        self.assertEqual(make_object('console', 0)['interaction']['action'], 'activate')

    def test_existing_door_and_lamp_touch_variants_remain_valid(self):
        for kind in ('door', 'lamp'):
            obj = make_object(kind, 0)
            obj['interaction'].update(trigger='touch', radius=.05)
            self.assertEqual(validate_objects([obj])[0], obj)


class SceneObjectPlaybackTests(unittest.TestCase):
    def test_static_props_stay_inactive_and_console_activates_near_root(self):
        tree = make_object('tree', 0, [0, 1.6, 0])
        console = make_object('console', 0, [2, .575, 0])
        positions = [[[0, 0, 0]], [[2, 0, 0]]]
        first = evaluate_objects([tree, console], positions, 0)
        last = evaluate_objects([tree, console], positions, 1)
        self.assertFalse(first[0]['active'])
        self.assertEqual(first[0]['position'], tree['position'])
        self.assertEqual(first[0]['color'], tree['color'])
        self.assertFalse(first[1]['active'])
        self.assertTrue(last[1]['active'])
        self.assertEqual(last[1]['position'], console['position'])
        self.assertEqual(last[1]['color'], console['color'])
    def test_pick_up_requires_explicit_hand_contact_and_tracks_that_hand(self):
        ball = make_object("ball", 0)
        # Joint 0 brushes the ball but is not declared as a hand. Joint 1
        # touches the ball only on frame 2.
        positions = [
            [[0, 0.15, 0], [2, 0.15, 0]],
            [[0, 0.15, 0], [0.21, 0.15, 0]],
            [[0, 0.15, 0], [0.18, 0.15, 0]],
            [[0, 0.15, 0], [1, 1.2, 0]],
        ]
        with self.assertRaisesRegex(ValueError, "hand_indices"):
            evaluate_objects([ball], positions, 0)
        self.assertFalse(evaluate_objects([ball], positions, 1, hand_indices=[1])[0]["active"])
        acquired = evaluate_objects([ball], positions, 2, hand_indices=[1])[0]
        self.assertTrue(acquired["active"])
        self.assertEqual(acquired["position"], [0.18, 0.15, 0.0])
        self.assertEqual(evaluate_objects([ball], positions, 3, hand_indices=[1])[0]["position"], [1, 1.2, 0])

    def test_scrubbing_and_replaying_give_the_same_states(self):
        door = make_object("door", 0, [2, 1, 0])
        lamp = make_object("lamp", 0, [4, 0.7, 0])
        chair = make_object("chair", 0, [6, 0.45, 0])
        positions = [
            [[0, 1, 0]],
            [[2, 1, 0]],
            [[4, 1, 0]],
            [[6, 1, 0]],
        ]
        objects = [door, lamp, chair]
        baseline = [evaluate_objects(objects, positions, i) for i in range(len(positions))]
        self.assertEqual([state["active"] for state in baseline[0]], [False, False, False])
        self.assertEqual([state["active"] for state in baseline[3]], [True, True, True])
        self.assertEqual(baseline[1][0]["position"][1], 3.0)
        self.assertNotEqual(baseline[2][1]["color"], lamp["color"])
        self.assertEqual(baseline[3][2]["position"], chair["position"])
        for i in (3, 1, 0, 2, 3):
            self.assertEqual(evaluate_objects(objects, positions, i), baseline[i])

    def test_proximity_uses_root_ground_position_and_ball_uses_sphere(self):
        chair = make_object("chair", 0, [1, 0.45, 0])
        ball = make_object("ball", 0)
        # The upper joint touches the chair, but the root remains distant.
        # The hand is inside the ball's bounding cube plus tolerance while
        # remaining outside the sphere plus tolerance.
        positions = [[[4, 0, 0], [1, 0.45, 0], [0.17, 0.32, 0]]]
        states = evaluate_objects([chair, ball], positions, 0, hand_indices=[2])
        self.assertEqual([state["active"] for state in states], [False, False])

    def test_malformed_motion_and_frame_are_rejected(self):
        lamp = make_object("lamp", 0)
        for positions, frame in (([], 0), ([[[0, 0, 0]]], 1), ([[[math.inf, 0, 0]]], 0), ([[[0, 0]]], 0)):
            with self.subTest(positions=positions, frame=frame), self.assertRaises(ValueError):
                evaluate_objects([lamp], positions, frame)
        with self.assertRaises(ValueError):
            evaluate_objects([make_object("ball", 0)], [[[0, 0, 0]]], 0, hand_indices=[2])


class SceneObjectRenderingTests(unittest.TestCase):
    def test_every_model_stays_inside_declared_size(self):
        for kind in KINDS:
            for size in ([1.2, 1.2, 1.2], [12, .05, .05], [.05, 12, .05]):
                if kind == 'ball' and len(set(size)) != 1:
                    continue
                obj = make_object(kind, 0)
                obj['size'] = size
                parts = ObjectSceneLayer(FakeServer())._build(obj)
                self.assertTrue(parts)
                for handle, offset, _ in parts:
                    if handle.shape == 'box':
                        bounds = np.asarray(handle.geometry['dimensions']) / 2
                    elif handle.shape == 'sphere':
                        bounds = np.repeat(handle.geometry['radius'], 3)
                    else:
                        bounds = np.max(np.abs(handle.geometry['vertices']), axis=0)
                    with self.subTest(kind=kind, size=size, shape=handle.shape):
                        self.assertTrue(np.all(np.abs(offset) + bounds <= np.asarray(size) / 2 + 1e-6))

    def test_material_details_and_console_screen_follow_state(self):
        class FakeAtmosphere:
            def __init__(self, server):
                self.calls = []

            def update(self, effects, seconds, lighting):
                self.calls.append((effects, seconds, lighting))

        module = types.SimpleNamespace(SceneAtmosphereLayer=FakeAtmosphere)
        with patch.dict(sys.modules, {'scene_atmosphere': module}):
            server = FakeServer()
            layer = ObjectSceneLayer(server)
            console = make_object('console', 0)
            inactive = {'id': console['id'], 'position': console['position'],
                        'color': console['color'], 'active': False}
            layer.update([console], [inactive])
            screen = server.scene.items['/objects/console-0/screen']
            key = server.scene.items['/objects/console-0/key0']
            inactive_key_color = key.color
            self.assertEqual(screen.color, (34, 88, 110))
            layer.update([console], {'objects': [dict(inactive, active=True)],
                                     'effects': [{'kind': 'fog'}], 'seconds': 1.5,
                                     'lighting': 'sunset'})
            self.assertEqual(screen.color, (87, 242, 222))
            self.assertEqual(key.color, inactive_key_color)
            self.assertEqual(layer._atmosphere.calls[-1][1:], (1.5, 'sunset'))
            layer.update([console], [inactive])
            self.assertEqual(layer._atmosphere.calls[-1], ([], 0.0, 'neutral'))


if __name__ == "__main__":
    unittest.main()
