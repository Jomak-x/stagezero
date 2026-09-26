"""Procedural asset safety, geometry, rendering, and playback checks."""

import copy
import json
import math
import struct
import unittest

import numpy as np

from asset_geometry import compile_asset, expanded_count, mesh_to_glb, validate_assets
from object_scene import ObjectSceneLayer
from scene_objects import evaluate_objects, validate_objects


def asset():
    return {'id': 'building-a', 'name': 'Brick building', 'parts': [
        {'shape': 'box', 'position': [0, 0, 0], 'size': [.8, .8, .6], 'color': [170, 77, 59]},
        {'shape': 'box', 'position': [-.3, -.2, -.31], 'size': [.08, .08, .02],
         'color': [255, 210, 122], 'repeat': {'count': [4, 3, 1], 'step': [.2, .2, 0]}},
    ]}


def custom(action='none'):
    trigger = {'none': 'none', 'open': 'proximity', 'pick_up': 'touch'}[action]
    radius = 0 if trigger == 'none' else (.05 if trigger == 'touch' else .5)
    return {'id': 'city-1', 'name': 'Building', 'kind': 'custom', 'asset': 'building-a',
            'position': [12, 5, -25], 'size': [8, 10, 6], 'color': [255, 255, 255],
            'yaw': 90, 'interaction': {'action': action, 'trigger': trigger, 'radius': radius}}


class GeometryTests(unittest.TestCase):
    def test_validates_repeat_bounds_and_detaches_input(self):
        source = asset()
        result = validate_assets([source])
        self.assertEqual(expanded_count(result[0]), 13)
        source['parts'][0]['color'][0] = 0
        self.assertEqual(result[0]['parts'][0]['color'][0], 170)
        for change in (
            lambda a: a['parts'][1]['repeat']['count'].__setitem__(0, 33),
            lambda a: a['parts'][1]['repeat']['step'].__setitem__(0, .3),
            lambda a: a['parts'][0]['position'].__setitem__(0, math.nan),
            lambda a: a['parts'][0].update(code='exec()'),
            lambda a: a['parts'][0].update(shape='file'),
            lambda a: a['parts'][0]['size'].__setitem__(0, 0),
        ):
            candidate = asset(); change(candidate)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_assets([candidate])
        over = asset()
        over['parts'][1]['repeat'] = {'count': [9, 9, 7], 'step': [0, 0, 0]}
        with self.assertRaisesRegex(ValueError, 'expanded'):
            validate_assets([over])
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            validate_assets([asset(), asset()])

    def test_compiles_one_mesh_with_repeated_window_colors_and_valid_glb(self):
        vertices, faces, colors = compile_asset(asset())
        self.assertEqual(vertices.shape[1], 3)
        self.assertEqual(faces.shape[1], 3)
        self.assertEqual(colors.shape, vertices.shape)
        self.assertEqual({tuple(c) for c in colors}, {(170, 77, 59), (255, 210, 122)})
        self.assertTrue(np.all(vertices >= -.500001) and np.all(vertices <= .500001))
        self.assertGreaterEqual(len(vertices), 13 * 8)
        glb = mesh_to_glb(vertices, faces, colors)
        magic, version, length = struct.unpack_from('<4sII', glb)
        self.assertEqual((magic, version, length), (b'glTF', 2, len(glb)))
        json_length = struct.unpack_from('<I', glb, 12)[0]
        document = json.loads(glb[20:20+json_length])
        self.assertEqual(document['meshes'][0]['primitives'][0]['attributes']['COLOR_0'], 2)
        self.assertTrue(document['accessors'][2]['normalized'])
        # COLOR_0 is linear. The first authored sRGB brick color (170, 77,
        # 59) must be encoded as its linear byte equivalent, not copied raw.
        color_offset = 20 + json_length + 8 + document['bufferViews'][2]['byteOffset']
        self.assertEqual(tuple(glb[color_offset:color_offset + 3]), (103, 19, 11))

    def test_rotated_part_still_fits_unit_bounds(self):
        source = {'id': 'angled', 'name': 'Angled beam', 'parts': [
            {'shape': 'box', 'position': [0, 0, 0], 'size': [.5, .1, .1], 'rotation': [0, 0, 45], 'color': [1, 2, 3]}]}
        vertices, _, _ = compile_asset(source)
        self.assertLessEqual(np.max(np.abs(vertices)), .5)
        source['parts'][0]['position'][0] = .4
        with self.assertRaisesRegex(ValueError, 'bounds'):
            validate_assets([source])


class FakeHandle:
    def __init__(self, glb_data, wxyz):
        self.glb_data, self.wxyz = glb_data, wxyz
        self.position = None
        self.removed = False

    def remove(self):
        self.removed = True


class FakeScene:
    def __init__(self):
        self.calls = []

    def add_glb(self, name, glb_data, *, wxyz):
        handle = FakeHandle(glb_data, wxyz)
        self.calls.append((name, handle))
        return handle


class RenderingTests(unittest.TestCase):
    def test_custom_objects_share_compilation_and_asset_edits_rebuild(self):
        server = type('Server', (), {'scene': FakeScene()})()
        layer = ObjectSceneLayer(server)
        layer._atmosphere = type('Atmosphere', (), {'update': lambda self, *args: None})()
        source = asset()
        first = custom()
        second = copy.deepcopy(first)
        second.update(id='city-2', position=[20, 5, -25])
        states = [{'id': obj['id'], 'position': obj['position'], 'color': obj['color'], 'active': False}
                  for obj in (first, second)]
        layer.update([first, second], {'objects': states, 'assets': [source]})
        self.assertEqual(len(server.scene.calls), 2)
        self.assertEqual(len(layer._mesh_cache), 1)
        self.assertIs(server.scene.calls[0][1].glb_data, server.scene.calls[1][1].glb_data)
        self.assertAlmostEqual(server.scene.calls[0][1].wxyz[0], math.sqrt(.5))
        self.assertEqual(server.scene.calls[0][1].position, tuple(first['position']))
        previous = [handle for _, handle in server.scene.calls]
        source['parts'][0]['color'] = [10, 20, 30]
        layer.update([first, second], {'objects': states, 'assets': [source]})
        self.assertTrue(all(handle.removed for handle in previous))
        self.assertEqual(len(server.scene.calls), 4)
        with self.assertRaisesRegex(ValueError, 'missing asset'):
            missing_layer = ObjectSceneLayer(server)
            missing_layer._atmosphere = layer._atmosphere
            missing_layer.update([first], {'objects': states[:1], 'assets': []})

    def test_custom_open_and_pickup_use_recorded_motion(self):
        door = custom('open')
        door['position'] = [2, 1, 0]
        door['size'] = [1, 2, 1]
        picked = custom('pick_up')
        picked['id'] = 'picked'
        picked['position'] = [0, 1, 0]
        picked['size'] = [1, 1, 1]
        validate_objects([door, picked])
        motion = [[[0, 0, 0], [2, 2, 0]], [[2, 0, 0], [0, 1, 0]], [[3, 0, 0], [5, 2, 0]]]
        states = evaluate_objects([door, picked], motion, 2, hand_indices=[1])
        self.assertEqual(states[0]['position'], [2., 3., 0.])
        self.assertEqual(states[1]['position'], [5., 2., 0.])
        self.assertEqual([s['active'] for s in states], [True, True])


if __name__ == '__main__':
    unittest.main()
