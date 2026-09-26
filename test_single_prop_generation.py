"""One-prompt custom prop generation and additive scene authoring."""

import copy
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from object_directing import ObjectDirectorSession
from scene_asset_library import AssetLibrary
from scene_composition import validate_scene
from scene_effects import make_effect
from scene_environments import make_room
from scene_objects import make_object
from single_prop_generation import SinglePropGenerator
from test_live_motion import ControlledBackend


def example_asset():
    return {'id': 'telescope', 'name': 'Brass telescope', 'parts': [
        {'shape': 'cylinder', 'position': [0, .1, 0], 'size': [.18, .8, .18], 'color': [170, 125, 56]},
        {'shape': 'box', 'position': [0, -.35, 0], 'size': [.5, .15, .5], 'color': [94, 62, 38]},
        {'shape': 'sphere', 'position': [0, .42, 0], 'size': [.2, .12, .2], 'color': [195, 151, 65]},
    ]}


def response():
    return {'assets': [example_asset()], 'placement': {
        'size': [1.2, 1.8, .8], 'position': [2, .9, 1.5], 'yaw': 25}}


class SinglePropTests(unittest.TestCase):
    def test_one_request_creates_one_reusable_custom_prop(self):
        with tempfile.TemporaryDirectory() as folder:
            gateway = Mock()
            gateway.request_json.return_value = response()
            result = SinglePropGenerator(gateway, AssetLibrary(folder)).generate('a brass telescope')
            self.assertEqual(gateway.request_json.call_count, 1)
            self.assertIn('exactly ONE', gateway.request_json.call_args.args[0])
            self.assertEqual(result['object']['kind'], 'custom')
            self.assertEqual(result['object']['asset'], result['asset']['id'])
            self.assertEqual(result['object']['size'], [1.2, 1.8, .8])
            self.assertEqual(len(AssetLibrary(folder).load_all()), 1)

    def test_invalid_shape_does_not_save_an_asset(self):
        with tempfile.TemporaryDirectory() as folder:
            gateway = Mock()
            bad = response()
            bad['assets'][0]['parts'][0]['shape'] = 'script'
            gateway.request_json.return_value = bad
            library = AssetLibrary(folder)
            with self.assertRaises(ValueError):
                SinglePropGenerator(gateway, library).generate('a telescope')
            self.assertEqual(gateway.request_json.call_count, 2)
            self.assertEqual(library.load_all(), [])


class AppendObjectTests(unittest.TestCase):
    def setUp(self):
        backend = ControlledBackend()
        backend.release.set()
        self.session = ObjectDirectorSession(
            backend, np.zeros((20, 34, 3), dtype=np.float32),
            np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1)))

    def test_custom_append_promotes_v2_and_preserves_scene_and_gate(self):
        existing = make_object('door', 0)
        self.session.set_objects([existing])
        before_gate = copy.deepcopy(self.session.scene['gate'])
        with tempfile.TemporaryDirectory() as folder:
            gateway = Mock()
            gateway.request_json.return_value = response()
            generator = SinglePropGenerator(gateway, AssetLibrary(folder))
            added = self.session.append_custom_object('a telescope', generator)
            self.assertEqual(self.session.scene_document()['version'], 3)
            self.assertEqual(self.session.scene['gate'], before_gate)
            self.assertEqual(self.session.scene['objects'][0], existing)
            self.assertEqual(len(self.session.scene['assets']), 1)
            self.assertEqual(added['asset'], self.session.scene['assets'][0]['id'])
            self.assertEqual(validate_scene(self.session.scene_document()), self.session.scene_document())
            second = self.session.append_custom_object('another telescope', generator)
            self.assertNotEqual(second['id'], added['id'])
            self.assertEqual(len(self.session.scene['assets']), 1)
            self.assertEqual(len(self.session.scene['objects']), 3)

    def test_offline_append_keeps_existing_objects(self):
        self.session.set_objects([make_object('door', 0)])
        appended = self.session.append_catalog_object('a door')
        self.assertEqual(len(self.session.scene['objects']), 2)
        self.assertEqual(len({obj['id'] for obj in self.session.scene['objects']}), 2)
        self.assertEqual(appended['kind'], 'door')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            self.session.append_catalog_object('a door and a lamp')
        self.assertEqual(len(self.session.scene['objects']), 2)

    def test_custom_append_preserves_existing_v3_camera_effects_and_assets(self):
        room = make_room()
        room['effects'] = [make_effect('fireflies', 0)]
        self.session.set_scene(room)
        before = self.session.scene_document()
        with tempfile.TemporaryDirectory() as folder:
            gateway = Mock()
            gateway.request_json.return_value = response()
            self.session.append_custom_object('a telescope', SinglePropGenerator(gateway, AssetLibrary(folder)))
        after = self.session.scene_document()
        self.assertEqual(after['camera'], before['camera'])
        self.assertEqual(after['effects'], before['effects'])
        self.assertEqual(after['lighting'], before['lighting'])
        self.assertEqual(after['objects'][:len(before['objects'])], before['objects'])
        self.assertEqual(after['assets'][:len(before['assets'])], before['assets'])

    def test_scene_edit_can_preserve_gate_while_replacement_resets_it(self):
        room = make_room()
        self.session.scene['gate']['enabled'] = True
        self.session.set_scene(room, reset_gate=False)
        self.assertTrue(self.session.scene['gate']['enabled'])
        edited = self.session.scene_document()
        edited['lighting'] = 'moonlight'
        self.session.set_scene(edited, reset_gate=False)
        self.assertTrue(self.session.scene['gate']['enabled'])
        self.assertEqual(self.session.scene_document()['lighting'], 'moonlight')
        self.session.set_scene(room)
        self.assertFalse(self.session.scene['gate']['enabled'])

    def test_stale_click_revision_rejects_before_gateway_request(self):
        click_revision = self.session.project_revision
        self.session.set_objects([make_object('lamp', 0)])
        generator = Mock()
        with self.assertRaisesRegex(ValueError, 'Project changed before'):
            self.session.append_custom_object('a telescope', generator, expected_revision=click_revision)
        generator.generate.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'Project changed before'):
            self.session.generate_scene('a city', generator=generator, expected_revision=click_revision)
        generator.generate.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'Project changed before'):
            self.session.append_catalog_object('a door', expected_revision=click_revision)
        self.assertEqual([o['kind'] for o in self.session.scene_document()['objects']], ['lamp'])

    def test_revision_change_and_invalid_asset_leave_scene_untouched(self):
        initial = self.session.scene_document()
        revision = self.session.project_revision
        class Invalid:
            def generate(self, _):
                return {'asset': {'id': 'bad', 'name': 'Bad', 'parts': []},
                        'object': {'id': 'bad', 'kind': 'custom', 'asset': 'bad'}}
        with self.assertRaises(ValueError):
            self.session.append_custom_object('bad prop', Invalid())
        self.assertEqual(self.session.scene_document(), initial)
        self.assertEqual(self.session.project_revision, revision)
        class Concurrent:
            def generate(inner, _):
                self.session.set_objects([make_object('chair', 0)])
                return {'asset': example_asset(), 'object': {
                    'id': 'generated-prop', 'name': 'Brass telescope', 'kind': 'custom',
                    'asset': 'telescope', 'size': [1, 1, 1], 'position': [2, .5, 1],
                    'color': [255, 255, 255], 'interaction': {'action': 'none', 'trigger': 'none', 'radius': 0}}}
        with self.assertRaisesRegex(ValueError, 'Project changed'):
            self.session.append_custom_object('telescope', Concurrent())
        self.assertEqual([o['kind'] for o in self.session.scene_document()['objects']], ['chair'])


if __name__ == '__main__':
    unittest.main()
