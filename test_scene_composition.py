"""Offline end-to-end checks for versioned scene generation and persistence."""

import copy
import json
import tempfile
import threading
import unittest

import numpy as np
import requests

from object_directing import ObjectDirectorSession
from scene_composition import PRESETS, generate_recipe, make_preset, validate_scene
from scene_effects import MAX_EFFECTS, make_effect
from scene_generation import LocalSceneGenerator, SceneGenerator
from scene_objects import MAX_OBJECTS, make_object
from takes import encode_project
from test_live_motion import ControlledBackend, wait_until


class SceneCompositionTests(unittest.TestCase):
    def test_every_preset_has_expected_content_palette_and_layout(self):
        expected = {
            'Neon research lab': (9, ['portal', 'sparks'], 'neon', [84, 225, 247]),
            'Enchanted grove': (14, ['fireflies', 'portal'], 'moonlight', [52, 110, 93]),
            'Cozy living room': (8, [], 'warm', [85, 135, 140]),
            'Industrial yard': (9, ['sparks', 'smoke', 'rain'], 'sunset', [179, 117, 59]),
            'Winter plaza': (8, ['snow'], 'moonlight', [185, 206, 220]),
        }
        self.assertEqual(set(PRESETS), set(expected))
        for name in PRESETS:
            with self.subTest(name=name):
                count, effect_kinds, lighting, accent = expected[name]
                scene = make_preset(name, seed=17)
                self.assertEqual(scene, validate_scene(scene))
                self.assertEqual(scene['version'], 2)
                self.assertEqual(len(scene['objects']), count)
                self.assertEqual([fx['kind'] for fx in scene['effects']], effect_kinds)
                self.assertEqual(scene['lighting'], lighting)
                self.assertIn(accent, [obj['color'] for obj in scene['objects']])
                self.assertEqual(len({obj['id'] for obj in scene['objects']}), count)
                self.assertEqual([fx['seed'] for fx in scene['effects']], list(range(17, 17 + len(effect_kinds))))
                self.assertEqual(make_preset(name, seed=17), scene)

    def test_offline_recipe_counts_colors_and_effect_only_scene(self):
        scene = generate_recipe('Three red chairs, two blue lamps, smoke and portal', seed=9)
        self.assertEqual([obj['kind'] for obj in scene['objects']], ['chair'] * 3 + ['lamp'] * 2)
        self.assertEqual([obj['color'] for obj in scene['objects'][:3]], [[218, 70, 72]] * 3)
        self.assertEqual([obj['color'] for obj in scene['objects'][3:]], [[58, 120, 210]] * 2)
        self.assertEqual([fx['kind'] for fx in scene['effects']], ['smoke', 'portal'])
        self.assertEqual(scene['lighting'], 'neutral')
        effects_only = generate_recipe('Fireflies and a portal', seed=5)
        self.assertEqual(effects_only['objects'], [])
        self.assertEqual([fx['kind'] for fx in effects_only['effects']], ['fireflies', 'portal'])
        self.assertEqual(effects_only, validate_scene(effects_only))

    def test_scene_document_limits_and_untrusted_fields(self):
        good = make_preset('Winter plaza')
        invalid = (
            dict(good, version=1),
            dict(good, name='  '),
            dict(good, lighting='ultraviolet'),
            dict(good, objects=[make_object('chair', i) for i in range(MAX_OBJECTS + 1)]),
            dict(good, effects=[make_effect('rain', i) for i in range(MAX_EFFECTS + 1)]),
            dict(good, effects=[dict(make_effect('snow', 0), code='run()')]),
            dict(good, objects=[dict(make_object('lamp', 0), url='file:///private')]),
            dict(good, extra='ignored'),
        )
        for item in invalid:
            with self.subTest(item=str(item)[:70]), self.assertRaises(ValueError):
                validate_scene(item)
        for prompt in ('13 chairs', '41 chairs', 'a mysterious staircase'):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                generate_recipe(prompt)
        detached = validate_scene(good)
        good['effects'][0]['color'][0] = 0
        self.assertNotEqual(detached['effects'][0]['color'], good['effects'][0]['color'])


class _FakeResponse:
    def __init__(self, body=b'', status_code=200):
        self.body = body
        self.status_code = status_code
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def iter_content(self, _chunk_size):
        yield self.body


class _FakeTransport:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.response


class SceneGenerationTests(unittest.TestCase):
    def test_gateway_repairs_schema_once_and_propagates_transport_failure(self):
        from unittest.mock import Mock
        expected = make_preset('Winter plaza')
        gateway = Mock()
        gateway.request_json.side_effect = [{'bad': 'scene'}, expected]
        self.assertEqual(SceneGenerator(gateway).generate('winter plaza'), expected)
        self.assertEqual(gateway.request_json.call_count, 2)
        self.assertIn('failed validation', gateway.request_json.call_args.args[0])
        gateway.reset_mock()
        gateway.request_json.side_effect = [{'bad': 'scene'}, {'still': 'bad'}]
        with self.assertRaises(ValueError):
            SceneGenerator(gateway).generate('winter plaza')
        self.assertEqual(gateway.request_json.call_count, 2)
        gateway.reset_mock()
        gateway.request_json.side_effect = ValueError('HTTP failure')
        with self.assertRaises(ValueError):
            SceneGenerator(gateway).generate('winter plaza')
        self.assertEqual(gateway.request_json.call_count, 1)

    def test_gateway_returns_strict_full_scene_and_prompt(self):
        expected = make_preset('Neon research lab', seed=2)

        class Gateway:
            def __init__(self, response):
                self.response = response
                self.calls = []

            def request_json(self, system, prompt, max_tokens):
                self.calls.append((system, prompt, max_tokens))
                return self.response

        gateway = Gateway(expected)
        scene = SceneGenerator(gateway).generate('  neon laboratory  ')
        self.assertEqual(scene, expected)
        self.assertEqual(gateway.calls[0][1:], ('neon laboratory', 8000))
        self.assertIn('EFFECT CATALOG', gateway.calls[0][0])
        gateway.response = dict(expected, effects=[dict(make_effect('rain', 0), code='unsafe')])
        with self.assertRaises(ValueError):
            SceneGenerator(gateway).generate('rain')

    def test_local_ollama_success_and_failure_use_fake_transport(self):
        expected = make_preset('Enchanted grove', seed=3)
        envelope = {'message': {'content': json.dumps(expected)}}
        response = _FakeResponse(json.dumps(envelope).encode())
        transport = _FakeTransport(response)
        generator = LocalSceneGenerator(model='test-local-model', transport=transport)
        self.assertEqual(generator.generate('  enchanted grove  '), expected)
        self.assertTrue(response.closed)
        args, kwargs = transport.calls[0]
        self.assertEqual(args, ('http://127.0.0.1:11434/api/chat',))
        self.assertEqual(kwargs['json']['model'], 'test-local-model')
        self.assertEqual(kwargs['json']['messages'][-1]['content'], 'enchanted grove')
        self.assertEqual(kwargs['timeout'], (3, 90))
        self.assertFalse(kwargs['allow_redirects'])
        for bad_response, error in (
            (_FakeResponse(status_code=503), None),
            (_FakeResponse(b'not json'), None),
            (_FakeResponse(json.dumps({'message': {'content': '{bad'}}).encode()), None),
            (_FakeResponse(json.dumps({'message': {'content': json.dumps(dict(expected, lighting='bad'))}}).encode()), None),
            (None, requests.Timeout('offline')),
        ):
            with self.subTest(error=error, body=getattr(bad_response, 'body', b'')[:30]):
                fake = _FakeTransport(bad_response, error)
                with self.assertRaises(ValueError):
                    LocalSceneGenerator(model='test-local-model', transport=fake).generate('forest')
                if bad_response is not None:
                    self.assertTrue(bad_response.closed)


class ScenePersistenceTests(unittest.TestCase):
    def setUp(self):
        backend = ControlledBackend()
        backend.release.set()
        self.session = ObjectDirectorSession(
            backend,
            np.zeros((20, 34, 3), dtype=np.float32),
            np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1)),
        )
        self.session.set_mode('Live ARDY')

    def _generate_take(self):
        self.session.submit('walk')
        wait_until(lambda: not self.session.busy)
        self.session.pause()

    def test_project_save_load_preserves_effects_lighting_and_name(self):
        original = make_preset('Industrial yard', seed=7)
        self.session.set_scene(original)
        self._generate_take()
        with tempfile.TemporaryDirectory() as folder:
            _, payload = self.session.save_project(folder, 'complete-scene')
        self.session.set_scene(make_preset('Cozy living room'))
        self.session.load_project(payload)
        self.assertEqual(self.session.scene_document(), original)

    def test_v1_object_document_import_and_invalid_load_preserves_live_scene(self):
        objects = [make_object('lamp', 0)]
        self.session.load_scene_document({'version': 1, 'objects': objects})
        self.assertEqual(self.session.scene_document(), {'version': 2, 'name': 'Imported objects',
                                                         'objects': objects, 'effects': [], 'lighting': 'neutral'})
        current = make_preset('Winter plaza')
        self.session.set_scene(current)
        self._generate_take()
        take = self.session.takes[self.session.active_take]
        revision = self.session.project_revision
        bad_scene = dict(self.session.scene, effects=[dict(make_effect('snow', 0), script='unsafe')])
        data = encode_project(self.session.takes, self.session.active_take, self.session.frame, bad_scene)
        with self.assertRaises(ValueError):
            self.session.load_project(data)
        self.assertIs(self.session.takes[self.session.active_take], take)
        self.assertEqual(self.session.project_revision, revision)
        self.assertEqual(self.session.scene_document(), current)

    def test_stale_async_result_cannot_replace_newer_scene(self):
        older = make_preset('Neon research lab')
        newer = make_preset('Winter plaza')
        started, release = threading.Event(), threading.Event()

        class SlowGenerator:
            def generate(self, _prompt):
                started.set()
                if not release.wait(2):
                    raise TimeoutError('test did not release scene generator')
                return copy.deepcopy(older)

        errors = []

        def work():
            try:
                self.session.generate_scene('lab', generator=SlowGenerator())
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=work)
        thread.start()
        self.assertTrue(started.wait(1))
        self.session.set_scene(newer)
        release.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)
        self.assertIn('Project changed', str(errors[0]))
        self.assertEqual(self.session.scene_document(), newer)


if __name__ == '__main__':
    unittest.main()
