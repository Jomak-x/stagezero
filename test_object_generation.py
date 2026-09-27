"""Offline coverage for functional prop generation and project integration."""

import json
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
import requests

from object_directing import ObjectDirectorSession
from object_generation import GatewayGenerator, MAX_RESPONSE_BYTES, generate_local
from scene_objects import make_object
from takes import encode_project
from test_live_motion import ControlledBackend, wait_until


class FakeResponse:
    def __init__(self, content=b'', status_code=200):
        self.content = content
        self.status_code = status_code
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def iter_content(self, _chunk_size):
        yield self.content


class FakeTransport:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.response


def gateway_envelope(objects):
    return json.dumps({'choices': [{'message': {'content': json.dumps({'objects': objects})}}]}).encode()


class GatewayGenerationTests(unittest.TestCase):
    def gateway(self, response=None, error=None):
        transport = FakeTransport(response, error)
        return GatewayGenerator('https://gateway.example/v1/', 'test-model', 'test-secret', transport), transport

    def test_neon_models_follow_generation_role(self):
        config = {'NEON_AI_GATEWAY_BASE_URL': 'https://branch.example',
                  'NEON_AI_GATEWAY_TOKEN': 'test-secret'}
        with mock.patch('object_generation.gateway_config', return_value=config):
            models = {stage: GatewayGenerator.from_env(stage=stage).model
                      for stage in (None, 'assets', 'layout')}
        self.assertEqual(models, {None: 'gpt-5-6-sol', 'assets': 'gpt-6-astra',
                                  'layout': 'gpt-5-6-sol'})

    def test_explicit_models_and_provider_override_neon_defaults(self):
        config = {'NEON_AI_GATEWAY_BASE_URL': 'https://branch.example',
                  'NEON_AI_GATEWAY_TOKEN': 'test-secret',
                  'STAGEZERO_OBJECT_MODEL': 'general-choice',
                  'STAGEZERO_SCENE_ASSET_MODEL': 'asset-choice',
                  'STAGEZERO_SCENE_LAYOUT_MODEL': 'layout-choice'}
        with mock.patch('object_generation.gateway_config', return_value=config):
            self.assertEqual(GatewayGenerator.from_env().model, 'general-choice')
            self.assertEqual(GatewayGenerator.from_env(stage='assets').model, 'asset-choice')
            self.assertEqual(GatewayGenerator.from_env(stage='layout').model, 'layout-choice')
        config.update({'STAGEZERO_OBJECT_API_BASE': 'https://other.example/v1',
                       'STAGEZERO_OBJECT_API_KEY': 'other-secret'})
        config.pop('STAGEZERO_SCENE_ASSET_MODEL')
        config.pop('STAGEZERO_SCENE_LAYOUT_MODEL')
        with mock.patch('object_generation.gateway_config', return_value=config):
            gateway = GatewayGenerator.from_env(stage='assets')
        self.assertEqual(gateway.url, 'https://other.example/v1/chat/completions')
        self.assertEqual(gateway.model, 'general-choice')

    def test_missing_model_or_token_fails_closed_for_other_gateways(self):
        cases = ({'STAGEZERO_OBJECT_API_BASE': 'https://other.example/v1',
                  'STAGEZERO_OBJECT_API_KEY': 'other-secret'},
                 {'NEON_AI_GATEWAY_BASE_URL': 'https://branch.example'})
        for config in cases:
            with self.subTest(config=tuple(config)), mock.patch(
                    'object_generation.gateway_config', return_value=config):
                with self.assertRaises(ValueError):
                    GatewayGenerator.from_env(stage='assets')

    def test_valid_json_scene_uses_bounded_gateway_request(self):
        objects = [make_object('lamp', 0), make_object('ball', 1)]
        response = FakeResponse(gateway_envelope(objects))
        generator, transport = self.gateway(response)

        self.assertEqual(generator.generate('  a lamp and a ball  '), objects)
        self.assertTrue(response.closed)
        args, kwargs = transport.calls[0]
        self.assertEqual(args, ('https://gateway.example/v1/chat/completions',))
        self.assertEqual(kwargs['headers']['Authorization'], 'Bearer test-secret')
        self.assertEqual(kwargs['json']['messages'][-1]['content'], 'a lamp and a ball')
        self.assertEqual(kwargs['json']['response_format'], {'type': 'json_object'})
        self.assertEqual(kwargs['timeout'], (10, 45))
        self.assertTrue(kwargs['stream'])
        self.assertFalse(kwargs['allow_redirects'])

    def test_malformed_or_untrusted_gateway_output_is_rejected(self):
        bad_objects = [dict(make_object('chair', 0), script='run code')]
        bodies = (
            b'not json',
            b'{"choices": []}',
            json.dumps({'choices': [{'message': {'content': 'not json'}}]}).encode(),
            json.dumps({'choices': [{'message': {'content': '{}'}}]}).encode(),
            gateway_envelope(bad_objects),
        )
        for body in bodies:
            with self.subTest(body=body[:40]):
                response = FakeResponse(body)
                generator, _ = self.gateway(response)
                with self.assertRaises(ValueError):
                    generator.generate('a chair')
                self.assertTrue(response.closed)

    def test_response_size_http_error_and_timeout_are_rejected(self):
        cases = (
            (FakeResponse(b'x' * (MAX_RESPONSE_BYTES + 1)), None),
            (FakeResponse(b'bad gateway', status_code=502), None),
            (None, requests.Timeout('offline timeout')),
        )
        for response, error in cases:
            with self.subTest(status=getattr(response, 'status_code', None), error=error):
                generator, _ = self.gateway(response, error)
                with self.assertRaises(ValueError):
                    generator.generate('a door')
                if response is not None:
                    self.assertTrue(response.closed)

    def test_prompt_and_gateway_configuration_bounds(self):
        generator, transport = self.gateway(FakeResponse(gateway_envelope([make_object('door', 0)])))
        for prompt in ('', '  ', 'x' * 2001, None):
            with self.subTest(prompt=str(prompt)[:20]), self.assertRaises(ValueError):
                generator.generate(prompt)
        self.assertEqual(transport.calls, [])
        for url in ('http://gateway.example/v1', 'https://user:pass@gateway.example/v1',
                    'https://gateway.example/v1?token=secret'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                GatewayGenerator(url, 'model', 'key', transport)

    def test_offline_recipes_are_explicit_and_validated(self):
        objects = generate_local('Please add a door, lamp, ball, and chair')
        self.assertEqual([obj['kind'] for obj in objects], ['door', 'lamp', 'ball', 'chair'])
        self.assertEqual(len({obj['id'] for obj in objects}), 4)
        with self.assertRaises(ValueError):
            generate_local('Please add a staircase')
        with self.assertRaises(ValueError):
            generate_local('x' * 2001)


class ObjectDirectorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.backend = ControlledBackend()
        self.backend.release.set()
        self.session = ObjectDirectorSession(
            self.backend,
            np.zeros((20, 34, 3), dtype=np.float32),
            np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1)),
        )
        self.session.set_mode('Live ARDY')

    def generate_take(self):
        self.session.submit('walk')
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        return self.session.takes[self.session.active_take]

    def test_objects_survive_project_save_and_load(self):
        objects = [make_object('lamp', 0), make_object('door', 1)]
        self.session.set_objects(objects)
        self.generate_take()
        with tempfile.TemporaryDirectory() as folder:
            _path, data = self.session.save_project(folder, 'objects')
        self.session.set_objects([make_object('ball', 0)])
        self.session.load_project(data)
        self.assertEqual(self.session.scene['objects'], objects)

    def test_invalid_saved_scene_does_not_replace_current_state(self):
        current_objects = [make_object('lamp', 0)]
        self.session.set_objects(current_objects)
        take = self.generate_take()
        active, frame, revision = self.session.active_take, self.session.frame, self.session.project_revision
        invalid_scene = dict(self.session.scene, objects=[dict(make_object('ball', 0), code='unsafe')])
        data = encode_project(self.session.takes, active, frame, invalid_scene)

        with self.assertRaises(ValueError):
            self.session.load_project(data)
        self.assertIs(self.session.takes[active], take)
        self.assertEqual(self.session.active_take, active)
        self.assertEqual(self.session.frame, frame)
        self.assertEqual(self.session.project_revision, revision)
        self.assertEqual(self.session.scene['objects'], current_objects)

    def test_stale_gateway_result_cannot_replace_newer_objects(self):
        started, release = threading.Event(), threading.Event()
        proposed = [make_object('door', 0)]
        newer = [make_object('lamp', 0)]

        class DelayedGenerator:
            def generate(self, _prompt):
                started.set()
                if not release.wait(2):
                    raise TimeoutError('test did not release generator')
                return proposed

        errors = []

        def generate():
            try:
                self.session.generate_objects('a door', DelayedGenerator())
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=generate)
        thread.start()
        self.assertTrue(started.wait(1))
        self.session.set_objects(newer)
        release.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)
        self.assertEqual(self.session.scene['objects'], newer)


if __name__ == '__main__':
    unittest.main()
