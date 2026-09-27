"""Offline Gemini protocol, configuration, and pipeline integration tests."""
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests

from gemini_character_reference import CharacterGenerationCancelled
from gemini_character_reference import GeminiCharacterReference, DEFAULT_MODEL
from tests.character_http_fixtures import FakeResponse, FakeTransport, response, glb
from test_neon_character_reference import image_bytes
from character_pipeline import SelfHostedCharacterGenerator


def image_response(**extra):
    return response({'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
        {'text': 'Here is your character'},
        {'inlineData': {'mimeType': 'image/png', 'data': base64.b64encode(image_bytes()).decode()}, **extra},
    ]}}]})


class GeminiCharacterTests(unittest.TestCase):
    def adapter(self, replies, **kwargs):
        transport = FakeTransport(replies)
        return GeminiCharacterReference('secret-key', transport=transport, **kwargs), transport

    def test_direct_gemini_request_and_validated_png(self):
        reply = image_response()
        generator, transport = self.adapter([reply])
        progress = []
        self.assertEqual(generator.generate('space explorer', progress.append), image_bytes())
        method, url, options = transport.calls[0]
        self.assertEqual(method, 'post')
        self.assertEqual(url, f'https://generativelanguage.googleapis.com/v1beta/models/{DEFAULT_MODEL}:generateContent')
        self.assertEqual(options['headers'], {'x-goog-api-key': 'secret-key'})
        self.assertEqual(options['json']['generationConfig']['responseModalities'], ['TEXT', 'IMAGE'])
        prompt = options['json']['contents'][0]['parts'][0]['text']
        self.assertIn('space explorer', prompt)
        self.assertIn('A-pose', prompt)
        self.assertFalse(options['allow_redirects'])
        self.assertTrue(reply.closed)
        self.assertEqual(progress[-1], 'Character reference ready')

    def test_config_file_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'characters.env'
            path.write_text("export GEMINI_API_KEY='file-key'\nMESHY_API_KEY=irrelevant\n")
            self.assertEqual(GeminiCharacterReference.from_env(environ={}, config_path=path)._api_key, 'file-key')
            configured = GeminiCharacterReference.from_env(environ={'GEMINI_API_KEY': 'env-key',
                'STAGEZERO_GEMINI_CHARACTER_MODEL': 'gemini-other-image',
                'STAGEZERO_CHARACTER_IMAGE_MODEL': 'gpt-legacy'}, config_path=path)
            self.assertEqual(configured._api_key, 'env-key')
            self.assertEqual(configured.model, 'gemini-other-image')
            path.write_text('GEMINI_API_KEY=$(danger)\n')
            with self.assertRaises(ValueError):
                GeminiCharacterReference.from_env(environ={}, config_path=path)
            path.write_text('x' * 16385)
            with self.assertRaisesRegex(ValueError, 'too large'):
                GeminiCharacterReference.from_env(environ={}, config_path=path)
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'Set GEMINI_API_KEY'):
                GeminiCharacterReference.from_env(environ={}, config_path=path)

    def test_invalid_prompts_and_models_never_send(self):
        generator, transport = self.adapter([])
        for prompt in ('', ' ', None, 'x' * 801):
            with self.assertRaises(ValueError):
                generator.generate(prompt)
        self.assertEqual(transport.calls, [])
        for model in ('gpt-6-astra', 'gemini-test/else', 'gemini-x?key=secret'):
            with self.assertRaises(ValueError):
                GeminiCharacterReference('secret', model=model)

    def test_errors_sanitized_and_responses_closed(self):
        for reply in (FakeResponse(b'secret-key', 403), FakeResponse(b'secret-key', 429),
                      FakeResponse(b'secret-key', 500), FakeResponse(b'secret-key', 302),
                      FakeResponse(b'not json'), requests.Timeout('secret-key')):
            generator, _ = self.adapter([reply])
            with self.assertRaises(ValueError) as error:
                generator.generate('explorer')
            self.assertNotIn('secret-key', str(error.exception))
            if isinstance(reply, FakeResponse):
                self.assertTrue(reply.closed)

    def test_missing_blocked_malformed_and_thought_images_rejected(self):
        documents = [None, [], {}, {'candidates': None}, {'candidates': [None]},
            {'promptFeedback': {'blockReason': 'SAFETY'}},
            {'candidates': [{'finishReason': 'SAFETY', 'content': {'parts': []}}]},
            {'candidates': [{'content': {'parts': None}}]}]
        for mime, data in [('image/jpeg', 'AAAA'), ('image/png', '!!'),
                           ('image/png', base64.b64encode(b'not a PNG').decode())]:
            documents.append({'candidates': [{'content': {'parts': [
                {'inlineData': {'mimeType': mime, 'data': data}}]}}]})
        for reply in [response(doc) for doc in documents] + [image_response(thought=True)]:
            generator, _ = self.adapter([reply])
            with self.assertRaises(ValueError):
                generator.generate('explorer')

    def test_size_limits(self):
        generator, _ = self.adapter([image_response()])
        with patch('gemini_character_reference.MAX_RESPONSE_BYTES', 10), self.assertRaisesRegex(ValueError, 'too large'):
            generator.generate('explorer')
        generator, _ = self.adapter([image_response()])
        with patch('gemini_character_reference.MAX_IMAGE_BYTES', 10), self.assertRaisesRegex(ValueError, 'invalid character image'):
            generator.generate('explorer')

    def test_cancellation_and_deadline(self):
        generator, transport = self.adapter([])
        with self.assertRaises(CharacterGenerationCancelled):
            generator.generate('explorer', cancelled=lambda: True)
        self.assertEqual(transport.calls, [])
        reply = image_response()
        generator, _ = self.adapter([reply])
        checks = iter([False, False, True])
        with self.assertRaises(CharacterGenerationCancelled):
            generator.generate('explorer', cancelled=lambda: next(checks))
        self.assertTrue(reply.closed)
        generator, _ = self.adapter([image_response()], clock=iter([0, 0, 241]).__next__)
        with self.assertRaises(TimeoutError):
            generator.generate('explorer')
        generator, _ = self.adapter([image_response()])
        with patch('gemini_character_reference.validate_reference') as validate:
            stopped = [False]
            validate.side_effect = lambda _: stopped.__setitem__(0, True)
            with self.assertRaises(CharacterGenerationCancelled):
                generator.generate('explorer', cancelled=lambda: stopped[0])

    def test_factory_uses_gemini_and_png_reaches_worker(self):
        reference, gemini = self.adapter([image_response()])
        with patch('character_pipeline.Path.read_text', return_value='worker-token'), patch.object(
                GeminiCharacterReference, 'from_env', return_value=reference) as factory:
            pipeline = SelfHostedCharacterGenerator.from_env()
        factory.assert_called_once_with()
        job = 'a' * 32
        gpu = FakeTransport([response({'ready': True}), response({'id': job}),
                             response({'status': 'succeeded'}), FakeResponse(glb())])
        pipeline.transport = gpu
        self.assertEqual(pipeline.source, 'gemini-trellis')
        self.assertEqual(pipeline.generate('explorer'), glb())
        self.assertEqual(gpu.calls[1][2]['data'], image_bytes())
        self.assertEqual(len(gemini.calls), 1)
