"""Offline Gemini text protocol and Gemini-to-Neon handoff tests."""
import base64
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import requests

from gemini_character_design import GeminiCharacterDesigner, CharacterGenerationCancelled, DEFAULT_MODEL
from neon_character_reference import NeonCharacterReference
from tests.character_http_fixtures import FakeTransport, FakeResponse, response
from test_neon_character_reference import FakeTransport as NeonTransport, FakeResponse as NeonResponse, image_bytes


def design_response(text='An explorer in a fitted teal suit with orange shoulders.'):
    return response({'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
        {'text': 'private reasoning', 'thought': True}, {'text': text}]}}]})


class GeminiDesignTests(unittest.TestCase):
    def designer(self, replies, **kwargs):
        transport = FakeTransport(replies)
        return GeminiCharacterDesigner('secret-key', transport=transport, **kwargs), transport

    def test_text_request_and_neon_image_handoff(self):
        designer, transport = self.designer([design_response()])
        image = {'type': 'response.output_item.done', 'item': {'type': 'image_generation_call',
                 'result': base64.b64encode(image_bytes()).decode()}}
        neon_transport = NeonTransport(NeonResponse([image]))
        neon = NeonCharacterReference('https://branch.example', 'neon-secret',
                                      designer=designer, transport=neon_transport)
        self.assertEqual(neon.generate('Explorer'), image_bytes())
        self.assertEqual(len(transport.calls), 1)
        _, url, options = transport.calls[0]
        self.assertIn(DEFAULT_MODEL + ':generateContent', url)
        self.assertEqual(options['headers'], {'x-goog-api-key': 'secret-key'})
        self.assertNotIn('responseModalities', options['json']['generationConfig'])
        self.assertIn('under 120 words', options['json']['systemInstruction']['parts'][0]['text'])
        self.assertFalse(options['allow_redirects'])
        self.assertEqual(len(neon_transport.requests), 1)  # No Neon text-design request.
        payload = neon_transport.requests[0][1]['json']
        self.assertIn('teal suit', payload['input'])
        self.assertNotIn('private reasoning', payload['input'])
        self.assertEqual(payload['tools'][0]['type'], 'image_generation')

    def test_private_config_and_environment_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'characters.env'
            path.write_text("export GEMINI_API_KEY='file-key'\n")
            self.assertEqual(GeminiCharacterDesigner.from_env(environ={}, config_path=path)._api_key, 'file-key')
            d = GeminiCharacterDesigner.from_env(environ={'GEMINI_API_KEY': 'env-key',
                'STAGEZERO_GEMINI_CHARACTER_DESIGN_MODEL': 'gemini-text'}, config_path=path)
            self.assertEqual((d._api_key, d.model), ('env-key', 'gemini-text'))
            path.write_text('GEMINI_API_KEY=$(danger)')
            with self.assertRaises(ValueError):
                GeminiCharacterDesigner.from_env(environ={}, config_path=path)
            path.write_text('x' * 16385)
            with self.assertRaisesRegex(ValueError, 'too large'):
                GeminiCharacterDesigner.from_env(environ={}, config_path=path)
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'Set GEMINI_API_KEY'):
                GeminiCharacterDesigner.from_env(environ={}, config_path=path)

    def test_invalid_inputs_and_models(self):
        d, t = self.designer([])
        for prompt in ('', ' ', None, 'x' * 801):
            with self.assertRaises(ValueError): d.generate(prompt)
        self.assertEqual(t.calls, [])
        for model in ('gpt-astra', 'gemini-test/key', 'gemini-test?key=secret'):
            with self.assertRaises(ValueError): GeminiCharacterDesigner('key', model=model)

    def test_errors_do_not_leak_or_fallback_to_neon(self):
        for reply in (FakeResponse(b'secret-key', 403), FakeResponse(b'secret-key', 429),
                      FakeResponse(b'secret-key', 500), FakeResponse(b'secret-key', 302),
                      requests.Timeout('secret-key'), FakeResponse(b'bad JSON'),
                      design_response('x'*1601), design_response(''), response(None),
                      response({'candidates': [{'finishReason': 'SAFETY'}]}),
                      response({'candidates': [{'content': {'parts': None}}]})):
            d, _ = self.designer([reply])
            transport = NeonTransport()
            neon = NeonCharacterReference('https://branch.example', 'token', designer=d, transport=transport)
            with self.assertRaises(ValueError) as error: neon.generate('Explorer')
            self.assertNotIn('secret-key', str(error.exception))
            self.assertEqual(transport.requests, [])
            if isinstance(reply, FakeResponse): self.assertTrue(reply.closed)

    def test_cancel_deadline_and_response_bound(self):
        d,t = self.designer([])
        with self.assertRaises(CharacterGenerationCancelled): d.generate('Explorer', cancelled=lambda: True)
        self.assertEqual(t.calls, [])
        reply = design_response()
        d,_ = self.designer([reply])
        checks=iter([False,False,True])
        with self.assertRaises(CharacterGenerationCancelled): d.generate('Explorer',cancelled=lambda:next(checks))
        self.assertTrue(reply.closed)
        d,_=self.designer([design_response()],clock=iter([0,0,61]).__next__)
        with self.assertRaises(TimeoutError): d.generate('Explorer')
        d,_=self.designer([design_response()])
        with patch('gemini_character_design.MAX_RESPONSE_BYTES',10),self.assertRaisesRegex(ValueError,'too large'):
            d.generate('Explorer')

    def test_factory_injects_gemini_into_neon(self):
        from character_pipeline import SelfHostedCharacterGenerator
        with patch('character_pipeline.Path.read_text',return_value='token'), patch.object(
                GeminiCharacterDesigner,'from_env') as gemini, patch.object(NeonCharacterReference,'from_env') as neon:
            pipeline=SelfHostedCharacterGenerator.from_env()
        neon.assert_called_once_with(designer=gemini.return_value)
        self.assertIs(pipeline.reference_generator,neon.return_value)
        self.assertEqual(pipeline.source,'gemini-neon-trellis')

    def test_exact_output_boundary_and_truncation(self):
        d, _ = self.designer([design_response('x' * 1600)])
        self.assertEqual(len(d.generate('Explorer')), 1600)
        for finish in ('MAX_TOKENS', None):
            d, _ = self.designer([response({'candidates': [{'finishReason': finish,
                'content': {'parts': [{'text': 'Truncated brief'}]}}]})])
            with self.assertRaisesRegex(ValueError, 'no character design'):
                d.generate('Explorer')

    def test_enclosing_neon_deadline_stops_after_design(self):
        from unittest.mock import Mock
        designer = Mock()
        clock = [0]
        def generate(*args, **kwargs):
            clock[0] = 11
            return 'A teal explorer'
        designer.generate.side_effect = generate
        transport = NeonTransport()
        neon = NeonCharacterReference('https://branch.example', 'token', designer=designer,
                                      transport=transport, deadline=10, clock=lambda: clock[0])
        with self.assertRaises(TimeoutError):
            neon.generate('Explorer')
        self.assertEqual(transport.requests, [])
