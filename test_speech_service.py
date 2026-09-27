"""Offline tests for the voice direction transcription adapter."""

import os
import unittest
from unittest.mock import patch

import requests

from speech_service import ElevenLabsSpeech, MAX_RECORDING_BYTES, SpeechError


class FakeResponse:
    def __init__(self, document=None, status_code=200, content=b'{}', error=None):
        self.document = document
        self.status_code = status_code
        self.content = content
        self.error = error

    def json(self):
        if self.error is not None:
            raise self.error
        return self.document

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError('private response body: secret-key')


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error is not None:
            raise self.error
        return self.response


class SpeechServiceTests(unittest.TestCase):
    def setUp(self):
        self.audio = b'0' * 100

    def test_transcribes_with_bounded_request_and_trimmed_text(self):
        session = FakeSession(FakeResponse({'text': '  Walk to the door.  '}))
        speech = ElevenLabsSpeech('secret-key', session=session)
        self.assertEqual(speech.transcribe(self.audio, 'audio/webm;codecs=opus'), 'Walk to the door.')
        url, kwargs = session.calls[0]
        self.assertEqual(url, 'https://api.elevenlabs.io/v1/speech-to-text')
        self.assertEqual(kwargs['headers'], {'xi-api-key': 'secret-key'})
        self.assertEqual(kwargs['timeout'], (10, 90))
        self.assertEqual(kwargs['data'], {'model_id': 'scribe_v2', 'tag_audio_events': 'false'})
        filename, stream, mime = kwargs['files']['file']
        self.assertEqual((filename, stream.getvalue(), mime), ('recording.webm', self.audio, 'audio/webm'))

    def test_from_env_and_missing_key(self):
        session = FakeSession(FakeResponse({'text': 'Hello'}))
        with patch.dict(os.environ, {'ELEVENLABS_API_KEY': 'secret-key'}):
            self.assertEqual(ElevenLabsSpeech.from_env(session=session).transcribe(self.audio, 'audio/wav'), 'Hello')
        with patch.dict(os.environ, {'ELEVENLABS_API_KEY': ''}):
            with self.assertRaisesRegex(SpeechError, 'ELEVENLABS_API_KEY'):
                ElevenLabsSpeech.from_env(session=session).transcribe(self.audio, 'audio/wav')

    def test_rejects_unsupported_or_unbounded_recording_before_request(self):
        session = FakeSession(FakeResponse({'text': 'Hello'}))
        speech = ElevenLabsSpeech('secret-key', session=session)
        for audio, mime in ((b'', 'audio/wav'), (b'x' * 99, 'audio/wav'),
                            (b'x' * (MAX_RECORDING_BYTES + 1), 'audio/wav'),
                            ('not bytes', 'audio/wav'), (self.audio, 'text/plain'),
                            (self.audio, None)):
            with self.subTest(mime=mime, size=len(audio)):
                with self.assertRaises(SpeechError):
                    speech.transcribe(audio, mime)
        self.assertEqual(session.calls, [])

    def test_permission_and_key_errors_are_safe(self):
        for response, expected in (
            (FakeResponse({'detail': {'status': 'missing_permissions', 'message': 'secret-key'}}, 403), 'Speech to Text permission'),
            (FakeResponse({'detail': 'secret-key'}, 401), 'rejected the API key'),
            (FakeResponse(status_code=403, error=ValueError('secret-key')), 'rejected the API key'),
        ):
            with self.subTest(expected=expected):
                speech = ElevenLabsSpeech('secret-key', session=FakeSession(response))
                with self.assertRaises(SpeechError) as raised:
                    speech.transcribe(self.audio, 'audio/ogg')
                self.assertIn(expected, str(raised.exception))
                self.assertNotIn('secret-key', str(raised.exception))

    def test_timeout_and_http_errors_do_not_leak_secret(self):
        for session in (FakeSession(error=requests.Timeout('secret-key')),
                        FakeSession(FakeResponse(status_code=500))):
            with self.subTest(session=session):
                with self.assertRaises(SpeechError) as raised:
                    ElevenLabsSpeech('secret-key', session=session).transcribe(self.audio, 'audio/mp4')
                self.assertEqual(str(raised.exception), 'Speech service request failed; retry shortly')
                self.assertNotIn('secret-key', str(raised.exception))

    def test_malformed_or_oversized_response(self):
        cases = (
            (FakeResponse(error=ValueError('secret-key')), 'Speech service request failed'),
            (FakeResponse({'transcript': 'ignored'}), 'No speech was heard'),
            (FakeResponse({'text': '   '}), 'No speech was heard'),
            (FakeResponse({'text': 'x' * 2001}), 'Direction is too long'),
            (FakeResponse({'text': 'okay'}, content=b'x' * (1024 * 1024 + 1)), 'Speech response is too large'),
        )
        for response, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(SpeechError) as raised:
                    ElevenLabsSpeech('secret-key', session=FakeSession(response)).transcribe(self.audio, 'audio/mpeg')
                self.assertIn(expected, str(raised.exception))
                self.assertNotIn('secret-key', str(raised.exception))


if __name__ == '__main__':
    unittest.main()
