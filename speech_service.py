"""Server-side speech-to-text adapter for voice directions."""

from __future__ import annotations

import io
import os

import requests


MAX_RECORDING_BYTES = 8 * 1024 * 1024
MAX_TRANSCRIPT_LENGTH = 2000
MAX_RESPONSE_BYTES = 1024 * 1024
_FORMATS = {
    'audio/webm': 'webm',
    'audio/mp4': 'm4a',
    'audio/ogg': 'ogg',
    'audio/wav': 'wav',
    'audio/mpeg': 'mp3',
}


class SpeechError(RuntimeError):
    """A user-safe speech service error; never includes API response bodies."""


class ElevenLabsSpeech:
    def __init__(self, api_key: str, *, session=None):
        self.api_key = api_key
        self.session = session if session is not None else requests.Session()

    @classmethod
    def from_env(cls, *, session=None):
        return cls(os.environ.get('ELEVENLABS_API_KEY', ''), session=session)

    def _post(self, path: str, **kwargs):
        if not self.api_key:
            raise SpeechError('Set ELEVENLABS_API_KEY to use speech')
        try:
            response = self.session.post(
                'https://api.elevenlabs.io' + path,
                headers={'xi-api-key': self.api_key}, timeout=(10, 90), **kwargs)
            if response.status_code in (401, 403):
                try:
                    detail = response.json().get('detail', {})
                    code = detail.get('status') if isinstance(detail, dict) else None
                except (TypeError, ValueError, AttributeError):
                    code = None
                if code == 'missing_permissions':
                    raise SpeechError('Enable Speech to Text permission on the configured ElevenLabs API key')
                raise SpeechError('ElevenLabs rejected the API key; check the saved key and permissions')
            response.raise_for_status()
            if len(response.content) > MAX_RESPONSE_BYTES:
                raise SpeechError('Speech response is too large')
            return response.json()
        except SpeechError:
            raise
        except Exception:
            raise SpeechError('Speech service request failed; retry shortly') from None

    def transcribe(self, audio: bytes, mime_type: str) -> str:
        if not isinstance(audio, bytes) or not 100 <= len(audio) <= MAX_RECORDING_BYTES:
            raise SpeechError('Recording is empty or too large')
        basic_type = mime_type.split(';', 1)[0].strip().lower() if isinstance(mime_type, str) else ''
        extension = _FORMATS.get(basic_type)
        if extension is None:
            raise SpeechError('Unsupported recording format')
        document = self._post(
            '/v1/speech-to-text',
            files={'file': ('recording.' + extension, io.BytesIO(audio), basic_type)},
            data={'model_id': 'scribe_v2', 'tag_audio_events': 'false'})
        text = document.get('text') if isinstance(document, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise SpeechError('No speech was heard in the recording')
        if len(text.strip()) > MAX_TRANSCRIPT_LENGTH:
            raise SpeechError('Direction is too long; try a shorter recording')
        return text.strip()
