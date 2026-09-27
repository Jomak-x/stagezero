"""Server-side speech-to-text adapter for voice directions."""

from __future__ import annotations

import io
import base64
import math
import os
import re
import uuid

import requests


MAX_RECORDING_BYTES = 8 * 1024 * 1024
MAX_TRANSCRIPT_LENGTH = 2000
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_SPEECH_BYTES = 8 * 1024 * 1024
MAX_TTS_RESPONSE_BYTES = 12 * 1024 * 1024
VOICE_ID_PATTERN = re.compile(r'^[A-Za-z0-9]{8,64}$')
_FORMATS = {
    'audio/webm': 'webm',
    'audio/mp4': 'm4a',
    'audio/ogg': 'ogg',
    'audio/wav': 'wav',
    'audio/mpeg': 'mp3',
}


class SpeechError(RuntimeError):
    """A user-safe speech service error; never includes API response bodies."""


def _mp3_duration(audio: bytes) -> float:
    """Count complete MPEG layer III frames; alignment may extend this duration."""
    offset = 0
    if audio[:3] == b'ID3' and len(audio) >= 10:
        offset = (10 + ((audio[6] & 127) << 21) + ((audio[7] & 127) << 14)
                  + ((audio[8] & 127) << 7) + (audio[9] & 127))
    seconds = 0.0
    bitrates = {
        3: (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
        2: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
    }
    while offset + 4 <= len(audio):
        header = int.from_bytes(audio[offset:offset + 4], 'big')
        version, layer = (header >> 19) & 3, (header >> 17) & 3
        bitrate_index, sample_index = (header >> 12) & 15, (header >> 10) & 3
        if ((header >> 21) != 0x7ff or version == 1 or layer != 1 or
                bitrate_index in (0, 15) or sample_index == 3):
            break
        rate = (44100, 48000, 32000)[sample_index]
        rate //= 1 if version == 3 else 2 if version == 2 else 4
        bitrate = bitrates[3 if version == 3 else 2][bitrate_index] * 1000
        samples = 1152 if version == 3 else 576
        size = (144 if version == 3 else 72) * bitrate // rate + ((header >> 9) & 1)
        if offset + size > len(audio):
            break
        offset += size
        seconds += samples / rate
    return seconds


class ElevenLabsSpeech:
    def __init__(self, api_key: str, *, session=None,
                 model_id: str = 'eleven_flash_v2_5'):
        self.api_key = api_key
        self.session = session if session is not None else requests.Session()
        self.model_id = model_id

    @classmethod
    def from_env(cls, *, session=None):
        return cls(os.environ.get('ELEVENLABS_API_KEY', ''), session=session,
                   model_id=os.environ.get('ELEVENLABS_TTS_MODEL', 'eleven_flash_v2_5'))

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

    def list_voice_options(self) -> list[dict[str, str]]:
        """Return at most two usable account voices per requested gender."""
        if not self.api_key:
            raise SpeechError('Set ELEVENLABS_API_KEY to generate character voices')
        result = []
        seen = set()
        for gender in ('male', 'female'):
            try:
                response = self.session.get(
                    'https://api.elevenlabs.io/v2/voices',
                    headers={'xi-api-key': self.api_key},
                    params={'gender': gender, 'page_size': 20, 'voice_type': 'default'},
                    timeout=(10, 20))
                if response.status_code in (401, 403):
                    raise SpeechError('ElevenLabs rejected the API key or voice access')
                response.raise_for_status()
                if len(response.content) > MAX_RESPONSE_BYTES:
                    raise SpeechError('Voice catalog is too large')
                document = response.json()
            except SpeechError:
                raise
            except Exception:
                raise SpeechError('Voice catalog request failed; retry shortly') from None
            voices = document.get('voices') if isinstance(document, dict) else None
            if not isinstance(voices, list):
                raise SpeechError('Voice catalog returned unusable data')
            count = 0
            for voice in voices:
                if not isinstance(voice, dict):
                    continue
                voice_id, name, labels = voice.get('voice_id'), voice.get('name'), voice.get('labels')
                if (not isinstance(voice_id, str) or not VOICE_ID_PATTERN.fullmatch(voice_id)
                        or not isinstance(name, str) or not name.strip() or voice_id in seen):
                    continue
                if isinstance(labels, dict) and labels.get('gender') not in (None, gender):
                    continue
                seen.add(voice_id)
                result.append({'id': voice_id, 'name': name.strip()[:80], 'gender': gender})
                count += 1
                if count == 2:
                    break
        return result

    def synthesize(self, text: str, voice_id: str) -> dict:
        """Generate MP3 with timing; keep the API key and raw errors server-side."""
        if not self.api_key:
            raise SpeechError('Set ELEVENLABS_API_KEY to generate character voices')
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 1000:
            raise SpeechError('Dialogue must be 1–1000 characters')
        if not isinstance(voice_id, str) or not VOICE_ID_PATTERN.fullmatch(voice_id):
            raise SpeechError('Choose an available voice')
        try:
            response = self.session.post(
                'https://api.elevenlabs.io/v1/text-to-speech/' + voice_id + '/with-timestamps',
                headers={'xi-api-key': self.api_key},
                params={'output_format': 'mp3_44100_128'},
                json={'text': text.strip(), 'model_id': self.model_id}, timeout=(10, 90))
            if response.status_code in (401, 403):
                raise SpeechError('Enable Text to Speech permission on the configured ElevenLabs API key')
            response.raise_for_status()
            if len(response.content) > MAX_TTS_RESPONSE_BYTES:
                raise SpeechError('Generated speech is too large')
            document = response.json()
            encoded = document['audio_base64']
            if not isinstance(encoded, str) or len(encoded) > MAX_TTS_RESPONSE_BYTES:
                raise ValueError('invalid audio')
            audio = base64.b64decode(encoded, validate=True)
            if not 100 <= len(audio) <= MAX_SPEECH_BYTES:
                raise ValueError('invalid audio size')
            alignment = document.get('normalized_alignment') or document.get('alignment') or {}
            ends = alignment.get('character_end_times_seconds') or []
            aligned_duration = max((float(value) for value in ends), default=0.0)
            duration = max(_mp3_duration(audio), aligned_duration)
            if not math.isfinite(duration) or not 0 < duration <= 30:
                raise ValueError('invalid duration')
        except SpeechError:
            raise
        except (KeyError, TypeError, ValueError, base64.binascii.Error):
            raise SpeechError('Speech service returned unusable audio') from None
        except Exception:
            raise SpeechError('Speech service request failed; retry shortly') from None
        return {'audio_id': str(uuid.uuid4()), 'voice_id': voice_id,
                'audio': audio, 'duration_seconds': duration}
