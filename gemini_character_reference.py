"""Direct Gemini image generation for the existing private image-to-3D worker."""
import base64
import binascii
import json
import math
import os
import re
import shlex
from pathlib import Path
import time

import requests

from character_reference import validate_reference
from neon_character_reference import _reference_prompt

DEFAULT_MODEL = "gemini-3.1-flash-image"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class CharacterGenerationCancelled(ValueError):
    """Local cancellation; a provider request already in flight may finish."""


def _configured_key(*, environ, config_path=None, key_name="GEMINI_API_KEY"):
    path = Path(config_path) if config_path is not None else Path(__file__).resolve().parent / ".runtime/characters.env"
    key = environ.get(key_name)
    if key_name not in environ and path.is_file():
        if path.stat().st_size > 16384:
            raise ValueError("Character configuration file is too large")
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = shlex.split(line, comments=True)
                if parts[:1] == ["export"]:
                    parts = parts[1:]
                if len(parts) == 1 and parts[0].startswith(key_name + "="):
                    key = parts[0].split("=", 1)[1]
        except (UnicodeError, ValueError):
            raise ValueError("Invalid .runtime/characters.env") from None
    if (not isinstance(key, str) or not key or len(key) > 4096 or "$" in key or "`" in key
            or any(ord(c) < 33 or ord(c) > 126 for c in key)):
        raise ValueError("Set GEMINI_API_KEY in the environment or .runtime/characters.env")
    return key


class GeminiCharacterReference:
    def __init__(self, api_key, *, model=DEFAULT_MODEL, transport=None,
                 deadline=240.0, clock=None):
        self._api_key = _configured_key(environ={"GEMINI_API_KEY": api_key}, key_name="GEMINI_API_KEY")
        if not isinstance(model, str) or not re.fullmatch(r"gemini-[a-zA-Z0-9.-]{1,90}", model):
            raise ValueError("A Gemini image model is required")
        if type(deadline) not in (int, float) or not math.isfinite(deadline) or not 0 < deadline <= 600:
            raise ValueError("Invalid Gemini generation deadline")
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.transport = transport or requests
        self.deadline = deadline
        self.clock = clock or time.monotonic

    @classmethod
    def from_env(cls, *, environ=None, config_path=None, **kwargs):
        environ = os.environ if environ is None else environ
        key = _configured_key(environ=environ, config_path=config_path, key_name="GEMINI_API_KEY")
        kwargs.setdefault("model", environ.get("STAGEZERO_GEMINI_CHARACTER_MODEL", DEFAULT_MODEL))
        return cls(key, **kwargs)

    def generate(self, description, progress=lambda message: None, cancelled=lambda: False):
        # Reuse the established A-pose and framing constraints, without a second design call.
        prompt = _reference_prompt(description, "Preserve the requested appearance and choose coherent details")
        expires = self.clock() + self.deadline

        def check():
            if cancelled():
                raise CharacterGenerationCancelled("Character reference generation cancelled")
            remaining = expires - self.clock()
            if remaining <= 0:
                raise TimeoutError("Gemini character reference generation timed out")
            return remaining

        remaining = check()
        progress("1 / 3 · Designing the character with Gemini…")
        try:
            with self.transport.post(
                self.url, headers={"x-goog-api-key": self._api_key},
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}},
                timeout=(min(10, remaining), remaining), stream=True, allow_redirects=False,
            ) as response:
                check()
                if response.status_code in (401, 403):
                    raise ValueError("Gemini authentication failed; check GEMINI_API_KEY and image model access")
                if response.status_code == 429:
                    raise ValueError("Gemini quota exceeded; check API billing and quota or try again later")
                if response.status_code != 200:
                    raise ValueError(f"Gemini character request returned HTTP {response.status_code}")
                body = bytearray()
                for chunk in response.iter_content(65536):
                    check()
                    body.extend(chunk)
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise ValueError("Gemini character response is too large")
        except requests.RequestException:
            raise ValueError("Gemini character connection failed; try again") from None
        check()
        try:
            document = json.loads(body)
        except (ValueError, UnicodeError):
            raise ValueError("Gemini returned invalid JSON") from None
        candidates = document.get("candidates", []) if isinstance(document, dict) else []
        for candidate in candidates if isinstance(candidates, list) else []:
            if not isinstance(candidate, dict) or candidate.get("finishReason") not in (None, "STOP"):
                continue
            content = candidate.get("content")
            parts = content.get("parts", []) if isinstance(content, dict) else []
            for part in parts if isinstance(parts, list) else []:
                if not isinstance(part, dict) or part.get("thought"):
                    continue
                inline = part.get("inlineData")
                if not isinstance(inline, dict) or inline.get("mimeType") != "image/png":
                    continue
                encoded = inline.get("data")
                if not isinstance(encoded, str) or len(encoded) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
                    raise ValueError("Gemini returned an invalid character image")
                try:
                    png = base64.b64decode(encoded, validate=True)
                    validate_reference(png)
                except (ValueError, binascii.Error):
                    raise ValueError("Gemini returned an invalid character image") from None
                check()
                progress("Character reference ready")
                return png
        raise ValueError("Gemini returned no character image; try a different description")
