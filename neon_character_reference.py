"""Design a character and generate its 3D-ready reference PNG through Neon AI Gateway.

Neon documents image generation as a built-in tool on its OpenAI Responses
route, not as an Images API endpoint. Credentials remain in this server module.
"""

import base64
import binascii
from io import BytesIO
import json
import time
from urllib.parse import urlsplit

import requests
from PIL import Image

from object_generation import gateway_config


MAX_STREAM_BYTES = 48 * 1024 * 1024
MAX_EVENT_BYTES = 24 * 1024 * 1024
MAX_PNG_BYTES = 16 * 1024 * 1024
MAX_DESIGN_BYTES = 256 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

DESIGN_INSTRUCTIONS = (
    "You are designing one original character for an image-to-3D, animation-ready pipeline. "
    "Turn the user's idea into a vivid, visually specific character art brief. Preserve the user's "
    "identity, era, species, clothing, colors, and important distinctive features when given. "
    "For unspecified details, make coherent creative choices that produce a recognizable silhouette "
    "and appealing materials. For a named character, retain the iconic costume colors, mask or face, "
    "and recognizable motifs rather than replacing them with generic gear. Describe visible anatomy, "
    "face, hair or head covering, layered outfit, surface textures, and a restrained color palette. "
    "Keep the body and joints readable for rigging: "
    "two visible arms and legs, separated limbs, no wings or oversized parts obscuring the body. "
    "Integrate thematic details into clothing; avoid handheld objects, floating parts, scenery, "
    "lettering, and effects. Return one concrete visual description under 120 words. "
    "Do not include posing, camera, background, or instructions to the artist."
)


class NeonCharacterReferenceCancelled(ValueError):
    """The caller cancelled image generation."""


def _reference_prompt(description, design):
    if not isinstance(description, str) or not 1 <= len(description.strip()) <= 800:
        raise ValueError("Describe the character in 1–800 characters")
    description = description.strip()
    return (
        "Create ONE highly detailed, photographic-quality full-body character reference for image-to-3D reconstruction. "
        "Original request: " + description + ". Character design: " + design + ". "
        "Show exactly one person, head to toe, centered and directly front-facing at eye level, with "
        "orthographic-like projection. Use a symmetric neutral A-pose: shoulders level, arms angled "
        "down about 30 degrees away from the torso, elbows straight, open hands with all five fingers "
        "on each hand visibly separated, palms facing the camera. Leave clear background gaps between "
        "arms and torso, between each finger, and between both straight legs. Both feet point forward, "
        "are fully visible, and rest on the same baseline. Keep natural human proportions and readable "
        "joints, anatomically believable hands, a proportionate head, and a natural neck and shoulders. "
        "For a human character, show the convincing anatomy of an actual person beneath the clothing, "
        "with natural muscle transitions rather than inflated or sharply carved muscles. "
        "Give any visible face a neutral expression, eyes forward and mouth closed. Show the "
        "costume's actual form and physical material: fine fabric weave, precise stitching, realistic "
        "panel seams, subtle cloth tension and creases at joints, and distinct matte or glossy surfaces "
        "where appropriate to the design. Preserve fine costume details without painting false anatomy "
        "or dramatic shadows into the texture. The subject should have the photographic realism of "
        "a real person in a carefully made costume, never a toy, plastic mannequin, or flat illustration. Use broad, "
        "soft, even light with minimal baked shadows and a uniform light gray background. Fill about "
        "85 percent of the image height while keeping empty space around every limb. No scenery, "
        "props, web effects, motion, logos outside the costume, text, cropped parts, occlusion, "
        "strong perspective, or dramatic lighting."
    )


def _png_bytes(encoded):
    if not isinstance(encoded, str) or not encoded or len(encoded) > MAX_PNG_BYTES * 4 // 3 + 8:
        raise ValueError("Neon returned an invalid character image")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("Neon returned an invalid character image") from None
    if len(data) > MAX_PNG_BYTES or len(data) < 33 or not data.startswith(PNG_SIGNATURE):
        raise ValueError("Neon returned an invalid character PNG")
    if data[12:16] != b"IHDR":
        raise ValueError("Neon returned an invalid character PNG")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    if not 256 <= width <= 4096 or not 256 <= height <= 4096 or width * height > 16_777_216:
        raise ValueError("Neon returned an invalid character PNG size")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG" or image.size != (width, height):
                raise ValueError("Neon returned an invalid character PNG")
            image.verify()
    except (OSError, ValueError, SyntaxError):
        raise ValueError("Neon returned an invalid character PNG") from None
    return data


class NeonCharacterReference:
    """Bounded, cancellable Neon Responses image-generation adapter."""

    def __init__(self, base_url, api_key, *, model="gpt-6-astra", design_model="gpt-6-astra",
                 transport=None, deadline=240.0, clock=None):
        try:
            url = urlsplit(base_url)
            port = url.port
        except (TypeError, ValueError):
            raise ValueError("Neon gateway URL is invalid") from None
        if (url.scheme != "https" or not url.hostname or port not in (None, 443) or
                url.username or url.password or url.path not in ("", "/") or url.query or url.fragment):
            raise ValueError("Neon gateway URL must be a bare HTTPS host")
        if not isinstance(api_key, str) or not api_key or len(api_key) > 4096 or any(
                ord(c) < 33 or ord(c) > 126 for c in api_key):
            raise ValueError("A valid Neon gateway token is required")
        for candidate in (model, design_model):
            if not isinstance(candidate, str) or not candidate.startswith("gpt-") or len(candidate) > 100 or any(
                    c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-." for c in candidate):
                raise ValueError("A GPT model is required for Neon character generation")
        if not isinstance(deadline, (int, float)) or not 0 < deadline <= 600:
            raise ValueError("Invalid Neon image generation deadline")
        self.url = base_url.rstrip("/") + "/openai/v1/responses"
        self._api_key = api_key
        self.model = model
        self.design_model = design_model
        self.transport = transport or requests
        self.deadline = float(deadline)
        self.clock = clock or time.monotonic

    @classmethod
    def from_env(cls, **kwargs):
        config = gateway_config()
        base = config.get("NEON_AI_GATEWAY_BASE_URL")
        token = config.get("NEON_AI_GATEWAY_TOKEN")
        if not base or not token:
            raise ValueError("Configure Neon gateway URL and token in .runtime/objects.env or the environment")
        model = kwargs.pop("model", config.get("STAGEZERO_CHARACTER_IMAGE_MODEL", "gpt-6-astra"))
        design_model = kwargs.pop("design_model", config.get("STAGEZERO_CHARACTER_DESIGN_MODEL", "gpt-6-astra"))
        return cls(base, token, model=model, design_model=design_model, **kwargs)

    def _check(self, cancelled, expires):
        if cancelled():
            raise NeonCharacterReferenceCancelled("Character reference generation cancelled")
        remaining = expires - self.clock()
        if remaining <= 0:
            raise TimeoutError("Neon character reference generation timed out")
        return remaining

    @staticmethod
    def _image_from_event(event):
        kind = event.get("type")
        if kind == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "image_generation_call":
                return item.get("result")
        if kind == "response.completed":
            response = event.get("response")
            if isinstance(response, dict):
                output = response.get("output")
                for item in output if isinstance(output, list) else []:
                    if isinstance(item, dict) and item.get("type") == "image_generation_call":
                        return item.get("result")
        return None

    @staticmethod
    def _image_call_failed(event):
        if event.get("type") == "response.output_item.done":
            items = [event.get("item")]
        elif event.get("type") == "response.completed":
            response = event.get("response")
            items = response.get("output", []) if isinstance(response, dict) else []
        else:
            return False
        if not isinstance(items, list):
            return False
        return any(isinstance(item, dict) and item.get("type") == "image_generation_call"
                   and item.get("status") == "failed" for item in items)

    @staticmethod
    def _text_from_event(event):
        if event.get("type") == "response.output_item.done":
            items = [event.get("item")]
        elif event.get("type") == "response.completed":
            response = event.get("response")
            items = response.get("output", []) if isinstance(response, dict) else []
        else:
            return None
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            parts = [part.get("text") for part in content if isinstance(part, dict)
                     and part.get("type") == "output_text" and isinstance(part.get("text"), str)]
            if parts:
                return " ".join(parts)
        return None

    def _events(self, payload, cancelled, expires, *, max_stream_bytes):
        remaining = self._check(cancelled, expires)
        try:
            with self.transport.post(
                self.url,
                headers={"Authorization": "Bearer " + self._api_key,
                         "Accept": "text/event-stream"},
                json=payload,
                timeout=(min(10, remaining), min(30, remaining)),
                stream=True,
                allow_redirects=False,
            ) as response:
                if response.status_code != 200:
                    raise ValueError(f"Neon character request returned HTTP {response.status_code}")
                total = 0
                line = bytearray()
                for chunk in response.iter_content(8192):
                    self._check(cancelled, expires)
                    total += len(chunk)
                    if total > max_stream_bytes:
                        raise ValueError("Neon character response is too large")
                    for byte in chunk:
                        if byte != 10:
                            line.append(byte)
                            if len(line) > MAX_EVENT_BYTES:
                                raise ValueError("Neon character event is too large")
                            continue
                        raw = bytes(line).rstrip(b"\r")
                        line.clear()
                        if not raw.startswith(b"data: "):
                            continue
                        try:
                            event = json.loads(raw[6:])
                        except (UnicodeError, json.JSONDecodeError):
                            raise ValueError("Neon returned an invalid character event") from None
                        if not isinstance(event, dict):
                            raise ValueError("Neon returned an invalid character event")
                        if event.get("type") in ("response.failed", "response.incomplete", "error"):
                            raise ValueError("Neon could not generate the character")
                        yield event
        except requests.RequestException:
            raise ValueError("Neon character connection failed") from None

    def _design(self, description, progress, cancelled, expires):
        progress("Designing character with Neon")
        payload = {
            "model": self.design_model,
            "input": [{"role": "developer", "content": DESIGN_INSTRUCTIONS},
                      {"role": "user", "content": description}],
            "reasoning": {"effort": "medium"},
            "max_output_tokens": 500,
            "stream": True,
            "store": False,
        }
        for event in self._events(payload, cancelled, expires, max_stream_bytes=MAX_DESIGN_BYTES):
            design = self._text_from_event(event)
            if design is not None:
                design = " ".join(design.split())
                if not 1 <= len(design) <= 1600:
                    raise ValueError("Neon returned an invalid character design")
                self._check(cancelled, expires)
                return design
        raise ValueError("Neon returned no character design")

    def generate(self, prompt, progress=lambda message: None, cancelled=lambda: False) -> bytes:
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 800:
            raise ValueError("Describe the character in 1–800 characters")
        description = prompt.strip()
        if not callable(progress) or not callable(cancelled):
            raise ValueError("Character reference callbacks must be callable")
        expires = self.clock() + self.deadline
        self._check(cancelled, expires)
        design = self._design(description, progress, cancelled, expires)
        progress("Generating character reference with Neon")
        payload = {
            "model": self.model,
            "input": _reference_prompt(description, design),
            "tools": [{"type": "image_generation", "action": "generate", "size": "1024x1536",
                       "quality": "high", "output_format": "png"}],
            "tool_choice": {"type": "image_generation"},
            "stream": True,
            "store": False,
        }
        image_failed = False
        for event in self._events(payload, cancelled, expires, max_stream_bytes=MAX_STREAM_BYTES):
            image_failed |= self._image_call_failed(event)
            encoded = self._image_from_event(event)
            if encoded is not None:
                data = _png_bytes(encoded)
                self._check(cancelled, expires)
                progress("Character reference ready")
                return data
        if image_failed:
            raise ValueError("Neon image generation failed")
        raise ValueError("Neon returned no character image")
