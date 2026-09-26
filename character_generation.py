"""Generate a textured, self-contained character GLB with Meshy's Text to 3D API.

Meshy charges for submitted tasks. A local cancellation stops waiting and downloading,
but it does not cancel a task already submitted to Meshy.
API: https://docs.meshy.ai/en/api/text-to-3d
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import struct
import time
from urllib.parse import urlsplit

import requests


API_URL = "https://api.meshy.ai/openapi/v2/text-to-3d"
MAX_RESPONSE_BYTES = 128_000
MAX_GLB_BYTES = 40 * 1024 * 1024
MAX_CONFIG_BYTES = 16_384
_TASK_ID = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")


class CharacterGenerationCancelled(ValueError):
    """The caller stopped waiting; already-submitted remote work may continue."""


def _data_uri(uri: object) -> bool:
    return isinstance(uri, str) and uri.lower().startswith("data:") and "," in uri


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate character GLB JSON key")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ValueError("Invalid character GLB JSON number")


def validate_glb(data: bytes) -> bytes:
    """Validate GLB framing and core resource containment, returning the original bytes.

    This is an ingestion check, not a full glTF renderer/parser. GLB JSON may include
    embedded data URIs; buffer and image references to external files are refused.
    """
    if not isinstance(data, bytes) or len(data) > MAX_GLB_BYTES or len(data) < 20:
        raise ValueError("Invalid or oversized character GLB")
    magic, version, declared = struct.unpack_from("<4sII", data)
    if magic != b"glTF" or version != 2 or declared != len(data) or declared % 4:
        raise ValueError("Invalid character GLB header")

    chunks = []
    offset = 12
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("Truncated character GLB chunk")
        length, kind = struct.unpack_from("<I4s", data, offset)
        offset += 8
        if length % 4 or offset + length > len(data):
            raise ValueError("Invalid character GLB chunk length")
        chunks.append((kind, data[offset:offset + length]))
        offset += length
    if not chunks or chunks[0][0] != b"JSON" or len(chunks) > 2 or (
        len(chunks) == 2 and chunks[1][0] != b"BIN\0"
    ):
        raise ValueError("Invalid character GLB chunk layout")
    try:
        document = json.loads(chunks[0][1].rstrip(b" ").decode("utf-8"),
                              object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (UnicodeError, json.JSONDecodeError, ValueError):
        raise ValueError("Invalid character GLB JSON") from None
    if not isinstance(document, dict) or not isinstance(document.get("asset"), dict) or (
        document["asset"].get("version") != "2.0"
    ):
        raise ValueError("Invalid character glTF asset")

    buffers = document.get("buffers", [])
    images = document.get("images", [])
    views = document.get("bufferViews", [])
    if not all(isinstance(value, list) for value in (buffers, images, views)):
        raise ValueError("Invalid character GLB resources")
    embedded_buffer_count = 0
    for buffer in buffers:
        if not isinstance(buffer, dict) or type(buffer.get("byteLength")) is not int or buffer["byteLength"] < 0:
            raise ValueError("Invalid character GLB buffer")
        if "uri" in buffer:
            if not _data_uri(buffer["uri"]):
                raise ValueError("Character GLB references an external buffer")
        else:
            embedded_buffer_count += 1
            if embedded_buffer_count > 1 or len(chunks) < 2 or buffer["byteLength"] > len(chunks[1][1]) or (
                len(chunks[1][1]) - buffer["byteLength"] > 3
            ):
                raise ValueError("Character GLB is missing embedded buffer data")
    for view in views:
        if not isinstance(view, dict) or type(view.get("buffer")) is not int or not (0 <= view["buffer"] < len(buffers)):
            raise ValueError("Invalid character GLB buffer view")
        start = view.get("byteOffset", 0)
        length = view.get("byteLength")
        if type(start) is not int or type(length) is not int or start < 0 or length < 0 or (
            start + length > buffers[view["buffer"]]["byteLength"]
        ):
            raise ValueError("Invalid character GLB buffer view")
    for image in images:
        if not isinstance(image, dict):
            raise ValueError("Invalid character GLB image")
        if "uri" in image:
            if not _data_uri(image["uri"]):
                raise ValueError("Character GLB references an external image")
        elif type(image.get("bufferView")) is not int or not (0 <= image["bufferView"] < len(views)):
            raise ValueError("Character GLB image is missing embedded data")
    return data


def _configured_key(*, environ=None, config_path=None) -> str:
    """Read only MESHY_API_KEY without evaluating shell expressions."""
    environ = os.environ if environ is None else environ
    path = Path(config_path) if config_path is not None else Path(__file__).resolve().parent / ".runtime" / "characters.env"
    key = None
    if "MESHY_API_KEY" not in environ and path.is_file():
        if path.stat().st_size > MAX_CONFIG_BYTES:
            raise ValueError("Character configuration file is too large")
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                parts = shlex.split(line, comments=True)
                if parts[:1] == ["export"]:
                    parts = parts[1:]
                if len(parts) == 1 and parts[0].startswith("MESHY_API_KEY="):
                    key = parts[0].split("=", 1)[1]
        except (UnicodeError, ValueError):
            raise ValueError("Invalid .runtime/characters.env") from None
    key = environ.get("MESHY_API_KEY", key)
    if not isinstance(key, str) or not key or len(key) > 4096 or any(ord(char) < 33 or ord(char) > 126 for char in key) or "$" in key or "`" in key:
        raise ValueError("Set MESHY_API_KEY in the environment or .runtime/characters.env")
    return key


class MeshyCharacterGenerator:
    """Synchronous, cancellable preview/refine adapter with injectable transport."""

    def __init__(self, api_key: str, *, transport=None, poll_interval=5.0,
                 deadline=1800.0, max_polls=360, clock=None, sleep=None):
        if not isinstance(api_key, str) or not api_key or len(api_key) > 4096 or "$" in api_key or "`" in api_key or any(
            ord(char) < 33 or ord(char) > 126 for char in api_key
        ):
            raise ValueError("A valid Meshy API key is required")
        if not 0 < poll_interval <= 60 or not 0 < deadline <= 7200 or not 0 < max_polls <= 1440:
            raise ValueError("Invalid Meshy polling limits")
        self._api_key = api_key
        self.transport = transport or requests
        self.poll_interval = poll_interval
        self.deadline = deadline
        self.max_polls = max_polls
        self.clock = clock or time.monotonic
        self.sleep = sleep or time.sleep

    @classmethod
    def from_env(cls, **kwargs):
        environ = kwargs.pop("environ", None)
        config_path = kwargs.pop("config_path", None)
        return cls(_configured_key(environ=environ, config_path=config_path), **kwargs)

    def _check(self, cancelled, expires):
        if cancelled():
            raise CharacterGenerationCancelled("Character generation cancelled; a submitted Meshy task may continue")
        remaining = expires - self.clock()
        if remaining <= 0:
            raise TimeoutError("Character generation timed out; a submitted Meshy task may continue")
        return remaining

    def _request(self, method, url, *, expires, cancelled, payload=None, limit=MAX_RESPONSE_BYTES,
                 authenticated=True):
        remaining = self._check(cancelled, expires)
        headers = {"Authorization": "Bearer " + self._api_key} if authenticated else {}
        try:
            request = getattr(self.transport, method)
            kwargs = {"headers": headers, "timeout": (min(5, remaining), min(30, remaining)),
                      "stream": True, "allow_redirects": False}
            if payload is not None:
                kwargs["json"] = payload
            with request(url, **kwargs) as response:
                if not 200 <= response.status_code < 300:
                    raise ValueError(f"Meshy {method.upper()} returned HTTP {response.status_code}")
                raw_length = response.headers.get("Content-Length")
                if raw_length is not None:
                    try:
                        declared_length = int(raw_length)
                    except (TypeError, ValueError):
                        raise ValueError("Invalid Meshy response length") from None
                    if declared_length < 0:
                        raise ValueError("Invalid Meshy response length")
                    if declared_length > limit:
                        raise ValueError("Meshy response is too large")
                body = bytearray()
                for chunk in response.iter_content(64 * 1024):
                    self._check(cancelled, expires)
                    body.extend(chunk)
                    if len(body) > limit:
                        raise ValueError("Meshy response is too large")
                return bytes(body)
        except requests.RequestException:
            raise ValueError("Meshy connection failed") from None

    def _json(self, method, url, *, expires, cancelled, payload=None):
        body = self._request(method, url, expires=expires, cancelled=cancelled, payload=payload)
        try:
            result = json.loads(body)
        except (UnicodeError, json.JSONDecodeError):
            raise ValueError("Meshy returned invalid JSON") from None
        if not isinstance(result, dict):
            raise ValueError("Meshy returned invalid task data")
        return result

    def _create(self, payload, *, expires, cancelled):
        result = self._json("post", API_URL, expires=expires, cancelled=cancelled, payload=payload).get("result")
        if not isinstance(result, str) or not _TASK_ID.fullmatch(result):
            raise ValueError("Meshy returned an invalid task ID")
        return result

    def _poll(self, task_id, phase, *, expires, cancelled, progress, polls):
        while True:
            self._check(cancelled, expires)
            if polls[0] >= self.max_polls:
                raise TimeoutError("Character generation reached its polling limit; a submitted Meshy task may continue")
            polls[0] += 1
            task = self._json("get", API_URL + "/" + task_id, expires=expires, cancelled=cancelled)
            status = task.get("status")
            if status == "SUCCEEDED":
                progress(f"Meshy {phase} complete")
                return task
            if status in ("FAILED", "CANCELED"):
                raise ValueError(f"Meshy {phase} task {status.lower()}")
            if status not in ("PENDING", "IN_PROGRESS"):
                raise ValueError("Meshy returned an unknown task status")
            percent = task.get("progress")
            suffix = f" ({percent}%)" if type(percent) is int and 0 <= percent <= 100 else ""
            progress(f"Meshy {phase} {status.lower().replace('_', ' ')}{suffix}")
            self.sleep(min(self.poll_interval, self._check(cancelled, expires)))

    @staticmethod
    def _artifact_url(url):
        if not isinstance(url, str) or len(url) > 8192 or any(ord(char) < 32 for char in url):
            raise ValueError("Meshy returned an unsafe GLB URL")
        try:
            parsed = urlsplit(url)
            port = parsed.port
        except ValueError:
            raise ValueError("Meshy returned an unsafe GLB URL") from None
        # Meshy's documented signed model URLs use this exact public asset host.
        if parsed.scheme != "https" or parsed.hostname != "assets.meshy.ai" or port not in (None, 443) or (
            parsed.username is not None or parsed.password is not None or parsed.fragment or
            not parsed.path.lower().endswith(".glb")
        ):
            raise ValueError("Meshy returned an unsafe GLB URL")
        return url

    def generate(self, prompt, progress=lambda message: None, cancelled=lambda: False) -> bytes:
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 800:
            raise ValueError("Describe the character in 1–800 characters")
        prompt = prompt.strip()
        if not callable(progress) or not callable(cancelled):
            raise ValueError("Character generation callbacks must be callable")
        expires = self.clock() + self.deadline
        polls = [0]
        self._check(cancelled, expires)
        progress("Starting Meshy preview")
        preview_id = self._create({"mode": "preview", "prompt": prompt, "pose_mode": "a-pose",
                                   "target_formats": ["glb"]},
                                  expires=expires, cancelled=cancelled)
        self._poll(preview_id, "preview", expires=expires, cancelled=cancelled, progress=progress, polls=polls)
        self._check(cancelled, expires)
        progress("Starting Meshy texture refine")
        refine_id = self._create({"mode": "refine", "preview_task_id": preview_id,
                                  "target_formats": ["glb"]}, expires=expires, cancelled=cancelled)
        task = self._poll(refine_id, "refine", expires=expires, cancelled=cancelled,
                          progress=progress, polls=polls)
        urls = task.get("model_urls")
        url = self._artifact_url(urls.get("glb") if isinstance(urls, dict) else None)
        progress("Downloading textured character")
        data = self._request("get", url, expires=expires, cancelled=cancelled,
                             limit=MAX_GLB_BYTES, authenticated=False)
        return validate_glb(data)
