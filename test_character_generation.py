"""Offline Meshy protocol and GLB containment tests; never contacts Meshy."""

import json
from pathlib import Path
import struct
import tempfile
import unittest

import requests

from character_generation import (
    API_URL, MAX_GLB_BYTES, MAX_RESPONSE_BYTES, CharacterGenerationCancelled,
    MeshyCharacterGenerator, validate_glb,
)


def glb(document=None, binary=b"\x00\x00\x00\x00"):
    document = document or {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(binary)}]}
    encoded = json.dumps(document).encode()
    encoded += b" " * (-len(encoded) % 4)
    chunks = struct.pack("<I4s", len(encoded), b"JSON") + encoded
    if binary is not None:
        chunks += struct.pack("<I4s", len(binary), b"BIN\0") + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


class FakeResponse:
    def __init__(self, body=b"", status=200, headers=None):
        self.body, self.status_code = body, status
        self.headers = headers or {}
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def iter_content(self, chunk_size):
        for start in range(0, len(self.body), chunk_size):
            yield self.body[start:start + chunk_size]


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def _call(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def post(self, url, **kwargs):
        return self._call("post", url, **kwargs)

    def get(self, url, **kwargs):
        return self._call("get", url, **kwargs)


def response(document, status=200):
    return FakeResponse(json.dumps(document).encode(), status)


ASSET_URL = "https://assets.meshy.ai/user/tasks/2/output/model.glb?Expires=123&Signature=abc"


def success_responses(artifact=None, url=ASSET_URL):
    return [
        response({"result": "preview-1"}),
        response({"status": "PENDING", "progress": 0}),
        response({"status": "SUCCEEDED"}),
        response({"result": "refine-2"}),
        response({"status": "IN_PROGRESS", "progress": 50}),
        response({"status": "SUCCEEDED", "model_urls": {"glb": url}}),
        FakeResponse(glb() if artifact is None else artifact),
    ]


class MeshyGeneratorTests(unittest.TestCase):
    def generator(self, responses, **kwargs):
        transport = FakeTransport(responses)
        clock = kwargs.pop("clock", lambda: 0.0)
        sleep = kwargs.pop("sleep", lambda _: None)
        return MeshyCharacterGenerator("test-key", transport=transport, clock=clock,
                                       sleep=sleep, **kwargs), transport

    def test_preview_refine_download_and_bounded_requests(self):
        responses = success_responses()
        generator, transport = self.generator(responses)
        messages = []
        self.assertEqual(generator.generate("  a fully clothed adventurer  ", messages.append), glb())
        self.assertTrue(all(item.closed for item in responses))
        self.assertEqual([call[0] for call in transport.calls],
                         ["post", "get", "get", "post", "get", "get", "get"])
        self.assertEqual(transport.calls[0][2]["json"],
                         {"mode": "preview", "prompt": "a fully clothed adventurer",
                          "pose_mode": "a-pose", "target_formats": ["glb"]})
        self.assertEqual(transport.calls[3][2]["json"],
                         {"mode": "refine", "preview_task_id": "preview-1", "target_formats": ["glb"]})
        self.assertEqual(transport.calls[-1][1], ASSET_URL)
        self.assertEqual(transport.calls[-1][2]["headers"], {})
        for method, url, kwargs in transport.calls:
            self.assertTrue(kwargs["stream"])
            self.assertFalse(kwargs["allow_redirects"])
            self.assertLessEqual(kwargs["timeout"][0], 5)
            self.assertLessEqual(kwargs["timeout"][1], 30)
            if url.startswith(API_URL):
                self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertTrue(any("50%" in message for message in messages))

    def test_invalid_prompt_does_not_submit_paid_task(self):
        generator, transport = self.generator([])
        for prompt in (None, "", "  ", "x" * 801):
            with self.subTest(prompt=str(prompt)[:10]), self.assertRaises(ValueError):
                generator.generate(prompt)
        self.assertEqual(transport.calls, [])

    def test_failure_and_malformed_json_are_sanitized(self):
        cases = [
            ([response({"result": "preview-1"}), response({"status": "FAILED", "task_error": {"message": "secret"}})],
             "failed"),
            ([response({"result": "preview-1"}), response({"status": "WEIRD"})], "unknown"),
            ([FakeResponse(b"invalid")], "invalid JSON"),
            ([response({"result": "../oops"})], "invalid task ID"),
            ([FakeResponse(b"Bearer secret", 401)], "HTTP 401"),
            ([requests.Timeout("Bearer secret")], "connection failed"),
        ]
        for responses, expected in cases:
            with self.subTest(expected=expected):
                generator, transport = self.generator(responses)
                with self.assertRaises((ValueError, TimeoutError)) as context:
                    generator.generate("a character")
                self.assertIn(expected, str(context.exception))
                self.assertNotIn("secret", str(context.exception))
                self.assertEqual(transport.calls[0][0], "post")
                self.assertEqual(sum(method == "post" for method, _, _ in transport.calls), 1)

    def test_cancellation_stops_local_work_without_delete(self):
        generator, transport = self.generator([response({"result": "preview-1"}),
                                               response({"status": "PENDING"})])
        checks = iter([False, False, False, False, False, True])
        with self.assertRaises(CharacterGenerationCancelled) as context:
            generator.generate("a character", cancelled=lambda: next(checks))
        self.assertIn("may continue", str(context.exception))
        self.assertTrue(all(method in ("post", "get") for method, _, _ in transport.calls))
        self.assertEqual(sum(method == "post" for method, _, _ in transport.calls), 1)

    def test_poll_and_deadline_limits(self):
        generator, transport = self.generator([response({"result": "preview-1"}),
                                               response({"status": "PENDING"})], max_polls=1)
        with self.assertRaises(TimeoutError):
            generator.generate("a character")
        self.assertEqual(len(transport.calls), 2)
        ticks = iter([0, 0, 0, 10])
        generator, transport = self.generator([response({"result": "preview-1"})],
                                               clock=lambda: next(ticks), deadline=1)
        with self.assertRaises(TimeoutError):
            generator.generate("a character")
        self.assertEqual(len(transport.calls), 1)

    def test_response_limits_and_redirects(self):
        for bad in (FakeResponse(b"", headers={"Content-Length": str(MAX_RESPONSE_BYTES + 1)}),
                    FakeResponse(b"x" * (MAX_RESPONSE_BYTES + 1)),
                    FakeResponse(b"", status=302, headers={"Location": "http://localhost"})):
            generator, _ = self.generator([bad])
            with self.subTest(status=bad.status_code), self.assertRaises(ValueError):
                generator.generate("a character")
            self.assertTrue(bad.closed)

    def test_artifact_url_and_glb_rejection(self):
        bad_urls = ("http://assets.meshy.ai/model.glb", "https://localhost/model.glb",
                    "https://assets.meshy.ai.evil.test/model.glb",
                    "https://evil@assets.meshy.ai/model.glb",
                    "https://assets.meshy.ai:444/model.glb",
                    "https://assets.meshy.ai/model.glb#fragment",
                    "https://sub.assets.meshy.ai/model.glb")
        for url in bad_urls:
            with self.subTest(url=url):
                generator, transport = self.generator(success_responses(url=url))
                with self.assertRaisesRegex(ValueError, "unsafe GLB URL"):
                    generator.generate("a character")
                self.assertEqual(len(transport.calls), 6)
        generator, _ = self.generator(success_responses(artifact=b"not a GLB"))
        with self.assertRaises(ValueError):
            generator.generate("a character")

    def test_config_file_is_bounded_and_environment_wins(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "characters.env"
            path.write_text("# ignored\nexport MESHY_API_KEY='file-key'\nUNRELATED=ignored\n")
            self.assertEqual(MeshyCharacterGenerator.from_env(config_path=path,
                             environ={})._api_key, "file-key")
            self.assertEqual(MeshyCharacterGenerator.from_env(config_path=path,
                             environ={"MESHY_API_KEY": "env-key"})._api_key, "env-key")
            path.write_text("MESHY_API_KEY=$(danger)\n")
            with self.assertRaises(ValueError):
                MeshyCharacterGenerator.from_env(config_path=path, environ={})
            path.write_text("x" * 16_385)
            with self.assertRaisesRegex(ValueError, "too large"):
                MeshyCharacterGenerator.from_env(config_path=path, environ={})


class GLBValidationTests(unittest.TestCase):
    def test_valid_embedded_buffer_and_images(self):
        document = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": 4}],
                    "bufferViews": [{"buffer": 0, "byteLength": 4}],
                    "images": [{"bufferView": 0, "mimeType": "image/png"}]}
        data = glb(document)
        self.assertIs(validate_glb(data), data)
        data = glb({"asset": {"version": "2.0"}, "buffers": [{"byteLength": 1, "uri": "data:application/octet-stream;base64,AA=="}],
                    "images": [{"uri": "data:image/png;base64,AA=="}]}, binary=None)
        self.assertEqual(validate_glb(data), data)

    def test_rejects_malformed_or_external_glb(self):
        valid = glb()
        bad = [b"bad", valid[:-1], valid[:8] + struct.pack("<I", len(valid) + 4) + valid[12:],
               valid[:4] + struct.pack("<I", 1) + valid[8:],
               glb({"asset": {"version": "2.0"}, "buffers": [{"byteLength": 4, "uri": "https://evil/b.bin"}]}, binary=None),
               glb({"asset": {"version": "2.0"}, "images": [{"uri": "../texture.png"}]}, binary=None),
               glb({"asset": {"version": "2.0"}, "buffers": [{"byteLength": 100}]}, binary=b"\0" * 4),
               glb({"asset": {"version": "2.0"}, "buffers": [{"byteLength": 4}],
                    "bufferViews": [{"buffer": 0, "byteLength": 8}]}),
               b"\0" * (MAX_GLB_BYTES + 1)]
        for blob in bad:
            with self.subTest(length=len(blob)), self.assertRaises(ValueError):
                validate_glb(blob)


if __name__ == "__main__":
    unittest.main()
