import base64
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image

from neon_character_reference import (
    NeonCharacterReference,
    NeonCharacterReferenceCancelled,
)


def image_bytes():
    out = io.BytesIO()
    Image.new("RGB", (256, 384), "#ddd9d4").save(out, format="PNG")
    return out.getvalue()


class FakeResponse:
    def __init__(self, events, status=200):
        self.status_code = status
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def iter_content(self, _size):
        for event in self.events:
            raw = ("data: " + json.dumps(event) + "\n\n").encode()
            yield raw[:17]
            yield raw[17:]


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return self.responses.pop(0)


class NeonCharacterReferenceTests(unittest.TestCase):
    def gateway(self, events, status=200):
        design = {"type": "response.output_item.done", "item": {"type": "message", "content": [
            {"type": "output_text", "text": "A visually distinct explorer in a navy flight suit with warm brass trim."}]}}
        transport = FakeTransport(FakeResponse([design]), FakeResponse(events, status))
        return NeonCharacterReference("https://branch.example", "secret", transport=transport), transport

    def test_private_character_model_overrides_and_explicit_arguments(self):
        with tempfile.TemporaryDirectory() as folder:
            module_path = Path(folder) / "object_generation.py"
            private = Path(folder) / ".runtime" / "objects.env"
            private.parent.mkdir()
            private.write_text("NEON_AI_GATEWAY_BASE_URL=https://branch.example\n"
                               "NEON_AI_GATEWAY_TOKEN=test-secret\n"
                               "STAGEZERO_CHARACTER_IMAGE_MODEL=gpt-image-choice\n"
                               "STAGEZERO_CHARACTER_DESIGN_MODEL=gpt-design-choice\n")
            with mock.patch("object_generation.__file__", str(module_path)), mock.patch.dict(
                    os.environ, {}, clear=True):
                configured = NeonCharacterReference.from_env()
                explicit = NeonCharacterReference.from_env(model="gpt-explicit-choice")
                with mock.patch.dict(os.environ, {"STAGEZERO_CHARACTER_IMAGE_MODEL": "gpt-env-choice"}):
                    environment = NeonCharacterReference.from_env()
        self.assertEqual(configured.model, "gpt-image-choice")
        self.assertEqual(configured.design_model, "gpt-design-choice")
        self.assertEqual(explicit.model, "gpt-explicit-choice")
        self.assertEqual(explicit.design_model, "gpt-design-choice")
        self.assertEqual(environment.model, "gpt-env-choice")

    def test_generates_png_using_responses_image_tool(self):
        expected = image_bytes()
        event = {"type": "response.output_item.done", "item": {
            "type": "image_generation_call", "result": base64.b64encode(expected).decode()}}
        gateway, transport = self.gateway([event])
        progress = []
        self.assertEqual(gateway.generate("space explorer", progress=progress.append), expected)
        self.assertEqual(len(transport.requests), 2)
        design_url, design_options = transport.requests[0]
        self.assertEqual(design_url, "https://branch.example/openai/v1/responses")
        self.assertEqual(design_options["json"]["model"], "gpt-6-astra")
        self.assertEqual(design_options["json"]["input"][-1]["content"], "space explorer")
        self.assertEqual(design_options["json"]["reasoning"]["effort"], "medium")
        url, options = transport.requests[1]
        self.assertEqual(url, "https://branch.example/openai/v1/responses")
        self.assertEqual(options["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(options["json"]["model"], "gpt-6-astra")
        self.assertEqual(options["json"]["tools"][0]["type"], "image_generation")
        self.assertEqual(options["json"]["tools"][0]["action"], "generate")
        self.assertEqual(options["json"]["tools"][0]["quality"], "high")
        self.assertEqual(options["json"]["tool_choice"], {"type": "image_generation"})
        self.assertTrue(options["json"]["stream"])
        self.assertIn("full-body", options["json"]["input"])
        self.assertIn("navy flight suit", options["json"]["input"])
        self.assertIn("A-pose", options["json"]["input"])
        self.assertIn("mouth closed", options["json"]["input"])
        self.assertIn("fingers", options["json"]["input"])
        self.assertIn("photographic-quality", options["json"]["input"])
        self.assertIn("fine fabric weave", options["json"]["input"])
        self.assertIn("natural muscle transitions", options["json"]["input"])
        self.assertNotIn("not a photograph", options["json"]["input"])
        self.assertNotIn("3D maquette", options["json"]["input"])
        self.assertTrue(options["stream"])
        self.assertFalse(options["allow_redirects"])
        self.assertEqual(progress[-1], "Character reference ready")

    def test_errors_are_controlled_and_do_not_expose_response(self):
        bad = {"type": "response.output_item.done", "item": {
            "type": "image_generation_call", "result": base64.b64encode(b"not-png").decode()}}
        for events, status in (([], 403), ([{"type": "response.failed", "response": {
            "error": {"message": "private remote detail"}}}], 200), ([bad], 200), ([], 200)):
            with self.subTest(events=events, status=status):
                gateway, _ = self.gateway(events, status)
                with self.assertRaises(ValueError) as error:
                    gateway.generate("space explorer")
                self.assertNotIn("private remote detail", str(error.exception))
                self.assertNotIn("secret", str(error.exception))

    def test_completed_response_can_supply_image(self):
        expected = image_bytes()
        event = {"type": "response.completed", "response": {"output": [
            {"type": "message", "content": []},
            {"type": "image_generation_call", "result": base64.b64encode(expected).decode()},
        ]}}
        gateway, _ = self.gateway([event])
        self.assertEqual(gateway.generate("space explorer"), expected)

    def test_failed_image_call_does_not_infer_reason_from_assistant_text(self):
        failed = {"type": "response.output_item.done", "item": {
            "type": "image_generation_call", "status": "failed", "result": None}}
        gateway, _ = self.gateway([failed, {"type": "response.completed", "response": {
            "output": [{"type": "message", "content": [{"type": "output_text",
                "text": "Sorry, I can't generate that copyrighted character. private refusal detail"}]}]}}])
        with self.assertRaisesRegex(ValueError, "Neon image generation failed") as error:
            gateway.generate("space explorer")
        self.assertNotIn("private refusal detail", str(error.exception))

    def test_failed_image_call_without_refusal_has_generic_error(self):
        failed = {"type": "response.output_item.done", "item": {
            "type": "image_generation_call", "status": "failed", "result": None}}
        gateway, _ = self.gateway([failed])
        with self.assertRaisesRegex(ValueError, "Neon image generation failed"):
            gateway.generate("space explorer")

    def test_cancellation_prevents_request(self):
        gateway, transport = self.gateway([])
        with self.assertRaises(NeonCharacterReferenceCancelled):
            gateway.generate("space explorer", cancelled=lambda: True)
        self.assertEqual(transport.requests, [])

    def test_design_failure_stops_before_image_request(self):
        transport = FakeTransport(FakeResponse([{"type": "response.failed", "response": {
            "error": {"message": "private remote detail"}}}]))
        gateway = NeonCharacterReference("https://branch.example", "secret", transport=transport)
        with self.assertRaises(ValueError) as error:
            gateway.generate("space explorer")
        self.assertEqual(len(transport.requests), 1)
        self.assertNotIn("private remote detail", str(error.exception))

    def test_cancellation_between_design_and_image(self):
        gateway, transport = self.gateway([])
        cancelled = lambda: len(transport.requests) > 0
        with self.assertRaises(NeonCharacterReferenceCancelled):
            gateway.generate("space explorer", cancelled=cancelled)
        self.assertEqual(len(transport.requests), 1)

    def test_base_url_must_be_bare_https_host(self):
        for url in ("http://branch.example", "https://u:p@branch.example",
                    "https://branch.example/v1", "https://branch.example?token=x"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                NeonCharacterReference(url, "secret")


if __name__ == "__main__":
    unittest.main()
