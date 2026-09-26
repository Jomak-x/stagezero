"""Observable prompt-assistant behavior, with no Ollama or motion calls."""

import json
import unittest

from prompt_assistant import (
    MAX_PROMPT_CHARS,
    OllamaPromptProvider,
    needs_clarification,
    refine_prompt,
)


class PromptAssistantTests(unittest.TestCase):
    def test_bare_turn_back_asks_before_provider_and_keeps_current_angle_uninterpreted(self):
        calls = []

        def unavailable(*_args):
            calls.append(True)
            return {"ready": True, "refined_prompt": "wrong", "explanation": "wrong", "questions": []}

        result = refine_prompt("turn back currently45right", provider=unavailable)
        self.assertFalse(result.ready)
        self.assertEqual(result.refined_prompt, "")
        self.assertEqual(result.questions[0].id, "turn_back_meaning")
        self.assertIn("180°", result.questions[0].options[0])
        self.assertTrue(needs_clarification("turn back currently45right"))
        self.assertEqual(calls, [])

    def test_turkish_turn_back_asks_in_turkish_and_accepts_180_answer(self):
        prompt = "Şu an 45 derece sağa bakıyor, geri dön"
        first = refine_prompt(prompt, offline=True)
        self.assertFalse(first.ready)
        self.assertIn("hangi hareketi", first.questions[0].text)
        second = refine_prompt(prompt, {"turn_back_meaning": first.questions[0].options[0]}, offline=True)
        self.assertTrue(second.ready)
        self.assertIn("180°", second.refined_prompt)
        self.assertIn("45", second.refined_prompt)
        self.assertIn("to the right", second.refined_prompt)
        self.assertNotIn("sağa", second.refined_prompt)
        self.assertNotIn("225", second.refined_prompt)
        self.assertIn("Yerel", second.explanation)

    def test_explicit_turkish_turn_is_translated_offline(self):
        result = refine_prompt("Yerinde 180 derece geri dön", offline=True)
        self.assertTrue(result.ready)
        self.assertIn("turn 180° in place", result.refined_prompt)
        self.assertNotIn("geri dön", result.refined_prompt)
        self.assertNotIn("derece", result.refined_prompt)

    def test_requested_turkish_turn_side_is_not_called_current_heading(self):
        for side, english in (("sola", "left"), ("sağa", "right")):
            with self.subTest(side=side):
                result = refine_prompt(f"Yerinde 180 derece {side} geri dön", offline=True)
                self.assertTrue(result.ready)
                self.assertIn(f"Turn direction: 180° to the {english}.", result.refined_prompt)
                self.assertNotIn("Current facing direction", result.refined_prompt)

    def test_turkish_current_heading_and_requested_turn_remain_distinct(self):
        for separator in (", ", " "):
            with self.subTest(separator=separator):
                prompt = f"Şu an 45 derece sağa bakıyor{separator}yerinde 180 derece sola geri dön"
                result = refine_prompt(prompt, offline=True)
                self.assertTrue(result.ready)
                self.assertIn("Current facing direction: 45° to the right.", result.refined_prompt)
                self.assertIn("Turn direction: 180° to the left.", result.refined_prompt)
                self.assertNotIn("Current facing direction: 180°", result.refined_prompt)

    def test_explicit_180_in_place_skips_redundant_question(self):
        prompt = "Turn back 180° in place, slowly, from the current 45° right heading"
        self.assertFalse(needs_clarification(prompt))
        result = refine_prompt(prompt, offline=True)
        self.assertTrue(result.ready)
        self.assertIn("180° in place", result.refined_prompt)
        self.assertIn("45° right", result.refined_prompt)
        self.assertTrue(needs_clarification("Turn back after 180 seconds"))
        self.assertTrue(needs_clarification("turn back currently facing180degrees right"))
        self.assertTrue(needs_clarification("currently facing 180° right, turn back"))

    def test_freeform_answer_is_kept_without_inventing_angle(self):
        prompt = "Turn back slowly while holding the cup in the left hand"
        result = refine_prompt(prompt, {"turn_back_meaning": "Face the door behind me"}, offline=True)
        self.assertTrue(result.ready)
        self.assertIn("Face the door behind me", result.refined_prompt)
        self.assertIn("slowly", result.refined_prompt)
        self.assertIn("left hand", result.refined_prompt)
        self.assertNotIn("180", result.refined_prompt)

    def test_generic_turn_requires_direction_and_uses_answer(self):
        first = refine_prompt("Turn slowly", offline=True)
        self.assertFalse(first.ready)
        self.assertEqual(first.questions[0].id, "turn_direction")
        second = refine_prompt("Turn slowly", {"turn_direction": "left by 90 degrees"}, offline=True)
        self.assertTrue(second.ready)
        self.assertIn("left by 90 degrees", second.refined_prompt)
        self.assertFalse(needs_clarification(second.refined_prompt))
        self.assertFalse(needs_clarification("Turn off the light"))

    def test_resolved_turn_and_reference_do_not_retrigger_guard(self):
        for meaning in ("Return to the previous heading", "Return to the previous position",
                        "Turn 180° in place to face the opposite direction"):
            result = refine_prompt("Turn back slowly", {"turn_back_meaning": meaning}, offline=True)
            self.assertTrue(result.ready)
            self.assertFalse(needs_clarification(result.refined_prompt))
        resolved = refine_prompt("Walk over there", {"reference": "the marked doorway"}, offline=True)
        self.assertFalse(needs_clarification(resolved.refined_prompt))
        self.assertIn("Walk to the marked doorway", resolved.refined_prompt)

    def test_freeform_answer_modifiers_are_preserved(self):
        result = refine_prompt("Turn back from the current 45° right heading",
                               {"turn_back_meaning": "180° to the left, slowly"}, offline=True)
        self.assertTrue(result.ready)
        self.assertIn("left", result.refined_prompt)
        self.assertIn("slowly", result.refined_prompt)
        self.assertIn("45° right", result.refined_prompt)

    def test_injected_model_receives_prior_answers_and_history(self):
        seen = {}

        def provider(system, user):
            seen["system"] = system
            seen["user"] = json.loads(user)
            return {"ready": True, "refined_prompt": "A person waves gently with the left hand.",
                    "explanation": "I kept the requested hand and pace.", "questions": []}

        result = refine_prompt("Wave gently", {"hand": "left"}, provider=provider,
                               history=[{"question": "Which hand?", "answer": "left"}])
        self.assertTrue(result.ready)
        self.assertEqual(result.source, "injected")
        self.assertEqual(seen["user"]["answers"], {"hand": "left"})
        self.assertEqual(seen["user"]["history"][0]["answer"], "left")
        self.assertIn("Never invent", seen["system"])

    def test_yes_answer_to_model_question_is_preserved(self):
        seen = {}

        def provider(_system, user):
            seen.update(json.loads(user))
            return {"ready": True, "refined_prompt": "A person waves.",
                    "explanation": "Clarified the motion.", "questions": []}

        result = refine_prompt("Wave", {"include_pause": "yes"}, provider=provider)
        self.assertTrue(result.ready)
        self.assertEqual(seen["answers"]["include_pause"], "yes")

    def test_model_can_ask_three_questions_and_invalid_output_falls_back(self):
        question = lambda index: {"id": f"q{index}", "text": f"Question {index}?", "options": ["one", "two"]}
        good = lambda *_: {"ready": False, "refined_prompt": "", "explanation": "Need detail.",
                           "questions": [question(index) for index in range(3)]}
        result = refine_prompt("Wave", provider=good)
        self.assertFalse(result.ready)
        self.assertEqual(len(result.questions), 3)
        self.assertEqual(result.source, "injected")

        bad = lambda *_: {"ready": True, "refined_prompt": "A person waves.",
                          "explanation": "Changed.", "questions": [question(1)]}
        fallback = refine_prompt("Wave gently", provider=bad)
        self.assertTrue(fallback.ready)
        self.assertEqual(fallback.source, "offline")
        self.assertEqual(fallback.refined_prompt, "Wave gently")
        self.assertIn("invalid response", fallback.warning)

    def test_model_cannot_drop_or_invent_angle_or_flip_side(self):
        def provider(_system, _user):
            return {"ready": True, "refined_prompt": "A person turns 45° to the right.",
                    "explanation": "Clarified turn.", "questions": []}

        source = "Turn 180° in place from a 45° right heading"
        result = refine_prompt(source, provider=provider)
        self.assertEqual(result.source, "offline")
        self.assertEqual(result.refined_prompt, source)
        self.assertIn("invalid response", result.warning)

        def flipped(_system, _user):
            return {"ready": True, "refined_prompt": "A person turns 180° in place from a 45° left heading.",
                    "explanation": "Clarified turn.", "questions": []}

        result = refine_prompt(source, provider=flipped)
        self.assertEqual(result.source, "offline")
        self.assertIn("right", result.refined_prompt)

    def test_offline_never_calls_injected_provider(self):
        calls = []

        def forbidden(*_args):
            calls.append(True)
            return {"ready": True, "refined_prompt": "wrong", "explanation": "wrong", "questions": []}

        result = refine_prompt("Wave gently", provider=forbidden, offline=True)
        self.assertTrue(result.ready)
        self.assertEqual(result.source, "offline")
        self.assertEqual(calls, [])

    def test_input_and_output_limits_do_not_silently_drop_constraints(self):
        with self.assertRaises(ValueError):
            refine_prompt("x" * (MAX_PROMPT_CHARS + 1), offline=True)
        result = refine_prompt("Turn back " + "slowly " * 68,
                               {"turn_back_meaning": "Turn 180° in place"}, offline=True)
        self.assertFalse(result.ready)
        self.assertEqual(result.questions[0].id, "shorten")

    def test_missing_reference_stays_unready_until_answered(self):
        first = refine_prompt("Walk over there", offline=True)
        self.assertFalse(first.ready)
        self.assertEqual(first.questions[0].id, "reference")
        second = refine_prompt("Walk over there", {"reference": "the marked doorway"}, offline=True)
        self.assertTrue(second.ready)
        self.assertIn("marked doorway", second.refined_prompt)


class _Response:
    status_code = 200

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def iter_content(self, _chunk_size):
        yield self.body


class _Transport:
    def __init__(self, body):
        self.body = body
        self.request = None

    def post(self, url, **kwargs):
        self.request = (url, kwargs)
        return _Response(self.body)


class OllamaProviderTests(unittest.TestCase):
    def test_local_transport_uses_bounded_timeout_and_schema(self):
        document = {"ready": True, "refined_prompt": "A person waves.",
                    "explanation": "Clearer subject.", "questions": []}
        envelope = {"message": {"content": json.dumps(document)}}
        transport = _Transport(json.dumps(envelope).encode())
        result = OllamaPromptProvider(model="test-model", transport=transport)("system", "user")
        self.assertEqual(result, document)
        url, kwargs = transport.request
        self.assertEqual(url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(kwargs["timeout"], (3, 55))
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["json"]["model"], "test-model")
        self.assertEqual(kwargs["json"]["format"]["properties"]["questions"]["maxItems"], 3)


if __name__ == "__main__":
    unittest.main()
