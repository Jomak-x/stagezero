"""Observable prompt-assistant behavior, with no Ollama or motion calls."""

import json
import unittest
from unittest.mock import patch

from prompt_assistant import (
    GatewayPromptProvider,
    MAX_PROMPT_CHARS,
    MAX_RESPONSE_BYTES,
    OllamaPromptProvider,
    needs_clarification,
    refine_prompt,
)


class PromptAssistantTests(unittest.TestCase):
    def test_big_jump_asks_one_material_question_without_inventing_measurements(self):
        calls = []
        result = refine_prompt("make a nice big jump", provider=lambda *_: calls.append(True))
        self.assertFalse(result.ready)
        self.assertEqual([question.id for question in result.questions], ["jump_emphasis"])
        self.assertEqual(calls, [])
        self.assertTrue(needs_clarification("make a nice big jump"))
        self.assertFalse(needs_clarification("Make a high jump straight up, landing in place"))

    def test_big_jump_clarifications_accept_equivalent_vertical_and_landing_words(self):
        for refined in ("Make a high vertical jump and land in place.",
                        "Make a very high vertical jump and land in the same spot.",
                        "Make a notably strong and impressively high vertical jump and land in the same spot.",
                        "Make an exceptionally high vertical jump and land in the same spot.",
                        "Make one high vertical jump and land in the same spot.",
                        "Jump high straight upward once and land where you started."):
            with self.subTest(refined=refined):
                result = refine_prompt("make a nice big jump", {
                    "direction": "Straight up, landing in place", "size": "Height"},
                    provider=lambda *_: {"ready": True, "refined_prompt": refined,
                                         "explanation": "Kept the clarified jump.", "questions": []})
                self.assertTrue(result.ready, result.warning)
                self.assertIsNone(result.diagnostics.issue)

    def test_turkish_one_forward_jump_preserves_action_direction_and_count(self):
        for refined in ("Make one forward jump.", "Jump forwards once."):
            with self.subTest(refined=refined):
                result = refine_prompt("Öne doğru bir kere zıpla", provider=lambda *_: {
                    "ready": True, "refined_prompt": refined,
                    "explanation": "Yön ve tekrar korundu.", "questions": []})
                self.assertTrue(result.ready, result.warning)

    def test_equivalent_negative_travel_and_landing_pronouns_are_accepted(self):
        cases = (
            ("Jump once in place without moving forward.",
             "Make one vertical jump and land in the same spot."),
            ("Jump once in place without moving forwards.",
             "Make one vertical jump and land back where it started."),
            ("Jump once in place without moving backward.",
             "Make one vertical jump and land back where it started."),
        )
        for original, refined in cases:
            with self.subTest(original=original):
                result = refine_prompt(original, provider=lambda *_: {
                    "ready": True, "refined_prompt": refined,
                    "explanation": "Kept the landing location.", "questions": []})
                self.assertTrue(result.ready, result.warning)

    def test_direction_synonyms_do_not_accept_negated_or_changed_direction(self):
        cases = (
            ("Öne doğru bir kere zıpla", "Jump once without moving forwards."),
            ("Öne doğru bir kere zıpla", "Jump backwards once."),
            ("Jump once in place without moving forwards.",
             "Make one forward jump and land back where it started."),
            ("Jump once in place.", "Jump once and do not land back where it started."),
        )
        for original, refined in cases:
            with self.subTest(original=original):
                result = refine_prompt(original, provider=lambda *_: {
                    "ready": True, "refined_prompt": refined,
                    "explanation": "Changed.", "questions": []})
                self.assertFalse(result.ready)
                self.assertEqual(result.diagnostics.issue.category, "constraint")

    def test_real_constraint_failure_records_exact_rule_and_never_retries(self):
        calls = []
        document = {"ready": True, "refined_prompt": "Jump twice forward for 2 meters.",
                    "explanation": "Changed jump.", "questions": []}
        def provider(*_args):
            calls.append(True)
            return document
        result = refine_prompt("Öne doğru bir kere zıpla", provider=provider)
        self.assertFalse(result.ready)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.diagnostics.original_prompt, "Öne doğru bir kere zıpla")
        self.assertEqual(result.diagnostics.issue.category, "constraint")
        self.assertEqual(result.diagnostics.issue.field, "repetition_count")
        self.assertIn("repetition count", result.diagnostics.issue.message)
        self.assertEqual(result.diagnostics.attempts[0].parsed_response, document)
        self.assertEqual(json.loads(result.diagnostics.attempts[0].raw_response), document)

    def test_big_jump_cannot_invent_quantity_drop_height_or_add_travel(self):
        for refined in ("Make a high vertical jump 2 meters up and land in place.",
                        "Make a high vertical jump for 3 seconds and land in place.",
                        "Make a low vertical jump and land in place.",
                        "Make a high jump forward and land in place.",
                        "Make a high jump and land somewhere else.",
                        "Make one vertical jump, not high, and land in place.",
                        "Make one high vertical jump and do not land in the same spot."):
            with self.subTest(refined=refined):
                result = refine_prompt("make a nice big jump", {
                    "direction": "Straight up, landing in place", "size": "Height"},
                    provider=lambda *_: {"ready": True, "refined_prompt": refined,
                                         "explanation": "Clarified.", "questions": []})
                self.assertFalse(result.ready)
                self.assertEqual(result.diagnostics.issue.category, "constraint")

    def test_malformed_json_gets_one_repair_with_exact_feedback_and_original_context(self):
        requests = []
        raw = '{"ready":true,"refined_prompt":'
        def provider(_system, user):
            requests.append(json.loads(user))
            if len(requests) == 1:
                return raw
            return {"ready": True, "refined_prompt": "Wave gently with the left hand.",
                    "explanation": "Kept your action.", "questions": []}
        result = refine_prompt("Wave gently", {"hand": "left"}, provider=provider)
        self.assertTrue(result.ready, result.warning)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[1]["original_prompt"], "Wave gently")
        self.assertEqual(requests[1]["answers"], {"hand": "left"})
        issue = result.diagnostics.attempts[0].issue
        self.assertEqual(issue.category, "format")
        self.assertEqual(requests[1]["repair"]["feedback"], issue.message)
        self.assertEqual(requests[1]["repair"]["previous_response"], raw)
        self.assertEqual(result.diagnostics.attempts[0].raw_response, raw)
        self.assertIsNone(result.diagnostics.attempts[0].parsed_response)
        self.assertIsNone(result.diagnostics.issue)

    def test_response_field_failure_is_bounded_and_distinct_from_constraints(self):
        calls = []
        def provider(*_args):
            calls.append(True)
            return {"ready": True, "refined_prompt": "Wave.", "explanation": "", "questions": []}
        result = refine_prompt("Wave", provider=provider)
        self.assertFalse(result.ready)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.diagnostics.issue.category, "format")
        self.assertEqual(result.diagnostics.issue.field, "explanation")
        self.assertEqual(len(result.diagnostics.attempts), 2)

    def test_cancel_between_format_failure_and_repair_prevents_second_call(self):
        calls = []
        def provider(*_args):
            calls.append(True)
            return 'broken'
        result = refine_prompt("Wave", provider=provider, should_cancel=lambda: bool(calls))
        self.assertFalse(result.ready)
        self.assertEqual(calls, [True])
        self.assertEqual(result.diagnostics.issue.category, "cancelled")

    def test_transport_failure_never_enters_format_repair(self):
        calls = []
        def provider(*_args):
            calls.append(True)
            raise TimeoutError("timeout")
        result = refine_prompt("Wave", provider=provider)
        self.assertFalse(result.ready)
        self.assertEqual(calls, [True])
        self.assertEqual(result.diagnostics.issue.category, "transport")
        self.assertFalse(result.diagnostics.issue.retryable)

    def test_repair_still_rejects_a_semantically_changed_prompt(self):
        calls = []
        def provider(*_args):
            calls.append(True)
            return ('broken' if len(calls) == 1 else {
                "ready": True, "refined_prompt": "Jump backward once.",
                "explanation": "Changed.", "questions": []})
        result = refine_prompt("Öne doğru bir kere zıpla", provider=provider)
        self.assertFalse(result.ready)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.diagnostics.issue.category, "constraint")
        self.assertEqual(result.diagnostics.issue.field, "jump_direction")
        self.assertEqual(result.refined_prompt, "")

    def test_cancellation_before_provider_does_not_call_model(self):
        calls = []
        result = refine_prompt("Wave", provider=lambda *_: calls.append(True), should_cancel=lambda: True)
        self.assertFalse(result.ready)
        self.assertEqual(calls, [])
        self.assertEqual(result.diagnostics.attempts, ())
        self.assertEqual(result.diagnostics.issue.code, "cancelled")

    def test_diagnostics_keep_input_whitespace_and_sanitize_transport_exceptions(self):
        def provider(*_args):
            raise RuntimeError("authorization=secret-value https://private-token.example")
        result = refine_prompt(" Wave  gently ", provider=provider)
        self.assertEqual(result.diagnostics.original_prompt, " Wave  gently ")
        self.assertNotIn("secret-value", str(result))
        self.assertNotIn("private-token", str(result))

    def test_opt_in_development_log_contains_exact_attempt_diagnostics(self):
        document = {"ready": True, "refined_prompt": "Jump backward once.",
                    "explanation": "Changed.", "questions": []}
        with self.assertLogs("prompt_assistant", level="DEBUG") as logs:
            result = refine_prompt("Öne doğru bir kere zıpla", provider=lambda *_: document)
        record = json.loads(logs.records[0].getMessage().split("prompt_refinement ", 1)[1])
        self.assertEqual(record["original_prompt"], "Öne doğru bir kere zıpla")
        self.assertEqual(record["attempts"][0]["parsed_response"], document)
        self.assertEqual(json.loads(record["attempts"][0]["raw_response"]), document)
        self.assertEqual(record["issue"]["message"], result.diagnostics.issue.message)

    def test_person_article_is_not_treated_as_a_singular_jump_count(self):
        for refined in ("A person should jump forward.", "A tall person will jump forward.",
                        "A character can jump forward.", "A robot must jump forward."):
            with self.subTest(refined=refined):
                result = refine_prompt("Jump forward", provider=lambda *_: {
                    "ready": True, "refined_prompt": refined,
                    "explanation": "Clarified subject.", "questions": []})
                self.assertTrue(result.ready, result.warning)

    def test_explicit_low_jump_height_cannot_be_replaced_with_high(self):
        result = refine_prompt("Make a low jump.", provider=lambda *_: {
            "ready": True, "refined_prompt": "Make a high jump.",
            "explanation": "Changed height.", "questions": []})
        self.assertFalse(result.ready)
        self.assertEqual(result.diagnostics.issue.field, "jump_emphasis")

    def test_oversized_multibyte_response_is_bounded_and_not_retried(self):
        calls = []
        def provider(*_args):
            calls.append(True)
            return "ş" * MAX_RESPONSE_BYTES
        result = refine_prompt("Wave", provider=provider)
        self.assertFalse(result.ready)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.diagnostics.issue.code, "response_too_large")
        self.assertLessEqual(len(result.diagnostics.attempts[0].raw_response.encode("utf-8")), MAX_RESPONSE_BYTES)

    def test_explicit_repeat_counts_are_not_weakened_by_singular_jump_words(self):
        for original, refined in (("Jump forward twice", "Make one forward jump."),
                                  ("Jump forward twice", "Make a very long forward jump."),
                                  ("Make a very high vertical jump twice", "Make a very high vertical jump."),
                                  ("Öne doğru bir kere zıpla", "Jump forward twice.")):
            with self.subTest(original=original):
                result = refine_prompt(original, provider=lambda *_: {
                    "ready": True, "refined_prompt": refined,
                    "explanation": "Changed.", "questions": []})
                self.assertFalse(result.ready)
                self.assertEqual(result.diagnostics.issue.field, "repetition_count")

    def test_big_jump_selected_emphasis_is_used_without_another_question(self):
        prompt = "make a nice big jump"
        first = refine_prompt(prompt, offline=True)
        for answer, refined in zip(first.questions[0].options, (
                "Make one high vertical jump and land in the same spot.",
                "Make one long forward jump.", "Make one high, long forward jump.")):
            with self.subTest(answer=answer):
                result = refine_prompt(prompt, {"jump_emphasis": answer}, provider=lambda *_: {
                    "ready": True, "refined_prompt": refined,
                    "explanation": "Kept the selected emphasis.", "questions": []})
                self.assertTrue(result.ready, result.warning)
                self.assertEqual(len(result.diagnostics.attempts), 1)

    def test_horizontal_route_constraint_is_preserved_after_vertical_synonym_fix(self):
        result = refine_prompt("Jump straight forward once", provider=lambda *_: {
            "ready": True, "refined_prompt": "Jump forward once in a circle.",
            "explanation": "Changed.", "questions": []})
        self.assertFalse(result.ready)
        self.assertEqual(result.diagnostics.issue.field, "route_shape")

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
        self.assertFalse(fallback.ready)
        self.assertEqual(fallback.refined_prompt, "")
        self.assertIn("invalid", fallback.warning)

    def test_model_cannot_drop_or_invent_angle_or_flip_side(self):
        def provider(_system, _user):
            return {"ready": True, "refined_prompt": "A person turns 45° to the right.",
                    "explanation": "Clarified turn.", "questions": []}

        source = "Turn 180° in place from a 45° right heading"
        result = refine_prompt(source, provider=provider)
        self.assertEqual(result.source, "injected")
        self.assertFalse(result.ready)
        self.assertEqual(result.refined_prompt, "")
        self.assertIn("constraint", result.warning)

        def flipped(_system, _user):
            return {"ready": True, "refined_prompt": "A person turns 180° in place from a 45° left heading.",
                    "explanation": "Clarified turn.", "questions": []}

        result = refine_prompt(source, provider=flipped)
        self.assertFalse(result.ready)
        self.assertEqual(result.refined_prompt, "")

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

    def test_turkish_destination_is_asked_and_answered_without_guessing(self):
        prompt = "Oraya yürü"
        first = refine_prompt(prompt, offline=True)
        self.assertFalse(first.ready)
        self.assertEqual(first.questions[0].id, "reference")
        self.assertIn("Nereye", first.questions[0].text)
        answered = refine_prompt(prompt, {"reference": "kırmızı kapıya"}, offline=True)
        self.assertTrue(answered.ready)
        self.assertIn("kırmızı kapıya", answered.refined_prompt)
        grounded = refine_prompt(prompt, scene_context={"reference_target": "the red door"}, offline=True)
        self.assertTrue(grounded.ready)
        self.assertIn("red door", grounded.refined_prompt)
        self.assertFalse(needs_clarification(answered.refined_prompt))

        def translated(*_args):
            return {"ready": True, "refined_prompt": "Walk to the red door.",
                    "explanation": "Kept the target.", "questions": []}

        translated_result = refine_prompt(prompt, {"reference": "kırmızı kapıya"}, provider=translated)
        self.assertTrue(translated_result.ready)

    def test_lateral_jump_asks_reference_unless_explicit_or_grounded(self):
        prompt = "Sağa zıpla"
        first = refine_prompt(prompt, offline=True)
        self.assertFalse(first.ready)
        self.assertEqual(first.questions[0].id, "direction_reference")
        self.assertIn("ekranda", first.questions[0].text)
        answered = refine_prompt(prompt, {"direction_reference": "Karakterin sağına"}, offline=True)
        self.assertTrue(answered.ready)
        self.assertIn("Karakterin sağına", answered.refined_prompt)
        self.assertFalse(needs_clarification("Öne doğru zıpla"))
        self.assertFalse(needs_clarification("Jump to the character's right"))
        self.assertFalse(needs_clarification(prompt, {"direction_reference": "character"}))
        context_result = refine_prompt("Jump right", scene_context={"direction_reference": "character"},
                                       offline=True)
        self.assertTrue(context_result.ready)
        self.assertIn("character's right", context_result.refined_prompt)
        self.assertFalse(needs_clarification(context_result.refined_prompt))
        self.assertTrue(needs_clarification("The character should jump right"))
        framed = refine_prompt("Jump right", {"direction_reference": "Screen right"},
                               provider=lambda *_: {"ready": True,
                                   "refined_prompt": "Jump right relative to the screen.",
                                   "explanation": "Clarified reference.", "questions": []})
        self.assertTrue(framed.ready)
        left = refine_prompt("Sola zıpla", offline=True)
        self.assertEqual(left.questions[0].options,
                         ("Karakterin soluna", "Ekranda sola"))

    def test_unknown_street_end_asks_target_or_distance(self):
        prompt = "Sokağın sonuna yürü"
        first = refine_prompt(prompt, offline=True)
        self.assertFalse(first.ready)
        self.assertEqual(first.questions[0].id, "street_target")
        self.assertIn("hedef", first.questions[0].text)
        answered = refine_prompt(prompt, {"street_target": "yaklaşık 20 metre"}, offline=True)
        self.assertTrue(answered.ready)
        self.assertIn("20", answered.refined_prompt)
        grounded = refine_prompt(prompt, scene_context={"street_end_target": "the crosswalk"}, offline=True)
        self.assertTrue(grounded.ready)
        self.assertIn("crosswalk", grounded.refined_prompt)
        named = refine_prompt(prompt, scene_context={"targets": [
            {"id": "street-end", "name": "Street end", "object_id": "road", "position": [0, 0, 10]}]},
            offline=True)
        self.assertTrue(named.ready)
        self.assertIn("Street end", named.refined_prompt)
        self.assertFalse(needs_clarification("Walk 20 meters to the end of the street"))
        unanswered = refine_prompt(prompt, {"street_target": "I don't know"}, offline=True)
        self.assertFalse(unanswered.ready)
        self.assertFalse(needs_clarification(answered.refined_prompt))
        self.assertFalse(needs_clarification(grounded.refined_prompt))
        english = refine_prompt("Walk to the end of the street",
                                {"street_target": "the red door"}, offline=True)
        self.assertTrue(english.ready)
        self.assertFalse(needs_clarification(english.refined_prompt))

    def test_scene_numeric_target_is_preserved_and_contradictory_side_answer_reasked(self):
        scene = {"street_end_target": "20 meters forward"}
        provider = lambda *_: {"ready": True,
                               "refined_prompt": "Walk 20 meters forward to the end of the street.",
                               "explanation": "Kept the scene target.", "questions": []}
        result = refine_prompt("Walk to the end of the street", scene_context=scene,
                               provider=provider)
        self.assertTrue(result.ready)

        contradictory = refine_prompt("Jump left", {"direction_reference": "Character's right"},
                                      offline=True)
        self.assertFalse(contradictory.ready)
        self.assertEqual(contradictory.questions[0].id, "direction_reference")

    def test_provider_failure_does_not_show_unchanged_prompt_as_success(self):
        def unavailable(*_args):
            raise TimeoutError("model timed out")

        result = refine_prompt("Wave gently", provider=unavailable)
        self.assertFalse(result.ready)
        self.assertEqual(result.refined_prompt, "")
        self.assertEqual(result.questions, ())
        self.assertIn("timed out", result.warning)

    def test_model_must_preserve_action_distance_count_duration_and_target(self):
        def dropped(*_args):
            return {"ready": True, "refined_prompt": "Walk to the door.",
                    "explanation": "Improved.", "questions": []}

        result = refine_prompt("Walk 3 meters to the red door twice in 5 seconds", provider=dropped)
        self.assertFalse(result.ready)
        self.assertIn("constraint", result.warning.lower())

    def test_model_cannot_swap_distance_and_duration_or_lose_answered_target(self):
        def swapped(*_args):
            return {"ready": True, "refined_prompt": "Walk 5 meters to the red door in 3 seconds.",
                    "explanation": "Clear.", "questions": []}

        result = refine_prompt("Walk 3 meters to the red door in 5 seconds", provider=swapped)
        self.assertFalse(result.ready)

        def wrong_target(*_args):
            return {"ready": True, "refined_prompt": "Walk to the green door.",
                    "explanation": "Clear.", "questions": []}

        result = refine_prompt("Walk over there", {"reference": "the red door"}, provider=wrong_target)
        self.assertFalse(result.ready)

        def wrong_count(*_args):
            return {"ready": True, "refined_prompt": "Jump twice.",
                    "explanation": "Clear.", "questions": []}

        result = refine_prompt("Jump three times", provider=wrong_count)
        self.assertFalse(result.ready)

    def test_clear_forward_jump_passes_to_provider_and_unsupported_motion_is_explicit(self):
        seen = []

        def unsupported(_system, user):
            seen.append(json.loads(user))
            return {"ready": False, "refined_prompt": "", "questions": [],
                    "explanation": "This motion is unsupported by the current model."}

        result = refine_prompt("Jump forward", provider=unsupported)
        self.assertEqual(len(seen), 1)
        self.assertFalse(result.ready)
        self.assertFalse(result.questions)
        self.assertIn("unsupported", result.explanation)

    def test_model_cannot_change_duration_bound_continuity_pause_or_path(self):
        cases = (
            ("Run continuously for up to 30 seconds", "Run for 30 seconds.", None),
            ("Run forward continuously for 10 seconds",
             "Run forward, stop, then run for 10 seconds.", None),
            ("Run four steps, stop, then run four steps",
             "Run four steps, then run four steps.", None),
            ("Run four steps, pause, then run four steps",
             "Run four steps, then run four steps.", None),
            ("Run straight forward for 10 seconds",
             "Run forward in a circle for 10 seconds.", None),
            ("Run to the end of the street, at most 30 seconds",
             "Run to the end of the street for 30 seconds.",
             {"targets": [{"id": "street", "name": "Street End", "position": [0, 0, 10]}]}),
        )
        for original, changed, scene_context in cases:
            with self.subTest(original=original):
                result = refine_prompt(original, provider=lambda *_: {
                    "ready": True, "refined_prompt": changed,
                    "explanation": "Clarified.", "questions": []}, scene_context=scene_context)
                self.assertFalse(result.ready)
                self.assertEqual(result.refined_prompt, "")
                self.assertIn("constraint", result.warning)

    def test_equivalent_english_and_turkish_motion_qualifiers_are_accepted(self):
        cases = (
            ("Run continuously for up to 30 seconds",
             "Run without stopping for at most 30 seconds."),
            ("Run four steps, stop, then run four steps",
             "Run four steps, stop, then run four steps."),
            ("Run straight forward for 10 seconds",
             "Run forward in a straight line for 10 seconds."),
            ("En fazla 30 saniye kesintisiz düz koş, sonra durakla",
             "Run straight without stopping for at most 30 seconds, then pause."),
        )
        for original, equivalent in cases:
            with self.subTest(original=original):
                result = refine_prompt(original, provider=lambda *_: {
                    "ready": True, "refined_prompt": equivalent,
                    "explanation": "Clarified.", "questions": []})
                self.assertTrue(result.ready, result.warning)

    def test_stop_and_pause_negation_does_not_match_positive_action(self):
        cases = (
            ("Run then stop", "Run and do not stop."),
            ("Walk and pause", "Walk without a pause."),
        )
        for original, changed in cases:
            with self.subTest(original=original):
                result = refine_prompt(original, provider=lambda *_: {
                    "ready": True, "refined_prompt": changed,
                    "explanation": "Clarified.", "questions": []})
                self.assertFalse(result.ready)
                self.assertIn("constraint", result.warning)

    def test_hyphenated_non_stop_equals_continuous_without_added_stop(self):
        result = refine_prompt("Run non-stop for 10 seconds", provider=lambda *_: {
            "ready": True, "refined_prompt": "Run continuously for 10 seconds.",
            "explanation": "Clarified.", "questions": []})
        self.assertTrue(result.ready, result.warning)


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
    def test_gateway_malformed_content_keeps_exact_raw_text_for_bounded_repair(self):
        content = '  {"ready": true,  '
        envelope = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        transport = _Transport(json.dumps(envelope).encode())
        gateway = GatewayPromptProvider("https://example.test/v1", "test-model", "test-secret",
                                        transport=transport)
        result = refine_prompt("Wave", provider=gateway)
        self.assertFalse(result.ready)
        self.assertEqual(len(result.diagnostics.attempts), 2)
        for attempt in result.diagnostics.attempts:
            self.assertEqual(attempt.raw_response, content)
            self.assertEqual(attempt.issue.code, "invalid_json")
        self.assertNotIn("test-secret", str(result.diagnostics))

    def test_gateway_valid_content_retains_raw_spacing_and_parsed_fields(self):
        content = '  { "ready": true, "refined_prompt": "Wave.", "explanation": "Clearer.", "questions": [] }\n'
        envelope = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        gateway = GatewayPromptProvider("https://example.test/v1", "test-model", "test-secret",
                                        transport=_Transport(json.dumps(envelope).encode()))
        result = refine_prompt("Wave", provider=gateway)
        self.assertTrue(result.ready, result.warning)
        self.assertEqual(result.diagnostics.attempts[0].raw_response, content)
        self.assertEqual(result.diagnostics.attempts[0].parsed_response, json.loads(content))

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

    def test_configured_gateway_is_selected_and_parses_bounded_json(self):
        document = {"ready": True, "refined_prompt": "A person waves gently.",
                    "explanation": "Kept the action and speed.", "questions": []}
        envelope = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(document)}}]}
        transport = _Transport(json.dumps(envelope).encode())
        gateway = GatewayPromptProvider("https://example.test/v1", "test-model", "test-secret",
                                        transport=transport)
        with patch("object_generation.gateway_config", return_value={
                "NEON_AI_GATEWAY_BASE_URL": "https://example.test",
                "NEON_AI_GATEWAY_TOKEN": "test-secret",
                "STAGEZERO_OBJECT_MODEL": "test-model"}), \
             patch("prompt_assistant.GatewayPromptProvider.from_env", return_value=gateway):
            result = refine_prompt("Wave gently")
        self.assertTrue(result.ready)
        self.assertEqual(result.source, "gateway")
        url, kwargs = transport.request
        self.assertEqual(url, "https://example.test/v1/chat/completions")
        self.assertEqual(kwargs["json"]["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["timeout"], (10, 60))
        self.assertTrue(kwargs["headers"]["Authorization"].startswith("Bearer "))


if __name__ == "__main__":
    unittest.main()
