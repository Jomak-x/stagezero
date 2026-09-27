"""One Generate click refines, clarifies, and submits the validated direction."""

from threading import Event
from functools import partial
import unittest

from prompt_assistant import PromptAssistantResult, PromptQuestion, refine_prompt
import test_studio_ui as fixtures


def ready(text):
    return PromptAssistantResult(text, 'Direction validated.', (), True, 'test')


class PromptAssistantIntegrationTests(unittest.TestCase):
    def setUp(self):
        fixtures.StudioUITests.setUp(self)
        self.ui.prompt_assistant.refiner = lambda prompt, _answers, **_kwargs: ready(prompt)

    tearDown = fixtures.StudioUITests.tearDown
    _seed_take = fixtures.StudioUITests._seed_take
    _seed_segmented_take = fixtures.StudioUITests._seed_segmented_take

    def _finish_refinement(self):
        def finished():
            self.ui.update()
            return not self.ui.prompt_assistant._request_pending
        fixtures.wait_until(finished)

    def test_generate_submits_validated_text_to_director_backend_once(self):
        received = []
        original_generate = self.backend.generate

        def capture(request_id, prompt, history):
            received.append(prompt)
            return original_generate(request_id, prompt, history)

        self.backend.generate = capture
        self.ui.prompt.edit('Jump forward')
        self.ui.prompt_assistant.refiner = lambda *_args, **_kwargs: ready(
            'Jump forward once with a clear landing.')
        self.ui.generate.click()
        self._finish_refinement()
        self.assertTrue(self.backend.started.wait(1))
        self.assertEqual(received, ['Jump forward once with a clear landing.'])
        self.assertEqual(self.session.prompt, received[0])
        self.assertEqual(self.ui.prompt.value, received[0])
        visible = self.ui.prompt_assistant.generation_prompts.content
        self.assertIn('Original direction:', visible)
        self.assertIn('Jump forward', visible)
        self.assertIn('Validated direction used:', visible)
        self.assertIn(received[0], visible)
        self.backend.release.set()
        fixtures.wait_until(lambda: not self.session.busy)
        self.ui.update()
        self.assertIn('Validated direction used:',
                      self.ui.prompt_assistant.generation_prompts.content)

    def test_turkish_forward_jump_uses_real_validator_before_backend(self):
        received = []
        original_generate = self.backend.generate

        def capture(request_id, prompt, history):
            received.append(prompt)
            return original_generate(request_id, prompt, history)

        self.backend.generate = capture
        self.ui.prompt.edit('Öne doğru bir kere zıpla')
        self.ui.prompt_assistant.refiner = partial(refine_prompt, provider=lambda *_: {
            'ready': True, 'refined_prompt': 'Make one forward jump.',
            'explanation': 'Yön ve tekrar korundu.', 'questions': []})
        self.ui.generate.click()
        self._finish_refinement()
        self.assertTrue(self.backend.started.wait(1))
        self.assertEqual(received, ['Make one forward jump.'])
        self.assertIn('Öne doğru bir kere zıpla',
                      self.ui.prompt_assistant.generation_prompts.content)

    def test_big_jump_real_local_question_then_validated_backend_request(self):
        calls = []

        def provider(*_args):
            calls.append(1)
            return {'ready': True,
                    'refined_prompt': 'Make one high vertical jump and land in the same spot.',
                    'explanation': 'Clarified the height and landing.', 'questions': []}

        self.ui.prompt.edit('make a nice big jump')
        self.ui.prompt_assistant.refiner = partial(refine_prompt, provider=provider)
        self.ui.generate.click()
        self._finish_refinement()
        self.assertFalse(self.session.busy)
        self.assertEqual(calls, [])
        self.assertEqual(self.ui.prompt_assistant._result.questions[0].id, 'jump_emphasis')
        self.ui.prompt_assistant._rows[0][1].edit('Height (straight up, landing in place)')
        self.ui.prompt_assistant.continue_button.click()
        self._finish_refinement()
        self.assertTrue(self.backend.started.wait(1))
        self.assertEqual(calls, [1])
        self.assertEqual(self.session.prompt,
                         'Make one high vertical jump and land in the same spot.')

    def test_single_generate_then_answer_continues_to_backend(self):
        calls = []

        def refine(_prompt, answers, **_kwargs):
            calls.append(dict(answers))
            if not answers:
                return PromptAssistantResult('', 'Choose the intended turn.',
                    (PromptQuestion('turn', 'Which turn?', ('180° in place', 'Walk backward')),),
                    False, 'test')
            return ready('Turn 180° in place to face the opposite direction.')

        self.ui.prompt.edit('turn back')
        self.ui.prompt_assistant.refiner = refine
        self.ui.generate.click()
        self._finish_refinement()
        self.assertFalse(self.session.busy)
        self.assertIn('Waiting for your answer', self.ui.prompt_assistant.status.content)
        self.ui.prompt_assistant._rows[0][1].edit('180° in place')
        self.ui.prompt_assistant.continue_button.click()
        self._finish_refinement()
        self.assertTrue(self.backend.started.wait(1))
        self.assertEqual(calls, [{}, {'turn': '180° in place'}])
        self.assertEqual(self.session.prompt, 'Turn 180° in place to face the opposite direction.')

    def test_cancelled_question_allows_fresh_generate_attempt(self):
        calls = []

        def refine(_prompt, _answers, **_kwargs):
            calls.append(1)
            return PromptAssistantResult('', 'Choose the intended turn.',
                (PromptQuestion('turn', 'Which turn?', ('180° in place',)),), False, 'test')

        self.ui.prompt.edit('turn back')
        self.ui.prompt_assistant.refiner = refine
        self.ui.generate.click()
        self._finish_refinement()
        self.ui.prompt_assistant.cancel_button.click()
        self.ui.update()
        self.assertFalse(self.ui.generate.disabled)
        self.assertEqual(self.ui.generate.label, 'Generate motion')
        self.ui.generate.click()
        self._finish_refinement()
        self.assertEqual(calls, [1, 1])
        self.assertFalse(self.session.busy)

    def test_inflight_double_click_starts_one_refinement_and_generation(self):
        started, release = Event(), Event()
        calls = []

        def refine(_prompt, _answers, **_kwargs):
            calls.append(1)
            started.set()
            release.wait(1)
            return ready('Wave with the right hand.')

        self.ui.prompt.edit('wave')
        self.ui.prompt_assistant.refiner = refine
        try:
            self.ui.generate.click()
            self.assertTrue(started.wait(1))
            self.ui.generate.click()
            self.assertEqual(calls, [1])
            self.assertFalse(self.session.busy)
            release.set()
            self._finish_refinement()
            self.assertTrue(self.backend.started.wait(1))
            self.assertEqual(self.session.prompt, 'Wave with the right hand.')
        finally:
            release.set()

    def test_cancel_at_final_studio_submission_guard_prevents_backend_call(self):
        self.ui.prompt.edit('Wave')
        self.ui.prompt_assistant.refiner = lambda *_args, **_kwargs: ready(
            'Wave with the right hand.')
        assistant = self.ui.prompt_assistant
        original_guard = assistant._submission_guard

        def cancel_before_commit():
            assistant.cancel()
            return original_guard()

        assistant._submission_guard = cancel_before_commit
        self.ui.generate.click()
        self._finish_refinement()
        self.assertFalse(self.session.busy)
        self.assertFalse(self.backend.started.is_set())
        self.assertIn('cancelled', assistant.status.content)

    def test_changed_prompt_or_context_discards_inflight_response(self):
        for changed in ('prompt', 'duration', 'scene', 'motion_mode_roundtrip'):
            with self.subTest(changed=changed):
                started, release = Event(), Event()

                def refine(_prompt, _answers, **_kwargs):
                    started.set()
                    release.wait(1)
                    return ready('Old validated direction.')

                self.ui.prompt.edit('Old direction')
                self.ui.prompt_assistant.refiner = refine
                try:
                    self.ui.generate.click()
                    self.assertTrue(started.wait(1))
                    if changed == 'prompt':
                        self.ui.prompt.edit('New user direction')
                    elif changed == 'duration':
                        self.ui.duration_seconds.edit('5.0')
                    elif changed == 'motion_mode_roundtrip':
                        self.session.set_character_motion_enabled(False)
                        self.session.set_character_motion_enabled(True)
                    else:
                        self.session.scene['objects'] = [{
                            'id': 'marker', 'name': 'New marker', 'position': [1, 0, 2]}]
                    self.ui.update()
                    release.set()
                    self._finish_refinement()
                    self.assertFalse(self.session.busy)
                    self.assertFalse(self.backend.started.is_set())
                    self.assertNotEqual(self.ui.prompt.value, 'Old validated direction.')
                finally:
                    release.set()

    def test_failed_refinement_preserves_original_and_never_submits(self):
        for result in (
            PromptAssistantResult('', 'Response was malformed.', (), False, 'test',
                                  'Model returned invalid fields.'),
            PromptAssistantResult('', 'Network unavailable.', (), False, 'test',
                                  'Model request timed out.'),
        ):
            with self.subTest(warning=result.warning):
                self.ui.prompt.edit('Jump forward')
                self.ui.prompt_assistant.refiner = lambda *_args, **_kwargs: result
                self.ui.generate.click()
                self._finish_refinement()
                self.assertEqual(self.ui.prompt.value, 'Jump forward')
                self.assertFalse(self.session.busy)
                self.assertFalse(self.backend.started.is_set())
                self.assertIn(result.warning, self.ui.prompt_assistant.status.content)
                self.assertNotIn(result.warning, self.ui.prompt_assistant.explanation.content)
                self.ui.prompt_assistant.cancel_button.click()

    def test_refiner_exception_preserves_original_and_never_submits(self):
        def fail(*_args, **_kwargs):
            raise ConnectionError('Provider offline')

        self.ui.prompt.edit('Jump forward')
        self.ui.prompt_assistant.refiner = fail
        self.ui.generate.click()
        self._finish_refinement()
        self.assertEqual(self.ui.prompt.value, 'Jump forward')
        self.assertFalse(self.session.busy)
        self.assertIn('Provider offline', self.ui.prompt_assistant.status.content)

    def test_ambiguous_action_edit_preserves_existing_take(self):
        take = self._seed_segmented_take()
        original = take.positions.copy()
        self.ui._begin_action_edit(take, 0, 'replace')
        self.ui.update()
        self.ui.action_prompt.edit('turn back')
        self.ui.save_action.click()
        self.assertFalse(self.session.busy)
        self.assertIs(self.session.takes[take.id], take)
        self.assertTrue((take.positions == original).all())
        self.assertIn('Clarify', self.session.status)

    def test_assistant_hidden_for_static_character(self):
        self.session.set_character_motion_enabled(False)
        self.ui.update()
        self.assertFalse(self.ui.prompt_assistant.folder.visible)
        self.assertFalse(self.ui.action_assistant.folder.visible)

    def test_apply_rechecks_source_without_generating(self):
        self.ui.prompt.edit('turn back')
        context = self.ui._assistant_context(False)
        self.ui.prompt.edit('Wave with your left hand')
        self.assertFalse(self.ui.prompt_assistant.apply(
            'Turn 180 degrees in place.', context, 'turn back'))
        self.assertEqual(self.ui.prompt.value, 'Wave with your left hand')
        self.assertFalse(self.session.busy)


if __name__ == '__main__':
    unittest.main()
