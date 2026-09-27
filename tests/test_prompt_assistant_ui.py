"""Prompt assistant editor lifecycle and stale-result regressions."""

from threading import Event, Thread
from types import SimpleNamespace
from functools import partial
import time
import unittest

from prompt_assistant import PromptAssistantResult, PromptQuestion, refine_prompt
from prompt_assistant_ui import PromptAssistantUI


class Handle:
    def __init__(self, **values):
        self.writes = []
        self.callbacks = {'click': [], 'update': []}
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        if name not in ('writes', 'callbacks'):
            self.writes.append((name, value))
        object.__setattr__(self, name, value)

    def on_click(self, callback):
        self.callbacks['click'].append(callback)
        return callback

    def on_update(self, callback):
        self.callbacks['update'].append(callback)
        return callback

    def click(self):
        for callback in self.callbacks['click']:
            callback(SimpleNamespace())

    def edit(self, value):
        self.value = value
        for callback in self.callbacks['update']:
            callback(SimpleNamespace())

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class Gui:
    def __init__(self):
        self.handles = []

    def _new(self, **values):
        values.setdefault('visible', True)
        handle = Handle(**values)
        self.handles.append(handle)
        return handle

    def add_folder(self, label, *, expand_by_default=True):
        return self._new(label=label, expand_by_default=expand_by_default)

    def add_button(self, label, **_kwargs):
        return self._new(label=label, disabled=False)

    def add_html(self, content):
        return self._new(content=content)

    def add_dropdown(self, label, options, *, initial_value=None):
        return self._new(label=label, options=options,
                         value=initial_value or options[0], disabled=False)

    def add_text(self, label, *, initial_value='', **_kwargs):
        return self._new(label=label, value=initial_value, disabled=False)


def ready(text='Face the opposite direction.'):
    return PromptAssistantResult(text, 'Use this clear motion instruction.', (), True, 'test')


def question():
    return PromptAssistantResult('', 'Pick the intended turn.',
                                 (PromptQuestion('meaning', '<Turn back?>',
                                                 ('Turn 180° in place', 'Move backward')),),
                                 False, 'test')


class PromptAssistantUITests(unittest.TestCase):
    def setUp(self):
        self.gui = Gui()
        self.prompt = self.gui.add_text('Direction', initial_value='Turn back')
        self.target = ['take-1', 1, 1, 'direct']
        self.applied = []
        self.calls = []

        def context():
            return tuple(self.target)

        def apply(text, expected_context, expected_prompt):
            if tuple(self.target) != expected_context or self.prompt.value != expected_prompt:
                return False
            self.prompt.value = text
            self.applied.append(text)
            return True

        self.context = context
        self.apply = apply

    def make_ui(self, refiner, scene_context=None, *, auto_apply=False):
        kwargs = {'scene_context': scene_context} if scene_context is not None else {}
        kwargs['auto_apply'] = auto_apply
        ui = PromptAssistantUI(self.gui, self.prompt, self.context, self.apply, refiner,
                               **kwargs)
        ui.refresh(True, False)
        return ui

    @staticmethod
    def wait_for(ui):
        for _ in range(100):
            ui.refresh(True, False)
            if not ui._request_pending:
                return
            time.sleep(0.005)
        raise AssertionError('assistant request did not complete')

    def test_clarification_choices_freeform_and_explicit_use(self):
        def refiner(prompt, answers=None, *, offline=False):
            self.calls.append((prompt, answers, offline))
            return question() if offline or not answers else ready('Rotate 180° in place.')

        ui = self.make_ui(refiner)
        self.assertTrue(ui.clarify())
        self.assertEqual(self.applied, [])
        self.assertFalse(ui.folder.expand_by_default)
        self.assertIn('answer needed', ui.folder.label)
        self.assertIn('&lt;Turn back?&gt;', ui._rows[0][0].content)
        self.assertTrue(ui.continue_button.disabled)
        ui._rows[0][2].edit('Turn 180° in place, with feet planted')
        self.assertFalse(ui.continue_button.disabled)
        ui.continue_button.click()
        self.wait_for(ui)
        self.assertEqual(self.calls[-1][1], {'meaning': 'Turn 180° in place, with feet planted'})
        self.assertEqual(self.prompt.value, 'Turn back')
        ui.preview.edit('Rotate fully in place and face the opposite direction.')
        ui.use_prompt.click()
        self.assertEqual(self.applied, ['Rotate fully in place and face the opposite direction.'])

    def test_duplicate_click_does_not_start_second_request_and_idle_writes_zero(self):
        started = Event()
        release = Event()

        def refiner(_prompt, _answers):
            self.calls.append(1)
            started.set()
            release.wait(1)
            return ready()

        ui = self.make_ui(refiner)
        try:
            ui.improve.click()
            self.assertTrue(started.wait(1))
            ui.improve.click()
            self.assertEqual(len(self.calls), 1)
            release.set()
            self.wait_for(ui)
            for handle in self.gui.handles:
                handle.writes.clear()
            for _ in range(20):
                ui.refresh(True, False)
            self.assertEqual(sum(len(h.writes) for h in self.gui.handles), 0)
        finally:
            release.set()

    def test_concurrent_improve_callbacks_reserve_only_one_request(self):
        capture_started = Event()
        release_capture = Event()
        armed = False
        requests = []

        def context():
            if armed:
                capture_started.set()
                release_capture.wait(1)
            return tuple(self.target)

        def refiner(_prompt, _answers):
            requests.append(1)
            return ready()

        ui = PromptAssistantUI(self.gui, self.prompt, context, self.apply, refiner)
        ui.refresh(True, False)
        armed = True
        first = Thread(target=ui.improve.click)
        second = Thread(target=ui.improve.click)
        try:
            first.start()
            self.assertTrue(capture_started.wait(1))
            self.assertTrue(ui.blocks_generation())
            self.assertTrue(ui.improve.disabled)
            self.assertTrue(ui.cancel_button.visible)
            self.assertIn('Improving', ui.status.content)
            second.start()
            second.join(1)
            self.assertFalse(second.is_alive())
            release_capture.set()
            first.join(1)
            self.assertFalse(first.is_alive())
            self.wait_for(ui)
            self.assertEqual(len(requests), 1)
        finally:
            release_capture.set()

    def test_cancel_during_retry_validation_does_not_launch_stale_request(self):
        entered = Event()
        release = Event()
        armed = False
        calls = []

        def context():
            nonlocal armed
            if armed:
                armed = False
                entered.set()
                release.wait(1)
            return tuple(self.target)

        def refiner(_prompt, _answers):
            calls.append(1)
            raise RuntimeError('Temporary failure')

        ui = PromptAssistantUI(self.gui, self.prompt, context, self.apply, refiner)
        ui.refresh(True, False)
        ui.improve.click()
        self.wait_for(ui)
        self.assertTrue(ui.retry_button.visible)
        armed = True
        retry = Thread(target=ui.retry_button.click)
        try:
            retry.start()
            self.assertTrue(entered.wait(1))
            ui.cancel_button.click()
            release.set()
            retry.join(1)
            self.assertFalse(retry.is_alive())
            self.assertEqual(calls, [1])
            self.assertFalse(ui._request_pending)
        finally:
            release.set()

    def test_changed_prompt_take_clip_and_edit_context_discard_inflight_result(self):
        for change in ('prompt', 'take', 'clip', 'edit'):
            with self.subTest(change=change):
                started = Event()
                release = Event()

                def refiner(_prompt, _answers):
                    started.set()
                    release.wait(1)
                    return ready()

                ui = self.make_ui(refiner)
                try:
                    ui.improve.click()
                    self.assertTrue(started.wait(1))
                    if change == 'prompt':
                        self.prompt.edit('Newer direction')
                    else:
                        self.target[{'take': 0, 'clip': 2, 'edit': 3}[change]] = change
                    ui.refresh(True, False)
                    release.set()
                    self.wait_for(ui)
                    self.assertFalse(ui.use_prompt.visible)
                    ui.use_prompt.click()
                    self.assertEqual(self.applied, [])
                finally:
                    release.set()
                self.prompt.value = 'Turn back'
                self.target[:] = ['take-1', 1, 1, 'direct']

    def test_hidden_or_busy_cannot_apply_even_after_result(self):
        ui = self.make_ui(lambda _prompt, _answers: ready())
        ui.improve.click()
        self.wait_for(ui)
        ui.refresh(False, False)
        ui.use_prompt.click()
        self.assertEqual(self.applied, [])
        ui.refresh(True, False)
        ui.improve.click()
        self.wait_for(ui)
        ui.refresh(True, True)
        ui.use_prompt.click()
        self.assertEqual(self.applied, [])

    def test_failed_refiner_is_escaped_and_keeps_original_prompt(self):
        def refiner(_prompt, _answers):
            raise RuntimeError('<provider unavailable>')

        ui = self.make_ui(refiner)
        ui.improve.click()
        self.wait_for(ui)
        self.assertIn('&lt;provider unavailable&gt;', ui.status.content)
        self.assertFalse(ui.use_prompt.visible)
        self.assertEqual(self.prompt.value, 'Turn back')

    def test_offline_fallback_warning_and_source_are_visible(self):
        result = PromptAssistantResult('Turn in place.', 'Clarified the action.',
                                       (), True, 'offline', '<AI unavailable>')
        ui = self.make_ui(lambda _prompt, _answers: result)
        ui.improve.click()
        self.wait_for(ui)
        self.assertIn('Offline guidance', ui.explanation.content)
        self.assertIn('Clarified the action.', ui.explanation.content)
        self.assertIn('&lt;AI unavailable&gt;', ui.explanation.content)

    def test_followup_refinement_receives_question_text_and_answer_history(self):
        requests = []

        def refiner(_prompt, answers=None, *, offline=False, history=()):
            requests.append((dict(answers or {}), tuple(history), offline))
            if offline:
                return question()
            if 'speed' not in answers:
                return PromptAssistantResult('', 'How quickly?',
                                             (PromptQuestion('speed', 'What speed?',
                                                             ('Slow', 'Fast')),), False, 'test')
            return ready()

        ui = self.make_ui(refiner)
        self.assertTrue(ui.clarify())
        ui._rows[0][1].edit('Turn 180° in place')
        ui.continue_button.click()
        self.wait_for(ui)
        self.assertEqual(requests[1][1], ({'id': 'meaning', 'question': '<Turn back?>',
                                           'answer': 'Turn 180° in place'},))
        ui._rows[0][1].edit('Slow')
        ui.continue_button.click()
        self.wait_for(ui)
        self.assertEqual([entry['id'] for entry in requests[2][1]], ['meaning', 'speed'])
        self.assertEqual(requests[2][1][1]['question'], 'What speed?')
        self.assertEqual(requests[2][0]['speed'], 'Slow')

    def test_apply_callback_rejects_race_after_ui_validation(self):
        def raced_apply(_text, _context, _prompt):
            self.prompt.value = 'Changed in another callback'
            return False

        ui = PromptAssistantUI(self.gui, self.prompt, self.context, raced_apply,
                               lambda _prompt, _answers: ready())
        ui.refresh(True, False)
        ui.improve.click()
        self.wait_for(ui)
        ui.use_prompt.click()
        self.assertEqual(self.prompt.value, 'Changed in another callback')
        self.assertIn('changed', ui.status.content)

    def test_empty_input_and_failure_offer_explicit_retry_without_losing_draft(self):
        attempts = []

        def refiner(prompt, _answers):
            attempts.append(prompt)
            if len(attempts) == 1:
                raise RuntimeError('The model is unavailable')
            return ready('Jump forward.')

        self.prompt.edit('   ')
        ui = self.make_ui(refiner)
        self.assertTrue(ui.improve.disabled)
        ui.improve.click()
        self.assertEqual(attempts, [])

        self.prompt.edit('Jump forward')
        ui.refresh(True, False)
        self.assertFalse(ui.improve.disabled)
        ui.improve.click()
        self.wait_for(ui)
        self.assertEqual(self.prompt.value, 'Jump forward')
        self.assertTrue(ui.retry_button.visible)
        self.assertIn('The model is unavailable', ui.status.content)
        ui.retry_button.click()
        self.wait_for(ui)
        self.assertEqual(attempts, ['Jump forward', 'Jump forward'])
        self.assertEqual(ui.preview.value, 'Jump forward.')

    def test_question_answer_retry_and_cancel_ignore_late_response(self):
        started = Event()
        release = Event()
        attempts = []

        def refiner(_prompt, answers, *, history=()):
            attempts.append((dict(answers), tuple(history)))
            if not answers:
                return question()
            if len(attempts) == 2:
                raise RuntimeError('Temporary API failure')
            started.set()
            release.wait(1)
            return ready('Turn 180° in place.')

        ui = self.make_ui(refiner)
        ui.improve.click()
        self.wait_for(ui)
        self.assertTrue(ui.blocks_generation())
        ui._rows[0][2].edit('Turn 180° in place, slowly')
        ui.continue_button.click()
        self.wait_for(ui)
        self.assertTrue(ui.blocks_generation())
        self.assertTrue(ui.retry_button.visible)
        self.assertEqual(ui._rows[0][2].value, 'Turn 180° in place, slowly')
        ui._rows[0][2].edit('Turn 180° in place, without moving position')
        ui.retry_button.click()
        self.assertTrue(started.wait(1))
        self.assertTrue(ui.blocks_generation())
        self.assertIn('Improving', ui.status.content)
        ui.cancel_button.click()
        self.assertTrue(ui.blocks_generation())
        self.assertEqual(self.prompt.value, 'Turn back')
        self.assertFalse(ui.questions_folder.visible)
        release.set()
        ui.refresh(True, False)
        self.assertFalse(ui.use_prompt.visible)
        self.prompt.edit('Turn 180° in place')
        ui.refresh(True, False)
        self.assertFalse(ui.blocks_generation())
        self.assertEqual(attempts[2][0]['meaning'],
                         'Turn 180° in place, without moving position')
        self.assertEqual(attempts[2][1][0]['answer'],
                         'Turn 180° in place, without moving position')

    def test_refined_preview_after_clarification_blocks_until_used(self):
        def refiner(_prompt, answers):
            return ready('Turn 180° in place.') if answers else question()

        ui = self.make_ui(refiner)
        ui.improve.click()
        self.wait_for(ui)
        ui._rows[0][1].edit('Turn 180° in place')
        ui.continue_button.click()
        self.wait_for(ui)
        self.assertTrue(ui.blocks_generation())
        self.assertTrue(ui.improve.disabled)
        ui.use_prompt.click()
        self.assertFalse(ui.blocks_generation())
        self.assertEqual(self.prompt.value, 'Turn 180° in place.')

    def test_scene_context_is_passed_and_scene_change_invalidates_result(self):
        scene = {'direction_reference': 'character'}
        requests = []

        def refiner(_prompt, _answers, *, scene_context=None):
            requests.append(dict(scene_context or {}))
            return ready('Jump to the character’s right.')

        ui = self.make_ui(refiner, scene_context=lambda: scene)
        ui.improve.click()
        self.wait_for(ui)
        self.assertEqual(requests, [{'direction_reference': 'character'}])
        scene['direction_reference'] = 'screen'
        ui.refresh(True, False)
        self.assertFalse(ui.use_prompt.visible)
        self.assertEqual(self.prompt.value, 'Turn back')

    def test_nonready_model_response_without_question_is_retryable(self):
        failures = PromptAssistantResult('', 'Could not refine.', (), False,
                                         'ollama', 'Model request timed out')
        ui = self.make_ui(lambda _prompt, _answers: failures)
        ui.improve.click()
        self.wait_for(ui)
        self.assertTrue(ui.retry_button.visible)
        self.assertIn('Model request timed out', ui.status.content)
        self.assertEqual(self.prompt.value, 'Turn back')

    def test_failed_generation_rewrite_can_submit_original_direction(self):
        submissions = []

        def generate(text, context, scene, original, guard):
            with guard() as allowed:
                if allowed:
                    submissions.append((text, context, scene, original))
            return allowed

        failure = PromptAssistantResult('', 'Could not refine.', (), False,
                                        'gateway', 'Assistant changed a stated constraint')
        ui = PromptAssistantUI(self.gui, self.prompt, self.context, self.apply,
                               lambda _prompt, _answers: failure,
                               auto_apply=True, generate=generate)
        ui.refresh(True, False)
        self.assertTrue(ui.start_generation())
        self.wait_for(ui)
        self.assertTrue(ui.generate_original_button.visible)
        self.assertIn('action needed', ui.folder.label)
        ui.generate_original_button.click()
        self.assertEqual(submissions, [('Turn back', tuple(self.target), None, 'Turn back')])
        self.assertEqual(self.prompt.value, 'Turn back')
        self.assertFalse(ui.generate_original_button.visible)

    def test_original_direction_fallback_rejects_changed_context(self):
        submissions = []

        def generate(text, _context, _scene, _original, guard):
            with guard() as allowed:
                if allowed:
                    submissions.append(text)
            return True

        failure = PromptAssistantResult('', 'Could not refine.', (), False,
                                        'gateway', 'Model request timed out')
        ui = PromptAssistantUI(self.gui, self.prompt, self.context, self.apply,
                               lambda _prompt, _answers: failure,
                               auto_apply=True, generate=generate)
        ui.refresh(True, False)
        ui.start_generation()
        self.wait_for(ui)
        self.target[1] += 1
        ui.refresh(True, False)
        self.assertFalse(ui.generate_original_button.visible)
        ui.generate_original_button.click()
        self.assertEqual(submissions, [])

    def test_actual_refiner_asks_then_preserves_resolved_target(self):
        self.prompt.edit('Walk over there')
        ui = self.make_ui(partial(refine_prompt, offline=True))
        ui.improve.click()
        self.wait_for(ui)
        self.assertTrue(ui.questions_folder.visible)
        self.assertTrue(ui.blocks_generation())
        ui._rows[0][2].edit('the marked doorway')
        ui.continue_button.click()
        self.wait_for(ui)
        self.assertTrue(ui.use_prompt.visible)
        self.assertIn('marked doorway', ui.preview.value)
        self.assertEqual(self.prompt.value, 'Walk over there')

    def test_auto_apply_direct_result_and_revert_original(self):
        ui = self.make_ui(lambda _prompt, _answers: ready('Turn 180° in place.'),
                          auto_apply=True)
        ui.improve.click()
        self.wait_for(ui)
        self.assertEqual(self.prompt.value, 'Turn 180° in place.')
        self.assertEqual(self.applied, ['Turn 180° in place.'])
        self.assertIn('applied', ui.status.content)
        self.assertIn('Use this clear motion instruction.', ui.explanation.content)
        self.assertFalse(ui.use_prompt.visible)
        self.assertTrue(ui.revert_button.visible)
        self.assertFalse(ui.blocks_generation())
        ui.revert_button.click()
        self.assertEqual(self.prompt.value, 'Turn back')
        self.assertFalse(ui.revert_button.visible)

    def test_auto_apply_after_all_clarification_questions(self):
        def refiner(_prompt, answers):
            return ready('Turn 180° in place.') if answers else question()

        ui = self.make_ui(refiner, auto_apply=True)
        ui.improve.click()
        self.wait_for(ui)
        self.assertEqual(self.prompt.value, 'Turn back')
        ui._rows[0][1].edit('Turn 180° in place')
        ui.continue_button.click()
        self.wait_for(ui)
        self.assertEqual(self.prompt.value, 'Turn 180° in place.')
        self.assertEqual(self.applied, ['Turn 180° in place.'])
        self.assertTrue(ui.revert_button.visible)

    def test_auto_apply_reentrant_refresh_and_repeat_are_idempotent(self):
        calls = []
        ui = None

        def apply(text, context, source):
            calls.append((text, source))
            if tuple(self.target) != context or self.prompt.value != source:
                return False
            self.prompt.value = text
            ui.refresh(True, False)
            return True

        replies = iter((ready('Turn 180° in place.'), ready('Rotate slowly in place.')))
        ui = PromptAssistantUI(self.gui, self.prompt, self.context, apply,
                               lambda _prompt, _answers: next(replies), auto_apply=True)
        ui.refresh(True, False)
        ui.improve.click()
        self.wait_for(ui)
        for _ in range(5):
            ui.refresh(True, False)
        self.assertEqual(calls, [('Turn 180° in place.', 'Turn back')])
        self.assertTrue(ui.revert_button.visible)
        ui.improve.click()
        self.wait_for(ui)
        self.assertEqual(calls[-1], ('Rotate slowly in place.', 'Turn 180° in place.'))
        self.assertEqual(self.prompt.value, 'Rotate slowly in place.')
        ui.revert_button.click()
        self.assertEqual(self.prompt.value, 'Turn back')
        self.assertEqual(calls[-1], ('Turn back', 'Rotate slowly in place.'))

    def test_auto_apply_local_ready_clarification_uses_atomic_callback(self):
        def refiner(_prompt, _answers, *, offline=False):
            self.assertTrue(offline)
            return ready('Turn 180° in place.')

        ui = self.make_ui(refiner, auto_apply=True)
        self.assertFalse(ui.clarify())
        self.assertEqual(self.prompt.value, 'Turn 180° in place.')
        self.assertTrue(ui.revert_button.visible)

    def test_cancel_between_apply_and_submit_blocks_generation(self):
        submissions = []

        def generate(text, _context, _scene, _original, guard):
            with guard() as allowed:
                if allowed:
                    submissions.append(text)
            return allowed

        ui = PromptAssistantUI(self.gui, self.prompt, self.context, self.apply,
                               lambda _prompt, _answers: ready('Turn 180° in place.'),
                               auto_apply=True, generate=generate)
        original_render = ui._render
        cancelled = []

        def render_with_cancel():
            if ui._generation_submitting and not ui._applying and not cancelled:
                cancelled.append(True)
                ui.cancel()
            original_render()

        ui._render = render_with_cancel
        ui.refresh(True, False)
        ui.start_generation()
        self.wait_for(ui)
        self.assertEqual(cancelled, [True])
        self.assertEqual(submissions, [])
        self.assertIn('cancelled', ui.status.content)

    def test_cancel_during_apply_blocks_generation(self):
        submissions = []
        ui = None

        def apply(text, context, original):
            ui.cancel()
            return self.apply(text, context, original)

        def generate(text, _context, _scene, _original, guard):
            with guard() as allowed:
                if allowed:
                    submissions.append(text)
            return allowed

        ui = PromptAssistantUI(self.gui, self.prompt, self.context, apply,
                               lambda _prompt, _answers: ready('Turn 180° in place.'),
                               auto_apply=True, generate=generate)
        ui.refresh(True, False)
        ui.start_generation()
        self.wait_for(ui)
        self.assertEqual(submissions, [])
        self.assertIn('cancelled', ui.status.content)

        started, release = Event(), Event()

        def later_refine(_prompt, _answers):
            started.set()
            release.wait(1)
            return ready('A different improvement.')

        ui.refiner = later_refine
        try:
            self.assertIsNone(ui.improve)
            ui.start_generation()
            self.assertTrue(started.wait(1))
            ui.cancel()
            release.set()
            self.wait_for(ui)
            self.assertNotEqual(self.prompt.value, 'A different improvement.')
        finally:
            release.set()

    def test_auto_apply_callback_rejection_keeps_newer_user_text(self):
        def raced_apply(_text, _context, _source):
            self.prompt.value = 'My newer direction'
            return False

        ui = PromptAssistantUI(self.gui, self.prompt, self.context, raced_apply,
                               lambda _prompt, _answers: ready(), auto_apply=True)
        ui.refresh(True, False)
        ui.improve.click()
        self.wait_for(ui)
        self.assertEqual(self.prompt.value, 'My newer direction')
        self.assertIn('changed', ui.status.content)
        self.assertFalse(ui.revert_button.visible)

    def test_auto_apply_drops_revert_when_callback_switches_editor(self):
        def moved_apply(text, context, source):
            if tuple(self.target) != context or self.prompt.value != source:
                return False
            self.prompt.value = text
            self.target[0] = 'take-2'
            return True

        ui = PromptAssistantUI(self.gui, self.prompt, self.context, moved_apply,
                               lambda _prompt, _answers: ready(), auto_apply=True)
        ui.refresh(True, False)
        ui.improve.click()
        self.wait_for(ui)
        self.assertEqual(self.prompt.value, 'Face the opposite direction.')
        self.assertFalse(ui.revert_button.visible)
        ui.revert_button.click()
        self.assertEqual(self.prompt.value, 'Face the opposite direction.')

    def test_stale_auto_apply_cannot_bind_to_new_request_during_scene_check(self):
        entered, release, provider_release = Event(), Event(), Event()
        armed = False

        def scene_context():
            nonlocal armed
            if armed:
                armed = False
                entered.set()
                release.wait(1)
            return {'objects': []}

        def refiner(_prompt, _answers):
            provider_release.wait(1)
            return ready('New refinement')

        ui = self.make_ui(refiner, scene_context=scene_context, auto_apply=True)
        ui._capture_source()
        old_result = ready('Old refinement')
        ui._show_result(old_result)
        old_request_id = ui._request_id
        armed = True
        old_completion = Thread(target=ui._auto_apply_result,
                                args=(old_result, old_request_id))
        try:
            old_completion.start()
            self.assertTrue(entered.wait(1))
            ui.cancel_button.click()
            self.prompt.edit('New user draft')
            ui.improve.click()
            release.set()
            old_completion.join(1)
            self.assertFalse(old_completion.is_alive())
            self.assertEqual(self.prompt.value, 'New user draft')
            self.assertEqual(self.applied, [])
        finally:
            release.set()
            provider_release.set()

    def test_stale_revert_cannot_restore_new_editors_applied_record(self):
        entered, release = Event(), Event()
        armed = False

        def scene_context():
            nonlocal armed
            if armed:
                armed = False
                entered.set()
                release.wait(1)
            return {'objects': []}

        def refiner(prompt, _answers):
            return ready('First improvement' if prompt == 'Turn back'
                         else 'Second improvement')

        ui = self.make_ui(refiner, scene_context=scene_context, auto_apply=True)
        ui.improve.click()
        self.wait_for(ui)
        self.assertEqual(self.prompt.value, 'First improvement')
        armed = True
        stale_revert = Thread(target=ui.revert_button.click)
        try:
            stale_revert.start()
            self.assertTrue(entered.wait(1))
            self.target[0] = 'take-2'
            self.prompt.edit('Second draft')
            ui.refresh(True, False)
            ui.improve.click()
            self.wait_for(ui)
            release.set()
            stale_revert.join(1)
            self.assertFalse(stale_revert.is_alive())
            self.assertEqual(self.prompt.value, 'Second improvement')
            self.assertEqual(self.applied, ['First improvement', 'Second improvement'])
        finally:
            release.set()

    def test_auto_apply_does_not_overwrite_edit_context_cancel_or_error(self):
        for change in ('prompt', 'context', 'cancel', 'error'):
            with self.subTest(change=change):
                started, release = Event(), Event()

                def refiner(_prompt, _answers):
                    started.set()
                    release.wait(1)
                    if change == 'error':
                        raise RuntimeError('Provider unavailable')
                    return ready()

                ui = self.make_ui(refiner, auto_apply=True)
                try:
                    ui.improve.click()
                    self.assertTrue(started.wait(1))
                    if change == 'prompt':
                        self.prompt.edit('Changed while improving')
                    elif change == 'context':
                        self.target[0] = 'take-2'
                    elif change == 'cancel':
                        ui.cancel_button.click()
                    ui.refresh(True, False)
                    release.set()
                    self.wait_for(ui)
                    self.assertEqual(self.applied, [])
                    self.assertFalse(ui.revert_button.visible)
                finally:
                    release.set()
                    self.prompt.value = 'Turn back'
                    self.target[:] = ['take-1', 1, 1, 'direct']

    def test_auto_apply_revert_rejects_later_editor_edit_or_context_change(self):
        for change in ('prompt', 'context'):
            with self.subTest(change=change):
                ui = self.make_ui(lambda _prompt, _answers: ready(), auto_apply=True)
                ui.improve.click()
                self.wait_for(ui)
                if change == 'prompt':
                    self.prompt.edit('My own revision')
                else:
                    self.target[0] = 'take-2'
                ui.revert_button.click()
                self.assertEqual(self.prompt.value,
                                 'My own revision' if change == 'prompt' else
                                 'Face the opposite direction.')
                self.prompt.value = 'Turn back'
                self.target[:] = ['take-1', 1, 1, 'direct']
                self.applied.clear()


if __name__ == '__main__':
    unittest.main()
