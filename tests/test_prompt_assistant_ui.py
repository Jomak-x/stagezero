"""Prompt assistant editor lifecycle and stale-result regressions."""

from threading import Event
from types import SimpleNamespace
import time
import unittest

from prompt_assistant import PromptAssistantResult, PromptQuestion
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

    def make_ui(self, refiner):
        ui = PromptAssistantUI(self.gui, self.prompt, self.context, self.apply, refiner)
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
        self.assertTrue(ui.folder.expand_by_default)
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


if __name__ == '__main__':
    unittest.main()
