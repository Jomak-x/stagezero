"""Viser controls for reviewing an AI-improved motion direction."""

from html import escape
from inspect import Parameter, signature
from threading import Lock, Thread

from prompt_assistant import refine_prompt


CHOOSE = 'Choose an answer…'
MAX_PROMPT_LENGTH = 500


class PromptAssistantUI:
    """A draft assistant that never starts motion generation itself.

    ``context`` returns an equality-comparable token for the target editor,
    take, and clip revision. ``apply`` must atomically recheck that token and
    the original prompt before assigning the approved text. The caller owns
    any session lock; no assistant callback holds its private lock while
    calling either function or while invoking the refiner.
    """

    def __init__(self, gui, prompt_handle, context, apply, refiner=refine_prompt):
        self._gui = gui
        self.prompt_handle = prompt_handle
        self.context = context
        self.apply = apply
        self.refiner = refiner
        try:
            parameters = signature(refiner).parameters
            self._refiner_accepts_history = ('history' in parameters or any(
                parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()))
        except (TypeError, ValueError):
            self._refiner_accepts_history = True
        self._visible = True
        self._busy = False
        self._source_context = None
        self._source_prompt = None
        self._result = None
        self._answers = {}
        self._history = []
        self._request_pending = False
        self._request_id = 0
        self._completed = None
        self._preview_revision = 0
        self._request_preview_revision = 0
        self._error = ''
        self._mutex = Lock()
        self._rows = []

        self.folder = gui.add_folder('Prompt assistant', expand_by_default=True)
        with self.folder:
            self.improve = gui.add_button('Improve prompt', color='gray')
            self.status = gui.add_html('')
            self.questions_folder = gui.add_folder('Clarify your direction', expand_by_default=True)
            self.continue_button = gui.add_button('Continue and refine', color='gray')
            self.explanation = gui.add_html('')
            self.preview = gui.add_text('Refined direction', initial_value='', multiline=True)
            self.use_prompt = gui.add_button('Use prompt', color='green')

        @self.improve.on_click
        def _improve(_event):
            self._start()

        @self.continue_button.on_click
        def _continue(_event):
            self._continue()

        @self.preview.on_update
        def _preview_edited(_event):
            self._preview_revision += 1
            self._render()

        @self.use_prompt.on_click
        def _use(_event):
            self._use_prompt()

        self._render()

    @staticmethod
    def _set(handle, name, value):
        if getattr(handle, name) != value:
            setattr(handle, name, value)

    def _source_matches(self):
        return (self._source_context == self.context() and
                self._source_prompt == self.prompt_handle.value)

    def _invalidate(self):
        self._request_id += 1
        self._source_context = None
        self._source_prompt = None
        self._result = None
        self._answers = {}
        self._history = []
        self._request_pending = False
        self._error = ''
        with self._mutex:
            self._completed = None
        self._set(self.preview, 'value', '')

    def _add_row(self):
        with self.questions_folder:
            label = self._gui.add_html('')
            choice = self._gui.add_dropdown('Answer', (CHOOSE,), initial_value=CHOOSE)
            freeform = self._gui.add_text('Or describe it in your own words', initial_value='')
        row = (label, choice, freeform)
        self._rows.append(row)

        @choice.on_update
        def _choice_changed(_event):
            self._render()

        @freeform.on_update
        def _freeform_changed(_event):
            self._render()

    def _show_result(self, result):
        self._result = result
        self._error = ''
        questions = result.questions
        while len(self._rows) < len(questions):
            self._add_row()
        for index, question in enumerate(questions):
            label, choice, freeform = self._rows[index]
            self._set(label, 'content', f'<div class="sz-note">{escape(question.text)}</div>')
            options = (CHOOSE, *question.options)
            self._set(choice, 'options', options)
            previous = self._answers.get(question.id, '')
            self._set(choice, 'value', previous if previous in question.options else CHOOSE)
            self._set(freeform, 'value', '' if previous in question.options else previous)
        self._set(self.preview, 'value', result.refined_prompt if result.ready else '')
        self._render()

    def _answers_from_controls(self):
        answers = dict(self._answers)
        if self._result is not None:
            for question, (_, choice, freeform) in zip(self._result.questions, self._rows):
                answer = freeform.value.strip() or (choice.value if choice.value != CHOOSE else '')
                if answer:
                    answers[question.id] = answer
                else:
                    answers.pop(question.id, None)
        return answers

    def _valid_source(self):
        return (self._visible and not self._busy and
                isinstance(self.prompt_handle.value, str) and
                1 <= len(self.prompt_handle.value.strip()) <= MAX_PROMPT_LENGTH)

    def _start(self):
        if not self._valid_source() or self._request_pending:
            return
        self._source_context = self.context()
        self._source_prompt = self.prompt_handle.value
        self._answers = {}
        self._history = []
        self._result = None
        self._error = ''
        self._set(self.preview, 'value', '')
        self._launch()

    def _continue(self):
        if (not self._valid_source() or self._request_pending or
                self._result is None or not self._result.questions or
                not self._source_matches()):
            return
        answers = self._answers_from_controls()
        if any(not answers.get(question.id) for question in self._result.questions):
            return
        for question in self._result.questions:
            self._history = [entry for entry in self._history if entry['id'] != question.id]
            self._history.append({'id': question.id, 'question': question.text,
                                  'answer': answers[question.id]})
        self._history = self._history[-6:]
        self._answers = answers
        self._result = None
        self._error = ''
        self._set(self.preview, 'value', '')
        self._launch()

    def _launch(self):
        self._request_id += 1
        request_id = self._request_id
        prompt = self._source_prompt
        answers = dict(self._answers)
        history = tuple(dict(entry) for entry in self._history)
        self._request_pending = True
        self._request_preview_revision = self._preview_revision
        self._render()

        def work():
            try:
                if self._refiner_accepts_history:
                    result = self.refiner(prompt, answers, history=history)
                else:
                    result = self.refiner(prompt, answers)
                error = ''
            except Exception as exc:
                result = None
                error = str(exc) or type(exc).__name__
            with self._mutex:
                if request_id == self._request_id:
                    self._completed = (request_id, result, error)

        Thread(target=work, name='prompt-assistant', daemon=True).start()

    def clarify(self):
        """Show a local clarification before generation; this makes no network call."""
        if not self._valid_source() or self._request_pending:
            return False
        if (self._result is not None and self._result.questions and
                self._source_matches()):
            return True
        self._source_context = self.context()
        self._source_prompt = self.prompt_handle.value
        self._answers = {}
        self._history = []
        try:
            result = self.refiner(self._source_prompt, offline=True)
        except Exception as exc:
            self._error = str(exc) or type(exc).__name__
            self._render()
            return False
        self._show_result(result)
        return bool(result.questions)

    def _use_prompt(self):
        if (not self._valid_source() or self._request_pending or
                self._result is None or not self._result.ready or
                not self._source_matches()):
            return
        text = self.preview.value.strip()
        if not 1 <= len(text) <= MAX_PROMPT_LENGTH:
            return
        # The caller rechecks the context and prompt under its own session
        # lock. The helper never holds its private lock over that callback.
        if self.apply(text, self._source_context, self._source_prompt):
            self._invalidate()
            self._render()
        else:
            self._invalidate()
            self._error = 'The direction changed. Improve the current draft again.'
            self._render()

    def refresh(self, visible, busy):
        """Publish completed work and invalidate drafts for a changed editor."""
        self._visible = bool(visible)
        self._busy = bool(busy)
        if ((not self._visible or self._busy) and self._source_context is not None) or (
                self._source_context is not None and not self._source_matches()):
            self._invalidate()
        with self._mutex:
            completed = self._completed
            self._completed = None
        if completed is not None and completed[0] == self._request_id and self._request_pending:
            self._request_pending = False
            if self._preview_revision != self._request_preview_revision:
                self._error = 'The refined text changed while the assistant worked. Improve the current draft again.'
            elif completed[2]:
                self._error = completed[2]
            else:
                self._show_result(completed[1])
        self._render()

    def _render(self):
        self._set(self.folder, 'visible', self._visible)
        self._set(self.improve, 'disabled', not self._valid_source() or self._request_pending)
        questions = self._result.questions if self._result is not None else ()
        self._set(self.folder, 'label',
                  'Prompt assistant · answer needed' if questions else 'Prompt assistant')
        self._set(self.questions_folder, 'visible', self._visible and bool(questions))
        for index, (label, choice, freeform) in enumerate(self._rows):
            show = self._visible and index < len(questions)
            for handle in (label, choice, freeform):
                self._set(handle, 'visible', show)
            self._set(choice, 'disabled', self._busy or self._request_pending)
            self._set(freeform, 'disabled', self._busy or self._request_pending)
        can_continue = (questions and self._source_context is not None and
                        self._source_matches() and
                        all(self._answers_from_controls().get(q.id) for q in questions))
        self._set(self.continue_button, 'visible', self._visible and bool(questions))
        self._set(self.continue_button, 'disabled', self._busy or self._request_pending or not can_continue)
        ready = self._result is not None and self._result.ready
        self._set(self.preview, 'visible', self._visible and ready)
        self._set(self.preview, 'disabled', self._busy or self._request_pending)
        self._set(self.use_prompt, 'visible', self._visible and ready)
        can_use = (ready and self._source_context is not None and self._source_matches() and
                   1 <= len(self.preview.value.strip()) <= MAX_PROMPT_LENGTH)
        self._set(self.use_prompt, 'disabled', self._busy or self._request_pending or not can_use)
        explanation = ''
        if self._result is not None:
            source = {'ollama': 'Local AI', 'offline': 'Offline guidance',
                      'injected': 'Assistant'}.get(self._result.source, 'Assistant')
            explanation = ' · '.join(part for part in (
                source, self._result.explanation, self._result.warning) if part)
        self._set(self.explanation, 'content',
                  f'<div class="sz-note">{escape(explanation)}</div>' if explanation else '')
        self._set(self.explanation, 'visible', self._visible and bool(explanation))
        status = ('Improving your direction…' if self._request_pending else
                  self._error if self._error else
                  'Answer the questions, then continue.' if questions else '')
        self._set(self.status, 'content',
                  f'<div class="sz-note">{escape(status)}</div>' if status else '')
        self._set(self.status, 'visible', self._visible and bool(status))
