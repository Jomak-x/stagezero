"""Viser controls for refining and validating motion directions."""

from copy import deepcopy
from contextlib import contextmanager
from html import escape
from inspect import Parameter, signature
from threading import Lock, Thread

from prompt_assistant import refine_prompt


CHOOSE = 'Choose an answer…'
MAX_PROMPT_LENGTH = 500


class PromptAssistantUI:
    """A draft assistant with optional validated-generation continuation.

    ``context`` returns an equality-comparable token for the target editor,
    take, and clip revision. ``apply`` must atomically recheck that token and
    the original prompt before assigning the approved text. The caller owns
    any session lock; no assistant callback holds its private lock while
    calling either function or while invoking the refiner.
    """

    def __init__(self, gui, prompt_handle, context, apply, refiner=refine_prompt,
                 scene_context=None, auto_apply=False, generate=None):
        self._gui = gui
        self.prompt_handle = prompt_handle
        self.context = context
        self.apply = apply
        self.refiner = refiner
        self.scene_context = scene_context
        self.auto_apply = bool(auto_apply)
        self.generate = generate
        try:
            parameters = signature(refiner).parameters
            accepts_kwargs = any(parameter.kind == Parameter.VAR_KEYWORD
                                 for parameter in parameters.values())
            self._refiner_accepts_history = 'history' in parameters or accepts_kwargs
            self._refiner_accepts_scene_context = 'scene_context' in parameters or accepts_kwargs
            self._refiner_accepts_offline = 'offline' in parameters or accepts_kwargs
            self._refiner_accepts_should_cancel = 'should_cancel' in parameters or accepts_kwargs
        except (TypeError, ValueError):
            self._refiner_accepts_history = True
            self._refiner_accepts_scene_context = True
            self._refiner_accepts_offline = True
            self._refiner_accepts_should_cancel = True
        self._visible = True
        self._busy = False
        self._source_context = None
        self._source_prompt = None
        self._source_scene_context = None
        self._dismissed_source = None
        self._dismissed_prompt_only = None
        self._result = None
        self._clarification_flow = False
        self._answers = {}
        self._history = []
        self._request_pending = False
        self._reservation_token = None
        self._reservation_prompt = None
        self._request_id = 0
        self._source_epoch = 0
        self._completed = None
        self._preview_revision = 0
        self._request_preview_revision = 0
        self._error = ''
        self._notice = ''
        self._applied = None
        self._applying = False
        self._generation_requested = False
        self._generation_submitting = False
        self._generation_committed = False
        self._generation_cancelled = False
        self._generation_status = ''
        self._generation_seen_busy = False
        self._generation_original = ''
        self._generation_used = ''
        self._mutex = Lock()
        self._rows = []

        self.folder = gui.add_folder('Prompt assistant', expand_by_default=False)
        with self.folder:
            self.improve = (gui.add_button('Improve prompt', color='gray')
                            if self.generate is None else None)
            self.status = gui.add_html('')
            self.generation_prompts = gui.add_html('')
            self.retry_button = gui.add_button('Retry improvement', color='gray')
            self.generate_original_button = gui.add_button('Generate original direction', color='green')
            self.cancel_button = gui.add_button('Cancel clarification', color='gray')
            self.questions_folder = gui.add_folder('Clarify your direction', expand_by_default=True)
            self.continue_button = gui.add_button('Continue and refine', color='gray')
            self.explanation = gui.add_html('')
            self.preview = gui.add_text('Refined direction', initial_value='', multiline=True)
            self.use_prompt = gui.add_button('Use prompt', color='green')
            self.revert_button = gui.add_button('Revert original text', color='gray')

        if self.improve is not None:
            @self.improve.on_click
            def _improve(_event):
                self._start()

        @self.continue_button.on_click
        def _continue(_event):
            self._continue()

        @self.retry_button.on_click
        def _retry(_event):
            self._retry()

        @self.generate_original_button.on_click
        def _generate_original(_event):
            self._generate_original()

        @self.cancel_button.on_click
        def _cancel(_event):
            self.cancel()

        @self.preview.on_update
        def _preview_edited(_event):
            self._preview_revision += 1
            self._render()

        @self.use_prompt.on_click
        def _use(_event):
            self._use_prompt()

        @self.revert_button.on_click
        def _revert(_event):
            self._revert_original()

        self._render()

    @staticmethod
    def _set(handle, name, value):
        if getattr(handle, name) != value:
            setattr(handle, name, value)

    def _source_matches(self):
        return self._matches_snapshot(self._source_context, self._source_prompt,
                                      self._source_scene_context)

    def _matches_snapshot(self, context, prompt, scene_context):
        return (context == self.context() and prompt == self.prompt_handle.value and
                scene_context == self._scene_snapshot())

    def _dismissed_matches(self):
        return ((self._dismissed_prompt_only is not None and
                 self._dismissed_prompt_only == self.prompt_handle.value) or
                (self._dismissed_source is not None and
                 self._matches_snapshot(*self._dismissed_source)))

    def _scene_snapshot(self):
        return deepcopy(self.scene_context()) if self.scene_context is not None else None

    def _capture_source(self):
        context, prompt, scene_context = (self.context(), self.prompt_handle.value,
                                          self._scene_snapshot())
        with self._mutex:
            self._source_context = context
            self._source_prompt = prompt
            self._source_scene_context = scene_context
            self._source_epoch += 1

    def _reserve_request(self, expected_id=None):
        with self._mutex:
            if (self._applying or self._request_pending or self._reservation_token is not None or
                    (expected_id is not None and self._request_id != expected_id)):
                return None
            self._reservation_token = self._request_id
            self._reservation_prompt = self.prompt_handle.value
            token = self._reservation_token
        try:
            self._render()
        except Exception:
            self._release_reservation(token)
            raise
        return token

    def _release_reservation(self, token):
        with self._mutex:
            if self._reservation_token == token:
                self._reservation_token = None
                self._reservation_prompt = None

    def _refine(self, prompt, answers, history=(), *, offline=False,
                scene_context=None, should_cancel=None):
        kwargs = {}
        if self._refiner_accepts_history:
            kwargs['history'] = history
        if self._refiner_accepts_scene_context:
            kwargs['scene_context'] = scene_context
        if should_cancel is not None and self._refiner_accepts_should_cancel:
            kwargs['should_cancel'] = should_cancel
        if offline:
            if not self._refiner_accepts_offline:
                raise RuntimeError('Clarification is unavailable for this assistant.')
            kwargs['offline'] = True
        return self.refiner(prompt, answers, **kwargs)

    def _clear_state_locked(self, *, dismissed_source=None, dismissed_prompt_only=None):
        self._request_id += 1
        self._source_epoch += 1
        self._reservation_token = None
        self._reservation_prompt = None
        self._source_context = None
        self._source_prompt = None
        self._source_scene_context = None
        self._dismissed_source = dismissed_source
        self._dismissed_prompt_only = dismissed_prompt_only
        self._result = None
        self._clarification_flow = False
        self._answers = {}
        self._history = []
        self._request_pending = False
        self._error = ''
        self._notice = ''
        self._completed = None
        self._generation_requested = False
        self._generation_status = ''
        self._generation_seen_busy = False

    def _invalidate(self, *, dismissed_source=None, dismissed_prompt_only=None):
        with self._mutex:
            self._clear_state_locked(dismissed_source=dismissed_source,
                                     dismissed_prompt_only=dismissed_prompt_only)
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
        self._error = (result.warning or result.explanation or 'Could not improve the direction. Please retry.')
        if result.ready or result.questions:
            self._error = ''
        questions = result.questions
        if questions:
            self._clarification_flow = True
            if self._generation_requested:
                self._generation_status = 'waiting'
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

    def _answers_from_controls(self, result=None):
        answers = dict(self._answers)
        result = self._result if result is None else result
        if result is not None:
            for question, (_, choice, freeform) in zip(result.questions, self._rows):
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

    def start_generation(self):
        """Refine this draft and submit its validated result after any answers."""
        if (self.generate is None or not self.auto_apply or self._generation_submitting or
                self._generation_requested):
            return False
        if (self._result is not None and self._result.questions and
                self._source_matches()):
            self._generation_cancelled = False
            self._generation_requested = True
            self._generation_status = 'waiting'
            self._render()
            return True
        return self._start(for_generation=True)

    def _start(self, *, for_generation=False):
        if (self._applying or self._generation_submitting or not self._valid_source()
                or self._request_pending):
            return False
        if self._clarification_flow and self._source_matches():
            return False
        token = self._reserve_request()
        if token is None:
            return False
        try:
            context, prompt, scene_context = (self.context(), self.prompt_handle.value,
                                               self._scene_snapshot())
        except Exception as exc:
            with self._mutex:
                if self._reservation_token != token:
                    return
                self._reservation_token = None
                self._reservation_prompt = None
                self._error = str(exc) or type(exc).__name__
            self._render()
            return False
        with self._mutex:
            if self._reservation_token != token or self._request_id != token:
                return False
            self._source_context = context
            self._source_prompt = prompt
            self._source_scene_context = scene_context
            self._source_epoch += 1
            self._answers = {}
            self._history = []
            self._result = None
            self._clarification_flow = False
            self._error = ''
            self._notice = ''
            self._generation_cancelled = False
            self._generation_requested = for_generation
            self._generation_status = 'improving' if for_generation else ''
            if for_generation:
                self._generation_original = prompt
                self._generation_used = ''
                self._generation_seen_busy = False
        self._set(self.preview, 'value', '')
        self._launch(token)
        return True

    def _continue(self):
        with self._mutex:
            expected_id = self._request_id
            question_result = self._result
        if (not self._valid_source() or self._request_pending or
                question_result is None or not question_result.questions or
                not self._source_matches()):
            return
        token = self._reserve_request(expected_id)
        if token is None:
            return
        answers = self._answers_from_controls(question_result)
        if any(not answers.get(question.id) for question in question_result.questions):
            self._release_reservation(token)
            return
        history = list(self._history)
        for question in question_result.questions:
            history = [entry for entry in history if entry['id'] != question.id]
            history.append({'id': question.id, 'question': question.text,
                            'answer': answers[question.id]})
        with self._mutex:
            if (self._reservation_token != token or self._request_id != token or
                    self._result is not question_result):
                if self._reservation_token == token:
                    self._reservation_token = None
                    self._reservation_prompt = None
                return
            self._history = history[-6:]
            self._answers = answers
            self._error = ''
            if self._generation_requested:
                self._generation_status = 'improving'
        self._launch(token)

    def _retry(self):
        with self._mutex:
            expected_id = self._request_id
        if (not self._error or not self._valid_source() or self._request_pending or
                not self._source_matches()):
            return
        if self._result is not None and self._result.questions:
            self._continue()
            return
        token = self._reserve_request(expected_id)
        if token is None:
            return
        with self._mutex:
            if (self._reservation_token != token or self._request_id != token or
                    self._source_context is None or not self._error):
                if self._reservation_token == token:
                    self._reservation_token = None
                    self._reservation_prompt = None
                return
            self._error = ''
        self._launch(token)

    def cancel(self):
        """Discard the assistant draft; the user's source direction stays editable."""
        with self._mutex:
            if self._generation_cancelled:
                return
            if ((self._generation_submitting or
                 (self._generation_requested and self._applying)) and
                    not self._generation_committed):
                self._generation_cancelled = True
                self._generation_status = ''
                self._notice = 'Pending generation cancelled.'
                self._error = ''
                handoff_cancelled = True
            else:
                handoff_cancelled = False
            if not handoff_cancelled and (self._applying or
                    (self._source_context is None and self._reservation_token is None)):
                return
            if not handoff_cancelled:
                dismissed = self._dismissed_source
                dismissed_prompt_only = self._dismissed_prompt_only
                if self._clarification_flow or self._request_pending:
                    dismissed = (self._source_context, self._source_prompt,
                                 deepcopy(self._source_scene_context))
                elif self._reservation_token is not None and self._source_context is None:
                    dismissed_prompt_only = self._reservation_prompt
                self._clear_state_locked(dismissed_source=dismissed,
                                         dismissed_prompt_only=dismissed_prompt_only)
        if not handoff_cancelled:
            self._set(self.preview, 'value', '')
        self._render()

    def _generate_original(self):
        """Let an explicit user choice bypass a failed rewrite, with the usual context checks."""
        if (self.generate is None or not self.auto_apply or not self._valid_source() or
                self._request_pending or self._applying or self._generation_submitting or
                not self._source_matches()):
            return
        with self._mutex:
            if (not self._error or not self._generation_requested or
                    self._source_context is None or self._reservation_token is not None):
                return
            context = self._source_context
            scene_context = deepcopy(self._source_scene_context)
            original = self._source_prompt
            self._generation_submitting = True
            self._generation_committed = False
            self._generation_cancelled = False
            self._clear_state_locked()
            self._generation_original = original
            self._generation_used = original
            self._generation_status = 'generating'
        self._set(self.preview, 'value', '')
        self._render()
        try:
            outcome = self.generate(original, context, scene_context, original,
                                    self._submission_guard)
            submitted, message = outcome if isinstance(outcome, tuple) else (bool(outcome), '')
        except Exception as exc:
            submitted, message = False, str(exc) or type(exc).__name__
        with self._mutex:
            self._generation_submitting = False
            self._generation_committed = False
            if not submitted:
                self._generation_status = ''
                self._error = ('Pending generation cancelled.' if self._generation_cancelled else
                               message or 'Generation was not started. Check the direction and try again.')
        self._render()

    @contextmanager
    def _submission_guard(self):
        """Serialize a late cancellation with the final session submission."""
        with self._mutex:
            allowed = self._generation_submitting and not self._generation_cancelled
            if allowed:
                self._generation_committed = True
            yield allowed

    def blocks_generation(self):
        """Whether this source is awaiting an assistant clarification or preview."""
        return (self._visible and (
            self._generation_requested or self._generation_submitting or
            self._dismissed_matches() or
            (self._reservation_token is not None and
             self._reservation_prompt == self.prompt_handle.value) or
            (self._source_context is not None and self._source_matches() and
             (self._request_pending or self._clarification_flow or
              bool(self._result is not None and self._result.questions)))))

    def generation_in_progress(self):
        """Whether Generate should wait for this active assistant attempt."""
        return (self._visible and (
            self._generation_requested or self._generation_submitting or
            (self._reservation_token is not None and
             self._reservation_prompt == self.prompt_handle.value) or
            (self._source_context is not None and self._source_matches() and
             (self._request_pending or self._clarification_flow or
              bool(self._result is not None and self._result.questions)))))

    def _launch(self, token):
        with self._mutex:
            if self._reservation_token != token or self._request_id != token:
                return
            self._request_id += 1
            request_id = self._request_id
            prompt = self._source_prompt
            answers = dict(self._answers)
            history = tuple(dict(entry) for entry in self._history)
            scene_context = deepcopy(self._source_scene_context)
            self._request_pending = True
            self._reservation_token = None
            self._reservation_prompt = None
            self._request_preview_revision = self._preview_revision
        self._render()

        def work():
            def should_cancel():
                with self._mutex:
                    stale = request_id != self._request_id
                if stale:
                    return True
                try:
                    return not self._matches_snapshot(self._source_context, prompt,
                                                      scene_context)
                except Exception:
                    return True

            try:
                result = self._refine(prompt, answers, history,
                                      scene_context=scene_context,
                                      should_cancel=should_cancel)
                error = ''
            except Exception as exc:
                result = None
                error = str(exc) or type(exc).__name__
            with self._mutex:
                if request_id == self._request_id:
                    self._completed = (request_id, result, error)

        try:
            Thread(target=work, name='prompt-assistant', daemon=True).start()
        except Exception as exc:
            with self._mutex:
                if request_id != self._request_id:
                    return
                self._request_pending = False
                self._error = str(exc) or type(exc).__name__
            self._render()

    def clarify(self):
        """Show a local clarification before generation; this makes no network call."""
        if self._applying or not self._valid_source() or self._request_pending:
            return False
        if (self._result is not None and self._result.questions and
                self._source_matches()):
            return True
        self._capture_source()
        self._answers = {}
        self._history = []
        try:
            result = self._refine(self._source_prompt, None, offline=True,
                                  scene_context=deepcopy(self._source_scene_context))
        except Exception as exc:
            self._error = str(exc) or type(exc).__name__
            self._render()
            return False
        self._show_result(result)
        if self.auto_apply and result.ready and not result.questions:
            self._auto_apply_result(result, self._request_id)
        return bool(result.questions)

    def _use_prompt(self):
        if (self.auto_apply or not self._valid_source() or self._request_pending or
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

    def _applied_matches(self):
        if self._applied is None:
            return False
        context, _original, applied, scene_context = self._applied
        return self._matches_snapshot(context, applied, scene_context)

    @classmethod
    def _context_after_text_change(cls, before, after, original, applied):
        """Allow only the prompt field itself to change in an editor token."""
        if before == after:
            return True
        if isinstance(before, str) and isinstance(after, str):
            return ((before == original and after == applied) or
                    (before == original.strip() and after == applied.strip()))
        if isinstance(before, (tuple, list)) and type(before) is type(after):
            return len(before) == len(after) and all(
                cls._context_after_text_change(old, new, original, applied)
                for old, new in zip(before, after))
        if isinstance(before, dict) and isinstance(after, dict):
            return before.keys() == after.keys() and all(
                cls._context_after_text_change(before[key], after[key], original, applied)
                for key in before)
        return False

    def _auto_apply_result(self, result, request_id):
        with self._mutex:
            if (request_id != self._request_id or self._result is not result or
                    self._applying or self._source_context is None):
                return
            source_epoch = self._source_epoch
            context = self._source_context
            source_prompt = self._source_prompt
            scene_context = self._source_scene_context
            previous_applied = self._applied
            generate_requested = self._generation_requested

        text = result.refined_prompt.strip() if isinstance(result.refined_prompt, str) else ''
        if not 1 <= len(text) <= MAX_PROMPT_LENGTH:
            with self._mutex:
                if (request_id != self._request_id or source_epoch != self._source_epoch or
                        self._result is not result):
                    return
                self._result = None
                self._error = 'The assistant returned an invalid direction. Retry improvement.'
            self._render()
            return
        if (not self._valid_source() or
                not self._matches_snapshot(context, source_prompt, scene_context)):
            return
        original = source_prompt
        if (previous_applied is not None and
                self._matches_snapshot(previous_applied[0], previous_applied[2],
                                       previous_applied[3])):
            original = previous_applied[1]
        with self._mutex:
            if (request_id != self._request_id or source_epoch != self._source_epoch or
                    self._result is not result or self._applied is not previous_applied or
                    self._source_context is not context or
                    self._source_prompt != source_prompt or
                    self._source_scene_context is not scene_context or
                    self._applying or self._request_pending or not self._visible or self._busy):
                return
            self._applying = True
        # Studio's atomic callback calls update(), which re-enters refresh().
        # Keep this result intact until the callback and its nested update end.
        current = False
        safe_to_generate = False
        try:
            try:
                applied = self.apply(text, context, source_prompt)
            except Exception as exc:
                applied = False
                failure = str(exc) or type(exc).__name__
            else:
                failure = 'The direction changed. Improve the current draft again.'
            try:
                post_context = self.context() if applied else None
                post_scene = self._scene_snapshot() if applied else None
                post_prompt = self.prompt_handle.value if applied else None
            except Exception:
                post_context = post_scene = post_prompt = None
            structurally_ready = bool(
                applied and post_prompt == text and post_scene == scene_context and
                self._context_after_text_change(context, post_context, source_prompt, text))
            with self._mutex:
                current = request_id == self._request_id and source_epoch == self._source_epoch
                cancelled = self._generation_cancelled
                safe_to_generate = structurally_ready and not cancelled
                if current and generate_requested and safe_to_generate:
                    self._generation_submitting = True
                    self._generation_committed = False
            if current:
                self._invalidate()
                if applied:
                    self._notice = 'Improved direction applied to the editor.'
                    self._result = result
                    # Studio includes prompt text in its context token.
                    if structurally_ready:
                        self._applied = (post_context, original, text, scene_context)
                    else:
                        self._applied = None
                    if generate_requested:
                        self._generation_original = original
                        self._generation_used = text if safe_to_generate else ''
                        self._generation_status = 'generating' if safe_to_generate else ''
                        if cancelled:
                            self._notice = 'Pending generation cancelled.'
                        elif not safe_to_generate:
                            self._error = 'The direction or scene changed. Generation was not started.'
                else:
                    if cancelled:
                        self._notice = 'Pending generation cancelled.'
                    else:
                        self._error = failure
        finally:
            with self._mutex:
                self._applying = False
        if current:
            self._render()
        if current and generate_requested and safe_to_generate:
            with self._mutex:
                cancelled = self._generation_cancelled
            if cancelled:
                submitted, message = False, 'Pending generation cancelled.'
            else:
                try:
                    outcome = self.generate(text, post_context, scene_context, original,
                                            self._submission_guard)
                    submitted, message = outcome if isinstance(outcome, tuple) else (bool(outcome), '')
                except Exception as exc:
                    submitted, message = False, str(exc) or type(exc).__name__
            with self._mutex:
                self._generation_submitting = False
                self._generation_committed = False
                if not submitted:
                    self._generation_status = ''
                    if self._generation_cancelled:
                        self._notice = 'Pending generation cancelled.'
                    else:
                        self._error = message or 'Generation was not started. Check the direction and try again.'
            self._render()

    def _revert_original(self):
        with self._mutex:
            record = self._applied
            source_epoch = self._source_epoch
            request_id = self._request_id
        if (not self.auto_apply or record is None or not self._valid_source() or
                not self._matches_snapshot(record[0], record[2], record[3])):
            return
        with self._mutex:
            if (self._applied is not record or self._source_epoch != source_epoch or
                    self._request_id != request_id or self._applying or
                    self._request_pending or self._reservation_token is not None or
                    not self._visible or self._busy):
                return
            self._applying = True
        context, original, applied_text, _scene_context = record
        current = False
        try:
            try:
                reverted = self.apply(original, context, applied_text)
            except Exception as exc:
                reverted = False
                failure = str(exc) or type(exc).__name__
            else:
                failure = 'The direction changed. Revert was not applied.'
            with self._mutex:
                current = (self._applied is record and
                           self._source_epoch == source_epoch and
                           self._request_id == request_id)
            if current:
                self._applied = None
                self._invalidate()
                if reverted:
                    self._notice = 'Original direction restored.'
                else:
                    self._error = failure
        finally:
            with self._mutex:
                self._applying = False
        if current:
            self._render()

    def refresh(self, visible, busy):
        """Publish completed work and invalidate drafts for a changed editor."""
        self._visible = bool(visible)
        self._busy = bool(busy)
        if self._generation_status == 'generating':
            if self._busy:
                self._generation_seen_busy = True
            elif getattr(self, '_generation_seen_busy', False) and not self._generation_submitting:
                self._generation_status = ''
        if self._applying:
            return
        if self._applied is not None and not self._applied_matches():
            self._applied = None
            self._notice = ''
            if self._source_context is None:
                self._result = None
        if self._source_context is not None:
            if not self._source_matches():
                if self._generation_requested:
                    self._generation_original = ''
                    self._generation_used = ''
                self._invalidate()
            elif not self._visible or self._busy:
                dismissed = self._dismissed_source
                if self._clarification_flow or self._request_pending:
                    dismissed = (self._source_context, self._source_prompt,
                                 deepcopy(self._source_scene_context))
                self._invalidate(dismissed_source=dismissed,
                                 dismissed_prompt_only=self._dismissed_prompt_only)
        elif self._reservation_token is not None and (not self._visible or self._busy):
            self._invalidate(dismissed_prompt_only=self._reservation_prompt)
        if self._dismissed_source is not None and not self._dismissed_matches():
            self._dismissed_source = None
        if (self._dismissed_prompt_only is not None and
                self._dismissed_prompt_only != self.prompt_handle.value):
            self._dismissed_prompt_only = None
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
                if (self.auto_apply and completed[1].ready and
                        not completed[1].questions):
                    self._auto_apply_result(completed[1], completed[0])
        self._render()

    def _render(self):
        self._set(self.folder, 'visible', self._visible)
        if self.improve is not None:
            self._set(self.improve, 'disabled', not self._valid_source() or
                      self._request_pending or self._reservation_token is not None or
                      self._clarification_flow)
        retryable = bool(self._error and self._source_context is not None and
                         self._valid_source() and self._source_matches())
        self._set(self.retry_button, 'visible', self._visible and retryable)
        self._set(self.retry_button, 'disabled', self._busy or self._request_pending or
                  self._reservation_token is not None)
        self._set(self.cancel_button, 'visible', self._visible and
                  (self._source_context is not None or self._reservation_token is not None or
                   (self._generation_submitting and not self._generation_committed)))
        self._set(self.cancel_button, 'disabled', self._busy)
        questions = self._result.questions if self._result is not None else ()
        recovery = bool(self.generate is not None and self._generation_requested and
                        self._error and self._source_context is not None and
                        self._valid_source() and self._source_matches())
        self._set(self.folder, 'label',
                  'Prompt assistant · action needed' if recovery else
                  'Prompt assistant · answer needed' if questions else 'Prompt assistant')
        self._set(self.generate_original_button, 'visible', self._visible and recovery)
        self._set(self.generate_original_button, 'disabled', self._busy or self._request_pending or
                  self._reservation_token is not None or self._generation_submitting)
        self._set(self.questions_folder, 'visible', self._visible and bool(questions))
        for index, (label, choice, freeform) in enumerate(self._rows):
            show = self._visible and index < len(questions)
            for handle in (label, choice, freeform):
                self._set(handle, 'visible', show)
            self._set(choice, 'disabled', self._busy or self._request_pending or
                      self._reservation_token is not None)
            self._set(freeform, 'disabled', self._busy or self._request_pending or
                      self._reservation_token is not None)
        can_continue = (questions and self._source_context is not None and
                        self._source_matches() and
                        all(self._answers_from_controls().get(q.id) for q in questions))
        self._set(self.continue_button, 'visible', self._visible and bool(questions))
        self._set(self.continue_button, 'disabled', self._busy or self._request_pending or
                  self._reservation_token is not None or not can_continue)
        ready = self._result is not None and self._result.ready and not self.auto_apply
        self._set(self.preview, 'visible', self._visible and ready)
        self._set(self.preview, 'disabled', self._busy or self._request_pending or
                  self._reservation_token is not None)
        self._set(self.use_prompt, 'visible', self._visible and ready)
        can_use = (ready and self._source_context is not None and self._source_matches() and
                   1 <= len(self.preview.value.strip()) <= MAX_PROMPT_LENGTH)
        self._set(self.use_prompt, 'disabled', self._busy or self._request_pending or
                  self._reservation_token is not None or not can_use)
        can_revert = (self.auto_apply and self._applied is not None and
                      self._applied_matches())
        self._set(self.revert_button, 'visible', self._visible and bool(can_revert))
        self._set(self.revert_button, 'disabled', self._busy or self._request_pending or
                  self._reservation_token is not None or not can_revert)
        explanation = ''
        if self._result is not None and (self._result.ready or self._result.questions):
            source = {'ollama': 'Local AI', 'offline': 'Offline guidance',
                      'injected': 'Assistant'}.get(self._result.source, 'Assistant')
            explanation = ' · '.join(part for part in (
                source, self._result.explanation, self._result.warning) if part)
        self._set(self.explanation, 'content',
                  f'<div class="sz-note">{escape(explanation)}</div>' if explanation else '')
        self._set(self.explanation, 'visible', self._visible and bool(explanation))
        prompts = ''
        if self._generation_original:
            prompts = (f'<div class="sz-note"><strong>Original direction:</strong> '
                       f'{escape(self._generation_original)}</div>')
            if self._generation_used:
                prompts += (f'<div class="sz-note"><strong>Validated direction used:</strong> '
                            f'{escape(self._generation_used)}</div>')
        self._set(self.generation_prompts, 'content', prompts)
        self._set(self.generation_prompts, 'visible', self._visible and bool(prompts))
        prompt = self.prompt_handle.value
        source_hint = ('Enter a direction to improve.' if isinstance(prompt, str) and
                       not prompt.strip() else
                       f'Keep the direction under {MAX_PROMPT_LENGTH} characters.' if
                       isinstance(prompt, str) and len(prompt.strip()) > MAX_PROMPT_LENGTH else '')
        status = ('Generating motion from the validated direction…' if
                  self._generation_status == 'generating' else
                  'Improving your direction…' if
                  self._request_pending or self._reservation_token is not None else
                  self._error if self._error else
                  self._notice if self._notice else
                  'Waiting for your answer; motion generation will continue automatically.'
                  if questions and self._generation_requested else
                  'Answer the questions, then continue.' if questions else
                  'Use the clarified direction or edit the original before generating.'
                  if self._dismissed_matches() else source_hint)
        self._set(self.status, 'content',
                  f'<div class="sz-note" role="status" aria-live="polite">'
                  f'{escape(status)}</div>' if status else '')
        self._set(self.status, 'visible', self._visible and bool(status))
