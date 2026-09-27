"""Compact full-scene entry point and client-local scene editor."""

from html import escape
import math
from types import SimpleNamespace

from studio_navigation import navigate_tab
from story_workflow import StoryWorkflow


_RUNNING = frozenset(('planning', 'queued', 'running', 'speech_retrying'))
_LENGTHS = {'15 seconds': 15, '30 seconds': 30, '60 seconds': 60,
            '120 seconds': 120}
_CURRENT = 'Current scene'


class StoryControls:
    def __init__(self, gui, session, *, core_session=None, paired_session=None, cast_session=None, on_story_activate=None):
        self.gui = gui
        self.session = session
        self.core_session = core_session
        self.paired_session = paired_session
        self.cast_session = cast_session
        self.on_story_activate = on_story_activate
        self.workflow = StoryWorkflow(session)
        self.ids = {}
        self._views = {}
        self._drafts = {}
        self._automatic_attempts = set()
        self._movement_edit = None
        self._closed = False
        self.open_button = gui.add_button('Full scene')
        self.sidebar_status = gui.add_html('')

        @self.open_button.on_click
        def open_scene(event):
            self.open(event.client)

    @staticmethod
    def _set(handle, key, value):
        if getattr(handle, key) != value:
            setattr(handle, key, value)

    @staticmethod
    def _client_key(client):
        return getattr(client, 'client_id', None) or id(client)

    def _data(self, view):
        identifier = self.ids.get(view.jobs.value)
        return (identifier, self.workflow.snapshot(identifier)) if identifier else (None, None)

    def _is_current(self, view):
        return not self._closed and self._views.get(view.key) is view

    def _loaded_take(self, data):
        if not data or not data.get('loaded'):
            return None
        with self.session.lock:
            return self.session.takes.get(data.get('take_id'))

    def _view_take(self, view, data):
        if view.jobs.value == _CURRENT and view.current_take_id is not None:
            with self.session.lock:
                if self.session.active_take != view.current_take_id:
                    return None
                return self.session.takes.get(view.current_take_id)
        return self._loaded_take(data)

    def _scene_options(self, current_take_id=None):
        with self.session.lock:
            has_current = (self.session.active_take in self.session.takes or
                           current_take_id is not None)
        options = ((_CURRENT,) if has_current else ()) + tuple(self.ids)
        return options or ('No scenes yet',)

    def _close_view(self, view):
        if self._views.get(view.key) is not view:
            return
        self._drafts[view.key] = (view.prompt.value, view.length.value,
                                  view.seconds.value, view.jobs.value)
        self._views.pop(view.key)
        view.modal.close()

    def open(self, client=None):
        if self._closed:
            return
        key = self._client_key(client)
        old = self._views.get(key)
        if old is not None:
            self._close_view(old)
        # A client's GuiApi sends the modal only to that viewer. Server GuiApi
        # is retained as a fallback for environments without a client handle.
        gui = getattr(client, 'gui', None) or self.gui
        modal = gui.add_modal('Full scene', size='md', show_close_button=False)
        with self.session.lock:
            current_take_id = (self.session.active_take
                               if self.session.active_take in self.session.takes else None)
        options = self._scene_options(current_take_id)
        prompt_draft, length_draft, seconds_draft, selected_job = self._drafts.get(
            key, ('', 'Auto', '30', next(reversed(self.ids), _CURRENT if current_take_id else 'No scenes yet')))
        view = SimpleNamespace(key=key, modal=modal, error='', action_map={},
                               action_token=None, selected_index=None,
                               current_take_id=current_take_id, bound_take=None,
                               rebound_edit_revision=None)
        self._views[key] = view
        with modal:
            view.status = gui.add_html('')
            view.estimate = gui.add_html('')
            view.tabs = gui.add_tab_group()
            with view.tabs.add_tab('Create'):
                view.prompt = gui.add_text('What happens?', initial_value=prompt_draft,
                                           multiline=True,
                                           hint='Example: Walk forward, wave, then turn and sit.')
                view.length = gui.add_dropdown('Length', ('Auto',) + tuple(_LENGTHS) + ('Custom',),
                                               initial_value=length_draft)
                view.seconds = gui.add_text('Seconds', initial_value=seconds_draft,
                                             hint='0.16–120')
                view.generate = gui.add_button('Generate scene')
                view.jobs = gui.add_dropdown('Scenes', options)
                if selected_job in options:
                    self._set(view.jobs, 'value', selected_job)
                view.cancel = gui.add_button('Cancel generation', color='gray')
                view.review = gui.add_button('Review scene')
                view.load = gui.add_button('Edit movements', color='gray')
                with gui.add_folder('Details', expand_by_default=False) as details:
                    view.details = details
                    view.warnings = gui.add_html('')
                    view.plan = gui.add_html('')
            with view.tabs.add_tab('Refine'):
                view.source = gui.add_html('')
                view.actions = gui.add_dropdown('Movement', ('No movements yet',))
                view.refresh_action = gui.add_button('Refresh movement', color='gray')
                view.action_prompt = gui.add_text('Change this movement', initial_value='', multiline=True)
                with gui.add_folder('Timing', expand_by_default=False) as timing:
                    view.timing = timing
                    view.action_length = gui.add_dropdown('Length', ('Auto', 'Custom'),
                                                          initial_value='Auto')
                    view.action_seconds = gui.add_text('Seconds', initial_value='4.16',
                                                        hint='0.16–30')
                view.edit = gui.add_button('Update movement')
                view.cancel_edit = gui.add_button('Cancel update', color='gray')
                view.undo = gui.add_button('Undo update', color='gray')
                view.play = gui.add_button('Preview movement', color='gray')
            view.close = gui.add_button('Close', color='gray')

        @view.close.on_click
        def close(_):
            self._close_view(view)

        @view.generate.on_click
        def generate(_):
            if not self._is_current(view):
                return
            try:
                if not 1 <= len(view.prompt.value.strip()) <= 2000:
                    raise ValueError('Describe what happens in the scene.')
                with self.session.lock:
                    native_active = self._native_active()
                    if self.session.busy:
                        raise ValueError('Wait for the current motion generation to finish.')
                    if not native_active and not self.session.character_motion_enabled:
                        raise ValueError('Select a motion-ready character before generating a scene.')
                if view.length.value == 'Auto':
                    seconds = None
                else:
                    seconds = (_LENGTHS[view.length.value] if view.length.value in _LENGTHS
                               else float(view.seconds.value))
                    if not math.isfinite(seconds) or not 0.16 <= seconds <= 120:
                        raise ValueError('Enter a scene length from 0.16 to 120 seconds.')
                identifier = self.workflow.submit(view.prompt.value, seconds=seconds)
                label = f'{len(self.ids) + 1:02d} · {view.prompt.value.strip()[:52]}'
                self.ids[label] = identifier
                self._set(view.jobs, 'options', tuple(self.ids))
                self._set(view.jobs, 'value', label)
                view.error = ''
                view.current_take_id = None
                view.action_token = None
                view.bound_take = None
            except (ValueError, RuntimeError) as exc:
                view.error = str(exc)
            self.update()

        @view.jobs.on_update
        def jobs(event):
            if event.client is not None and self._is_current(view):
                view.error = ''
                with self.session.lock:
                    view.current_take_id = (self.session.active_take
                                            if view.jobs.value == _CURRENT else None)
                view.action_token = None
                view.selected_index = None
                view.bound_take = None
                self.update()

        @view.length.on_update
        def length(event):
            if event.client is not None and self._is_current(view):
                self._refresh_view(view)

        @view.cancel.on_click
        def cancel(_):
            if not self._is_current(view):
                return
            identifier, data = self._data(view)
            if identifier and data and data['status'] in _RUNNING | {'speech_failed'}:
                self.workflow.cancel(identifier)
            self.update()

        @view.load.on_click
        def load(_):
            if not self._is_current(view):
                return
            with self.session.lock:
                busy = self.session.busy
            if busy:
                view.error = 'Wait for the current motion to finish before editing.'
                self.update()
                return
            identifier, data = self._data(view)
            if identifier and data and data['status'] == 'completed' and not data.get('loaded'):
                if not self._load(view, identifier):
                    self.update()
                    return
            if self._view_take(view, self._data(view)[1]) is not None:
                navigate_tab(view.tabs, 1, _.client)
                view.error = ''
            else:
                view.error = 'Select a ready scene before editing movements.'
            self.update()

        @view.review.on_click
        def review(_):
            if not self._is_current(view):
                return
            identifier, data = self._data(view)
            if identifier and data and data['status'] == 'completed' and not data.get('loaded'):
                if not self._load(view, identifier):
                    self.update()
                    return
                _, data = self._data(view)
            take = self._view_take(view, data)
            if take is None:
                view.error = 'The scene take is unavailable. Load the scene again.'
            else:
                with self.session.lock:
                    if self.session.takes.get(take.id) is not take:
                        view.error = 'The scene take changed. Select it again before reviewing.'
                    elif self.session.busy:
                        view.error = 'Wait for the current motion to finish before reviewing.'
                    elif self._paired_activation_error():
                        view.error = self._paired_activation_error()
                    else:
                        self._activate_g1()
                        self.session.set_mode('Live ARDY')
                        self.session.select_take(take.id)
                        view.current_take_id = None
                        view.action_token = None
                        view.error = ''
                        self._close_view(view)
            self.update()

        @view.actions.on_update
        def actions(event):
            if event.client is not None and self._is_current(view):
                view.selected_index = view.action_map.get(view.actions.value)
                _, data = self._data(view)
                with self.session.lock:
                    self._bind_action(view, self._view_take(view, data), view.selected_index)
                view.error = ''
                self._refresh_view(view)

        @view.action_length.on_update
        def action_length(event):
            if event.client is not None and self._is_current(view):
                self._refresh_view(view)

        @view.refresh_action.on_click
        def refresh_action(_):
            if not self._is_current(view):
                return
            _, data = self._data(view)
            with self.session.lock:
                take = self._view_take(view, data)
                self._bind_action(view, take, view.selected_index)
            view.error = ''
            self._refresh_view(view)

        @view.edit.on_click
        def edit(_):
            if not self._is_current(view):
                return
            _, data = self._data(view)
            take = self._view_take(view, data)
            with self.session.lock:
                index = view.selected_index
                if (take is None or self.session.takes.get(take.id) is not take
                        or index is None or not 0 <= index < len(take.segments)
                        or view.action_token is None
                        or self._action_token(take, index) != view.action_token):
                    view.error = 'The selected movement changed. Select it again before updating.'
                elif self.session.busy:
                    view.error = 'Wait for the current motion to finish before updating.'
                else:
                    try:
                        seconds = None
                        if view.action_length.value == 'Custom':
                            try:
                                seconds = float(view.action_seconds.value)
                            except ValueError as exc:
                                raise ValueError('Enter a movement length from 0.16 to 30 seconds.') from exc
                            if not math.isfinite(seconds) or not 0.16 <= seconds <= 30:
                                raise ValueError('Enter a movement length from 0.16 to 30 seconds.')
                        if not 1 <= len(view.action_prompt.value.strip()) <= 500:
                            raise ValueError('Enter a movement direction (1–500 characters).')
                        self._activate_g1()
                        self.session.set_mode('Live ARDY')
                        self.session.select_take(take.id)
                        if not self.session.submit_action_edit(
                                view.action_prompt.value, index, 'replace', seconds=seconds,
                                automatic_timing=view.action_length.value == 'Auto'):
                            raise ValueError(self.session.status)
                        self._movement_edit = {'version': self.session.version,
                                               'take_id': take.id, 'owner': view.key,
                                               'revision': self.session.action_edit_revision,
                                               'index': index,
                                               'prompt': view.action_prompt.value.strip(),
                                               'result': None}
                        view.error = ''
                    except ValueError as exc:
                        view.error = str(exc)
            self.update()

        @view.cancel_edit.on_click
        def cancel_edit(_):
            if not self._is_current(view):
                return
            _, data = self._data(view)
            with self.session.lock:
                take = self._view_take(view, data)
                edit = self._movement_edit
                if (take is None or edit is None or edit['owner'] != view.key
                        or edit['take_id'] != take.id or not self.session.busy
                        or self.session.version != edit['version']
                        or self.session.active_take != take.id):
                    view.error = 'No movement update from this popup is running.'
                else:
                    self.session.seek(self.session.frame)
                    self.session.status = 'Movement update cancelled · original take preserved'
                    edit['result'] = self.session.status
                    view.error = ''
            self.update()

        @view.undo.on_click
        def undo(_):
            if not self._is_current(view):
                return
            _, data = self._data(view)
            take = self._view_take(view, data)
            with self.session.lock:
                if (take is None or self.session.takes.get(take.id) is not take
                        or self.session.active_take != take.id or self.session.busy
                        or not self._can_undo_scene(take)):
                    view.error = 'Select this scene take and wait for motion to finish before undoing.'
                elif not self.session.undo_action_edit():
                    view.error = self.session.status
                else:
                    view.error = ''
                    if self._movement_edit is not None and self._movement_edit['take_id'] == take.id:
                        self._movement_edit['result'] = 'Movement edit undone · previous take restored'
            self.update()

        @view.play.on_click
        def play(_):
            if not self._is_current(view):
                return
            _, data = self._data(view)
            take = self._view_take(view, data)
            with self.session.lock:
                index = view.selected_index
                if (take is None or self.session.takes.get(take.id) is not take
                        or index is None or not 0 <= index < len(take.segments)
                        or self.session.busy or not self.session.character_motion_enabled):
                    view.error = 'Select a loaded movement and wait for motion to finish.'
                else:
                    self._activate_g1()
                    self.session.set_mode('Live ARDY')
                    self.session.select_take(take.id)
                    self.session.seek(take.segments[index]['start'])
                    self.session.play()
                    view.error = ''
                    self._close_view(view)
            self.update()

        self._refresh_view(view)
        return view

    def owns_take(self, take_id):
        """Recognize transient scene jobs, including legacy edits without beat IDs."""
        for identifier in tuple(self.ids.values()):
            data = self.workflow.snapshot(identifier)
            if data.get('loaded') and data.get('take_id') == take_id:
                return True
        return False

    def open_for_movement(self, client, take_id, index):
        """Open the exact active scene movement requested by a timeline click."""
        owned = self.owns_take(take_id)
        with self.session.lock:
            take = self.session.takes.get(take_id)
            if (take is None or self.session.active_take != take_id or
                    not isinstance(index, int) or not 0 <= index < len(take.segments) or
                    not (owned or take.segments[index].get('beat_id'))):
                return False
        view = self.open(client)
        if view is None:
            return False
        with self.session.lock:
            if (self.session.active_take != take_id or
                    self.session.takes.get(take_id) is not take or
                    index >= len(take.segments) or
                    not (owned or take.segments[index].get('beat_id'))):
                self._close_view(view)
                return False
            self._set(view.jobs, 'value', _CURRENT)
            view.current_take_id = take_id
            view.action_token = None
            view.bound_take = None
            view.selected_index = index
        self._refresh_view(view)
        choice = next((label for label, value in view.action_map.items()
                       if value == index), None)
        if choice is None:
            self._close_view(view)
            return False
        with self.session.lock:
            if self.session.active_take != take_id or self.session.takes.get(take_id) is not take:
                self._close_view(view)
                return False
            self._set(view.actions, 'value', choice)
            self._bind_action(view, take, index)
        navigate_tab(view.tabs, 1, client)
        return True

    def _native_active(self):
        return any(native is not None and native.snapshot().get('active', False)
                   for native in (self.core_session, self.paired_session, self.cast_session))

    def _paired_activation_error(self):
        for label, native in (('paired', self.paired_session), ('cast', self.cast_session)):
            state = native.snapshot() if native is not None else {}
            if state.get('capturing'):
                return f'Wait for {label} playback export to finish before loading this scene.'
            if state.get('busy'):
                return f'Finish or cancel {label} generation before loading this scene.'
        return None

    def _activate_g1(self):
        error = self._paired_activation_error()
        if error:
            raise ValueError(error)
        if self.core_session is not None and self.core_session.snapshot()['active']:
            self.core_session.deactivate()
        # The viewer also owns paired visibility and motion enablement. Its
        # explicit handoff is needed even when Native Core was never active.
        if self.on_story_activate is not None:
            self.on_story_activate()

    def _load(self, view, identifier, *, automatic=False):
        try:
            error = self._paired_activation_error()
            if error:
                if automatic:
                    return False
                raise ValueError(error)
            if automatic and self._native_active():
                return False
            loaded = self.workflow.load(identifier, automatic=automatic)
            if not loaded:
                if automatic:
                    return False
                raise ValueError('This scene is not ready to load yet.')
            # Explicit Load means the user wants to see the G1 result. An
            # automatic load only runs if both native modes are inactive.
            if not automatic:
                self._activate_g1()
            view.current_take_id = None
            view.action_token = None
            view.bound_take = None
            view.error = ''
            return True
        except (ValueError, RuntimeError) as exc:
            view.error = str(exc)
            return False

    @staticmethod
    def _action_token(take, index):
        segment = take.segments[index]
        return (id(take), index, segment['start'], segment['end'], segment['prompt'])

    def _bind_action(self, view, take, index):
        if take is None or index is None or not 0 <= index < len(take.segments):
            view.action_token = None
            return
        segment = take.segments[index]
        view.action_token = self._action_token(take, index)
        view.bound_take = take
        self._set(view.action_length, 'value', 'Auto')
        self._set(view.action_prompt, 'value', segment['prompt'])
        self._set(view.action_seconds, 'value',
                  f'{(segment["end"] - segment["start"])/25:.2f}')

    def _can_undo_scene(self, take):
        undo = getattr(self.session, '_undo_action_edit', None)
        return (take is not None and undo is not None and
                undo[0].id == take.id and self.session.can_undo_action_edit)

    def _movement_state(self, view, take):
        """Report only an edit submitted through this popup and its final outcome."""
        with self.session.lock:
            edit = self._movement_edit
            if edit is None or take is None or edit['take_id'] != take.id:
                return None, False
            active = self.session.busy and self.session.version == edit['version']
            if active:
                return self.session.status, edit['owner'] == view.key
            if edit['result'] is None:
                if self.session.action_edit_revision > edit['revision']:
                    edit['result'] = ('Movement updated · Undo is available'
                                      if not self.session.status.startswith('Regenerated')
                                      else self.session.status)
                elif self.session.version != edit['version']:
                    edit['result'] = 'Movement update cancelled · original take preserved'
                else:
                    edit['result'] = (self.session.status
                                      if self.session.status.startswith('Action edit failed')
                                      else 'Movement update stopped · original take preserved')
            return edit['result'], False

    def _refresh_view(self, view):
        if self._views.get(view.key) is not view:
            return
        with self.session.lock:
            active_take_id = (self.session.active_take
                              if self.session.active_take in self.session.takes else None)
        options = self._scene_options(active_take_id)
        selected = view.jobs.value
        self._set(view.jobs, 'options', options)
        if selected not in options:
            self._set(view.jobs, 'value', next(reversed(self.ids),
                                               _CURRENT if active_take_id else 'No scenes yet'))
            view.action_token = None
            view.selected_index = None
            view.bound_take = None
        if view.jobs.value == _CURRENT and view.current_take_id != active_take_id:
            previous_take_id = view.current_take_id
            view.current_take_id = active_take_id
            view.action_token = None
            view.selected_index = None
            if previous_take_id is not None and active_take_id is not None:
                view.error = 'Current scene changed. Refresh movement before updating.'
        identifier, data = self._data(view)
        take = self._view_take(view, data)
        movement_status, can_cancel = self._movement_state(view, take)
        state = data['status'] if data else 'ready'
        detail = view.error or ((data.get('error') or data.get('speech_error')) if data else '') or ''
        if data and state in _RUNNING:
            progress = data.get('progress') or {}
            completed = progress.get('completed_beats', 0)
            total = progress.get('total_beats', 0)
            attempt = data.get('attempt', 1)
            label = (f'Retrying scene · attempt {attempt}/{data.get("max_attempts", 3)}'
                     if attempt > 1 else 'Creating scene')
            detail = detail or (f'{label} · {completed}/{total} movements' if total
                                else label + '…')
        if movement_status and not view.error:
            failed = 'failed' in movement_status.lower() or 'stopped' in movement_status.lower()
            style = 'sz-progress error' if failed else 'sz-progress'
            if self.session.busy:
                message = 'Updating movement…'
            elif failed:
                message = movement_status
            elif 'cancelled' in movement_status.lower():
                message = 'Update cancelled'
            elif 'undone' in movement_status.lower():
                message = 'Update undone'
            else:
                message = 'Movement updated'
            status_content = f'<div class="{style}" role="status">{escape(message)}</div>'
        elif view.error or detail:
            style = 'sz-progress error' if view.error or state in ('failed', 'speech_failed') else 'sz-progress'
            status_content = f'<div class="{style}" role="status">{escape(str(detail))}</div>'
        elif state == 'completed':
            status_content = '<div class="sz-progress" role="status">Scene ready</div>'
        elif state == 'cancelled':
            status_content = '<div class="sz-progress" role="status">Generation cancelled</div>'
        else:
            status_content = ''
        self._set(view.status, 'content', status_content)
        self._set(view.cancel, 'disabled', state not in _RUNNING | {'speech_failed'})
        self._set(view.cancel, 'visible', state in _RUNNING | {'speech_failed'})
        ready = take is not None or bool(data and state == 'completed')
        self._set(view.load, 'disabled', not ready or self.session.busy)
        self._set(view.load, 'visible', ready)
        self._set(view.jobs, 'visible', len(options) > 1)
        self._set(view.seconds, 'visible', view.length.value == 'Custom')
        native_active = self._native_active()
        self._set(view.generate, 'disabled', self.session.busy or
                  (not native_active and not self.session.character_motion_enabled))
        plan = data.get('plan') if data else None
        beats = (plan.get('beats') or []) if plan else []
        if take is not None and take.segments:
            frames = sum(segment['end'] - segment['start'] for segment in take.segments)
            count = len(take.segments)
            label = 'Current scene:'
        elif beats:
            frames = sum(round(beat['seconds'] * 25) for beat in beats)
            count = len(beats)
            label = 'Estimated'
        else:
            frames = 0
        if frames:
            duration = f'{frames / 25:.2f}'.rstrip('0').rstrip('.')
            self._set(view.estimate, 'content',
                      f'<div class="sz-note">{label} {duration}s · {count} '
                      f'{"movement" if count == 1 else "movements"}</div>')
        else:
            self._set(view.estimate, 'content', '')
        html = ''
        if plan:
            html = '<div class="sz-note">Planned movements</div><ol>' + ''.join(
                f'<li>{escape(beat["prompt"])} · {beat["seconds"]:g}s</li>' for beat in beats) + '</ol>'
        self._set(view.plan, 'content', html)
        warnings = plan.get('warnings', []) if plan else []
        self._set(view.warnings, 'content', ''.join(
            f'<div class="sz-note">{escape(str(w))}</div>' for w in warnings))
        self._set(view.details, 'visible', bool(plan or warnings))
        self._set(view.cancel_edit, 'disabled', not can_cancel)
        self._set(view.cancel_edit, 'visible', can_cancel)
        if take is not None:
            self._set(view.source, 'content',
                      f'<div class="sz-preview"><b>{escape(take.name)}</b> · '
                      f'{len(take.segments)} movements</div>')
        elif view.current_take_id is not None:
            self._set(view.source, 'content',
                      '<div class="sz-preview invalid">This take is no longer available.</div>')
        else:
            self._set(view.source, 'content',
                      '<div class="sz-note">Generate a scene to refine its movements.</div>')
        self._set(view.review, 'disabled', not ready or self.session.busy)
        self._set(view.review, 'visible', ready)
        if take is None:
            view.action_map = {}
            view.action_token = None
            view.selected_index = None
            self._set(view.actions, 'options', ('No movements yet',))
        else:
            choices = {f'{index + 1:02d} · {(segment["end"] - segment["start"])/25:.2f}s long · {segment["prompt"][:60]}': index
                       for index, segment in enumerate(take.segments)}
            view.action_map = choices
            options = tuple(choices) or ('No movements yet',)
            previous = view.actions.value
            self._set(view.actions, 'options', options)
            if previous not in choices:
                index = (view.selected_index if view.selected_index is not None and
                         0 <= view.selected_index < len(choices) else 0)
                self._set(view.actions, 'value', options[index])
                view.selected_index = choices.get(options[index])
            elif view.selected_index is None:
                view.selected_index = choices[previous]
        index = view.selected_index
        token = self._action_token(take, index) if take is not None and index is not None and 0 <= index < len(take.segments) else None
        edit = self._movement_edit
        own_completed_edit = (edit is not None and take is not None and
                              edit['owner'] == view.key and edit['take_id'] == take.id and
                              edit['index'] == index and
                              self.session.action_edit_revision == edit['revision'] + 1 and
                              not self.session.busy and token != view.action_token and
                              view.rebound_edit_revision != edit['revision'] and
                              take.segments[index]['prompt'] == edit['prompt'])
        if own_completed_edit:
            self._bind_action(view, take, index)
            view.rebound_edit_revision = edit['revision']
        elif take is not None and view.bound_take is None and token is not None:
            self._bind_action(view, take, index)
        if view.action_token is not None and token != view.action_token:
            view.action_token = None
        editable = token is not None and token == view.action_token and self.session.character_motion_enabled
        self._set(view.refresh_action, 'visible', take is not None and token is not None and
                  view.action_token is None and not self.session.busy)
        self._set(view.refresh_action, 'disabled', take is None or self.session.busy)
        self._set(view.actions, 'disabled', take is None or not take.segments or self.session.busy)
        self._set(view.actions, 'visible', take is not None and bool(take.segments))
        self._set(view.timing, 'visible', take is not None and bool(take.segments))
        self._set(view.action_seconds, 'visible', view.action_length.value == 'Custom')
        for handle in (view.action_prompt, view.action_seconds, view.edit, view.play):
            self._set(handle, 'disabled', not editable or self.session.busy)
        for handle in (view.action_prompt, view.action_seconds, view.edit, view.play):
            self._set(handle, 'visible', take is not None and bool(take.segments))
        self._set(view.undo, 'disabled', take is None or self.session.busy or
                  self.session.active_take != take.id or not self._can_undo_scene(take))
        self._set(view.undo, 'visible', take is not None and not self.session.busy and
                  self.session.active_take == take.id and self._can_undo_scene(take))

    def update(self):
        if self._closed:
            return
        # Jobs advance independently of any open modal. Automatic loading
        # preserves the one-prompt flow without replacing either native mode.
        for identifier in tuple(self.ids.values()):
            data = self.workflow.snapshot(identifier)
            if (data and data['status'] == 'completed' and not data.get('loaded')
                    and not self.session.busy and
                    not self._native_active() and not self._paired_activation_error()):
                attempt = (identifier, self.session.project_revision)
                if attempt not in self._automatic_attempts:
                    self._automatic_attempts.add(attempt)
                    try:
                        self.workflow.load(identifier, automatic=True)
                    except (ValueError, RuntimeError):
                        pass  # The explicit Load action surfaces the reason.
        for view in tuple(self._views.values()):
            self._refresh_view(view)
        running = sum((self.workflow.snapshot(identifier) or {}).get('status') in _RUNNING
                      for identifier in self.ids.values())
        completed = sum((self.workflow.snapshot(identifier) or {}).get('status') == 'completed'
                        for identifier in self.ids.values())
        summary = (f'{running} generating · {completed} completed' if self.ids else
                   'Build a sequence up to 120 seconds, then refine each movement.')
        self._set(self.sidebar_status, 'content', f'<div class="sz-note">{escape(summary)}</div>')

    def close(self):
        if self._closed:
            return
        self._closed = True
        for view in tuple(self._views.values()):
            self._close_view(view)
        self.workflow.close()
