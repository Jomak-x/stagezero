"""Compact full-scene entry point and client-local scene editor."""

from html import escape
import math
from types import SimpleNamespace

from story_workflow import StoryWorkflow


_RUNNING = frozenset(('planning', 'queued', 'running', 'speech_retrying'))
_DEFAULT_PROMPT = ('A person walks forward, stops, waves to a friend, '
                   'then celebrates with both arms raised.')


class StoryControls:
    def __init__(self, gui, session, *, core_session=None, on_story_activate=None):
        self.gui = gui
        self.session = session
        self.core_session = core_session
        self.on_story_activate = on_story_activate
        self.workflow = StoryWorkflow(session)
        self.ids = {}
        self._views = {}
        self._drafts = {}
        self._automatic_attempts = set()
        self._movement_edit = None
        self._closed = False
        self.open_button = gui.add_button('Full scene · one prompt', color='green')
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
        if view.current_take_id is not None:
            with self.session.lock:
                return self.session.takes.get(view.current_take_id)
        return self._loaded_take(data)

    def _close_view(self, view):
        if self._views.get(view.key) is not view:
            return
        self._drafts[view.key] = (view.prompt.value, view.seconds.value, view.jobs.value)
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
        modal = gui.add_modal('Create a full scene', size='xl', show_close_button=False)
        prompt_draft, seconds_draft, selected_job = self._drafts.get(
            key, (_DEFAULT_PROMPT, '60', next(iter(self.ids), 'No scenes yet')))
        view = SimpleNamespace(key=key, modal=modal, error='', action_map={},
                               action_token=None, selected_index=None,
                               current_take_id=None)
        self._views[key] = view
        with modal:
            gui.add_html('<div class="sz-intro"><b>One prompt, a complete performance</b>'
                         'Describe the sequence of movements. Generation can continue while this window is closed. '
                         'The completed take stays editable movement by movement.</div>')
            view.close = gui.add_button('Close', color='gray')
            view.status = gui.add_html('')
            tabs = gui.add_tab_group()
            with tabs.add_tab('Create'):
                view.prompt = gui.add_text('Describe the full scene', initial_value=prompt_draft,
                                           multiline=True)
                view.seconds = gui.add_text('Scene length (seconds, 0.16–120)', initial_value=seconds_draft)
                view.generate = gui.add_button('Generate full scene', color='green')
                view.jobs = gui.add_dropdown('Scene jobs', tuple(self.ids) or ('No scenes yet',))
                if selected_job in self.ids:
                    self._set(view.jobs, 'value', selected_job)
                view.cancel = gui.add_button('Cancel selected scene', color='gray')
                view.load = gui.add_button('Load completed scene', color='gray')
                view.review = gui.add_button('Review scene take', color='gray')
                view.warnings = gui.add_html('')
                with gui.add_folder('Planned movements', expand_by_default=False):
                    view.plan = gui.add_html('')
            with tabs.add_tab('Refine'):
                gui.add_html('<div class="sz-note">Choose a movement and press Select movement. '
                             'Updating it regenerates that movement and reconnects the following movements.</div>')
                view.source = gui.add_html('')
                view.use_current = gui.add_button('Use current take', color='gray')
                view.actions = gui.add_dropdown('Movement', ('No movements yet',))
                view.select_action = gui.add_button('Select movement', color='gray')
                view.action_prompt = gui.add_text('Movement direction', initial_value='', multiline=True)
                view.action_seconds = gui.add_text('Movement length (seconds, 0.16–30)', initial_value='4.16')
                view.edit = gui.add_button('Regenerate movement', color='green')
                view.cancel_edit = gui.add_button('Cancel movement update', color='gray')
                view.undo = gui.add_button('Undo latest movement edit', color='gray')
                view.play = gui.add_button('Play selected movement', color='gray')
                view.pause = gui.add_button('Pause preview', color='gray')

        @view.close.on_click
        def close(_):
            self._close_view(view)

        @view.generate.on_click
        def generate(_):
            if not self._is_current(view):
                return
            try:
                with self.session.lock:
                    core_active = (self.core_session is not None and
                                   self.core_session.snapshot()['active'])
                    if self.session.busy:
                        raise ValueError('Wait for the current motion generation to finish.')
                    if not core_active and not self.session.character_motion_enabled:
                        raise ValueError('Select a motion-ready character before generating a scene.')
                seconds = float(view.seconds.value)
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
            except (ValueError, RuntimeError) as exc:
                view.error = str(exc)
            self.update()

        @view.jobs.on_update
        def jobs(event):
            if event.client is not None and self._is_current(view):
                view.error = ''
                view.current_take_id = None
                view.action_token = None
                view.selected_index = None
                self.update()

        @view.use_current.on_click
        def use_current(_):
            if not self._is_current(view):
                return
            with self.session.lock:
                take = self.session.takes.get(self.session.active_take)
                if take is None:
                    view.error = 'Select a saved take in the project first.'
                elif self.session.busy:
                    view.error = 'Wait for the current motion to finish before reviewing a take.'
                else:
                    self._activate_g1()
                    self.session.set_mode('Live ARDY')
                    self.session.select_take(take.id)
                    view.current_take_id = take.id
                    view.selected_index = None
                    view.action_token = None
                    view.error = ''
            self.update()

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
            identifier, _ = self._data(view)
            if identifier:
                self._load(view, identifier)
            self.update()

        @view.review.on_click
        def review(_):
            if not self._is_current(view):
                return
            _, data = self._data(view)
            take = self._loaded_take(data)
            if take is None:
                view.error = 'The scene take is unavailable. Load the scene again.'
            else:
                with self.session.lock:
                    if self.session.takes.get(take.id) is not take:
                        view.error = 'The scene take changed. Select it again before reviewing.'
                    elif self.session.busy:
                        view.error = 'Wait for the current motion to finish before reviewing.'
                    else:
                        self._activate_g1()
                        self.session.set_mode('Live ARDY')
                        self.session.select_take(take.id)
                        view.current_take_id = None
                        view.action_token = None
                        view.error = ''
            self.update()

        @view.actions.on_update
        def actions(event):
            if event.client is not None and self._is_current(view):
                view.selected_index = view.action_map.get(view.actions.value)
                view.action_token = None
                self._refresh_view(view)

        @view.select_action.on_click
        def select_action(_):
            if not self._is_current(view):
                return
            _, data = self._data(view)
            with self.session.lock:
                take = self._view_take(view, data)
                index = view.action_map.get(view.actions.value)
                if take is None or index is None or not 0 <= index < len(take.segments):
                    view.error = 'Load a scene and choose a movement first.'
                else:
                    view.selected_index = index
                    view.action_token = self._action_token(take, index)
                    segment = take.segments[index]
                    self._set(view.action_prompt, 'value', segment['prompt'])
                    self._set(view.action_seconds, 'value',
                              f'{(segment["end"] - segment["start"])/25:.2f}')
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
                        seconds = float(view.action_seconds.value)
                        if not math.isfinite(seconds) or not 0.16 <= seconds <= 30:
                            raise ValueError('Enter a movement length from 0.16 to 30 seconds.')
                        if not 1 <= len(view.action_prompt.value.strip()) <= 500:
                            raise ValueError('Enter a movement direction (1–500 characters).')
                        self._activate_g1()
                        self.session.set_mode('Live ARDY')
                        self.session.select_take(take.id)
                        if not self.session.submit_action_edit(
                                view.action_prompt.value, index, 'replace', seconds=seconds):
                            raise ValueError(self.session.status)
                        self._movement_edit = {'version': self.session.version,
                                               'take_id': take.id, 'owner': view.key,
                                               'revision': self.session.action_edit_revision,
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
            self.update()

        @view.pause.on_click
        def pause(_):
            if not self._is_current(view):
                return
            with self.session.lock:
                self.session.pause()
            self.update()

        self._refresh_view(view)
        return view

    def _activate_g1(self):
        if self.core_session is not None and self.core_session.snapshot()['active']:
            self.core_session.deactivate()
            if self.on_story_activate is not None:
                self.on_story_activate()

    def _load(self, view, identifier, *, automatic=False):
        try:
            loaded = self.workflow.load(identifier, automatic=automatic)
            if not loaded:
                if automatic:
                    return False
                raise ValueError('This scene is not ready to load yet.')
            # Explicit Load means the user wants to see the G1 result. An
            # automatic load only runs if Native Core is already inactive.
            if not automatic:
                self._activate_g1()
            view.current_take_id = None
            view.action_token = None
            view.error = ''
            return True
        except (ValueError, RuntimeError) as exc:
            view.error = str(exc)
            return False

    @staticmethod
    def _action_token(take, index):
        segment = take.segments[index]
        return (id(take), index, segment['start'], segment['end'], segment['prompt'])

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
        identifier, data = self._data(view)
        job_take = self._loaded_take(data)
        take = self._view_take(view, data)
        movement_status, can_cancel = self._movement_state(view, take)
        state = data['status'] if data else 'ready'
        detail = view.error or ((data.get('error') or data.get('speech_error')) if data else '') or ''
        if data and data.get('loaded'):
            detail = detail or 'Loaded as an editable take. Pick a movement below to refine it.'
        elif data and state in _RUNNING:
            progress = data.get('progress') or {}
            detail = detail or (f'{progress.get("completed_beats", 0)}/{progress.get("total_beats", 0)} movements · '
                                f'{progress.get("completed_chunks", 0)}/{progress.get("total_chunks", 0)} motion chunks')
        if movement_status and not view.error:
            failed = 'failed' in movement_status.lower() or 'stopped' in movement_status.lower()
            style = 'sz-progress error' if failed else 'sz-progress'
            status_content = (f'<div class="{style}" role="status"><b>Movement update</b> '
                              f'{escape(movement_status)}</div>')
        else:
            status_content = (f'<div class="sz-progress" role="status"><b>{escape(state.capitalize())}</b> '
                              f'{escape(str(detail))}</div>')
        self._set(view.status, 'content', status_content)
        self._set(view.cancel, 'disabled', state not in _RUNNING | {'speech_failed'})
        self._set(view.load, 'disabled', state != 'completed' or bool(data and data.get('loaded')))
        core_active = self.core_session is not None and self.core_session.snapshot()['active']
        self._set(view.generate, 'disabled', self.session.busy or
                  (not core_active and not self.session.character_motion_enabled))
        plan = data.get('plan') if data else None
        html = ''
        if plan:
            beats = plan.get('beats') or []
            html = '<div class="sz-note">Planned movements</div><ol>' + ''.join(
                f'<li>{escape(beat["prompt"])} · {beat["seconds"]:g}s</li>' for beat in beats) + '</ol>'
        self._set(view.plan, 'content', html)
        warnings = plan.get('warnings', []) if plan else []
        self._set(view.warnings, 'content', ''.join(
            f'<div class="sz-note">{escape(str(w))}</div>' for w in warnings))
        self._set(view.cancel_edit, 'disabled', not can_cancel)
        with self.session.lock:
            current_take = self.session.takes.get(self.session.active_take)
        self._set(view.use_current, 'disabled', current_take is None or self.session.busy)
        if take is not None:
            source = ('Project take' if view.current_take_id is not None else 'Completed scene job')
            self._set(view.source, 'content',
                      f'<div class="sz-preview"><b>{source}: {escape(take.name)}</b> · '
                      f'{len(take.segments)} movements</div>')
        elif view.current_take_id is not None:
            self._set(view.source, 'content',
                      '<div class="sz-preview invalid">The chosen project take is no longer available. '
                      'Press Use current take to choose another.</div>')
        else:
            self._set(view.source, 'content',
                      '<div class="sz-note">Load a completed scene job, or press Use current take '
                      'to refine a take already in this project.</div>')
        self._set(view.review, 'disabled', job_take is None or self.session.busy)
        if take is None:
            view.action_map = {}
            view.action_token = None
            view.selected_index = None
            self._set(view.actions, 'options', ('No movements yet',))
        else:
            choices = {f'{index + 1:02d} · {segment["start"]/25:.2f}s · {segment["prompt"][:60]}': index
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
                view.action_token = None
            elif view.selected_index is None:
                view.selected_index = choices[previous]
        index = view.selected_index
        token = self._action_token(take, index) if take is not None and index is not None and 0 <= index < len(take.segments) else None
        if view.action_token is not None and token != view.action_token:
            view.action_token = None
        editable = token is not None and token == view.action_token and self.session.character_motion_enabled
        self._set(view.actions, 'disabled', take is None or not take.segments or self.session.busy)
        self._set(view.select_action, 'disabled', take is None or not take.segments or self.session.busy)
        for handle in (view.action_prompt, view.action_seconds, view.edit, view.play):
            self._set(handle, 'disabled', not editable or self.session.busy)
        self._set(view.undo, 'disabled', take is None or self.session.busy or
                  self.session.active_take != take.id or not self._can_undo_scene(take))
        self._set(view.pause, 'disabled', not self.session.playing)

    def update(self):
        if self._closed:
            return
        # Jobs advance independently of any open modal. Automatic loading
        # preserves the simple one-prompt flow when Native Core is inactive.
        for identifier in tuple(self.ids.values()):
            data = self.workflow.snapshot(identifier)
            if (data and data['status'] == 'completed' and not data.get('loaded')
                    and not self.session.busy and
                    (self.core_session is None or not self.core_session.snapshot()['active'])):
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
