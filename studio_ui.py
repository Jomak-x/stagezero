"""Studio controls built on public Viser GUI handles."""
from html import escape
import json
import math
import time
import viser
from takes import MAX_TAKES
from duration_planning import plan_duration
from studio_guide import GUIDE_HTML
from studio_navigation import navigate_tab
from prompt_assistant import needs_clarification
from prompt_assistant_ui import PromptAssistantUI
from scene_targets import resolve_targets
from upload_events import install_upload_snapshots

CREATE = 'Create new'
EXTEND = 'Add action to end'
REPLACE = 'Change ending (new version)'
AUTO = 'Auto'
SET_DURATION = 'Set new motion length'
TARGET_TOTAL = 'Set total scene length'
ROUTINE_STATUS_PREFIXES = (
    'Recorded playback', 'Ready for an instruction', 'Instruction changed',
    'Recorded generated take', 'Paused recorded preview', 'Paused at playhead',
    'Replaying stored motion', 'Rewound', 'Reset ·', 'New take ·',
)

STYLE = """<style>
:root { --mantine-font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.mantine-Paper-root { background: #111923; }
.mantine-ScrollArea-viewport { scrollbar-color: #344453 #111923; }
.mantine-Tabs-list { padding: 0 10px; border-bottom: 1px solid #2a3644; gap: 0; }
.mantine-Tabs-tab { flex: 1; padding: 9px 5px !important; font-size: 12px; font-weight: 600; color: #8fa3b8; }
.mantine-Tabs-tab[data-active] { color: #87e8cd; border-color: #87e8cd; background: #182b30; }
.mantine-Tabs-panel { padding-top: 6px; }
.mantine-Button-root { min-height: 32px; border-radius: 7px; transition: background 120ms; }
.mantine-Button-root[data-variant=outline] { border-color: #35495a; color: #d2e3ed; }
.mantine-Button-root:disabled { opacity: .42; }
.mantine-Input-input { background: #0c121b; border-color: #304052; border-radius: 6px; color: #e3edf6; }
.mantine-Input-input:focus { border-color: #83dec7; }
.mantine-Flex-root:has(> div > .mantine-Textarea-root) { flex-direction: column; align-items: stretch !important; gap: 6px; }
.mantine-Flex-root:has(> div > .mantine-Textarea-root) > div { width: 100% !important; }
.mantine-Textarea-input { min-height: 68px; line-height: 1.5; padding: 8px; }
.mantine-Text-root label { color: #a0b1c1; letter-spacing: 0; font-size: 11px; }
.mantine-Checkbox-label { color: #b6c7d5; }
.sz-sub { color: #8fa3b8; font-size: 12px; line-height: 1.6; }
.sz-section { margin: 5px 12px 7px; color: #f0f6fa; font-size: 13px; font-weight: 600; }
.sz-section small { display: block; color: #8fa3b8; font-size: 11px; font-weight: 400; line-height: 1.45; margin-top: 2px; }
.sz-status { margin: 5px 12px 7px; color: #9bc6bc; font-size: 11px; line-height: 1.35; }
.sz-status span { color: #d7b5a6; }
.sz-count { margin: -2px 12px 5px; color: #91a9b8; font-size: 11px; }
.sz-count.invalid { color: #ffab9c; }
.sz-note { margin: 4px 12px 7px; padding-left: 8px; border-left: 2px solid #39545b; color: #93abba; font-size: 11px; line-height: 1.45; }
.sz-preview { margin: 5px 12px 7px; padding: 7px 8px; border: 1px solid #36554f; border-radius: 7px; background: #152a29; color: #cfe9e5; font-size: 11px; line-height: 1.45; }
.sz-preview.invalid { border-color: #805145; color: #ffb9a9; }
.sz-intro { margin: 7px 12px 11px; padding: 10px; border: 1px solid #36554f; border-radius: 8px; background: #152a29; color: #d5eee8; font-size: 12px; line-height: 1.5; }
.sz-intro b { display: block; color: #f0f6fa; font-size: 14px; margin-bottom: 3px; }
.sz-progress { margin: 7px 12px; padding: 9px 10px; border: 1px solid #42665f; border-radius: 8px; background: #19302e; color: #d5eee8; font-size: 12px; line-height: 1.5; }
.sz-progress.error { border-color: #805145; background: #302320; color: #ffcfbf; }
.sz-project-status { margin: 2px 12px 8px; color: #9bc6bc; font-size: 11px; line-height: 1.4; }
.sz-segments { display: flex; margin: 8px 12px; height: 7px; gap: 3px; }
.sz-segments i { background: #518c88; border-radius: 3px; min-width: 3px; }
.sz-segments i.active { background: #a6f2db; }
</style>"""


def section(gui, title, description=''):
    gui.add_html(f'<div class="sz-section">{escape(title)}<small>{escape(description)}</small></div>')


class StudioUI:
    def __init__(self, server, session, camera, project_folder, scene_controls, character_controls=None, *, core_session=None, core_controls=None, on_story_activate=None):
        install_upload_snapshots(server)
        self.server, self.session, self.camera = server, session, camera
        self.core_session = core_session
        self._core_visibility = []
        self._legacy_motion_controls = []
        self.folder = project_folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.take_map, self.saved_map, self.segment_map = {}, {}, {}
        self.last_take = None
        self._published_take_selection = None
        self._last_action_choice = None
        self.action_edit = None
        self._action_source = None
        self._last_timeline_nonce = None
        self._submitted_action_edit = None
        self._action_request_context = None
        self._generation_request_context = None
        self._generation_started_at = None
        self.take_slots = []
        self.slot_ids = ()
        gui = server.gui
        gui.add_html(STYLE)
        self.status = gui.add_html('')
        from story_controls import StoryControls
        self.story_controls = StoryControls(gui, session, core_session=core_session,
                                            on_story_activate=on_story_activate)
        self.playhead = gui.add_html('')
        self.transport = gui.add_button_group('Playback', ('Start', 'Play', 'Pause'))
        self.quick_actions = gui.add_button_group('Quick actions', ('New take', 'Guide'))
        self.project_name = gui.add_text('Project name', initial_value='My performance')
        self.save = gui.add_button('Save project + download', icon=viser.Icon.DOWNLOAD, color='gray')
        self.files = gui.add_html('')
        self.tabs = gui.add_tab_group()
        with self.tabs.add_tab('Motion'):
            if core_controls is not None:
                core_controls(gui)
            legacy_before = set(vars(self))
            # Timeline clicks arrive through a normal Viser text update. The
            # browser keeps this control hidden; JSON identifies the take and
            # action so an old click cannot edit a newly selected take.
            self.timeline_command = gui.add_text('Timeline action command', initial_value='')
            self.timeline_command.visible = False
            self.motion_intro = gui.add_html('')
            self.ideas_folder = gui.add_folder('Try a direction', expand_by_default=True)
            with self.ideas_folder:
                self.ideas = gui.add_button_group('Example directions', ('Wave', 'Walk', 'Dance'))
            self.action_heading = gui.add_html('')
            self.action_note = gui.add_html('')
            self.action_prompt = gui.add_text('Direction', initial_value='', multiline=True)
            self.action_assistant = self._make_prompt_assistant(gui, action=True)
            self.action_duration = gui.add_text('Length (s)', initial_value='4.16')
            self.action_feedback = gui.add_html('')
            self.save_action = gui.add_button('Update motion', icon=viser.Icon.SPARKLES)
            self.action_progress = gui.add_html('')
            self.cancel_action = gui.add_button('Cancel', color='gray')
            self.undo_action = gui.add_button('Undo edit', color='gray')
            self.editor_context = gui.add_html('')
            self.add_to_end = gui.add_button('Add action', color='green')
            self.edit_action = gui.add_dropdown('Action', (CREATE, EXTEND, REPLACE), initial_value=EXTEND if session.active_take in session.takes else CREATE)
            self.edit_explainer = gui.add_html('')
            self.replace_time = gui.add_text('Keep motion before this time (seconds)', initial_value='0.00')
            self.prompt = gui.add_text('Direction', initial_value='', multiline=True)
            self.prompt_assistant = self._make_prompt_assistant(gui, action=False)
            self.prompt_count = gui.add_html('')
            self.duration_mode = gui.add_dropdown('Length', (AUTO, SET_DURATION, TARGET_TOTAL), initial_value=AUTO)
            self.duration_seconds = gui.add_text('New motion length (seconds, 0.16–30)', initial_value='4.16')
            self.duration_preview = gui.add_html('')
            self.generate = gui.add_button('Generate', icon=viser.Icon.SPARKLES)
            self.cancel = gui.add_button('Cancel generation', color='gray', visible=False)
            self.motion_progress = gui.add_html('')
            self.advanced_folder = gui.add_folder('Advanced: change ending as a new version', expand_by_default=False)
            with self.advanced_folder:
                self.advanced_replace = gui.add_button('Change ending', color='gray')
            self.cancel_alternate = gui.add_button('Back to take', color='gray')
            self._legacy_motion_controls = [value for name, value in vars(self).items()
                if name not in legacy_before and hasattr(value, 'visible') and name != 'timeline_command']
        with self.tabs.add_tab('Takes'):
            with gui.add_folder('Precise playback', expand_by_default=False):
                self.frames = gui.add_button_group('Frame', ('−1 frame', '+1 frame', 'End'))
                self.seek_time = gui.add_text('Seek to second', initial_value='0.00')
                self.seek_go = gui.add_button('Go to time', color='gray')
            section(gui, 'Selected take', 'Click an action on the timeline to edit it, or add to the end.')
            self.take_info = gui.add_html('')
            self.prepare_extend = gui.add_button('Add action to end', color='gray')
            self.prepare_replace = gui.add_button('Change ending · new version', color='gray')
            self.remove_take = gui.add_button('Remove selected take', color='gray')
            self.undo_remove = gui.add_button('Undo remove', color='gray')
            section(gui, 'Your takes', 'Select a version to watch or change. Each generated ending is saved as a separate take.')
            self.takes = gui.add_dropdown('Selected take', ('No takes yet',))
            self.takes.visible = False  # Keep its state for older clients; buttons are the visible selector.
            self.empty_takes = gui.add_html('<div class="sz-note">No saved takes yet. Use New take to make one.</div>')
            for _ in range(MAX_TAKES):
                button = gui.add_button('Edit take', color='gray', visible=False)
                self.take_slots.append(button)
            self.take_name = gui.add_text('Take name', initial_value='')
            self.rename = gui.add_button('Rename take', color='gray')
            self.duplicate = gui.add_button('Make a copy of this take', color='gray')
            self.trim = gui.add_button('Save a shorter copy ending here', color='gray')
            section(gui, 'Motion in this take', 'Jump to an action or reuse its direction for another version.')
            self.segments = gui.add_dropdown('Generated motion', ('No generated actions',))
            self.jump = gui.add_button_group('Find motion', ('Previous', 'Go to action', 'Next'))
            self.reuse = gui.add_button('Copy this direction to Motion', color='gray')
            self.action = gui.add_html('')
            self.speed = gui.add_dropdown('Speed', ('0.25×', '0.5×', '1×', '1.5×', '2×'), initial_value='1×')
            self.loop = gui.add_checkbox('Loop playback', initial_value=False)
        with self.tabs.add_tab('Scene'):
            scene_controls(gui)
        if character_controls is not None:
            with self.tabs.add_tab('Character'):
                character_controls(gui)
        with self.tabs.add_tab('View'):
            section(gui, 'Camera')
            camera.build_gui(gui)
        with self.tabs.add_tab('Project'):
            section(gui, 'Open or start a project', 'Use Save project + download above to keep the current project.')
            self.saved = gui.add_dropdown('Saved projects', ('No saved projects',))
            self.open = gui.add_button('Open selected project', color='gray')
            self.upload = gui.add_upload_button('Open project file', mime_type='.npz')
            self.clear = gui.add_button('New project · back up current', color='gray')
            gui.add_html('<div class="sz-note">Opening a project automatically backs up the current takes. All connected viewers share the same project.</div>')
            with gui.add_folder('Generation diagnostics', expand_by_default=False):
                self.performance = gui.add_markdown('No generation in this session.')
            with gui.add_folder('Advanced: playback source', expand_by_default=False):
                self.mode = gui.add_dropdown('Source', ('Recorded preview', 'Live ARDY'), initial_value=session.mode)
                gui.add_html('<div class="sz-note">Recorded preview is the included example. Generating or selecting a take switches to your live project automatically.</div>')
            gui.add_markdown('[ARDY G1 model](https://huggingface.co/nvidia/ARDY-G1-RP-25FPS-Horizon52) · [Recorded motion source](https://huggingface.co/datasets/bones-studio/seed)')
        self.guide_tab_index = 6 if character_controls is not None else 5
        with self.tabs.add_tab('Guide'):
            gui.add_html(GUIDE_HTML)
            self.guide_new_take = gui.add_button('Start a new take', color='green')
            self.guide_browse_takes = gui.add_button('Browse takes', color='gray')
            self.guide_scene = gui.add_button('Add scene objects', color='gray')
        self.refresh_saved()
        self.bind()
        self.update()

    def _set(self, handle, property_name, value):
        """Viser setters broadcast to clients, so publish changed properties only."""
        if (property_name == 'visible' and self.core_session is not None
                and self.core_session.snapshot()['active']
                and any(handle is item for item in self._legacy_motion_controls)):
            self._core_visibility = [(h, v) for h, v in self._core_visibility if h is not handle]
            self._core_visibility.append((handle, value))
            value = False
        if getattr(handle, property_name) != value:
            setattr(handle, property_name, value)

    @staticmethod
    def _valid_prompt(value):
        return 1 <= len(value.strip()) <= 500

    def _prompt_feedback(self):
        count = len(self.prompt.value.strip())
        invalid = not 1 <= count <= 500
        message = 'Enter a direction' if count == 0 else '500 character limit' if count > 500 else 'Direction ready'
        self._set(self.prompt_count, 'content',
                  f'<div class="sz-count{" invalid" if invalid else ""}">{count}/500 · {message}</div>')
        self._set(self.prompt_count, 'visible', invalid and self.prompt.visible)

    def _clear_action_edit(self):
        self.action_edit = None
        self._action_source = None
        self._submitted_action_edit = None
        self._action_request_context = None

    def _begin_action_edit(self, take, index, operation):
        segment = take.segments[index]
        self.action_edit = (take.id, index, operation)
        self._action_source = take
        self._submitted_action_edit = None
        self._action_request_context = None
        self._set(self.action_prompt, 'value', segment['prompt'] if operation == 'replace' else '')
        self._set(self.action_duration, 'value',
                  f'{(segment["end"] - segment["start"])/25:.2f}' if operation == 'replace' else '4.16')

    def _action_duration_value(self):
        try:
            seconds = float(self.action_duration.value)
            if not math.isfinite(seconds) or not 0.16 <= seconds <= 30:
                raise ValueError
            return seconds, None
        except ValueError:
            return None, 'Enter an action length from 0.16 to 30 seconds.'

    def _assistant_context(self, action):
        s = self.session
        with s.lock:
            return (s.active_take, s.project_revision, s.clip_revision, s.version,
                    s.character_motion_enabled, s.busy,
                    self._action_context() if action else self._generation_context())

    def _make_prompt_assistant(self, gui, *, action):
        target = self.action_prompt if action else self.prompt

        def apply(text, expected_context, expected_prompt):
            s = self.session
            with s.lock:
                if (s.busy or not s.character_motion_enabled or not target.visible
                        or self._assistant_context(action) != expected_context
                        or target.value != expected_prompt or not self._valid_prompt(text)):
                    return False
                self._set(target, 'value', text)
                if not action:
                    s.edit_prompt(text)
            self.update()
            return True

        def generate_validated(text, expected_context, expected_scene, _original,
                               submission_guard):
            s = self.session
            with s.lock:
                if (s.busy or not s.character_motion_enabled or not target.visible or
                        target.value != text or not self._valid_prompt(text) or
                        self._assistant_context(False) != expected_context or
                        self._assistant_scene_context() != expected_scene):
                    return False, 'The direction or scene changed. Generation was not started.'
                take = s.takes.get(s.active_take)
                plan, error = self._generation_plan(take)
                if error is not None:
                    s.status = error
                    return False, error
                choice, at_frame, _, duration = plan
                with submission_guard() as allowed:
                    if not allowed:
                        return False, 'Pending generation cancelled.'
                    if s.mode != 'Live ARDY':
                        s.set_mode('Live ARDY')
                        self._set(self.mode, 'value', 'Live ARDY')
                    s.submit(text,
                             seconds=None if self.duration_mode.value == AUTO else duration.seconds,
                             edit_mode={CREATE:'new', EXTEND:'extend', REPLACE:'replace'}[choice],
                             at_frame=at_frame)
                started = s.busy
                if started:
                    self._generation_request_context = self._generation_context()
                    self._generation_started_at = time.perf_counter()
                else:
                    error = s.status
            self.update()
            return (True, '') if started else (False, error)

        return PromptAssistantUI(gui, target, lambda: self._assistant_context(action), apply,
                                 scene_context=self._assistant_scene_context, auto_apply=True,
                                 generate=None if action else generate_validated)

    def _assistant_scene_context(self):
        """Expose scene facts without treating an editor selection as a motion goal."""
        with self.session.lock:
            scene = self.session.scene
            objects = scene.get('objects', [])
            targets = resolve_targets(scene.get('targets', []), objects)
            return {
                'objects': [{key: obj[key] for key in ('id', 'name', 'position') if key in obj}
                            for obj in objects],
                'targets': [{key: target[key] for key in ('id', 'name', 'object_id', 'position')}
                            for target in targets],
            }

    def _action_context(self):
        return (self.action_edit, id(self._action_source), self.session.project_revision,
                self.action_prompt.value.strip(), self.action_duration.value)

    def _generation_context(self):
        return (self.session.active_take, self.session.project_revision,
                self.edit_action.value, self.prompt.value.strip(),
                self.duration_mode.value, self.duration_seconds.value, self.replace_time.value)

    def _progress_content(self, status, failed=False):
        elapsed = ''
        if not failed and self._generation_started_at is not None:
            elapsed = f' · {int(max(0, time.perf_counter() - self._generation_started_at))} s elapsed'
        detail = (' Review the direction and length above, then retry.' if failed else '')
        style = 'sz-progress error' if failed else 'sz-progress'
        return f'<div class="{style}" role="status" aria-live="polite">{escape(status)}{elapsed}{detail}</div>'

    def _refresh_action_edit(self, take, busy):
        edit = self.action_edit
        editing = edit is not None
        if editing and (take is None or take is not self._action_source or
                        take.id != edit[0] or edit[1] >= len(take.segments)):
            self._clear_action_edit()
            editing = False
        submitted = self._submitted_action_edit
        if editing and submitted is not None and not busy and self.session.action_edit_revision > submitted:
            self._clear_action_edit()
            editing = False
        elif not busy and submitted is not None:
            # Failed or cancelled generation leaves the form ready to retry.
            self._submitted_action_edit = None

        for control in (self.action_heading, self.action_note, self.action_prompt,
                        self.action_duration, self.action_feedback, self.save_action,
                        self.cancel_action):
            self._set(control, 'visible', editing)
        alternate = take is not None and self.edit_action.value == REPLACE
        show_form = not editing and (take is None or not take.segments or alternate)
        self._set(self.ideas_folder, 'visible', show_form)
        self._set(self.advanced_folder, 'visible', take is not None and bool(take.segments) and not editing and not alternate)
        self._set(self.cancel_alternate, 'visible', alternate and not editing)
        self._set(self.cancel_alternate, 'disabled', busy)
        self._set(self.undo_action, 'visible', take is not None and self.session.can_undo_action_edit)
        self._set(self.undo_action, 'disabled', busy)
        self._set(self.editor_context, 'visible', not editing)
        # The action dropdown remains as a compatibility handle for older
        # clients; the visible routes are New take, Add action and Change ending.
        self._set(self.edit_action, 'visible', False)
        for control in (self.prompt, self.prompt_count, self.duration_mode,
                        self.generate, self.ideas):
            self._set(control, 'visible', show_form)
        self._set(self.add_to_end, 'visible', take is not None and not editing and not alternate)
        self._set(self.add_to_end, 'disabled', take is None or busy)
        if not editing:
            return
        _, index, operation = edit
        action_number = index + 1
        title = (f'Edit action {action_number}' if operation == 'replace'
                 else f'Add before action {action_number}' if operation == 'insert_before'
                 else 'Add action to end' if index == len(take.segments) - 1
                 else f'Add after action {action_number}')
        self._set(self.action_heading, 'content', f'<div class="sz-section">{escape(title)}</div>')
        following = len(take.segments) - index - (operation != 'insert_before')
        affected = (f'{following} following action' + ('s' if following != 1 else '') + ' will regenerate.'
                    if following else 'No following actions need regeneration.')
        change = 'Updating this action' if operation == 'replace' else 'Adding this action'
        self._set(self.action_note, 'content',
                  f'<div class="sz-note">{change} regenerates motion from here. '
                  f'{affected} Original motion stays until completion; Undo edit restores it afterward.</div>')
        seconds, error = self._action_duration_value()
        if not self._valid_prompt(self.action_prompt.value):
            error = 'Enter a direction (1–500 characters).'
        self._set(self.action_feedback, 'visible', error is not None)
        self._set(self.action_feedback, 'content',
                  f'<div class="sz-preview invalid">{escape(error)}</div>' if error else '')
        retry = (self.session.status.startswith('Action edit failed') and
                 self._action_request_context == self._action_context())
        self._set(self.save_action, 'label',
                  'Updating motion…' if busy and operation == 'replace' else
                  'Adding action…' if busy else
                  'Retry update motion' if retry and operation == 'replace' else
                  'Retry add action' if retry else
                  'Update motion' if operation == 'replace' else 'Add action')
        self._set(self.save_action, 'disabled', busy or not self.session.character_motion_enabled or error is not None)
        self._set(self.action_prompt, 'disabled', busy)
        self._set(self.action_duration, 'disabled', busy)
        self._set(self.cancel_action, 'disabled', busy)

    def _generation_plan(self, take):
        """Resolve the explicit UI choice without moving the playback position."""
        choice = self.edit_action.value
        if choice not in (CREATE, EXTEND, REPLACE):
            return None, 'Choose what to make.'
        if choice == CREATE and take is not None:
            return None, 'Use New take to start a blank draft.'
        if choice != CREATE and take is None:
            return None, 'Select a take in Takes first.'
        at_frame = None
        prefix_frames = 0
        if choice == EXTEND:
            prefix_frames = len(take.positions)
        elif choice == REPLACE:
            try:
                at_seconds = float(self.replace_time.value)
                if not math.isfinite(at_seconds):
                    raise ValueError
                at_frame = round(at_seconds * 25)
                if at_seconds < 0 or not 0 <= at_frame < len(take.positions):
                    raise ValueError
            except ValueError:
                return None, f'Enter a time from 0 to {(len(take.positions)-1)/25:.2f} seconds.'
            if 0 < at_frame < 4:
                return None, 'Use 0 to replace the whole take, or 0.16 seconds or later to keep its beginning.'
            prefix_frames = at_frame
        seconds = None
        if self.duration_mode.value != AUTO:
            try:
                input_seconds = float(self.duration_seconds.value)
                if not math.isfinite(input_seconds):
                    raise ValueError
                if self.duration_mode.value == TARGET_TOTAL:
                    new_frames = round(input_seconds * 25) - prefix_frames
                    if not 4 <= new_frames <= 750:
                        raise ValueError
                    seconds = new_frames / 25
                else:
                    seconds = input_seconds
                if not 0.16 <= seconds <= 30:
                    raise ValueError
            except ValueError:
                if self.duration_mode.value == TARGET_TOTAL:
                    return None, f'Total length must add 0.16–30 seconds beyond the kept {prefix_frames/25:.2f} seconds.'
                return None, 'Enter a new motion length from 0.16 to 30 seconds.'
        try:
            duration = plan_duration(self.prompt.value, seconds=seconds)
        except ValueError as exc:
            return None, str(exc)
        return (choice, at_frame, prefix_frames, duration), None

    def _refresh_generation(self, take, busy):
        choice = self.edit_action.value
        show_form = self.action_edit is None and (take is None or not take.segments or choice == REPLACE)
        context = ('A changed ending makes a separate take; the original stays available.'
                   if choice == REPLACE else '')
        self._set(self.edit_explainer, 'content',
                  f'<div class="sz-note">{escape(context)}</div>' if context else '')
        self._set(self.edit_explainer, 'visible', bool(context) and show_form)
        self._set(self.replace_time, 'visible', choice == REPLACE and show_form)
        self._set(self.replace_time, 'disabled', busy or choice != REPLACE)
        self._set(self.duration_seconds, 'visible', self.duration_mode.value != AUTO and show_form)
        self._set(self.duration_seconds, 'disabled', busy or self.duration_mode.value == AUTO)
        length_label = ('Target total scene length (seconds)' if self.duration_mode.value == TARGET_TOTAL
                        else 'New motion length (seconds, 0.16–30)')
        self._set(self.duration_seconds, 'label', length_label)
        self._set(self.edit_action, 'disabled', busy)
        self._set(self.duration_mode, 'disabled', busy)

        plan, error = self._generation_plan(take)
        if plan is None:
            preview = f'<div class="sz-preview invalid">{escape(error)}</div>'
        else:
            _, _, prefix_frames, duration = plan
            new_total = (prefix_frames + duration.frames) / 25
            if choice == CREATE:
                outcome = f'New take: {new_total:.2f} seconds.'
            elif choice == EXTEND:
                outcome = f'Current take: {prefix_frames/25:.2f} seconds → {new_total:.2f} seconds total.'
            else:
                outcome = (f'Keep first {prefix_frames/25:.2f} seconds; new ending makes a '
                           f'{new_total:.2f} second take. Original stays available.')
            estimate = (f'{duration.label}: {duration.seconds:.2f} seconds of new motion'
                        if self.duration_mode.value == AUTO else f'Generate {duration.seconds:.2f} seconds of new motion')
            preview = f'<div class="sz-preview">{escape(estimate)}. {escape(outcome)}</div>'
        self._set(self.duration_preview, 'content', preview)
        self._set(self.duration_preview, 'visible', self._valid_prompt(self.prompt.value) and show_form)
        at_limit = len(self.session.takes) >= MAX_TAKES and choice != EXTEND
        assistant_pending = self.prompt_assistant.generation_in_progress()
        disabled = (busy or assistant_pending or not self.session.character_motion_enabled or
                    not self._valid_prompt(self.prompt.value) or error is not None or at_limit)
        self._set(self.generate, 'disabled', disabled)
        retry = (self.session.status.startswith('Generation failed') and
                 self._generation_request_context == self._generation_context())
        self._set(self.generate, 'label', 'Generating…' if busy else
                  'Improving direction…' if assistant_pending and self.prompt_assistant._request_pending else
                  'Answer prompt questions' if assistant_pending else
                  'Take limit reached' if at_limit else
                  'Retry generation' if retry else 'Generate motion')

    def refresh_saved(self):
        self.saved_map = {p.name: p for p in sorted(self.folder.glob('*.stagezero.npz'), key=lambda p: p.stat().st_mtime, reverse=True)}
        self._set(self.saved, 'options', tuple(self.saved_map) or ('No saved projects',))
        if self.saved.value not in self.saved.options:
            self._set(self.saved, 'value', self.saved.options[0])
        self._set(self.open, 'disabled', not self.saved_map)

    def bind(self):
        s = self.session

        def begin_new_take(client):
            with s.lock:
                if s.busy: return
                if s.mode != 'Live ARDY': s.set_mode('Live ARDY')
                if s.new_take() is False: return
                self._clear_action_edit()
                self._set(self.mode, 'value', 'Live ARDY')
                self._set(self.edit_action, 'value', CREATE)
                self._set(self.prompt, 'value', '')
                s.edit_prompt('')
                self._set(self.duration_mode, 'value', AUTO)
                self._set(self.replace_time, 'value', '0.00')
            self.update()
            navigate_tab(self.tabs, 0, client)

        @self.quick_actions.on_click
        def quick_action(e):
            if self.quick_actions.value == 'New take':
                begin_new_take(e.client)
            elif self.quick_actions.value == 'Guide':
                navigate_tab(self.tabs, self.guide_tab_index, e.client)

        @self.guide_new_take.on_click
        def guide_new_take(e):
            begin_new_take(e.client)

        @self.guide_browse_takes.on_click
        def guide_browse_takes(e):
            navigate_tab(self.tabs, 1, e.client)

        @self.guide_scene.on_click
        def guide_scene(e):
            navigate_tab(self.tabs, 2, e.client)

        @self.transport.on_click
        def transport(_):
            transport_command(self.transport.value)

        @self.frames.on_click
        def frame_step(_):
            transport_command(self.frames.value)

        def transport_command(value):
            if self.core_session is not None and self.core_session.snapshot()['active']:
                core = self.core_session
                state = core.snapshot()
                if value == 'Play': core.play()
                elif value == 'Pause': core.pause()
                elif value == 'Start': core.seek(0)
                elif value == 'End': core.seek(max(0, state['total_frames']-1))
                elif value == '−1 frame': core.seek(max(0, state['frame']-1))
                elif value == '+1 frame': core.seek(state['frame']+1)
                return
            with s.lock:
                if not s.character_motion_enabled and value != 'Pause':
                    return
                if s.busy and value not in ('Play', 'Pause'):
                    return
                if s.kind == 'reference' and value not in ('Play', 'Pause'):
                    return
                if value == 'Play': s.play()
                elif value == 'Pause': s.pause()
                elif value == 'Start': s.seek(0)
                elif value == 'End': s.seek(len(s.positions)-1)
                elif value == '−1 frame': s.seek(s.frame-1)
                elif value == '+1 frame': s.seek(s.frame+1)

        @self.seek_go.on_click
        def seek_time(_):
            with s.lock:
                if not s.character_motion_enabled or s.busy or s.kind == 'reference':
                    return
                try:
                    seconds = float(self.seek_time.value)
                    duration = len(s.positions)/s.fps
                    if not 0 <= seconds <= duration:
                        raise ValueError
                except ValueError:
                    s.status = f'Enter a time from 0 to {len(s.positions)/s.fps:.2f} s'
                    return
                s.seek(min(round(seconds*s.fps), len(s.positions)-1))

        @self.mode.on_update
        def mode(e):
            if e.client is not None:
                with s.lock:
                    if s.busy:
                        self._set(self.mode, 'value', s.mode)
                    elif self.mode.value != s.mode:
                        self._clear_action_edit()
                        s.set_mode(self.mode.value)

        @self.timeline_command.on_update
        def timeline_command(e):
            if e.client is None:
                return
            try:
                command = json.loads(self.timeline_command.value)
                take_id = command['take_id']
                index = command['index']
                operation = command['operation']
                nonce = command['nonce']
            except (TypeError, ValueError, KeyError):
                return
            if (type(take_id) is not str or type(index) is not int or
                    operation not in ('replace', 'insert_before', 'insert_after',
                                      'start', 'play', 'pause') or
                    type(nonce) not in (str, int) or nonce == self._last_timeline_nonce):
                return
            with s.lock:
                take = s.takes.get(s.active_take)
                if take is None or take.id != take_id or not 0 <= index < len(take.segments):
                    return
                if not s.character_motion_enabled and operation != 'pause':
                    return
                if operation in ('start', 'play', 'pause'):
                    if s.busy and operation != 'pause':
                        return
                    self._last_timeline_nonce = nonce
                    if operation == 'play': s.play()
                    elif operation == 'pause': s.pause()
                    else: s.seek(0)
                    self.update()
                    return
                if s.busy:
                    return
                self._last_timeline_nonce = nonce
                self._begin_action_edit(take, index, operation)
                selected = take.segments[index]
                s.seek(selected['end'] - 1 if operation == 'insert_after' else selected['start'])
                s.status = f'Editing action {index + 1} · write a direction and update motion'
            self.update()
            navigate_tab(self.tabs, 0, e.client)

        @self.action_prompt.on_update
        def action_prompt(e):
            if e.client is not None:
                self.update()

        @self.action_duration.on_update
        def action_duration(e):
            if e.client is not None:
                self.update()

        @self.save_action.on_click
        def save_action(_):
            with s.lock:
                edit = self.action_edit
                take = s.takes.get(s.active_take)
                if (not s.character_motion_enabled or s.busy or edit is None or take is not self._action_source or
                        take.id != edit[0] or
                        edit[1] >= len(take.segments) or not self._valid_prompt(self.action_prompt.value)):
                    return
                seconds, error = self._action_duration_value()
                if error is not None:
                    s.status = error
                    return
                assistant_pending = self.action_assistant.blocks_generation()
                if (assistant_pending or
                        needs_clarification(self.action_prompt.value,
                                            scene_context=self._assistant_scene_context())):
                    if not assistant_pending:
                        self.action_assistant.clarify()
                    s.status = 'Clarify the direction in Prompt assistant before updating motion'
                    self.update()
                    return
                revision = s.action_edit_revision
                if s.submit_action_edit(self.action_prompt.value, edit[1], edit[2], seconds=seconds):
                    self._submitted_action_edit = revision
                    self._action_request_context = self._action_context()
                    self._generation_started_at = time.perf_counter()
            self.update()

        @self.cancel_action.on_click
        def cancel_action(_):
            with s.lock:
                if s.busy:
                    return
                self._clear_action_edit()
            self.update()

        @self.undo_action.on_click
        def undo_action(_):
            with s.lock:
                if not s.busy and s.undo_action_edit():
                    self._clear_action_edit()
            self.update()

        @self.add_to_end.on_click
        def add_to_end(e):
            with s.lock:
                take = s.takes.get(s.active_take)
                if s.busy or take is None:
                    return
                if take.segments:
                    self._begin_action_edit(take, len(take.segments) - 1, 'insert_after')
                else:
                    self._clear_action_edit()
                    self._set(self.edit_action, 'value', EXTEND)
                s.status = 'Ready to add an action at the end'
            self.update()
            navigate_tab(self.tabs, 0, e.client)

        @self.prompt.on_update
        def prompt(e):
            if e.client is not None:
                with s.lock:
                    if s.busy: return
                    s.edit_prompt(self.prompt.value)
                self._prompt_feedback()
                self.update()

        @self.edit_action.on_update
        def edit_action(e):
            if e.client is not None:
                self._clear_action_edit()
                if self.edit_action.value == CREATE and s.active_take is not None:
                    begin_new_take(e.client)
                    return
                self.update()

        @self.replace_time.on_update
        def replace_time(e):
            if e.client is not None:
                self.update()

        @self.duration_mode.on_update
        def duration_mode(e):
            if e.client is not None:
                self.update()

        @self.duration_seconds.on_update
        def duration_seconds(e):
            if e.client is not None:
                self.update()

        @self.ideas.on_click
        def idea(_):
            with s.lock:
                if s.busy: return
                self._set(self.prompt, 'value', {'Wave':'A person waves with their right hand.', 'Walk':'A person walks forward at a relaxed pace.', 'Dance':'A person dances with small rhythmic steps and relaxed arm movements.'}[self.ideas.value])
                s.edit_prompt(self.prompt.value)
                self._prompt_feedback()
            self.update()

        @self.generate.on_click
        def generate(_):
            with s.lock:
                if not s.character_motion_enabled or s.busy or not self._valid_prompt(self.prompt.value):
                    return
                take = s.takes.get(s.active_take)
                _, error = self._generation_plan(take)
                if error is not None:
                    s.status = error
                    return
            self.prompt_assistant.start_generation()
            self.update()

        @self.cancel.on_click
        def cancel(_):
            with s.lock:
                if not s.busy: return
                s.seek(s.frame)
                s.status = 'Generation cancelled · stored motion preserved'
            self.update()

        @self.takes.on_update
        def take(e):
            if e.client is not None:
                # Capture the requested ID before waiting for the viewer's
                # session lock. An update may rebuild the option map meanwhile.
                take_id = self.take_map.get(self.takes.value)
                if take_id is None: return
                with s.lock:
                    if s.busy: return
                    if s.mode != 'Live ARDY': s.set_mode('Live ARDY')
                    self._set(self.mode, 'value', 'Live ARDY')
                    self._clear_action_edit()
                    s.select_take(take_id)

        for slot_index, button in enumerate(self.take_slots):
            @button.on_click
            def open_take(e, index=slot_index):
                # Resolve the clicked row before acquiring the session lock. A
                # concurrent viewer update may republish the rows while we wait.
                take_id = self.slot_ids[index] if index < len(self.slot_ids) else None
                with s.lock:
                    if s.busy or take_id not in s.takes: return
                    if s.mode != 'Live ARDY': s.set_mode('Live ARDY')
                    self._set(self.mode, 'value', 'Live ARDY')
                    self._clear_action_edit()
                    s.select_take(take_id)
                    self._set(self.edit_action, 'value', EXTEND)
                self.update()
                navigate_tab(self.tabs, 0, e.client)

        @self.prepare_extend.on_click
        def prepare_extend(e):
            with s.lock:
                take = s.takes.get(s.active_take)
                if s.busy or take is None: return
                if take.segments:
                    self._begin_action_edit(take, len(take.segments) - 1, 'insert_after')
                else:
                    self._clear_action_edit()
                    self._set(self.edit_action, 'value', EXTEND)
                s.status = 'Ready to add an action at the end'
            self.update()
            navigate_tab(self.tabs, 0, e.client)

        @self.prepare_replace.on_click
        def prepare_replace(e):
            with s.lock:
                if s.busy or s.active_take not in s.takes: return
                self._clear_action_edit()
                self._set(self.edit_action, 'value', REPLACE)
                self._set(self.replace_time, 'value', f'{s.frame/s.fps:.2f}')
                s.status = 'Ready to change this ending · choose the time to keep'
            self.update()
            navigate_tab(self.tabs, 0, e.client)

        @self.advanced_replace.on_click
        def advanced_replace(e):
            prepare_replace(e)

        @self.cancel_alternate.on_click
        def cancel_alternate(_):
            with s.lock:
                if s.busy: return
                self._set(self.edit_action, 'value', EXTEND)
            self.update()

        @self.rename.on_click
        def rename(_):
            with s.lock:
                if not s.busy: s.rename_active_take(self.take_name.value)
        @self.duplicate.on_click
        def duplicate(_):
            with s.lock:
                if not s.busy: s.duplicate_active_take()
        @self.remove_take.on_click
        def remove_take(_):
            with s.lock:
                if not s.busy: s.remove_active_take()
            self.update()
        @self.undo_remove.on_click
        def undo_remove(_):
            with s.lock:
                if not s.busy: s.undo_remove_take()
            self.update()
        @self.trim.on_click
        def trim(_):
            with s.lock:
                if not s.busy: s.trim_after_playhead()
        @self.speed.on_update
        def speed(e):
            if e.client is not None: s.set_playback_speed(float(self.speed.value[:-1]))
        @self.loop.on_update
        def loop(e):
            if e.client is not None: s.set_loop(self.loop.value)
        @self.jump.on_click
        def jump(_):
            with s.lock:
                if s.busy or s.mode != 'Live ARDY': return
                if self.jump.value == 'Previous': s.previous_action()
                elif self.jump.value == 'Next': s.next_action()
                elif self.segments.value in self.segment_map: s.seek(self.segment_map[self.segments.value]['start'])
        @self.reuse.on_click
        def reuse(_):
            with s.lock:
                if s.busy: return
                seg = self.segment_map.get(self.segments.value)
                if seg:
                    self._set(self.prompt, 'value', seg['prompt'])
                    s.edit_prompt(self.prompt.value)
                    self._prompt_feedback()
                    s.status = 'Direction copied · choose how to use it in Motion'
            self.update()
            navigate_tab(self.tabs, 0, _.client)

        @self.save.on_click
        def save(e):
            try:
                with s.lock:
                    if s.busy: return
                path, data = s.save_project(self.folder, self.project_name.value)
            except Exception as exc:
                s.project_status = f'Save failed: {exc}'
                self.update()
                return
            self.refresh_saved()
            self._set(self.saved, 'value', path.name)
            if e.client is not None:
                try:
                    e.client.send_file_download(path.name, data)
                except Exception as exc:
                    s.project_status += f' · Download failed: {exc}'
            self.update()

        @self.open.on_click
        def open_saved(_):
            path = self.saved_map.get(self.saved.value)
            if path:
                try: self.open_data(path.read_bytes())
                except OSError as exc: s.project_status = f'Open failed: {exc}'
        @self.upload.on_upload
        def upload(event): self.open_data(event.file.content)
        @self.clear.on_click
        def clear(_):
            try:
                with s.lock:
                    if s.busy: return
                    s.new_project(self.folder)
                    self._clear_action_edit()
                    self._set(self.mode, 'value', s.mode)
                    self._set(self.edit_action, 'value', CREATE)
                    self._set(self.prompt, 'value', '')
                self.refresh_saved()
                self.update()
            except Exception as exc: s.project_status = f'Backup failed; project retained: {exc}'

    def open_data(self, data):
        try:
            with self.session.lock:
                if self.session.busy: return
                self.session.save_project(self.folder, 'before-open-backup')
                self.session.load_project(data)
                self._clear_action_edit()
                self._set(self.mode, 'value', self.session.mode)
                self._set(self.edit_action, 'value', EXTEND if self.session.active_take else CREATE)
                self._set(self.prompt, 'value', '')
                self.session.edit_prompt('')
            self.refresh_saved()
            self.update()
        except Exception as exc:
            self.session.project_status = f'Open failed; current takes preserved: {exc}'

    def update(self):
        """Synchronize the sidebar after the viewer advances the session clock."""
        self.story_controls.update()
        s = self.session
        if self.core_session is None or not self.core_session.snapshot()['active']:
            for handle, visible in self._core_visibility:
                self._set(handle, 'visible', visible)
            self._core_visibility = []
        with s.lock:
            live = s.mode == 'Live ARDY'
            # Viser button groups cannot be disabled; static previews hide
            # motion controls while their callbacks remain guarded as well.
            self._set(self.frames, 'visible', s.character_motion_enabled)
            take = s.takes.get(s.active_take)
            has_clip = s.kind in ('recorded', 'generated')
            last_frame = len(s.positions)-1
            if s.busy: state = 'Generating'
            elif s.kind == 'reference': state = 'Reference pose'
            elif s.playing: state = 'Playing'
            elif s.frame >= last_frame: state = 'Finished'
            else: state = 'Paused'
            source = take.name if live and take else 'Ready to create' if live else 'Example preview'
            detail = '' if s.status.startswith(ROUTINE_STATUS_PREFIXES) else s.status
            detail_html = f' <span>· {escape(detail)}</span>' if detail else ''
            self._set(self.status, 'content', f'<div class="sz-status">{state} · {escape(source)}{detail_html}</div>')
            # Playback and clock live in the bottom timeline toolbar.
            self._set(self.playhead, 'visible', False)
            self._set(self.transport, 'visible', False)
            self._set(self.seek_go, 'disabled', not s.character_motion_enabled or not has_clip or s.busy)
            self._set(self.seek_time, 'disabled', not s.character_motion_enabled or not has_clip or s.busy)
            self._set(self.mode, 'disabled', s.busy)
            self._set(self.prompt, 'disabled', s.busy)
            self._set(self.guide_new_take, 'disabled', s.busy or len(s.takes) >= MAX_TAKES)
            self._refresh_action_edit(take, s.busy)
            self._prompt_feedback()
            self._refresh_generation(take, s.busy)
            self.prompt_assistant.refresh(
                visible=self.prompt.visible and s.character_motion_enabled, busy=s.busy)
            self.action_assistant.refresh(
                visible=self.action_prompt.visible and s.character_motion_enabled, busy=s.busy)
            self._set(self.cancel, 'visible', s.busy)
            if s.busy:
                if self._generation_started_at is None:
                    self._generation_started_at = time.perf_counter()
                progress = self._progress_content(s.status)
                action_progress = self.action_edit is not None
            else:
                action_progress = self.action_edit is not None and (
                    s.status.startswith('Action edit failed') and
                    self._action_request_context == self._action_context())
                motion_progress = (s.status.startswith('Generation failed') and
                                   self._generation_request_context == self._generation_context())
                progress = self._progress_content(s.status, failed=True) if action_progress or motion_progress else ''
                self._generation_started_at = None
            self._set(self.action_progress, 'visible', action_progress)
            self._set(self.action_progress, 'content', progress if action_progress else '')
            self._set(self.motion_progress, 'visible', not action_progress and bool(progress))
            self._set(self.motion_progress, 'content', progress if not action_progress else '')
            intro = ('Select a motion-ready character to generate motion.' if not s.character_motion_enabled else
                     'Choose Wave, Walk, or Dance below, or write your own direction. '
                     'Set a length, then press Generate motion.')
            heading = 'Create your first motion' if not s.takes else 'Start another take'
            self._set(self.motion_intro, 'content',
                      f'<div class="sz-intro"><b>{heading}</b>{intro}</div>')
            self._set(self.motion_intro, 'visible', take is None and not s.busy)
            if take is None:
                editor_state = 'New take draft' if live else 'Start a new take'
            elif self.edit_action.value == REPLACE:
                editor_state = f'New ending for {take.name}'
            else:
                action_count = len(take.segments)
                editor_state = (f'{take.name} · {len(take.positions)/25:.2f} s · '
                                f'{action_count} action{"s" if action_count != 1 else ""}. '
                                'Click a timeline action to edit.')
            self._set(self.editor_context, 'content', f'<div class="sz-note">{escape(editor_state)}</div>')
            self.take_map = {f'{i+1:02d} · {t.name} · {len(t.positions)/25:.2f}s':t.id for i,t in enumerate(s.takes.values())}
            options = tuple(self.take_map) or ('No takes yet',)
            self._set(self.takes, 'options', options)
            selection = next((n for n,t in self.take_map.items() if t == s.active_take), options[0])
            selection_key = (s.active_take, options)
            if selection_key != self._published_take_selection:
                self._set(self.takes, 'value', selection)
                self._published_take_selection = selection_key
            self._set(self.takes, 'disabled', not s.takes or s.busy)
            self.slot_ids = tuple(s.takes)
            self._set(self.empty_takes, 'visible', not s.takes)
            for index, button in enumerate(self.take_slots):
                if index < len(self.slot_ids):
                    row = s.takes[self.slot_ids[index]]
                    marker = '● ' if row.id == s.active_take else ''
                    self._set(button, 'label', f'{marker}Edit {row.name} · {len(row.positions)/25:.2f}s')
                    self._set(button, 'visible', True)
                    self._set(button, 'disabled', s.busy)
                else:
                    self._set(button, 'visible', False)
            active_id = (take.id, take.name) if take else None
            if active_id != self.last_take:
                self.last_take = active_id
                self._set(self.take_name, 'value', take.name if take else '')
            for control in (self.rename,self.duplicate,self.take_name):
                self._set(control, 'disabled', take is None or s.busy)
            self._set(self.remove_take, 'disabled', take is None or s.busy)
            self._set(self.undo_remove, 'disabled', s.busy or not s.can_undo_take_removal)
            for control in (self.prepare_extend, self.prepare_replace):
                self._set(control, 'disabled', take is None or s.busy)
            self._set(self.trim, 'disabled', not live or take is None or s.busy or s.frame < 3 or s.frame >= last_frame)
            self._set(self.save, 'disabled', s.busy)
            self._set(self.open, 'disabled', not self.saved_map or s.busy)
            self._set(self.upload, 'disabled', s.busy)
            self._set(self.clear, 'disabled', s.busy)
            self._set(self.reuse, 'disabled', take is None or s.busy or not take.segments)
            self._set(self.segments, 'disabled', take is None or s.busy or not take.segments)
            self._set(self.take_info, 'content', f'<div class="sz-note">{escape(take.name)} · {len(take.positions)/25:.2f} s · {len(take.segments)} actions</div>' if take else '<div class="sz-note">No take selected. Use New take to begin.</div>')
            self.segment_map = {f'{i+1:02d} · {seg["start"]/25:.2f}s · {seg["prompt"][:30]}':seg for i,seg in enumerate(take.segments)} if take else {}
            segments = tuple(self.segment_map) or ('No generated actions',)
            self._set(self.segments, 'options', segments)
            self._set(self.action, 'content', f'<div class="sz-note">At playhead: {escape(s.current_action() or "No generated action")}</div>')
            project_status = s.project_status or 'No project save in this session.'
            self._set(self.files, 'content',
                      f'<div class="sz-project-status" role="status">{escape(project_status)}</div>')
            if s.metrics:
                self._set(self.performance, 'content', f'GPU generation: **{s.metrics["generation_seconds"]:.2f} s** · Received: **{s.metrics["command_to_received_seconds"]:.2f} s**')

            if self.core_session is not None:
                core = self.core_session.snapshot()
                if core['active']:
                    remembered = {id(h) for h, _ in self._core_visibility}
                    for handle in self._legacy_motion_controls:
                        if id(handle) not in remembered:
                            self._core_visibility.append((handle, handle.visible))
                        if handle.visible:
                            handle.visible = False
                    self._set(self.status, 'content', '<div class="sz-status">Scene direction · '
                              + escape(str(core['status'])) + '</div>')
