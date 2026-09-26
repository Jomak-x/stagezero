"""Studio controls built on public Viser GUI handles."""
from html import escape
import math
import viser
from takes import MAX_TAKES
from duration_planning import plan_duration

CREATE = 'New take'
EXTEND = 'Add action'
REPLACE = 'Edit ending'
EDIT = 'Edit action'
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
.mantine-Tabs-panel { padding: 2px 4px 8px; }
.mantine-Tabs-list { flex-wrap: nowrap; }
.mantine-Modal-content { border: 1px solid #344453; border-radius: 18px; }
.mantine-Modal-title { font-weight: 700; font-size: 20px; }
.mantine-Button-root { min-height: 38px; border-radius: 9px; transition: background 120ms; }
.mantine-Button-root[data-variant=outline] { border-color: #35495a; color: #d2e3ed; }
.mantine-Button-root:disabled { opacity: .42; }
.mantine-Input-input { background: #0c121b; border-color: #304052; border-radius: 6px; color: #e3edf6; }
.mantine-Input-input:focus { border-color: #83dec7; }
.mantine-Flex-root:has(> div > .mantine-Textarea-root) { flex-direction: column; align-items: stretch !important; gap: 6px; }
.mantine-Flex-root:has(> div > .mantine-Textarea-root) > div { width: 100% !important; }
.mantine-Textarea-input { min-height: 80px; line-height: 1.5; padding: 8px; }
.mantine-Text-root label { color: #a0b1c1; letter-spacing: 0; font-size: 11px; }
.mantine-Checkbox-label { color: #b6c7d5; }
.sz-brand { display: flex; align-items: baseline; gap: 9px; padding: 8px 12px; border-bottom: 1px solid #293746; margin-bottom: 3px; }
.sz-brand strong { font-size: 15px; letter-spacing: -.5px; color: #f0f6fa; }
.sz-brand span { color: #8fa3b8; font-size: 11px; }
.sz-eyebrow { font-size: 10px; color: #83dec7; letter-spacing: 2px; font-weight: 650; text-transform: uppercase; }
.sz-brand h1 { margin: 2px 0; font-size: 21px; letter-spacing: -1px; font-weight: 650; color: #f0f6fa; }
.sz-sub { color: #8fa3b8; font-size: 12px; line-height: 1.6; }
.sz-section { margin: 5px 12px 7px; color: #f0f6fa; font-size: 13px; font-weight: 600; }
.sz-section small { display: block; color: #8fa3b8; font-size: 11px; font-weight: 400; line-height: 1.45; margin-top: 2px; }
.sz-status { margin: 0 12px 4px; padding: 5px 8px; border: 0; border-radius: 7px; background: #15252c; }
.sz-status b { color: #9fe7d5; font-size: 12px; }
.sz-status p { margin: 2px 0 0; color: #a7bdcb; font-size: 11px; line-height: 1.45; }
.sz-clock { margin: 0 12px 2px; font-variant-numeric: tabular-nums; color: #e6f4f6; font-size: 15px; font-weight: 650; }
.sz-clock small { color: #96acb9; font-size: 11px; font-weight: 400; }
.sz-count { margin: -2px 12px 5px; color: #91a9b8; font-size: 11px; }
.sz-count.invalid { color: #ffab9c; }
.sz-hud { position: fixed; left: 26px; top: 24px; z-index: 5; pointer-events: none; }
.sz-hud strong { display: block; font-size: 13px; color: #deedf4; margin-top: 8px; }
.sz-hud span { font-size: 11px; color: #8eabbc; }
.sz-note { margin: 4px 12px 7px; padding-left: 8px; border-left: 2px solid #39545b; color: #93abba; font-size: 11px; line-height: 1.45; }
.sz-step { margin: 7px 12px 3px; color: #e9f5f5; font-size: 12px; font-weight: 650; }
.sz-step span { display: inline-block; width: 20px; height: 20px; margin-right: 7px; border-radius: 50%; background: #22514c; color: #c1f4e7; text-align: center; line-height: 20px; font-size: 11px; }
.sz-preview { margin: 5px 12px 7px; padding: 7px 8px; border: 0; border-radius: 7px; background: #152a29; color: #cfe9e5; font-size: 11px; line-height: 1.45; }
.sz-preview.invalid { border-color: #805145; color: #ffb9a9; }
.sz-segments { display: flex; margin: 8px 12px; height: 7px; gap: 3px; }
.sz-segments i { background: #518c88; border-radius: 3px; min-width: 3px; }
.sz-segments i.active { background: #a6f2db; }
@media (max-width: 600px) { .sz-hud { top: 14px; left: 14px; } }
.sz-sequence-summary { margin: 10px 12px 6px; display:flex; justify-content:space-between; color:#bed3de; font-size:11px; }
.sz-sequence-summary span { color:#7ddfc2; }
.sz-action-list { display:flex; gap:8px; overflow-x:auto; margin:0 12px 8px; padding-bottom:4px; scrollbar-width:thin; }
.sz-action-card { display:flex; gap:8px; min-width:145px; max-width:220px; padding:7px 9px; border-radius:10px; background:linear-gradient(120deg,#193e40,#182933); flex-shrink:0; }
.sz-action-card > span { color:#83dec7; font-size:11px; }
.sz-action-card b { display:block; max-width:175px; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; color:#e1eeef; font-size:11px; font-weight:500; }
.sz-action-card small { color:#88aeb6; font-size:10px; }
.sz-sequence-empty { margin: 8px 12px 12px; color:#7e96a8; font-size:11px; }
</style>"""


def section(gui, title, description=''):
    gui.add_html(f'<div class="sz-section">{escape(title)}<small>{escape(description)}</small></div>')


def step(gui, number, title):
    gui.add_html(f'<div class="sz-step"><span>{number}</span>{escape(title)}</div>')


class StudioUI:
    def __init__(self, server, session, camera, project_folder, scene_controls, character_controls=None):
        self.server, self.session, self.camera = server, session, camera
        self.folder = project_folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.take_map, self.saved_map, self.segment_map = {}, {}, {}
        self.last_take = None
        self._sequence_take_id = None
        self._published_take_selection = None
        self._last_action_choice = None
        self._action_selection_key = None
        self._published_mode = session.mode
        gui = server.gui
        gui.add_html(STYLE)
        self.status = gui.add_html('')
        self.playhead = gui.add_html('')
        self.play_toggle = gui.add_button('Play')
        self.restart = gui.add_button('Back to start', color='gray')
        tabs = gui.add_tab_group()
        with tabs.add_tab('Motion'):
            section(gui, 'Build your sequence')
            self.takes = gui.add_dropdown('Your take', ('No takes yet',))
            self.edit_action = gui.add_dropdown('Compose', (CREATE, EXTEND, EDIT), initial_value=CREATE)
            self.edit_explainer = gui.add_html('')
            self.segments = gui.add_dropdown('Select action', ('No generated actions',))
            self.undo = gui.add_button('Undo last edit', color='gray', visible=False)
            self.replace_time = gui.add_text('Replace from second', initial_value='0.00')
            self.prompt = gui.add_text('Describe the action', initial_value='A person waves with their right hand.', multiline=True)
            self.prompt_count = gui.add_html('')
            self.duration_preview = gui.add_html('')
            self.generate = gui.add_button('Generate motion', icon=viser.Icon.SPARKLES)
            self.delete_action = gui.add_button('Delete selected action', color='red')
            self.cancel = gui.add_button('Cancel generation', color='gray', visible=False)
            self.append_saved = gui.add_button('＋ Add a saved take', color='gray')
            self.sequence_preview = gui.add_html('')
            self.ideas = gui.add_button_group('Try an idea', ('Wave', 'Walk', 'Dance'))
            with gui.add_folder('Timing · automatic', expand_by_default=False):
                self.duration_mode = gui.add_dropdown('Length', (AUTO, SET_DURATION, TARGET_TOTAL), initial_value=AUTO)
                self.duration_seconds = gui.add_text('New motion length (seconds, 0.16–30)', initial_value='4.16')
                gui.add_html('<div class="sz-note">Motion sets the scene length. Backgrounds, characters and effects follow playback automatically.</div>')
            with gui.add_folder('Take options · rename & delete', expand_by_default=False):
                self.take_info = gui.add_html('')
                self.prepare_extend = gui.add_button('Add motion to this take', color='gray', visible=False)
                self.prepare_replace = gui.add_button('Edit from playhead', color='gray', visible=False)
                self.take_name = gui.add_text('Take name', initial_value='')
                self.rename = gui.add_button('Rename', color='gray')
                self.duplicate = gui.add_button('Duplicate take', color='gray')
                self.delete_take = gui.add_button('Delete this take', color='red')
                self.trim = gui.add_button('Trim after playhead · save copy', color='gray')
                self.jump = gui.add_button_group('Find action', ('Previous', 'Go to action', 'Next'))
                self.reuse = gui.add_button('Reuse action prompt', color='gray')
                self.action = gui.add_html('')
            with gui.add_folder('Playback settings', expand_by_default=False):
                self.frames = gui.add_button_group('Frame', ('−1 frame', '+1 frame', 'End'))
                self.seek_time = gui.add_text('Seek to second', initial_value='0.00')
                self.seek_go = gui.add_button('Go to time', color='gray')
                self.speed = gui.add_dropdown('Speed', ('0.25×', '0.5×', '1×', '1.5×', '2×'), initial_value='1×')
                self.loop = gui.add_checkbox('Loop playback', initial_value=False)
        with tabs.add_tab('Background'):
            scene_controls(gui)
        if character_controls is not None:
            with tabs.add_tab('Character'):
                character_controls(gui)
        with tabs.add_tab('Project'):
            self.project_name = gui.add_text('Project name', initial_value='My performance')
            self.save = gui.add_button('Save project + download', icon=viser.Icon.DOWNLOAD)
            with gui.add_folder('Open or start a project', expand_by_default=False):
                self.saved = gui.add_dropdown('Saved projects', ('No saved projects',))
                self.open = gui.add_button('Open selected project', color='gray')
                self.upload = gui.add_upload_button('Open project file', mime_type='.npz')
                self.clear = gui.add_button('New project · back up current', color='gray')
            self.files = gui.add_markdown('')
            with gui.add_folder('Camera', expand_by_default=False):
                camera.build_gui(gui)
            with gui.add_folder('Advanced', expand_by_default=False):
                self.performance = gui.add_markdown('No generation in this session.')
                self.mode = gui.add_dropdown('Source', ('Recorded preview', 'Live ARDY'), initial_value=session.mode)
        self.refresh_saved()
        self.bind()
        self.update()

    @staticmethod
    def _set(handle, property_name, value):
        """Viser setters broadcast to clients, so publish changed properties only."""
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
                  f'<div class="sz-count invalid">{count}/500 · {message}</div>' if invalid else '')

    def _generation_plan(self, take):
        """Resolve the explicit UI choice without moving the playback position."""
        choice = self.edit_action.value
        if choice not in (CREATE, EXTEND, REPLACE, EDIT):
            return None, 'Choose what to make.'
        if choice != CREATE and take is None:
            return None, 'Choose a current take above first.'
        at_frame = None
        prefix_frames = 0
        if choice == EXTEND:
            prefix_frames = len(take.positions)
        elif choice == EDIT:
            segment = self.segment_map.get(self.segments.value)
            if segment is None or segment not in take.segments:
                return None, 'Select an action to edit.'
            at_frame = segment['start']
            prefix_frames = len(take.positions) - (segment['end'] - segment['start'])
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
        context = {
            CREATE: 'Start a separate take. Your other takes stay saved.',
            EXTEND: 'This action plays after the last action in your take.',
            EDIT: 'Select an action, rewrite its direction, then generate. The other actions stay in order.',
            REPLACE: 'Regenerate from the time below. Your original is kept as a separate take.',
        }.get(choice, 'Choose what to make.')
        self._set(self.edit_explainer, 'content', f'<div class="sz-note">{escape(context)}</div>')
        self._set(self.replace_time, 'visible', choice == REPLACE)
        self._set(self.replace_time, 'disabled', busy or choice != REPLACE)
        self._set(self.duration_seconds, 'visible', self.duration_mode.value != AUTO)
        self._set(self.duration_seconds, 'disabled', busy or self.duration_mode.value == AUTO)
        length_label = ('Target total scene length (seconds)' if self.duration_mode.value == TARGET_TOTAL
                        else 'New motion length (seconds, 0.16–30)')
        self._set(self.duration_seconds, 'label', length_label)
        self._set(self.edit_action, 'disabled', busy)
        self._set(self.edit_action, 'visible', True)
        self._set(self.takes, 'visible', bool(self.session.takes))
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
            elif choice == EDIT:
                outcome = f'Replace only the selected action · {new_total:.2f}s total. Other actions stay.'
            else:
                outcome = (f'Keep first {prefix_frames/25:.2f} seconds; new ending makes a '
                           f'{new_total:.2f} second take. Original stays available.')
            estimate = (f'{duration.label}: {duration.seconds:.2f} seconds of new motion'
                        if self.duration_mode.value == AUTO else f'Generate {duration.seconds:.2f} seconds of new motion')
            preview = f'<div class="sz-preview">{escape(outcome)}{" · " + escape(duration.label) if self.duration_mode.value == AUTO else ""}</div>'
        self._set(self.duration_preview, 'content', preview)
        at_limit = len(self.session.takes) >= MAX_TAKES and choice not in (EXTEND, EDIT)
        disabled = busy or not self._valid_prompt(self.prompt.value) or error is not None or at_limit
        self._set(self.generate, 'disabled', disabled)
        self._set(self.generate, 'label', 'Generating…' if busy else 'Take limit reached' if at_limit else {CREATE: 'Generate first action', EXTEND: '＋ Generate next action', REPLACE: 'Generate new ending', EDIT: 'Replace selected action'}[choice])

    @staticmethod
    def _sequence_html(take):
        if take is None:
            return '<div class="sz-sequence-empty">Your first action starts here.</div>'
        cards = ''.join(
            f'<div class="sz-action-card"><span>{i + 1:02d}</span><div><b>{escape(seg["prompt"] or "Motion")}</b>'
            f'<small>{seg["start"]/25:.1f}–{seg["end"]/25:.1f}s</small></div></div>'
            for i, seg in enumerate(take.segments)
        )
        return (f'<div class="sz-sequence-summary">{len(take.segments)} action{ "s" if len(take.segments) != 1 else ""}'
                f' <span>{len(take.positions)/25:.2f}s total</span></div>'
                f'<div class="sz-action-list">{cards}</div>')

    def _open_take_picker(self, event):
        s = self.session
        with s.lock:
            if s.busy or s.active_take not in s.takes:
                return
            target_id = s.active_take
            available = [(t.id, t.name, len(t.positions)/25) for t in s.takes.values() if t.id != target_id]
        panel = event.client.gui if event.client is not None and hasattr(event.client, 'gui') else self.server.gui
        modal = panel.add_modal('Add a saved take', size='md', show_close_button=True)
        completed = False
        with modal:
            feedback = panel.add_markdown('Choose a take to play after the current sequence. Its length is added automatically.'
                                          if available else 'No other takes yet. Choose New take to create another, then return here to join them.')
            for identifier, name, seconds in available:
                choose = panel.add_button(f'＋ {name} · {seconds:.2f}s')
                def append(_, source_id=identifier):
                    nonlocal completed
                    with s.lock:
                        if completed:
                            return
                        if s.active_take != target_id:
                            feedback.content = 'The selected take changed. Close this window and choose your destination take again.'
                            return
                        try:
                            s.append_take(source_id)
                            completed = True
                        except ValueError as exc:
                            feedback.content = str(exc)
                            return
                        self._set(self.edit_action, 'value', EXTEND)
                    modal.close()
                    self.update()
                choose.on_click(append)
            close = panel.add_button('Back to editor', color='gray')
            close.on_click(lambda _: modal.close())

    def _use_selected_action(self):
        s = self.session
        segment = self.segment_map.get(self.segments.value)
        if segment is None:
            return
        s.seek(segment['start'])
        self._set(self.edit_action, 'value', EDIT)
        self._set(self.prompt, 'value', segment['prompt'])
        s.edit_prompt(segment['prompt'])
        self._prompt_feedback()

    def refresh_saved(self):
        self.saved_map = {p.name: p for p in sorted(self.folder.glob('*.stagezero.npz'), key=lambda p: p.stat().st_mtime, reverse=True)}
        self._set(self.saved, 'options', tuple(self.saved_map) or ('No saved projects',))
        if self.saved.value not in self.saved.options:
            self._set(self.saved, 'value', self.saved.options[0])
        self._set(self.open, 'disabled', not self.saved_map)

    def bind(self):
        s = self.session
        @self.play_toggle.on_click
        def toggle_playback(_):
            with s.lock:
                if s.busy or s.kind not in ('recorded', 'generated'):
                    return
                s.pause() if s.playing else s.play()
            self.update()

        @self.restart.on_click
        def back_to_start(_):
            transport_command('Start')
            self.update()

        @self.frames.on_click
        def frame_step(_):
            transport_command(self.frames.value)

        def transport_command(value):
            with s.lock:
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
                if s.busy or s.kind == 'reference':
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
                        s.set_mode(self.mode.value)

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
                if s.busy:
                    return
                if self.edit_action.value == CREATE:
                    if s.mode != 'Live ARDY':
                        s.set_mode('Live ARDY')
                    self._set(self.mode, 'value', 'Live ARDY')
                    s.new_take()
                elif self.edit_action.value == EDIT:
                    if self.segments.value not in self.segment_map and self.segment_map:
                        self._set(self.segments, 'value', next(iter(self.segment_map)))
                    self._use_selected_action()
                elif self.edit_action.value == EXTEND:
                    self._set(self.segments, 'value', 'Select an action')
                if self.edit_action.value == REPLACE:
                    self._set(self.replace_time, 'value', f'{s.frame/s.fps:.2f}')
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
                if s.busy or not self._valid_prompt(self.prompt.value):
                    return
                take = s.takes.get(s.active_take)
                plan, error = self._generation_plan(take)
                if error is not None:
                    s.status = error
                    return
                choice, at_frame, _, duration = plan
                if s.mode != 'Live ARDY':
                    s.set_mode('Live ARDY')
                    self._set(self.mode, 'value', 'Live ARDY')
                try:
                    s.submit(self.prompt.value,
                             seconds=None if self.duration_mode.value == AUTO else duration.seconds,
                             edit_mode={CREATE:'new', EXTEND:'extend', REPLACE:'replace', EDIT:'action'}[choice],
                             at_frame=at_frame)
                except ImportError:
                    s.status = 'Motion generation is unavailable: a required component is missing. Your takes are unchanged.'
            self.update()

        @self.segments.on_update
        def select_action(e):
            if e.client is not None:
                with s.lock:
                    if s.busy: return
                    self._use_selected_action()
                self.update()

        @self.delete_action.on_click
        def delete_action(_):
            with s.lock:
                if s.busy: return
                take = s.takes.get(s.active_take)
                segment = self.segment_map.get(self.segments.value)
                if take is None or segment not in take.segments: return
                try:
                    s.delete_action(take.segments.index(segment))
                except ValueError as exc:
                    s.status = str(exc)
            self.update()

        @self.delete_take.on_click
        def delete_take(_):
            with s.lock:
                if s.busy: return
                try:
                    s.delete_take()
                except ValueError as exc:
                    s.status = str(exc)
            self.update()

        @self.undo.on_click
        def undo(_):
            with s.lock:
                if s.busy: return
                s.undo_edit()
            self.update()

        @self.append_saved.on_click
        def append_saved(event):
            self._open_take_picker(event)

        @self.cancel.on_click
        def cancel(_):
            with s.lock:
                if not s.busy: return
                s.seek(s.frame)
                s.status = 'Generation cancelled · stored motion preserved'

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
                    s.select_take(take_id)

        @self.prepare_extend.on_click
        def prepare_extend(_):
            with s.lock:
                if s.busy or s.active_take not in s.takes: return
                self._set(self.edit_action, 'value', EXTEND)
                s.status = 'Ready to extend this take · describe the next action above, then generate'

        @self.prepare_replace.on_click
        def prepare_replace(_):
            with s.lock:
                if s.busy or s.active_take not in s.takes: return
                self._set(self.edit_action, 'value', REPLACE)
                self._set(self.replace_time, 'value', f'{s.frame/s.fps:.2f}')
                s.status = 'Ready to change this ending · choose the time above, then describe the new action'

        @self.rename.on_click
        def rename(_):
            with s.lock:
                if not s.busy: s.rename_active_take(self.take_name.value)
        @self.duplicate.on_click
        def duplicate(_):
            with s.lock:
                if not s.busy: s.duplicate_active_take()
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
                    s.status = 'Direction copied · choose how to use it above'

        @self.save.on_click
        def save(e):
            try:
                with s.lock:
                    if s.busy: return
                path, data = s.save_project(self.folder, self.project_name.value)
                self.refresh_saved()
                self._set(self.saved, 'value', path.name)
                if e.client is not None: e.client.send_file_download(path.name, data)
            except Exception as exc: s.project_status = f'Save failed: {exc}'

        @self.open.on_click
        def open_saved(_):
            path = self.saved_map.get(self.saved.value)
            if path:
                try: self.open_data(path.read_bytes())
                except OSError as exc: s.project_status = f'Open failed: {exc}'
        @self.upload.on_upload
        def upload(_): self.open_data(self.upload.value.content)
        @self.clear.on_click
        def clear(_):
            try:
                with s.lock:
                    if s.busy: return
                    s.new_project(self.folder)
                    self._set(self.mode, 'value', s.mode)
                self.refresh_saved()
            except Exception as exc: s.project_status = f'Backup failed; project retained: {exc}'

    def open_data(self, data):
        try:
            with self.session.lock:
                if self.session.busy: return
                if self.session.takes:
                    self.session.save_project(self.folder, 'before-open-backup')
                self.session.load_project(data)
                self._set(self.mode, 'value', self.session.mode)
            self.refresh_saved()
        except Exception as exc:
            self.session.project_status = f'Open failed; current takes preserved: {exc}'

    def update(self):
        """Synchronize the sidebar after the viewer advances the session clock."""
        s = self.session
        with s.lock:
            live = s.mode == 'Live ARDY'
            take = s.takes.get(s.active_take)
            selected_id = take.id if take else None
            if selected_id != self._sequence_take_id and not s.busy:
                self._sequence_take_id = selected_id
                self._set(self.edit_action, 'value', EXTEND if take else CREATE)
            has_clip = s.kind in ('recorded', 'generated')
            last_frame = len(s.positions)-1
            elapsed = s.frame/s.fps if has_clip else 0.
            # A clip of N frames occupies N/FPS seconds; its final sample is at (N-1)/FPS.
            duration = len(s.positions)/s.fps if has_clip else 0.
            if s.busy: state = 'Generating'
            elif s.kind == 'reference': state = 'Reference pose'
            elif s.playing: state = 'Playing'
            elif s.frame >= last_frame: state = 'Finished'
            else: state = 'Paused'
            source = take.name if live and take else 'Ready to create' if live else 'Example preview'
            detail = '' if s.status.startswith(ROUTINE_STATUS_PREFIXES) else s.status
            if 'failed motion-quality checks' in detail:
                detail = 'That action could not be generated. Try a simpler movement or add a saved take. Your sequence is unchanged.'
            detail_html = f'<p>{escape(detail)}</p>' if detail else ''
            self._set(self.status, 'content', f'<div class="sz-status"><b>● {state} · {escape(source)}</b>{detail_html}</div>')
            self._set(self.playhead, 'content', f'<div class="sz-clock">{elapsed:.2f} <small>/ {duration:.2f} s</small></div>')
            self._set(self.play_toggle, 'disabled', not has_clip or s.busy)
            self._set(self.restart, 'disabled', not has_clip or s.busy)
            self._set(self.play_toggle, 'label', 'Pause' if s.playing else 'Replay' if has_clip and s.frame >= last_frame else 'Resume' if has_clip and s.frame > 0 else 'Play')
            self._set(self.seek_go, 'disabled', not has_clip or s.busy)
            self._set(self.seek_time, 'disabled', not has_clip or s.busy)
            self._set(self.mode, 'disabled', s.busy)
            if self._published_mode != s.mode:
                self._set(self.mode, 'value', s.mode)
                self._published_mode = s.mode
            self._set(self.prompt, 'disabled', s.busy)
            self._prompt_feedback()
            self.segment_map = {f'{i+1:02d} · {seg["start"]/25:.2f}–{seg["end"]/25:.2f}s · {seg["prompt"]}':seg for i,seg in enumerate(take.segments)} if take else {}
            segments = ('Select an action', *self.segment_map) if self.segment_map else ('No generated actions',)
            action_key = (selected_id, segments)
            if action_key != self._action_selection_key:
                previous = self.segments.value
                self._set(self.segments, 'options', segments)
                self._set(self.segments, 'value', previous if previous in segments else segments[0])
                self._action_selection_key = action_key
            self._set(self.segments, 'visible', take is not None)
            self._set(self.delete_action, 'visible', take is not None and self.edit_action.value == EDIT)
            self._set(self.delete_action, 'disabled', s.busy or self.segments.value not in self.segment_map)
            self._set(self.delete_take, 'disabled', s.busy or take is None)
            self._set(self.undo, 'visible', bool(getattr(s, 'can_undo', False)))
            self._set(self.undo, 'disabled', s.busy)
            self._refresh_generation(take, s.busy)
            self._set(self.cancel, 'visible', s.busy)
            self._set(self.append_saved, 'disabled', take is None or s.busy)
            self._set(self.sequence_preview, 'content', self._sequence_html(take))
            self._set(self.sequence_preview, 'visible', take is None)
            self.take_map = {f'{i+1:02d} · {t.name} · {len(t.positions)/25:.2f}s':t.id for i,t in enumerate(s.takes.values())}
            options = tuple(self.take_map) if take is not None else ('New take · not generated yet', *self.take_map)
            self._set(self.takes, 'options', options)
            selection = next((n for n,t in self.take_map.items() if t == s.active_take), options[0])
            selection_key = (s.active_take, options)
            if selection_key != self._published_take_selection:
                self._set(self.takes, 'value', selection)
                self._published_take_selection = selection_key
            self._set(self.takes, 'disabled', not s.takes or s.busy)
            active_id = (take.id, take.name) if take else None
            if active_id != self.last_take:
                self.last_take = active_id
                self._set(self.take_name, 'value', take.name if take else '')
            for control in (self.rename,self.duplicate,self.take_name):
                self._set(control, 'disabled', take is None or s.busy)
            for control in (self.prepare_extend, self.prepare_replace):
                self._set(control, 'disabled', take is None or s.busy)
            self._set(self.trim, 'disabled', not live or take is None or s.busy or s.frame < 3 or s.frame >= last_frame)
            self._set(self.save, 'disabled', not s.takes or s.busy)
            self._set(self.open, 'disabled', not self.saved_map or s.busy)
            self._set(self.upload, 'disabled', s.busy)
            self._set(self.clear, 'disabled', s.busy)
            self._set(self.reuse, 'disabled', take is None or s.busy or not take.segments)
            self._set(self.segments, 'disabled', take is None or s.busy or not take.segments)
            self._set(self.take_info, 'content', f'<div class="sz-note">{len(take.positions)/25:.2f} seconds · {len(take.segments)} actions in this take.</div>' if take else '<div class="sz-note">Your generated takes will appear here.</div>')
            self._set(self.action, 'content', f'<div class="sz-note">At playhead: {escape(s.current_action() or "No generated action")}</div>')
            self._set(self.files, 'content', s.project_status)
            if s.metrics:
                self._set(self.performance, 'content', f'GPU generation: **{s.metrics["generation_seconds"]:.2f} s** · Received: **{s.metrics["command_to_received_seconds"]:.2f} s**')
