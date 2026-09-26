"""Studio controls built on public Viser GUI handles."""
from html import escape
import math
import viser
from takes import MAX_TAKES
from duration_planning import plan_duration

CREATE = 'Create new take'
EXTEND = 'Extend selected take'
REPLACE = 'Replace ending from time'
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
.sz-brand { display: flex; align-items: baseline; gap: 9px; padding: 5px 12px; border-bottom: 1px solid #293746; margin-bottom: 3px; }
.sz-brand strong { font-size: 15px; letter-spacing: -.5px; color: #f0f6fa; }
.sz-brand span { color: #8fa3b8; font-size: 11px; }
.sz-eyebrow { font-size: 10px; color: #83dec7; letter-spacing: 2px; font-weight: 650; text-transform: uppercase; }
.sz-brand h1 { margin: 2px 0; font-size: 21px; letter-spacing: -1px; font-weight: 650; color: #f0f6fa; }
.sz-sub { color: #8fa3b8; font-size: 12px; line-height: 1.6; }
.sz-section { margin: 5px 12px 7px; color: #f0f6fa; font-size: 13px; font-weight: 600; }
.sz-section small { display: block; color: #8fa3b8; font-size: 11px; font-weight: 400; line-height: 1.45; margin-top: 2px; }
.sz-status { margin: 0 12px 4px; padding: 5px 8px; border: 1px solid #2c424a; border-radius: 7px; background: #15252c; }
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
.sz-preview { margin: 5px 12px 7px; padding: 7px 8px; border: 1px solid #36554f; border-radius: 7px; background: #152a29; color: #cfe9e5; font-size: 11px; line-height: 1.45; }
.sz-preview.invalid { border-color: #805145; color: #ffb9a9; }
.sz-segments { display: flex; margin: 8px 12px; height: 7px; gap: 3px; }
.sz-segments i { background: #518c88; border-radius: 3px; min-width: 3px; }
.sz-segments i.active { background: #a6f2db; }
@media (max-width: 600px) { .sz-hud { top: 14px; left: 14px; } }
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
        self._published_take_selection = None
        self._last_action_choice = None
        gui = server.gui
        gui.add_html(STYLE)
        gui.add_html('<div class="sz-hud"><div class="sz-eyebrow">STAGEZERO / VIEWPORT</div><strong>Motion studio</strong><span>Drag to pan · Alt-drag to orbit · pinch/Alt-scroll to zoom · WASD/QE</span></div>')
        gui.add_html('<div class="sz-brand"><strong>StageZero.</strong><span>Motion studio</span></div>')
        self.status = gui.add_html('')
        self.playhead = gui.add_html('')
        self.transport = gui.add_button_group('Playback', ('Start', 'Play', 'Pause'))
        tabs = gui.add_tab_group()
        with tabs.add_tab('Direct'):
            step(gui, 1, 'Choose what to make')
            self.edit_action = gui.add_dropdown('Action', (CREATE, EXTEND, REPLACE), initial_value=CREATE)
            self.edit_explainer = gui.add_html('')
            self.replace_time = gui.add_text('Keep motion before this time (seconds)', initial_value='0.00')
            step(gui, 2, 'Describe the motion')
            self.prompt = gui.add_text('What should the person do?', initial_value='A person waves with their right hand.', multiline=True)
            self.prompt_count = gui.add_html('')
            step(gui, 3, 'Choose how long')
            self.duration_mode = gui.add_dropdown('Length', (AUTO, SET_DURATION, TARGET_TOTAL), initial_value=AUTO)
            self.duration_seconds = gui.add_text('New motion length (seconds, 0.16–30)', initial_value='4.16')
            self.duration_preview = gui.add_html('')
            self.generate = gui.add_button('Generate', icon=viser.Icon.SPARKLES)
            self.cancel = gui.add_button('Cancel generation', color='gray', visible=False)
            with gui.add_folder('Example directions', expand_by_default=False):
                self.ideas = gui.add_button_group('Try an example', ('Wave', 'Walk', 'Dance'))
            with gui.add_folder('How editing works', expand_by_default=False):
                gui.add_html('<div class="sz-note">The selected take sets scene playback length. Props and effects follow its timeline. Drag the bottom time ruler to inspect motion.</div>')
                gui.add_html('<div class="sz-note">Create new starts another take. Extend adds motion to the end of the selected take. Replace ending keeps motion before your chosen time and creates a separate take with new motion after it; the original stays available. Choose a take in Takes first for either edit.</div>')
        with tabs.add_tab('Takes'):
            with gui.add_folder('Precise playback', expand_by_default=False):
                self.frames = gui.add_button_group('Frame', ('−1 frame', '+1 frame', 'End'))
                self.seek_time = gui.add_text('Seek to second', initial_value='0.00')
                self.seek_go = gui.add_button('Go to time', color='gray')
            section(gui, 'Your takes', 'Select a version to watch or change. Each generated ending is saved as a separate take.')
            self.takes = gui.add_dropdown('Selected take', ('No takes yet',))
            self.take_info = gui.add_html('')
            self.prepare_extend = gui.add_button('Prepare to add motion at the end', color='gray')
            self.prepare_replace = gui.add_button('Prepare to change the ending', color='gray')
            gui.add_html('<div class="sz-note">Choose an edit above, then open Direct to write the new direction and duration. Changing an ending keeps the original take. Motion before the chosen time stays in the new take; motion after it is replaced there.</div>')
            self.take_name = gui.add_text('Take name', initial_value='')
            self.rename = gui.add_button('Rename take', color='gray')
            self.duplicate = gui.add_button('Make a copy of this take', color='gray')
            self.trim = gui.add_button('Save a shorter copy ending here', color='gray')
            section(gui, 'Motion in this take', 'Jump to an action or reuse its direction for another version.')
            self.segments = gui.add_dropdown('Generated motion', ('No generated actions',))
            self.jump = gui.add_button_group('Find motion', ('Previous', 'Go to action', 'Next'))
            self.reuse = gui.add_button('Copy this direction to Direct', color='gray')
            self.action = gui.add_html('')
            self.speed = gui.add_dropdown('Speed', ('0.25×', '0.5×', '1×', '1.5×', '2×'), initial_value='1×')
            self.loop = gui.add_checkbox('Loop playback', initial_value=False)
        with tabs.add_tab('Scene'):
            scene_controls(gui)
        if character_controls is not None:
            with tabs.add_tab('Character'):
                character_controls(gui)
        with tabs.add_tab('Camera'):
            section(gui, 'Camera')
            camera.build_gui(gui)
        with tabs.add_tab('Project'):
            section(gui, 'Project')
            self.project_name = gui.add_text('Project name', initial_value='My performance')
            self.save = gui.add_button('Save project + download', icon=viser.Icon.DOWNLOAD)
            self.saved = gui.add_dropdown('Saved projects', ('No saved projects',))
            self.open = gui.add_button('Open selected project', color='gray')
            self.upload = gui.add_upload_button('Open project file', mime_type='.npz')
            self.clear = gui.add_button('New project · back up current', color='gray')
            self.files = gui.add_markdown('')
            gui.add_html('<div class="sz-note">Opening a project automatically backs up the current takes. All connected viewers share the same project.</div>')
            with gui.add_folder('Generation diagnostics', expand_by_default=False):
                self.performance = gui.add_markdown('No generation in this session.')
            with gui.add_folder('Advanced: playback source', expand_by_default=False):
                self.mode = gui.add_dropdown('Source', ('Recorded preview', 'Live ARDY'), initial_value=session.mode)
                gui.add_html('<div class="sz-note">Recorded preview is the included example. Generating or selecting a take switches to your live project automatically.</div>')
            gui.add_markdown('[ARDY G1 model](https://huggingface.co/nvidia/ARDY-G1-RP-25FPS-Horizon52) · [Recorded motion source](https://huggingface.co/datasets/bones-studio/seed)')
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
                  f'<div class="sz-count{" invalid" if invalid else ""}">{count}/500 · {message}</div>')

    def _generation_plan(self, take):
        """Resolve the explicit UI choice without moving the playback position."""
        choice = self.edit_action.value
        if choice not in (CREATE, EXTEND, REPLACE):
            return None, 'Choose what to make.'
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
        context = {
            CREATE: 'Start a separate take from the reference pose. Existing takes stay in the project.',
            EXTEND: 'Add motion after the selected take ends. Its existing motion stays in this take.',
            REPLACE: 'Keep motion before the time below and make a new ending in a separate take. The original stays available. This replaces the rest of the new take, not a middle section.',
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
        at_limit = len(self.session.takes) >= MAX_TAKES and choice != EXTEND
        disabled = busy or not self.session.character_motion_enabled or not self._valid_prompt(self.prompt.value) or error is not None or at_limit
        self._set(self.generate, 'disabled', disabled)
        self._set(self.generate, 'label', 'Generating…' if busy else 'Take limit reached' if at_limit else 'Generate motion')

    def refresh_saved(self):
        self.saved_map = {p.name: p for p in sorted(self.folder.glob('*.stagezero.npz'), key=lambda p: p.stat().st_mtime, reverse=True)}
        self._set(self.saved, 'options', tuple(self.saved_map) or ('No saved projects',))
        if self.saved.value not in self.saved.options:
            self._set(self.saved, 'value', self.saved.options[0])
        self._set(self.open, 'disabled', not self.saved_map)

    def bind(self):
        s = self.session
        @self.transport.on_click
        def transport(_):
            transport_command(self.transport.value)

        @self.frames.on_click
        def frame_step(_):
            transport_command(self.frames.value)

        def transport_command(value):
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
                plan, error = self._generation_plan(take)
                if error is not None:
                    s.status = error
                    return
                choice, at_frame, _, duration = plan
                if s.mode != 'Live ARDY':
                    s.set_mode('Live ARDY')
                    self._set(self.mode, 'value', 'Live ARDY')
                s.submit(self.prompt.value,
                         seconds=None if self.duration_mode.value == AUTO else duration.seconds,
                         edit_mode={CREATE:'new', EXTEND:'extend', REPLACE:'replace'}[choice],
                         at_frame=at_frame)

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
                s.status = 'Ready to extend this take · open Direct, add a direction, then generate'

        @self.prepare_replace.on_click
        def prepare_replace(_):
            with s.lock:
                if s.busy or s.active_take not in s.takes: return
                self._set(self.edit_action, 'value', REPLACE)
                self._set(self.replace_time, 'value', f'{s.frame/s.fps:.2f}')
                s.status = 'Ready to change this ending · open Direct and choose the time to keep'

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
                    s.status = 'Direction copied · open Direct and choose how to use it'

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
            # Viser button groups cannot be disabled; static previews hide
            # motion controls while their callbacks remain guarded as well.
            self._set(self.transport, 'visible', s.character_motion_enabled)
            self._set(self.frames, 'visible', s.character_motion_enabled)
            take = s.takes.get(s.active_take)
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
            detail_html = f'<p>{escape(detail)}</p>' if detail else ''
            self._set(self.status, 'content', f'<div class="sz-status"><b>● {state} · {escape(source)}</b>{detail_html}</div>')
            self._set(self.playhead, 'content', f'<div class="sz-clock">{elapsed:.2f} <small>/ {duration:.2f} s</small></div>')
            self._set(self.seek_go, 'disabled', not has_clip or s.busy)
            self._set(self.seek_time, 'disabled', not has_clip or s.busy)
            self._set(self.mode, 'disabled', s.busy)
            self._set(self.prompt, 'disabled', s.busy)
            self._prompt_feedback()
            self._refresh_generation(take, s.busy)
            self._set(self.cancel, 'visible', s.busy)
            self.take_map = {f'{i+1:02d} · {t.name} · {len(t.positions)/25:.2f}s':t.id for i,t in enumerate(s.takes.values())}
            options = tuple(self.take_map) or ('No takes yet',)
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
            self._set(self.take_info, 'content', f'<div class="sz-note">{len(take.positions)/25:.2f} seconds · {len(take.segments)} generated actions. Select it to preview, or prepare an edit below.</div>' if take else '<div class="sz-note">No take selected yet. Open Direct to create one.</div>')
            self.segment_map = {f'{i+1:02d} · {seg["start"]/25:.2f}s · {seg["prompt"][:30]}':seg for i,seg in enumerate(take.segments)} if take else {}
            segments = tuple(self.segment_map) or ('No generated actions',)
            self._set(self.segments, 'options', segments)
            self._set(self.action, 'content', f'<div class="sz-note">At playhead: {escape(s.current_action() or "No generated action")}</div>')
            self._set(self.files, 'content', s.project_status)
            if s.metrics:
                self._set(self.performance, 'content', f'GPU generation: **{s.metrics["generation_seconds"]:.2f} s** · Received: **{s.metrics["command_to_received_seconds"]:.2f} s**')
