"""Small prompt-first interface for bounded, explicitly staged cast performances."""
from html import escape
from pathlib import Path
import threading


class PromptSceneControls:
    def __init__(self, gui, session, *, backgrounds, on_generate, on_background,
                 on_frame, on_export, on_open, output_root, on_play=None, get_background=None):
        self.session = session
        self.on_generate, self.on_background = on_generate, on_background
        self.on_frame, self.on_export, self.on_open = on_frame, on_export, on_open
        self.on_play = on_play or session.play
        self.output_root = Path(output_root)
        self.backgrounds = dict(backgrounds)
        self.get_background = get_background
        self._lock = threading.RLock()
        self._syncing_background = False
        self._notice = ''
        self._exporting = False
        self._last_plan = None
        gui.add_markdown('### Describe your scene\nChoose a background, then describe one, two, or three people. Cast, places and duration are planned automatically.')
        self.background = gui.add_dropdown('Background', tuple(backgrounds), initial_value=next(iter(backgrounds)))
        self.prompt = gui.add_text('What happens?', initial_value='Two people meet and shake hands.', multiline=True)
        self.generate = gui.add_button('Generate', color='green')
        self.cancel = gui.add_button('Cancel', color='gray')
        self.status = gui.add_html('')
        self.play = gui.add_button('Play', color='green')
        self.pause = gui.add_button('Pause', color='gray')
        self.frame = gui.add_button('Frame everyone', color='gray')
        with gui.add_folder('Plan and timing', expand_by_default=False):
            self.plan = gui.add_html('The plan will appear here after generation.')
        with gui.add_folder('Files and variations', expand_by_default=False):
            self.seed = gui.add_number('Variation', initial_value=42, min=0, max=2**32-1, step=1)
            self.retry = gui.add_button('Retry', color='gray')
            self.save = gui.add_button('Save performance', color='gray')
            self.open = gui.add_upload_button('Open performance', mime_type='.npz')
            self.export = gui.add_button('Export video', color='gray')
        gui.add_markdown('Research preview. Pair interactions are generated jointly; other people hold their pose during that beat. Foot planting and physical contact still need review.')

        @self.generate.on_click
        def generate(event):
            self._call(lambda: self.on_generate(self.prompt.value, int(self.seed.value), event.client))

        @self.background.on_update
        def background(event):
            if self._syncing_background and event.client is None: return
            self._call(lambda: self.on_background(self.background.value))

        @self.cancel.on_click
        def cancel(event): self._call(self.session.cancel, allow_busy=True)
        @self.retry.on_click
        def retry(event): self._call(self.session.retry)
        @self.play.on_click
        def play(event): self._call(self.on_play)
        @self.pause.on_click
        def pause(event): self._call(self.session.pause)
        @self.frame.on_click
        def frame(event): self._call(lambda: self.on_frame(event.client))
        @self.save.on_click
        def save(event):
            def action():
                data = self.session.save()
                self.output_root.mkdir(parents=True, exist_ok=True)
                import time
                path = self.output_root / f'{time.time_ns()}.cast.stagezero.npz'
                path.write_bytes(data)
                if event.client: event.client.send_file_download(path.name, data)
                self._notice = 'Saved complete performance.'
            self._call(action)
        @self.open.on_upload
        def open_file(event):
            self._call(lambda: self.on_open(bytes(event.target.value.content)))
        @self.export.on_click
        def export(event):
            with self._lock:
                if self._exporting: return
                try:
                    self._require_idle()
                    if not self.session.snapshot().get('total_frames'):
                        raise ValueError('Open or generate a complete performance before exporting.')
                except (ValueError, RuntimeError) as exc:
                    self._notice = str(exc)
                    self.tick()
                    return
                self._exporting = True
                self._notice = 'Exporting the complete performance…'
                self.tick()
            def run():
                try:
                    path = self.on_export(event.client)
                    if event.client: event.client.send_file_download(path.name, path.read_bytes())
                    notice = 'Exported the complete performance.'
                except Exception as exc:
                    notice = str(exc)
                finally:
                    with self._lock:
                        self._notice = notice
                        self._exporting = False
                        self.tick()
            threading.Thread(target=run, daemon=True).start()
        self.tick()

    def _require_idle(self, *, allow_busy=False):
        state = self.session.snapshot()
        if self._exporting or state.get('capturing'):
            raise RuntimeError('Wait for video export to finish.')
        if state.get('busy') and not allow_busy:
            raise RuntimeError('Finish or cancel generation before changing the scene.')

    def _call(self, action, *, allow_busy=False):
        with self._lock:
            self._notice = ''
            try:
                self._require_idle(allow_busy=allow_busy)
                action()
            except (ValueError, RuntimeError, OSError) as exc:
                self._notice = str(exc)
            self.tick()

    @staticmethod
    def _set(handle, key, value):
        if getattr(handle, key, None) != value: setattr(handle, key, value)

    def tick(self):
        with self._lock:
            self._tick()

    def _sync_background(self):
        if self.get_background is None:
            return
        scene = self.get_background()
        label = next((name for name, document in self.backgrounds.items() if document == scene), None)
        options = tuple(self.backgrounds)
        if label is None:
            label = 'Saved background'
            if isinstance(scene, dict) and scene.get('name'):
                label += ' · ' + str(scene['name'])
            # This option describes an imported scene, not a different preset.
            if label in self.backgrounds:
                label += ' (project)'
            options += (label,)
        self._syncing_background = True
        try:
            self._set(self.background, 'options', options)
            self._set(self.background, 'value', label)
        finally:
            self._syncing_background = False

    def _tick(self):
        self._sync_background()
        state = self.session.snapshot()
        busy = bool(state.get('busy') or state.get('capturing') or self._exporting)
        ready = bool(state.get('total_frames'))
        self._set(self.generate, 'disabled', busy or not self.prompt.value.strip())
        self._set(self.cancel, 'disabled', not state.get('busy', False))
        for item in (self.background, self.prompt, self.seed, self.open): self._set(item, 'disabled', busy)
        for item in (self.play, self.pause, self.frame, self.save, self.export): self._set(item, 'disabled', busy or not ready)
        self._set(self.retry, 'disabled', busy or not (state.get('error') or state.get('failure')))
        message = self._notice or state.get('error') or state.get('failure') or (state.get('progress') if state.get('busy') else None) or state.get('status') or 'Ready. Duration is automatic.'
        if isinstance(message, dict):
            phase = message.get('phase')
            if phase == 'planning': message = 'Planning the cast, places and action timings…'
            elif phase in ('core', 'approach'):
                done, total = message.get('completed_windows', 0), message.get('total_windows', '?')
                message = f'Generating movement · {done}/{total} steps ready…'
            elif phase == 'pair_approach_attempt':
                attempt = message.get('attempt', 1)
                message = 'Generating the approach…' if attempt == 1 else f'Trying a wider approach · attempt {attempt} of {message.get("maximum_attempts", 3)}…'
            elif phase == 'pair_generation': message = 'Generating the shared interaction…'
            elif phase == 'beat': message = f'Building action {message.get("beat_index", 0)+1} of {message.get("total_beats", "?")}…'
            else: message = 'Checking the complete scene…'
        self._set(self.status, 'content', '<p>' + escape(str(message)) + '</p>')
        clip = self.session.timeline_clip()
        if clip is not None and clip is not self._last_plan:
            meta = clip.metadata
            plan = meta.get('plan', {})
            beats = plan.get('beats', [])
            lines = [f'<b>{escape(str(plan.get("title", "Performance")))}</b>',
                     f'{len(clip.actor_ids)} {"person" if len(clip.actor_ids) == 1 else "people"} · {clip.frames / clip.fps:.1f} seconds']
            for beat in beats:
                lines.append(escape(f'{", ".join(beat["actor_ids"])}: {beat["prompt"]} ({beat["seconds"]:g}s)'))
            for warning in plan.get('warnings', []): lines.append(escape(str(warning)))
            if meta.get('wall_seconds') is not None:
                replayed = any(source.get("replayed_native_source") for source in meta.get("sources", []))
                label = "Rebuilt with archived interactions in" if replayed else "Generated in"
                lines.append(f"{label} {float(meta['wall_seconds']):.1f}s")
            self._set(self.plan, 'content', '<p>' + '<br>'.join(lines) + '</p>')
            if plan.get('prompt'):
                self._set(self.prompt, 'value', plan['prompt'])
            self._last_plan = clip
