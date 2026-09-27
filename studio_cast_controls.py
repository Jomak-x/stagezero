"""Compact prompt-first cast controls inside the main Studio Motion tab."""

from html import escape
from pathlib import Path
import threading
import time


DEFAULT_PROMPT = 'Two people meet and shake hands.'


class StudioCastControls:
    """Present the validated one-to-three-person pipeline without owning playback."""

    def __init__(self, gui, cast_session, *, on_generate, on_frame, on_export,
                 on_open, provider_available, output_root):
        self.session = cast_session
        self.on_generate = on_generate
        self.on_frame = on_frame
        self.on_export = on_export
        self.on_open = on_open
        self.provider_available = provider_available
        self.output_root = Path(output_root)
        self._lock = threading.RLock()
        self._notice = ''
        self._exporting = False
        self._last_clip = None
        self._sync_prompt_from_archive = True

        gui.add_markdown('Describe what happens with one, two, or three people. Cast and duration are planned automatically. Try “Three people celebrate together in place.”')
        self.prompt = gui.add_text('What happens?', initial_value=DEFAULT_PROMPT, multiline=True)
        self.generate = gui.add_button('Generate', color='green')
        self.cancel = gui.add_button('Cancel', color='gray')
        self.status = gui.add_html('')
        with gui.add_folder('Plan and timing', expand_by_default=False):
            self.plan = gui.add_html('The plan will appear after generation.')
        with gui.add_folder('Files and variations', expand_by_default=False):
            self.seed = gui.add_number('Variation', initial_value=42, min=0, max=2**32-1, step=1)
            self.retry = gui.add_button('Retry', color='gray')
            self.save = gui.add_button('Save performance', color='gray')
            self.open = gui.add_upload_button('Open performance', mime_type='.npz')
            self.export = gui.add_button('Export video', color='gray')
            self.frame = gui.add_button('Frame everyone', color='gray')
        gui.add_markdown('Research preview · paired movement is generated jointly; a third person uses independent generated and authored observer motion. Review foot planting and contact before use. Noncommercial research terms apply.')

        @self.generate.on_click
        def generate(event):
            self._call(lambda: self.on_generate(self.prompt.value, int(self.seed.value), event.client),
                       require_provider=True, require_active=True)

        @self.cancel.on_click
        def cancel(_event):
            self._call(self.session.cancel, allow_generation=True)

        @self.retry.on_click
        def retry(_event):
            self._call(self.session.retry, require_provider=True, require_active=True)

        @self.frame.on_click
        def frame(event):
            self._call(lambda: self.on_frame(event.client), require_clip=True)

        @self.save.on_click
        def save(event):
            def action():
                data = self.session.save()
                self.output_root.mkdir(parents=True, exist_ok=True)
                path = self.output_root / f'{time.time_ns()}.cast.stagezero.npz'
                path.write_bytes(data)
                if event.client:
                    event.client.send_file_download(path.name, data)
                self._notice = 'Saved complete performance.'
            self._call(action, require_clip=True)

        @self.open.on_upload
        def open_file(event):
            def action():
                before = self.session.timeline_clip()
                self.on_open(bytes(event.target.value.content))
                if self.session.timeline_clip() is not before:
                    self._sync_prompt_from_archive = True
                self._notice = 'Opened performance.'
            self._call(action)

        @self.export.on_click
        def export(event):
            with self._lock:
                self._notice = ''
                try:
                    self._require_idle(require_clip=True)
                except (ValueError, RuntimeError) as exc:
                    self._notice = str(exc)
                    self.tick()
                    return
                self._exporting = True
                self._notice = 'Exporting the complete performance…'
                self.tick()

            def run():
                try:
                    path = Path(self.on_export(event.client))
                    if event.client:
                        event.client.send_file_download(path.name, path.read_bytes())
                    notice = 'Exported the complete performance.'
                except Exception as exc:
                    notice = str(exc)
                with self._lock:
                    self._notice = notice
                    self._exporting = False
                    self.tick()

            threading.Thread(target=run, name='studio-cast-export', daemon=True).start()

        self.tick()

    @staticmethod
    def _set(handle, key, value):
        if getattr(handle, key, None) != value:
            setattr(handle, key, value)

    def _provider_ready(self):
        return bool(self.provider_available() if callable(self.provider_available)
                    else self.provider_available)

    def _require_idle(self, *, allow_generation=False, require_clip=False,
                      require_provider=False, require_active=False):
        state = self.session.snapshot()
        if self._exporting or state.get('capturing'):
            raise RuntimeError('Wait for video export to finish.')
        if state.get('busy') and not allow_generation:
            raise RuntimeError('Finish or cancel generation before changing the performance.')
        if require_provider and not self._provider_ready():
            raise RuntimeError('AI cast needs both motion providers configured.')
        if require_active and not state.get('active'):
            raise RuntimeError('Select AI cast in Motion first.')
        if require_clip and not state.get('total_frames'):
            raise ValueError('Open or generate a complete performance first.')

    def _call(self, action, *, allow_generation=False, require_clip=False,
              require_provider=False, require_active=False):
        with self._lock:
            self._notice = ''
            try:
                self._require_idle(allow_generation=allow_generation,
                                   require_clip=require_clip,
                                   require_provider=require_provider,
                                   require_active=require_active)
                action()
            except (ValueError, RuntimeError, OSError) as exc:
                self._notice = str(exc)
            self.tick()

    @staticmethod
    def _progress_message(progress):
        if not isinstance(progress, dict):
            return str(progress)
        phase = progress.get('phase')
        if phase == 'planning':
            return 'Planning the cast and action timings…'
        if phase in ('core', 'approach'):
            return (f'Generating movement · {progress.get("completed_windows", 0)}/'
                    f'{progress.get("total_windows", "?")} steps ready…')
        if phase == 'pair_approach_attempt':
            attempt = progress.get('attempt', 1)
            return ('Generating the approach…' if attempt == 1 else
                    f'Trying a wider approach · attempt {attempt} of '
                    f'{progress.get("maximum_attempts", 3)}…')
        if phase == 'pair_generation':
            return 'Generating the shared interaction…'
        if phase == 'beat':
            return (f'Building action {progress.get("beat_index", 0) + 1} '
                    f'of {progress.get("total_beats", "?")}…')
        return 'Checking the complete performance…'

    @staticmethod
    def _plan_html(clip):
        metadata = clip.metadata or {}
        plan = metadata.get('plan') or {}
        lines = [f'<b>{escape(str(plan.get("title", "Performance")))}</b>',
                 f'{len(clip.actor_ids)} {"person" if len(clip.actor_ids) == 1 else "people"} · '
                 f'{clip.frames / clip.fps:.1f} seconds']
        for beat in plan.get('beats', []):
            if not isinstance(beat, dict):
                continue
            people = ', '.join(map(str, beat.get('actor_ids', [])))
            prompt = str(beat.get('prompt', ''))
            seconds = beat.get('seconds')
            suffix = f' ({seconds:g}s)' if isinstance(seconds, (float, int)) else ''
            lines.append(escape(f'{people}: {prompt}{suffix}'))
            for actor, action in beat.get('concurrent_solos', {}).items():
                lines.append(escape(f'At the same time · {actor}: {action}'))
        for warning in plan.get('warnings', []):
            lines.append(escape(str(warning)))
        if metadata.get('wall_seconds') is not None:
            try:
                seconds = float(metadata['wall_seconds'])
                replayed = any(isinstance(source, dict) and source.get('replayed_native_source')
                               for source in metadata.get('sources', []))
                label = 'Rebuilt with archived interactions in' if replayed else 'Generated in'
                lines.append(f'{label} {seconds:.1f}s')
            except (ValueError, TypeError):
                pass
        return '<p>' + '<br>'.join(lines) + '</p>'

    def mark_loaded(self):
        """Refresh plan and original prompt after a project opens outside this panel."""
        with self._lock:
            self._sync_prompt_from_archive = True
            self._last_clip = None
            self.tick()

    def tick(self):
        with self._lock:
            state = self.session.snapshot()
            available = self._provider_ready()
            busy = bool(state.get('busy') or state.get('capturing') or self._exporting)
            ready = bool(state.get('total_frames'))
            self._set(self.generate, 'disabled', busy or not available or
                      not state.get('active') or not self.prompt.value.strip())
            self._set(self.cancel, 'disabled', not state.get('busy') or
                      state.get('capturing') or self._exporting)
            for handle in (self.prompt, self.seed, self.open):
                self._set(handle, 'disabled', busy)
            for handle in (self.frame, self.save, self.export):
                self._set(handle, 'disabled', busy or not ready)
            self._set(self.retry, 'disabled', busy or not available or
                      not state.get('active') or not state.get('failure'))

            message = (self._notice or state.get('failure') or
                       (state.get('progress') if state.get('busy') else None) or
                       (None if available else 'AI cast needs both motion providers configured.') or
                       state.get('status') or 'Ready. Duration is automatic.')
            clip = self.session.timeline_clip()
            fallback = (clip.metadata or {}).get('concurrency_fallback') if clip else None
            if fallback and not busy and not state.get('failure') and not self._notice:
                message = 'Requested simultaneous action was not achieved; preserved the original performance. ' + str(fallback)
            self._set(self.status, 'content', '<p>' +
                      escape(self._progress_message(message)) + '</p>')

            clip = self.session.timeline_clip()
            if clip is not self._last_clip:
                self._set(self.plan, 'content', self._plan_html(clip) if clip else
                          'The plan will appear after generation.')
                if clip and self._sync_prompt_from_archive:
                    plan = (clip.metadata or {}).get('plan') or {}
                    original = plan.get('prompt') or (clip.metadata or {}).get('prompt')
                    if original:
                        self._set(self.prompt, 'value', str(original))
                self._last_clip = clip
                self._sync_prompt_from_archive = False
