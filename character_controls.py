"""Generate, cast, and animate characters directly inside the studio."""
import threading
import time
import json
from pathlib import Path
import tempfile
import uuid
import re
from io import BytesIO
import numpy as np
from PIL import Image

from character_pipeline import SelfHostedCharacterGenerator
from character_library import CharacterLibrary, preview_transform


_JOURNAL_STAGES = frozenset({
    'Connecting to the character engine',
    'Designing character with Neon',
    'Generating character reference with Neon',
    'Character reference ready',
    '1 / 3 · Designing the character with Neon…',
    '1 / 3 · Reusing your completed design…',
    '2 / 3 · Building geometry and textures on our GPU…',
    '3 / 3 · Loading the textured 3D character…',
    'Fitting and checking the character body',
    *(f'2 / 3 · {stage}…' for stage in (
        'Checking GPU memory', 'Waiting for the shared GPU to have enough free memory',
        'Preparing the reconstruction model',
        'Loading the reconstruction model', 'Reconstructing geometry and materials',
        'Preparing reference detail', 'Baking detailed textures',
        'Restoring face and clothing detail', 'Exporting the textured mesh',
        'Finishing the character',
    )),
})


class CharacterControls:
    def __init__(self, server, folder, camera=None, generator_factory=None,
                 actor_factory=None, default_actor_visibility=None):
        self.server = server
        self.library = CharacterLibrary(folder)
        self.camera = camera
        self.generator_factory = generator_factory or SelfHostedCharacterGenerator.from_env
        self.lock = threading.RLock()
        self.cancelled = threading.Event()
        self.busy = False
        self.preview = None
        self.preview_id = None
        self.labels = {}
        self.actor_factory = actor_factory
        self.default_actor_visibility = default_actor_visibility or (lambda visible: None)
        self.last_pose = None
        self.progress_started = None
        self.progress_message = None
        self.progress_id = None
        self.progress_finished = None
        self._retry_reference = None

    def build(self, gui):
        self.gui = gui
        self._generation_modal = None
        self._modal_status = None
        self._modal_reference = None
        self._modal_progress = None
        self.progress_percent = 0
        gui.add_html('<style>img[alt="Current character"]{max-height:150px;width:100%;object-fit:contain;border-radius:10px}</style>')
        self.browse = gui.add_button('Browse character presets')
        self.open_generator = gui.add_button('Generate a character')
        self.status = gui.add_markdown('Choose a ready-made character or create your own.')
        self.progress_bar = gui.add_html('')
        self.progress_bar.visible = False
        self.reference_view = gui.add_image(np.zeros((1, 1, 3), dtype=np.uint8), label='Current character', visible=False)
        with gui.add_folder('Saved characters & import', expand_by_default=False):
            self.selection = gui.add_dropdown('Your characters', ('No characters yet',))
            self.show = gui.add_button('Use selected character')
            self.frame = gui.add_button('Focus on character')
            self.hide = gui.add_button('Use original actor')
            self.upload = gui.add_upload_button('Import character GLB', mime_type='.glb')
        # Lightweight GUI adapters can render the same form inline.
        if not hasattr(gui, 'add_modal'):
            self._build_generation_form(gui)
        self.refresh()

        @self.browse.on_click
        def browse(event):
            self.open_presets(event)

        @self.open_generator.on_click
        def open_generator(event):
            self.open_generation(event)

        @self.show.on_click
        def show(event):
            self.attempt(lambda: self.display(self.selected(), event.client))

        @self.frame.on_click
        def frame(event):
            if self.preview is not None:
                self.frame_client(event.client)

        @self.hide.on_click
        def hide(_):
            with self.lock:
                if self.preview is not None:
                    self.preview.remove()
                    self.preview = None
                    self.preview_id = None
                self.default_actor_visibility(True)
                self.library.set_active(None)
                self._set_status('Default actor active. Your generated cast is ready to use again.')

        @self.upload.on_upload
        def upload(event):
            def action():
                with self.lock:
                    if self.busy:
                        raise ValueError('Wait for generation to finish before importing')
                    entry = self.library.save(self.upload.value.content, self.upload.value.name, 'import')
                    self.refresh(entry['id'])
                    self.display(entry['id'], event.client)
                    self._set_status('Character added to your cast and placed in the scene.')
            self.attempt(action)

    def _build_generation_form(self, gui):
        self.prompt = gui.add_text('Describe your character', multiline=True,
            initial_value=getattr(self, '_last_prompt',
                'An explorer in a brown leather jacket, olive cargo pants and hiking boots'))
        self.generate = gui.add_button('Generate & use character', disabled=self.busy)
        self.cancel = gui.add_button('Stop generation', disabled=not self.busy)
        self.generate.on_click(self._start_generation)

        @self.cancel.on_click
        def cancel(_):
            with self.lock:
                if self.busy and self.progress_finished is not None and not self.progress_finished.is_set():
                    self.cancelled.set()
                    self.cancel.disabled = True
                    self._set_status('Stopping generation… Your existing character is preserved.')

    def open_generation(self, event):
        if not hasattr(self.gui, 'add_modal'):
            return
        if self._generation_modal is not None:
            self._close_modal(self._generation_modal)
        gui = event.client.gui if getattr(event.client, 'gui', None) else self.gui
        modal = gui.add_modal('Create a character', size='lg', show_close_button=True)
        self._generation_modal = modal
        with modal:
            gui.add_markdown('Describe a person. We’ll design, build and add them automatically.')
            self._build_generation_form(gui)
            self._modal_progress = gui.add_html(self.progress_bar.content)
            self._modal_progress.visible = self.busy
            self._modal_status = gui.add_markdown(self.status.content)
            self._modal_reference = gui.add_image(self.reference_view.image,
                                                  visible=self.reference_view.visible)
            close = gui.add_button('Keep working in the scene' if self.busy else 'Back to scene')
            @close.on_click
            def close_modal(_):
                self._last_prompt = self.prompt.value
                self._close_modal(modal)
                self._generation_modal = None
                self._modal_status = self._modal_reference = self._modal_progress = None
        if self.busy:
            self._render_progress()

    @staticmethod
    def _close_modal(modal):
        # The client close icon may already have removed the modal server-side.
        registry = getattr(getattr(modal, '_gui_api', None), '_modal_handle_from_uuid', None)
        if registry is not None and getattr(modal, '_uuid', None) not in registry:
            return
        try:
            modal.close()
        except KeyError:
            pass

    @staticmethod
    def _card(gui, name, detail, reference):
        if reference:
            with Image.open(BytesIO(reference)) as image:
                image.thumbnail((256, 384))
                gui.add_image(np.asarray(image.convert('RGB')), label=name)
        else:
            gui.add_image(np.full((256, 192, 3), (35, 47, 61), dtype=np.uint8), label=name)

    def open_presets(self, event):
        from character_presets import PRESETS, preset_reference
        gui = event.client.gui if getattr(event.client, 'gui', None) else self.gui
        if not hasattr(gui, 'add_modal'):
            return
        modal = gui.add_modal('Choose your character', size='lg', show_close_button=True)
        with modal:
            gallery_status = gui.add_markdown('')
            gui.add_markdown('Ready to animate. Choose a character to put them in your scene.')
            for preset in PRESETS:
                reference = preset_reference(preset['key'])
                self._card(gui, preset['name'], preset['detail'], reference)
                choose = gui.add_button('Use ' + preset['name'])
                @choose.on_click
                def use_preset(click, key=preset['key']):
                    try:
                        self.use_preset(key, click.client)
                        self._close_modal(modal)
                    except (ValueError, OSError) as exc:
                        gallery_status.content = f'Could not load character: {exc}'
                        self._set_status(gallery_status.content)
            saved = [entry for entry in self.library.entries()
                     if not str(entry.get('source', '')).startswith('preset:')]
            if saved:
                gui.add_markdown('### Your creations')
            for entry in saved:
                self._card(gui, entry['name'], 'Saved character', self._optional_reference(entry['id']))
                choose = gui.add_button('Use ' + entry['name'][:45])
                @choose.on_click
                def use_saved(click, identifier=entry['id']):
                    try:
                        with self.lock:
                            if self.busy:
                                raise ValueError('Stop generation before switching characters')
                            self.display(identifier, click.client)
                        self._close_modal(modal)
                    except (ValueError, OSError) as exc:
                        gallery_status.content = f'Could not load character: {exc}'
                        self._set_status(gallery_status.content)
            close = gui.add_button('Back to scene')
            close.on_click(lambda _: self._close_modal(modal))

    def use_preset(self, key, client=None):
        from character_presets import PRESETS, preset_data
        with self.lock:
            if self.busy:
                raise ValueError('Your new character is being generated; stop it before switching')
            preset = next((item for item in PRESETS if item['key'] == key), None)
            if preset is None:
                raise ValueError('Unknown character preset')
            entry = next((item for item in self.library.entries()
                          if item.get('source') == 'preset:' + key), None)
            if entry is None:
                data, reference = preset_data(key)
                entry = self.library.save(data, preset['name'], 'preset:' + key, reference=reference)
            self.refresh(entry['id'])
            self.display(entry['id'], client)

    def _set_status(self, content):
        self.status.content = content
        if self._modal_status is not None:
            self._modal_status.content = content

    def _start_generation(self, event):
        with self.lock:
            if self.busy:
                return
            description = self.prompt.value.strip()
            self._last_prompt = description
            self.progress_percent = 0
            if not 1 <= len(description) <= 800:
                self._set_status('Enter a character description of 1–800 characters.')
                return
            if self._retry_reference is not None and self._retry_reference[0] != description:
                self._retry_reference = None
            retry_reference = self._retry_reference
            self.cancelled = threading.Event()
            cancelled = self.cancelled
            self.reference_view.visible = False
            if self._modal_reference is not None:
                self._modal_reference.visible = False
            self.busy = True
            self.generate.disabled = True
            self.upload.disabled = True
            self.cancel.disabled = False
            self.progress_started = time.monotonic()
            self.progress_id = uuid.uuid4().hex
            request_id = self.progress_id
            finished = threading.Event()
            self.progress_finished = finished
            self.progress_message = 'Connecting to the character engine'
            self._render_progress()
            self._record_progress('running', description)

        def active():
            return self.progress_id == request_id and not finished.is_set() and not cancelled.is_set()

        def heartbeat():
            while not finished.wait(1):
                with self.lock:
                    if self.busy and active():
                        self._render_progress()

        def progress(message):
            with self.lock:
                if active():
                    if message != self.progress_message:
                        self.progress_message = message
                        self._record_progress('running', description)
                    self._render_progress()

        threading.Thread(target=heartbeat, daemon=True).start()

        def run():
            try:
                generator = self.generator_factory()
                if retry_reference is not None and hasattr(generator, 'use_reference'):
                    generator.use_reference(description, retry_reference[1])
                if hasattr(generator, 'on_reference'):
                    def reference_ready(data):
                        with self.lock:
                            if active():
                                CharacterLibrary._validate_reference(data)
                                self.show_reference(data)
                                self._retry_reference = (description, data)
                    generator.on_reference = reference_ready
                data = generator.generate(description, progress=progress, cancelled=cancelled.is_set)
                with self.lock:
                    if not active():
                        return
                    entry = self.library.save(data, description, getattr(generator, 'source', 'neon-trellis'),
                                              reference=getattr(generator, 'reference_image', None))
                    self.refresh(entry['id'])
                    self.progress_message = 'Fitting and checking the character body'
                    self._render_progress()
                    self.display(entry['id'], event.client)
                    self._retry_reference = None
                    self.progress_percent = 100
                    self._render_progress_bar()
                    self._set_status('Your character is ready. Press Play or describe an action.')
                    self._record_progress('succeeded', description, character_id=entry['id'])
                    finished.set()
            except (ValueError, OSError) as exc:
                with self.lock:
                    if active():
                        self._set_status(f'**Generation stopped:** {exc}\n\nTry again or choose a ready-made character in Browse presets.')
                        self._record_progress('failed', description)
                        finished.set()
            except Exception:
                with self.lock:
                    if active():
                        self._set_status('**Generation stopped:** The character could not be completed. Your current character is unchanged.\n\nRetry the same description.')
                        self._record_progress('failed', description)
                        finished.set()
            finally:
                finished.set()
                with self.lock:
                    self.busy = False
                    self.generate.disabled = False
                    self.upload.disabled = False
                    self.cancel.disabled = True
                    if cancelled.is_set():
                        self._set_status('Generation stopped. Your existing character is preserved.')
                        self._record_progress('cancelled', description)
        threading.Thread(target=run, daemon=True).start()

    def _render_progress_bar(self):
        percent = self.progress_percent
        label = 'Ready' if percent == 100 else 'Estimated progress'
        content = (f'<div style="display:flex;justify-content:space-between;font-size:12px;margin:5px 0 8px">'
                   f'<span>{label}</span><strong>{percent}%</strong></div>'
                   f'<div role="progressbar" aria-label="Character generation" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percent}" '
                   'style="height:7px;border-radius:8px;background:rgba(128,128,128,.2);overflow:hidden">'
                   f'<div style="height:100%;width:{percent}%;background:#51d7bf;border-radius:8px;transition:width .5s"></div></div>')
        self.progress_bar.content = content
        self.progress_bar.visible = True
        if self._modal_progress is not None:
            self._modal_progress.content = content
            self._modal_progress.visible = True

    def _render_progress(self):
        elapsed = max(0, int(time.monotonic() - self.progress_started))
        minutes, seconds = divmod(elapsed, 60)
        message = self.progress_message or 'Getting started'
        # The worker reports stages, not measurable completion. Never fake time-based progress.
        milestones = (('Connecting', 5), ('1 / 3', 15), ('reference ready', 28),
                      ('2 / 3', 35), ('Loading the reconstruction', 42),
                      ('Reconstructing', 55), ('Preparing reference', 68),
                      ('Baking', 76), ('Restoring', 85), ('Exporting', 91),
                      ('Finishing', 94), ('3 / 3', 96), ('Fitting', 98))
        for text, percent in milestones:
            if text.lower() in message.lower():
                self.progress_percent = max(self.progress_percent, percent)
        self._render_progress_bar()
        label = message
        if 'waiting for the shared gpu' in message.lower():
            label = 'Waiting for the character engine to become available'
        elif '1 / 3' in message or 'Neon' in message:
            label = 'Designing the character’s appearance'
        elif '2 / 3' in message:
            label = ('Adding texture and detail' if self.progress_percent >= 68
                     else 'Building the 3D character')
        elif '3 / 3' in message or 'Fitting' in message:
            label = 'Getting your character ready to animate'
        self._set_status(f'**{label}** · {minutes}:{seconds:02d} elapsed')

    def _record_progress(self, state, prompt, *, character_id=None):
        # Keep diagnostics bounded and omit all user/provider text, which may contain secrets.
        doc = {'id': self.progress_id, 'state': state,
               'prompt_length': min(len(prompt), 800),
               'stage': (self.progress_message if isinstance(self.progress_message, str)
                         and self.progress_message in _JOURNAL_STAGES else 'Generating character'),
               'elapsed_seconds': min(round(max(0, time.monotonic() - self.progress_started), 1), 86400.0)}
        if isinstance(character_id, str) and re.fullmatch(r'[a-f0-9]{32}', character_id):
            doc['character_id'] = character_id
        try:
            path = self.library.folder / '.last-generation.json'
            staged = None
            try:
                with tempfile.NamedTemporaryFile('w', dir=self.library.folder,
                                                 prefix='.last-generation-', suffix='.tmp',
                                                 delete=False) as stream:
                    staged = Path(stream.name)
                    stream.write(json.dumps(doc, indent=2) + '\n')
                staged.replace(path)
            finally:
                if staged is not None:
                    staged.unlink(missing_ok=True)
        except OSError:
            pass

    def attempt(self, action):
        try:
            with self.lock:
                action()
        except (ValueError, OSError) as exc:
            self._set_status(f'Could not load character: {exc}')

    def selected(self):
        identifier = self.labels.get(self.selection.value)
        if identifier is None:
            raise ValueError('Generate or import a character first')
        return identifier

    def refresh(self, selected=None):
        self.labels = {f"{entry['name']} · {entry['id'][:8]}": entry['id'] for entry in self.library.entries()}
        self.selection.options = tuple(self.labels) or ('No characters yet',)
        if selected:
            self.selection.value = next(label for label, identifier in self.labels.items() if identifier == selected)
        elif self.selection.value not in self.selection.options:
            self.selection.value = self.selection.options[0]
        self.show.disabled = not bool(self.labels)

    def display(self, identifier, client=None):
        if self.preview is not None and self.preview_id == identifier:
            self.show_reference(self._optional_reference(identifier))
            self.frame_client(client)
            return
        data = self.library.read(identifier)
        reference = self._optional_reference(identifier)
        # Prepare and pose the replacement before removing the current actor.
        if self.actor_factory is not None:
            self._set_status('Fitting character to motion…')
            handle = self.actor_factory(data, identifier)
            try:
                if self.last_pose is not None:
                    handle.update(*self.last_pose)
            except Exception:
                handle.remove()
                raise
        else:
            scale, position = preview_transform(data)
            position = (position[0] - 2.5, position[1], position[2])
            handle = self.server.scene.add_glb(f'/character/{identifier}', data, scale=scale, position=position)
        if self.preview is not None:
            self.preview.remove()
        self.preview, self.preview_id = handle, identifier
        self.default_actor_visibility(False)
        self.library.set_active(identifier)
        self.show_reference(reference)
        self.frame_client(client)
        self._set_status('Character active. Press Play or describe an action.')

    def _optional_reference(self, identifier):
        try:
            return self.library.read_reference(identifier)
        except (ValueError, OSError):
            return None

    def show_reference(self, data):
        if data is None:
            self.reference_view.visible = False
            if self._modal_reference is not None:
                self._modal_reference.visible = False
            return
        with Image.open(BytesIO(data)) as image:
            image.thumbnail((384, 576))
            self.reference_view.image = np.asarray(image.convert('RGB'))
        self.reference_view.visible = True
        if self._modal_reference is not None:
            self._modal_reference.image = self.reference_view.image
            self._modal_reference.visible = True

    def update_pose(self, positions, rotations):
        with self.lock:
            self.last_pose = (np.asarray(positions).copy(), np.asarray(rotations).copy())
            if self.preview is not None and self.actor_factory is not None:
                self.preview.update(*self.last_pose)

    def restore_active(self):
        identifier = self.library.active()
        if identifier:
            self.refresh(identifier)
            self.attempt(lambda: self.display(identifier))

    def frame_client(self, client, *, reset_follow=True):
        if client is not None:
            if self.camera and reset_follow:
                self.camera.follow = False
                if self.camera.follow_handle is not None:
                    self.camera.follow_handle.value = False
            root = self.last_pose[0][0] if self.last_pose is not None else np.array([0., .85, 0.])
            floor = (float(np.min(self.last_pose[0][[6, 7, 13, 14], 1]))
                     if self.last_pose is not None else 0.)
            target = np.array([root[0], floor + .85, root[2]])
            offset = np.array([1.7, .45, 2.5])
            if self.last_pose is not None:
                forward = self.last_pose[1][0] @ np.array([0., 0., 1.])
                yaw = np.arctan2(forward[0], forward[2])
                c, s = np.cos(yaw), np.sin(yaw)
                offset = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ offset
            client.camera.position = tuple(target + offset)
            client.camera.look_at = tuple(target)
            client.camera.up_direction = (0, 1, 0)
            client.camera.fov = .65
