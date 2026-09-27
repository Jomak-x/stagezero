"""A compact, cancellable character creation panel for the existing cast."""
from __future__ import annotations

from io import BytesIO
import threading
import time

import numpy as np
from PIL import Image

from character_pipeline import SelfHostedCharacterGenerator
from character_reference import validate_reference


_STAGES = {
    '1 / 3 · Designing the character with Neon…': 'Designing appearance',
    '1 / 3 · Reusing your completed design…': 'Reusing appearance',
    '2 / 3 · Building geometry and textures on our GPU…': 'Building 3D character',
    '3 / 3 · Loading the textured 3D character…': 'Finishing character',
    'Character reference ready': 'Appearance ready',
}
_GPU_STAGES = frozenset({
    'Checking GPU memory', 'Waiting for the shared GPU to have enough free memory',
    'Preparing the reconstruction model', 'Loading the reconstruction model',
    'Reconstructing geometry and materials', 'Preparing reference detail',
    'Baking detailed textures', 'Restoring face and clothing detail',
    'Exporting the textured mesh', 'Finishing the character',
})
_ERRORS = {
    'Character worker is not configured; run the character backend launcher':
        'Character creation is not set up. Start the character backend.',
    'Cannot reach the character GPU; start its backend and private tunnel':
        'Cannot reach the character backend. Start it and try again.',
    'Character GPU is still being set up; try again when it is ready':
        'Character backend is starting. Try again shortly.',
    'Character GPU is busy; wait for its current generation to finish':
        'Character backend is busy. Try again shortly.',
    'The shared GPU is still busy; try again when other GPU work finishes':
        'Character backend is busy. Try again shortly.',
    'Character generation timed out': 'Creation timed out. Try again.',
}


class CharacterCreation:
    """Generate one model at a time and hand it to CharacterControls' cast."""

    def __init__(self, server, controls, *, generator_factory=None):
        self.server = server
        self.controls = controls
        self.generator_factory = generator_factory or SelfHostedCharacterGenerator.from_env
        self.lock = threading.RLock()
        self.busy = False
        self._request_id = 0
        self._cancelled = None
        self._finished = None
        self._reference_cache = None
        self.prompt = self.create = self.cancel = self.status = self.reference_view = None

    def build_gui(self, gui):
        with gui.add_folder('Create from description', expand_by_default=False):
            self.prompt = gui.add_text('Describe a person', initial_value='', multiline=True)
            self.create = gui.add_button('Create character')
            self.cancel = gui.add_button('Cancel creation', visible=False)
            self.status = gui.add_html('<div class="sz-status">Describe a full-body person, then create.</div>')
            self.reference_view = gui.add_image(np.zeros((1, 1, 3), dtype=np.uint8),
                                                label='Appearance', visible=False)

        @self.create.on_click
        def create(event):
            self.start(getattr(getattr(event, 'client', None), 'client_id', None))

        @self.cancel.on_click
        def cancel(_):
            self.stop()

    def _set_status(self, message):
        # All messages passed here are fixed strings from this module, never
        # provider output, prompt text, or exception details.
        self.status.content = f'<div class="sz-status">{message}</div>'

    def _active(self, request_id, cancelled):
        return self.busy and self._request_id == request_id and not cancelled.is_set()

    def _set_busy(self, value):
        self.busy = value
        self.create.disabled = value
        self.prompt.disabled = value
        self.cancel.visible = value
        self.cancel.disabled = not value

    def start(self, client_id=None):
        with self.lock:
            if self.busy:
                return False
            description = self.prompt.value.strip()
            if not 1 <= len(description) <= 800:
                self._set_status('Describe a person in 1–800 characters.')
                return False
            if self._reference_cache is not None and self._reference_cache[0] != description:
                self._reference_cache = None
            cached = self._reference_cache
            self._request_id += 1
            request_id = self._request_id
            cancelled = self._cancelled = threading.Event()
            finished = self._finished = threading.Event()
            started = time.monotonic()
            selection_revision = self.controls.selection_revision
            stage = ['Connecting to character backend']
            self.reference_view.visible = False
            self._set_busy(True)
            self._set_status('Connecting to character backend · 0:00')

        def progress(message):
            safe = _STAGES.get(message)
            if safe is None and isinstance(message, str) and message.startswith('2 / 3 · ') and message.endswith('…'):
                detail = message[len('2 / 3 · '):-1]
                if detail in _GPU_STAGES:
                    safe = detail
            with self.lock:
                if self._active(request_id, cancelled):
                    stage[0] = safe or 'Creating character'
                    render_progress()

        def render_progress():
            elapsed = max(0, int(time.monotonic() - started))
            minutes, seconds = divmod(elapsed, 60)
            self._set_status(f'{stage[0]} · {minutes}:{seconds:02d}')

        def reference_ready(data):
            with self.lock:
                if not self._active(request_id, cancelled):
                    return
            validate_reference(data)
            with Image.open(BytesIO(data)) as image:
                image.thumbnail((320, 480))
                pixels = np.asarray(image.convert('RGB'))
            with self.lock:
                if self._active(request_id, cancelled):
                    self._reference_cache = (description, data)
                    self.reference_view.image = pixels
                    self.reference_view.visible = True

        def heartbeat():
            while not finished.wait(1):
                with self.lock:
                    if self._active(request_id, cancelled):
                        render_progress()

        def run():
            try:
                generator = self.generator_factory()
                if cached is not None and hasattr(generator, 'use_reference'):
                    generator.use_reference(description, cached[1])
                if hasattr(generator, 'on_reference'):
                    generator.on_reference = reference_ready
                data = generator.generate(description, progress=progress,
                                          cancelled=cancelled.is_set)
                with self.lock:
                    if not self._active(request_id, cancelled):
                        return
                    stage[0] = 'Checking character'
                    render_progress()
                # Inspect/persist can take substantial CPU time. Let Cancel
                # proceed, then check the request again before any activation.
                asset_id = self.controls.add_generated_file(data, description)
                with self.lock:
                    if not self._active(request_id, cancelled):
                        return
                    if (client_id is not None and
                            self.controls.select_generated(asset_id, client_id, selection_revision)):
                        self._set_status('Character saved. Ready in your character list.')
                    else:
                        self._set_status('Character saved. Choose it from the character list.')
                    self._reference_cache = None
            except (ValueError, OSError) as exc:
                with self.lock:
                    if self._active(request_id, cancelled):
                        self._set_status(_ERRORS.get(str(exc), 'Could not create this character. Try again.'))
            except Exception:
                with self.lock:
                    if self._active(request_id, cancelled):
                        self._set_status('Could not create this character. Try again.')
            finally:
                finished.set()
                with self.lock:
                    if self._request_id == request_id:
                        if cancelled.is_set():
                            self._set_status('Creation stopped. Your current character is unchanged.')
                        self._set_busy(False)

        threading.Thread(target=heartbeat, daemon=True).start()
        threading.Thread(target=run, daemon=True).start()
        return True

    def stop(self):
        with self.lock:
            if not self.busy or self._finished.is_set():
                return False
            self._cancelled.set()
            self.cancel.disabled = True
            self._set_status('Stopping creation…')
            return True
