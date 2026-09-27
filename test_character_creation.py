"""Small UI and async handoff tests without network or GPU calls."""
from io import BytesIO
import threading
import time
from types import SimpleNamespace
import unittest

from PIL import Image

from character_creation import CharacterCreation


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('Timed out waiting for character creation')
        time.sleep(.005)


def png():
    output = BytesIO()
    Image.new('RGB', (24, 32), 'blue').save(output, format='PNG')
    return output.getvalue()


class Handle(SimpleNamespace):
    def on_click(self, callback):
        self.click = callback
        return callback

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class Gui:
    def __init__(self):
        self.folders = []
        self.labels = []

    def add_folder(self, label, **kwargs):
        self.folders.append((label, kwargs))
        return Handle()

    def add_text(self, label, initial_value='', **kwargs):
        self.labels.append(label)
        return Handle(value=initial_value, disabled=False)

    def add_button(self, label, **kwargs):
        self.labels.append(label)
        return Handle(disabled=False, visible=kwargs.get('visible', True))

    def add_html(self, content):
        return Handle(content=content)

    def add_image(self, image, **kwargs):
        return Handle(image=image, visible=kwargs.get('visible', True))


class Cast:
    def __init__(self):
        self.added = []
        self.selected = []
        self.active = 'existing-character'
        self.selection_revision = 0

    def add_generated_file(self, data, prompt):
        self.added.append((data, prompt))
        return f'asset-{len(self.added)}'

    def select(self, asset_id, client_id):
        self.selected.append((asset_id, client_id))
        self.selection_revision += 1

    def select_generated(self, asset_id, client_id, expected_revision):
        if self.selection_revision != expected_revision:
            return False
        self.select(asset_id, client_id)
        return True


class Generator:
    def __init__(self, result=b'glb', error=None, reference=None):
        self.result, self.error, self.reference = result, error, reference
        self.on_reference = None
        self.started = threading.Event()
        self.release = threading.Event()
        self.prompts = []
        self.seeded = None
        self.progress = None

    def use_reference(self, prompt, reference):
        self.seeded = (prompt, reference)

    def generate(self, prompt, progress, cancelled):
        self.prompts.append(prompt)
        self.progress = progress
        progress('private stage: token=secret')
        if self.reference is not None and self.on_reference is not None:
            self.on_reference(self.seeded[1] if self.seeded else self.reference)
        self.started.set()
        if not self.release.wait(2):
            raise TimeoutError('Generator was not released')
        if self.error:
            raise self.error
        return self.result


class CharacterCreationTests(unittest.TestCase):
    def setUp(self):
        self.cast = Cast()
        self.generators = []
        self.pending = []
        self.gui = Gui()
        self.creation = CharacterCreation(SimpleNamespace(), self.cast,
                                          generator_factory=self.next_generator)
        self.creation.build_gui(self.gui)

    def tearDown(self):
        for generator in self.generators:
            generator.release.set()
        wait_until(lambda: not self.creation.busy)

    def next_generator(self):
        return self.pending.pop(0)

    def start(self, generator, client_id='client-1'):
        self.generators.append(generator)
        self.pending.append(generator)
        self.creation.create.click(SimpleNamespace(client=SimpleNamespace(client_id=client_id))
                                   if client_id else SimpleNamespace(client=None))
        self.assertTrue(generator.started.wait(1))

    def test_panel_is_small_and_hidden_until_used(self):
        self.assertEqual(self.gui.folders, [('Create from description', {'expand_by_default': False})])
        self.assertEqual(self.gui.labels, ['Describe a person', 'Create character', 'Cancel creation'])
        self.assertEqual(self.creation.prompt.value, '')
        self.assertFalse(self.creation.cancel.visible)
        self.assertFalse(self.creation.reference_view.visible)
        self.assertFalse(self.creation.start())
        self.assertEqual(self.cast.added, [])

    def test_success_saves_then_selects_using_initiating_client(self):
        generator = Generator(reference=png())
        self.creation.prompt.value = '  A realistic explorer  '
        self.start(generator)
        self.assertTrue(self.creation.busy)
        self.assertTrue(self.creation.create.disabled)
        self.assertTrue(self.creation.cancel.visible)
        self.assertTrue(self.creation.reference_view.visible)
        self.assertNotIn('secret', self.creation.status.content)
        generator.release.set()
        wait_until(lambda: not self.creation.busy)
        self.assertEqual(self.cast.added, [(b'glb', 'A realistic explorer')])
        self.assertEqual(self.cast.selected, [('asset-1', 'client-1')])
        self.assertFalse(self.creation.cancel.visible)
        self.assertEqual(self.cast.active, 'existing-character')  # Browser ack owns activation.

    def test_no_client_saves_without_selecting(self):
        generator = Generator()
        self.creation.prompt.value = 'Explorer'
        self.start(generator, client_id=None)
        generator.release.set()
        wait_until(lambda: not self.creation.busy)
        self.assertEqual(self.cast.added, [(b'glb', 'Explorer')])
        self.assertEqual(self.cast.selected, [])

    def test_gemini_setup_error_is_actionable(self):
        message = 'Set GEMINI_API_KEY in the environment or .runtime/characters.env'
        generator = Generator(error=ValueError(message))
        self.creation.prompt.value = 'Explorer'
        self.start(generator)
        generator.release.set()
        wait_until(lambda: not self.creation.busy)
        self.assertIn('GEMINI_API_KEY', self.creation.status.content)
        self.assertIn('.runtime/characters.env', self.creation.status.content)
        self.assertEqual(self.cast.added, [])

    def test_failure_preserves_current_cast_and_reuses_matching_reference(self):
        reference = png()
        first = Generator(error=ValueError('secret=credential'), reference=reference)
        self.creation.prompt.value = 'Explorer'
        self.start(first)
        first.release.set()
        wait_until(lambda: not self.creation.busy)
        self.assertEqual(self.cast.added, [])
        self.assertEqual(self.cast.active, 'existing-character')
        self.assertNotIn('credential', self.creation.status.content)

        second = Generator(result=b'new-glb', reference=png())
        self.start(second)
        self.assertEqual(second.seeded, ('Explorer', reference))
        first.on_reference(b'invalid stale image')
        first.progress('secret=credential')
        second.release.set()
        wait_until(lambda: not self.creation.busy)
        self.assertEqual(self.cast.added, [(b'new-glb', 'Explorer')])
        self.assertEqual(self.cast.selected, [('asset-1', 'client-1')])

    def test_cancel_discards_late_result_and_allows_new_request(self):
        first = Generator(reference=png())
        self.creation.prompt.value = 'Explorer'
        self.start(first)
        self.creation.cancel.click(None)
        self.assertFalse(first.release.is_set())
        first.release.set()
        wait_until(lambda: not self.creation.busy)
        self.assertEqual(self.cast.added, [])
        self.assertEqual(self.cast.selected, [])
        self.assertIn('stopped', self.creation.status.content)
        second = Generator()
        self.creation.prompt.value = 'Pilot'
        self.start(second)
        self.assertIsNone(second.seeded)
        second.release.set()
        wait_until(lambda: not self.creation.busy)
        self.assertEqual(self.cast.added, [(b'glb', 'Pilot')])

    def test_cast_rejection_does_not_select_or_change_active_character(self):
        def reject(data, prompt):
            raise ValueError('private parser detail: secret')
        self.cast.add_generated_file = reject
        generator = Generator()
        self.creation.prompt.value = 'Explorer'
        self.start(generator)
        generator.release.set()
        wait_until(lambda: not self.creation.busy)
        self.assertEqual(self.cast.selected, [])
        self.assertEqual(self.cast.active, 'existing-character')
        self.assertNotIn('secret', self.creation.status.content)

    def test_cancel_during_slow_save_keeps_model_saved_but_never_selects_it(self):
        saving = threading.Event()
        release_save = threading.Event()
        original_add = self.cast.add_generated_file

        def slow_add(data, prompt):
            saving.set()
            self.assertTrue(release_save.wait(2))
            return original_add(data, prompt)

        self.cast.add_generated_file = slow_add
        generator = Generator()
        self.creation.prompt.value = 'Explorer'
        self.start(generator)
        generator.release.set()
        self.assertTrue(saving.wait(1))
        self.assertTrue(self.creation.stop())
        self.assertTrue(self.creation.busy)
        release_save.set()
        wait_until(lambda: not self.creation.busy)
        self.assertEqual(self.cast.added, [(b'glb', 'Explorer')])
        self.assertEqual(self.cast.selected, [])
        self.assertEqual(self.cast.active, 'existing-character')
        self.assertIn('stopped', self.creation.status.content)

    def test_user_selection_during_generation_wins(self):
        generator = Generator()
        self.creation.prompt.value = 'Explorer'
        self.start(generator)
        self.cast.select('another-character', 'client-1')
        generator.release.set()
        wait_until(lambda: not self.creation.busy)
        self.assertEqual(self.cast.added, [(b'glb', 'Explorer')])
        self.assertEqual(self.cast.selected, [('another-character', 'client-1')])
        self.assertIn('Choose it from the character list', self.creation.status.content)

    def test_prompt_is_bounded_and_duplicate_start_is_ignored(self):
        self.creation.prompt.value = 'x' * 801
        self.assertFalse(self.creation.start())
        self.assertEqual(self.cast.added, [])
        generator = Generator()
        self.creation.prompt.value = 'Explorer'
        self.start(generator)
        self.assertFalse(self.creation.start())
        generator.release.set()
        wait_until(lambda: not self.creation.busy)
        self.assertEqual(generator.prompts, ['Explorer'])


if __name__ == '__main__':
    unittest.main()
