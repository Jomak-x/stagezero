"""Local character library and Viser-style control tests without provider calls."""
from pathlib import Path
from io import BytesIO
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import threading
import time
import unittest

import numpy as np
import trimesh
from PIL import Image

from character_controls import CharacterControls
from character_library import CharacterLibrary, preview_transform
from test_studio_ui import Gui


def box_glb(extents=(1.0, 2.0, 0.5), center=(0.0, 0.0, 0.0)):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh.export(file_type='glb')


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('Timed out waiting for character generation')
        time.sleep(0.005)


class CharacterGui(Gui):
    def add_image(self, image, **kwargs):
        return self._handle(image=image, visible=kwargs.get('visible', True))

    def add_button(self, label, **kwargs):
        handle = super().add_button(label, **kwargs)
        handle.disabled = kwargs.get('disabled', False)
        return handle


class PreviewHandle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class Scene:
    def __init__(self):
        self.added = []

    def add_glb(self, name, data, **kwargs):
        handle = PreviewHandle()
        self.added.append((name, data, kwargs, handle))
        return handle


class Client:
    def __init__(self):
        self.camera = SimpleNamespace(position=None, look_at=None, up_direction=None)
        self.downloads = []

    def send_file_download(self, name, data):
        self.downloads.append((name, data))


class ControlledGenerator:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.progress = None
        self.on_reference = None

    def generate(self, prompt, progress, cancelled):
        self.calls.append(prompt)
        self.progress = progress
        progress('Fake generator running')
        self.started.set()
        if not self.release.wait(2):
            raise TimeoutError('Test generator was not released')
        if self.error:
            raise self.error
        return self.result


class ReferenceGenerator(ControlledGenerator):
    def __init__(self, reference, result=None, error=None):
        super().__init__(result=result, error=error)
        self.reference = reference
        self.seeded = None
        self.generated_reference = False
        self.reference_image = None

    def use_reference(self, prompt, png):
        self.seeded = (prompt, png)

    def generate(self, prompt, progress, cancelled):
        self.calls.append(prompt)
        self.progress = progress
        self.generated_reference = self.seeded is None
        self.reference_image = self.reference if self.seeded is None else self.seeded[1]
        self.on_reference(self.reference_image)
        self.started.set()
        if not self.release.wait(2):
            raise TimeoutError('Test generator was not released')
        if self.error:
            raise self.error
        return self.result


class CharacterLibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.library = CharacterLibrary(self.folder)
        self.data = box_glb()

    def tearDown(self):
        self.temp.cleanup()

    def test_save_roundtrip_survives_library_recreation(self):
        entry = self.library.save(self.data, '  Copper\n  astronaut  ', 'import')
        self.assertEqual(entry['name'], 'Copper astronaut')
        self.assertFalse(entry['animation_ready'])
        self.assertEqual(entry['source'], 'import')
        restarted = CharacterLibrary(self.folder)
        self.assertEqual(restarted.entries(), [entry])
        self.assertEqual(restarted.read(entry['id']), self.data)

    def test_malformed_glb_and_manifest_are_rejected(self):
        with self.assertRaises(ValueError):
            self.library.save(b'not a glb', 'broken', 'import')
        self.assertEqual(list(self.folder.iterdir()), [])
        entry = self.library.save(self.data, 'Valid', 'import')
        (self.folder / f"{entry['id']}.json").write_text('{broken')
        self.assertEqual(self.library.entries(), [])
        (self.folder / f"{entry['id']}.glb").write_bytes(b'broken')
        with self.assertRaises(ValueError):
            self.library.read(entry['id'])

    def test_identifier_path_traversal_is_refused(self):
        for identifier in ('../outside', 'a' * 32 + '/../outside', 'A' * 32, ''):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                self.library.read(identifier)

    def test_reference_is_saved_with_character_and_checked_on_read(self):
        buffer = BytesIO()
        Image.new('RGB', (32, 48), 'gray').save(buffer, format='PNG')
        reference = buffer.getvalue()
        entry = self.library.save(self.data, 'Explorer', 'neon-trellis2', reference=reference)
        restarted = CharacterLibrary(self.folder)
        self.assertEqual(restarted.read_reference(entry['id']), reference)
        self.assertIsNone(restarted.read_reference('f' * 32))
        (self.folder / (entry['id'] + '.png')).write_bytes(b'bad image')
        with self.assertRaises(ValueError):
            restarted.read_reference(entry['id'])

    def test_active_state_write_does_not_follow_old_temporary_symlink(self):
        entry = self.library.save(self.data, 'Explorer', 'import')
        target = self.folder / 'unrelated.txt'
        target.write_text('keep this content')
        (self.folder / '.active.tmp').symlink_to(target)
        self.library.set_active(entry['id'])
        self.assertEqual(target.read_text(), 'keep this content')
        self.assertEqual(self.library.active(), entry['id'])

    def test_preview_transform_normalizes_height_and_floor(self):
        data = box_glb(extents=(1.0, 2.0, 0.5), center=(3.0, 5.0, -2.0))
        scale, position = preview_transform(data)
        self.assertAlmostEqual(scale, 0.85)
        self.assertTrue(np.allclose(position, (-0.05, -3.4, 1.7)))
        bounds = np.array([[2.5, 4.0, -2.25], [3.5, 6.0, -1.75]])
        transformed = bounds * scale + np.array(position)
        self.assertTrue(np.allclose(transformed[:, 1], [0.0, 1.7]))
        self.assertAlmostEqual(transformed[:, 0].mean(), 2.5)
        self.assertAlmostEqual(transformed[:, 2].mean(), 0.0)

    def test_preview_rejects_flat_or_implausibly_wide_geometry(self):
        for extents in ((1.0, 0.0000001, 1.0), (30.0, 1.0, 1.0)):
            with self.subTest(extents=extents), self.assertRaises(ValueError):
                preview_transform(box_glb(extents=extents))


class CharacterControlTests(unittest.TestCase):
    @staticmethod
    def png(color):
        buffer = BytesIO()
        Image.new('RGB', (32, 48), color).save(buffer, format='PNG')
        return buffer.getvalue()

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.data = box_glb()
        self.scene = Scene()
        self.server = SimpleNamespace(scene=self.scene)
        self.client = Client()
        self.camera = SimpleNamespace(follow=True, follow_handle=SimpleNamespace(value=True))
        self.generators = []
        self.pending_generators = []
        self.controls = CharacterControls(self.server, self.temp.name, self.camera,
                                          generator_factory=self.next_generator)
        self.gui = CharacterGui()
        self.controls.build(self.gui)

    def tearDown(self):
        for generator in self.generators:
            generator.release.set()
        wait_until(lambda: not self.controls.busy)
        self.temp.cleanup()

    def next_generator(self):
        return self.pending_generators.pop(0)

    def click(self, handle):
        handle.callbacks['click'](SimpleNamespace(client=self.client))

    def upload(self, data, name='Imported astronaut.glb'):
        self.controls.upload.value = SimpleNamespace(content=data, name=name)
        self.controls.upload.callbacks['upload'](SimpleNamespace(client=self.client))

    def start(self, generator):
        self.generators.append(generator)
        self.pending_generators.append(generator)
        self.click(self.controls.generate)
        self.assertTrue(generator.started.wait(1))

    def journal(self):
        return json.loads((Path(self.temp.name) / '.last-generation.json').read_text())

    def test_successful_async_generation_saves_and_previews(self):
        generator = ControlledGenerator(result=self.data)
        self.controls.prompt.value = '  Amber astronaut  '
        self.start(generator)
        self.assertTrue(self.controls.busy)
        self.assertTrue(self.controls.generate.disabled)
        self.assertTrue(self.controls.upload.disabled)
        self.assertFalse(self.controls.cancel.disabled)
        generator.release.set()
        wait_until(lambda: not self.controls.busy)
        entries = self.controls.library.entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(generator.calls, ['Amber astronaut'])
        self.assertEqual(self.controls.library.read(entries[0]['id']), self.data)
        self.assertEqual(self.controls.preview_id, entries[0]['id'])
        self.assertEqual(len(self.scene.added), 1)
        self.assertEqual(self.scene.added[0][1], self.data)
        self.assertEqual(self.client.camera.look_at, (0., .85, 0))
        self.assertEqual(self.client.camera.position, (1.7, 1.3, 2.5))
        self.assertEqual(self.client.camera.fov, .65)
        self.assertFalse(self.camera.follow)
        self.assertFalse(self.camera.follow_handle.value)
        self.assertIn('character is ready', self.controls.status.content)
        self.assertEqual(self.journal()['state'], 'succeeded')

    def test_heartbeat_updates_elapsed_and_does_not_replace_final_status(self):
        generator = ControlledGenerator(result=self.data)
        self.start(generator)
        with self.controls.lock:
            self.controls.progress_started -= 65
        wait_until(lambda: '1:0' in self.controls.status.content, timeout=1.5)
        generator.release.set()
        wait_until(lambda: not self.controls.busy)
        final_status = self.controls.status.content
        final_journal = self.journal()
        generator.progress('Late progress after completion')
        self.assertEqual(self.controls.status.content, final_status)
        self.assertEqual(self.journal(), final_journal)

    def test_cancelled_request_callbacks_cannot_change_retry(self):
        first = ControlledGenerator(result=self.data)
        self.start(first)
        self.click(self.controls.cancel)
        first.release.set()
        wait_until(lambda: not self.controls.busy)
        self.assertEqual(self.journal()['state'], 'cancelled')
        second = ControlledGenerator(result=self.data)
        self.start(second)
        running_status = self.controls.status.content
        running_journal = self.journal()
        first.progress('stale stage with credential=old-secret')
        first.on_reference(b'not an image')
        self.assertEqual(self.controls.status.content, running_status)
        self.assertEqual(self.journal(), running_journal)
        second.release.set()
        wait_until(lambda: not self.controls.busy)
        self.assertEqual(self.journal()['state'], 'succeeded')

    def test_gpu_failure_retries_same_reference_and_stale_callback_cannot_replace_it(self):
        png = self.png('gray')
        first = ReferenceGenerator(png, error=ValueError('GPU memory busy'))
        self.controls.prompt.value = 'Explorer'
        self.start(first)
        first.release.set()
        wait_until(lambda: not self.controls.busy)
        self.assertEqual(self.controls._retry_reference, ('Explorer', png))

        second = ReferenceGenerator(self.png('blue'), result=self.data)
        self.start(second)
        self.assertEqual(second.seeded, ('Explorer', png))
        self.assertFalse(second.generated_reference)
        first.on_reference(self.png('red'))
        self.assertEqual(self.controls._retry_reference, ('Explorer', png))
        second.release.set()
        wait_until(lambda: not self.controls.busy)
        self.assertIsNone(self.controls._retry_reference)
        entry = self.controls.library.entries()[0]
        self.assertEqual(self.controls.library.read_reference(entry['id']), png)

    def test_changed_prompt_discards_failed_reference(self):
        first = ReferenceGenerator(self.png('gray'), error=ValueError('GPU memory busy'))
        self.controls.prompt.value = 'Explorer'
        self.start(first)
        first.release.set()
        wait_until(lambda: not self.controls.busy)
        self.controls.prompt.value = 'Pilot'
        second = ReferenceGenerator(self.png('blue'), result=self.data)
        self.start(second)
        self.assertIsNone(second.seeded)
        self.assertTrue(second.generated_reference)
        second.release.set()
        wait_until(lambda: not self.controls.busy)

    def test_journal_omits_prompt_stage_and_error_credentials(self):
        secret = 'Bearer super-secret-token'
        self.controls.prompt.value = f'A pilot with a badge reading {secret}'
        generator = ControlledGenerator(error=ValueError(f'upstream failed: {secret}'))
        self.start(generator)
        generator.progress(f'Private upstream stage: {secret}')
        running = self.journal()
        self.assertEqual(running['stage'], 'Generating character')
        generator.release.set()
        wait_until(lambda: not self.controls.busy)
        journal_text = (Path(self.temp.name) / '.last-generation.json').read_text()
        self.assertNotIn(secret, journal_text)
        self.assertLess(len(journal_text), 512)
        self.assertEqual(self.journal()['state'], 'failed')
        self.assertEqual(set(self.journal()), {'id', 'state', 'prompt_length', 'stage', 'elapsed_seconds'})

    def test_journal_write_does_not_follow_old_temporary_symlink(self):
        target = Path(self.temp.name) / 'unrelated.txt'
        target.write_text('keep this content')
        (Path(self.temp.name) / '.last-generation.tmp').symlink_to(target)
        generator = ControlledGenerator(result=self.data)
        self.start(generator)
        generator.release.set()
        wait_until(lambda: not self.controls.busy)
        self.assertEqual(target.read_text(), 'keep this content')
        self.assertEqual(self.journal()['state'], 'succeeded')

    def test_failure_preserves_existing_library_and_preview(self):
        self.upload(self.data)
        original_id = self.controls.preview_id
        original_handle = self.controls.preview
        generator = ControlledGenerator(error=ValueError('provider rejected prompt'))
        self.start(generator)
        generator.release.set()
        wait_until(lambda: not self.controls.busy)
        self.assertEqual([entry['id'] for entry in self.controls.library.entries()], [original_id])
        self.assertIs(self.controls.preview, original_handle)
        self.assertFalse(original_handle.removed)
        self.assertIn('provider rejected prompt', self.controls.status.content)

    def test_stop_waiting_discards_late_success(self):
        self.upload(self.data)
        original_id = self.controls.preview_id
        original_handle = self.controls.preview
        generator = ControlledGenerator(result=box_glb(center=(2, 2, 0)))
        self.start(generator)
        self.click(self.controls.cancel)
        self.assertIn('Stopping generation', self.controls.status.content)
        generator.release.set()
        wait_until(lambda: not self.controls.busy)
        self.assertIn('Generation stopped', self.controls.status.content)
        self.assertEqual([entry['id'] for entry in self.controls.library.entries()], [original_id])
        self.assertIs(self.controls.preview, original_handle)
        self.assertFalse(original_handle.removed)
        self.assertEqual(len(self.scene.added), 1)

    def test_duplicate_generation_is_ignored_while_busy(self):
        generator = ControlledGenerator(result=self.data)
        self.start(generator)
        self.click(self.controls.generate)
        self.assertEqual(generator.calls, [self.controls.prompt.value.strip()])
        generator.release.set()
        wait_until(lambda: not self.controls.busy)
        self.assertEqual(len(self.controls.library.entries()), 1)

    def test_import_and_invalid_import_preserves_current_character(self):
        self.upload(self.data)
        entry = self.controls.library.entries()[0]
        self.assertEqual(self.controls.preview_id, entry['id'])
        original_handle = self.controls.preview
        self.upload(b'invalid', 'bad.glb')
        self.assertEqual(len(self.controls.library.entries()), 1)
        self.assertIs(self.controls.preview, original_handle)
        self.assertIn('Could not load character', self.controls.status.content)
        self.click(self.controls.hide)
        self.assertTrue(original_handle.removed)
        self.assertIsNone(self.controls.preview)
        self.click(self.controls.show)
        self.assertEqual(self.controls.preview_id, entry['id'])

    def test_repreview_same_character_does_not_duplicate_scene_handle(self):
        self.upload(self.data)
        original_handle = self.controls.preview
        self.click(self.controls.show)
        self.assertEqual(len(self.scene.added), 1)
        self.assertFalse(original_handle.removed)
        self.assertIs(self.controls.preview, original_handle)

    def test_frame_new_client_preserves_shared_follow_setting(self):
        self.upload(self.data)
        self.camera.follow = True
        self.camera.follow_handle.value = True
        newcomer = Client()
        self.controls.frame_client(newcomer, reset_follow=False)
        self.assertTrue(self.camera.follow)
        self.assertTrue(self.camera.follow_handle.value)
        self.assertEqual(newcomer.camera.look_at, (0., .85, 0.))
        self.assertEqual(newcomer.camera.position, (1.7, 1.3, 2.5))
        self.assertEqual(newcomer.camera.fov, .65)

    def test_focus_follows_character_elevation_on_stairs(self):
        positions = np.zeros((34, 3))
        positions[:, 1] = 2.
        positions[0, 1] = 2.8
        self.controls.last_pose = (positions, np.tile(np.eye(3), (34, 1, 1)))
        newcomer = Client()
        self.controls.frame_client(newcomer)
        self.assertEqual(newcomer.camera.look_at, (0., 2.85, 0.))
        self.assertAlmostEqual(newcomer.camera.position[1], 3.3)

    def test_corrupt_optional_reference_does_not_block_saved_model(self):
        image = BytesIO()
        Image.new('RGB', (32, 48), 'gray').save(image, format='PNG')
        entry = self.controls.library.save(self.data, 'Explorer', 'import', reference=image.getvalue())
        (Path(self.temp.name) / (entry['id'] + '.png')).write_bytes(b'corrupt preview')
        self.controls.refresh(entry['id'])
        self.click(self.controls.show)
        self.assertEqual(self.controls.preview_id, entry['id'])
        self.assertFalse(self.controls.reference_view.visible)
        self.assertIn('Character active', self.controls.status.content)

    def test_generated_actor_receives_motion_and_default_is_restored(self):
        actors = []
        visibility = []
        class Actor(PreviewHandle):
            def __init__(self):
                super().__init__()
                self.poses = []
            def update(self, positions, rotations):
                self.poses.append((positions.copy(), rotations.copy()))
        def factory(data, identifier):
            actor = Actor()
            actors.append(actor)
            return actor
        self.controls.actor_factory = factory
        self.controls.default_actor_visibility = visibility.append
        positions = np.zeros((34, 3))
        rotations = np.tile(np.eye(3), (34, 1, 1))
        self.controls.update_pose(positions, rotations)
        self.upload(self.data)
        self.assertEqual(len(actors[0].poses), 1)
        self.assertEqual(visibility, [False])
        identifier = self.controls.preview_id
        self.assertEqual(CharacterLibrary(self.temp.name).active(), identifier)
        positions[0, 0] = 3
        self.controls.update_pose(positions, rotations)
        self.assertEqual(actors[0].poses[-1][0][0, 0], 3)
        self.click(self.controls.hide)
        self.assertTrue(actors[0].removed)
        self.assertEqual(visibility[-1], True)
        self.assertIsNone(self.controls.library.active())

    def test_failed_actor_fit_preserves_current_actor(self):
        self.upload(self.data)
        original = self.controls.preview
        identifier = self.controls.preview_id
        def fail(data, name):
            raise ValueError('Cannot fit this body')
        self.controls.actor_factory = fail
        self.upload(box_glb(extents=(1, 3, .5)))
        self.assertIs(self.controls.preview, original)
        self.assertFalse(original.removed)
        self.assertEqual(self.controls.library.active(), identifier)
        self.assertIn('Cannot fit', self.controls.status.content)


class ModalCharacterGui(CharacterGui):
    def __init__(self):
        super().__init__()
        self.modals = []

    def add_modal(self, title, **kwargs):
        class Modal:
            closed = False
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def close(self):
                if self.closed:
                    raise KeyError('Modal already closed')
                self.closed = True
        modal = Modal()
        modal.title = title
        self.modals.append(modal)
        return modal


class CharacterGalleryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.gui = ModalCharacterGui()
        self.controls = CharacterControls(SimpleNamespace(scene=Scene()), self.temp.name)
        self.controls.build(self.gui)
        self.client = Client()
        self.client.gui = self.gui
        self.event = SimpleNamespace(client=self.client)

    def tearDown(self):
        self.temp.cleanup()

    def test_presets_are_bundled_textured_humanoids_and_never_need_generator(self):
        from character_presets import PRESETS, preset_data
        from character_actor import _mesh_from_glb
        self.assertEqual(len(PRESETS), 3)
        for preset in PRESETS:
            with self.subTest(preset=preset['key']):
                data, reference = preset_data(preset['key'])
                mesh = _mesh_from_glb(data)
                CharacterLibrary._validate_reference(reference)
                extent = mesh.extents
                self.assertTrue(.4 <= extent[1] <= 4)
                self.assertTrue(1.5 <= extent[1] / extent[0] <= 6)
                self.assertGreater(len(mesh.faces), 1000)
                self.assertGreaterEqual(min(mesh.visual.material.baseColorTexture.size), 1024)

    def test_every_preset_fits_and_animates_through_the_real_actor_factory(self):
        from character_presets import PRESETS, preset_data
        from character_actor import GeneratedCharacterActor, _BONES
        from test_character_actor import _FakeScene
        names = {name: i for i, name in enumerate(dict.fromkeys(joint for _, joint, _ in _BONES))}
        skeleton = SimpleNamespace(bone_index=names, nbjoints=len(names),
                                   neutral_joints=np.zeros((len(names), 3), dtype=np.float32))
        positions = np.zeros((len(names), 3), dtype=np.float32)
        positions[:, 1] = .75
        rotations = np.broadcast_to(np.eye(3), (len(names), 3, 3)).copy()
        for preset in PRESETS:
            with self.subTest(preset=preset['key']):
                scene = _FakeScene()
                data, _ = preset_data(preset['key'])
                actor = GeneratedCharacterActor(SimpleNamespace(scene=scene), data, preset['key'], skeleton)
                actor.update(positions, rotations)
                self.assertTrue(np.isfinite(actor.deform_vertices()).all())
                self.assertGreater(len(scene.messages[-1].texture_png) if hasattr(scene.messages[-1], 'texture_png')
                                   else len(scene.messages[-1].props.texture_png), 1000)
                actor.remove()
                self.assertTrue(actor.handle.removed)

    def test_gallery_selection_is_immediate_and_reuses_saved_preset(self):
        self.controls.generator_factory = lambda: self.fail('Presets must not call generation')
        self.controls.open_presets(self.event)
        modal = self.gui.modals[-1]
        button = next(h for h in self.gui.handles if getattr(h, 'label', None) == 'Use Explorer')
        button.callbacks['click'](self.event)
        self.assertTrue(modal.closed)
        first = self.controls.preview_id
        self.assertIsNotNone(first)
        self.controls.use_preset('explorer', self.client)
        self.assertEqual(self.controls.preview_id, first)
        self.assertEqual(len(self.controls.library.entries()), 1)

    def test_closing_generator_does_not_cancel_and_reopening_closed_modal_works(self):
        self.controls.open_generation(self.event)
        self.controls.prompt.value = 'A silver-haired pilot'
        back = next(h for h in reversed(self.gui.handles) if getattr(h, 'label', None) == 'Back to scene')
        back.callbacks['click'](self.event)
        self.assertFalse(self.controls.cancelled.is_set())
        self.controls.open_generation(self.event)
        self.assertEqual(self.controls.prompt.value, 'A silver-haired pilot')
        self.assertFalse(self.gui.modals[-1].closed)

    def test_progress_is_stage_based_monotonic_and_never_100_before_completion(self):
        self.controls.open_generation(self.event)
        self.controls.progress_started = time.monotonic() - 3600
        self.controls.progress_message = '2 / 3 · Reconstructing geometry and materials…'
        self.controls._render_progress()
        self.assertEqual(self.controls.progress_percent, 55)
        self.assertIn('Estimated progress', self.controls.progress_bar.content)
        self.controls.progress_message = '2 / 3 · Waiting for the shared GPU to have enough free memory…'
        self.controls._render_progress()
        self.assertEqual(self.controls.progress_percent, 55)
        self.controls.progress_message = 'Fitting and checking the character body'
        self.controls._render_progress()
        self.assertEqual(self.controls.progress_percent, 98)
        self.assertEqual(self.controls._modal_progress.content, self.controls.progress_bar.content)
        self.assertEqual(self.controls._modal_status.content, self.controls.status.content)


if __name__ == '__main__':
    unittest.main()
