"""Current cast catalog and bundled preset tests; no GPU or provider calls."""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from types import SimpleNamespace
import json
import unittest

import numpy as np
import trimesh
from PIL import Image

from character_controls import CharacterControls
from character_library import CharacterLibrary, preview_transform
from character_presets import PRESETS, preset_data
from retargeting import _default_skeleton, neutral_source_pose


def box_glb(extents=(1., 2., .5)):
    return trimesh.creation.box(extents=extents).export(file_type='glb')


class CharacterLibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.library = CharacterLibrary(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_save_roundtrip_and_reference_validation(self):
        image = BytesIO()
        Image.new('RGB', (24, 32), 'blue').save(image, format='PNG')
        data = box_glb()
        entry = self.library.save(data, '  Copper\n astronaut ', 'import', reference=image.getvalue())
        restarted = CharacterLibrary(self.temp.name)
        self.assertEqual(restarted.entries(), [entry])
        self.assertEqual(restarted.read(entry['id']), data)
        self.assertEqual(restarted.read_reference(entry['id']), image.getvalue())
        self.assertFalse(entry['animation_ready'])
        self.assertEqual(entry['name'], 'Copper astronaut')

    def test_invalid_models_and_identifier_traversal_are_rejected(self):
        with self.assertRaises(ValueError):
            self.library.save(b'bad', 'bad', 'import')
        for identifier in ('../outside', 'A' * 32, 'a' * 32 + '/..'):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                self.library.read(identifier)
        entry = self.library.save(box_glb(), 'Valid', 'import')
        (Path(self.temp.name) / f"{entry['id']}.json").write_text('{broken')
        self.assertEqual(self.library.entries(), [])
        (Path(self.temp.name) / f"{entry['id']}.glb").write_bytes(b'broken')
        with self.assertRaises(ValueError):
            self.library.read(entry['id'])

    def test_active_state_is_atomic_and_preview_fits_floor(self):
        entry = self.library.save(box_glb(), 'Valid', 'import')
        target = Path(self.temp.name) / 'unrelated.txt'
        target.write_text('keep')
        (Path(self.temp.name) / '.active.tmp').symlink_to(target)
        self.library.set_active(entry['id'])
        self.assertEqual(target.read_text(), 'keep')
        self.assertEqual(self.library.active(), entry['id'])
        scale, position = preview_transform(box_glb())
        self.assertAlmostEqual(scale, .85)
        self.assertAlmostEqual(position[1], .85)


class Renderer:
    def __init__(self):
        self.loads = []
        self.commits = []
        self.rejections = []
        self.poses = []

    def load(self, asset_id, data, client_id, **kwargs):
        self.loads.append((asset_id, data, client_id, kwargs))
        return len(self.loads)

    def poll(self):
        pass

    def reject(self, revision):
        self.rejections.append(revision)

    def commit(self, revision):
        self.commits.append(revision)
        return True

    def set_pose(self, pose, **kwargs):
        self.poses.append(pose)

    def restore_g1(self):
        pass


class PresetCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.skeleton = _default_skeleton()
        positions, rotations = neutral_source_pose(self.skeleton)
        self.session = SimpleNamespace(lock=RLock(), positions=positions[None],
                                       rotations=rotations[None], frame=0, motion_enabled=True)
        self.session.set_character_motion_enabled = lambda enabled: setattr(self.session, 'motion_enabled', enabled)
        self.client = SimpleNamespace(camera=SimpleNamespace(position=None, look_at=None,
                                                              up_direction=None, fov=None))
        self.server = SimpleNamespace(get_clients=lambda: {'client': self.client})
        self.renderer = Renderer()
        self.controls = CharacterControls(self.server, self.session, self.skeleton,
                                          self.temp.name, renderer=self.renderer)

    def tearDown(self):
        self.temp.cleanup()

    def test_all_bundled_presets_are_self_contained_and_fit_for_motion(self):
        for preset in PRESETS:
            with self.subTest(preset=preset['key']):
                data, image = preset_data(preset['key'])
                self.assertGreater(len(data), 100_000)
                self.assertGreater(len(image), 1000)
                asset_id = self.controls.use_preset(preset['key'], 'client')
                self.assertNotEqual(self.controls.active_id, asset_id,
                                    'Browser acknowledgement must commit the swap')
                self.assertTrue(self.controls.entries[asset_id].compatibility.motion_ready)
                revision = self.renderer.loads[-1][3]
                self.assertTrue(revision['required_nodes'])
                self.controls._on_result('client', asset_id, len(self.renderer.loads), 'loaded', None)
                self.controls.tick(pose_key=preset['key'])
                self.assertEqual(self.controls.active_id, asset_id)
                self.assertTrue(self.session.motion_enabled)
                self.assertIsNotNone(self.client.camera.look_at)

    def test_preset_reuses_saved_fit_after_restart_and_invalid_key_is_rejected(self):
        first = self.controls.use_preset('ranger', 'client')
        self.assertEqual(self.controls.use_preset('ranger', 'client'), first)
        self.assertEqual(len(self.controls.entries), 1)
        restarted = CharacterControls(self.server, self.session, self.skeleton,
                                      self.temp.name, renderer=Renderer())
        self.assertEqual(restarted.use_preset('ranger', 'client'), first)
        self.assertEqual(len(restarted.entries), 1)
        with self.assertRaises(ValueError):
            restarted.use_preset('../unknown', 'client')

    def test_invalid_upload_and_failed_load_keep_existing_actor(self):
        first = self.controls.use_preset('ranger', 'client')
        self.controls._on_result('client', first, 1, 'loaded', None)
        self.controls.tick()
        with self.assertRaises(ValueError):
            self.controls.add_file(b'bad', 'bad.glb')
        second = self.controls.use_preset('explorer', 'client')
        self.controls._on_result('client', second, 2, 'failed', 'Browser rejected model')
        self.controls.tick()
        self.assertEqual(self.controls.active_id, first)
        self.assertTrue(self.session.motion_enabled)
        self.assertIn('Load failed', self.controls.status)

    def test_legacy_flat_catalog_import_preserves_source_and_active_choice(self):
        with TemporaryDirectory() as source_dir:
            source = CharacterLibrary(source_dir)
            entry = source.save(preset_data('ranger')[0], 'My saved ranger', 'neon-trellis')
            source.set_active(entry['id'])
            original = source.read(entry['id'])
            mapping = self.controls.import_legacy_catalog(source_dir)
            self.assertIn(entry['id'], mapping)
            self.assertEqual(self.controls._initial_id, mapping[entry['id']])
            self.assertEqual(source.read(entry['id']), original)
            self.assertEqual(source.active(), entry['id'])
            self.assertEqual(self.controls.import_legacy_catalog(source_dir), mapping)
            self.assertEqual(len(self.controls.entries), 1)


if __name__ == '__main__':
    unittest.main()
