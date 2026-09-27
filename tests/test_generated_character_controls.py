"""Generated models use the existing catalog and browser-confirmed selection."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from character_controls import CharacterControls
from generated_character_rig import GeneratedCharacterRetargeter, export_generated_character
from live_motion import MotionSession
from retargeting import _default_skeleton, neutral_source_pose
from test_generated_character_rig import fixture
from tests.test_character_controls import BrowserBridge, CharacterGui


class GeneratedCharacterCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skeleton = _default_skeleton()
        cls.source = fixture()  # Synthetic full-body mesh, no provider or disk asset.
        cls.rigged = export_generated_character(cls.source, cls.skeleton)

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        positions, rotations = neutral_source_pose(self.skeleton)
        self.session = MotionSession(None, np.tile(positions, (4, 1, 1)),
                                     np.tile(rotations, (4, 1, 1, 1)))
        self.server = SimpleNamespace(get_clients=lambda: {})
        with patch('character_controls.GlbCharacterRenderer', BrowserBridge):
            self.controls = CharacterControls(self.server, self.session, self.skeleton, self.root)

    def add_generated(self, prompt='Original explorer'):
        # Rig export itself is covered by test_generated_character_rig. Reuse its
        # validated output so these tests exercise catalog writes and selection.
        with patch('generated_character_rig.export_generated_character', return_value=self.rigged) as export:
            asset_id = self.controls.add_generated_file(self.source, prompt)
        export.assert_called_once_with(self.source, self.skeleton)
        return asset_id

    def test_close_cancels_in_flight_creation(self):
        from unittest.mock import Mock
        self.controls.creation = Mock()
        self.controls.close()
        self.controls.creation.stop.assert_called_once_with()

    def test_generated_model_saves_and_activates_only_after_browser_ack(self):
        revision = self.controls.selection_revision
        asset_id = self.add_generated('  Original explorer  ')
        entry = self.controls.entries[asset_id]
        self.assertIsInstance(entry.retargeter, GeneratedCharacterRetargeter)
        self.assertTrue(entry.compatibility.motion_ready)
        self.assertEqual(entry.asset.display_name, 'Generated - Original explorer')
        self.assertIsNone(self.controls.active_id)
        self.assertEqual(self.controls.selection_revision, revision)
        sidecar = json.loads((self.root / asset_id / 'generated.json').read_text())
        self.assertEqual(sidecar, {'version': 1, 'source': 'gemini-neon-trellis',
                                   'asset_id': asset_id, 'prompt': 'Original explorer'})

        self.assertTrue(self.controls.select_generated(asset_id, 'client-1', revision))
        self.assertIsNone(self.controls.active_id)
        self.assertEqual(self.controls.renderer.pending[:2], ('client-1', asset_id))
        self.controls.renderer.respond()
        self.controls.tick()
        self.assertEqual(self.controls.active_id, asset_id)
        self.assertIsInstance(self.controls.active_entry.retargeter, GeneratedCharacterRetargeter)
        gui = CharacterGui()
        with patch('character_controls.ScopedUploadLimits', autospec=True):
            self.controls.build_gui(gui)
        self.controls.tick()
        self.assertFalse(gui.handles['Load rig mapping'].visible)
        self.assertFalse(self.controls._mapping_folder.visible)

    def test_newer_user_selection_prevents_generated_model_from_loading(self):
        revision = self.controls.selection_revision
        asset_id = self.add_generated()
        self.controls.select(None, 'client-1')  # User explicitly chose G1 while job ran.
        self.assertFalse(self.controls.select_generated(asset_id, 'client-1', revision))
        self.assertIsNone(self.controls.renderer.pending)
        self.assertIsNone(self.controls.active_id)
        self.assertIn(asset_id, self.controls.entries)  # It remains available to choose.

    def test_restart_restores_generated_fit_only_with_trusted_sidecar(self):
        asset_id = self.add_generated()
        with patch('character_controls.GlbCharacterRenderer', BrowserBridge):
            restarted = CharacterControls(self.server, self.session, self.skeleton, self.root)
        self.assertIn(asset_id, restarted.entries)
        self.assertIsInstance(restarted.entries[asset_id].retargeter, GeneratedCharacterRetargeter)
        self.assertTrue(restarted.entries[asset_id].compatibility.motion_ready)

        (self.root / asset_id / 'generated.json').unlink()
        with patch('character_controls.GlbCharacterRenderer', BrowserBridge):
            ordinary = CharacterControls(self.server, self.session, self.skeleton, self.root)
        self.assertIn(asset_id, ordinary.entries)
        self.assertNotIsInstance(ordinary.entries[asset_id].retargeter, GeneratedCharacterRetargeter)

    def test_invalid_generated_mesh_does_not_enter_catalog(self):
        before = list(self.root.iterdir())
        with self.assertRaises(ValueError):
            self.controls.add_generated_file(b'bad GLB', 'Explorer')
        self.assertEqual(list(self.root.iterdir()), before)
        self.assertEqual(self.controls.entries, {})

    def test_maximum_length_emoji_prompt_restores_with_bounded_sidecar(self):
        prompt = '🧑' * 800
        asset_id = self.add_generated(prompt)
        sidecar = self.root / asset_id / 'generated.json'
        self.assertLessEqual(sidecar.stat().st_size, 4096)
        self.assertEqual(json.loads(sidecar.read_text())['prompt'], prompt)
        with patch('character_controls.GlbCharacterRenderer', BrowserBridge):
            restarted = CharacterControls(self.server, self.session, self.skeleton, self.root)
        self.assertIsInstance(restarted.entries[asset_id].retargeter, GeneratedCharacterRetargeter)

    def test_legacy_neon_origin_still_uses_generated_rig(self):
        asset_id = self.add_generated()
        sidecar = self.root / asset_id / 'generated.json'
        doc = json.loads(sidecar.read_text())
        doc['source'] = 'neon-trellis'
        sidecar.write_text(json.dumps(doc))
        self.assertTrue(self.controls._generated_origin(asset_id))
        with patch('character_controls.GlbCharacterRenderer', BrowserBridge):
            restarted = CharacterControls(self.server, self.session, self.skeleton, self.root)
        self.assertIsInstance(restarted.entries[asset_id].retargeter, GeneratedCharacterRetargeter)


if __name__ == '__main__':
    unittest.main()
