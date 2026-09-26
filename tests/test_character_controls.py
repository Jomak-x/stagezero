"""Character swaps are committed by the browser, independently of motion state."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np

from character_controls import CharacterControls
from character_assets import import_glb, inspect_glb
from live_motion import MotionSession
from retargeting import detect_rig_profile, neutral_source_pose
from upload_events import UploadEvent, UploadFile
from tests.glb_fixtures import base_document_and_binary, make_glb, make_humanoid_glb, make_static_glb


class BrowserBridge:
    def __init__(self, server, on_result):
        self.on_result = on_result
        self.rev = 0
        self.pending = None
        self.active = None
        self.poses = []

    def load(self, asset_id, glb_data, initiating_client_id, **kwargs):
        self.rev += 1
        self.pending = (initiating_client_id, asset_id, self.rev)
        return self.rev

    def respond(self, pending=None, status='loaded'):
        self.on_result(*(pending or self.pending), status, 'Invalid GLB' if status == 'error' else None)

    def commit(self, revision):
        if self.pending is None or self.pending[2] != revision:
            return False
        self.active, self.pending = self.pending, None
        return True

    def reject(self, revision):
        self.pending = None

    def set_pose(self, pose, revision=None):
        self.poses.append((revision, np.asarray(pose).copy()))

    def restore_g1(self):
        self.rev += 1
        self.pending = self.active = None

    def poll(self):
        pass


class GuiHandle(SimpleNamespace):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def on_upload(self, callback):
        self.upload = callback
        return callback

    def on_update(self, callback):
        self.update = callback
        return callback

    def on_click(self, callback):
        self.click = callback
        return callback


class CharacterGui:
    """Public GUI surface; real transfer handling is tested separately."""
    def __init__(self):
        self.handles = {}

    def add_html(self, content):
        return GuiHandle(content=content)

    add_markdown = add_html

    def add_folder(self, label, **kwargs):
        return GuiHandle()

    def add_button(self, label, **kwargs):
        handle = GuiHandle(visible=True, disabled=False, value=None)
        self.handles[label] = handle
        return handle

    add_upload_button = add_button

    def add_dropdown(self, label, options):
        handle = self.add_button(label)
        handle.options, handle.value = options, options[0]
        return handle


class CharacterControlsTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        positions, rotations = neutral_source_pose()
        self.session = MotionSession(None, np.tile(positions, (4, 1, 1)), np.tile(rotations, (4, 1, 1, 1)))
        with patch('character_controls.GlbCharacterRenderer', BrowserBridge):
            self.controls = CharacterControls(SimpleNamespace(), self.session, None, Path(self.directory.name))
        self.bridge = self.controls.renderer
        for name, content in (('rigged', make_humanoid_glb('mixamo')), ('static', make_static_glb())):
            asset = inspect_glb(content, display_name=name + '.glb')
            self.controls.entries[asset.sha256] = self.controls._entry(asset)
            setattr(self, name, asset.sha256)

    def test_only_initiating_success_commits_and_preserves_motion(self):
        self.session.play()
        self.session.motion = np.zeros((4, 414))
        history = self.session.motion
        self.controls.select(self.rigged, 1)
        self.controls.tick((0, 0))
        self.assertIsNone(self.controls.active_id)
        self.bridge.respond((2, self.rigged, self.bridge.rev))
        self.controls.tick((0, 0))
        self.assertIsNone(self.controls.active_id)
        self.bridge.respond()
        self.controls.tick((0, 0))
        self.assertEqual(self.controls.active_id, self.rigged)
        self.assertIs(self.session.motion, history)
        self.assertTrue(self.session.playing)
        self.assertGreater(len(self.bridge.poses), 0)

    def test_failed_static_swap_keeps_active_character_and_playback(self):
        self.controls.select(self.rigged, 1)
        self.bridge.respond()
        self.controls.tick((0, 0))
        self.session.play()
        self.controls.select(self.static, 1)
        self.bridge.respond(status='error')
        self.controls.tick((0, 0))
        self.assertEqual(self.controls.active_id, self.rigged)
        self.assertTrue(self.session.playing)
        self.assertTrue(self.session.character_motion_enabled)
        self.assertEqual(self.controls.mapping_asset_id, self.rigged)

    def test_static_success_stops_playback_and_g1_restores_capability(self):
        self.session.play()
        source_clip = self.session.positions
        self.controls.select(self.static, 1)
        self.bridge.respond()
        self.controls.tick((0, 0))
        self.assertEqual(self.controls.active_id, self.static)
        self.assertFalse(self.session.playing)
        self.assertFalse(self.session.character_motion_enabled)
        self.assertIs(self.session.positions, source_clip)
        self.controls.select(None, 1)
        self.assertTrue(self.session.character_motion_enabled)
        self.assertIsNone(self.controls.active_id)

    def test_stale_success_cannot_replace_latest_selection(self):
        self.controls.select(self.rigged, 1)
        old = self.bridge.pending
        self.controls.select(self.static, 1)
        self.bridge.respond(old)
        self.controls.tick((0, 0))
        self.assertIsNone(self.controls.active_id)
        self.bridge.respond()
        self.controls.tick((0, 0))
        self.assertEqual(self.controls.active_id, self.static)

    def test_paused_pose_is_applied_to_new_character_and_framed(self):
        self.controls.select(self.rigged, 1)
        self.bridge.respond()
        self.controls.tick((0, 0))
        count = len(self.bridge.poses)
        self.controls.select(self.rigged, 1)
        self.bridge.respond()
        self.controls.tick((0, 0))
        self.assertGreater(len(self.bridge.poses), count)
        camera = SimpleNamespace()
        self.controls.frame_character(SimpleNamespace(camera=camera))
        self.assertTrue(np.isfinite(camera.position).all())
        self.assertGreater(np.linalg.norm(camera.position - camera.look_at), 0)

    def test_static_commit_replaces_travelling_character_camera_root(self):
        self.session.positions[:, :, 0] += 10.
        self.controls.select(self.rigged, 1)
        self.bridge.respond()
        self.controls.tick((0, 0))
        self.assertGreater(self.controls.actor_root()[0], 5.)
        self.controls.select(self.static, 1)
        self.bridge.respond()
        self.controls.tick((0, 0))
        expected = self.controls.active_entry.asset.bounds.mean(axis=0)
        expected[1] = 0.
        np.testing.assert_allclose(self.controls.actor_root(), expected)

    def build_upload_gui(self):
        gui = CharacterGui()
        with patch('character_controls.ScopedUploadLimits', autospec=True):
            self.controls.build_gui(gui)
        return gui

    def mapping_payload(self):
        profile = detect_rig_profile(self.controls.entries[self.rigged].asset)
        return UploadFile('first.json', json.dumps({
            'bones': dict(profile.bones), 'root_scale': 1.7,
        }).encode())

    def test_glb_callback_imports_completed_file_when_live_handle_has_newer_file(self):
        gui = self.build_upload_gui()
        handle = gui.handles['Load GLB']
        document, binary = base_document_and_binary()
        document['asset']['generator'] = 'completed first transfer'
        payload = make_glb(document, binary)
        expected = inspect_glb(payload)
        handle.value = UploadFile('later.glb', make_humanoid_glb('mixamo'))
        event = UploadEvent(SimpleNamespace(client_id=1), 1, handle, 'glb-a',
                            UploadFile('first.glb', payload))

        handle.upload(event)

        self.assertEqual(self.controls.mapping_asset_id, expected.asset_id)
        imported = self.controls.entries[expected.asset_id].asset
        self.assertEqual(imported.display_name, 'first.glb')
        self.assertEqual(imported.glb_bytes, payload)

    def test_mapping_callback_uses_completed_file_when_live_handle_has_newer_file(self):
        gui = self.build_upload_gui()
        root = Path(self.directory.name)
        import_glb(make_humanoid_glb('mixamo'), root)
        self.controls.select(self.rigged, 1)
        self.controls.tick()
        handle = gui.handles['Load rig mapping']
        handle.value = UploadFile('later.json', b'invalid later JSON')
        event = UploadEvent(SimpleNamespace(client_id=1), 1, handle, 'mapping-a',
                            self.mapping_payload())

        handle.upload(event)

        saved = json.loads((root / self.rigged / 'mapping.json').read_text())
        self.assertEqual(saved['root_scale'], 1.7)
        self.assertEqual(self.controls.entries[self.rigged].retargeter.profile.root_scale, 1.7)

    def test_delayed_mapping_cannot_write_to_new_or_returned_selection(self):
        gui = self.build_upload_gui()
        root = Path(self.directory.name)
        import_glb(make_humanoid_glb('mixamo'), root)
        other = self.controls.add_file(make_humanoid_glb('g1'), 'Other.glb')
        self.controls.select(self.rigged, 1)
        self.controls.tick()
        original = gui.handles['Load rig mapping']
        original.value = self.mapping_payload()
        event = UploadEvent(SimpleNamespace(client_id=1), 1, original, 'mapping-a',
                            self.mapping_payload())
        callback = original.upload
        original_scale = self.controls.entries[self.rigged].retargeter.profile.root_scale

        self.controls.select(other, 1)
        self.controls.tick()
        callback(event)

        self.assertFalse((root / other / 'mapping.json').exists())
        self.assertFalse((root / self.rigged / 'mapping.json').exists())
        self.assertEqual(self.controls.mapping_asset_id, other)
        self.assertIsNot(gui.handles['Load rig mapping'], original)

        self.controls.select(self.rigged, 1)
        self.controls.tick()
        callback(event)

        self.assertFalse((root / self.rigged / 'mapping.json').exists())
        self.assertEqual(self.controls.entries[self.rigged].retargeter.profile.root_scale, original_scale)

    def test_full_saved_catalog_reimport_preserves_custom_mapping_and_name(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_humanoid_glb('mixamo')
            asset = import_glb(data, root, display_name='Saved performer')
            profile = detect_rig_profile(asset)
            mapping = {'schema_version': 1, 'bones': dict(profile.bones), 'root_scale': 1.7}
            mapping_file = root / asset.asset_id / 'mapping.json'
            mapping_file.write_text(json.dumps(mapping))
            manifest_file = root / asset.asset_id / 'manifest.json'
            original_manifest = manifest_file.read_bytes()
            for index in range(15):
                document, binary = base_document_and_binary()
                document['asset']['generator'] = f'catalog filler {index}'
                import_glb(make_glb(document, binary), root, display_name=f'Filler {index}')

            with patch('character_controls.GlbCharacterRenderer', BrowserBridge):
                controls = CharacterControls(SimpleNamespace(), self.session, None, root)
            self.assertEqual(len(controls.entries), 16)
            original_entry = controls.entries[asset.asset_id]
            self.assertEqual(original_entry.retargeter.profile.root_scale, 1.7)

            self.assertEqual(controls.add_file(data, 'Renamed duplicate.glb'), asset.asset_id)
            self.assertIs(controls.entries[asset.asset_id], original_entry)
            self.assertEqual(original_entry.asset.display_name, 'Saved performer')
            self.assertEqual(original_entry.retargeter.profile.root_scale, 1.7)
            self.assertEqual(manifest_file.read_bytes(), original_manifest)
            self.assertEqual(json.loads(mapping_file.read_text()), mapping)
            self.assertEqual(len(controls.entries), 16)

            document, binary = base_document_and_binary()
            document['asset']['generator'] = 'new asset after full catalog'
            with self.assertRaisesRegex(ValueError, 'Character library is full'):
                controls.add_file(make_glb(document, binary), 'new.glb')


if __name__ == '__main__':
    unittest.main()
