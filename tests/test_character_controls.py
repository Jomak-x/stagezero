"""Character swaps are committed by the browser, independently of motion state."""
import json
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np

from character_controls import CharacterControls
from character_assets import DEFAULT_LIMITS, import_glb, inspect_glb
from character_diagnostics import standing_reference
from character_geometry import posed_minimum_y
from live_motion import MotionSession
from retargeting import detect_rig_profile, neutral_source_pose
from tests.glb_fixtures import base_document_and_binary, make_glb, make_humanoid_glb, make_static_glb


class BrowserBridge:
    def __init__(self, server, on_result):
        self.on_result = on_result
        self.rev = 0
        self.pending = None
        self.active = None
        self.poses = []
        self.loads = []

    def load(self, asset_id, glb_data, initiating_client_id, **kwargs):
        self.rev += 1
        self.pending = (initiating_client_id, asset_id, self.rev)
        self.loads.append((asset_id, kwargs))
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
    def remove(self):
        self.removed = True

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
    """Small UI boundary fake exposing the same visible actions as Viser."""
    def __init__(self):
        self.handles = {}

    def add_html(self, content):
        return GuiHandle(content=content)

    def add_markdown(self, content):
        return GuiHandle(content=content)

    def add_folder(self, label, **kwargs):
        return GuiHandle()

    def add_button(self, label, **kwargs):
        handle = GuiHandle(visible=kwargs.get('visible', True), disabled=False, value=None)
        self.handles[label] = handle
        return handle

    def add_upload_button(self, label, **kwargs):
        return self.add_button(label, **kwargs)

    def add_dropdown(self, label, options):
        handle = self.add_button(label)
        handle.options, handle.value = options, options[0]
        return handle


def upload_event(handle, client, file):
    """Model the file captured at completion, independently of live value."""
    return SimpleNamespace(client=client, client_id=client.client_id, target=handle, file=file)


class CharacterControlsTests(unittest.TestCase):
    def test_gui_upload_budget_can_hold_the_full_glb_and_mapping(self):
        gui = CharacterGui()
        with patch('character_controls.ScopedUploadLimits', autospec=True) as limiter:
            self.controls.build_gui(gui)
        limiter.assert_called_once_with(
            self.server, max_total_bytes=DEFAULT_LIMITS.max_file_bytes + 1024 * 1024)
        registrations = limiter.return_value.register.call_args_list
        self.assertEqual(len(registrations), 2)
        self.assertIs(registrations[0].args[0], gui.handles['Load GLB'])
        self.assertEqual(registrations[0].kwargs['max_bytes'], DEFAULT_LIMITS.max_file_bytes)
        self.assertIs(registrations[1].args[0], gui.handles['Load rig mapping'])
        self.assertEqual(registrations[1].kwargs['max_bytes'], 1024 * 1024)

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        positions, rotations = neutral_source_pose()
        self.session = MotionSession(None, np.tile(positions, (4, 1, 1)), np.tile(rotations, (4, 1, 1, 1)))
        self.clients = {}
        self.server = SimpleNamespace(get_clients=lambda: self.clients)
        with patch('character_controls.GlbCharacterRenderer', BrowserBridge):
            self.controls = CharacterControls(self.server, self.session, None, Path(self.directory.name))
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
        self.controls.open_static_preview(1)
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
        self.controls.open_static_preview(1)
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
        self.controls.open_static_preview(1)
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
        self.controls.open_static_preview(1)
        self.bridge.respond()
        self.controls.tick((0, 0))
        expected = self.controls.active_entry.asset.bounds.mean(axis=0)
        expected[1] = 0.
        np.testing.assert_allclose(self.controls.actor_root(), expected)

    def test_static_selection_requires_consent_and_preserves_playing_actor(self):
        self.controls.select(self.rigged, 1)
        self.bridge.respond()
        self.controls.tick((0, 0))
        self.session.play()
        loads = len(self.bridge.loads)
        self.controls.select(self.static, 1)
        self.controls.tick((0, 0))
        self.assertEqual(len(self.bridge.loads), loads)
        self.assertEqual(self.controls.active_id, self.rigged)
        self.assertEqual(self.controls.mapping_asset_id, self.static)
        self.assertEqual(self.controls.compatibility.status, 'static_preview')
        self.assertTrue(self.session.character_motion_enabled)
        self.assertTrue(self.session.playing)
        self.controls.choose_another_file()
        self.assertEqual(self.controls.active_id, self.rigged)
        self.assertEqual(self.controls.mapping_asset_id, self.rigged)

    def test_unconsented_candidate_cancels_older_browser_activation(self):
        self.controls.select(self.rigged, 1)
        old = self.bridge.pending
        self.controls.select(self.static, 1)
        self.bridge.respond(old)
        self.controls.tick((0, 0))
        self.assertIsNone(self.controls.active_id)
        self.assertIsNone(self.bridge.pending)
        self.assertTrue(self.session.character_motion_enabled)
        self.controls.select(None, 1)
        self.controls.open_static_preview(1)
        self.assertIsNone(self.bridge.pending)

    def test_saved_static_startup_requires_consent_and_reconnect_does_not_consent(self):
        imported = import_glb(make_static_glb(), Path(self.directory.name), display_name='Saved static')
        with patch('character_controls.GlbCharacterRenderer', BrowserBridge):
            controls = CharacterControls(self.server, self.session, None, Path(self.directory.name))
        controls.set_initial_asset(imported.asset_id)
        controls.on_client_connect(SimpleNamespace(client_id=1))
        controls.on_client_connect(SimpleNamespace(client_id=2))
        self.assertIsNone(controls.active_id)
        self.assertIsNone(controls.renderer.pending)
        self.assertEqual(controls.compatibility.status, 'static_preview')
        controls.open_static_preview(2)
        controls.renderer.respond()
        controls.tick()
        self.assertEqual(controls.active_id, imported.asset_id)
        self.assertFalse(self.session.character_motion_enabled)

    def test_malformed_upload_never_enters_catalog_or_replaces_actor(self):
        self.controls.select(self.rigged, 1)
        self.bridge.respond()
        self.controls.tick()
        before = set(self.controls.entries)
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            self.controls.add_file(b'broken glb', 'bad.glb')
        self.assertEqual(self.controls.compatibility.status, 'unsupported')
        self.assertEqual(set(self.controls.entries), before)
        self.assertEqual(list(Path(self.directory.name).glob('*/asset.glb')), [])
        self.assertEqual(self.controls.active_id, self.rigged)

    def test_mapping_revalidation_preserves_static_actor_until_browser_commit(self):
        original = inspect_glb(make_humanoid_glb('mixamo'))
        profile = detect_rig_profile(original)
        doc = copy.deepcopy(original.document)
        for index, node in enumerate(doc['nodes']):
            node['name'] = f'custom_{index}'
        asset_id = self.controls.add_file(make_glb(doc, original.binary_chunk), 'Custom.glb')
        self.controls.select(asset_id, 1)
        self.assertEqual(self.controls.compatibility.status, 'mapping_required')
        self.assertIsNone(self.bridge.pending)
        self.controls.open_static_preview(1)
        self.bridge.respond()
        self.controls.tick()
        self.assertFalse(self.session.character_motion_enabled)
        self.assertIsNone(self.controls.active_entry.retargeter)
        with self.assertRaisesRegex(ValueError, 'Missing required'):
            self.controls.apply_mapping(asset_id, {'bones': {}}, 1)
        self.assertEqual(self.controls.active_id, asset_id)
        self.assertFalse(self.session.character_motion_enabled)
        self.assertEqual(self.controls.compatibility.status, 'mapping_required')
        self.controls.apply_mapping(asset_id, {'bones': dict(profile.bones)}, 1)
        self.assertFalse(self.session.character_motion_enabled)
        self.assertIsNone(self.controls.active_entry.retargeter)
        self.bridge.respond()
        self.controls.tick()
        self.assertTrue(self.session.character_motion_enabled)
        self.assertIsNotNone(self.controls.active_entry.retargeter)
        entry = self.controls.active_entry
        stand = entry.retargeter.retarget(*standing_reference(entry.retargeter.skeleton))
        self.assertAlmostEqual(posed_minimum_y(entry.asset, stand.world_matrices) + entry.ground_offset, 0)

    def test_mapping_is_not_a_fix_for_unskinned_geometry(self):
        with self.assertRaisesRegex(ValueError, 'cannot create bones'):
            self.controls.apply_mapping(self.static, {'bones': {}}, 1)
        self.assertIsNone(self.bridge.pending)

    def test_gui_only_offers_static_consent_and_another_file_for_unskinned_upload(self):
        gui = CharacterGui()
        with patch('character_controls.ScopedUploadLimits', autospec=True):
            self.controls.build_gui(gui)
        client = SimpleNamespace(client_id=1)
        upload = gui.handles['Load GLB']
        completed_file = SimpleNamespace(content=make_static_glb(), name='static.glb')
        upload.value = SimpleNamespace(content=b'newer bad upload', name='later.glb')
        upload.upload(upload_event(upload, client, completed_file))
        self.controls.tick()
        selected = self.controls.entries[self.controls.mapping_asset_id].asset
        self.assertEqual(selected.display_name, completed_file.name)
        self.assertEqual(selected.glb_bytes, completed_file.content)
        self.assertTrue(gui.handles['Open static preview'].visible)
        self.assertTrue(gui.handles['Choose another file'].visible)
        self.assertFalse(gui.handles['Load rig mapping'].visible)
        self.assertTrue(gui.handles['Load rig mapping'].disabled)
        self.assertIsNone(self.bridge.pending)
        self.assertTrue(self.session.character_motion_enabled)
        gui.handles['Open static preview'].click(SimpleNamespace(client=client))
        self.bridge.respond()
        self.controls.tick()
        self.assertFalse(self.session.character_motion_enabled)
        self.assertFalse(gui.handles['Open static preview'].visible)
        self.assertIn('motion generation disabled', self.controls.status)

    def test_gui_new_invalid_upload_cancels_previous_pending_and_shows_only_retry(self):
        gui = CharacterGui()
        with patch('character_controls.ScopedUploadLimits', autospec=True):
            self.controls.build_gui(gui)
        self.controls.select(self.rigged, 1)
        stale = self.bridge.pending
        upload = gui.handles['Load GLB']
        upload.value = SimpleNamespace(content=b'bad', name='bad.glb')
        client = SimpleNamespace(client_id=1)
        upload.upload(upload_event(upload, client, upload.value))
        self.bridge.respond(stale)
        self.controls.tick()
        self.assertIsNone(self.controls.active_id)
        self.assertEqual(self.controls.compatibility.status, 'unsupported')
        self.assertTrue(gui.handles['Choose another file'].visible)
        self.assertFalse(gui.handles['Open static preview'].visible)
        self.assertFalse(gui.handles['Load rig mapping'].visible)
        gui.handles['Choose another file'].click(SimpleNamespace(client=client))
        self.controls.tick()
        self.assertIsNone(self.controls.compatibility)
        self.assertTrue(self.session.character_motion_enabled)

    def test_cancelled_upload_cannot_publish_a_late_failure_report(self):
        from character_compatibility import unsupported_character
        gui = CharacterGui()
        with patch('character_controls.ScopedUploadLimits', autospec=True):
            self.controls.build_gui(gui)
        upload = gui.handles['Load GLB']
        upload.value = SimpleNamespace(content=b'bad', name='bad.glb')

        def inspect_after_cancellation(*args, **kwargs):
            self.controls.choose_another_file()
            return None, unsupported_character('old failure')

        with patch('character_controls.inspect_character', side_effect=inspect_after_cancellation):
            client = SimpleNamespace(client_id=1)
            upload.upload(upload_event(upload, client, upload.value))
        self.assertIsNone(self.controls.compatibility)
        self.assertNotIn('old failure', self.controls.status)

    def test_invalid_upload_hides_mapping_action_for_preserved_active_rig(self):
        gui = CharacterGui()
        with patch('character_controls.ScopedUploadLimits', autospec=True):
            self.controls.build_gui(gui)
        self.controls.select(self.rigged, 1)
        self.bridge.respond()
        self.controls.tick()
        self.assertTrue(gui.handles['Load rig mapping'].visible)
        upload = gui.handles['Load GLB']
        upload.value = SimpleNamespace(content=b'bad', name='bad.glb')
        client = SimpleNamespace(client_id=1)
        upload.upload(upload_event(upload, client, upload.value))
        self.controls.tick()
        self.assertEqual(self.controls.active_id, self.rigged)
        self.assertEqual(self.controls.compatibility.status, 'unsupported')
        self.assertTrue(self.session.character_motion_enabled)
        self.assertTrue(gui.handles['Choose another file'].visible)
        self.assertFalse(gui.handles['Load rig mapping'].visible)
        self.assertTrue(gui.handles['Load rig mapping'].disabled)
        gui.handles['Choose another file'].click(SimpleNamespace(client=client))
        self.controls.tick()
        self.assertTrue(gui.handles['Load rig mapping'].visible)

    def test_waist_origin_static_offset_reaches_renderer_root_and_camera(self):
        doc, binary = base_document_and_binary()
        doc['nodes'][0]['translation'] = [0, -.555, 0]
        asset_id = self.controls.add_file(make_glb(doc, binary), 'Waist origin.glb')
        self.controls.select(asset_id, 1)
        self.controls.open_static_preview(1)
        self.assertAlmostEqual(self.bridge.loads[-1][1]['ground_offset'], .555)
        self.bridge.respond()
        self.controls.tick()
        self.assertAlmostEqual(self.controls.actor_root()[1], .555)
        camera = SimpleNamespace()
        self.controls.frame_character(SimpleNamespace(camera=camera))
        self.assertAlmostEqual(camera.look_at[1], .2, places=6)
        self.assertEqual(self.controls.active_entry.asset.glb_bytes, make_glb(doc, binary))

    def test_rig_carrier_offset_is_fixed_and_vertical_motion_survives(self):
        self.controls.select(self.rigged, 1)
        self.bridge.respond()
        self.controls.tick((0, 0))
        before = self.controls.actor_root()
        offset = self.controls.active_entry.ground_offset
        self.session.positions[1, :, 1] += .75
        self.session.frame = 1
        self.controls.tick((0, 1))
        after = self.controls.actor_root()
        self.assertAlmostEqual(after[1] - before[1], .75 * self.controls.active_entry.retargeter.root_scale)
        self.assertEqual(self.controls.active_entry.ground_offset, offset)
        self.assertEqual(len(self.bridge.loads), 1)
        camera = SimpleNamespace()
        self.controls.frame_character(SimpleNamespace(camera=camera))
        expected = self.controls.active_entry.asset.bounds.mean(axis=0)
        expected += after - self.controls.active_entry.retargeter.retarget(*neutral_source_pose()).root_position
        np.testing.assert_allclose(camera.look_at, expected)

    def test_reconnecting_view_frames_grounded_static_geometry_not_source_origin(self):
        doc, binary = base_document_and_binary()
        doc['nodes'][0]['translation'] = [0, -100., 0]
        asset_id = self.controls.add_file(make_glb(doc, binary), 'Distant origin.glb')
        self.controls.select(asset_id, 1)
        self.controls.open_static_preview(1)
        self.bridge.respond()
        self.controls.tick()
        client = SimpleNamespace(client_id=2, camera=SimpleNamespace())
        self.controls.on_client_connect(client)
        self.assertAlmostEqual(client.camera.look_at[1], .2, places=6)
        self.assertLess(np.linalg.norm(client.camera.position), 5.)
        self.assertEqual(self.controls.active_id, asset_id)
        self.assertEqual(len(self.bridge.loads), 1)

    def test_initial_rig_load_frames_only_initiator_after_committed_pose(self):
        self.clients.update({index: SimpleNamespace(client_id=index, camera=SimpleNamespace(
            position=np.array([20., 20., 20.]), look_at=np.array([20., 20., 20.]))) for index in (1, 2)})
        self.session.positions[:, :, 0] += 10.
        self.controls.set_initial_asset(self.rigged)
        self.controls.on_client_connect(self.clients[1])
        np.testing.assert_array_equal(self.clients[1].camera.look_at, [20., 20., 20.])
        self.bridge.respond()
        self.controls.tick()
        self.assertGreater(self.clients[1].camera.look_at[0], 5.)
        self.assertLess(self.clients[1].camera.look_at[1], 5.)
        np.testing.assert_array_equal(self.clients[2].camera.look_at, [20., 20., 20.])
        expected = SimpleNamespace(camera=SimpleNamespace())
        self.controls.frame_character(expected)
        np.testing.assert_allclose(self.clients[1].camera.look_at, expected.camera.look_at)

    def test_failed_load_and_static_candidate_do_not_reframe_but_commit_does(self):
        client = SimpleNamespace(client_id=1, camera=SimpleNamespace(look_at=np.array([20., 20., 20.])))
        self.clients[1] = client
        self.controls.select(self.rigged, 1)
        self.bridge.respond(status='error')
        self.controls.tick()
        np.testing.assert_array_equal(client.camera.look_at, [20., 20., 20.])
        doc, binary = base_document_and_binary()
        doc['nodes'][0]['translation'] = [0, -100., 0]
        asset_id = self.controls.add_file(make_glb(doc, binary), 'Distant origin.glb')
        self.controls.select(asset_id, 1)
        self.controls.tick()
        np.testing.assert_array_equal(client.camera.look_at, [20., 20., 20.])
        self.controls.open_static_preview(1)
        self.bridge.respond()
        self.controls.tick()
        self.assertAlmostEqual(client.camera.look_at[1], .2, places=6)

    def test_delayed_mapping_callback_cannot_write_to_new_target_or_returned_target(self):
        gui = CharacterGui()
        with patch('character_controls.ScopedUploadLimits', autospec=True):
            self.controls.build_gui(gui)
        root = Path(self.directory.name)
        import_glb(make_humanoid_glb('mixamo'), root)
        other = self.controls.add_file(make_humanoid_glb('g1'), 'Other.glb')
        self.controls.select(self.rigged, 1)
        self.controls.tick()
        original_handle = gui.handles['Load rig mapping']
        profile = detect_rig_profile(self.controls.entries[self.rigged].asset)
        payload = json.dumps({'bones': dict(profile.bones), 'root_scale': 1.7}).encode()
        original_handle.value = SimpleNamespace(content=payload, name='mapping.json')
        client = SimpleNamespace(client_id=1)
        completed_file = SimpleNamespace(content=payload, name='mapping.json')
        delayed_event = upload_event(original_handle, client, completed_file)
        delayed_callback = original_handle.upload
        self.controls.select(other, 1)
        self.controls.tick()
        self.assertIsNot(gui.handles['Load rig mapping'], original_handle)
        delayed_callback(delayed_event)
        self.assertFalse((root / other / 'mapping.json').exists())
        self.assertFalse((root / self.rigged / 'mapping.json').exists())
        self.assertEqual(self.controls.mapping_asset_id, other)
        self.controls.select(self.rigged, 1)
        self.controls.tick()
        delayed_callback(delayed_event)
        self.assertFalse((root / self.rigged / 'mapping.json').exists())
        current_handle = gui.handles['Load rig mapping']
        current_handle.value = SimpleNamespace(content=b'newer invalid JSON', name='later.json')
        current_handle.upload(upload_event(current_handle, client, completed_file))
        self.assertEqual(json.loads((root / self.rigged / 'mapping.json').read_text())['root_scale'], 1.7)
        self.assertFalse((root / other / 'mapping.json').exists())

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
                controls = CharacterControls(self.server, self.session, None, root)
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
