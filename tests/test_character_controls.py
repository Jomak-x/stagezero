"""Character swaps are committed by the browser, independently of motion state."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np

from character_controls import CharacterControls
from character_assets import inspect_glb
from live_motion import MotionSession
from retargeting import neutral_source_pose
from tests.glb_fixtures import make_humanoid_glb, make_static_glb


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


if __name__ == '__main__':
    unittest.main()
