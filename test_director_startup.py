"""Studio startup configuration and bounded local GLB selection."""

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from character_controls import CharacterControls
from director_viewer import MAX_STARTUP_GLB_BYTES, build_parser, load_startup_glb
from live_motion import MotionSession
from preview import ROOT
from retargeting import neutral_source_pose
from tests.glb_fixtures import make_humanoid_glb
from tests.test_character_controls import BrowserBridge


class StartupControls:
    def __init__(self):
        self.received = None
        self.initial_asset = None

    def add_file(self, data, name):
        self.received = (data, name)
        return 'imported-asset-id'

    def set_initial_asset(self, asset_id):
        self.initial_asset = asset_id


class DirectorStartupTests(unittest.TestCase):
    def test_default_startup_keeps_live_backend_and_recorded_source(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.recording, ROOT / 'assets/recorded_g1.csv')
        self.assertEqual(args.token_path, ROOT / '.runtime/api-token')
        self.assertEqual(args.backend_url, 'http://127.0.0.1:8765')
        self.assertEqual(args.characters, ROOT / '.runtime/characters')
        self.assertEqual(args.environment, 'studio')
        self.assertIsNone(args.glb)

    def test_private_runtime_paths_glb_and_lighting_can_be_selected(self):
        args = build_parser().parse_args([
            '--recording', '/private/motion.csv', '--token-path', '/private/token',
            '--backend-url', 'http://127.0.0.1:2342', '--glb', '/private/actor.glb',
            '--characters', '/private/catalog', '--environment', 'warehouse',
        ])
        self.assertEqual(args.recording, Path('/private/motion.csv'))
        self.assertEqual(args.token_path, Path('/private/token'))
        self.assertEqual(args.backend_url, 'http://127.0.0.1:2342')
        self.assertEqual(args.glb, Path('/private/actor.glb'))
        self.assertEqual(args.characters, Path('/private/catalog'))
        self.assertEqual(args.environment, 'warehouse')
        self.assertEqual(build_parser().parse_args(['--environment', 'none']).environment, 'none')

    def test_local_glb_is_prepared_for_first_browser_connection(self):
        with TemporaryDirectory() as directory:
            glb = Path(directory) / 'actor.glb'
            glb.write_bytes(b'glb-data')
            controls = StartupControls()
            load_startup_glb(build_parser(), controls, glb)
            self.assertEqual(controls.received, (b'glb-data', 'actor.glb'))
            self.assertEqual(controls.initial_asset, 'imported-asset-id')

    def test_real_rigged_glb_activates_on_first_browser_connection(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            glb = root / 'performer.glb'
            glb.write_bytes(make_humanoid_glb('mixamo'))
            positions, rotations = neutral_source_pose()
            session = MotionSession(None, np.tile(positions, (4, 1, 1)),
                                    np.tile(rotations, (4, 1, 1, 1)))
            with patch('character_controls.GlbCharacterRenderer', BrowserBridge):
                controls = CharacterControls(SimpleNamespace(), session, None, root / 'catalog')
            load_startup_glb(build_parser(), controls, glb)
            self.assertIsNone(controls.active_id)
            client = SimpleNamespace(client_id=7)
            controls.on_client_connect(client)
            self.assertEqual(controls.renderer.pending[0], client.client_id)
            controls.renderer.respond()
            controls.tick((0, 0))
            self.assertEqual(controls.active_id, controls.renderer.active[1])
            self.assertTrue(session.character_motion_enabled)

    def test_oversized_glb_is_rejected_before_import(self):
        with TemporaryDirectory() as directory:
            glb = Path(directory) / 'oversized.glb'
            with glb.open('wb') as target:
                target.seek(MAX_STARTUP_GLB_BYTES)
                target.write(b'x')
            controls = StartupControls()
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
                load_startup_glb(build_parser(), controls, glb)
            self.assertEqual(raised.exception.code, 2)
            self.assertIsNone(controls.received)
            self.assertIsNone(controls.initial_asset)

    def test_missing_glb_is_reported_as_a_startup_argument_error(self):
        with TemporaryDirectory() as directory:
            controls = StartupControls()
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
                load_startup_glb(build_parser(), controls, Path(directory) / 'missing.glb')
            self.assertEqual(raised.exception.code, 2)
            self.assertIsNone(controls.initial_asset)


if __name__ == '__main__':
    unittest.main()
