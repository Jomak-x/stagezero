"""Studio context wrapper keeps native source exact and preserves failed work."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from native_pair_clip import load_source
from native_pair_context import build_studio_context
from native_pair_transition import shared_place_pair


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'review/two-character/native-recovery/originals/handshake_seed42.npz'
SCENE = {'version': 2, 'name': 'Empty stage', 'objects': [],
         'effects': [], 'lighting': 'neutral'}
PLACEMENT = {'x': 3., 'z': -2., 'yaw_degrees': 75.}


def _result(pair, yaw, translation):
    prefix = pair.joints[:4].astype(np.float64).copy()
    prefix[..., 0] -= .5
    suffix = pair.joints[-4:].astype(np.float64).copy()
    suffix[..., 2] += .5
    local = np.concatenate((prefix, pair.joints, suffix))
    world = shared_place_pair(local, yaw=yaw, translation=translation)
    spans = ((0, 4, 'ardy_core'), (4, 4 + pair.frames, 'intergen'),
             (4 + pair.frames, len(local), 'ardy_core'))
    segments = [{'label': source, 'source': source, 'kind': 'paired_action',
                 'start_frame': first, 'end_frame_exclusive': last,
                 'frames': last-first} for first, last, source in spans]
    return local, {'joints': world, 'metadata': {
        'model': 'ARDY Core + InterGen', 'fps': 30, 'frames': len(local),
        'segments': segments, 'source_pair_preserved_exactly_after_shared_placement': True}}


class NativePairContextTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.pair = load_source(SOURCE)
        self.client = object()

    def _archive_dir(self):
        paths = list(Path(self.folder.name).iterdir())
        self.assertEqual(len(paths), 1)
        return paths[0]

    def test_world_planning_inverse_once_and_exact_native_segment(self):
        seen = {}

        def fake_build(pair, client, scene, *, yaw, translation, cancelled, on_core_chunk):
            self.assertIs(pair, self.pair)
            self.assertIs(client, self.client)
            self.assertEqual(scene, SCENE)
            self.assertAlmostEqual(yaw, np.deg2rad(75.))
            np.testing.assert_array_equal(translation, [3., 0., -2.])
            self.assertFalse(cancelled())
            expected, result = _result(pair, yaw, translation)
            seen['expected'], seen['world'] = expected, result['joints'].copy()
            return result

        def fake_geometry(world, scene):
            np.testing.assert_array_equal(world, seen['world'])
            self.assertEqual(scene, SCENE)
            return {'checked_world': True}

        with patch('native_pair_context.build_ardy_pair_context', side_effect=fake_build), \
             patch('native_pair_geometry.check_native_pair_geometry', side_effect=fake_geometry):
            result = build_studio_context(self.pair, self.client, SCENE, PLACEMENT, self.folder.name)
        np.testing.assert_allclose(result.joints[:4], seen['expected'][:4], atol=1e-12)
        np.testing.assert_allclose(result.joints[-4:], seen['expected'][-4:], atol=1e-12)
        np.testing.assert_array_equal(result.joints[4:4+self.pair.frames], self.pair.joints)
        self.assertIsNone(result.features)
        archive_dir = self._archive_dir()
        with np.load(archive_dir/'source-pair.npz', allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['joints'], self.pair.joints)
            np.testing.assert_array_equal(archive['features'], self.pair.features)
        with np.load(archive_dir/'composed-review.npz', allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['joints'], result.joints)
        report = json.loads((archive_dir/'report.json').read_text())
        self.assertEqual(report['scene_geometry'], {'checked_world': True})
        self.assertFalse((archive_dir/'failure.json').exists())

    def test_geometry_rejection_retains_native_and_completed_core_archive(self):
        def fake_build(pair, client, scene, *, yaw, translation, cancelled, on_core_chunk):
            core = SimpleNamespace(positions=np.zeros((2, 40, 27, 3), np.float32),
                                   rotations=np.zeros((2, 40, 27, 3, 3), np.float32),
                                   native_features=np.ones((2, 40, 8), np.float32), fps=20)
            on_core_chunk('approach', 0, core, {'request_id': 'one'})
            return _result(pair, yaw, translation)[1]

        with patch('native_pair_context.build_ardy_pair_context', side_effect=fake_build), \
             patch('native_pair_geometry.check_native_pair_geometry', side_effect=ValueError('scene solid')):
            with self.assertRaisesRegex(ValueError, 'scene solid'):
                build_studio_context(self.pair, self.client, SCENE, PLACEMENT, self.folder.name)
        archive_dir = self._archive_dir()
        self.assertTrue((archive_dir/'source-pair.npz').is_file())
        with np.load(archive_dir/'source-pair.npz', allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['joints'], self.pair.joints)
            np.testing.assert_array_equal(archive['features'], self.pair.features)
        with np.load(archive_dir/'approach-0.core.npz', allow_pickle=False) as archive:
            self.assertEqual(archive['positions'].shape, (2, 40, 27, 3))
            self.assertEqual(archive['native_features'].shape, (2, 40, 8))
        self.assertFalse((archive_dir/'composed-review.npz').exists())
        failure = json.loads((archive_dir/'failure.json').read_text())
        self.assertTrue(failure['source_retained'])

    def test_partial_core_failure_preserves_completed_raw_window(self):
        def fake_build(pair, client, scene, *, yaw, translation, cancelled, on_core_chunk):
            core = SimpleNamespace(positions=np.ones((2, 40, 27, 3), np.float32),
                                   rotations=np.zeros((2, 40, 27, 3, 3), np.float32),
                                   native_features=np.ones((2, 40, 8), np.float32), fps=20)
            on_core_chunk('approach', 0, core, {'request_id': 'completed-window'})
            raise RuntimeError('second Core window failed')

        with patch('native_pair_context.build_ardy_pair_context', side_effect=fake_build):
            with self.assertRaisesRegex(RuntimeError, 'second Core window failed'):
                build_studio_context(self.pair, self.client, SCENE, PLACEMENT, self.folder.name)
        archive_dir = self._archive_dir()
        self.assertTrue((archive_dir/'source-pair.npz').is_file())
        self.assertTrue((archive_dir/'approach-0.core.npz').is_file())
        self.assertFalse((archive_dir/'composed-review.npz').exists())
        self.assertTrue(json.loads((archive_dir/'failure.json').read_text())['source_retained'])


if __name__ == '__main__':
    unittest.main()
