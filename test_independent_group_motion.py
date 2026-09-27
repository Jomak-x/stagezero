"""Focused stdlib contract checks for the opt-in group motion probe."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cast_performance import CastPerformance, decode_project, encode_project
from independent_group_motion import _geometry, _overlay_activity, generate_independent_tracks, overlay_third_track
from native_pair_transition import CORE27_NAMES, core27_to_native22
from paired_scene import EMPTY_SCENE


def _core_pose(x):
    points = np.empty((27, 3), dtype=np.float64)
    for i, name in enumerate(CORE27_NAMES):
        side = -1 if name.startswith('Left') else 1 if name.startswith('Right') else 0
        height = (.9 if name == 'Hips' else 1.2 if 'Spine' in name else
                  1.55 if 'Shoulder' in name or 'Neck' in name else
                  1.75 if 'Head' in name else .05 if 'Toe' in name or 'Foot' in name else
                  .45 if 'Leg' in name else 1.1)
        points[i] = (x + side * (.14 + i * .002), height + i * .0001, i * .002)
    return points


class FakeClient:
    def __init__(self, *, malformed=False):
        self.requests = []
        self.malformed = malformed

    def wait(self, request, cancelled):
        self.requests.append(request)
        aid = request['actor_ids'][0]
        x = request.get('initial_placements', {}).get(aid, {}).get('position_xz', [0, 0])[0]
        points = np.repeat(_core_pose(x)[None, None], 40, axis=1)
        native = np.full((1, 40, 330), float(request['seed']) / 1e6 + len(self.requests))
        if self.malformed:
            points[0, 0, 0, 0] = np.nan
        return [SimpleNamespace(actor_ids=(aid,), fps=20, frames=40,
                                positions=points, rotations=np.zeros((1, 40, 27, 3, 3)),
                                native_features=native)]


def _actors(n=2):
    return [{'id': f'a{i}', 'prompt': 'waves hello', 'x': float(i*3), 'z': 0., 'yaw': 0.}
            for i in range(n)]


def _saved_source():
    base = core27_to_native22(_core_pose(0))
    joints = np.repeat(base[None, None], 60, axis=0)
    joints = np.repeat(joints, 3, axis=1)
    joints[:, 0, :, 0] -= 3
    joints[:, 1, :, 0] += 3
    ids = ('a0', 'a1', 'a2')
    metadata = {'fps': 30, 'frames': 60,
                'segment_activity': [{'start_frame': 0, 'end_frame_exclusive': 60,
                                      'active_actor_ids': ['a0', 'a1'],
                                      'contact_actor_ids': ['a0', 'a1']}]}
    clip = CastPerformance(ids, joints, metadata=metadata)
    cast = [{'id': aid, 'name': aid, 'color': [1, 2, 3]} for aid in ids]
    return clip, encode_project(clip, cast, EMPTY_SCENE)


class IndependentGroupMotionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_native_history_is_private_and_every_horizon_is_archived(self):
        with patch('independent_group_motion._geometry', return_value={'checked': True}):
            client = FakeClient()
            result = generate_independent_tracks(client, EMPTY_SCENE, actors=_actors(), seconds=6,
                                                 seed=7, output_root=self.root)
        self.assertEqual(result['performance'].frames, 180)
        self.assertEqual(len(client.requests), 6)
        self.assertEqual([r['seed'] for r in client.requests], [7]*3 + [1016]*3)
        self.assertEqual([r['stage_kind'] for r in client.requests],
                         ['approach', 'continuation', 'continuation']*2)
        for index in (1, 2, 4, 5):
            expected = float(client.requests[index-1]['seed']) / 1e6 + index
            np.testing.assert_array_equal(client.requests[index]['history']['native_features'],
                                          np.full((1, 40, 330), expected))
        manifest = json.loads(result['manifest'].read_text())
        self.assertEqual(manifest['actor_seeds'], {'a0': 7, 'a1': 1016})
        self.assertEqual(len(manifest['sources']), 6)
        with np.load(manifest['sources'][0]['path'], allow_pickle=False) as archive:
            self.assertEqual(archive['positions'].shape, (1, 40, 27, 3))
            self.assertEqual(archive['native_features'].shape, (1, 40, 330))

    def test_bad_core_output_is_preserved_before_rejection(self):
        with self.assertRaisesRegex(ValueError, 'Core returned incompatible'):
            generate_independent_tracks(FakeClient(malformed=True), EMPTY_SCENE,
                                        actors=_actors(1), seconds=2, seed=1, output_root=self.root)
        folder = next(self.root.iterdir())
        manifest = json.loads((folder/'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'rejected')
        with np.load(manifest['sources'][0]['path'], allow_pickle=False) as archive:
            self.assertTrue(np.isnan(archive['positions'][0, 0, 0, 0]))

    def test_actor_cap_and_cancellation(self):
        with self.assertRaisesRegex(ValueError, '1–10'):
            generate_independent_tracks(FakeClient(), EMPTY_SCENE, actors=_actors(11),
                                        seconds=2, seed=1, output_root=self.root)
        client = FakeClient()
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            generate_independent_tracks(client, EMPTY_SCENE, actors=_actors(1),
                                        seconds=6, seed=1, output_root=self.root,
                                        cancelled=lambda: len(client.requests) >= 1)
        folder = next(self.root.iterdir())
        manifest = json.loads((folder/'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'cancelled')
        self.assertEqual(len(manifest['sources']), 1)

    def test_full_interval_overlay_preserves_pair_after_roundtrip(self):
        source, data = _saved_source()
        with patch('independent_group_motion._geometry', return_value={'checked': True}):
            result = overlay_third_track(FakeClient(), data, prompt='wave', start_frame=0,
                                         end_frame=60, seed=5, output_root=self.root)
        self.assertEqual(result['status'], 'accepted_by_geometry_gates')
        self.assertFalse(result['fallback'])
        saved, _, _, _ = decode_project(result['project_bytes'])
        self.assertEqual(saved.joints[:, :2].tobytes(), source.joints[:, :2].tobytes())
        self.assertEqual(saved.joints.dtype, source.joints.dtype)
        self.assertEqual((result['folder']/'original.cast.stagezero.npz').read_bytes(), data)
        self.assertIn('a2', saved.metadata['segment_activity'][0]['active_actor_ids'])

    def test_overlay_geometry_rejection_returns_exact_original(self):
        source, data = _saved_source()
        with patch('independent_group_motion._geometry', side_effect=ValueError('moving third intersects pair')):
            result = overlay_third_track(FakeClient(), data, prompt='wave', start_frame=0,
                                         end_frame=60, seed=5, output_root=self.root)
        self.assertTrue(result['fallback'])
        self.assertEqual(result['project_bytes'], data)
        self.assertTrue((result['folder']/'candidate.npz').exists())
        self.assertEqual(result['performance'].joints.tobytes(), source.joints.tobytes())

    def test_production_cast_rejects_ten_tracks(self):
        joints = np.zeros((60, 10, 22, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, '1–3'):
            CastPerformance([f'a{i}' for i in range(10)], joints)

    def test_moving_root_rejection_keeps_measurements(self):
        base = core27_to_native22(_core_pose(0))
        joints = np.repeat(base[None, None], 60, axis=0)
        joints = np.repeat(joints, 2, axis=1)
        joints[:, 1, :, 0] += 3
        joints[30:, 1, :, 0] -= 3
        activities = [{'start_frame': 0, 'end_frame_exclusive': 60,
                       'active_actor_ids': ['a0', 'a1'], 'contact_actor_ids': []}]
        with self.assertRaisesRegex(ValueError, 'body-proxy overlap') as caught:
            _geometry(joints, EMPTY_SCENE, ('a0', 'a1'), activities)
        self.assertIn('body_clearance', caught.exception.geometry_diagnostics)

    def test_overlay_activity_splits_source_pair_schedule(self):
        metadata = {'segments': [{'label': 'pair', 'source': 'intergen', 'kind': 'paired_action',
                                  'start_frame': 0, 'end_frame_exclusive': 120, 'frames': 120}],
                    'segment_activity': [{'start_frame': 0, 'end_frame_exclusive': 120,
                                          'active_actor_ids': ['a0', 'a1'], 'held_actor_ids': ['a2'],
                                          'contact_actor_ids': ['a0', 'a1'],
                                          'observer_control_spans': [{'actor_id': 'a0',
                                              'start_frame': 0, 'end_frame_exclusive': 120,
                                              'source': 'existing'}]}]}
        _overlay_activity(metadata, 30, 90, 'a2')
        self.assertEqual([(s['start_frame'], s['end_frame_exclusive']) for s in metadata['segments']],
                         [(0, 30), (30, 90), (90, 120)])
        self.assertEqual([s['frames'] for s in metadata['segments']], [30, 60, 30])
        for index, activity in enumerate(metadata['segment_activity']):
            self.assertEqual(activity['contact_actor_ids'], ['a0', 'a1'])
            if index == 1:
                self.assertIn('a2', activity['active_actor_ids'])
                self.assertNotIn('a2', activity['held_actor_ids'])
                self.assertEqual(len(activity['observer_control_spans']), 2)
            else:
                self.assertNotIn('a2', activity['active_actor_ids'])
                self.assertIn('a2', activity['held_actor_ids'])
                self.assertEqual(len(activity['observer_control_spans']), 1)


if __name__ == '__main__':
    unittest.main()
