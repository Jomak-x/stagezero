"""Focused continuation regressions for the local grounded review server."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

import grounded_server


def _motion(frames: int, offset: float = 0.0):
    positions = np.zeros((1, frames, 27, 3), dtype=np.float32)
    positions[0, :, :, 0] = np.arange(frames, dtype=np.float32)[:, None] * 0.01 + offset
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (1, frames, 27, 3, 3)).copy()
    return positions, rotations


class _Rig:
    def __init__(self, *, fail=False):
        self.fail = fail

    def clip_payload(self, positions, rotations, **options):
        if self.fail:
            raise RuntimeError('retarget failed')
        return {'fitted_positions': positions.tolist(), 'fitted_rotations': rotations.tolist(), 'character_provenance': {'floor_offsets': options.get('floor_offsets') or [0.]}}


class GroundedContinuationTests(TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.token = self.root / 'token'
        self.token.write_text('local-token')
        self.original_p, self.original_r = _motion(48)
        self.native = np.arange(48 * 330, dtype=np.float32).reshape(1, 48, 330)
        self.source_dir = self.root / 'source-clips'
        self.source_dir.mkdir()
        self.source = self.source_dir / 'input.npz'
        np.savez_compressed(self.source, positions=self.original_p, rotations=self.original_r,
                            native_features=self.native,
                            metadata=json.dumps({'actor_ids': ['human'], 'label': 'Input'}))
        self.original_sha = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.root_patch = patch.object(grounded_server, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        args = SimpleNamespace(clips=[str(self.source_dir)], token_file=self.token,
                               realtime_url='http://127.0.0.1:8769', characters=['unused'])
        self.review = grounded_server.Review(args)
        self.source_id = next(iter(self.review.clips()))

    def _await_job(self, job_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = self.review.jobs[job_id]
            if job['status'] != 'running':
                return job
            time.sleep(.005)
        self.fail('Continuation worker did not finish')

    def _client(self, *, fail=False):
        produced_p, produced_r = _motion(80, 10)
        calls = []

        class Client:
            def __init__(self, url, token, *, job_timeout):
                assert token == 'local-token'
                assert job_timeout == 60

            def wait(self, body):
                calls.append(body)
                if fail:
                    raise RuntimeError('model failed')
                return [SimpleNamespace(positions=produced_p[:, :40], rotations=produced_r[:, :40]),
                        SimpleNamespace(positions=produced_p[:, 40:], rotations=produced_r[:, 40:])]

        return Client, calls, produced_p, produced_r

    def test_continuation_preserves_every_source_frame_and_uses_last_native_history(self):
        client, calls, generated_p, generated_r = self._client()
        with patch('realtime_client.RealtimeClient', client), patch.object(self.review, 'rig', return_value=_Rig()):
            pending = self.review.continue_motion({'clip_id': self.source_id, 'prompt': 'Walk together'})
            job = self._await_job(pending['id'])
        self.assertEqual(job['status'], 'complete', job)
        self.assertEqual(job['preserved_prefix_frames'], 48)
        np.testing.assert_array_equal(calls[0]['history']['native_features'], self.native[:, -40:])
        self.assertEqual(calls[0]['actor_ids'], ['human'])
        target = self.review.generated[job['clip_id']][0]
        with np.load(target, allow_pickle=False) as data:
            np.testing.assert_array_equal(data['positions'][:, :48], self.original_p)
            np.testing.assert_array_equal(data['rotations'][:, :48], self.original_r)
            np.testing.assert_array_equal(data['positions'][:, 48:], generated_p)
            np.testing.assert_array_equal(data['rotations'][:, 48:], generated_r)
            metadata = json.loads(data['metadata'].item())
        self.assertEqual(metadata['seed'], calls[0]['seed'])
        self.assertEqual(metadata['frames'], 128)
        self.assertEqual(metadata['request_id'], calls[0]['request_id'])
        self.assertEqual(metadata['stage_kind'], 'continuation')
        self.assertEqual(metadata['parent_file_sha256'], self.original_sha)
        self.assertEqual(metadata['preserved_prefix_frames'], 48)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.original_sha)

    def test_live_direction_only_replaces_unplayed_suffix(self):
        client, calls, generated_p, generated_r = self._client()
        with patch('realtime_client.RealtimeClient', client), patch.object(self.review, 'rig', return_value=_Rig()):
            source_fit = self.review.clip(self.source_id)
            pending = self.review.continue_motion({'clip_id': self.source_id, 'prompt': 'Turn left', 'commit_frame': 40})
            job = self._await_job(pending['id'])
            self.assertEqual(job['status'], 'complete', job)
            result = self.review.clip(job['clip_id'])
        np.testing.assert_array_equal(calls[0]['history']['native_features'], self.native[:, :40])
        np.testing.assert_array_equal(np.asarray(result['positions'])[:, :40], self.original_p[:, :40])
        np.testing.assert_array_equal(np.asarray(result['fitted_positions'])[:, :40], np.asarray(source_fit['fitted_positions'])[:, :40])
        np.testing.assert_array_equal(np.asarray(result['fitted_rotations'])[:, :40], np.asarray(source_fit['fitted_rotations'])[:, :40])
        self.assertEqual(result['metadata']['replaced_unplayed_frames'], 8)
        self.assertEqual(result['metadata']['retarget_floor_offsets'], source_fit['metadata']['retarget_floor_offsets'])
        self.assertEqual(len(result['positions'][0]), 120)

    def test_failed_retarget_does_not_publish_result_or_modify_source(self):
        client, _, _, _ = self._client()
        with patch('realtime_client.RealtimeClient', client), patch.object(self.review, 'rig', return_value=_Rig(fail=True)):
            pending = self.review.continue_motion({'clip_id': self.source_id, 'prompt': 'Walk together'})
            job = self._await_job(pending['id'])
        self.assertEqual(job['status'], 'failed', job)
        self.assertIn('retarget failed', job['error'])
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.original_sha)
        self.assertEqual(set(self.review.clips()), {self.source_id},
                         'A failed retarget must not be advertised as a playable clip')

    def test_failed_model_request_leaves_source_and_generated_catalog_unchanged(self):
        client, _, _, _ = self._client(fail=True)
        with patch('realtime_client.RealtimeClient', client):
            pending = self.review.continue_motion({'clip_id': self.source_id, 'prompt': 'Walk together'})
            job = self._await_job(pending['id'])
        self.assertEqual(job['status'], 'failed', job)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.original_sha)
        self.assertEqual(set(self.review.clips()), {self.source_id})

    def test_pair_continuation_is_blocked_before_generation(self):
        np.savez_compressed(self.source, positions=np.repeat(self.original_p, 2, axis=0),
                            rotations=np.repeat(self.original_r, 2, axis=0))
        with self.assertRaisesRegex(ValueError, 'replay-only'):
            self.review.continue_motion({'clip_id': self.source_id, 'prompt': 'Dance together'})
        self.assertEqual(self.review.jobs, {})

    def test_soma_import_continuation_is_blocked_before_generation(self):
        np.savez_compressed(self.source, positions=self.original_p, rotations=self.original_r,
                            metadata=json.dumps({'soma_conversion': {'adapter': 'SOMA77'}}))
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            self.review.continue_motion({'clip_id': self.source_id, 'prompt': 'Guard'})
        self.assertEqual(self.review.jobs, {})

    def test_explicit_contact_retarget_mode_survives_continuation(self):
        options={'preserve_feet': True, 'preserve_wrists': True, 'wrist_target_space': 'native_world'}
        np.savez_compressed(self.source, positions=self.original_p, rotations=self.original_r,
                            native_features=self.native, metadata=json.dumps({'retarget_options': options}))
        client, _, _, _ = self._client()
        with patch('realtime_client.RealtimeClient', client), patch.object(self.review, 'rig', return_value=_Rig()):
            pending=self.review.continue_motion({'clip_id':self.source_id,'prompt':'Stand'})
            job=self._await_job(pending['id'])
            self.assertEqual(job['status'],'complete',job)
            result=self.review.clip(job['clip_id'])
        self.assertEqual(result['metadata']['retarget_options'], options)

    def test_live_tail_and_reused_fit_match_full_result(self):
        client, _, _, _=self._client()
        rig=_Rig()
        with patch('realtime_client.RealtimeClient',client), patch.object(self.review,'rig',return_value=rig):
            source=self.review.clip(self.source_id)
            with patch.object(rig,'clip_payload', wraps=rig.clip_payload) as fit:
                job=self._await_job(self.review.continue_motion({'clip_id':self.source_id,'prompt':'Guard','commit_frame':40})['id'])
                self.assertEqual(job['status'],'complete',job)
                self.assertEqual(fit.call_count,1)
                self.assertEqual(fit.call_args.args[0].shape[1],80)
            full=self.review.clip(job['clip_id'])
            tail=self.review.clip_tail(job['clip_id'],40)
            self.assertEqual(tail['start_frame'],40)
            self.assertEqual(tail['source_clip_id'],self.source_id)
            self.assertEqual(tail['frames'],120)
            for field in ('positions','rotations','fitted_positions','fitted_rotations'):
                combined=np.concatenate((np.asarray(source[field])[:,:40],np.asarray(tail[field])),axis=1)
                np.testing.assert_array_equal(combined,full[field])
            with self.assertRaises(ValueError):self.review.clip_tail(job['clip_id'],120)
