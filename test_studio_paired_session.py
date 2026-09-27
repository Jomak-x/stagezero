"""Offline paired research session lifecycle, archive, and request contract checks."""
import io
import json
import threading
import time
import unittest
from unittest import mock
from zipfile import ZipFile

import numpy as np

from paired_scene import (ACTOR_IDS, EMPTY_SCENE, LICENSE, decode_project, encode_project,
                          join_chunks, request_body, validate_request)
from realtime_backend import validate_job
from realtime_clip import CanonicalClip
from studio_core_session import CoreStudioSession
from scene_objects import make_object
from studio_paired_session import PairedStudioSession


def chunk(body, index, *, source='intergen', native=False, value=0.):
    metadata = {'request_id': body['request_id'], 'stage_kind': 'paired',
                'chunk_index': index, 'start_frame': index * 40, 'frames': 40,
                'pair_sequence_id': body['pair_sequence_id'], 'source_start_frame': index * 40,
                'source_total_frames': body['frames'], 'seed': body['seed']}
    positions = np.zeros((2, 40, 27, 3), dtype=np.float32)
    positions[1, :, :, 0] = 1.
    positions[..., 2] = value
    return CanonicalClip(positions, np.broadcast_to(np.eye(3), (2, 40, 27, 3, 3)),
                         20, ACTOR_IDS, source, metadata,
                         np.zeros((2, 40, 330)) if native else None)


def complete_clip():
    body = request_body('A friendly paired gesture', 7301, 120)
    return join_chunks([chunk(body, i) for i in range(3)], body)


def wait_for(predicate, timeout=3.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError('Condition did not finish')


class FakeClient:
    def __init__(self):
        self.calls = []
        self.health_calls = 0
        self.enabled = True
        self.started = threading.Event()
        self.partial = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.fail = False
        self.ignore_cancel = False
        self.bad_source = False
        self.short = False
        self.torso_overlap = False
        self.active = 0
        self.max_active = 0

    def health(self):
        self.health_calls += 1
        return {'pair_research_enabled': self.enabled}

    def status(self, request_id):
        return {'request_id': request_id, 'metadata': {'model': 'InterGen', 'license': LICENSE}}

    def wait(self, body, on_chunk, *, cancelled):
        validate_job(body, pair_enabled=True)
        self.calls.append(body)
        self.active += 1
        self.max_active = max(self.active, self.max_active)
        self.started.set()
        clips = []
        try:
            for index in range(body['frames'] // 40):
                if index == 1:
                    while not self.release.wait(.005):
                        if cancelled() and not self.ignore_cancel:
                            raise RuntimeError('cancelled')
                    if self.fail:
                        raise RuntimeError('simulated failure')
                    if self.short:
                        return clips
                clip = chunk(body, index, source='ardy_core' if self.bad_source else 'intergen',
                             value=.1 * len(self.calls))
                if self.torso_overlap and index == body['frames'] // 40 - 1:
                    positions = clip.positions.copy()
                    positions[1, -1, 4] = positions[0, -1, 4] + [.2, 0., 0.]
                    clip = CanonicalClip(positions, clip.rotations, 20, ACTOR_IDS, 'intergen', clip.metadata)
                clips.append(clip)
                on_chunk(clip)
                self.partial.set()
            return clips
        finally:
            self.active -= 1


class PairedSessionTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.session = PairedStudioSession(self.client)
        self.addCleanup(self.session.close)
        self.addCleanup(self.client.release.set)

    def generate(self, **kwargs):
        self.session.activate()
        return self.session.generate('Two people exchange a friendly gesture', 7301, **kwargs)

    def completed(self):
        self.generate()
        wait_for(lambda: not self.session.snapshot()['busy'])
        self.assertIsNone(self.session.snapshot()['failure'])
        self.session.pause()
        return self.session.timeline_clip()

    def test_activation_no_network_and_requires_explicit_research_generation(self):
        with self.assertRaisesRegex(RuntimeError, 'Activate'):
            self.session.generate('wave', 1)
        self.session.activate()
        self.session.tick()
        self.assertEqual(self.client.health_calls, 0)
        self.assertEqual(self.client.calls, [])
        self.assertIsNone(self.session.timeline_clip())
        with PairedStudioSession() as offline:
            offline.activate()
            with self.assertRaisesRegex(RuntimeError, 'not configured'):
                offline.generate('wave', 1)

    def test_full_pair_commits_once_without_native_history_or_core_state(self):
        with CoreStudioSession() as core:
            before = core.snapshot()
            clip = self.completed()
            self.assertEqual(core.snapshot(), before)
        state = self.session.snapshot()
        self.assertEqual(clip.frames, 120)
        self.assertEqual(clip.source, 'intergen')
        self.assertIsNone(clip.native_features)
        self.assertEqual(state['revision'], 1)
        self.assertTrue(state['research_only'])
        self.assertEqual(state['mode'], 'research')
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.client.max_active, 1)
        self.assertNotIn('history', self.client.calls[0])
        self.assertNotIn('root_targets', self.client.calls[0])
        self.assertEqual(state['segments'][0]['end'], 120)

    def test_partial_output_keeps_previous_clip_scene_and_revision_until_complete(self):
        old = self.completed()
        scene = self.session.scene_document
        self.client.release.clear(); self.client.partial.clear()
        pending_scene = {**EMPTY_SCENE, 'name': 'New backdrop'}
        self.generate(scene_document=pending_scene)
        self.assertTrue(self.client.partial.wait(1))
        self.assertTrue(self.session.snapshot()['pending'])
        self.assertEqual(self.session.snapshot()['pending_chunks'], 1)
        self.assertIs(self.session.timeline_clip(), old)
        self.assertEqual(self.session.scene_document, scene)
        self.assertEqual(self.session.revision, 1)
        pending_scene['name'] = 'Caller changed input after submission'
        self.client.release.set()
        wait_for(lambda: not self.session.snapshot()['busy'])
        self.assertEqual(self.session.scene_document['name'], 'New backdrop')
        self.assertEqual(self.session.revision, 2)
        self.assertIsNot(self.session.timeline_clip(), old)

    def test_failed_or_incomplete_generation_keeps_last_good_project(self):
        old = self.completed()
        scene = self.session.scene_document
        for kind in ('fail', 'short', 'bad_source'):
            setattr(self.client, kind, True)
            self.generate(scene_document={**EMPTY_SCENE, 'name': 'Uncommitted'})
            wait_for(lambda: not self.session.snapshot()['busy'])
            self.assertTrue(self.session.snapshot()['failure'])
            self.assertIs(self.session.timeline_clip(), old)
            self.assertEqual(self.session.scene_document, scene)
            self.assertEqual(self.session.revision, 1)
            setattr(self.client, kind, False)

    def test_disabled_research_service_fails_before_job_submission(self):
        self.client.enabled = False
        self.generate()
        wait_for(lambda: not self.session.snapshot()['busy'])
        self.assertEqual(self.client.calls, [])
        self.assertIn('does not expose', self.session.snapshot()['failure'])

    def test_cancel_and_deactivate_discard_late_output(self):
        old = self.completed()
        for action in (self.session.cancel, self.session.deactivate):
            self.client.release.clear(); self.client.partial.clear(); self.client.ignore_cancel = True
            self.generate()
            self.assertTrue(self.client.partial.wait(1))
            action()
            self.client.release.set()
            wait_for(lambda: not self.session.snapshot()['busy'])
            self.assertIs(self.session.timeline_clip(), old)
            self.assertIsNone(self.session.snapshot()['failure'])
            self.assertEqual(self.session.revision, 1)

    def test_no_concurrent_job_when_previous_request_is_draining(self):
        self.client.release.clear(); self.client.ignore_cancel = True
        self.generate()
        self.assertTrue(self.client.partial.wait(1))
        self.session.cancel()
        with self.assertRaisesRegex(RuntimeError, 'running or cancelling'):
            self.generate()
        self.client.release.set()
        wait_for(lambda: not self.session.snapshot()['busy'])
        self.completed()
        self.assertEqual(self.client.max_active, 1)

    def test_invalid_input_preserves_pending_request(self):
        self.client.release.clear()
        self.generate()
        self.assertTrue(self.client.partial.wait(1))
        epoch = self.session.snapshot()['epoch']
        with self.assertRaises(ValueError):
            self.session.generate('x' * 501, 1)
        self.assertEqual(self.session.snapshot()['epoch'], epoch)
        self.client.release.set()
        wait_for(lambda: not self.session.snapshot()['busy'])
        self.assertEqual(self.session.snapshot()['total_frames'], 120)

    def test_invalid_scene_fails_before_network_and_copy_is_private(self):
        self.session.activate()
        bad = {**EMPTY_SCENE, 'code': 'do not execute'}
        with self.assertRaises(ValueError):
            self.session.generate('wave', 1, scene_document=bad)
        self.assertEqual(self.client.health_calls, 0)
        copy = self.session.scene_document
        copy['objects'].append({})
        self.assertEqual(self.session.scene_document['objects'], [])

    def test_scene_overlap_and_unsupported_floor_keep_old_motion_and_scene(self):
        old = self.completed()
        original_scene = self.session.scene_document
        wall = make_object('wall', 0)
        wall.update(position=[0., .5, .2], size=[2., 1., .5])
        floor = make_object('platform', 0)
        floor.update(position=[10., .05, 0.], size=[6., .1, 6.])
        for obstacle in (wall, floor):
            self.generate(scene_document={**EMPTY_SCENE, 'objects': [obstacle]})
            wait_for(lambda: not self.session.snapshot()['busy'])
            self.assertTrue(self.session.snapshot()['failure'])
            self.assertIs(self.session.timeline_clip(), old)
            self.assertEqual(self.session.scene_document, original_scene)
        with self.assertRaises(ValueError):
            self.session.load(encode_project(old, {**EMPTY_SCENE, 'objects': [wall]}))
        self.assertIs(self.session.timeline_clip(), old)

    def test_last_frame_upper_torso_overlap_rejected_while_roots_remain_separated(self):
        old = self.completed()
        original_scene = self.session.scene_document
        self.client.torso_overlap = True
        self.generate(scene_document={**EMPTY_SCENE, 'name': 'Rejected backdrop'})
        wait_for(lambda: not self.session.snapshot()['busy'])
        self.assertIn('torso overlap', self.session.snapshot()['failure'])
        self.assertIs(self.session.timeline_clip(), old)
        self.assertEqual(self.session.scene_document, original_scene)
        self.assertEqual(self.session.revision, 1)

    def test_close_jointly_generated_pair_is_not_rejected_by_core_root_disc_gate(self):
        body = request_body('Paired close gesture', 7301)
        chunks = []
        for index in range(3):
            original = chunk(body, index)
            positions = original.positions.copy()
            positions[1, :, :, 0] = .635
            positions[1, :, 16] = positions[0, :, 16]  # Intentional hand contact.
            chunks.append(CanonicalClip(positions, original.rotations, 20, ACTOR_IDS,
                                        'intergen', original.metadata))
        clip = join_chunks(chunks, body)
        state = self.session.load(encode_project(clip, EMPTY_SCENE))
        self.assertEqual(state['total_frames'], 120)
        self.assertFalse(state['physical_contact_verified'])

    def test_playback_seek_pause_restart_and_end_at_20fps(self):
        now = [10.]
        with PairedStudioSession(clock=lambda: now[0]) as session:
            session.load(encode_project(complete_clip(), EMPTY_SCENE, 12))
            self.assertFalse(session.snapshot()['playing'])
            session.play(); now[0] += .5
            self.assertEqual(session.tick()['frame'], 22)
            session.pause(); now[0] += 2
            self.assertEqual(session.tick()['frame'], 22)
            session.seek(30); session.restart(); now[0] += 1
            self.assertEqual(session.tick()['frame'], 20)
            now[0] += 20
            self.assertEqual(session.tick()['frame'], 119)
            self.assertFalse(session.snapshot()['playing'])
            session.play()
            self.assertEqual(session.snapshot()['frame'], 0)
            for frame in (-1, 120, True):
                with self.assertRaises(ValueError): session.seek(frame)

    def test_exact_archive_load_is_offline_and_separate_from_core(self):
        original = self.completed()
        self.session.seek(32)
        content = self.session.save()
        with PairedStudioSession() as restored:
            state = restored.load(content)
            self.assertTrue(state['active']); self.assertFalse(state['playing'])
            self.assertEqual(state['frame'], 32)
            np.testing.assert_array_equal(restored.timeline_clip().positions, original.positions)
            np.testing.assert_array_equal(restored.timeline_clip().rotations, original.rotations)
            self.assertEqual(restored.scene_document, self.session.scene_document)
            self.assertEqual(restored.timeline_clip().metadata['license'], LICENSE)
        with CoreStudioSession() as native:
            with self.assertRaises(ValueError): native.load(content)

    def test_load_during_generation_invalidates_pending_but_preserves_loaded_project(self):
        saved = encode_project(complete_clip(), {**EMPTY_SCENE, 'name': 'Saved backdrop'}, 8)
        self.client.release.clear(); self.client.ignore_cancel = True
        self.generate()
        self.assertTrue(self.client.partial.wait(1))
        self.session.load(saved)
        loaded = self.session.timeline_clip()
        self.client.release.set()
        wait_for(lambda: not self.session.snapshot()['busy'])
        self.assertIs(self.session.timeline_clip(), loaded)
        self.assertEqual(self.session.snapshot()['frame'], 8)
        self.assertEqual(self.session.scene_document['name'], 'Saved backdrop')

    def test_bad_load_does_not_replace_clip_or_cancel_pending(self):
        old = self.completed()
        self.client.release.clear()
        self.generate(); self.assertTrue(self.client.partial.wait(1))
        epoch = self.session.snapshot()['epoch']
        with self.assertRaises(ValueError): self.session.load(b'not a project')
        self.assertIs(self.session.timeline_clip(), old)
        self.assertEqual(self.session.snapshot()['epoch'], epoch)

    def test_close_cancels_late_result_and_forbids_new_work(self):
        self.client.release.clear(); self.client.ignore_cancel = True
        self.generate(); self.assertTrue(self.client.partial.wait(1))
        self.session.close(timeout=0)
        self.client.release.set()
        wait_for(lambda: not self.session.snapshot()['busy'])
        self.assertIsNone(self.session.timeline_clip())
        self.assertFalse(self.session.available)
        with self.assertRaises(RuntimeError): self.session.activate()


class PairedArchiveTests(unittest.TestCase):
    def rewrite(self, content, edit):
        with np.load(io.BytesIO(content), allow_pickle=False) as data:
            values = {key: data[key] for key in data.files}
        metadata = json.loads(values['metadata'].item())
        edit(metadata, values)
        values['metadata'] = np.array(json.dumps(metadata))
        output = io.BytesIO(); np.savez_compressed(output, **values)
        return output.getvalue()

    def test_request_matches_real_backend_and_bounds(self):
        for frames in (40, 80, 120):
            body = request_body('A friendly gesture', 2**32 - 1, frames)
            validate_job(body, pair_enabled=True)
        for prompt, seed, frames in [('', 1, 120), ('a', True, 120), ('a', -1, 120),
                                    ('a', 2**32, 120), ('a', 1, 121), ('a', 1, True)]:
            with self.assertRaises(ValueError): request_body(prompt, seed, frames)
        body = request_body('a', 1)
        with self.assertRaises(ValueError): validate_request({**body, 'history': {}})

    def test_rejects_native_features_incomplete_source_and_changed_chunk_identity(self):
        body = request_body('a', 1)
        good = [chunk(body, i) for i in range(3)]
        for values in [good[:2], [chunk(body, 0, native=True)] + good[1:],
                       [chunk(body, 0, source='ardy_core')] + good[1:],
                       [good[1], good[0], good[2]]]:
            with self.assertRaises(ValueError): join_chunks(values, body)

    def test_malformed_research_archive_rejected_before_mutation(self):
        original = encode_project(complete_clip(), EMPTY_SCENE)
        mutations = [lambda m, a: m.update(format='stagezero_core'),
                     lambda m, a: m.update(research_only=False),
                     lambda m, a: m.update(license='unrestricted'),
                     lambda m, a: m.update(frame=120),
                     lambda m, a: m.update(frame=True),
                     lambda m, a: m['scene_document'].update(script='no'),
                     lambda m, a: m['clip_metadata']['request'].update(frames=40),
                     lambda m, a: m['clip_metadata'].update(physical_contact_verified=True),
                     lambda m, a: a.update(native_features=np.zeros((2,120,330))),
                     lambda m, a: a.update(positions=np.zeros((2,121,27,3))),
                     lambda m, a: a.update(rotations=np.zeros((2,120,27,3,3))),
                     lambda m, a: a.update(positions=np.full((2,120,27,3), np.nan))]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                decode_project(self.rewrite(original, mutation))

    def test_archive_rejects_duplicate_members_and_object_arrays(self):
        original = encode_project(complete_clip(), EMPTY_SCENE)
        duplicated = io.BytesIO()
        with ZipFile(io.BytesIO(original)) as source, ZipFile(duplicated, 'w') as target:
            for name in source.namelist(): target.writestr(name, source.read(name))
            target.writestr('extra.npy', b'no')
        with self.assertRaises(ValueError): decode_project(duplicated.getvalue())
        corrupt = self.rewrite(original, lambda m,a: a.update(positions=np.array([object()],dtype=object)))
        with self.assertRaises(ValueError): decode_project(corrupt)


if __name__ == '__main__':
    unittest.main()
