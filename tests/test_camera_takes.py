"""Cameras survive project and take editing without invoking real inference."""
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from directing import DirectorSession
from takes import decode_project, encode_project
from test_live_motion import ControlledBackend, wait_until


def rewrite_manifest(data, rewrite):
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    manifest = json.loads(str(arrays['manifest']))
    rewrite(manifest)
    arrays['manifest'] = np.array(json.dumps(manifest))
    out = io.BytesIO()
    np.savez_compressed(out, **arrays)
    return out.getvalue()


class CameraTakeTests(unittest.TestCase):
    def setUp(self):
        self.backend = ControlledBackend()
        self.backend.release.set()
        self.session = DirectorSession(self.backend, np.zeros((20, 34, 3), dtype=np.float32),
                                       np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1)))
        self.session.set_mode('Live ARDY')
        self.take = self.generate('initial', seconds=.8)
        self.camera = self.session.add_camera([1, 2, 3], [2, 0, 0, 0], 1.0)

    def finish(self):
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        return self.session.takes[self.session.active_take]

    def generate(self, prompt, **kwargs):
        self.session.submit(prompt, **kwargs)
        return self.finish()

    def add_cut(self, frame, camera=None):
        return self.session.add_camera_cut(self.take.id, frame, (camera or self.camera)['id'])

    def archive(self):
        return encode_project(self.session.takes, self.session.active_take, self.session.frame,
                              self.session.scene, self.session.cameras)

    def test_camera_crud_is_detached_dirty_and_never_generates_motion(self):
        revision = self.session.project_revision
        calls = len(self.backend.histories)
        copied = self.session.duplicate_camera(self.camera['id'])
        self.assertNotEqual(copied['id'], self.camera['id'])
        self.assertNotEqual(copied['name'], self.camera['name'])
        copied['position'][0] = 100
        self.assertEqual(self.session.cameras[-1]['position'], [1.0, 2.0, 3.0])
        updated = self.session.update_camera(self.camera['id'], name='Close', position=[3, 4, 5],
                                             wxyz=[0, 0, 3, 0], fov=.75)
        self.assertEqual(updated['wxyz'], [0.0, 0.0, 1.0, 0.0])
        self.assertEqual(self.session.project_revision, revision + 2)
        self.session.remove_camera(copied['id'])
        self.assertEqual(len(self.session.cameras), 1)
        self.assertEqual(self.session.project_revision, revision + 3)
        self.assertIn('Unsaved', self.session.project_status)
        self.assertEqual(len(self.backend.histories), calls)

    def test_invalid_camera_update_does_not_change_state(self):
        before = json.dumps(self.session.cameras)
        revision = self.session.project_revision
        for fields in ({'id': 'new-id'}, {'fov': float('nan')}, {'position': [0, 0]},
                       {'name': ''}, {'wxyz': [0, 0, 0, 0]}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.session.update_camera(self.camera['id'], **fields)
            self.assertEqual(json.dumps(self.session.cameras), before)
            self.assertEqual(self.session.project_revision, revision)

    def test_cut_add_replace_move_and_clear_preserve_unique_frames(self):
        calls = len(self.backend.histories)
        later = self.add_cut(8)
        self.assertEqual(later['frame'], 8)
        self.assertEqual([cut['frame'] for cut in self.take.camera_cuts], [0, 8])
        other = self.session.duplicate_camera(self.camera['id'])
        replaced = self.add_cut(8, other)
        self.assertEqual(replaced['id'], later['id'])
        self.assertEqual(self.take.camera_cuts[1]['camera_id'], other['id'])
        moving = self.add_cut(16)
        moved = self.session.update_camera_cut(self.take.id, moving['id'], frame=8)
        self.assertEqual(moved['id'], moving['id'])
        self.assertEqual([cut['frame'] for cut in self.take.camera_cuts], [0, 8])
        self.assertEqual(self.take.camera_cuts[-1]['id'], moving['id'])
        self.session.remove_camera_cut(self.take.id, moved['id'])
        self.assertEqual(len(self.take.camera_cuts), 1)
        self.session.clear_camera_cuts(self.take.id)
        self.assertEqual(self.take.camera_cuts, [])
        self.assertEqual(len(self.backend.histories), calls)

    def test_cut_can_replace_frame_zero_but_anchor_cannot_move_or_be_deleted(self):
        anchor = self.add_cut(0)
        moving = self.add_cut(8)
        moved = self.session.update_camera_cut(self.take.id, moving['id'], frame=0)
        self.assertEqual(self.take.camera_cuts, [moved])
        self.assertNotEqual(moved['id'], anchor['id'])
        before = [dict(cut) for cut in self.take.camera_cuts]
        revision = self.session.project_revision
        with self.assertRaises(ValueError):
            self.session.update_camera_cut(self.take.id, moved['id'], frame=4)
        with self.assertRaises(ValueError):
            self.session.remove_camera_cut(self.take.id, moved['id'])
        for frame in (-1, 20, 1.5, True):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                self.add_cut(frame)
        with self.assertRaises(ValueError):
            self.session.add_camera_cut(self.take.id, 8, 'missing-camera')
        self.assertEqual(self.take.camera_cuts, before)
        self.assertEqual(self.session.project_revision, revision)

    def test_camera_edits_allowed_during_generation_but_cut_edits_rejected(self):
        cut = self.add_cut(0)
        self.backend.release.clear()
        self.backend.started.clear()
        try:
            self.session.submit('waiting', seconds=.8, edit_mode='extend')
            self.assertTrue(self.backend.started.wait(1))
            self.session.update_camera(self.camera['id'], name='During generation')
            extra = self.session.duplicate_camera(self.camera['id'])
            self.session.remove_camera(extra['id'])
            actions = [lambda: self.add_cut(8),
                       lambda: self.session.update_camera_cut(self.take.id, cut['id'], camera_id=self.camera['id']),
                       lambda: self.session.remove_camera_cut(self.take.id, cut['id']),
                       lambda: self.session.clear_camera_cuts(self.take.id)]
            for action in actions:
                with self.assertRaisesRegex(ValueError, 'generation'):
                    action()
            self.assertTrue(self.session.busy)
            self.assertEqual(self.take.camera_cuts, [cut])
        finally:
            self.backend.release.set()
        self.finish()

    def test_duplicate_and_trim_copy_cuts_without_mutating_source(self):
        self.add_cut(8)
        self.add_cut(16)
        original = [dict(cut) for cut in self.take.camera_cuts]
        duplicate = self.session.duplicate_active_take()
        self.assertEqual(duplicate.camera_cuts, original)
        self.session.update_camera_cut(duplicate.id, duplicate.camera_cuts[-1]['id'], frame=17)
        self.assertEqual(self.take.camera_cuts, original)
        self.session.select_take(self.take.id)
        self.session.seek(7)
        trimmed = self.session.trim_after_playhead()
        self.assertEqual(len(trimmed.positions), 8)
        self.assertEqual([cut['frame'] for cut in trimmed.camera_cuts], [0])
        self.assertEqual(self.take.camera_cuts, original)

    def test_extend_and_replace_ending_preserve_absolute_cut_times(self):
        self.add_cut(8)
        self.add_cut(16)
        original = [dict(cut) for cut in self.take.camera_cuts]
        extended = self.generate('extend', seconds=.8, edit_mode='extend')
        self.assertEqual(extended.camera_cuts, original)
        branch = self.generate('short ending', seconds=.16, edit_mode='replace', at_frame=8)
        self.assertNotEqual(branch.id, extended.id)
        self.assertEqual(len(branch.positions), 12)
        self.assertEqual([cut['frame'] for cut in branch.camera_cuts], [0, 8])
        self.assertEqual(extended.camera_cuts, original)

    def test_action_insert_keeps_absolute_cuts_and_undo_restores_snapshot(self):
        self.add_cut(8)
        self.add_cut(16)
        original = [dict(cut) for cut in self.take.camera_cuts]
        self.assertTrue(self.session.submit_action_edit('insert', 0, 'insert_before', seconds=.16))
        inserted = self.finish()
        self.assertEqual(len(inserted.positions), 24)
        self.assertEqual(inserted.camera_cuts, original)
        self.assertTrue(self.session.undo_action_edit())
        self.assertIs(self.session.takes[self.take.id], self.take)
        self.assertEqual(self.take.camera_cuts, original)

    def test_action_shorten_truncates_cuts_and_undo_prevents_dangling_camera(self):
        self.add_cut(0)
        late_camera = self.session.duplicate_camera(self.camera['id'])
        self.add_cut(16, late_camera)
        original = [dict(cut) for cut in self.take.camera_cuts]
        self.assertTrue(self.session.submit_action_edit('short', 0, 'replace', seconds=.16))
        shortened = self.finish()
        self.assertEqual(len(shortened.positions), 4)
        self.assertEqual([cut['frame'] for cut in shortened.camera_cuts], [0])
        with self.assertRaisesRegex(ValueError, 'Undo'):
            self.session.remove_camera(late_camera['id'])
        self.assertTrue(self.session.undo_action_edit())
        self.assertEqual(self.session.takes[self.take.id].camera_cuts, original)
        self.archive()

    def test_cut_edit_after_action_edit_invalidates_undo_before_camera_delete(self):
        self.add_cut(16)
        self.assertTrue(self.session.submit_action_edit('short', 0, 'replace', seconds=.16))
        self.finish()
        self.assertTrue(self.session.can_undo_action_edit)
        self.session.clear_camera_cuts(self.take.id)
        self.assertFalse(self.session.can_undo_action_edit)
        self.session.remove_camera(self.camera['id'])
        self.assertFalse(self.session.undo_action_edit())
        self.archive()

    def test_removed_take_camera_reference_survives_undo(self):
        self.add_cut(8)
        self.session.rename_active_take('Opening shot')
        self.assertTrue(self.session.remove_active_take())
        with self.assertRaisesRegex(ValueError, 'removed take “Opening shot”.*Restore.*Undo'):
            self.session.remove_camera(self.camera['id'])
        self.assertTrue(self.session.undo_remove_take())
        self.assertEqual([cut['frame'] for cut in self.take.camera_cuts], [0, 8])
        self.session.clear_camera_cuts(self.take.id)
        self.session.remove_camera(self.camera['id'])
        self.assertEqual(self.session.cameras, [])

    def test_camera_deletion_names_each_using_take(self):
        self.add_cut(0)
        self.session.rename_active_take('First performance')
        self.session.duplicate_active_take()
        self.session.rename_active_take('Alternate performance')
        with self.assertRaisesRegex(ValueError, 'First performance.*Alternate performance'):
            self.session.remove_camera(self.camera['id'])

    def test_superseded_undo_does_not_leave_an_undeletable_orphan_camera(self):
        self.add_cut(0)
        late_camera = self.session.duplicate_camera(self.camera['id'])
        self.add_cut(16, late_camera)
        self.assertTrue(self.session.submit_action_edit('short', 0, 'replace', seconds=.16))
        self.finish()
        self.assertTrue(self.session.can_undo_action_edit)
        self.generate('extend', seconds=.16, edit_mode='extend')
        self.assertFalse(self.session.can_undo_action_edit)
        self.session.remove_camera(late_camera['id'])
        self.assertNotIn(late_camera['id'], [camera['id'] for camera in self.session.cameras])
        self.assertFalse(self.session.undo_action_edit())
        self.archive()

    def test_save_rejects_metadata_over_load_limit_before_resetting_project(self):
        self.add_cut(8)
        epoch = self.session.camera_project_id
        with tempfile.TemporaryDirectory() as directory, patch('takes.MAX_METADATA_CHARS', 100):
            with self.assertRaisesRegex(ValueError, 'metadata'):
                self.session.new_project(directory)
        self.assertIs(self.session.takes[self.take.id], self.take)
        self.assertEqual(self.session.camera_project_id, epoch)
        self.assertEqual(self.session.cameras, [self.camera])

    def test_v2_save_load_and_new_project_rotate_epoch_and_roundtrip_camera_only_project(self):
        self.add_cut(8)
        old_project = self.session.camera_project_id
        self.session.seek(7)
        with tempfile.TemporaryDirectory() as directory:
            _, data = self.session.save_project(directory, 'with-cameras')
            with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                manifest = json.loads(str(archive['manifest']))
            self.assertEqual(manifest['version'], 2)
            self.assertEqual(manifest['cameras'], self.session.cameras)
            self.session.load_project(data)
            self.assertNotEqual(self.session.camera_project_id, old_project)
            self.assertEqual(self.session.frame, 7)
            self.assertEqual(self.session.cameras, manifest['cameras'])
            self.assertEqual(self.session.takes[self.take.id].camera_cuts, self.take.camera_cuts)
            loaded_project = self.session.camera_project_id
            self.session.new_project(directory)
            self.assertNotEqual(self.session.camera_project_id, loaded_project)
            self.assertEqual(self.session.cameras, [])
            camera = self.session.add_camera([0, 3, -4], [1, 0, 0, 0], 1)
            _, data = self.session.save_project(directory, 'camera-only')
            self.session.load_project(data)
            self.assertEqual(self.session.takes, {})
            self.assertEqual(self.session.cameras, [camera])
            self.assertIsNone(self.session.active_take)

    def test_v1_project_loads_without_cameras_or_cuts(self):
        self.add_cut(8)
        def legacy(manifest):
            manifest['version'] = 1
            del manifest['cameras']
            for take in manifest['takes']:
                del take['camera_cuts']
        data = rewrite_manifest(self.archive(), legacy)
        self.session.load_project(data)
        self.assertEqual(self.session.cameras, [])
        self.assertEqual(self.session.takes[self.take.id].camera_cuts, [])
        self.assertEqual(len(decode_project(data)), 4)

    def test_invalid_v2_camera_or_cut_load_preserves_entire_live_project(self):
        self.add_cut(8)
        data = self.archive()
        original_cameras = self.session.cameras
        original_epoch = self.session.camera_project_id
        revision = self.session.project_revision
        invalid = [
            lambda doc: doc['cameras'].append(dict(doc['cameras'][0])),
            lambda doc: doc['cameras'][0].update(wxyz=[0, 0, 0, 0]),
            lambda doc: doc['cameras'][0].update(fov=float('nan')),
            lambda doc: doc['takes'][0]['camera_cuts'][0].update(camera_id='missing'),
            lambda doc: doc['takes'][0]['camera_cuts'][0].update(frame=1),
            lambda doc: doc['takes'][0]['camera_cuts'][1].update(frame=0),
            lambda doc: doc['takes'][0]['camera_cuts'][1].update(frame=20),
            lambda doc: doc.pop('cameras'),
        ]
        for rewrite in invalid:
            with self.subTest(rewrite=rewrite), self.assertRaises(ValueError):
                self.session.load_project(rewrite_manifest(data, rewrite))
            self.assertIs(self.session.takes[self.take.id], self.take)
            self.assertIs(self.session.cameras, original_cameras)
            self.assertEqual(self.session.camera_project_id, original_epoch)
            self.assertEqual(self.session.project_revision, revision)
