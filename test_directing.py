"""Offline tests use controlled synthetic transport, never presented as AI motion."""
import io
import json
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from directing import DirectorSession
from takes import encode_project, decode_project
from test_live_motion import ControlledBackend, wait_until


class ActionBackend(ControlledBackend):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.fail_prompt = None

    def generate(self, request_id, prompt, history):
        self.calls.append((prompt, None if history is None else history.copy()))
        if prompt == self.fail_prompt:
            raise RuntimeError('Selected action failed')
        return super().generate(request_id, prompt, history)


class DirectingTests(unittest.TestCase):
    def setUp(self):
        self.backend = ControlledBackend()
        self.backend.release.set()
        self.session = DirectorSession(self.backend, np.zeros((20, 34, 3), dtype=np.float32), np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1)))
        self.session.set_mode('Live ARDY')

    def generate(self, prompt='old'):
        self.session.submit(prompt)
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        return self.session.takes[self.session.active_take]

    def test_extend_records_both_segments(self):
        first = self.generate()
        self.session.seek(103)
        second = self.generate('new')
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(second.positions), 208)
        np.testing.assert_array_equal(first.positions, second.positions[:104])
        self.assertEqual([s['prompt'] for s in second.segments], ['old', 'new'])
        self.assertEqual(self.session.frame, 104)

    def test_branch_preserves_original_and_exact_prefix(self):
        original = self.generate()
        self.session.seek(40)
        branch = self.generate('new')
        self.assertNotEqual(original.id, branch.id)
        self.assertEqual(branch.parent, original.id)
        self.assertEqual(branch.branch_frame, 40)
        self.assertIs(self.session.takes[original.id], original)
        for name in ('positions', 'rotations', 'motion'):
            np.testing.assert_array_equal(getattr(original, name)[:41], getattr(branch, name)[:41])
        self.assertEqual(len(branch.positions), 145)
        self.assertEqual(branch.segments[0]['end'], 41)

    def test_roundtrip_preserves_all_motion_and_playhead(self):
        self.generate()
        self.session.seek(30)
        self.generate('new')
        self.session.seek(60)
        with tempfile.TemporaryDirectory() as folder:
            path, data = self.session.save_project(folder, '../../Take one')
            self.assertEqual(path.parent.name, folder.split('/')[-1])
            before = self.session.takes.copy()
            self.session.new_take()
            self.session.load_project(data)
            self.assertEqual(self.session.frame, 60)
            self.assertFalse(self.session.playing)
            for tid, t in before.items():
                np.testing.assert_array_equal(t.motion, self.session.takes[tid].motion)
                self.assertEqual(t.segments, self.session.takes[tid].segments)

    def test_reset_and_new_take_preserve_existing_takes(self):
        t = self.generate()
        self.session.seek(55)
        self.session.reset()
        self.assertEqual(self.session.frame, 0)
        self.assertEqual(self.session.active_take, t.id)
        self.session.new_take()
        self.assertIsNone(self.session.active_take)
        self.assertIn(t.id, self.session.takes)
        self.assertEqual(self.session.kind, 'reference')
        self.assertEqual(len(self.session.positions), 1)
        self.session.seek(10)
        self.assertEqual(self.session.frame, 0)

    def test_remove_child_and_undo_restores_selection_and_archive(self):
        parent = self.generate()
        self.session.seek(40)
        child = self.generate('branch')
        self.session.seek(25)
        self.assertTrue(self.session.remove_active_take())
        self.assertEqual(self.session.active_take, parent.id)
        self.assertNotIn(child.id, self.session.takes)
        self.assertTrue(self.session.can_undo_take_removal)
        self.assertTrue(self.session.undo_remove_take())
        self.assertEqual(self.session.active_take, child.id)
        self.assertEqual(self.session.frame, 25)
        self.assertEqual(self.session.takes[child.id].parent, parent.id)
        self.assertFalse(self.session.can_undo_take_removal)
        self.assertFalse(self.session.undo_remove_take())
        data = encode_project(self.session.takes, child.id, self.session.frame, self.session.scene)
        loaded, _, _, _ = decode_project(data)
        self.assertEqual(loaded[child.id].parent, parent.id)

    def test_remove_parent_detaches_child_and_undo_restores_provenance(self):
        parent = self.generate()
        self.session.seek(40)
        child = self.generate('branch')
        self.session.select_take(parent.id)
        self.assertTrue(self.session.remove_active_take())
        self.assertEqual(self.session.active_take, child.id)
        self.assertIsNone(child.parent)
        self.assertIsNone(child.branch_frame)
        data = encode_project(self.session.takes, child.id, 0, self.session.scene)
        loaded, _, _, _ = decode_project(data)
        self.assertIsNone(loaded[child.id].parent)
        self.assertTrue(self.session.undo_remove_take())
        self.assertEqual(child.parent, parent.id)
        self.assertEqual(child.branch_frame, 40)
        data = encode_project(self.session.takes, parent.id, 0, self.session.scene)
        loaded, _, _, _ = decode_project(data)
        self.assertEqual(loaded[child.id].parent, parent.id)

    def test_remove_last_take_enters_draft_and_can_undo(self):
        take = self.generate()
        self.assertTrue(self.session.remove_active_take())
        self.assertEqual(self.session.takes, {})
        self.assertIsNone(self.session.active_take)
        self.assertEqual(self.session.kind, 'reference')
        self.assertEqual(len(self.session.positions), 1)
        self.session.scene['gate']['position'] = [2.0, 0.0, 1.0]
        with tempfile.TemporaryDirectory() as folder:
            _, data = self.session.save_project(folder, 'empty')
        loaded, active, frame, scene = decode_project(data)
        self.assertEqual(loaded, {})
        self.assertIsNone(active)
        self.assertEqual(frame, 0)
        self.assertEqual(scene['gate']['position'], [2.0, 0.0, 1.0])
        self.assertTrue(self.session.undo_remove_take())
        self.assertIs(self.session.takes[take.id], take)
        self.assertEqual(self.session.active_take, take.id)
        self.session.load_project(data)
        self.assertEqual(self.session.takes, {})
        self.assertIsNone(self.session.active_take)
        self.assertEqual(len(self.session.positions), 1)
        self.assertFalse(self.session.can_undo_take_removal)

    def test_remove_and_undo_refuse_while_generating(self):
        take = self.generate()
        self.backend.started.clear()
        self.backend.release.clear()
        self.session.seek(40)
        self.session.submit('pending')
        self.assertTrue(self.backend.started.wait(1))
        self.assertFalse(self.session.new_take())
        self.assertEqual(self.session.active_take, take.id)
        self.assertFalse(self.session.remove_active_take())
        self.assertIn(take.id, self.session.takes)
        self.session.seek(0)
        self.assertTrue(self.session.remove_active_take())
        self.session.submit('pending again')
        self.assertFalse(self.session.undo_remove_take())
        self.assertEqual(self.session.takes, {})
        self.session.seek(0)
        self.backend.release.set()
        self.assertTrue(self.session.undo_remove_take())

    def test_loading_project_clears_old_removal_undo(self):
        take = self.generate()
        data = encode_project(self.session.takes, take.id, 0, self.session.scene)
        self.assertTrue(self.session.remove_active_take())
        self.session.load_project(data)
        self.assertFalse(self.session.can_undo_take_removal)

    def test_archive_rejects_dangling_and_cyclic_branch_links(self):
        parent = self.generate()
        self.session.seek(40)
        child = self.generate('branch')
        child.parent = None
        with self.assertRaisesRegex(ValueError, 'branch provenance'):
            encode_project(self.session.takes, child.id, 0, self.session.scene)
        child.parent = parent.id
        parent.parent = child.id
        parent.branch_frame = 3
        with self.assertRaisesRegex(ValueError, 'Cyclic branch provenance'):
            encode_project(self.session.takes, child.id, 0, self.session.scene)

    def test_seek_cancels_pending_result(self):
        t = self.generate()
        self.backend.started.clear(); self.backend.release.clear()
        self.session.seek(40); self.session.submit('new')
        self.assertTrue(self.backend.started.wait(1))
        self.session.seek(20); self.backend.release.set()
        self.backend.started.clear()
        self.session.submit('old')
        wait_until(lambda: not self.session.busy)
        self.assertEqual(len(self.session.takes), 2)
        self.assertEqual(self.session.takes[self.session.active_take].branch_frame, 20)
        self.assertIs(self.session.takes[t.id], t)

    def test_switching_mode_preserves_takes(self):
        t = self.generate()
        self.session.set_mode('Recorded preview')
        self.assertEqual(self.session.kind, 'recorded')
        self.session.set_mode('Live ARDY')
        self.assertEqual(self.session.active_take, t.id)
        self.assertEqual(self.session.kind, 'generated')

    def test_invalid_load_is_transactional(self):
        t = self.generate()
        with self.assertRaises(Exception): self.session.load_project(b'not a project')
        self.assertIs(self.session.takes[t.id], t)
        self.assertEqual(self.session.active_take, t.id)

    def test_incompatible_saved_motion_rejected(self):
        self.generate()
        data = encode_project(self.session.takes, self.session.active_take, 0, self.session.scene)
        with np.load(io.BytesIO(data), allow_pickle=False) as src:
            arrays = {k: src[k] for k in src.files}
        arrays['t0_motion'] = np.zeros((104, 27))
        buffer = io.BytesIO(); np.savez_compressed(buffer, **arrays)
        with self.assertRaises(ValueError): decode_project(buffer.getvalue())

    def test_too_early_branch_explains_required_history(self):
        self.generate(); self.session.seek(0); self.session.submit('new')
        self.assertFalse(self.session.busy)
        self.assertIn('0.12', self.session.status)
        self.assertEqual(len(self.session.takes), 1)

    def test_gate_rewinds_and_survives_save(self):
        self.session.scene['gate'] = dict(position=[1, 0, 1], radius=.65, enabled=True)
        t = self.generate()
        # Simulate a recorded event at a later frame to test clock restoration.
        t.events = [dict(type='gate_open', frame=40)]
        self.session.seek(60); self.assertTrue(self.session.gate_open())
        data = encode_project(self.session.takes, t.id, 60, self.session.scene)
        self.session.seek(20); self.assertFalse(self.session.gate_open())
        self.session.load_project(data); self.assertTrue(self.session.gate_open())
        self.session.reset(); self.assertFalse(self.session.gate_open())


class ActionEditingTests(unittest.TestCase):
    def setUp(self):
        self.backend = ActionBackend()
        self.backend.release.set()
        self.session = DirectorSession(self.backend, np.zeros((20, 34, 3), dtype=np.float32),
                                       np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1)))
        self.session.set_mode('Live ARDY')

    def generate(self, prompt='old'):
        self.session.submit(prompt)
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        return self.session.takes[self.session.active_take]

    def test_static_character_blocks_action_edit_until_motion_is_enabled(self):
        before = self.generate('first')
        self.backend.calls.clear()
        self.session.set_character_motion_enabled(False)
        revision = self.session.action_edit_revision
        for operation in ('replace', 'insert_before', 'insert_after'):
            with self.subTest(operation=operation):
                self.assertFalse(self.session.submit_action_edit('changed', 0, operation))
                self.assertFalse(self.session.busy)
                self.assertIs(self.session.takes[before.id], before)
                self.assertEqual(self.session.action_edit_revision, revision)
        self.assertEqual(self.backend.calls, [])
        self.session.set_character_motion_enabled(True)
        self.assertTrue(self.session.submit_action_edit('changed', 0, 'replace'))
        wait_until(lambda: not self.session.busy)
        self.assertEqual([prompt for prompt, _ in self.backend.calls], ['changed'])
        self.assertEqual(self.session.takes[before.id].segments[0]['prompt'], 'changed')

    def sequence(self):
        original = self.generate('first')
        self.session.seek(len(original.positions) - 1)
        self.session.submit('second', seconds=2)
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        take = self.session.takes[self.session.active_take]
        self.session.seek(len(take.positions) - 1)
        self.session.submit('third', seconds=3)
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        return self.session.takes[self.session.active_take]

    def test_insert_middle_keeps_prefix_and_regenerates_suffix_in_order(self):
        before = self.sequence()
        self.backend.calls.clear()
        revision = self.session.action_edit_revision
        self.assertTrue(self.session.submit_action_edit('inserted', 1, 'insert_before', seconds=1))
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        after = self.session.takes[before.id]
        self.assertEqual(len(self.session.takes), 1)
        self.assertEqual((after.id, after.name), (before.id, before.name))
        self.assertEqual([(s['prompt'], s['end'] - s['start']) for s in after.segments],
                         [('first', 104), ('inserted', 25), ('second', 50), ('third', 75)])
        self.assertEqual([prompt for prompt, _ in self.backend.calls], ['inserted', 'second', 'third'])
        np.testing.assert_array_equal(after.motion[:104], before.motion[:104])
        np.testing.assert_array_equal(self.backend.calls[0][1], before.motion[52:104])
        self.assertEqual(self.backend.calls[1][1].shape, (52, 414))
        self.assertEqual(self.session.action_edit_revision, revision + 1)
        self.assertTrue(self.session.can_undo_action_edit)
        data = encode_project(self.session.takes, after.id, 0, self.session.scene)
        loaded, _, _, _ = decode_project(data)
        self.assertEqual(len(loaded[after.id].positions), 254)

    def test_replace_middle_preserves_duration_by_default_and_undo_restores_snapshot(self):
        before = self.sequence()
        self.session.seek(120)
        self.backend.calls.clear()
        self.assertTrue(self.session.submit_action_edit('changed', 1, 'replace'))
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        after = self.session.takes[before.id]
        self.assertIsNot(after, before)
        self.assertEqual(len(after.positions), len(before.positions))
        self.assertEqual([s['prompt'] for s in after.segments], ['first', 'changed', 'third'])
        self.assertEqual([prompt for prompt, _ in self.backend.calls], ['changed', 'third'])
        np.testing.assert_array_equal(after.motion[:104], before.motion[:104])
        self.assertTrue(self.session.undo_action_edit())
        self.assertIs(self.session.takes[before.id], before)
        self.assertEqual(self.session.frame, 120)
        self.assertFalse(self.session.can_undo_action_edit)

    def test_failure_and_cancellation_preserve_original(self):
        before = self.sequence()
        revision = self.session.action_edit_revision
        self.backend.fail_prompt = 'third'
        self.assertTrue(self.session.submit_action_edit('inserted', 1, 'insert_before', seconds=1))
        wait_until(lambda: not self.session.busy)
        self.assertIs(self.session.takes[before.id], before)
        self.assertEqual(self.session.action_edit_revision, revision)
        self.assertIn('Original take preserved', self.session.status)
        self.backend.fail_prompt = None
        self.backend.started.clear()
        self.backend.release.clear()
        self.assertTrue(self.session.submit_action_edit('inserted', 1, 'insert_after', seconds=1))
        self.assertTrue(self.backend.started.wait(1))
        self.session.seek(20)
        self.backend.release.set()
        time.sleep(.1)
        self.assertIs(self.session.takes[before.id], before)
        self.assertEqual(self.session.action_edit_revision, revision)

    def test_shortening_parent_detaches_out_of_range_child_and_undo_restores_link(self):
        parent = self.sequence()
        self.session.seek(220)
        child = self.generate('branch')
        self.session.select_take(parent.id)
        self.assertTrue(self.session.submit_action_edit('short', 1, 'replace', seconds=.16))
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        self.assertIsNone(child.parent)
        self.assertIsNone(child.branch_frame)
        edited = self.session.takes[parent.id]
        data = encode_project(self.session.takes, parent.id, 0, self.session.scene)
        loaded, _, _, _ = decode_project(data)
        self.assertIsNone(loaded[child.id].parent)
        self.assertTrue(self.session.undo_action_edit())
        self.assertIs(self.session.takes[parent.id], parent)
        self.assertEqual(child.parent, parent.id)
        self.assertEqual(child.branch_frame, 220)

    def test_edit_detaches_child_even_when_new_length_is_unchanged(self):
        parent = self.sequence()
        self.session.seek(120)
        child = self.generate('branch')
        self.session.select_take(parent.id)
        self.assertTrue(self.session.submit_action_edit('changed', 1, 'replace'))
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        self.assertEqual(len(self.session.takes[parent.id].positions), len(parent.positions))
        self.assertIsNone(child.parent)
        self.assertTrue(self.session.undo_action_edit())
        self.assertEqual(child.parent, parent.id)
        self.assertEqual(child.branch_frame, 120)

    def test_editing_inherited_prefix_clears_own_parent_link(self):
        parent = self.sequence()
        self.session.seek(120)
        child = self.generate('branch')
        self.assertEqual(child.parent, parent.id)
        self.assertTrue(self.session.submit_action_edit('changed', 0, 'replace'))
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        edited = self.session.takes[child.id]
        self.assertIsNone(edited.parent)
        self.assertIsNone(edited.branch_frame)
        self.assertTrue(self.session.undo_action_edit())
        self.assertIs(self.session.takes[child.id], child)
        self.assertEqual(child.parent, parent.id)

    def test_undo_detaches_new_branches_and_rename_clears_undo(self):
        parent = self.sequence()
        self.assertTrue(self.session.submit_action_edit('changed', 1, 'replace'))
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        edited = self.session.takes[parent.id]
        self.session.seek(120)
        new_child = self.generate('branch')
        self.assertEqual(new_child.parent, edited.id)
        self.assertTrue(self.session.undo_action_edit())
        self.assertIsNone(new_child.parent)
        self.assertTrue(self.session.submit_action_edit('changed again', 1, 'replace'))
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        self.assertTrue(self.session.rename_active_take('Renamed take'))
        self.assertFalse(self.session.can_undo_action_edit)

    def test_replace_short_partial_action_keeps_its_one_frame_duration(self):
        first = self.generate('first')
        self.session.seek(103)
        second = self.generate('second')
        self.session.seek(104)
        partial = self.generate('branch')
        self.assertEqual([(s['prompt'], s['end'] - s['start']) for s in partial.segments],
                         [('first', 104), ('second', 1), ('branch', 104)])
        self.assertTrue(self.session.submit_action_edit('changed', 1, 'replace'))
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        edited = self.session.takes[partial.id]
        self.assertEqual([(s['prompt'], s['end'] - s['start']) for s in edited.segments],
                         [('first', 104), ('changed', 1), ('branch', 104)])
        self.assertEqual(len(edited.positions), len(partial.positions))

    def test_invalid_and_busy_requests_do_not_start_edit(self):
        before = self.generate()
        self.assertFalse(self.session.submit_action_edit('bad', 1, 'replace'))
        self.assertFalse(self.session.submit_action_edit('bad', 0, 'unknown'))
        self.assertFalse(self.session.submit_action_edit('', 0, 'replace'))
        self.assertIs(self.session.takes[before.id], before)
        self.backend.started.clear()
        self.backend.release.clear()
        self.assertTrue(self.session.submit_action_edit('valid', 0, 'replace'))
        self.assertTrue(self.backend.started.wait(1))
        self.assertFalse(self.session.submit_action_edit('newer', 0, 'replace'))
        self.assertFalse(self.session.undo_action_edit())
        self.session.seek(0)
        self.backend.release.set()

    def test_budget_rejection_and_project_load_clear_undo(self):
        before = self.generate()
        self.backend.calls.clear()
        with patch('directing.MAX_TOTAL_FRAMES', len(before.positions)):
            self.assertFalse(self.session.submit_action_edit('inserted', 0, 'insert_after', seconds=1))
        self.assertEqual(self.backend.calls, [])
        self.assertIs(self.session.takes[before.id], before)
        self.assertTrue(self.session.submit_action_edit('changed', 0, 'replace'))
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        self.assertTrue(self.session.can_undo_action_edit)
        saved = encode_project(self.session.takes, before.id, 0, self.session.scene)
        self.session.load_project(saved)
        self.assertFalse(self.session.can_undo_action_edit)
        self.assertFalse(self.session.undo_action_edit())


if __name__ == '__main__': unittest.main()
