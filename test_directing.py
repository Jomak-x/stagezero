"""Offline tests use controlled synthetic transport, never presented as AI motion."""
import io
import json
import tempfile
import unittest
import numpy as np
from directing import DirectorSession
from takes import encode_project, decode_project
from test_live_motion import ControlledBackend, wait_until


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


if __name__ == '__main__': unittest.main()
