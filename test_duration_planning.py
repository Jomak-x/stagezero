"""Duration planning and transactional multi-chunk directing tests."""
import threading
import time
import unittest

import numpy as np

from directing import DirectorSession
from duration_planning import plan_duration
from takes import decode_project, encode_project
from test_live_motion import result, wait_until


class SequencedBackend:
    def __init__(self):
        self.histories = []
        self.calls = 0
        self.fail_at = None
        self.block_at = None
        self.blocked = threading.Event()
        self.release = threading.Event()

    def cancel(self, request_id):
        pass  # Test local version checks when the transport ignores cancellation.

    def generate(self, request_id, prompt, history):
        self.calls += 1
        index = self.calls
        self.histories.append(None if history is None else history.copy())
        if index == self.block_at:
            self.blocked.set()
            self.release.wait(2)
        if index == self.fail_at:
            raise RuntimeError('planned backend failure')
        return result(request_id, marker=index)


class DurationPlanningTests(unittest.TestCase):
    def test_auto_discloses_heuristic_or_explicit_prompt_seconds(self):
        self.assertEqual(plan_duration('walk').frames, 104)
        self.assertEqual(plan_duration('walk then jump').frames, 208)
        plan = plan_duration('walk for 10 seconds')
        self.assertEqual(plan.frames, 250)
        self.assertIn('stated in prompt', plan.label)
        self.assertEqual(plan_duration('walk for 90 seconds').frames, 750)
        self.assertEqual(plan_duration('walk for 0.01 seconds').label,
                         'Auto · prompt duration raised to 0.16 s minimum')

    def test_numeric_duration_quantized_and_bounded(self):
        self.assertEqual(plan_duration('walk', 10).frames, 250)
        self.assertEqual(plan_duration('walk', .17).seconds, .16)
        self.assertEqual(plan_duration('walk', 1.16 - 1.0).frames, 4)
        for invalid in (0, .15, 30.01, float('nan'), float('inf'), True, 'bogus'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                plan_duration('walk', invalid)


class DurationDirectingTests(unittest.TestCase):
    def setUp(self):
        self.backend = SequencedBackend()
        self.session = DirectorSession(self.backend, np.zeros((20, 34, 3), dtype=np.float32),
                                       np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1)))
        self.session.set_mode('Live ARDY')

    def submit_and_wait(self, prompt='walk', seconds=None, edit_mode='new', at_frame=None):
        self.session.submit(prompt, seconds=seconds, edit_mode=edit_mode, at_frame=at_frame)
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        return self.session.takes[self.session.active_take]

    def test_exact_length_and_chunk_history(self):
        take = self.submit_and_wait(seconds=10)
        self.assertEqual(len(take.positions), 250)
        self.assertEqual(self.backend.calls, 3)
        self.assertIsNone(self.backend.histories[0])
        self.assertEqual([h.shape for h in self.backend.histories[1:]], [(52, 414), (52, 414)])
        np.testing.assert_array_equal(self.backend.histories[1], np.ones((52, 414)))
        np.testing.assert_array_equal(self.backend.histories[2], np.full((52, 414), 2))
        self.assertEqual(take.segments[0]['end'], 250)
        self.assertEqual(self.session.planned_seconds, 10)

    def test_explicit_edit_modes_and_full_replacement_preserve_original(self):
        first = self.submit_and_wait(seconds=1)
        self.session.seek(0)
        extended = self.submit_and_wait('extend', 1, 'extend')
        self.assertEqual(extended.id, first.id)
        self.assertEqual(len(extended.positions), 50)
        self.assertEqual([s['end'] for s in extended.segments], [25, 50])
        original = extended
        replaced = self.submit_and_wait('replace', 1, 'replace', at_frame=0)
        self.assertNotEqual(replaced.id, original.id)
        self.assertEqual(len(replaced.positions), 25)
        self.assertIs(self.session.takes[original.id], original)
        self.assertIsNone(replaced.parent)
        self.assertIsNone(replaced.branch_frame)
        branch = self.submit_and_wait('branch', 1, 'replace', at_frame=10)
        self.assertEqual(branch.parent, replaced.id)
        self.assertEqual(branch.branch_frame, 9)
        self.assertEqual(len(branch.positions), 35)
        archive = encode_project(self.session.takes, branch.id, 0, self.session.scene)
        restored, active, _, _ = decode_project(archive)
        self.assertEqual(active, branch.id)
        self.assertEqual(len(restored[branch.id].positions), 35)
        self.assertEqual(len(restored[original.id].positions), 50)

    def test_later_chunk_failure_keeps_source_unchanged(self):
        first = self.submit_and_wait(seconds=1)
        before_revision = self.session.project_revision
        self.backend.fail_at = self.backend.calls + 2
        self.session.submit('failed extension', seconds=10, edit_mode='extend')
        wait_until(lambda: not self.session.busy)
        self.assertIs(self.session.takes[first.id], first)
        self.assertEqual(len(first.positions), 25)
        self.assertEqual(self.session.project_revision, before_revision)
        self.assertIn('Original take preserved', self.session.status)

    def test_cancel_during_later_chunk_discards_private_result(self):
        first = self.submit_and_wait(seconds=1)
        before_revision = self.session.project_revision
        self.backend.block_at = self.backend.calls + 2
        self.session.submit('cancelled extension', seconds=10, edit_mode='extend')
        self.assertTrue(self.backend.blocked.wait(1))
        self.session.seek(0)
        self.backend.release.set()
        time.sleep(.1)
        self.assertFalse(self.session.busy)
        self.assertIs(self.session.takes[first.id], first)
        self.assertEqual(len(first.positions), 25)
        self.assertEqual(self.session.project_revision, before_revision)

    def test_reject_unsupported_duration_without_starting_backend(self):
        self.session.submit('walk', seconds=31, edit_mode='new')
        self.assertFalse(self.session.busy)
        self.assertEqual(self.backend.calls, 0)
        self.assertIn('30 seconds', self.session.status)


if __name__ == '__main__':
    unittest.main()
