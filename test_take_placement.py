"""Placement moves world motion while preserving saved media and safe undo."""
import unittest

import numpy as np

from directing import DirectorSession
from takes import Take, validate_take
from test_live_motion import ControlledBackend, result


class PlacementTests(unittest.TestCase):
    def setUp(self):
        backend = ControlledBackend()
        recorded = np.zeros((20, 34, 3), dtype=np.float32)
        rotations = np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1))
        self.session = DirectorSession(backend, recorded, rotations)
        self.session.set_mode('Live ARDY')
        generated = result('test')
        self.parent = Take('parent', 'Parent', generated['positions'].copy(),
                           generated['rotations'].copy(), generated['motion'].copy(),
                           segments=[dict(start=0, end=104, prompt='walk')])
        self.child = Take('child', 'Child', generated['positions'].copy(),
                          generated['rotations'].copy(), generated['motion'].copy(),
                          segments=[dict(start=0, end=104, prompt='walk')],
                          parent='parent', branch_frame=40,
                          events=[dict(type='gate_open', frame=0)],
                          dialogue=[dict(text='Hello', voice_id='voice', audio_id='audio',
                                         start_frame=5, end_frame=20)],
                          audio_assets={'audio': b'wave'})
        self.session.takes = {'parent': self.parent, 'child': self.child}
        self.session._select('child')
        self.session.scene['gate'] = {'position': [1., 0., 1.], 'radius': .5, 'enabled': True}

    def test_move_clears_stale_gate_and_branch_link_but_keeps_voice(self):
        changed = self.session.set_start_pose(3., 4., 45.)
        validate_take(changed)
        self.assertEqual(changed.events, [])
        self.assertIsNone(changed.parent)
        self.assertIsNone(changed.branch_frame)
        self.assertEqual(changed.dialogue, self.child.dialogue)
        self.assertEqual(changed.audio_assets, self.child.audio_assets)
        np.testing.assert_allclose(changed.positions[0, 0, [0, 2]], [3., 4.], atol=1e-5)
        self.assertTrue(self.session.can_undo_start_pose)
        self.assertTrue(self.session.undo_start_pose())
        self.assertIs(self.session.takes['child'], self.child)
        self.assertEqual(self.session.takes['child'].events, [dict(type='gate_open', frame=0)])
        self.assertEqual(self.session.takes['child'].parent, 'parent')

    def test_consecutive_drag_moves_coalesce_into_one_undo(self):
        self.session.set_start_pose(2., 3., 0.)
        self.session.set_start_pose(4., 5., 90.)
        self.assertTrue(self.session.undo_start_pose())
        self.assertIs(self.session.takes['child'], self.child)
        self.assertFalse(self.session.can_undo_start_pose)

    def test_draft_move_keeps_single_reference_frame(self):
        self.session.new_take()
        self.session.set_start_pose(2., -3., 0.)
        self.assertEqual(len(self.session.positions), 1)
        self.assertEqual(self.session.get_start_pose()[:2], (2., -3.))
        self.assertTrue(self.session.undo_start_pose())
        self.assertEqual(len(self.session.positions), 1)


if __name__ == '__main__':
    unittest.main()
