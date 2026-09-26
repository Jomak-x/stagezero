"""Static previews suspend inference without discarding source motion history."""
import unittest
import numpy as np
from live_motion import MotionSession
from directing import DirectorSession
from tests.test_controller_edges import SimulatedTransport


class CharacterMotionTests(unittest.TestCase):
    def make_session(self, cls=MotionSession):
        backend = SimulatedTransport()
        self.addCleanup(backend.release_all)
        session = cls(backend, np.zeros((20, 34, 3), dtype=np.float32),
                      np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1)))
        return session, backend

    def test_static_preview_blocks_play_and_generate_for_both_sessions(self):
        for cls in (MotionSession, DirectorSession):
            with self.subTest(cls=cls.__name__):
                session, backend = self.make_session(cls)
                session.play()
                self.assertTrue(session.playing)
                session.set_character_motion_enabled(False)
                session.play()
                self.assertFalse(session.playing)
                session.set_mode('Live ARDY')
                session.submit('A person waves.')
                self.assertFalse(session.busy)
                self.assertEqual(backend.prompts, [])
                self.assertIn('character', session.status.lower())

    def test_motion_capable_swap_preserves_playback_and_history_identity(self):
        session, _ = self.make_session()
        session.motion = np.zeros((20, 414), dtype=np.float32)
        history = session.motion
        session.frame = 7
        session.play()
        session.set_character_motion_enabled(True)
        self.assertTrue(session.playing)
        self.assertEqual(session.frame, 7)
        self.assertIs(session.motion, history)

    def test_static_transition_rejects_late_response_before_next_job(self):
        session, backend = self.make_session()
        session.set_mode('Live ARDY')
        old = backend.plan('old')
        new = backend.plan('new', marker=3)
        original = session.positions
        session.submit('old')
        self.assertTrue(old.started.wait(2))
        session.set_character_motion_enabled(False)
        self.assertFalse(session.busy)
        self.assertFalse(session.playing)
        session.set_character_motion_enabled(True)
        session.submit('new')
        old.release.set()
        self.assertTrue(new.started.wait(2))
        self.assertIs(session.positions, original)
        self.assertEqual(session.kind, 'reference')
        session.set_character_motion_enabled(False)


if __name__ == '__main__':
    unittest.main()
