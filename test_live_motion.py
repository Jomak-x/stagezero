"""Regression tests for cancellation, pause, reset, failure and rig validation."""
import threading
import time
import unittest
import numpy as np
from live_motion import MotionSession, MODEL, validate_result


def result(request_id, marker=1):
    return {"positions": np.full((104, 34, 3), marker, dtype=np.float32),
            "rotations": np.tile(np.eye(3, dtype=np.float32), (104, 34, 1, 1)),
            "motion": np.full((104, 414), marker, dtype=np.float32),
            "metadata": {"request_id": request_id, "model": MODEL, "fps": 25, "generation_seconds": 0.1}}


class ControlledBackend:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.error = False
        self.histories = []
    def cancel(self, request_id):
        pass  # Deliberately ignore cancel, to test rejection even if the backend cannot cancel.
    def generate(self, request_id, prompt, history):
        self.histories.append(history)
        self.started.set()
        self.release.wait(2)
        if self.error:
            raise RuntimeError("Backend unavailable")
        return result(request_id, 2 if prompt == "new" else 1)


def wait_until(predicate):
    end = time.perf_counter() + 3
    while time.perf_counter() < end:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out")


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.b = ControlledBackend()
        self.s = MotionSession(self.b, np.zeros((20, 34, 3), dtype=np.float32), np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1)))
        self.s.set_mode("Live ARDY")

    def test_latest_instruction_wins(self):
        self.s.submit("old"); self.assertTrue(self.b.started.wait(1))
        self.s.submit("new"); self.b.release.set()
        wait_until(lambda: not self.s.busy)
        self.assertEqual(self.s.positions[0, 0, 0], 2)

    def test_edit_during_generation_rejects_result(self):
        self.s.submit("old"); self.assertTrue(self.b.started.wait(1))
        self.s.edit_prompt("different draft"); self.b.release.set()
        time.sleep(0.1)
        self.assertEqual(self.s.kind, "reference")
        self.assertIn("Instruction changed", self.s.status)

    def test_pause_during_generation_is_preserved(self):
        self.s.submit("old"); self.assertTrue(self.b.started.wait(1))
        self.s.pause(); self.b.release.set()
        wait_until(lambda: not self.s.busy)
        self.assertEqual(self.s.kind, "generated")
        self.assertFalse(self.s.playing)

    def test_reset_discards_pending_result(self):
        self.s.submit("old"); self.assertTrue(self.b.started.wait(1))
        self.s.reset(); self.b.release.set(); time.sleep(0.1)
        self.assertEqual(self.s.kind, "reference")
        self.assertIsNone(self.s.motion)
        self.assertFalse(self.s.playing)

    def test_fallback_discards_pending_result(self):
        self.s.submit("old"); self.assertTrue(self.b.started.wait(1))
        self.s.set_mode("Recorded preview"); self.b.release.set(); time.sleep(0.1)
        self.assertEqual(self.s.kind, "recorded")
        self.s.play(); self.assertTrue(self.s.playing)

    def test_failure_and_retry(self):
        self.b.error = True; self.b.release.set(); self.s.submit("old")
        wait_until(lambda: not self.s.busy)
        self.assertIn("Generation failed", self.s.status)
        self.b.error = False; self.s.submit("new")
        wait_until(lambda: not self.s.busy)
        self.assertEqual(self.s.kind, "generated")

    def test_history_ends_at_displayed_frame(self):
        self.b.release.set(); self.s.submit("old")
        wait_until(lambda: not self.s.busy)
        self.s.pause(); self.s.frame = 40
        self.s.submit("new")
        wait_until(lambda: not self.s.busy)
        self.assertEqual(self.b.histories[-1].shape, (40, 414))

    def test_reject_wrong_rig_and_nonfinite(self):
        r = result("id"); r["positions"] = r["positions"][:, :27]
        with self.assertRaises(ValueError): validate_result(r, "id")
        r = result("id"); r["positions"][0, 0, 0] = np.nan
        with self.assertRaises(ValueError): validate_result(r, "id")


if __name__ == "__main__": unittest.main()
