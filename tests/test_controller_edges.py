"""Controller edge cases using only event-gated, simulated transport.

These synthetic arrays are contract fixtures, not ARDY output. No Backend,
HTTP connection, token, recorded asset, or inference process is used.
"""

import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from live_motion import MODEL, MotionSession


def generated_result(request_id, marker):
    motion = np.full((104, 414), marker, dtype=np.float32)
    for frame in range(104):
        motion[frame] += frame
    return {
        "positions": np.full((104, 34, 3), marker, dtype=np.float32),
        "rotations": np.tile(np.eye(3, dtype=np.float32), (104, 34, 1, 1)),
        "motion": motion,
        "metadata": {"request_id": request_id, "model": MODEL, "fps": 25,
                     "generation_seconds": 0.1},
    }


class SimulatedCall:
    def __init__(self, marker, error, transform):
        self.started = threading.Event()
        self.release = threading.Event()
        self.marker = marker
        self.error = error
        self.transform = transform
        self.request_id = None
        self.history = None


class SimulatedTransport:
    """Each response stays pending until the test releases its own gate."""

    def __init__(self):
        self.planned = {}
        self.prompts = []

    def plan(self, prompt, marker=1, error=None, transform=None):
        call = SimulatedCall(marker, error, transform)
        self.planned[prompt] = call
        return call

    def cancel(self, request_id):
        pass  # Deliberately simulate a backend that cannot cancel in-flight work.

    def generate(self, request_id, prompt, history):
        self.prompts.append(prompt)
        call = self.planned[prompt]
        call.request_id = request_id
        call.history = history
        call.started.set()
        if not call.release.wait(5):
            raise RuntimeError("Test did not release its simulated response")
        if call.error is not None:
            raise call.error
        result = generated_result(request_id, call.marker)
        return result if call.transform is None else call.transform(result)

    def release_all(self):
        for call in self.planned.values():
            call.release.set()


class ControllerEdgeTests(unittest.TestCase):
    def setUp(self):
        self.transport = SimulatedTransport()
        self.addCleanup(self.transport.release_all)
        self.session = MotionSession(
            self.transport,
            np.zeros((20, 34, 3), dtype=np.float32),
            np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1)),
        )
        self.session.set_mode("Live ARDY")

    def start(self, prompt, **kwargs):
        call = self.transport.plan(prompt, **kwargs)
        self.session.submit(prompt)
        self.assertTrue(call.started.wait(3), f"Simulated request did not start: {prompt}")
        return call

    def finish(self, call):
        call.release.set()
        # Gates determine request ordering; bounded polling only observes the
        # controller's eventual commit, after transport.generate has returned.
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with self.session.lock:
                if not self.session.busy:
                    return
            time.sleep(0.001)
        self.fail("Controller did not finish the simulated request")

    def displayed_clip(self, prompt="baseline", marker=1):
        self.finish(self.start(prompt, marker=marker))
        self.assertEqual(self.session.kind, "generated")
        # Advance playback without relying on scheduler speed or sleeping.
        with patch("live_motion.time.perf_counter", autospec=True, return_value=100.0):
            self.session.play()
        with patch("live_motion.time.perf_counter", autospec=True, return_value=101.0):
            self.assertEqual(self.session.tick()[1], 25)
            self.session.pause()
        return self.snapshot()

    def snapshot(self):
        return {
            "positions": self.session.positions.copy(),
            "rotations": self.session.rotations.copy(),
            "motion": self.session.motion.copy(),
            "frame": self.session.frame,
            "revision": self.session.clip_revision,
        }

    def assert_clip_unchanged(self, before):
        for name in ("positions", "rotations", "motion"):
            np.testing.assert_array_equal(getattr(self.session, name), before[name])
        self.assertEqual(self.session.frame, before["frame"])
        self.assertEqual(self.session.clip_revision, before["revision"])
        self.assertEqual(self.session.kind, "generated")
        self.assertFalse(self.session.playing)

    def test_burst_replacement_skips_queued_prompts_and_ignores_stale_outcomes(self):
        for label, error in (("success", None), ("failure", RuntimeError("stale failure"))):
            with self.subTest(stale_outcome=label):
                old = self.start(f"old {label}", error=error)
                revision = self.session.clip_revision
                latest = self.transport.plan(f"latest {label}", marker=9)
                self.session.submit(f"superseded one {label}")
                self.session.submit(f"superseded two {label}")
                self.session.submit(f"latest {label}")
                old.release.set()
                # Starting the next call proves the stale result/error has
                # already passed through the controller's single worker.
                self.assertTrue(latest.started.wait(3))
                self.assertTrue(self.session.busy)
                self.assertEqual(self.session.clip_revision, revision)
                self.assertIn("Generating", self.session.status)
                self.finish(latest)
                self.assertEqual(self.session.positions[0, 0, 0], 9)
                self.assertEqual(self.session.metrics["request_id"], latest.request_id)
                self.assertNotIn("failed", self.session.status)
        self.assertEqual(self.transport.prompts,
                         ["old success", "latest success", "old failure", "latest failure"])

    def test_last_pause_or_resume_intent_wins_while_pending(self):
        for final_playing in (False, True):
            with self.subTest(final_playing=final_playing):
                call = self.start(f"pause sequence {final_playing}")
                for _ in range(3):
                    self.session.pause()
                    self.session.play()
                    self.assertFalse(self.session.playing)
                    self.assertTrue(self.session.busy)
                self.session.pause()
                if final_playing:
                    self.session.play()
                self.finish(call)
                self.assertEqual(self.session.kind, "generated")
                self.assertEqual(self.session.playing, final_playing)

    def test_reset_clears_displayed_history_before_replacement_finishes(self):
        before = self.displayed_clip()
        old = self.start("continue before reset", marker=2)
        np.testing.assert_array_equal(old.history, before["motion"][2:26])
        self.session.pause()
        self.session.reset()
        self.assertFalse(self.session.busy)
        self.assertFalse(self.session.playing)
        self.assertEqual(self.session.frame, 0)
        self.assertEqual(self.session.kind, "reference")
        self.assertIsNone(self.session.motion)
        np.testing.assert_array_equal(self.session.positions, self.session.recorded[0])
        reset_revision = self.session.clip_revision

        fresh = self.transport.plan("fresh after reset", marker=3)
        self.session.submit("fresh after reset")
        old.release.set()
        self.assertTrue(fresh.started.wait(3))
        self.assertIsNone(fresh.history)
        self.assertEqual(self.session.kind, "reference")
        self.assertEqual(self.session.clip_revision, reset_revision)
        self.assertIsNone(self.session.motion)
        self.session.pause()
        self.finish(fresh)
        self.assertEqual(self.session.positions[0, 0, 0], 3)
        self.assertFalse(self.session.playing)
        self.assertEqual(self.session.metrics["request_id"], fresh.request_id)

    def test_transport_failure_preserves_displayed_clip_and_retry_history(self):
        before = self.displayed_clip()
        failed = self.start("disconnected", error=RuntimeError("simulated connection loss"))
        self.finish(failed)
        self.assertIn("Generation failed", self.session.status)
        self.assert_clip_unchanged(before)

        retry = self.start("recovered", marker=7)
        np.testing.assert_array_equal(retry.history, before["motion"][2:26])
        self.assert_clip_unchanged(before)
        self.session.pause()
        self.finish(retry)
        self.assertEqual(self.session.positions[0, 0, 0], 7)
        self.assertEqual(self.session.metrics["request_id"], retry.request_id)
        self.assertFalse(self.session.playing)
        self.assertNotIn("failed", self.session.status)

    def test_malformed_responses_preserve_scene_and_allow_recovery(self):
        cases = (
            ("missing arrays", lambda r: {"metadata": r["metadata"]}),
            ("missing request ID", lambda r: {**r, "metadata": {"model": MODEL, "fps": 25}}),
            ("foreign request", lambda r: {**r, "metadata": {**r["metadata"], "request_id": "other"}}),
            ("wrong model", lambda r: {**r, "metadata": {**r["metadata"], "model": "other"}}),
            ("wrong fps", lambda r: {**r, "metadata": {**r["metadata"], "fps": 60}}),
            ("short clip", lambda r: {**r, "positions": r["positions"][:-1]}),
            ("wrong features", lambda r: {**r, "motion": r["motion"][:, :-1]}),
            ("wrong rotation shape", lambda r: {**r, "rotations": r["rotations"][:, :-1]}),
            ("nonfinite motion", lambda r: {**r, "motion": r["motion"] * np.inf}),
            ("nonfinite rotations", lambda r: {**r, "rotations": r["rotations"] + np.inf}),
            ("scaled rotations", lambda r: {**r, "rotations": r["rotations"] * 2}),
            ("reflected rotations", lambda r: {**r, "rotations": r["rotations"] * [-1, 1, 1]}),
        )
        for label, transform in cases:
            with self.subTest(response=label):
                before = self.displayed_clip(f"baseline {label}")
                bad = self.start(f"malformed {label}", marker=5, transform=transform)
                self.finish(bad)
                self.assertIn("Generation failed", self.session.status)
                self.assert_clip_unchanged(before)
                retry = self.start(f"retry {label}", marker=8)
                np.testing.assert_array_equal(retry.history, before["motion"][2:26])
                self.finish(retry)
                self.assertEqual(self.session.positions[0, 0, 0], 8)
                self.assertEqual(self.session.metrics["request_id"], retry.request_id)
                self.assertNotIn("failed", self.session.status)

    def test_prompt_boundaries_are_checked_before_transport(self):
        for invalid in ("", " \n\t ", "x" * 501):
            with self.subTest(length=len(invalid)):
                self.session.submit(invalid)
                self.assertFalse(self.session.busy)
                self.assertIn("1–500 characters", self.session.status)
        self.assertEqual(self.transport.prompts, [])

        boundary = "x" * 500
        call = self.transport.plan(boundary)
        self.session.submit("  " + boundary + "\n")
        self.assertTrue(call.started.wait(3))
        self.assertEqual(self.transport.prompts, [boundary])
        self.finish(call)
        self.assertEqual(self.session.kind, "generated")


if __name__ == "__main__":
    unittest.main()
