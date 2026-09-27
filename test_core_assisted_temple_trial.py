"""Offline adapter checks; no ARDY weights, CUDA, or network access required."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from interaction_runtime import validate_request
from experiments.core_assisted_temple_trial import _DirectCoreClient, _embedding_key


class FakeInteractionRuntime:
    """Validate real request schema and return bounded, distinguishable arrays."""

    def __init__(self, *, cancel_after_window=False):
        self.requests = []
        self.cancel_after_window = cancel_after_window
        self.motion = np.arange(40 * 330, dtype=np.float32).reshape(1, 40, 330)
        self.positions = np.zeros((1, 40, 27, 3), dtype=np.float32)
        self.positions[..., 0] = 1.25
        self.positions[..., 1] = 2.5
        self.positions[..., 2] = -3.75
        self.rotations = np.broadcast_to(
            np.eye(3, dtype=np.float32), (1, 40, 27, 3, 3)).copy()

    def generate(self, body, *, is_cancelled, on_window):
        self.requests.append(validate_request(body))
        on_window(0, [0], {"motion": self.motion, "positions": self.positions,
                           "rotations": self.rotations})
        if self.cancel_after_window:
            self.cancel_after_window = False
            self._cancel_event.set()
            if is_cancelled(body["request_id"]):
                raise RuntimeError("generation cancelled after complete window")
        return ({"actor_0_motion": self.motion[0],
                 "actor_0_positions": self.positions[0],
                 "actor_0_rotations": self.rotations[0]},
                {"request_id": body["request_id"], "fps": 20,
                 "frames": 40, "native_conditions": True})


class DirectCoreClientTests(unittest.TestCase):
    def make_body(self):
        return {
            "request_id": "terrain-window-1",
            "stage_kind": "approach",
            "frames": 40,
            "prompt": "A person walks naturally.",
            "actor_ids": ["walker"],
            "actor_prompts": {"walker": "A person walks forward naturally."},
            "seed": 33,
            "root_targets": {"walker": [{"frame": 24, "position_xz": [2.0, -4.0],
                                          "heading": 1.2, "root_height": 1.05}]},
            "coordinate_frames_y": {"walker": 1.5},
            "initial_placements": {"walker": {"position_xz": [1.0, 3.0], "yaw": 2.4}},
        }

    def test_exact_terrain_planner_prompts_have_cached_embedding_classes(self):
        self.assertEqual(_embedding_key("A person turns in place, keeping an upright posture."),
                         "stand")
        self.assertEqual(_embedding_key("A person walks forward naturally."), "walk")
        self.assertEqual(_embedding_key(
            "A person walks forward naturally, then slows to a relaxed stop."), "walk")

    def test_none_cancel_callback_validates_schema_and_preserves_native_arrays(self):
        runtime = FakeInteractionRuntime()
        prompts = {}
        with tempfile.TemporaryDirectory() as folder:
            client = _DirectCoreClient(runtime, Path(folder), prompts, seed=257)
            clips = client.wait(self.make_body(), cancelled=None)
            self.assertEqual(len(clips), 1)
            clip = clips[0]
            normalized = runtime.requests[0]["actors"][0]
            self.assertEqual(normalized["id"], "walker")
            self.assertEqual(normalized["prompt"], "A person walks forward naturally.")
            self.assertEqual(normalized["seed"], 257)
            self.assertEqual(normalized["initial_position_xz"], [1.0, 3.0])
            self.assertEqual(normalized["initial_yaw"], 2.4)
            self.assertEqual(normalized["coordinate_frame_y"], 1.5)
            self.assertEqual(normalized["root_targets"], [
                {"frame": 24, "position_xz": [2.0, -4.0],
                 "heading": 1.2, "root_height": 1.05}])
            self.assertEqual(prompts, {"A person walks forward naturally.": "walk"})
            np.testing.assert_array_equal(clip.native_features, runtime.motion)
            np.testing.assert_array_equal(clip.positions, runtime.positions)
            self.assertEqual(float(clip.positions[0, 0, 0, 1]), 2.5)
            self.assertEqual(client.window_index, 1)
            archive_path = Path(folder) / "native_windows" / "window-000.npz"
            with np.load(archive_path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["motion"], runtime.motion)
            sidecar = json.loads(archive_path.with_suffix(".json").read_text())
            self.assertTrue(sidecar["native_features_preserved"])

    def test_exact_native_history_is_forwarded_for_continuation(self):
        runtime = FakeInteractionRuntime()
        history = (np.arange(8 * 330, dtype=np.float32).reshape(1, 8, 330) / 7.)
        body = self.make_body()
        body.pop("initial_placements")
        body["history"] = {"native_features": history.tolist()}
        body["request_id"] = "terrain-window-2"
        with tempfile.TemporaryDirectory() as folder:
            client = _DirectCoreClient(runtime, Path(folder), {}, seed=33)
            clip, = client.wait(body, cancelled=lambda: False)
        np.testing.assert_array_equal(runtime.requests[0]["actors"][0]["history"], history[0])
        np.testing.assert_array_equal(clip.native_features, runtime.motion)

    def test_cancel_before_runtime_does_not_generate_or_write_a_window(self):
        runtime = FakeInteractionRuntime()
        event = threading.Event()
        event.set()
        with tempfile.TemporaryDirectory() as folder:
            client = _DirectCoreClient(runtime, Path(folder), {}, seed=33)
            with self.assertRaisesRegex(RuntimeError, "cancelled before"):
                client.wait(self.make_body(), cancelled=event.is_set)
            self.assertEqual(runtime.requests, [])
            self.assertFalse((Path(folder) / "native_windows").exists())

    def test_completed_window_is_checkpointed_before_later_cancellation(self):
        runtime = FakeInteractionRuntime(cancel_after_window=True)
        event = threading.Event()
        runtime._cancel_event = event
        with tempfile.TemporaryDirectory() as folder:
            client = _DirectCoreClient(runtime, Path(folder), {}, seed=33)
            with self.assertRaisesRegex(RuntimeError, "after complete window"):
                client.wait(self.make_body(), cancelled=event.is_set)
            archive_path = Path(folder) / "native_windows" / "window-000.npz"
            self.assertTrue(archive_path.is_file())
            with np.load(archive_path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["motion"], runtime.motion)
            self.assertEqual(client.window_index, 1)


if __name__ == "__main__":
    unittest.main()
