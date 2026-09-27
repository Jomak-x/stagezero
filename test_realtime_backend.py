"""CPU contract and recovery checks for the isolated realtime job service."""
from __future__ import annotations

import io
import threading
import time
import unittest

import numpy as np

from realtime_backend import CoreStageAdapter, JobManager, validate_job
from interaction_runtime import validate_request


def request(request_id="job-1", **changes):
    value = {"request_id": request_id, "stage_kind": "approach", "frames": 80,
             "prompt": "Two people walk toward one another.", "actor_ids": ["a", "b"], "seed": 7}
    value.update(changes)
    return value


def identity_rotations(frames=40):
    return np.broadcast_to(np.eye(3, dtype=np.float32), (2, frames, 27, 3, 3)).copy()


class FakeRuntime:
    model = object()
    device = "cpu"

    def __init__(self):
        self.calls = []

    def generate(self, body, *, is_cancelled, deadline, on_window):
        self.calls.append(body)
        arrays = {"motion": np.zeros((2, 40, 330), dtype=np.float32),
                  "positions": np.zeros((2, 40, 27, 3), dtype=np.float32),
                  "rotations": identity_rotations()}
        for index in range(body["frames"] // 40):
            if is_cancelled(body["request_id"]):
                raise RuntimeError("cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out")
            arrays["positions"][:, :, 0, 0] = index
            on_window(index, [0, 1], arrays)
        return {}, {"model": "fake", "frames": body["frames"]}


class GatedRuntime(FakeRuntime):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate(self, body, *, is_cancelled, deadline, on_window):
        self.entered.set()
        self.release.wait(2)
        return super().generate(body, is_cancelled=is_cancelled, deadline=deadline, on_window=on_window)


def wait_status(manager, request_id, expected, timeout=2):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        state = manager.snapshot(request_id)
        if state and state["status"] == expected:
            return state
        time.sleep(.005)
    raise AssertionError(f"{request_id} did not reach {expected}: {manager.snapshot(request_id)}")


class RealtimeBackendTests(unittest.TestCase):
    def test_terrain_fields_are_explicit_and_validated(self):
        ordinary = validate_job(request())["core_request"]
        self.assertTrue(all("coordinate_frame_y" not in a for a in ordinary["actors"]))
        body = request(coordinate_frames_y={"a": 2.5}, root_targets={
            "a": [{"frame": 39, "position_xz": [0, 1], "root_height": 3.45}]})
        actors = validate_request(validate_job(body)["core_request"])["actors"]
        self.assertEqual(actors[0]["coordinate_frame_y"], 2.5)
        self.assertEqual(actors[0]["root_targets"][0]["root_height"], 3.45)
        self.assertNotIn("coordinate_frame_y", actors[1])
        for origins in ({"unknown": 1}, {"a": float("nan")}, {"a": 26}):
            with self.subTest(origins=origins), self.assertRaises(ValueError):
                validate_job(request(coordinate_frames_y=origins))
        for height in (float("inf"), -26):
            with self.subTest(height=height), self.assertRaises(ValueError):
                validate_job(request(root_targets={"a": [
                    {"frame": 39, "position_xz": [0, 1], "root_height": height}]}))

    def test_contract_limits_and_research_gate(self):
        normalized = validate_job(request(root_targets={"a": [{"frame": 39, "position_xz": [1, 2]}]}))
        self.assertEqual(normalized["core_request"]["actors"][0]["root_targets"][0]["frame"], 39)
        # The adapter passes a raw request to InteractionRuntime, which is the
        # authoritative second validation boundary before model inference.
        validate_request(normalized["core_request"])
        placed = validate_job(request(initial_placements={"a": {"position_xz": [-1.2, 0]},
                                                         "b": {"position_xz": [1.2, 0]}}))
        self.assertEqual(validate_request(placed["core_request"])["actors"][1]["initial_position_xz"], [1.2, 0.0])
        with self.assertRaisesRegex(ValueError, "disabled"):
            validate_job(request(stage_kind="paired"))
        with self.assertRaisesRegex(ValueError, "multiple"):
            validate_job(request(frames=39))
        with self.assertRaisesRegex(ValueError, "target"):
            validate_job(request(stage_kind="transition", history={"native_features": np.zeros((2, 4, 330)).tolist()}))
        target = {"positions": np.zeros((2, 40, 27, 3)).tolist(),
                  "rotations": identity_rotations().tolist()}
        job = validate_job(request(stage_kind="transition", frames=40, target=target,
                                   history={"native_features": np.zeros((2, 4, 330)).tolist()}))
        self.assertEqual(job["target"]["positions"].shape, (2, 40, 27, 3))
        with self.assertRaisesRegex(ValueError, "completed Core"):
            manager = JobManager(CoreStageAdapter(FakeRuntime()))
            manager.core.reference_provider("missing", 0, "RightHand", np.zeros(3))
        hand = {"source_job_id": "prior-core", "source_frame": 79, "hand": "RightHand",
                "position_xyz": [.2, .95, 1.18], "frames": [39, 59, 79, 99]}
        action = validate_job(request(stage_kind="continuation", actor_ids=["a"], frames=120,
                                      history={"native_features": np.zeros((1, 4, 330)).tolist()},
                                      hand_target=hand))
        self.assertEqual(action["hand_target"]["frames"], hand["frames"])
        with self.assertRaisesRegex(ValueError, "one-actor"):
            validate_job(request(stage_kind="continuation", history={"native_features": np.zeros((2, 4, 330)).tolist()},
                                 hand_target=hand))

    def test_real_core_windows_become_fetchable_chunks(self):
        runtime = FakeRuntime()
        manager = JobManager(CoreStageAdapter(runtime))
        manager.submit(request())
        state = wait_status(manager, "job-1", "complete")
        self.assertEqual(state["available_chunks"], [0, 1])
        self.assertEqual(len(runtime.calls), 1)
        status, payload = manager.chunk("job-1", 1)
        self.assertEqual(status, 200)
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            self.assertEqual(data["positions"].shape, (2, 40, 27, 3))
            self.assertEqual(data["native_features"].shape, (2, 40, 330))
            self.assertTrue(np.all(data["positions"][:, :, 0, 0] == 1))

    def test_queued_cancel_and_worker_recovery(self):
        runtime = GatedRuntime()
        manager = JobManager(CoreStageAdapter(runtime), max_queued=1)
        manager.submit(request("first", frames=40))
        self.assertTrue(runtime.entered.wait(1))
        manager.submit(request("cancelled", frames=40))
        self.assertEqual(manager.cancel("cancelled")["status"], "cancelled")
        runtime.release.set()
        wait_status(manager, "first", "complete")
        manager.submit(request("after", frames=40))
        wait_status(manager, "after", "complete")
        self.assertEqual(len(runtime.calls), 2)


if __name__ == "__main__":
    unittest.main()
