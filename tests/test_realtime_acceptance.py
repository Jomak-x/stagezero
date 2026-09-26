"""CPU counterexamples and mock-labelled protocol tests for realtime acceptance.

These tests cannot establish GPU latency or motion quality. The actual service
soak is ``experiments/verify_realtime_pipeline.py``.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realtime_clip import CanonicalClip
from realtime_director import RealtimeDirector, StageSpec
from realtime_backend import JobManager, validate_job
from scene_beats import build_scene
from experiments.generate_scene_showcase import _service_body, _stage_specs
from experiments.verify_realtime_pipeline import (
    Client, assert_chunk, collect_job, exact_disk_roundtrip, immutable_prefix,
    load_chunk, playback_clock, seam_report, validate_showcase_artifacts,
)


def synthetic_arrays(actors=1, frames=40, *, marker=0.):
    positions = np.zeros((actors, frames, 27, 3), np.float32)
    positions[..., 1] = 1
    positions[..., 0] = marker
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32),
                                (actors, frames, 27, 3, 3)).copy()
    features = np.full((actors, frames, 330), marker, np.float32)
    return {"positions": positions, "rotations": rotations, "native_features": features}


def synthetic_clip(actors=("a",), *, marker=0., frames=40):
    arrays = synthetic_arrays(len(actors), frames, marker=marker)
    return CanonicalClip.from_arrays(**arrays, actor_ids=tuple(actors), source="ardy_core")


def mock_npz(arrays, request_id="mock-1", chunk_index=0):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays,
                        metadata=np.array(json.dumps({"request_id": request_id,
                                                      "chunk_index": chunk_index,
                                                      "start_frame": chunk_index * 40,
                                                      "frames": 40, "fps": 20,
                                                      "actor_ids": ["a"],
                                                      "stage_kind": "approach",
                                                      "skeleton": "core27"})))
    return stream.getvalue()


class IndependentMetricTests(unittest.TestCase):
    def test_detects_root_joint_and_rotation_corruption(self):
        arrays = synthetic_arrays(frames=80)
        assert_chunk({name: value[:, :40] for name, value in arrays.items()}, actors=1)
        arrays["positions"][:, 40:, :, 0] = .16
        seam = seam_report(arrays["positions"], (40,))
        self.assertFalse(seam["passed"])
        self.assertGreater(seam["seams"][0]["root_step_m"], .15)
        bad = synthetic_arrays()
        bad["rotations"][0, 0, 0] *= 2
        with self.assertRaisesRegex(AssertionError, "rotations"):
            assert_chunk(bad, actors=1)

    def test_clock_counts_late_chunks_and_preserves_absolute_time(self):
        clock = playback_clock([1., 2.9, 5.2])
        self.assertEqual(clock["underruns"], 1)
        self.assertAlmostEqual(clock["max_late_s"], .2)
        with self.assertRaises(AssertionError):
            playback_clock([2., 1.])

    def test_prefix_and_exact_wire_disk_roundtrip(self):
        before = synthetic_arrays(frames=40)["positions"]
        after = np.concatenate([before, synthetic_arrays(frames=40, marker=.2)["positions"]], axis=1)
        self.assertTrue(immutable_prefix(before, after))
        after[0, 0, 0, 0] = .01
        self.assertFalse(immutable_prefix(before, after))
        blob = mock_npz(synthetic_arrays())
        with tempfile.TemporaryDirectory() as temp:
            self.assertTrue(exact_disk_roundtrip(blob, Path(temp) / "chunk.npz", 1))
        loaded, meta = load_chunk(blob, 1)
        self.assertEqual(loaded["positions"].shape, (1, 40, 27, 3))
        self.assertEqual(meta["request_id"], "mock-1")

    def test_wrong_actor_count_or_nonfinite_rejected(self):
        data = synthetic_arrays(actors=2)
        with self.assertRaises(AssertionError):
            assert_chunk(data, actors=1)
        data["native_features"][1, 0, 0] = np.nan
        with self.assertRaises(AssertionError):
            assert_chunk(data, actors=2)


class MockProtocolTests(unittest.TestCase):
    """An in-memory fake response stream tests parsing; never real inference."""

    class FakeClient(Client):
        def __init__(self, token):
            super().__init__("http://127.0.0.1:1", token, 2)
            self.blob = mock_npz(synthetic_arrays(), request_id="mock-1")

        def request(self, method, path, body=None):
            def reply(code, value):
                return code, json.dumps(value).encode()

            if self.token != "test-token":
                return reply(401, {"error": "Unauthorized"})
            if method == "GET" and path == "/health":
                return reply(200, {"ready": True, "model": "MOCK-Core"})
            if method == "POST" and path == "/v1/realtime/jobs":
                if body.get("frames") != 40:
                    return reply(400, {"error": "invalid frames"})
                return reply(202, {"request_id": body["request_id"], "status": "queued",
                                   "status_url": "/v1/realtime/jobs/" + body["request_id"]})
            if method == "GET" and path == "/v1/realtime/jobs/mock-1":
                return reply(200, {"request_id": "mock-1", "status": "completed",
                                   "produced_chunks": 1, "total_chunks": 1})
            if method == "GET" and path == "/v1/realtime/jobs/mock-1/chunks/0":
                return 200, self.blob
            if method == "DELETE":
                return reply(200, {"cancelled": True})
            return reply(404, {"error": "missing"})

    def test_mock_job_chunk_arrival_and_auth(self):
        client = self.FakeClient("test-token")
        body = {"request_id": "mock-1", "stage_kind": "approach", "frames": 40,
                "prompt": "Walk.", "actor_ids": ["a"], "seed": 1}
        job = collect_job(client, body, poll_s=.01, deadline_s=2)
        self.assertEqual(job["arrays"]["positions"].shape, (1, 40, 27, 3))
        self.assertEqual(job["clock"]["underruns"], 0)
        self.assertEqual(self.FakeClient("bad").request("GET", "/health")[0], 401)
        self.assertEqual(client.cancel("mock-1")[0], 200)


class ActualJobManagerProtocolTests(unittest.TestCase):
    """Exercise production queue and cancellation with an explicit CPU adapter."""

    def test_queued_cancel_discards_output_and_failure_recovers(self):
        started = threading.Event()
        release = threading.Event()

        class CPUAdapter:
            def generate(self, job, on_chunk, is_cancelled, deadline):
                if job["request_id"] == "lead":
                    started.set()
                    if not release.wait(2):
                        raise TimeoutError("mock lead timed out")
                if job["request_id"] == "failure":
                    raise RuntimeError("planned CPU adapter failure")
                on_chunk(0, mock_npz(synthetic_arrays(), request_id=job["request_id"]))
                return {"generator": "CPU-test-adapter"}

        manager = JobManager(CPUAdapter(), max_queued=2, job_timeout=5)

        def body(request_id):
            return {"request_id": request_id, "stage_kind": "approach", "frames": 40,
                    "prompt": "A person walks.", "actor_ids": ["a"], "seed": 1}

        try:
            manager.submit(body("lead"))
            self.assertTrue(started.wait(1))
            manager.submit(body("cancel-me"))
            self.assertEqual(manager.cancel("cancel-me")["status"], "cancelled")
            release.set()

            def await_status(request_id):
                until = time.monotonic() + 5
                while time.monotonic() < until:
                    state = manager.snapshot(request_id)
                    if state["status"] in {"complete", "failed", "cancelled"}:
                        return state
                    time.sleep(.005)
                self.fail(f"{request_id} did not reach terminal state")

            self.assertEqual(await_status("lead")["status"], "complete")
            self.assertEqual(await_status("cancel-me")["produced_chunks"], 0)
            self.assertEqual(manager.chunk("cancel-me", 0)[0], 409)
            manager.submit(body("failure"))
            self.assertEqual(await_status("failure")["status"], "failed")
            manager.submit(body("recovery"))
            self.assertEqual(await_status("recovery")["status"], "complete")
            self.assertEqual(manager.chunk("recovery", 0)[0], 200)
        finally:
            release.set()


class SceneServiceContractTests(unittest.TestCase):
    def test_all_first_beats_pair_transitions_and_native_hand_goal_validate(self):
        for name in ("gate_meet_handshake", "staged_fight", "object_reach_inspect"):
            with self.subTest(scenario=name):
                plan = build_scene(name)
                director = RealtimeDirector(plan["actor_ids"], mode="research" if name != "object_reach_inspect" else "production")
                director.queue_sequence(_stage_specs(plan))
                first = director.claim_request()
                validate_job(_service_body(plan, plan["beats"][0], first, pair=None), pair_enabled=True)
                if name != "object_reach_inspect":
                    paired = next(beat for beat in plan["beats"] if beat["kind"] == "paired")
                    meta = paired["metadata"]
                    validate_job({"request_id": f"{name}-pair", "stage_kind": "paired",
                                  "frames": meta["source_total_frames"], "prompt": paired["prompt"],
                                  "actor_ids": plan["actor_ids"], "seed": plan["seed"],
                                  "pair_sequence_id": meta["pair_sequence_id"],
                                  "source_start_frame": 0,
                                  "source_total_frames": meta["source_total_frames"]}, pair_enabled=True)
                    transition = next(beat for beat in plan["beats"] if beat["kind"] == "transition")
                    history = synthetic_clip(tuple(plan["actor_ids"]))
                    pair_arrays = synthetic_arrays(actors=2)
                    pair_clip = CanonicalClip.from_arrays(
                        pair_arrays["positions"], pair_arrays["rotations"],
                        actor_ids=tuple(plan["actor_ids"]), source="intergen")
                    request = type("Request", (), {"request_id": f"{name}-transition", "history": history})()
                    validate_job(_service_body(plan, transition, request, pair=pair_clip), pair_enabled=True)
                else:
                    action = next(beat for beat in plan["beats"] if beat["kind"] == "action")
                    action["metadata"]["native_hand_target"].update(
                        source_job_id="prior-real-job", source_frame=79)
                    request = type("Request", (), {"request_id": "object-action", "history": synthetic_clip(("inspector",))})()
                    validate_job(_service_body(plan, action, request, pair=None), pair_enabled=True)

class DirectorAcceptanceTests(unittest.TestCase):
    def test_continuous_clock_stale_interruption_and_failure_recovery(self):
        director = RealtimeDirector(("a",), target_buffer_frames=40, max_buffer_frames=80,
                                    clock=lambda: 0.)
        director.queue_sequence((StageSpec("Walk", frames=80),))
        first = director.claim_request()
        self.assertIsNotNone(first)
        self.assertTrue(director.complete(first.request_id, synthetic_clip(marker=1)))
        before = director.timeline_clip().positions.copy()
        director.play(now=0.)
        self.assertEqual(director.tick(now=.5)["frame"], 10)
        queued = director.claim_request()
        self.assertIsNotNone(queued)
        director.interrupt("Turn away")
        self.assertIn(queued.request_id, director.take_cancellations())
        self.assertFalse(director.complete(queued.request_id, synthetic_clip(marker=99)))
        np.testing.assert_array_equal(director.timeline_clip().positions, before)
        replacement = director.claim_request()
        self.assertTrue(director.fail(replacement.request_id, "mock failure"))
        self.assertEqual(director.snapshot()["phase"], "generation_failed")
        self.assertTrue(director.retry())
        retry = director.claim_request()
        self.assertTrue(director.complete(retry.request_id, synthetic_clip(marker=2)))
        after = director.timeline_clip().positions
        self.assertTrue(immutable_prefix(before, after))
        self.assertEqual(after.shape[1], 80)
        director.tick(now=10.)
        self.assertEqual(director.snapshot()["frame"], 79)
        self.assertFalse(director.playing)

    def test_two_actor_exact_save_load_arrays_and_clock(self):
        director = RealtimeDirector(("a", "b"))
        director.queue_sequence((StageSpec("Two people walk together", frames=80),))
        for marker in (1., 2.):
            request = director.claim_request()
            self.assertIsNotNone(request)
            self.assertTrue(director.complete(request.request_id,
                                              synthetic_clip(("a", "b"), marker=marker)))
        original = director.timeline_clip()
        self.assertEqual(original.positions.shape, (2, 80, 27, 3))
        loaded = RealtimeDirector.load_project(director.save_project())
        restored = loaded.timeline_clip()
        np.testing.assert_array_equal(restored.positions, original.positions)
        np.testing.assert_array_equal(restored.rotations, original.rotations)
        np.testing.assert_array_equal(restored.native_features, original.native_features)
        self.assertEqual(loaded.segments, director.segments)

    def test_saved_showcase_verifier_rejects_raw_project_tampering(self):
        name = "object_reach_inspect"
        director = RealtimeDirector(("inspector",))
        director.queue_sequence((StageSpec("Walk and inspect", frames=600),))
        director.play(now=0.)
        for index in range(15):
            request = director.claim_request()
            if request is None:
                director.tick(now=2. * (index + 1))
                request = director.claim_request()
            self.assertIsNotNone(request)
            self.assertTrue(director.complete(request.request_id, synthetic_clip(("inspector",))))
            director.tick(now=2. * (index + 1))
        clip = director.timeline_clip()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / f"{name}.stagezero-realtime.npz").write_bytes(director.save_project())
            tampered = clip.positions.copy()
            tampered[0, 0, 0, 0] = .1
            np.savez_compressed(folder / f"{name}.raw.npz", positions=tampered,
                                rotations=clip.rotations,
                                metadata=np.array(json.dumps({"fps": 20, "actor_ids": ["inspector"]})))
            (folder / f"{name}.report.json").write_text(json.dumps({
                "scenario": name, "status": "complete", "plan": build_scene(name),
                "metrics": {"complete": True}, "jobs": []}))
            report = validate_showcase_artifacts(folder)
            item = next(row for row in report["scenarios"] if row["scenario"] == name)
            self.assertFalse(item["passed"])
            self.assertIn("saved project and raw motion arrays differ", item["error"]["message"])


if __name__ == "__main__":
    unittest.main()
