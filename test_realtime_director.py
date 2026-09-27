"""CPU protocol tests for buffered real-time motion orchestration."""

from __future__ import annotations

import io
import json
import unittest

import numpy as np

from realtime_clip import CanonicalClip
from realtime_director import RealtimeDirector, StageSpec, UnsupportedAction


def clip(ids=("a", "b"), frames=40, value=0, source="ardy_core", native=True, metadata=None):
    actors = len(ids)
    positions = np.full((actors, frames, 27, 3), value, np.float32)
    positions[:, :, :, 1] = 0
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32),
                                (actors, frames, 27, 3, 3)).copy()
    features = np.full((actors, frames, 330), value, np.float32) if native else None
    return CanonicalClip(positions, rotations, 20, ids, source,
                         {"seed": int(value)} if metadata is None else metadata, features)


class ClipContractTests(unittest.TestCase):
    def test_copies_arrays_and_keeps_shared_actor_time(self):
        original = clip(value=2)
        with self.assertRaises(ValueError):
            original.positions[0, 0, 0, 0] = 99
        with self.assertRaises(ValueError):
            original.positions.setflags(write=True)
        subset = original.slice_frames(5, 11)
        self.assertEqual(subset.positions.shape, (2, 6, 27, 3))
        self.assertEqual(subset.native_features.shape, (2, 6, 330))
        self.assertEqual(subset.actor_ids, ("a", "b"))
        with self.assertRaises(TypeError):
            original.metadata["seed"] = 88

    def test_rejects_nan_wrong_shape_bad_rotation_and_ids(self):
        base = clip()
        with self.assertRaises(ValueError):
            CanonicalClip(base.positions[:, :, :26], base.rotations, 20, ("a", "b"), "ardy_core")
        bad = base.positions.copy()
        bad[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            CanonicalClip(bad, base.rotations, 20, ("a", "b"), "ardy_core")
        bad_r = base.rotations.copy()
        bad_r[0, 0, 0] *= 2
        with self.assertRaises(ValueError):
            CanonicalClip(base.positions, bad_r, 20, ("a", "b"), "ardy_core")
        with self.assertRaises(ValueError):
            CanonicalClip(base.positions, base.rotations, 20, ("a", "a"), "ardy_core")


class DirectorTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.director = RealtimeDirector(("a", "b"), clock=lambda: self.now)

    def _commit(self, value=1, source="ardy_core", native=True):
        request = self.director.claim_request()
        self.assertIsNotNone(request)
        self.assertTrue(self.director.complete(request.request_id,
                                               clip(value=value, source=source, native=native)))
        return request

    def test_buffered_playback_and_underrun_never_blocks_or_fabricates(self):
        self.director.submit_instruction("Walk together", frames=120)
        self.director.play(now=0)
        self.assertEqual(self.director.tick(now=.5)["phase"], "buffering")
        self._commit(1)
        self.assertEqual(self.director.tick(now=1.0)["frame"], 10)
        self.assertEqual(self.director.tick(now=3.0)["frame"], 39)
        self.assertEqual(self.director.snapshot()["phase"], "buffering")
        self.assertEqual(self.director.total_frames, 40)
        self._commit(2)
        self.assertEqual(self.director.tick(now=3.05)["frame"], 40)
        self.assertEqual(self.director.frame_pose()[0][0, 0, 0], 2)
        self.assertEqual(self.director.total_frames, 80)

    def test_prefetch_bounded_by_target_and_resumes_after_playhead_moves(self):
        self.director.submit_instruction("Walk", frames=200)
        self._commit(1)
        self._commit(2)
        self.assertIsNone(self.director.claim_request())
        self.director.play(now=0)
        self.director.tick(now=2.0)
        self.assertIsNotNone(self.director.claim_request())

    def test_stale_completion_after_interrupt_is_discarded(self):
        self.director.submit_instruction("Walk", frames=80)
        first = self.director.claim_request()
        self.director.interrupt("Turn away")
        self.assertEqual(self.director.take_cancellations(), (first.request_id,))
        self.assertFalse(self.director.complete(first.request_id, clip(value=99)))
        self.assertEqual(self.director.total_frames, 0)
        second = self.director.claim_request()
        self.assertNotEqual(first.epoch, second.epoch)
        self.assertEqual(second.prompt, "Turn away")
        self.assertTrue(self.director.complete(second.request_id, clip(value=3)))
        self.assertEqual(self.director.total_frames, 40)

    def test_paired_interrupt_requires_generated_release_and_keeps_prefix(self):
        director = RealtimeDirector(("a", "b"), mode="research")
        director.queue_sequence((
            StageSpec("Approach", kind="approach"),
            StageSpec("Shake hands and release", kind="paired_action", source="intergen",
                      metadata={"pair_sequence_id": "pair-1", "source_start_frame": 0,
                                "source_total_frames": 80, "seed": 7}),
            StageSpec("Shake hands and release", kind="exit", source="intergen",
                      metadata={"pair_sequence_id": "pair-1", "source_start_frame": 40,
                                "source_total_frames": 80, "seed": 7, "release_window": [40, 80]}),
        ))
        first = director.claim_request()
        director.complete(first.request_id, clip(value=1))
        second = director.claim_request()
        director.complete(second.request_id, clip(value=1.1, source="intergen", native=False,
                                                  metadata=second.metadata))
        self.assertTrue(director.snapshot()["contact_open"])
        director.interrupt("Walk to the door")
        director.seek(40)
        third = director.claim_request()
        self.assertEqual(third.stage_kind, "exit")
        self.assertEqual(third.history.frames, 40)
        self.assertAlmostEqual(third.history.positions[0, -1, 0, 0], 1.1, places=5)
        director.complete(third.request_id, clip(value=1.2, source="intergen", native=False,
                                                 metadata=third.metadata))
        self.assertFalse(director.snapshot()["contact_open"])
        director.seek(80)
        after = director.claim_request()
        self.assertEqual(after.prompt, "Walk to the door")
        self.assertEqual(after.start_frame, 120)
        self.assertEqual(director.frame_pose(0)[0][0, 0, 0], 1)
        self.assertAlmostEqual(director.frame_pose(40)[0][0, 0, 0], 1.1, places=5)

    def test_pair_without_same_sequence_exit_is_rejected(self):
        director = RealtimeDirector(("a", "b"), mode="research", target_buffer_frames=160)
        with self.assertRaises(UnsupportedAction):
            director.queue_sequence((StageSpec("Contact", kind="paired_action", source="intergen",
                                               metadata={"pair_sequence_id": "p"}),))
        self.assertEqual(director.snapshot()["queued_stages"], 0)

    def test_failure_on_exit_preserves_committed_contact(self):
        director = RealtimeDirector(("a", "b"), mode="research", target_buffer_frames=160)
        pair = {"pair_sequence_id": "p", "source_total_frames": 80, "seed": 5}
        director.queue_sequence((
            StageSpec("Contact and release", kind="paired_action", source="intergen",
                      metadata={**pair, "source_start_frame": 0}),
            StageSpec("Contact and release", kind="exit", source="intergen",
                      metadata={**pair, "source_start_frame": 40, "release_window": [40, 80]}),
        ))
        request = director.claim_request()
        director.complete(request.request_id, clip(source="intergen", native=False,
                                                   metadata=request.metadata))
        release = director.claim_request()
        director.fail(release.request_id, "cached exit unavailable")
        self.assertEqual(director.snapshot()["phase"], "generation_failed")
        self.assertEqual(director.total_frames, 40)
        self.assertTrue(director.snapshot()["contact_open"])
        self.assertIsNone(director.claim_request())
        director.retry()
        self.assertEqual(director.claim_request().stage_kind, "exit")

    def test_interrupt_mid_pair_preserves_remaining_cached_pair_and_exit(self):
        director = RealtimeDirector(("a", "b"), mode="research", target_buffer_frames=160)
        common = {"pair_sequence_id": "contact-2", "seed": 9, "source_total_frames": 120}
        director.queue_sequence((
            StageSpec("Contact then release", kind="paired_action", frames=80,
                      source="intergen", metadata={**common, "source_start_frame": 0}),
            StageSpec("Contact then release", kind="exit", source="intergen",
                      metadata={**common, "source_start_frame": 80,
                                "release_window": [80, 120]}),
            StageSpec("Old next instruction"),
        ))
        first = director.claim_request()
        self.assertEqual(first.metadata["source_start_frame"], 0)
        director.complete(first.request_id, clip(value=1, source="intergen", native=False,
                                                 metadata=first.metadata))
        second = director.claim_request()
        self.assertEqual(second.metadata["source_start_frame"], 40)
        director.interrupt("New next instruction")
        self.assertEqual(director.take_cancellations(), ())
        self.assertTrue(director.complete(second.request_id,
                                          clip(value=1.1, source="intergen", native=False,
                                               metadata=second.metadata)))
        exit_request = director.claim_request()
        self.assertEqual(exit_request.stage_kind, "exit")
        self.assertEqual(exit_request.metadata["source_start_frame"], 80)
        director.complete(exit_request.request_id,
                          clip(value=1.2, source="intergen", native=False,
                               metadata=exit_request.metadata))
        director.seek(80)
        next_request = director.claim_request()
        self.assertEqual(next_request.prompt, "New next instruction")
        self.assertEqual(next_request.start_frame, 120)
        self.assertAlmostEqual(director.frame_pose(40)[0][0, 0, 0], 1.1, places=5)
        self.assertAlmostEqual(director.frame_pose(80)[0][0, 0, 0], 1.2, places=5)

    def test_wrong_pair_slice_and_large_boundary_do_not_commit(self):
        pair = {"pair_sequence_id": "p3", "seed": 11, "source_total_frames": 80}
        for wrong_provenance in (True, False):
            with self.subTest(wrong_provenance=wrong_provenance):
                director = RealtimeDirector(("a", "b"), mode="research", target_buffer_frames=160)
                director.queue_sequence((
                    StageSpec("Shake and release", kind="paired_action", source="intergen",
                              metadata={**pair, "source_start_frame": 0}),
                    StageSpec("Shake and release", kind="exit", source="intergen",
                              metadata={**pair, "source_start_frame": 40,
                                        "release_window": [40, 80]}),
                ))
                first = director.claim_request()
                director.complete(first.request_id,
                                  clip(value=1, source="intergen", native=False,
                                       metadata=first.metadata))
                second = director.claim_request()
                metadata = dict(second.metadata)
                if wrong_provenance:
                    metadata["source_start_frame"] = 0
                with self.assertRaises(ValueError):
                    director.complete(second.request_id,
                                      clip(value=1.1 if wrong_provenance else 4,
                                           source="intergen", native=False,
                                           metadata=metadata))
                self.assertEqual(director.total_frames, 40)
                self.assertEqual(director.snapshot()["phase"], "generation_failed")
                self.assertTrue(director.snapshot()["contact_open"])

    def test_production_gate_rejects_intergen_before_queue_mutation(self):
        with self.assertRaises(UnsupportedAction):
            self.director.queue_sequence((StageSpec("Walk"),
                                          StageSpec("Shake hands", kind="paired_action", source="intergen")))
        self.assertEqual(self.director.snapshot()["queued_stages"], 0)

    def test_bad_completion_preserves_committed_prefix_and_explicit_failure(self):
        self.director.submit_instruction("Walk", frames=80)
        self._commit(1)
        request = self.director.claim_request()
        with self.assertRaises(ValueError):
            self.director.complete(request.request_id, clip(value=2, source="intergen"))
        self.assertEqual(self.director.total_frames, 40)
        self.assertEqual(self.director.snapshot()["phase"], "generation_failed")
        self.assertEqual(self.director.frame_pose(0)[0][0, 0, 0], 1)

    def test_save_load_preserves_exact_motion_provenance_and_queue(self):
        self.director.project_metadata = {"scene": {"version": 2,
                                                    "objects": [{"id": "gate", "position": [0, 1, 2]}]}}
        self.director.queue_sequence((StageSpec("Walk", frames=80, metadata={"seed": 7}),))
        request = self.director.claim_request()
        self.director.complete(request.request_id,
                               clip(value=4, metadata={"seed": 4,
                                                       "nested": {"window": [1, 2]}}))
        payload = self.director.save_project()
        restored = RealtimeDirector.load_project(payload)
        self.assertEqual(restored.total_frames, 40)
        self.assertEqual(restored.segments[0]["metadata"], {"seed": 7})
        self.assertEqual(restored.segments[0]["clip_metadata"]["nested"]["window"], [1, 2])
        self.assertEqual(restored.project_metadata, self.director.project_metadata)
        self.assertEqual(restored.snapshot()["queued_stages"], 1)
        self.assertEqual(restored.claim_request().start_frame, 40)
        self.assertTrue(np.array_equal(restored.frame_pose(0)[0], self.director.frame_pose(0)[0]))

    def test_project_metadata_rejects_nonfinite_on_save(self):
        self.director.submit_instruction("Walk")
        self._commit()
        self.director.project_metadata = {"scene": {"bad": float("nan")}}
        with self.assertRaisesRegex(ValueError, "finite JSON"):
            self.director.save_project()

    def test_load_rejects_untrusted_object_array_without_pickle(self):
        payload = self.director
        payload.submit_instruction("Walk")
        self._commit()
        good = payload.save_project()
        with np.load(io.BytesIO(good), allow_pickle=False) as data:
            fields = {name: data[name] for name in data.files}
        fields["c0_positions"] = np.array([{"bad": True}], dtype=object)
        bad = io.BytesIO()
        np.savez_compressed(bad, **fields)
        with self.assertRaises(ValueError):
            RealtimeDirector.load_project(bad.getvalue())

    def test_save_load_mid_contact_preserves_required_cached_exit(self):
        director = RealtimeDirector(("a", "b"), mode="research")
        common = {"pair_sequence_id": "persisted", "seed": 3,
                  "source_total_frames": 80}
        director.queue_sequence((
            StageSpec("Meet then release", kind="paired_action", source="intergen",
                      metadata={**common, "source_start_frame": 0}),
            StageSpec("Meet then release", kind="exit", source="intergen",
                      metadata={**common, "source_start_frame": 40,
                                "release_window": [40, 80]}),
        ))
        request = director.claim_request()
        director.complete(request.request_id,
                          clip(source="intergen", native=False,
                               metadata=request.metadata))
        loaded = RealtimeDirector.load_project(director.save_project())
        self.assertTrue(loaded.snapshot()["contact_open"])
        exit_request = loaded.claim_request()
        self.assertEqual(exit_request.stage_kind, "exit")
        self.assertEqual(exit_request.metadata["pair_sequence_id"], "persisted")


if __name__ == "__main__":
    unittest.main()
