"""Contract and lifecycle checks; fake poses are not motion-quality evidence."""
import copy
import json
import unittest

import numpy as np

from core_choreography import PRESETS, build_choreography, choreography_preset, validate_plan
from realtime_backend import validate_job
from realtime_client import request_body
from realtime_clip import MAX_CLIP_FRAMES
from realtime_director import RealtimeDirector
from studio_core_session import CoreStudioSession
from test_studio_core_session import FakeClient, clip, wait_for

IDS = ("actor_1", "actor_2")
PLACEMENTS = {IDS[0]: {"position_xz": [-1.2, 0], "yaw": 0},
              IDS[1]: {"position_xz": [1.2, 0], "yaw": 0}}


def short_plan():
    plan = choreography_preset("feint_dodge")
    plan["beats"] = plan["beats"][:1]
    return plan


class PlanValidationTests(unittest.TestCase):
    def test_presets_validate_compile_and_detach_caller_data(self):
        for name in PRESETS:
            with self.subTest(name=name):
                plan = choreography_preset(name)
                original = copy.deepcopy(plan)
                stages, report = build_choreography(plan, initial_placements=PLACEMENTS)
                self.assertEqual(report["frames"], 160 if name == "pose_duet" else 240)
                self.assertEqual(len(stages), 4 if name == "pose_duet" else 6)
                self.assertEqual(plan, original)
                self.assertTrue(all(s.frames == 40 and s.source == "ardy_core" for s in stages))
                self.assertFalse(report["physical_contact_verified"])
                plan["beats"][0]["actor_prompts"][IDS[0]] = "Mutated"
                self.assertNotEqual(stages[0].actor_prompts[IDS[0]], "Mutated")
                self.assertNotEqual(report["plan"]["beats"][0]["actor_prompts"][IDS[0]], "Mutated")

    def test_untrusted_plan_rejects_unknown_fields_partial_roles_and_bad_types(self):
        mutations = [
            lambda p: p.update(source="intergen"),
            lambda p: p.update(seed=True),
            lambda p: p.update(seed=2**32),
            lambda p: p.update(version=True),
            lambda p: p["beats"][0].update(seconds=2.),
            lambda p: p["beats"][0].update(seconds=3),
            lambda p: p["beats"][0].update(contact={}),
            lambda p: p["beats"][0]["actor_prompts"].pop(IDS[1]),
            lambda p: p["beats"][0]["actor_prompts"].update(actor_1="x" * 501),
            lambda p: p["beats"][0].update(headings={IDS[0]: 0, IDS[1]: 0}),
            lambda p: p["beats"][0].update(root_offsets={IDS[0]: [0, 0]}),
            lambda p: p["beats"][0].update(root_offsets={IDS[0]: [float("nan"), 0], IDS[1]: [0, 0]}),
        ]
        for mutate in mutations:
            p = short_plan()
            mutate(p)
            with self.subTest(plan=p), self.assertRaises(ValueError):
                validate_plan(p)
        with self.assertRaises(ValueError):
            validate_plan(short_plan(), ("one",))

    def test_total_duration_and_duplicate_beats_are_bounded(self):
        p = short_plan()
        p["beats"] *= 2
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_plan(p)
        p["beats"] = [{**p["beats"][0], "name": str(i), "seconds": 4} for i in range(5)]
        with self.assertRaisesRegex(ValueError, "16 seconds"):
            validate_plan(p)

    def test_four_second_spatial_beat_splits_dense_goals_without_repeating_first_horizon(self):
        p = short_plan()
        p["beats"][0].update(seconds=4, root_offsets={a: [0, .8] for a in IDS}, headings={a: .3 for a in IDS})
        stages, report = build_choreography(p, initial_placements=PLACEMENTS)
        self.assertEqual(len(stages), 2)
        for index, stage in enumerate(stages):
            targets = stage.metadata["root_targets"][IDS[0]]
            self.assertEqual([x["frame"] for x in targets], [7, 15, 23, 31, 39])
            self.assertAlmostEqual(targets[-1]["position_xz"][1], .4 * (index + 1))
            self.assertEqual(targets[-1]["heading"], .3)
        self.assertEqual(report["beats"][0]["end_frame"], 80)
        director = RealtimeDirector(IDS)
        director.queue_sequence(stages)
        request = director.claim_request()
        validate_job(request_body(request))
        self.assertTrue(director.complete(request.request_id, clip(IDS)))
        second = request_body(director.claim_request())
        validate_job(second)
        self.assertIn("native_features", second["history"])
        self.assertNotIn("initial_placements", second)
        self.assertAlmostEqual(second["root_targets"][IDS[0]][-1]["position_xz"][1], .8)

    def test_continuing_roots_override_stale_placements_without_replacement(self):
        prior = clip(IDS, value=4)
        p = short_plan()
        p["beats"][0]["root_offsets"] = {a: [0, 0] for a in IDS}
        stages, report = build_choreography(p, initial_placements=PLACEMENTS, last_clip=prior)
        self.assertEqual(report["origins_xz"], {IDS[0]: [4., 0.], IDS[1]: [7., 0.]})
        self.assertNotIn("initial_placements", stages[0].metadata)
        self.assertEqual(stages[0].metadata["root_targets"][IDS[0]][-1]["position_xz"], [4., 0.])

    def test_crossing_pair_paths_and_worker_bounds_reject_before_generation(self):
        p = short_plan()
        p["beats"][0]["root_offsets"] = {IDS[0]: [2, 0], IDS[1]: [-2, 0]}
        with self.assertRaisesRegex(ValueError, "separation"):
            build_choreography(p, initial_placements=PLACEMENTS)
        p["beats"][0]["root_offsets"] = {a: [1, 0] for a in IDS}
        places = copy.deepcopy(PLACEMENTS)
        places[IDS[1]]["position_xz"] = [24.5, 0]
        with self.assertRaisesRegex(ValueError, "25"):
            build_choreography(p, initial_placements=places)


class SessionChoreographyTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.session = CoreStudioSession(self.client)
        self.session.start(2, placements=PLACEMENTS)
        self.addCleanup(self.session.close)
        self.addCleanup(self.client.release.set)

    def test_invalid_plan_preserves_pending_request_and_queue(self):
        self.client.release.clear()
        self.session.choreograph(choreography_preset("dance_response"))
        self.assertTrue(self.client.started.wait(1))
        before = self.session.snapshot()
        p = short_plan()
        p["beats"][0]["actor_prompts"].pop(IDS[1])
        with self.assertRaises(ValueError):
            self.session.choreograph(p)
        after = self.session.snapshot()
        self.assertEqual(before["inflight_request_id"], after["inflight_request_id"])
        self.assertEqual(before["queued_stages"], after["queued_stages"])
        self.assertEqual(self.client.active, 1)

    def test_capacity_rejection_preserves_existing_queue(self):
        # Freeze executor to exercise the boundary without 15,000 fake frames.
        self.session._run_generation = lambda: None
        self.session.choreograph(short_plan())
        self.session._director._frames = MAX_CLIP_FRAMES - 20
        before = self.session.snapshot()["queued_stages"]
        with self.assertRaisesRegex(ValueError, "15000"):
            self.session.choreograph(short_plan())
        self.assertEqual(before, self.session.snapshot()["queued_stages"])
        self.session._director._frames = 0

    def test_spatial_queue_cannot_anchor_to_unknown_future_roots(self):
        self.client.release.clear()
        self.session.choreograph(short_plan())
        self.assertTrue(self.client.started.wait(1))
        p = short_plan()
        p["beats"][0]["root_offsets"] = {a: [0, 0] for a in IDS}
        before = self.session.snapshot()
        with self.assertRaisesRegex(ValueError, "known committed roots"):
            self.session.choreograph(p, queued=True)
        self.assertEqual(before["inflight_request_id"], self.session.snapshot()["inflight_request_id"])

    def test_shared_beats_keep_native_prefix_and_roundtrip_plan(self):
        self.session.direct({a: "Stand" for a in IDS})
        wait_for(lambda: self.session.snapshot()["total_frames"] == 40)
        prefix = self.session.timeline_clip()
        report = self.session.choreograph(choreography_preset("dance_response"))
        def advance():
            total = self.session.snapshot()["total_frames"]
            if total:
                self.session.seek(total - 1)
            return self.session.snapshot()["queued_stages"] == 0
        wait_for(advance)
        result = self.session.timeline_clip()
        self.assertEqual(result.frames, 280)
        np.testing.assert_array_equal(prefix.native_features, result.native_features[:, :40])
        np.testing.assert_array_equal(prefix.positions, result.positions[:, :40])
        for request in self.client.calls[1:]:
            self.assertIn("native_features", request["history"])
            self.assertNotIn("initial_placements", request)
        self.assertEqual(self.client.calls[3]["actor_prompts"], report["plan"]["beats"][1]["actor_prompts"])
        with CoreStudioSession() as loaded:
            loaded.load(self.session.save())
            np.testing.assert_array_equal(loaded.timeline_clip().native_features, result.native_features)
            self.assertEqual(loaded._director.project_metadata["studio_core"]["last_choreography"]["plan"], report["plan"])
            self.assertEqual(loaded._director.segments[1]["metadata"]["choreography_plan"], report["plan"])

    def test_cancel_retains_committed_pair_frames_and_drops_remaining_beats(self):
        self.session.choreograph(choreography_preset("dance_response"))
        wait_for(lambda: self.session.snapshot()["total_frames"] == 80)
        prefix = self.session.timeline_clip().native_features.copy()
        self.session.cancel()
        self.assertEqual(self.session.snapshot()["queued_stages"], 0)
        np.testing.assert_array_equal(prefix, self.session.timeline_clip().native_features)


if __name__ == "__main__":
    unittest.main()
