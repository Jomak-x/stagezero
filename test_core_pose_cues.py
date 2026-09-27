"""Trusted native cue contracts and transforms, without models or network."""
import copy
import math
import unittest

import numpy as np

from core_choreography import build_choreography, choreography_preset, validate_plan
from core_pose_cues import (ASSET, ASSET_SHA256, _asset_payload, _library,
    cue_targets, heading, measure_cue_result, profile_metadata)
from realtime_backend import validate_job
from realtime_client import request_body
from realtime_clip import CanonicalClip
from realtime_director import RealtimeDirector

IDS = ("actor_1", "actor_2")
PLACEMENTS = {IDS[0]: {"position_xz": [-1.2, 0.], "yaw": math.pi/2},
              IDS[1]: {"position_xz": [1.2, 0.], "yaw": -math.pi/2}}


def history():
    source = _library()[0]["victory"]
    ps, rs = [], []
    for i in range(2):
        p = np.repeat(source.positions[0, :1], 40, axis=0).copy()
        p[..., 0] += [-1., 1.][i] - p[0, 0, 0]
        p[..., 2] -= p[0, 0, 2]
        ps.append(p); rs.append(np.repeat(source.rotations[0, :1], 40, axis=0))
    return CanonicalClip(np.asarray(ps), np.asarray(rs), 20, IDS, "ardy_core", {}, np.zeros((2, 40, 330), np.float32))


class PoseCuesTests(unittest.TestCase):
    def test_pinned_asset_has_provenance_and_detects_corruption(self):
        clips, metadata = _asset_payload(ASSET.read_bytes())
        self.assertEqual(set(clips), {"kick", "duck", "victory"})
        self.assertEqual(metadata["cues"]["kick"]["source_frames"], [107, 108, 109, 110])
        for cue in clips:
            self.assertEqual(len(metadata["cues"][cue]["archive_sha256"]), 64)
            self.assertEqual(clips[cue].native_features.shape, (1, 4, 330))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            _asset_payload(ASSET.read_bytes() + b'corrupt')

    def test_strict_recipe_and_minimum_initial_spacing(self):
        plan = choreography_preset("pose_duet")
        self.assertEqual(plan["recipe"], "pose_duet_v1")
        for mutate in (lambda p: p.update(recipe="other"),
                       lambda p: p["beats"][0].update(seconds=4),
                       lambda p: p["beats"][0]["actor_prompts"].update(actor_1="Unrelated"),
                       lambda p: p["beats"][0].update(root_offsets={a: [0, 0] for a in IDS})):
            candidate = copy.deepcopy(plan); mutate(candidate)
            with self.assertRaises(ValueError): validate_plan(candidate)
        close = copy.deepcopy(PLACEMENTS); close[IDS[1]]["position_xz"][0] = .7
        with self.assertRaisesRegex(ValueError, "2.25"):
            build_choreography(plan, initial_placements=close)
        stages, report = build_choreography(plan, initial_placements=PLACEMENTS)
        self.assertEqual([s.metadata["seed"] for s in stages], [6201, 6202, 6203, 6204])
        self.assertEqual(report["frames"], 160)
        self.assertEqual(stages[1].metadata["pose_cue_profile"]["asset_sha256"], ASSET_SHA256)

    def test_swapping_prompts_swaps_model_cues(self):
        plan = choreography_preset("pose_duet")
        for beat in plan["beats"]:
            p = beat["actor_prompts"]; p[IDS[0]], p[IDS[1]] = p[IDS[1]], p[IDS[0]]
        stages, _ = build_choreography(plan, initial_placements=PLACEMENTS)
        self.assertEqual(stages[1].metadata["pose_cue_profile"]["cue_ids"], ["duck", "kick"])
        self.assertEqual(stages[2].metadata["pose_cue_profile"]["cue_ids"], ["kick", "duck"])

    def test_production_requests_are_native_history4_and_retry_seeds_are_stable(self):
        stages, _ = build_choreography(choreography_preset("pose_duet"), initial_placements=PLACEMENTS)
        director = RealtimeDirector(IDS)
        director.queue_sequence(stages)
        first = director.claim_request(); body = request_body(first); validate_job(body)
        self.assertNotIn("history", body); self.assertEqual(body["stage_kind"], "approach")
        director.complete(first.request_id, history())
        request = director.claim_request(); body = request_body(request); validate_job(body)
        self.assertEqual(np.asarray(body["history"]["native_features"]).shape, (2, 4, 330))
        self.assertEqual(request.history.frames, 40)
        self.assertEqual(body["stage_kind"], "transition")
        self.assertNotIn("root_targets", body)
        self.assertNotIn("initial_placements", body)
        self.assertEqual(body["seed"], 6202)
        director.fail(request.request_id, "test transport failure")
        director.retry()
        retry = request_body(director.claim_request())
        self.assertEqual(retry["seed"], body["seed"])
        np.testing.assert_array_equal(retry["target"]["positions"], body["target"]["positions"])

    def test_actual_cues_have_correct_world_alignment_and_preserve_height(self):
        h = history(); before = h.positions.copy()
        target, provenance = cue_targets(h, ["kick", "duck"])
        p = np.asarray(target["positions"])
        kick = p[0, -1, 21] - p[0, -1, 0]
        self.assertAlmostEqual(math.atan2(kick[0], kick[2]), math.pi/2, places=5)
        self.assertAlmostEqual(heading(p[1, -1]), -math.pi/2, places=5)
        for i, cue in enumerate(("kick", "duck")):
            np.testing.assert_array_equal(p[i, -4:, :, 1], _library()[0][cue].positions[0, :, :, 1])
            np.testing.assert_allclose(p[i, 36, 0, [0, 2]], h.positions[i, -1, 0, [0, 2]], atol=1e-6)
        np.testing.assert_array_equal(h.positions, before)
        self.assertFalse(provenance["model_outcome_verified"])
        target, _ = cue_targets(h, ["victory", "victory"])
        p = np.asarray(target["positions"])
        self.assertAlmostEqual(heading(p[0, -1]), 0., places=5)
        self.assertAlmostEqual(heading(p[1, -1]), 0., places=5)

    def test_cue_error_measurement_reports_miss_without_rejecting_output(self):
        h = history(); target, _ = cue_targets(h, ["kick", "duck"])
        p = np.asarray(target["positions"], np.float32); p[:, -4:, 10, 0] += .5
        clip = CanonicalClip(p, np.asarray(target["rotations"]), 20, IDS, "ardy_core")
        report = measure_cue_result({"stage_kind": "transition", "target": target, "actor_ids": list(IDS)}, clip)
        self.assertAlmostEqual(report["per_actor"][0]["max_joint_error_m"], .5, places=5)
        self.assertFalse(report["quality_gate_applied"])
        self.assertIsNone(measure_cue_result({"stage_kind": "continuation"}, clip))


if __name__ == '__main__': unittest.main()
