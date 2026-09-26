"""CPU checks for geometry-grounded scene planning and director integration."""

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.generate_scene_showcase import (PlaybackMonitor, _service_body, _stage_specs,
                                                 place_pair, run_scene)
from realtime_backend import validate_job
from realtime_clip import CanonicalClip
from realtime_director import RealtimeDirector, StageSpec
from scene_beats import FIGHT_PAIR_PROMPT, RECOMMENDED_SEEDS, SCENARIOS, build_scene


class SceneBeatTests(unittest.TestCase):
    def test_default_seeds_use_verified_scenes_and_allow_overrides(self):
        self.assertEqual(set(RECOMMENDED_SEEDS), set(SCENARIOS))
        for name, seed in RECOMMENDED_SEEDS.items():
            self.assertEqual(build_scene(name)["seed"], seed)
            self.assertEqual(build_scene(name, 7)["seed"], 7)
        fight = build_scene("staged_fight")
        pair = [beat for beat in fight["beats"] if beat["source"] == "intergen"]
        self.assertEqual([beat["prompt"] for beat in pair], [FIGHT_PAIR_PROMPT] * 2)
        self.assertEqual(pair[0]["metadata"]["pair_sequence_id"], "fight_screen_2_42")
        self.assertEqual(fight["provenance"]["screened_pair_candidate"]["prompt_index"], 2)

    def test_all_scenes_cover_30_seconds_with_dense_native_root_goals(self):
        for name in SCENARIOS:
            with self.subTest(name=name):
                plan = build_scene(name)
                self.assertEqual(plan["total_frames"], 600)
                self.assertEqual(plan["beats"][0]["start_frame"], 0)
                self.assertEqual(plan["beats"][-1]["end_frame"], 600)
                self.assertEqual(sum(beat["frames"] for beat in plan["beats"]), 600)
                for beat in plan["beats"]:
                    self.assertEqual(beat["frames"] % 40, 0)
                    for actor in beat["actors"].values():
                        targets = actor["root_targets"]
                        if targets:
                            self.assertEqual(targets[-1]["frame"], beat["end_frame"] - 1)
                            self.assertTrue(all(target["frame"] < beat["end_frame"] for target in targets))
                            for frame in range(beat["start_frame"] + 39, beat["end_frame"], 40):
                                self.assertIn(frame, [target["frame"] for target in targets])

    def test_planner_routes_are_preserved_and_object_reach_is_native_reference_gated(self):
        gate = build_scene("gate_meet_handshake")
        route = gate["route_plans"]["visitor_gate"]
        self.assertEqual(route["verb"], "go_through")
        self.assertEqual(route["geometry"]["source"], "procedural_arch")
        self.assertEqual([w["role"] for w in route["waypoints"][-3:]],
                         ["entry", "center", "exit"])
        object_plan = build_scene("object_reach_inspect")
        self.assertEqual(object_plan["route_plans"]["console_approach"]["verb"], "approach")
        reach = next(beat for beat in object_plan["beats"] if beat["id"] == "reach_inspect")
        self.assertTrue(reach["quality_gates"][0]["native_reference_required"])
        self.assertEqual(reach["metadata"]["object_cue"], "touch_only_no_grasp")

    def test_pair_action_and_release_share_one_model_sequence(self):
        for name in ("gate_meet_handshake", "staged_fight"):
            with self.subTest(name=name):
                plan = build_scene(name)
                pair = [beat for beat in plan["beats"] if beat["source"] == "intergen"]
                self.assertEqual(len(pair), 2)
                self.assertEqual(pair[0]["kind"], "paired")
                self.assertEqual(pair[1]["kind"], "exit")
                self.assertEqual(pair[0]["prompt"], pair[1]["prompt"])
                self.assertEqual(pair[0]["metadata"]["pair_sequence_id"],
                                 pair[1]["metadata"]["pair_sequence_id"])
                self.assertEqual(pair[0]["metadata"]["source_start_frame"], 0)
                self.assertEqual(pair[1]["metadata"]["source_start_frame"], pair[0]["frames"])
                self.assertEqual(pair[1]["metadata"]["release_window"], [80, 120])
                director = RealtimeDirector(plan["actor_ids"], mode="research")
                director.queue_sequence(_stage_specs(plan))

    def test_first_service_job_keeps_initial_placement_and_local_targets(self):
        for name in SCENARIOS:
            with self.subTest(name=name):
                plan = build_scene(name)
                director = RealtimeDirector(plan["actor_ids"], mode="research" if len(plan["actor_ids"]) == 2 else "production")
                director.queue_sequence(_stage_specs(plan))
                request = director.claim_request()
                body = _service_body(plan, plan["beats"][0], request, pair=None)
                normalized = validate_job(body, pair_enabled=True)
                self.assertEqual(normalized["frames"], plan["beats"][0]["frames"])
                self.assertEqual(set(body["initial_placements"]), set(plan["actor_ids"]))
                self.assertEqual(max(t["frame"] for t in next(iter(body["root_targets"].values()))),
                                 plan["beats"][0]["frames"] - 1)

    def test_shared_pair_placement_preserves_relative_motion(self):
        positions = np.zeros((2, 40, 27, 3), dtype=np.float32)
        positions[0, ..., 0] = -1
        positions[1, ..., 0] = 1
        positions[..., 1] = 1
        rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 40, 27, 3, 3)).copy()
        pair = CanonicalClip(positions, rotations, 20, ("a", "b"), "intergen")
        placed, info = place_pair(pair, np.array([[0, -1], [0, 1]]))
        np.testing.assert_allclose(np.linalg.norm(placed.positions[0] - placed.positions[1], axis=-1),
                                   np.linalg.norm(pair.positions[0] - pair.positions[1], axis=-1), atol=1e-6)
        np.testing.assert_allclose(placed.positions[:, 0, 0][:, (0, 2)], [[0, -1], [0, 1]], atol=1e-6)
        self.assertLess(info["max_entry_root_mismatch_m"], 1e-6)
        self.assertAlmostEqual(abs(info["yaw_radians"]), math.pi / 2)

    def test_startup_wait_is_not_counted_as_playback_underrun(self):
        director = RealtimeDirector(("a",))
        director.queue_sequence((StageSpec("walk", frames=40),))
        monitor = PlaybackMonitor(director, realtime=False)
        monitor.pulse()
        self.assertIsNone(monitor.first_pose_seconds)
        self.assertEqual(monitor.buffering_events, 0)
        positions = np.ones((1, 40, 27, 3), dtype=np.float32)
        rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (1, 40, 27, 3, 3)).copy()
        clip = CanonicalClip(positions, rotations, 20, ("a",), "ardy_core")
        request = director.claim_request()
        director.complete(request.request_id, clip)
        monitor.pulse()
        self.assertEqual(monitor.finish(wait_for_playback=False)["buffer_underrun_events"], 0)

    def test_mock_service_assembles_all_frames_but_does_not_claim_fake_quality(self):
        class FakeClient:
            def health(self):
                return {"ready": True, "fake": True}

            def status(self, request_id):
                return {"request_id": request_id, "status": "complete", "metadata": {"fake": True}}

            def wait(self, body, on_chunk=None, cancelled=lambda: False):
                clips = []
                ids = tuple(body["actor_ids"])
                for index in range(body["frames"] // 40):
                    positions = np.ones((len(ids), 40, 27, 3), dtype=np.float32)
                    for actor_index, actor_id in enumerate(ids):
                        if body["stage_kind"] == "paired":
                            xz = (-.5, 0) if actor_index == 0 else (.5, 0)
                        elif body["stage_kind"] == "transition":
                            entry = body["target"]["positions"][actor_index][0][0]
                            xz = (entry[0], entry[2])
                        else:
                            goals = body.get("root_targets", {}).get(actor_id, [])
                            xz = goals[-1]["position_xz"] if goals else (0, 0)
                        positions[actor_index, :, :, 0] = xz[0]
                        positions[actor_index, :, :, 2] = xz[1]
                    rotations = np.broadcast_to(np.eye(3, dtype=np.float32),
                                                (len(ids), 40, 27, 3, 3)).copy()
                    source = "intergen" if body["stage_kind"] == "paired" else "ardy_core"
                    native = None if source == "intergen" else np.zeros((len(ids), 40, 330), dtype=np.float32)
                    clip = CanonicalClip(positions, rotations, 20, ids, source,
                                         {"request_id": body["request_id"], "chunk_index": index}, native)
                    clips.append(clip)
                    if on_chunk is not None:
                        on_chunk(clip)
                return clips

        plan = build_scene("gate_meet_handshake")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "quality gates failed"):
                run_scene(plan, FakeClient(), path, realtime=False)
            report = json.loads((path / "gate_meet_handshake.report.json").read_text())
            self.assertEqual(report["status"], "quality_failed")
            self.assertEqual(report["metrics"]["shape"], [2, 600, 27, 3])
            self.assertFalse(report["metrics"]["complete"])
            self.assertTrue((path / "gate_meet_handshake.stagezero-realtime.npz").exists())
            self.assertTrue((path / "gate_meet_handshake.pair-source.npz").exists())


if __name__ == "__main__":
    unittest.main()
