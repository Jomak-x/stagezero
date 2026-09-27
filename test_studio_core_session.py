"""Native controller lifecycle and exact motion guarantees, without inference."""
import io
import json
import threading
import time
import unittest

import numpy as np

from realtime_backend import validate_job
from realtime_clip import CanonicalClip
from realtime_director import RealtimeDirector, StageSpec
from scene_objects import make_object
from studio_core_session import CoreStudioSession, EMPTY_SCENE
from pathlib import Path


def clip(ids=("actor_1",), value=1., *, collide=False, source="ardy_core", native=True):
    p = np.zeros((len(ids), 40, 27, 3), dtype=np.float32)
    p[..., 1] = 1.
    for i in range(len(ids)):
        p[i, ..., 0] = value + (0 if collide else i * 3.)
    r = np.broadcast_to(np.eye(3, dtype=np.float32), (len(ids), 40, 27, 3, 3))
    f = np.full((len(ids), 40, 330), value, np.float32) if native else None
    return CanonicalClip(p, r, 20, ids, source, {}, f)


class FakeClient:
    def __init__(self):
        self.calls = []
        self.errors = 0
        self.collide = False
        self.jump = False
        self.release = threading.Event()
        self.release.set()
        self.started = threading.Event()
        self.ignore_cancel = False
        self.active = 0
        self.max_active = 0
        self.ready = True

    def health(self):
        return {"ready": self.ready, "model": "ARDY-Core-RP-20FPS-Horizon40"}

    def wait(self, body, *, cancelled):
        validate_job(body)  # Use the real HTTP request contract, including native history.
        self.calls.append(body)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            while not self.release.wait(.005):
                if cancelled() and not self.ignore_cancel:
                    raise RuntimeError("cancelled")
            if self.errors:
                self.errors -= 1
                raise RuntimeError("offline test failure")
            return [clip(tuple(body["actor_ids"]), value=(10. if self.jump else 1. + .1 * (len(self.calls) - 1)), collide=self.collide)]
        finally:
            self.active -= 1


def wait_for(predicate, timeout=3.):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("condition did not complete")


class CoreStudioTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.session = CoreStudioSession(self.client)
        self.addCleanup(self.session.close)
        self.addCleanup(self.client.release.set)

    def start(self, actors=1, scene=None):
        self.session.start(actors, EMPTY_SCENE if scene is None else scene)

    def direct(self, seconds=2, **kwargs):
        ids = self.session.snapshot()["actor_ids"]
        return self.session.direct({aid: "Wave in place" for aid in ids}, seconds, **kwargs)

    def test_explicit_activation_no_network_and_offline_playback(self):
        self.assertFalse(self.session.active)
        self.assertFalse(self.session.snapshot()["initialized"])
        with self.assertRaisesRegex(RuntimeError, "Activate"):
            self.direct()
        self.start()
        self.session.tick()
        self.session.activate()
        self.assertEqual(self.client.calls, [])
        self.assertIsNone(self.session._thread)
        with CoreStudioSession() as offline:
            offline.start()
            with self.assertRaisesRegex(RuntimeError, "not configured"):
                offline.direct({"actor_1": "Walk"})
            self.assertIsNone(offline._thread)

    def test_city_encounter_validates_before_replacing_existing_motion(self):
        self.start()
        self.direct()
        wait_for(lambda: self.session.snapshot()["total_frames"] == 40)
        before = self.session.timeline_clip().positions.copy()
        with self.assertRaisesRegex(ValueError, "route_variant"):
            self.session.run_city_encounter(EMPTY_SCENE, route_variant="unsafe")
        np.testing.assert_array_equal(before, self.session.timeline_clip().positions)

        city = json.loads((Path(__file__).parent / "examples/scenes/city-boulevard.json").read_text())
        self.client.ready = False
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            self.session.run_city_encounter(city)
        np.testing.assert_array_equal(before, self.session.timeline_clip().positions)
        self.client.ready = True
        report = self.session.run_city_encounter(city, route_variant="west", timing_variant="brisk", seed=42)
        self.assertEqual(report["contact_mode"], "staged_no_contact")
        self.assertEqual(self.session.snapshot()["actor_ids"], ("actor_1", "actor_2"))
        self.assertIn("Staged, no-contact", self.session.snapshot()["status"])
        self.assertEqual(self.session._director.project_metadata["studio_core"]["city_encounter"], report)

    def test_bounded_lookahead_and_request_schema_with_native_prefix(self):
        self.start(2)
        self.direct(12)
        wait_for(lambda: self.session.snapshot()["total_frames"] == 80)
        self.session.pause()
        prefix = self.session.timeline_clip().positions.copy()
        self.assertEqual(len(self.client.calls), 2)
        self.assertEqual(self.client.calls[0]["actor_prompts"]["actor_2"], "Wave in place")
        self.assertIn("native_features", self.client.calls[1]["history"])
        self.assertNotIn("initial_placements", self.client.calls[1])
        self.session.seek(79)
        wait_for(lambda: self.session.snapshot()["total_frames"] == 160)
        np.testing.assert_array_equal(prefix, self.session.timeline_clip().positions[:, :80])
        self.assertEqual(self.client.max_active, 1)
        self.assertEqual(self.session.snapshot()["fps"], 20)
        self.assertEqual(len(self.session.snapshot()["segments"]), 4)

    def test_requested_action_durations_finish_exactly(self):
        for seconds in (2, 6, 12):
            self.start()
            self.direct(seconds)
            def advance():
                total = self.session.snapshot()["total_frames"]
                if total:
                    self.session.seek(total - 1)
                return self.session.snapshot()["queued_stages"] == 0
            wait_for(advance)
            self.assertEqual(self.session.snapshot()["total_frames"], seconds * 20)
        self.assertEqual(self.client.max_active, 1)

    def test_failure_retry_keeps_last_good_prefix(self):
        self.start()
        self.direct()
        wait_for(lambda: self.session.snapshot()["total_frames"] == 40)
        before = self.session.timeline_clip().positions.copy()
        self.client.errors = 1
        self.direct()
        wait_for(lambda: self.session.snapshot()["failure"] is not None)
        np.testing.assert_array_equal(before, self.session.timeline_clip().positions)
        self.assertTrue(self.session.retry())
        wait_for(lambda: self.session.snapshot()["total_frames"] == 80)
        np.testing.assert_array_equal(before, self.session.timeline_clip().positions[:, :40])
        self.assertIsNone(self.session.snapshot()["failure"])

    def test_transport_reconfiguration_retains_failed_intent_for_retry(self):
        self.start()
        self.direct()
        wait_for(lambda: self.session.snapshot()["total_frames"] == 40)
        before = self.session.timeline_clip().positions.copy()
        self.client.errors = 1
        self.direct()
        wait_for(lambda: self.session.snapshot()["failure"] is not None)
        repaired = FakeClient()
        self.session.configure_client(repaired)
        self.assertTrue(self.session.retry())
        wait_for(lambda: self.session.snapshot()["total_frames"] == 80)
        self.assertEqual(repaired.calls[0]["actor_prompts"], {"actor_1": "Wave in place"})
        np.testing.assert_array_equal(before, self.session.timeline_clip().positions[:, :40])

    def test_mode_switch_ignores_late_result_and_drops_pending(self):
        self.start()
        self.client.release.clear()
        self.client.ignore_cancel = True
        self.direct(12)
        self.assertTrue(self.client.started.wait(1))
        self.session.deactivate()
        self.client.release.set()
        wait_for(lambda: self.client.active == 0)
        self.assertEqual(self.session.snapshot()["total_frames"], 0)
        self.assertEqual(self.session.snapshot()["queued_stages"], 0)
        self.session.activate()
        self.assertEqual(len(self.client.calls), 1)

    def test_scene_change_invalidates_pending_but_identical_scene_does_not(self):
        self.start()
        self.client.release.clear()
        self.client.ignore_cancel = True
        self.direct()
        self.assertTrue(self.client.started.wait(1))
        self.assertFalse(self.session.update_scene(dict(EMPTY_SCENE)))
        epoch = self.session.snapshot()["epoch"]
        changed = {**EMPTY_SCENE, "name": "Changed"}
        self.assertTrue(self.session.update_scene(changed))
        self.assertGreater(self.session.snapshot()["epoch"], epoch)
        self.client.release.set()
        wait_for(lambda: self.client.active == 0)
        self.assertEqual(self.session.snapshot()["total_frames"], 0)

    def test_scene_change_notice_persists_across_append_and_exact_archive(self):
        self.start()
        self.direct()
        wait_for(lambda: self.session.snapshot()["total_frames"] == 40)
        wall = make_object("wall", 0)
        wall.update(position=[10., 1., 0.], size=[.4, 2., 2.])
        changed = {**EMPTY_SCENE, "objects": [wall]}
        self.session.update_scene(changed)
        self.assertTrue(self.session.snapshot()["scene_changed_since_motion"])
        self.assertIn("Stored motion was created against an earlier layout", self.session.snapshot()["status"])
        self.direct()
        wait_for(lambda: self.session.snapshot()["total_frames"] == 80)
        self.assertTrue(self.session.snapshot()["scene_changed_since_motion"])
        with CoreStudioSession() as restored:
            restored.load(self.session.save())
            self.assertTrue(restored.snapshot()["scene_changed_since_motion"])
            self.assertIn("earlier layout", restored.snapshot()["status"])
            restored.reset(scene_document=changed)
            self.assertFalse(restored.snapshot()["scene_changed_since_motion"])
        self.session.load_example()
        self.assertFalse(self.session.snapshot()["scene_changed_since_motion"])

    def test_scene_edits_before_motion_and_non_geometry_changes_do_not_flag_stored_motion(self):
        self.start()
        wall = make_object("wall", 0)
        wall.update(position=[10., 1., 0.], size=[.4, 2., 2.])
        changed = {**EMPTY_SCENE, "objects": [wall]}
        self.session.update_scene(changed)
        self.direct()
        wait_for(lambda: self.session.snapshot()["total_frames"] == 40)
        self.assertFalse(self.session.snapshot()["scene_changed_since_motion"])
        self.session.update_scene({**changed, "name": "Renamed only"})
        self.assertFalse(self.session.snapshot()["scene_changed_since_motion"])
        with CoreStudioSession() as restored:
            restored.load(self.session.save())
            self.assertFalse(restored.snapshot()["scene_changed_since_motion"])

    def test_redirect_cancels_only_uncommitted_intent(self):
        self.start()
        self.direct()
        wait_for(lambda: self.session.snapshot()["total_frames"] == 40)
        before = self.session.timeline_clip().positions.copy()
        self.client.release.clear()
        self.client.started.clear()
        self.direct(6)
        self.assertTrue(self.client.started.wait(1))
        self.session.direct({"actor_1": "Turn"}, 2)
        self.client.release.set()
        wait_for(lambda: self.session.snapshot()["total_frames"] == 80)
        np.testing.assert_array_equal(before, self.session.timeline_clip().positions[:, :40])
        self.assertEqual(self.client.calls[-1]["actor_prompts"], {"actor_1": "Turn"})
        self.assertNotIn("initial_placements", self.client.calls[-1])
        self.assertEqual(self.session.snapshot()["queued_stages"], 0)

    def test_save_load_exact_native_arrays_scene_and_no_auto_inference(self):
        self.start()
        self.direct(6)
        wait_for(lambda: self.session.snapshot()["total_frames"] == 80)
        original = self.session.timeline_clip()
        self.session.pause()
        self.session.seek(17)
        content = self.session.save_project()
        with CoreStudioSession(self.client) as loaded:
            loaded.load_project(content)
            self.assertIsNone(loaded._thread)
            self.assertFalse(loaded.snapshot()["playing"])
            self.assertEqual(loaded.snapshot()["frame"], 17)
            self.assertEqual(loaded.scene_document, EMPTY_SCENE)
            copied = loaded.timeline_clip()
            np.testing.assert_array_equal(original.positions, copied.positions)
            np.testing.assert_array_equal(original.rotations, copied.rotations)
            np.testing.assert_array_equal(original.native_features, copied.native_features)
            metadata = loaded._director.project_metadata["studio_core"]
            self.assertEqual(metadata["original_scene_document"], EMPTY_SCENE)
            loaded.play()
            self.assertIsNone(loaded._thread)

    def test_archive_preserves_per_actor_prompt_provenance_and_readable_summary(self):
        self.start(2)
        prompts = {"actor_1": "Wave with your left hand.", "actor_2": "Turn your head to the right."}
        self.session.direct(prompts, 2)
        wait_for(lambda: self.session.snapshot()["total_frames"] == 40)
        with CoreStudioSession() as restored:
            restored.load(self.session.save())
            segment = restored.snapshot()["segments"][0]
            self.assertEqual(segment["metadata"]["actor_prompts"], prompts)
            self.assertIn(prompts["actor_1"], segment["prompt"])
            self.assertIn(prompts["actor_2"], segment["prompt"])
        self.start()
        self.session.direct({"actor_1": "Slowly raise both arms."}, 2)
        wait_for(lambda: self.session.snapshot()["total_frames"] == 40)
        self.assertEqual(self.session.snapshot()["segments"][0]["prompt"], "Slowly raise both arms.")

    def test_queue_and_duration_validation_does_not_discard_good_pending_work(self):
        self.start()
        self.client.release.clear()
        self.direct(6)
        self.assertTrue(self.client.started.wait(1))
        self.direct(2, queued=True)
        self.assertEqual(self.session.snapshot()["queued_stages"], 2)
        for seconds in (True, 3, -2, float("nan")):
            with self.assertRaises(ValueError):
                self.direct(seconds)
        with self.assertRaises(ValueError):
            self.session.direct({"actor_2": "Wrong cast"}, 2)
        self.assertEqual(self.session.snapshot()["queued_stages"], 2)

    def test_collision_failure_preserves_previous_good_timeline(self):
        self.start(2)
        self.direct()
        wait_for(lambda: self.session.snapshot()["total_frames"] == 40)
        prefix = self.session.timeline_clip().positions.copy()
        self.client.collide = True
        self.direct()
        wait_for(lambda: self.session.snapshot()["failure"] is not None)
        self.assertIn("root separation proxy", self.session.snapshot()["failure"])
        np.testing.assert_array_equal(prefix, self.session.timeline_clip().positions)

    def test_discontinuous_continuation_preserves_previous_good_timeline(self):
        self.start()
        self.direct()
        wait_for(lambda: self.session.snapshot()["total_frames"] == 40)
        prefix = self.session.timeline_clip().positions.copy()
        self.client.jump = True
        self.direct()
        wait_for(lambda: self.session.snapshot()["failure"] is not None)
        self.assertIn("root discontinuity", self.session.snapshot()["failure"])
        np.testing.assert_array_equal(prefix, self.session.timeline_clip().positions)

    def test_free_motion_leaving_authored_platform_is_rejected_before_commit(self):
        floor = make_object("platform", 0)
        floor.update(position=[0., -.05, 0.], size=[3., .1, 3.])
        self.start(scene={**EMPTY_SCENE, "objects": [floor]})
        self.direct()
        wait_for(lambda: self.session.snapshot()["total_frames"] == 40)
        prefix = self.session.timeline_clip().positions.copy()
        self.client.jump = True
        self.direct()
        wait_for(lambda: self.session.snapshot()["failure"] is not None)
        self.assertEqual(self.session.snapshot()["total_frames"], 40)
        np.testing.assert_array_equal(prefix, self.session.timeline_clip().positions)
        self.assertIn("authored floors", self.session.snapshot()["geometry_check"])

    def test_scene_solid_proxy_rejects_overlapping_chunk(self):
        wall = make_object("wall", 0)
        wall.update(position=[1., 1., 0.], size=[.4, 2., 2.])
        self.start(scene={**EMPTY_SCENE, "objects": [wall]})
        self.direct()
        wait_for(lambda: self.session.snapshot()["failure"] is not None)
        self.assertEqual(self.session.snapshot()["total_frames"], 0)
        self.assertIn("scene solid proxies", self.session.snapshot()["failure"])

    def test_navigation_real_schema_and_invalid_target_preserves_queue(self):
        arch = make_object("arch", 0)
        arch.update(position=[0., 1.5, 3.], size=[3., 3., .4])
        self.start(scene={**EMPTY_SCENE, "objects": [arch]})
        self.client.release.clear()
        route = self.session.navigate("actor_1", arch["id"], "go_through")
        self.assertTrue(self.client.started.wait(1))
        self.assertGreater(route["schedule"]["frames"], 40)
        self.assertIn("root_targets", self.client.calls[0])
        pending = self.session.snapshot()["queued_stages"]
        with self.assertRaises(ValueError):
            self.session.navigate("actor_1", "missing", "approach")
        self.assertEqual(self.session.snapshot()["queued_stages"], pending)

    def test_closed_door_cannot_be_used_as_a_passage(self):
        door = make_object("door", 0)
        door.update(position=[0., 1.2, 3.], size=[2., 2.4, .25])
        self.start(scene={**EMPTY_SCENE, "objects": [door]})
        with self.assertRaises(ValueError):
            self.session.navigate("actor_1", door["id"], "go_through")
        self.assertEqual(self.client.calls, [])
        self.assertIsNone(self.session._thread)

    def test_research_and_non_studio_archives_rejected_without_replacing_take(self):
        self.session.load_example()
        previous = self.session.timeline_clip().positions.copy()
        good = self.session.save()
        for mode, source in (("research", "ardy_core"), ("production", "intergen")):
            with np.load(io.BytesIO(good), allow_pickle=False) as z:
                values = {key: z[key] for key in z.files}
            manifest = json.loads(values["manifest"].item())
            manifest["mode"] = mode
            if source == "intergen":
                for c, seg in zip(manifest["chunks"], manifest["segments"]):
                    c["source"] = seg["source"] = source
            values["manifest"] = np.array(json.dumps(manifest))
            out = io.BytesIO(); np.savez_compressed(out, **values)
            with self.assertRaises(ValueError):
                self.session.load(out.getvalue())
            np.testing.assert_array_equal(previous, self.session.timeline_clip().positions)
        with self.assertRaises(ValueError):
            self.session.load(b"x" * 64_000_001)

    def test_example_uses_empty_reference_stage_after_crowded_scene(self):
        wall = make_object("wall", 0)
        wall.update(position=[0., 1., 0.], size=[8., 2., 8.])
        crowded = {**EMPTY_SCENE, "name": "Authored crowded scene", "objects": [wall]}
        original = json.dumps(crowded, sort_keys=True)
        self.start(scene=crowded)
        self.session.load_example()
        self.assertEqual(self.session.scene_document["objects"], [])
        self.assertEqual(self.session.scene_document["name"], "Native Core martial arts reference")
        self.assertEqual(json.dumps(crowded, sort_keys=True), original)
        self.assertEqual(self.session._director.project_metadata["studio_core"]["original_scene_document"],
                         self.session.scene_document)
        self.assertEqual(self.client.calls, [])

    def test_default_cast_marks_use_clear_scene_geometry(self):
        from studio_interaction_scene import recommend_placements
        wall = make_object("wall", 0)
        wall.update(position=[0., 1., 0.], size=[4., 2., 4.])
        document = {**EMPTY_SCENE, "objects": [wall]}
        self.start(2, scene=document)
        expected = recommend_placements(document, 2)
        self.assertEqual(list(self.session._placements.values()), expected)
        self.assertGreaterEqual(np.linalg.norm(np.subtract(expected[0]["position_xz"],
                                                          expected[1]["position_xz"])), 1.5)

    def test_example_offline_exact_and_close_bounded(self):
        self.session.load_example()
        self.assertTrue(self.session.snapshot()["initialized"])
        self.assertEqual(self.session.snapshot()["total_frames"], 160)
        self.assertEqual(self.client.calls, [])
        self.assertIsNone(self.session._thread)
        self.session.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.session.activate()

    def test_cancel_pending_public_director_retains_committed_clip(self):
        d = RealtimeDirector(("actor_1",))
        d.queue_sequence((StageSpec("Walk", frames=80),))
        first = d.claim_request()
        d.complete(first.request_id, clip())
        stale = d.claim_request()
        d.cancel_pending()
        self.assertEqual(d.total_frames, 40)
        self.assertEqual(d.take_cancellations(), (stale.request_id,))
        self.assertFalse(d.complete(stale.request_id, clip(value=99)))
        self.assertIsNone(d.claim_request())


if __name__ == "__main__":
    unittest.main()
