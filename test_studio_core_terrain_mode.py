"""Normal Core terrain-mode isolation and transactional publish contracts."""
import io
import hashlib
import json
from pathlib import Path
import threading
import tempfile
import time
import unittest
import zipfile
from unittest.mock import patch

import numpy as np

from realtime_clip import CanonicalClip
from studio_core_session import CoreStudioSession, EMPTY_SCENE
from studio_core_terrain_state import native_digest
from studio_core_renderer import DEFAULT_ASSETS
from terrain_assisted_session import (AssistedTerrainResult, NativeTerrainResult,
                                      PresentationClip, load_assisted_result,
                                      load_native_terrain_result)

RIG_HASH = hashlib.sha256(DEFAULT_ASSETS[0].read_bytes()).hexdigest()


def _wait(predicate, timeout=3.):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("terrain worker did not reach expected state")


def _candidate(scene, previous=None):
    start = 0 if previous is None else previous.native_clip.frames
    frames = start+40
    p = np.zeros((1, frames, 27, 3), np.float32)
    p[..., 1] = .95
    p[0, :, 0, 0] = np.arange(frames, dtype=np.float32)*.01
    r = np.broadcast_to(np.eye(3, dtype=np.float32), (1, frames, 27, 3, 3)).copy()
    f = np.broadcast_to(np.arange(frames, dtype=np.float32)[None, :, None],
                        (1, frames, 330)).copy()
    native = CanonicalClip(p, r, 20, ("actor_1",), "offline_combined", {}, f)
    dp = np.zeros((1, frames, 17, 3), np.float32)
    dp[..., 1] = .95
    dp[0, :, 0, 0] = p[0, :, 0, 0]
    dr = np.broadcast_to(np.eye(3, dtype=np.float32), (1, frames, 17, 3, 3)).copy()
    stance = np.ones((1, frames, 2), np.bool_)
    display = PresentationClip(dp, dr, stance, 20, ("actor_1",), (RIG_HASH,),
                               {"actions": [{} for _ in range(frames//40)]})
    route = {"verb": "cross", "target_id": "bridge", "schedule": {"frames": 40}}
    actions = ([] if previous is None else list(previous.report["actions"])) + [
        {"native_span": [start, frames], "route": route}]
    report = {"accepted": False, "native_sha256": native_digest(native),
              "rig_asset_sha256": [RIG_HASH], "actions": actions,
              "terrain_start_frame": 0}
    routes = tuple([route for _ in actions])
    return AssistedTerrainResult(scene, native, display, report, routes, ())


class TerrainModeTests(unittest.TestCase):
    def setUp(self):
        self.client = object()
        self.session = CoreStudioSession(self.client)
        self.session.start(1, EMPTY_SCENE)
        self.addCleanup(self.session.close)

    def test_toggle_is_separate_and_commits_complete_paired_stream(self):
        ordinary = self.session._director
        self.session.set_terrain_aware(True)
        self.assertIsNone(self.session.timeline_clip())
        with patch("terrain_assisted_session.run_assisted_terrain_commands",
                   side_effect=lambda scene, text, **kw: _candidate(scene, kw["committed_assisted"])):
            self.session.spatial_commands("actor_1", "cross the bridge")
            _wait(lambda: not self.session.snapshot()["terrain_pending"])
        terrain = self.session.timeline_clip()
        self.assertEqual(terrain.frames, 40)
        self.assertEqual(self.session.terrain_presentation().frames, 40)
        self.assertIn("Human17", self.session.snapshot()["geometry_check"])
        self.assertNotIn("no pose correction", self.session.snapshot()["geometry_check"])
        np.testing.assert_array_equal(terrain.native_features, self.session.terrain_result().native_clip.native_features)
        revision = self.session.snapshot()["terrain_display_revision"]
        self.session.set_terrain_aware(False)
        self.assertIs(self.session._director, ordinary)
        self.assertIsNone(self.session.terrain_presentation())
        self.session.set_terrain_aware(True)
        self.assertEqual(self.session.timeline_clip().frames, 40)
        self.assertGreater(self.session.snapshot()["terrain_display_revision"], revision)

    def test_cancel_scene_change_and_failed_retry_keep_prior_take(self):
        self.session.set_terrain_aware(True)
        gate = threading.Event()
        started = threading.Event()

        def slow(scene, text, **kw):
            started.set()
            gate.wait(1.)  # Deliberately ignores cancellation like a late HTTP reply.
            return _candidate(scene, kw["committed_assisted"])

        with patch("terrain_assisted_session.run_assisted_terrain_commands", side_effect=slow):
            self.session.spatial_commands("actor_1", "cross the bridge")
            self.assertTrue(started.wait(1.))
            self.session.cancel()
            self.assertFalse(self.session.snapshot()["terrain_pending"])
            gate.set()
            _wait(lambda: self.session._terrain_thread is not None and not self.session._terrain_thread.is_alive())
            self.assertIsNone(self.session.timeline_clip())

        with patch("terrain_assisted_session.run_assisted_terrain_commands",
                   side_effect=lambda scene, text, **kw: _candidate(scene, kw["committed_assisted"])):
            self.session.spatial_commands("actor_1", "cross the bridge")
            _wait(lambda: self.session.timeline_clip() is not None)
        prefix = self.session.timeline_clip().native_features.copy()
        display = self.session.terrain_presentation().positions.copy()

        def broken(scene, text, **kw):
            raise ValueError("measured route rejected")

        with patch("terrain_assisted_session.run_assisted_terrain_commands", side_effect=broken):
            self.session.spatial_commands("actor_1", "open the gate")
            _wait(lambda: not self.session.snapshot()["terrain_pending"])
        self.assertIn("failed", self.session.snapshot()["terrain_status"])
        self.assertTrue(self.session.snapshot()["terrain_retry_available"])
        np.testing.assert_array_equal(prefix, self.session.timeline_clip().native_features)
        np.testing.assert_array_equal(display, self.session.terrain_presentation().positions)

        with patch("terrain_assisted_session.run_assisted_terrain_commands",
                   side_effect=lambda scene, text, **kw: _candidate(scene, kw["committed_assisted"])):
            self.assertTrue(self.session.retry())
            _wait(lambda: self.session.snapshot()["total_frames"] == 80)
        np.testing.assert_array_equal(prefix, self.session.timeline_clip().native_features[:, :40])
        np.testing.assert_array_equal(display, self.session.terrain_presentation().positions[:, :40])

        scene = dict(EMPTY_SCENE, name="Different authored place")
        retained = self.session.terrain_result()
        self.assertFalse(self.session.update_scene(scene))
        self.assertIs(self.session.terrain_result(), retained)
        self.assertEqual(self.session.scene_document, EMPTY_SCENE)
        self.session.start_terrain(scene, {"actor_1": {"position_xz": [0., 0.], "yaw": 0.}})
        self.assertIsNone(self.session.timeline_clip())
        self.assertIsNone(self.session.terrain_presentation())

    def test_save_load_then_append_preserves_exact_native_and_rig_prefix(self):
        self.session.set_terrain_aware(True)
        observed = []

        def runner(scene, text, **kw):
            observed.append(kw["committed_assisted"])
            return _candidate(scene, kw["committed_assisted"])

        with patch("terrain_assisted_session.run_assisted_terrain_commands", side_effect=runner):
            self.session.spatial_commands("actor_1", "cross the bridge")
            _wait(lambda: self.session.timeline_clip() is not None)
            first = self.session.terrain_result()
            saved = self.session.save()
            restored = CoreStudioSession(self.client)
            self.addCleanup(restored.close)
            restored.load(saved)
            self.assertTrue(restored.snapshot()["terrain_aware"])
            self.assertEqual(restored.terrain_presentation().frames, 40)
            restored.spatial_commands("actor_1", "open the gate")
            _wait(lambda: restored.snapshot()["total_frames"] == 80)
            self.assertIsNotNone(observed[-1])
            np.testing.assert_array_equal(observed[-1].native_clip.native_features,
                                          first.native_clip.native_features)
            np.testing.assert_array_equal(restored.timeline_clip().native_features[:, :40],
                                          first.native_clip.native_features)
            np.testing.assert_array_equal(restored.terrain_presentation().positions[:, :40],
                                          first.presentation.positions)
            self.assertEqual(restored.snapshot()["terrain_display_revision"] > 0, True)

    def test_ordinary_entry_points_reject_mixed_stream_in_terrain_mode(self):
        self.session.set_terrain_aware(True)
        with self.assertRaisesRegex(ValueError, "Switch off terrain-aware"):
            self.session.direct({"actor_1": "walk"})
        with self.assertRaisesRegex(ValueError, "Switch off terrain-aware"):
            self.session.choreograph({"beats": []})
        with self.assertRaisesRegex(ValueError, "spatial commands"):
            self.session.navigate("actor_1", "missing")
        self.assertIsNone(self.session.timeline_clip())

    def test_start_terrain_preserves_an_existing_two_actor_ordinary_cast(self):
        self.session.start(2, EMPTY_SCENE)
        ordinary = self.session._director
        ordinary_scene = self.session.scene_document
        ordinary_placements = self.session._placements.copy()
        scene = dict(EMPTY_SCENE, name="Independent terrain stage")
        placement = {"actor_1": {"position_xz": [1., 2.], "yaw": 0.}}
        state = self.session.start_terrain(scene, placement)
        self.assertTrue(state["terrain_aware"])
        self.assertEqual(state["actor_ids"], ("actor_1",))
        self.assertEqual(self.session.scene_document["name"], "Independent terrain stage")
        self.session.set_terrain_aware(False)
        self.assertIs(self.session._director, ordinary)
        self.assertEqual(self.session.snapshot()["actor_ids"], ("actor_1", "actor_2"))
        self.assertEqual(self.session.scene_document, ordinary_scene)
        self.assertEqual(self.session._placements, ordinary_placements)
        self.session.set_terrain_aware(True)
        self.assertEqual(self.session.scene_document["name"], "Independent terrain stage")

    def test_first_time_terrain_start_restores_uninitialized_ordinary_slot(self):
        new = CoreStudioSession(self.client)
        self.addCleanup(new.close)
        self.assertFalse(new.snapshot()["initialized"])
        new.start_terrain(EMPTY_SCENE, {"actor_1": {"position_xz": [0., 0.], "yaw": 0.}})
        self.assertTrue(new.snapshot()["initialized"])
        new.set_terrain_aware(False)
        self.assertFalse(new.snapshot()["initialized"])
        self.assertEqual(new.snapshot()["total_frames"], 0)
        new.set_terrain_aware(True)
        self.assertTrue(new.snapshot()["initialized"])

    def test_loading_terrain_project_does_not_replace_existing_two_actor_cast(self):
        self.session.set_terrain_aware(True)
        with patch("terrain_assisted_session.run_assisted_terrain_commands",
                   side_effect=lambda scene, text, **kw: _candidate(scene, kw["committed_assisted"])):
            self.session.spatial_commands("actor_1", "cross the bridge")
            _wait(lambda: self.session.timeline_clip() is not None)
        saved = self.session.save()
        other = CoreStudioSession(self.client)
        self.addCleanup(other.close)
        other.start(2, dict(EMPTY_SCENE, name="Ordinary two-actor take"))
        ordinary = other._director
        other.load(saved)
        self.assertTrue(other.snapshot()["terrain_aware"])
        other.set_terrain_aware(False)
        self.assertIs(other._director, ordinary)
        self.assertEqual(other.snapshot()["actor_ids"], ("actor_1", "actor_2"))
        self.assertEqual(other.scene_document["name"], "Ordinary two-actor take")

    def test_corrupt_presentation_rejected_before_replacing_visible_take(self):
        self.session.set_terrain_aware(True)
        with patch("terrain_assisted_session.run_assisted_terrain_commands",
                   side_effect=lambda scene, text, **kw: _candidate(scene, kw["committed_assisted"])):
            self.session.spatial_commands("actor_1", "cross the bridge")
            _wait(lambda: self.session.timeline_clip() is not None)
        before = self.session.terrain_result()
        original = self.session.save()
        output = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(output, "w") as target:
            for name in source.namelist():
                raw = source.read(name)
                if name == "terrain_positions.npy":
                    values = np.load(io.BytesIO(raw), allow_pickle=False)
                    changed = values.copy()
                    changed[0, 0, 0, 0] += 1.
                    data = io.BytesIO()
                    np.save(data, changed, allow_pickle=False)
                    raw = data.getvalue()
                target.writestr(name, raw)
        with self.assertRaisesRegex(ValueError, "presentation provenance"):
            self.session.load(output.getvalue())
        self.assertIs(self.session.terrain_result(), before)

    def test_foreign_rig_archive_rejected_before_replacing_visible_take(self):
        self.session.set_terrain_aware(True)
        with patch("terrain_assisted_session.run_assisted_terrain_commands",
                   side_effect=lambda scene, text, **kw: _candidate(scene, kw["committed_assisted"])):
            self.session.spatial_commands("actor_1", "cross the bridge")
            _wait(lambda: self.session.timeline_clip() is not None)
        before = self.session.terrain_result()
        output = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(self.session.save())) as source, zipfile.ZipFile(output, "w") as target:
            for name in source.namelist():
                raw = source.read(name)
                if name == "terrain_manifest.npy":
                    manifest = json.loads(str(np.load(io.BytesIO(raw), allow_pickle=False)))
                    manifest["rig_asset_sha256"] = ["b"*64]
                    encoded = io.BytesIO()
                    np.save(encoded, np.array(json.dumps(manifest)), allow_pickle=False)
                    raw = encoded.getvalue()
                target.writestr(name, raw)
        with self.assertRaisesRegex(ValueError, "different character assets"):
            self.session.load(output.getvalue())
        self.assertIs(self.session.terrain_result(), before)

    def test_forged_huge_presentation_shape_rejected_before_array_materialization(self):
        self.session.set_terrain_aware(True)
        with patch("terrain_assisted_session.run_assisted_terrain_commands",
                   side_effect=lambda scene, text, **kw: _candidate(scene, kw["committed_assisted"])):
            self.session.spatial_commands("actor_1", "cross the bridge")
            _wait(lambda: self.session.timeline_clip() is not None)
        before = self.session.terrain_result()
        forged = io.BytesIO()
        np.lib.format.write_array_header_1_0(forged, {
            "descr": "<f4", "fortran_order": False,
            "shape": (1, 100_000_000, 17, 3)})
        output = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(self.session.save())) as source, zipfile.ZipFile(output, "w") as target:
            for name in source.namelist():
                target.writestr(name, forged.getvalue() if name == "terrain_positions.npy"
                                else source.read(name))
        with self.assertRaisesRegex(ValueError, "array header"):
            self.session.load(output.getvalue())
        self.assertIs(self.session.terrain_result(), before)

    def test_cancel_then_new_request_serializes_inference_and_drops_old_result(self):
        self.session.set_terrain_aware(True)
        old_started, release_old = threading.Event(), threading.Event()
        new_started = threading.Event()

        def runner(scene, text, **kw):
            if text == "old route":
                old_started.set()
                release_old.wait(2.)  # Simulate a transport that ignores cancellation.
                return _candidate(scene, kw["committed_assisted"])
            new_started.set()
            return _candidate(scene, kw["committed_assisted"])

        with patch("terrain_assisted_session.run_assisted_terrain_commands", side_effect=runner):
            self.session.spatial_commands("actor_1", "old route")
            self.assertTrue(old_started.wait(1.))
            self.session.cancel()
            self.session.spatial_commands("actor_1", "new route")
            self.assertFalse(new_started.wait(.03))
            release_old.set()
            _wait(lambda: self.session.snapshot()["total_frames"] == 40)
            self.assertTrue(new_started.is_set())
            self.assertIn("ready", self.session.snapshot()["terrain_status"])

    def test_scene_change_during_generation_discards_late_route(self):
        self.session.set_terrain_aware(True)
        started, release = threading.Event(), threading.Event()

        def runner(scene, text, **kw):
            started.set()
            release.wait(1.)
            return _candidate(scene, kw["committed_assisted"])

        with patch("terrain_assisted_session.run_assisted_terrain_commands", side_effect=runner):
            self.session.spatial_commands("actor_1", "old scene route")
            self.assertTrue(started.wait(1.))
            self.session.update_scene(dict(EMPTY_SCENE, name="New authored scene"))
            release.set()
            _wait(lambda: not self.session._terrain_thread.is_alive())
        self.assertIsNone(self.session.timeline_clip())
        self.assertIsNone(self.session.terrain_presentation())
        self.assertEqual(self.session.scene_document["name"], "New authored scene")

    def test_native_ready_checkpoint_survives_rig_failure_and_retry_avoids_inference(self):
        self.session.set_terrain_aware(True)
        with tempfile.TemporaryDirectory() as folder:
            self.session._terrain_checkpoint_dir = Path(folder)
            candidate = _candidate(EMPTY_SCENE)
            native = NativeTerrainResult(EMPTY_SCENE, candidate.native_clip,
                                         candidate.routes, ((0, 40),),
                                         ({"completed": True},), (), None)

            def fails_after_native(scene, text, **kw):
                kw["on_native_ready"](native)
                raise ValueError("rig solve failed")

            with patch("terrain_assisted_session.run_assisted_terrain_commands",
                       side_effect=fails_after_native):
                self.session.spatial_commands("actor_1", "cross the bridge")
                _wait(lambda: not self.session.snapshot()["terrain_pending"])
            state = self.session.snapshot()
            self.assertTrue(state["terrain_retry_available"])
            self.assertIsNotNone(state["terrain_checkpoint_path"])
            self.assertTrue(Path(state["terrain_checkpoint_path"]).is_file())
            restored = load_native_terrain_result(state["terrain_checkpoint_path"])
            np.testing.assert_array_equal(restored.native_clip.native_features,
                                          native.native_clip.native_features)
            self.session.configure_client(None)
            with (patch("terrain_assisted_session.run_assisted_terrain_commands",
                        side_effect=AssertionError("native inference reran")),
                  patch("terrain_assisted_session.assist_native_terrain_result",
                        side_effect=lambda saved, **kw: candidate)):
                self.assertTrue(self.session.retry())
                _wait(lambda: self.session.snapshot()["total_frames"] == 40)
            self.assertFalse(self.session.snapshot()["terrain_retry_available"])

    def test_append_checkpoint_carries_prior_assisted_boundary(self):
        self.session.set_terrain_aware(True)
        with tempfile.TemporaryDirectory() as folder:
            self.session._terrain_checkpoint_dir = Path(folder)
            with patch("terrain_assisted_session.run_assisted_terrain_commands",
                       side_effect=lambda scene, text, **kw: _candidate(scene, kw["committed_assisted"])):
                self.session.spatial_commands("actor_1", "cross the bridge")
                _wait(lambda: self.session.snapshot()["total_frames"] == 40)
            prior = self.session.terrain_result()
            extended = _candidate(EMPTY_SCENE, prior)
            generated = extended.native_clip.slice_frames(40, 80)
            native = NativeTerrainResult(EMPTY_SCENE, generated,
                                         (extended.routes[-1],), ((40, 80),),
                                         ({"completed": True},), (), prior.native_clip)

            def fails_after_native(scene, text, **kw):
                kw["on_native_ready"](native)
                raise ValueError("rig solve failed")

            with patch("terrain_assisted_session.run_assisted_terrain_commands",
                       side_effect=fails_after_native):
                self.session.spatial_commands("actor_1", "open the gate")
                _wait(lambda: not self.session.snapshot()["terrain_pending"])
            state = self.session.snapshot()
            self.assertTrue(Path(state["terrain_checkpoint_prior_path"]).is_file())
            restored = load_assisted_result(state["terrain_checkpoint_prior_path"])
            np.testing.assert_array_equal(restored.native_clip.native_features,
                                          prior.native_clip.native_features)
            np.testing.assert_array_equal(restored.presentation.positions,
                                          prior.presentation.positions)


if __name__ == "__main__":
    unittest.main()
