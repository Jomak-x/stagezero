"""Offline transaction checks; no GPU or live Core session required."""
import tempfile
import io
import unittest
import zipfile
import json
import math
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from motion_bridge import _layout
from realtime_clip import CanonicalClip
from scene_composition import make_preset
from terrain_assisted_session import (
    PresentationClip, assist_native_terrain_result, generate_native_terrain_commands,
    load_assisted_result, load_native_terrain_result, run_assisted_terrain_commands,
    save_assisted_result, save_native_terrain_result,
)
from traversal_kit import TEMPLE_START_ROOT_XYZ, traversable_temple_scene


_, _, NEUTRAL = _layout()
IDS = ("actor_1",)
PLACEMENT = {"actor_1": {"position_xz": [0., 0.], "yaw": 0.}}


class TargetClient:
    """Return Core-shaped native horizons with unique feature tokens."""

    def __init__(self):
        self.root = np.array((0., .97, 0.))
        self.calls = []

    def wait(self, body, *, cancelled):
        self.calls.append(body)
        targets = body["root_targets"][IDS[0]]
        frames = [-1]+[item["frame"] for item in targets]
        points = [self.root]+[np.array((item["position_xz"][0],
                                        item["root_height"]+.02,
                                        item["position_xz"][1])) for item in targets]
        roots = np.stack([np.interp(np.arange(40), frames,
                                     np.asarray(points)[:, axis]) for axis in range(3)], axis=1)
        self.root = roots[-1]
        positions = (roots[:, None, :]+NEUTRAL[None]).astype(np.float32)[None]
        rotations = np.broadcast_to(np.eye(3, dtype=np.float32),
                                    (1, 40, 27, 3, 3)).copy()
        features = np.full((1, 40, 330), len(self.calls), np.float32)
        return [CanonicalClip(positions, rotations, 20, IDS, "ardy_core", {}, features)]


def fitted_assistor(native_positions, native_rotations, geometry, character, **kwargs):
    poses = [character.retarget(p, r) for p, r in zip(native_positions, native_rotations)]
    return (np.asarray([row["positions"] for row in poses]),
            np.asarray([row["rotations"] for row in poses]),
            np.ones((len(poses), 2), dtype=np.bool_),
            {"accepted": False, "rig_mesh_sha256": character.mesh_sha256,
             "rig_bone_count": 17, "numerical_rejections": []})


class OfflineTerrainSessionTests(unittest.TestCase):
    def test_relative_sequence_uses_previous_route_travel_heading(self):
        client = TargetClient()
        result = generate_native_terrain_commands(
            self.scene, "walk 0.5m left then walk 0.5m forward",
            actor_ids=IDS, actor_id=IDS[0], initial_placements=PLACEMENT,
            client=client)
        np.testing.assert_allclose(np.asarray(result.routes[0]["support_xyz"][-1])[[0, 2]],
                                   [-0.5, 0.], atol=.02)
        np.testing.assert_allclose(np.asarray(result.routes[1]["support_xyz"][-1])[[0, 2]],
                                   [-1., 0.], atol=.02)
        self.assertEqual(len(result.measurements), 2)
        self.assertTrue(all(row["completed"] for row in result.measurements))

    def test_repeated_submission_uses_committed_display_heading_for_planning_only(self):
        first = assist_native_terrain_result(self.native, assistor=fitted_assistor)
        original_native = first.native_clip.native_features.copy()
        rotations = first.presentation.rotations.copy()
        yaw = math.pi/2
        rotations[0, -1, 0] = np.array(((math.cos(yaw), 0., math.sin(yaw)),
                                        (0., 1., 0.),
                                        (-math.sin(yaw), 0., math.cos(yaw))))
        display = PresentationClip(first.presentation.positions, rotations,
                                   first.presentation.stance, first.presentation.fps,
                                   first.presentation.actor_ids,
                                   first.presentation.rig_asset_sha256,
                                   first.presentation.report)
        committed = replace(first, presentation=display)
        with patch("terrain_assisted_session.generate_native_terrain_commands",
                   side_effect=RuntimeError("captured")) as generate:
            with self.assertRaisesRegex(RuntimeError, "captured"):
                run_assisted_terrain_commands(
                    self.scene, "walk 0.5m forward", actor_ids=IDS,
                    actor_id=IDS[0], initial_placements=PLACEMENT,
                    client=TargetClient(), committed_assisted=committed)
        self.assertAlmostEqual(generate.call_args.kwargs["planning_heading"], yaw)
        np.testing.assert_array_equal(first.native_clip.native_features, original_native)

    def test_array_header_cannot_allocate_more_than_zip_payload(self):
        from terrain_assisted_session import _validate_array_headers
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {
            "descr": "<f8", "fortran_order": False, "shape": (10**12,)})
        packed = io.BytesIO()
        with zipfile.ZipFile(packed, "w") as archive:
            archive.writestr("positions.npy", header.getvalue())
        packed.seek(0)
        with zipfile.ZipFile(packed) as archive:
            with self.assertRaisesRegex(ValueError, "array header"):
                _validate_array_headers(archive, archive.infolist())

    @classmethod
    def setUpClass(cls):
        cls.scene = make_preset("Jungle temple")
        cls.client = TargetClient()
        cls.native = generate_native_terrain_commands(
            cls.scene, "walk 2m forward", actor_ids=IDS, actor_id=IDS[0],
            initial_placements=PLACEMENT, client=cls.client)

    def test_private_native_horizons_use_exact_feature_history(self):
        self.assertGreaterEqual(len(self.client.calls), 2)
        self.assertNotIn("history", self.client.calls[0])
        np.testing.assert_array_equal(
            np.asarray(self.client.calls[1]["history"]["native_features"]),
            np.ones((1, 40, 330), np.float32))
        self.assertEqual(self.native.action_spans, ((0, self.native.native_clip.frames),))
        self.assertTrue(self.native.measurements[0]["completed"])
        self.assertTrue(self.native.measurements[0]["native_root_body_checked"])
        self.assertFalse(self.native.measurements[0]["native_raw_feet_checked"])

    def test_native_result_can_be_saved_and_assisted_without_regeneration(self):
        with tempfile.TemporaryDirectory() as folder:
            native_path = Path(folder)/"native.npz"
            assisted_path = Path(folder)/"assisted.npz"
            save_native_terrain_result(self.native, native_path)
            loaded = load_native_terrain_result(native_path)
            np.testing.assert_array_equal(loaded.native_clip.native_features,
                                          self.native.native_clip.native_features)
            result = assist_native_terrain_result(loaded, assistor=fitted_assistor)
            self.assertEqual(result.presentation.positions.shape[2:], (17, 3))
            self.assertIsNotNone(result.native_clip.native_features)
            self.assertFalse(result.report["accepted"])
            save_assisted_result(result, assisted_path)
            replay = load_assisted_result(assisted_path)
            self.assertEqual(replay.presentation.rig_asset_sha256,
                             result.presentation.rig_asset_sha256)
            np.testing.assert_array_equal(replay.native_clip.native_features,
                                          result.native_clip.native_features)
            with zipfile.ZipFile(assisted_path, "a") as archive:
                archive.writestr("unexpected.npy", b"x")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                load_assisted_result(assisted_path)

    def test_cancellation_discards_provisional_result(self):
        client = TargetClient()
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            generate_native_terrain_commands(
                self.scene, "walk 2m forward", actor_ids=IDS, actor_id=IDS[0],
                initial_placements=PLACEMENT, client=client,
                cancelled=lambda: bool(client.calls))
        self.assertEqual(len(client.calls), 1)

    def test_native_prefix_is_used_for_history_but_needs_actual_display_boundary(self):
        root = np.array((0., .97, 0.), np.float32)
        positions = np.broadcast_to(root+NEUTRAL, (1, 40, 27, 3)).copy()
        rotations = np.broadcast_to(np.eye(3, dtype=np.float32),
                                    (1, 40, 27, 3, 3)).copy()
        prefix = CanonicalClip(positions, rotations, 20, IDS, "ardy_core", {},
                               np.full((1, 40, 330), 7., np.float32))
        client = TargetClient()
        result = generate_native_terrain_commands(
            self.scene, "walk 1m forward", actor_ids=IDS, actor_id=IDS[0],
            initial_placements=PLACEMENT, client=client,
            committed_prefix=prefix)
        np.testing.assert_array_equal(
            np.asarray(client.calls[0]["history"]["native_features"]),
            prefix.native_features)
        self.assertEqual(result.action_spans[0][0], 40)
        with self.assertRaisesRegex(ValueError, "previous actual presentation"):
            assist_native_terrain_result(result, assistor=fitted_assistor)

    def test_repeated_commands_preserve_paired_prefix_and_archive(self):
        first = assist_native_terrain_result(self.native, assistor=fitted_assistor)
        original_p = first.presentation.positions.copy()
        original_r = first.presentation.rotations.copy()
        client = TargetClient()
        client.root = first.native_clip.positions[0, -1, 0].copy()
        seen = []
        def boundary_assistor(*args, **kwargs):
            seen.append((kwargs.get("initial_assisted_positions"),
                         kwargs.get("initial_assisted_rotations")))
            return fitted_assistor(*args, **kwargs)
        second = run_assisted_terrain_commands(
            self.scene, "walk 1m forward", actor_ids=IDS, actor_id=IDS[0],
            initial_placements=PLACEMENT, client=client,
            committed_assisted=first, assistor=boundary_assistor)
        n = first.native_clip.frames
        np.testing.assert_array_equal(second.native_clip.positions[:, :n], first.native_clip.positions)
        np.testing.assert_array_equal(second.native_clip.rotations[:, :n], first.native_clip.rotations)
        np.testing.assert_array_equal(second.native_clip.native_features[:, :n], first.native_clip.native_features)
        np.testing.assert_array_equal(second.presentation.positions[:, :n], original_p)
        np.testing.assert_array_equal(second.presentation.rotations[:, :n], original_r)
        np.testing.assert_array_equal(second.presentation.stance[:, :n], first.presentation.stance)
        np.testing.assert_array_equal(seen[0][0], original_p[0, -1])
        np.testing.assert_array_equal(seen[0][1], original_r[0, -1])
        self.assertEqual(second.report["actions"][0], first.report["actions"][0])
        self.assertEqual(second.report["terrain_start_frame"], 0)
        self.assertEqual(len(second.routes), 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"appended.npz"
            save_assisted_result(second, path)
            loaded = load_assisted_result(path)
            np.testing.assert_array_equal(loaded.presentation.positions, second.presentation.positions)
            np.testing.assert_array_equal(loaded.native_clip.native_features, second.native_clip.native_features)
            self.assertEqual(loaded.report, json.loads(json.dumps(second.report)))

    def test_cancel_append_keeps_committed_prefix_immutable(self):
        first = assist_native_terrain_result(self.native, assistor=fitted_assistor)
        old = (first.native_clip.native_features.tobytes(), first.presentation.positions.tobytes(),
               first.presentation.rotations.tobytes())
        client = TargetClient()
        client.root = first.native_clip.positions[0, -1, 0].copy()
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            run_assisted_terrain_commands(
                self.scene, "walk 1m forward", actor_ids=IDS, actor_id=IDS[0],
                initial_placements=PLACEMENT, client=client, committed_assisted=first,
                assistor=fitted_assistor, cancelled=lambda: bool(client.calls))
        self.assertEqual(old, (first.native_clip.native_features.tobytes(),
                             first.presentation.positions.tobytes(), first.presentation.rotations.tobytes()))
        self.assertFalse(first.presentation.positions.flags.writeable)

    def test_append_rejects_changed_scene_and_native_history_before_generation(self):
        first = assist_native_terrain_result(self.native, assistor=fitted_assistor)
        client = TargetClient()
        scene = dict(self.scene, name="changed authored scene")
        with self.assertRaisesRegex(ValueError, "authored scene"):
            run_assisted_terrain_commands(scene, "walk 1m forward", actor_ids=IDS, actor_id=IDS[0],
                initial_placements=PLACEMENT, client=client, committed_assisted=first)
        prefix = replace(first.native_clip, native_features=first.native_clip.native_features+1.)
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            run_assisted_terrain_commands(self.scene, "walk 1m forward", actor_ids=IDS, actor_id=IDS[0],
                initial_placements=PLACEMENT, client=client, committed_assisted=first, committed_prefix=prefix)
        self.assertEqual(client.calls, [])

    def test_append_rejects_assistor_that_changes_committed_boundary(self):
        first = assist_native_terrain_result(self.native, assistor=fitted_assistor)
        client = TargetClient()
        client.root = first.native_clip.positions[0, -1, 0].copy()
        old = first.presentation.positions.tobytes()
        def changed_boundary(*args, **kwargs):
            p, r, stance, report = fitted_assistor(*args, **kwargs)
            p[0, :, 0] += .001
            return p, r, stance, report
        with self.assertRaisesRegex(ValueError, "committed boundary pose"):
            run_assisted_terrain_commands(self.scene, "walk 1m forward", actor_ids=IDS, actor_id=IDS[0],
                initial_placements=PLACEMENT, client=client, committed_assisted=first,
                assistor=changed_boundary)
        self.assertEqual(old, first.presentation.positions.tobytes())

    def test_elevated_door_reaction_origin_survives_next_submission(self):
        scene = traversable_temple_scene()
        client = TargetClient()
        client.root = np.asarray(TEMPLE_START_ROOT_XYZ)+[0., .02, 0.]
        placements = {IDS[0]: {"position_xz": [TEMPLE_START_ROOT_XYZ[0], TEMPLE_START_ROOT_XYZ[2]],
                               "yaw": float(np.pi)}}
        first = generate_native_terrain_commands(scene,
            "walk up Shallow temple stairs, cross Suspended temple bridge, open Temple gate",
            actor_ids=IDS, actor_id=IDS[0], initial_placements=placements, client=client)
        old_gate = next(row for row in first.reaction_states if row["id"] == "temple-gate")
        second = generate_native_terrain_commands(scene, "enter", actor_ids=IDS,
            actor_id=IDS[0], initial_placements=placements, client=client,
            committed_prefix=first.native_clip, terrain_start_frame=first.terrain_start_frame,
            previous_routes=first.routes)
        gate = next(row for row in second.reaction_states if row["id"] == "temple-gate")
        self.assertEqual(gate["trigger_frame"], old_gate["trigger_frame"])
        self.assertEqual(gate["opening_fraction"], 1.)
        self.assertTrue(second.measurements[0]["crossing_verified"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"append-native.npz"
            save_native_terrain_result(second, path)
            self.assertEqual(load_native_terrain_result(path).terrain_start_frame, 0)

    def test_callback_sees_complete_native_route_before_assistance(self):
        observed = []
        def fail_assist(*args, **kwargs):
            raise ValueError("visual candidate deliberately rejected")
        with self.assertRaisesRegex(ValueError, "deliberately rejected"):
            run_assisted_terrain_commands(
                self.scene, "walk 1m forward", actor_ids=IDS, actor_id=IDS[0],
                initial_placements=PLACEMENT, client=TargetClient(),
                assistor=fail_assist,
                on_native_ready=lambda result:
                    observed.append((result.native_clip.frames, len(result.routes),
                                     result.action_spans, result.measurements)))
        self.assertEqual(len(observed), 1)
        self.assertGreater(observed[0][0], 0)
        self.assertTrue(observed[0][3][0]["completed"])

    def test_full_stairs_bridge_gate_enter_native_transaction(self):
        client = TargetClient()
        client.root = np.asarray(TEMPLE_START_ROOT_XYZ)+[0., .02, 0.]
        placements = {IDS[0]: {"position_xz": [TEMPLE_START_ROOT_XYZ[0],
                                                TEMPLE_START_ROOT_XYZ[2]],
                                "yaw": float(np.pi)}}
        result = generate_native_terrain_commands(
            traversable_temple_scene(),
            "walk up Shallow temple stairs, cross Suspended temple bridge, "
            "open Temple gate, and enter",
            actor_ids=IDS, actor_id=IDS[0], initial_placements=placements,
            client=client)
        self.assertEqual([route["verb"] for route in result.routes],
                         ["ascend", "cross", "open", "go_through"])
        self.assertEqual(result.native_clip.frames, result.action_spans[-1][1])
        self.assertTrue(all(measurement["completed"] for measurement in result.measurements))
        self.assertTrue(result.measurements[2]["automatic_door_open_verified"])
        self.assertTrue(result.measurements[3]["crossing_verified"])
        gate = next(state for state in result.reaction_states if state["id"] == "temple-gate")
        self.assertEqual(gate["opening_fraction"], 1.)


if __name__ == "__main__":
    unittest.main()
