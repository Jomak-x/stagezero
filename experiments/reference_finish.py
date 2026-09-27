"""Experimental native reference targets for the final four frames of a Core job.

This constructs conditioning inputs only. The worker must generate the whole
40-frame transition from actual native history. Never append these target poses
as generated output; they do not establish that the model reached the finish.
Run this file with --self-test for CPU rigid-transform and contract checks.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from realtime_clip import CanonicalClip


def _heading(positions):
    hips = positions[19] - positions[23]  # Official Core RightUpLeg - LeftUpLeg.
    if np.linalg.norm(hips[[0, 2]]) < 1e-6:
        raise ValueError("Hip positions have no reliable planar heading")
    return math.atan2(float(hips[2]), -float(hips[0]))


def _digest(clip):
    """Stable array-content digest, explicitly distinct from an NPZ file hash."""
    digest = hashlib.sha256()
    descriptor = {"version": 1, "fps": clip.fps, "actor_ids": list(clip.actor_ids),
                  "arrays": "positions,rotations,native_features; little-endian float32 C order"}
    digest.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode())
    for name in ("positions", "rotations", "native_features"):
        array = np.asarray(getattr(clip, name), dtype="<f4", order="C")
        digest.update(json.dumps({"name": name, "shape": list(array.shape)}, sort_keys=True).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def make_finish_target(source_clip, history_clip, *, source_actor=1,
                       source_frames=None, desired_headings=None):
    """Return ``(target, provenance)`` for a native two-actor H40 transition.

    ``source_frames=None`` selects the final four contiguous reference frames.
    ``desired_headings`` optionally maps the current actor IDs to absolute yaw
    radians; otherwise each actor retains its latest root heading. Both source
    and current clips must retain native Core features. Target arrays are JSON
    lists with shapes [2,40,27,3] and [2,40,27,3,3]. Only slots 36:40 are used by
    the existing transition adapter; earlier slots repeat the first target pose.

    Apply one Y-axis rotation and XZ translation to all selected joints and
    rotations for each actor. Align the FIRST reference root to the LAST current
    root in XZ. Preserve reference root height and all reference Y coordinates,
    including any airborne height; there is no floor snap or contact repair.
    """
    for label, clip in (("source", source_clip), ("history", history_clip)):
        if not isinstance(clip, CanonicalClip) or clip.native_features is None:
            raise ValueError(f"{label} must be a native Core CanonicalClip")
    if len(history_clip.actor_ids) != 2:
        raise ValueError("The transition endpoint requires exactly two current actors")
    if type(source_actor) is not int or not 0 <= source_actor < len(source_clip.actor_ids):
        raise ValueError("source_actor must index a reference actor")
    if source_clip.frames < 4:
        raise ValueError("Reference must contain at least four frames")
    if source_frames is None:
        source_frames = list(range(source_clip.frames - 4, source_clip.frames))
    if (not isinstance(source_frames, (tuple, list)) or len(source_frames) != 4
            or any(type(f) is not int or not 0 <= f < source_clip.frames for f in source_frames)
            or any(b != a + 1 for a, b in zip(source_frames, source_frames[1:]))):
        raise ValueError("source_frames must identify exactly four contiguous reference frames")
    if desired_headings is not None:
        if not isinstance(desired_headings, dict) or set(desired_headings) != set(history_clip.actor_ids):
            raise ValueError("desired_headings must cover both current actor IDs")
        if any(type(yaw) not in (int, float) or not math.isfinite(yaw) or abs(yaw) > math.pi
               for yaw in desired_headings.values()):
            raise ValueError("Desired headings must be finite radians within ±pi")
    source_p = source_clip.positions[source_actor, source_frames].astype(np.float64)
    source_r = source_clip.rotations[source_actor, source_frames].astype(np.float64)
    source_origin = source_p[0, 0, [0, 2]]
    source_yaw = _heading(source_p[0])
    targets_p, targets_r, transforms = [], [], []
    for i, actor_id in enumerate(history_clip.actor_ids):
        current_yaw = _heading(history_clip.positions[i, -1])
        desired_yaw = current_yaw if desired_headings is None else float(desired_headings[actor_id])
        delta = math.atan2(math.sin(desired_yaw - source_yaw), math.cos(desired_yaw - source_yaw))
        c, s = math.cos(delta), math.sin(delta)
        rotation = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
        origin = history_clip.positions[i, -1, 0, [0, 2]].astype(np.float64)
        offset = np.zeros(3)
        offset[[0, 2]] = origin - (rotation @ np.array([source_origin[0], 0., source_origin[1]]))[[0, 2]]
        positions = source_p @ rotation.T + offset
        rotations = rotation @ source_r
        if np.max(np.abs(positions[:, 0][:, [0, 2]])) > 25:
            raise ValueError("Transformed target roots leave the Core ±25 m scene bounds")
        targets_p.append(np.concatenate((np.repeat(positions[:1], 36, axis=0), positions), axis=0))
        targets_r.append(np.concatenate((np.repeat(rotations[:1], 36, axis=0), rotations), axis=0))
        transforms.append({"actor_id": actor_id, "current_heading_radians": current_yaw,
                           "target_heading_radians": desired_yaw, "yaw_delta_radians": delta,
                           "translation_xyz_m": offset.tolist(), "rotation_matrix": rotation.tolist(),
                           "anchor_current_root_xz_m": origin.tolist()})
    target = {"positions": np.asarray(targets_p, dtype=np.float32).tolist(),
              "rotations": np.asarray(targets_r, dtype=np.float32).tolist()}
    provenance = {"version": 1, "experimental": True, "conditioning_target_only": True,
                  "source_array_sha256": _digest(source_clip),
                  "source_hash_contract": "CanonicalClip arrays and actor IDs/fps; not archive bytes",
                  "history_array_sha256": _digest(history_clip),
                  "source_actor_index": source_actor,
                  "source_actor_id": source_clip.actor_ids[source_actor],
                  "source_frames": list(source_frames), "source_heading_radians": source_yaw,
                  "source_anchor_root_xz_m": source_origin.tolist(),
                  "target_conditioned_frames": [36, 37, 38, 39],
                  "ignored_target_frames": [0, 36], "transforms": transforms,
                  "root_height_preserved": True, "model_outcome_verified": False,
                  "warning": "Reference goals only; generate and review the complete native transition. No output pose splicing, foot locking or contact solve."}
    return target, provenance


def _self_test():
    import unittest
    from realtime_backend import validate_job

    def rotation(yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]], dtype=np.float32)

    def fixture():
        p = np.zeros((2, 8, 27, 3), np.float32)
        r = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 8, 27, 3, 3)).copy()
        # Nonzero world origin, moving root, distinct joint offsets and heights.
        for frame in range(8):
            p[1, frame, :, 0] = 3. + frame * .01 + np.arange(27) * .02
            p[1, frame, :, 1] = .8 + np.arange(27) * .03
            p[1, frame, :, 2] = 4. + np.arange(27) * .01
        p[:, :, 19] = p[:, :, 0] + [-.1, 0., 0.]
        p[:, :, 23] = p[:, :, 0] + [.1, 0., 0.]
        f = np.zeros((2, 8, 330), np.float32)
        source = CanonicalClip(p, r, 20, ("source_a", "source_b"), "ardy_core", {}, f)
        p[0, :, 0] = [8., 1.3, 2.]
        p[1, :, 0] = [-3., 1.7, -2.]
        r[0] = rotation(math.pi / 2)
        r[1] = rotation(-math.pi / 2)
        for i in range(2):
            p[i, :, 19] = p[i, :, 0] + r[i, 0, 0] @ np.array([-.1, 0., 0.])
            p[i, :, 23] = p[i, :, 0] + r[i, 0, 0] @ np.array([.1, 0., 0.])
        history = CanonicalClip(p, r, 20, ("actor_1", "actor_2"), "ardy_core", {}, f)
        return source, history

    class ReferenceFinishTests(unittest.TestCase):
        def test_rigid_transform_height_and_contiguous_motion(self):
            source, history = fixture()
            original = source.positions.copy()
            target, provenance = make_finish_target(source, history)
            p, r = np.asarray(target["positions"]), np.asarray(target["rotations"])
            self.assertEqual(p.shape, (2, 40, 27, 3))
            self.assertEqual(r.shape, (2, 40, 27, 3, 3))
            for actor in range(2):
                np.testing.assert_allclose(p[actor, 36, 0, [0, 2]], history.positions[actor, -1, 0, [0, 2]], atol=1e-6)
                np.testing.assert_allclose(p[actor, 36:, :, 1], source.positions[1, -4:, :, 1], atol=1e-6)
                np.testing.assert_allclose(np.linalg.norm(p[actor, 36:, 10] - p[actor, 36:, 0], axis=-1),
                                           np.linalg.norm(source.positions[1, -4:, 10] - source.positions[1, -4:, 0], axis=-1), atol=1e-6)
                np.testing.assert_allclose(r[actor, 36, 0], history.rotations[actor, -1, 0], atol=1e-6)
                self.assertGreater(np.linalg.norm(p[actor, 39, 0] - p[actor, 36, 0]), .02)
            np.testing.assert_array_equal(source.positions, original)
            self.assertEqual(provenance["target_conditioned_frames"], [36, 37, 38, 39])
            self.assertFalse(provenance["model_outcome_verified"])
            validate_job({"request_id": "reference-finish-test", "stage_kind": "transition", "frames": 40,
                          "prompt": "Raise both arms", "actor_ids": list(history.actor_ids), "seed": 3,
                          "history": {"native_features": history.native_features.tolist()}, "target": target})

        def test_hash_reproducibility_and_explicit_heading(self):
            source, history = fixture()
            _, a = make_finish_target(source, history)
            target, b = make_finish_target(source, history, desired_headings={actor: 0. for actor in history.actor_ids})
            self.assertEqual(a["source_array_sha256"], b["source_array_sha256"])
            self.assertEqual(len(a["source_array_sha256"]), 64)
            np.testing.assert_allclose(np.asarray(target["rotations"])[0, 36, 0], np.eye(3), atol=1e-6)
            changed = source.positions.copy(); changed[1, -1, 10, 0] += .1
            new_source = CanonicalClip(changed, source.rotations, 20, source.actor_ids, "ardy_core", {}, source.native_features)
            _, changed_provenance = make_finish_target(new_source, history)
            self.assertNotEqual(a["source_array_sha256"], changed_provenance["source_array_sha256"])

        def test_invalid_reference_frames_and_headings(self):
            source, history = fixture()
            for kwargs in ({"source_actor": True}, {"source_actor": 2}, {"source_frames": [0, 1, 3, 4]},
                           {"source_frames": [0, 1, 2]}, {"desired_headings": {"actor_1": 0}},
                           {"desired_headings": {"actor_1": float("nan"), "actor_2": 0}}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    make_finish_target(source, history, **kwargs)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceFinishTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if sys.argv[1:] != ["--self-test"]:
        raise SystemExit("Import make_finish_target(source_clip, history_clip), or run --self-test; this helper does not generate motion.")
    raise SystemExit(_self_test())
