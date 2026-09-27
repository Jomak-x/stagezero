"""CPU-only contracts for the paired scene request and orchestration wrapper."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from native_pair_clip import NativePairClip
from paired_direction import (InteractionPromptPlanner, PairedSceneBuilder, library_clip,
                              preview_direction, resolve_meeting, validate_request)
from scene_objects import make_object


SCENE = {"version": 2, "name": "City", "objects": [], "effects": [], "lighting": "neutral"}


def request(**changes):
    value = {"actor_ids": ["ava", "ben"], "source": "handshake", "seed": 42,
             "prompt": "They shake hands and step apart.",
             "starts": [{"x": -2., "z": -1.}, {"x": 2., "z": -1.}],
             "meeting": {"x": 0., "z": 0., "yaw_degrees": 0.}, "target_id": None}
    value.update(changes)
    return value


def source_clip():
    joints = np.zeros((8, 2, 22, 3), dtype=np.float64)
    joints[:, 0, :, 0] = -0.8
    joints[:, 1, :, 0] = 0.8
    return NativePairClip(joints, metadata={"model": "InterGen", "prompt": "Handshake"})


class RequestTests(unittest.TestCase):
    def test_valid_request_is_detached_and_canonical(self):
        raw = request(actor_ids=("ava", "ben"), meeting={"x": 0, "z": 0})
        clean = validate_request(raw)
        self.assertEqual(clean["actor_ids"], ["ava", "ben"])
        self.assertEqual(clean["meeting"]["yaw_degrees"], 0.)
        clean["starts"][0]["x"] = 9
        self.assertEqual(raw["starts"][0]["x"], -2.)

    def test_invalid_ids_bounds_and_schema_fail_before_builder(self):
        invalid = (
            request(actor_ids=["ava", "ava"]), request(source="unknown"),
            request(actor_ids=[[], "ben"]),
            request(seed=-1), request(seed=True), request(prompt=""),
            request(starts=[{"x": 25, "z": 0}, {"x": 2, "z": 0}]),
            request(meeting={"x": float("nan"), "z": 0}),
            request(meeting={"x": 0, "z": 0, "yaw_degrees": 181}),
            request(target_id=""),
        )
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                validate_request(item)

    def test_library_rejects_unknown_source(self):
        with self.assertRaisesRegex(ValueError, "reviewed library"):
            library_clip("unknown")

    def test_landmark_is_resolved_beside_object(self):
        crate = make_object("crate", 0)
        crate["name"] = "Market crate"
        scene = dict(SCENE, objects=[crate])
        with patch("paired_direction.check_native_pair_geometry", return_value={}) as geometry:
            clean = resolve_meeting(request(target_id=crate["id"]), source_clip(), scene)
        self.assertNotEqual(clean["meeting"]["z"], crate["position"][2])
        self.assertEqual(clean["resolved_landmark"]["id"], crate["id"])
        geometry.assert_called()

    def test_canonical_scene_target_and_stale_id(self):
        crate = make_object("crate", 0)
        target = {"id": "meet-marker", "name": "Meeting marker", "object_id": crate["id"],
                  "kind": "landing", "local_position": [0., 0., 0.]}
        scene = dict(SCENE, objects=[crate], targets=[target])
        with patch("paired_direction.check_native_pair_geometry", return_value={}):
            clean = resolve_meeting(request(target_id=target["id"]), source_clip(), scene)
        self.assertEqual(clean["resolved_landmark"]["id"], target["id"])
        self.assertGreater(abs(clean["meeting"]["z"]), 1.5)
        with self.assertRaisesRegex(ValueError, "no longer exists"):
            resolve_meeting(request(target_id="deleted"), source_clip(), scene)

    def test_custom_interaction_preview_uses_reviewed_spatial_reference(self):
        with (patch("paired_direction.library_clip", return_value=source_clip()) as library,
              patch("paired_meetup.plan_meetup", return_value={"planned_only": True})):
            result = preview_direction(request(source="generate"), SCENE)
        library.assert_called_once_with("handshake")
        self.assertTrue(result["plan"]["planned_only"])
        self.assertIn("fresh motion is checked again", result["summary"])


class PlannerTests(unittest.TestCase):
    def test_exact_schema_and_bounded_prompt(self):
        gateway = Mock()
        gateway.request_json.return_value = {"interaction_prompt": " Two people dance together. "}
        self.assertEqual(InteractionPromptPlanner(gateway).plan("dance in the city"),
                         "Two people dance together.")
        self.assertEqual(gateway.request_json.call_args.args[1], "dance in the city")
        for bad in ({"interaction_prompt": "ok", "coordinates": [1, 2]},
                    {"interaction_prompt": ""}, {"interaction_prompt": "x" * 501},
                    {"interaction_prompt": 5}, ["not an object"]):
            gateway.request_json.return_value = bad
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "invalid description"):
                InteractionPromptPlanner(gateway).plan("dance")


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clip = source_clip()
        self.core = object()

    def _folders(self):
        return list(self.root.iterdir())

    def test_cancel_before_provider_retains_no_fictitious_source(self):
        provider = Mock()
        builder = PairedSceneBuilder(request(source="generate"), provider, self.core,
                                     self.root, planner=Mock())
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            builder(SCENE, cancelled=lambda: True)
        provider.generate.assert_not_called()
        folder, = self._folders()
        self.assertFalse((folder / "source-pair.npz").exists())
        self.assertEqual(json.loads((folder / "failure.json").read_text())["sources_retained"], False)

    def test_failed_route_keeps_generated_pair_archive(self):
        provider = Mock(generate=Mock(return_value=self.clip))
        planner = Mock(plan=Mock(return_value="They greet each other."))
        builder = PairedSceneBuilder(request(source="generate"), provider, self.core,
                                     self.root, planner=planner)
        with (patch("paired_direction.preview_direction", return_value={}),
              patch("paired_meetup.plan_meetup", side_effect=ValueError("Route blocked"))):
            with self.assertRaisesRegex(ValueError, "Route blocked"):
                builder(SCENE)
        provider.generate.assert_called_once()
        folder, = self._folders()
        with np.load(folder / "source-pair.npz") as saved:
            np.testing.assert_array_equal(saved["joints"], self.clip.joints)
        self.assertTrue(json.loads((folder / "failure.json").read_text())["sources_retained"])
        self.assertFalse((folder / "composed.npz").exists())

    def test_success_returns_exact_builder_clip_and_persists_report(self):
        placement = {"x": 1., "z": 2., "yaw_degrees": 0.}
        compiled = NativePairClip(self.clip.joints.copy(), metadata={"model": "ARDY Core + InterGen"})
        expected = {"clip": compiled, "placement": placement, "report": {"checked": True}}
        builder = PairedSceneBuilder(request(), None, self.core, self.root)
        with (patch("paired_direction.preview_direction", return_value={}),
              patch("paired_direction.library_clip", return_value=self.clip),
              patch("paired_meetup.plan_meetup", return_value={"routes": []}),
              patch("paired_meetup.build_meetup", return_value=expected) as build):
            result = builder(SCENE)
        build.assert_called_once()
        self.assertIs(result["clip"].__class__, NativePairClip)
        np.testing.assert_array_equal(result["clip"].joints, compiled.joints)
        self.assertEqual(result["placement"], placement)
        folder, = self._folders()
        self.assertTrue((folder / "source-pair.npz").exists())
        self.assertTrue((folder / "composed.npz").exists())
        self.assertEqual(json.loads((folder / "report.json").read_text()), expected["report"])
        self.assertEqual(result["clip"].metadata["direction_request"]["actor_ids"], ["ava", "ben"])


if __name__ == "__main__":
    unittest.main()
