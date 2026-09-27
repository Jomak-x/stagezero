"""Simulated transport tests for strict scene-grounded direction planning.

No test here contacts an AI provider or claims generated motion succeeded.
"""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import requests

from scene_ai_planner import SceneAIPlanner, compile_actions, plan_local, validate_plan
from scene_objects import make_object


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def iter_content(self, _size):
        yield self.payload


class FakeTransport:
    def __init__(self, response=None, error=None):
        self.response, self.error = response, error
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.response


def envelope(doc, *, finish_reason="stop", refusal=None):
    message = {"content": json.dumps(doc)}
    if refusal:
        message["refusal"] = refusal
    return json.dumps({"choices": [{"finish_reason": finish_reason, "message": message}]}).encode()


class SceneAIPlannerTests(unittest.TestCase):
    def setUp(self):
        arch = make_object("arch", 0)
        arch["name"] = "Gate"
        arch["position"] = [0., 1.5, 0.]
        arch["size"] = [2.7, 3., .38]
        ball = make_object("ball", 1)
        ball["position"] = [2., .15, 0.]
        self.scene = {"version": 2, "name": "Lab", "objects": [arch, ball],
                      "effects": [], "lighting": "neutral"}
        self.actors = [{"id": "alex", "name": "Alex", "position": [0., 0., -3.]},
                       {"id": "bea", "name": "Bea", "position": [2., 0., -3.]}]

    def test_rule_based_gate_command_uses_actual_arch_and_compiles_route(self):
        plan = plan_local("Alex go through the gate", self.scene, self.actors)
        self.assertEqual(plan, {"source": "rule_based", "actions": [
            {"actor_id": "alex", "verb": "go_through", "target_id": "arch-0"}]})
        route = compile_actions(plan, self.scene, self.actors)[0]["geometry"]
        self.assertEqual(route["target_id"], "arch-0")
        self.assertEqual([point["role"] for point in route["waypoints"]][-3:],
                         ["entry", "center", "exit"])
        self.assertFalse(route["assumptions"]["motion_following_verified"])

    def test_unknown_object_injection_and_unverified_passage_are_rejected(self):
        action = {"actor_id": "alex", "verb": "go_through", "target_id": "imaginary-gate"}
        with self.assertRaisesRegex(ValueError, "ungrounded"):
            validate_plan({"actions": [action]}, self.scene, self.actors)
        door = make_object("door", 2)
        scene = dict(self.scene, objects=self.scene["objects"] + [door])
        action["target_id"] = door["id"]
        with self.assertRaisesRegex(ValueError, "ungrounded"):
            validate_plan({"actions": [action]}, scene, self.actors)
        another_arch = make_object("arch", 3)
        another_arch["position"] = [4., 1.5, 0.]
        another_arch["size"] = [2.7, 3., .38]
        scene["objects"].append(another_arch)
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            plan_local("Alex go through the gate", scene, self.actors)
        transport = FakeTransport(FakeResponse(envelope({"actions": []})))
        planner = SceneAIPlanner("https://gateway.example/v1", "test-model", "test-secret",
                                 transport=transport)
        with self.assertRaisesRegex(ValueError, "specify an exact target ID"):
            planner.plan("Alex go through the gate", scene, self.actors)
        self.assertEqual(transport.calls, [])

    def test_custom_mesh_requires_trusted_opening_metadata(self):
        custom = {"id": "portal-1", "name": "Portal", "kind": "custom", "asset": "asset-1",
                  "position": [0., 1.5, 0.], "size": [3., 3., .5], "yaw": 0.}
        scene = {"version": 3, "objects": [custom], "assets": [{"id": "asset-1"}]}
        action = {"actor_id": "alex", "verb": "go_through", "target_id": "portal-1"}
        with self.assertRaisesRegex(ValueError, "ungrounded"):
            validate_plan({"actions": [action]}, scene, self.actors)
        affordances = {"portal-1": {"kind": "passage", "verified_open": True,
                                     "width_m": 2., "height_m": 2.5, "depth_m": .4, "floor_y_m": 0.}}
        self.assertEqual(validate_plan({"actions": [action]}, scene, self.actors, affordances)["actions"][0], action)
        route = compile_actions({"actions": [action]}, scene, self.actors, affordances)[0]["geometry"]
        self.assertEqual(route["geometry"]["source"], "explicit_affordance")

    def test_symbolic_social_actions_validate_but_do_not_fake_motion(self):
        plan = {"actions": [{"actor_id": "alex", "verb": "handoff", "target_id": "bea"}]}
        self.assertEqual(validate_plan(plan, self.scene, self.actors), plan)
        with self.assertRaisesRegex(ValueError, "symbolic"):
            compile_actions(plan, self.scene, self.actors)
        reach = {"actions": [{"actor_id": "alex", "verb": "reach", "target_id": "ball-1"}]}
        with self.assertRaisesRegex(ValueError, "symbolic"):
            compile_actions(reach, self.scene, self.actors)

    def test_two_routes_for_one_actor_chain_position_and_time(self):
        plan = {"actions": [
            {"actor_id": "alex", "verb": "approach", "target_id": "ball-1"},
            {"actor_id": "alex", "verb": "approach", "target_id": "arch-0"},
        ]}
        first, second = compile_actions(plan, self.scene, self.actors)
        first_end = first["geometry"]["waypoints"][-1]
        second_start = second["geometry"]["waypoints"][0]
        self.assertEqual(second_start["position_xz"], first_end["position_xz"])
        self.assertEqual(second_start["time_seconds"], first_end["time_seconds"])
        self.assertEqual(second["action"]["start_seconds"], first_end["time_seconds"])
        plan["actions"][1]["start_seconds"] = 0.0
        with self.assertRaisesRegex(ValueError, "before the preceding route ends"):
            compile_actions(plan, self.scene, self.actors)

    def test_extra_fields_bad_types_and_unknown_affordance_ids_fail_closed(self):
        action = {"actor_id": "alex", "verb": "approach", "target_id": "ball-1"}
        bad = (dict(action, code="import os"), dict(action, start_seconds=True),
               dict(action, start_seconds=float("nan")), dict(action, verb="delete"),
               dict(action, target_id="https://example.com"))
        for item in bad:
            with self.subTest(item=item), self.assertRaises(ValueError):
                validate_plan({"actions": [item]}, self.scene, self.actors)
        with self.assertRaises(ValueError):
            validate_plan({"actions": [action]}, self.scene, self.actors,
                          {"unknown": {"kind": "passage"}})
        elevated = [dict(self.actors[0], position=[0., 1.2, -3.])]
        with self.assertRaisesRegex(ValueError, "ground anchor"):
            validate_plan({"actions": [action]}, self.scene, elevated)

    def test_mock_openai_request_is_schema_bounded_and_triple_validated(self):
        doc = {"actions": [{"actor_id": "alex", "verb": "go_through", "target_id": "arch-0",
                            "start_seconds": 1.25}]}
        response = FakeResponse(envelope(doc))
        transport = FakeTransport(response)
        planner = SceneAIPlanner("https://api.openai.com/v1", "test-model", "test-secret", transport=transport)
        result = planner.plan("Alex through the gate", self.scene, self.actors)
        self.assertEqual(result["source"], "ai")
        self.assertEqual(result["actions"][0]["start_seconds"], 1.25)
        args, kwargs = transport.calls[0]
        self.assertEqual(args, ("https://api.openai.com/v1/chat/completions",))
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-secret")
        self.assertEqual(kwargs["json"]["response_format"]["type"], "json_schema")
        self.assertTrue(kwargs["json"]["response_format"]["json_schema"]["strict"])
        self.assertFalse(kwargs["allow_redirects"])
        self.assertTrue(kwargs["stream"])
        self.assertTrue(response.closed)

    def test_gateway_failure_and_invalid_output_never_become_rule_based_success(self):
        for response, error in ((FakeResponse(envelope({"actions": [
                {"actor_id": "alex", "verb": "go_through", "target_id": "made-up", "start_seconds": None}]})), None),
                (FakeResponse(envelope({"actions": []})), None),
                (FakeResponse(envelope({"actions": []}), 502), None),
                (None, requests.Timeout("offline"))):
            with self.subTest(status=getattr(response, "status_code", None), error=error):
                transport = FakeTransport(response, error)
                planner = SceneAIPlanner("https://gateway.example/v1", "test-model", "test-secret",
                                         transport=transport)
                with self.assertRaises(ValueError):
                    planner.plan("go through the gate", self.scene, self.actors)
                if response is not None:
                    self.assertTrue(response.closed)
        with mock.patch("scene_ai_planner._config", return_value={}):
            with self.assertRaisesRegex(ValueError, "no AI request was sent"):
                SceneAIPlanner.from_env()

    def test_model_cannot_substitute_gate_for_nonexistent_named_object(self):
        response = FakeResponse(envelope({"actions": [{"actor_id": "alex", "verb": "go_through",
                                                      "target_id": "arch-0", "start_seconds": 0}]}))
        planner = SceneAIPlanner("https://gateway.example/v1", "test-model", "test-secret",
                                 transport=FakeTransport(response))
        with self.assertRaisesRegex(ValueError, "was not named"):
            planner.plan("Alex go through the rocket hatch", self.scene, [self.actors[0]])

    def test_existing_private_neon_gateway_config_is_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "objects.env"
            config.write_text("NEON_AI_GATEWAY_BASE_URL=https://br-test-api.ai.us-east-2.aws.neon.tech\n"
                              "NEON_AI_GATEWAY_TOKEN=test-secret\nSTAGEZERO_OBJECT_MODEL=test-model\n")
            with mock.patch.dict(os.environ, {}, clear=True):
                planner = SceneAIPlanner.from_env(config_path=config)
        self.assertEqual(planner.url,
                         "https://br-test-api.ai.us-east-2.aws.neon.tech/v1/chat/completions")
        self.assertEqual(planner.model, "gpt-6-astra")

    def test_neon_planning_defaults_to_strong_model(self):
        values = {"NEON_AI_GATEWAY_BASE_URL": "https://branch.example",
                  "NEON_AI_GATEWAY_TOKEN": "test-secret",
                  "STAGEZERO_OBJECT_MODEL": "gpt-5-6-sol"}
        with mock.patch("scene_ai_planner._config", return_value=values):
            planner = SceneAIPlanner.from_env()
        self.assertEqual(planner.url, "https://branch.example/v1/chat/completions")
        self.assertEqual(planner.model, "gpt-6-astra")
        values["STAGEZERO_SCENE_AI_MODEL"] = "planner-choice"
        with mock.patch("scene_ai_planner._config", return_value=values):
            self.assertEqual(SceneAIPlanner.from_env().model, "planner-choice")

    def test_other_planning_providers_retain_explicit_model_or_fail_closed(self):
        values = {"STAGEZERO_SCENE_AI_API_BASE": "https://other.example/v1",
                  "STAGEZERO_SCENE_AI_API_KEY": "other-secret",
                  "NEON_AI_GATEWAY_BASE_URL": "https://branch.example",
                  "STAGEZERO_OBJECT_MODEL": "general-choice"}
        with mock.patch("scene_ai_planner._config", return_value=values):
            planner = SceneAIPlanner.from_env()
        self.assertEqual(planner.url, "https://other.example/v1/chat/completions")
        self.assertEqual(planner.model, "general-choice")
        values.pop("STAGEZERO_OBJECT_MODEL")
        with mock.patch("scene_ai_planner._config", return_value=values):
            with self.assertRaisesRegex(ValueError, "no AI request was sent"):
                SceneAIPlanner.from_env()


if __name__ == "__main__":
    unittest.main()
