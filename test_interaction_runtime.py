"""CPU contract tests for the isolated ARDY Core interaction runtime."""

from __future__ import annotations

import io
import json
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from interaction_runtime import (
    FEATURES, GenerationCancelled, InteractionRuntime, build_conditions,
    encode_npz, targets_in_window, validate_request,
)


def request(*actors, frames=80):
    return {"request_id": "scene-1", "frames": frames, "actors": list(actors)}


def actor(actor_id="a", seed=7, **values):
    return {"id": actor_id, "prompt": "Walk through the gate.", "seed": seed, **values}


class FakeRoot2DConstraintSet:
    def __init__(self, skeleton, frame_indices, root_2d, global_root_heading=None):
        self.frame_indices = frame_indices
        self.root_2d = root_2d
        self.global_root_heading = global_root_heading


class FakeRepresentation:
    fps = 20
    motion_rep_dim = FEATURES

    def unnormalize(self, motion):
        return motion.clone()

    def normalize(self, motion):
        return motion.clone()

    def get_root_pos(self, motion):
        return motion[..., :3]

    def rotate_to(self, motion, yaw):
        output = motion.clone()
        output[..., 3] = yaw[:, None]
        return output

    def translate_2d_to(self, motion, targets):
        output = motion.clone()
        output[..., 0] += targets[:, 0, None] - output[:, 0, 0, None]
        output[..., 2] += targets[:, 1, None] - output[:, 0, 2, None]
        return output

    def create_conditions_from_constraints_batched(self, per_actor, lengths, *, to_normalize, device):
        result = torch.zeros(len(per_actor), int(lengths.max()), FEATURES)
        mask = torch.zeros_like(result)
        for index, constraints in enumerate(per_actor):
            for constraint in constraints:
                frame = int(constraint.frame_indices[0])
                result[index, frame, 0] = constraint.root_2d[0, 0]
                result[index, frame, 2] = constraint.root_2d[0, 1]
                mask[index, frame, 0] = 1
                mask[index, frame, 2] = 1
                if constraint.global_root_heading is not None:
                    result[index, frame, 3] = constraint.global_root_heading[0]
                    mask[index, frame, 3] = 1
        return result, mask

    def inverse(self, motion, *, is_normalized):
        batch, frames, _ = motion.shape
        positions = torch.zeros(batch, frames, 27, 3)
        positions[..., 0] = motion[..., 0, None]
        positions[..., 2] = motion[..., 2, None]
        rotations = torch.eye(3).repeat(batch, frames, 27, 1, 1)
        return {"posed_joints": positions, "global_rot_mats": rotations}


class FakeModel:
    gen_horizon_len = 40
    num_frames_per_token = 4
    skeleton = types.SimpleNamespace(name="cskel27", nbjoints=27, root_idx=0)
    diffusion = types.SimpleNamespace(num_base_steps=10)

    def __init__(self):
        self.motion_rep = FakeRepresentation()
        self.calls = []

    def _encode_text(self, prompts):
        return torch.ones(len(prompts), 1), torch.ones(len(prompts), 1)

    def autoregressive_step(self, *, num_frames, motion_mask, observed_motion,
                            init_history_sequence, text_feat, init_global_translation,
                            init_first_heading_angle, **kwargs):
        batch = len(text_feat)
        initial = 0 if init_history_sequence is None else init_history_sequence.shape[1]
        self.calls.append({"batch": batch, "initial": initial, "num_frames": num_frames,
                           "mask": None if motion_mask is None else motion_mask.clone(),
                           "history": None if init_history_sequence is None else init_history_sequence.clone(),
                           "init_translation": init_global_translation,
                           "init_heading": init_first_heading_angle})
        output = torch.zeros(batch, num_frames, FEATURES)
        if initial:
            output[:, :initial] = init_history_sequence
        for frame in range(initial, num_frames):
            output[:, frame, 0] = output[:, frame - 1, 0] + .01 if frame else .01
        if motion_mask is not None:
            output = output * (1 - motion_mask) + observed_motion * motion_mask
        return output


class ContractTests(unittest.TestCase):
    def test_accepts_two_independent_histories_and_mixed_heading_targets(self):
        history = np.zeros((8, FEATURES), np.float32).tolist()
        normalized = validate_request(request(
            actor("a", history=history, root_targets=[{"frame": 39, "position_xz": [1, 2]}]),
            actor("b", seed=8, history=history, root_targets=[
                {"frame": 40, "position_xz": [-1, 2], "heading": 1.2}]), frames=80))
        self.assertEqual([x["id"] for x in normalized["actors"]], ["a", "b"])
        self.assertEqual([x["frame"] for x in normalized["actors"][1]["root_targets"]], [40])

    def test_rejects_nonfinite_values_duplicate_targets_and_wrong_core_history(self):
        bad_actors = [
            actor(root_targets=[{"frame": 0, "position_xz": [float("nan"), 0]}]),
            actor(root_targets=[{"frame": 0, "position_xz": [0, 0]},
                                        {"frame": 0, "position_xz": [1, 0]}]),
            actor(history=np.zeros((4, 414)).tolist()),
            actor(root_targets=[{"frame": 80, "position_xz": [0, 0]}]),
            actor(seed=True),
        ]
        for item in bad_actors:
            with self.subTest(item=item.keys()), self.assertRaises(ValueError):
                validate_request(request(item))

    def test_keyframes_include_actual_history_offset(self):
        item = validate_request(request(actor(root_targets=[
            {"frame": 39, "position_xz": [1, 0]},
            {"frame": 40, "position_xz": [1, 1], "heading": .5},
            {"frame": 79, "position_xz": [2, 1]},
        ])))["actors"][0]
        self.assertEqual([x["window_frame"] for x in targets_in_window(item, generated_offset=0, history_length=8)], [47])
        self.assertEqual([x["window_frame"] for x in targets_in_window(item, generated_offset=40, history_length=4)], [4, 43])

    def test_constraints_never_touch_history_and_preserve_optional_heading(self):
        model = FakeModel()
        first, second = validate_request(request(
            actor("a", root_targets=[{"frame": 40, "position_xz": [1, 2], "heading": 1.1}]),
            actor("b")))["actors"]
        with patch.dict(sys.modules, {"ardy": types.ModuleType("ardy"),
                                      "ardy.constraints": types.SimpleNamespace(Root2DConstraintSet=FakeRoot2DConstraintSet)}):
            observed, mask = build_conditions(model, [first, second], generated_offset=40,
                                              history_length=4, device="cpu")
        self.assertEqual(tuple(observed.shape), (2, 44, FEATURES))
        self.assertEqual(int(mask[:, :4].count_nonzero()), 0)
        self.assertAlmostEqual(float(observed[0, 4, 3]), 1.1)
        self.assertEqual(int(mask[1].count_nonzero()), 0)

    def test_genuine_two_actor_batch_and_native_waypoint_output(self):
        model = FakeModel()
        runtime = InteractionRuntime(model, device="cpu")
        body = request(actor("a", root_targets=[{"frame": 39, "position_xz": [1, 2]}]),
                       actor("b", root_targets=[{"frame": 79, "position_xz": [-1, 2], "heading": .5}]))
        with patch.dict(sys.modules, {"ardy": types.ModuleType("ardy"),
                                      "ardy.constraints": types.SimpleNamespace(Root2DConstraintSet=FakeRoot2DConstraintSet),
                                      "ardy.tools": types.SimpleNamespace(seed_everything=lambda seed: torch.manual_seed(seed))}):
            arrays, metadata = runtime.generate(body)
        self.assertEqual([(x["batch"], x["initial"]) for x in model.calls], [(2, 0), (2, 4)])
        self.assertEqual(arrays["actor_0_positions"].shape, (80, 27, 3))
        self.assertAlmostEqual(metadata["actors"][0]["root_targets"][0]["position_error_m"], 0)
        self.assertAlmostEqual(metadata["actors"][1]["root_targets"][0]["position_error_m"], 0)
        self.assertTrue(metadata["native_conditions"])
        self.assertEqual(metadata["actors"][0]["conditioned_frames"], [39])
        self.assertEqual(metadata["actors"][1]["conditioned_frames"], [79])
        with np.load(io.BytesIO(encode_npz(arrays, metadata)), allow_pickle=False) as saved:
            self.assertEqual(json.loads(str(saved["metadata"]))["actors"][1]["id"], "b")
            np.testing.assert_array_equal(saved["actor_1_motion"], arrays["actor_1_motion"])

    def test_distinct_seeds_keep_actor_calls_independent_and_history_immutable(self):
        model = FakeModel()
        history = np.zeros((8, FEATURES), np.float32)
        history[:, 0] = 2
        runtime = InteractionRuntime(model, device="cpu")
        body = request(actor("a", 11, history=history.tolist()),
                       actor("b", 12, history=history.tolist()), frames=40)
        with patch.dict(sys.modules, {"ardy": types.ModuleType("ardy"),
                                      "ardy.tools": types.SimpleNamespace(seed_everything=lambda seed: torch.manual_seed(seed))}):
            arrays, metadata = runtime.generate(body)
        self.assertEqual([x["batch"] for x in model.calls], [1, 1])
        self.assertEqual([x["initial"] for x in model.calls], [8, 8])
        self.assertEqual([x["seed"] for x in metadata["actors"]], [11, 12])
        self.assertFalse(metadata["native_conditions"])
        np.testing.assert_array_equal(model.calls[0]["history"][0].numpy(), history)
        self.assertEqual(arrays["actor_0_motion"].shape, (40, FEATURES))

    def test_cancellation_prevents_model_call(self):
        model = FakeModel()
        runtime = InteractionRuntime(model, device="cpu")
        with patch.dict(sys.modules, {"ardy": types.ModuleType("ardy"),
                                      "ardy.tools": types.SimpleNamespace(seed_everything=lambda seed: None)}):
            with self.assertRaises(GenerationCancelled):
                runtime.generate(request(actor()), is_cancelled=lambda _: True)
        self.assertEqual(model.calls, [])

    def test_cold_start_initial_placement_is_passed_to_native_model(self):
        model = FakeModel()
        runtime = InteractionRuntime(model, device="cpu")
        body = request(actor("a", initial_position_xz=[-1.5, 2], initial_yaw=1.2), frames=40)
        with patch.dict(sys.modules, {"ardy": types.ModuleType("ardy"),
                                      "ardy.tools": types.SimpleNamespace(seed_everything=lambda seed: None)}):
            _, metadata = runtime.generate(body)
        np.testing.assert_allclose(model.calls[0]["init_translation"].numpy(), [[-1.5, 0, 2]])
        np.testing.assert_allclose(model.calls[0]["init_heading"].numpy(), [1.2])
        self.assertEqual(metadata["actors"][0]["initial_position_xz"], [-1.5, 2.0])

    def test_history_placement_changes_only_copied_model_input(self):
        model = FakeModel()
        runtime = InteractionRuntime(model, device="cpu")
        history = np.zeros((4, FEATURES), np.float32)
        history[:, 0] = 5
        history[:, 2] = -3
        history[:, 3] = .1
        body = request(actor("a", history=history.tolist(), initial_position_xz=[1, 2],
                             initial_yaw=1.0), frames=40)
        original = json.dumps(body)
        with patch.dict(sys.modules, {"ardy": types.ModuleType("ardy"),
                                      "ardy.tools": types.SimpleNamespace(seed_everything=lambda seed: None)}):
            runtime.generate(body)
        placed = model.calls[0]["history"].numpy()[0]
        self.assertAlmostEqual(float(placed[0, 0]), 1)
        self.assertAlmostEqual(float(placed[0, 2]), 2)
        self.assertAlmostEqual(float(placed[0, 3]), 1)
        self.assertEqual(json.dumps(body), original)

    def test_trusted_extra_constraint_hook_is_not_client_supplied(self):
        with self.assertRaises(ValueError):
            validate_request(request(actor(hand_targets=[{"frame": 1}])))
        model = FakeModel()
        normalized = validate_request(request(actor()))["actors"]
        with patch.dict(sys.modules, {"ardy": types.ModuleType("ardy"),
                                      "ardy.constraints": types.SimpleNamespace(Root2DConstraintSet=FakeRoot2DConstraintSet)}):
            observed, mask = build_conditions(model, normalized, generated_offset=0,
                history_length=0, device="cpu", condition_hook=lambda **_: [FakeRoot2DConstraintSet(
                    model.skeleton, torch.tensor([20]), torch.tensor([[.5, .5]]))])
        self.assertEqual(int(mask.count_nonzero()), 2)
        self.assertAlmostEqual(float(observed[0, 20, 0]), .5)


if __name__ == "__main__":
    unittest.main()
