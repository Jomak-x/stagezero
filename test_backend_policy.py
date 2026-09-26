"""CPU-only integration checks for the Pod generation policy.

The official model entrypoints are stubbed before importing pod_backend. The
real torch CPU tensors and real motion_policy candidate selector are used.
"""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

from motion_quality import HANDS, SHOULDERS


ROOT = Path(__file__).resolve().parent


def load_backend(token_path):
    fake_ardy = types.ModuleType("ardy")
    fake_model = types.ModuleType("ardy.model")
    fake_tools = types.ModuleType("ardy.tools")
    fake_psutil = types.ModuleType("psutil")
    fake_goals = types.ModuleType("motion_action_goals")
    fake_goals.build_action_conditions = lambda *args, **kwargs: (None, None, {"action": args[1]})
    fake_model.load_model = lambda *args, **kwargs: None
    fake_tools.seed_everything = lambda seed: None
    fake_psutil.Process = lambda: types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=0))
    spec = importlib.util.spec_from_file_location("pod_backend_policy_test", ROOT / "pod_backend.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"ardy": fake_ardy, "ardy.model": fake_model, "ardy.tools": fake_tools,
                                       "psutil": fake_psutil, "motion_action_goals": fake_goals}), \
         mock.patch.dict(os.environ, {"STAGEZERO_TOKEN_FILE": str(token_path)}):
        spec.loader.exec_module(module)
    return module


class FakeMotionRep:
    def __init__(self):
        self.inverse_calls = []
        self.unsafe_markers = set()

    def inverse(self, motion, *, is_normalized):
        assert is_normalized is True
        marker = float(motion[0, 0, 0])
        frames = motion.shape[1]
        self.inverse_calls.append((frames, marker))
        positions = np.zeros((1, frames, 34, 3), dtype=np.float32)
        positions[:, :, 0, 1] = 1.0
        positions[:, :, SHOULDERS, 1] = 1.4
        positions[:, :, HANDS, 1] = 1.0
        # Candidate 1 is the expressive alternative: both hands overhead.
        if frames == 104 and marker == 21:
            positions[:, :, HANDS, 1] = 1.7
        if frames == 104 and marker in self.unsafe_markers:
            positions[:, :, 7, 1] = -.06
        rotations = np.tile(np.eye(3, dtype=np.float32), (1, frames, 34, 1, 1))
        return {"posed_joints": torch.from_numpy(positions),
                "global_rot_mats": torch.from_numpy(rotations)}


class FakeModel:
    def __init__(self, on_step=None):
        self.num_frames_per_token = 4
        self.diffusion = types.SimpleNamespace(num_base_steps=8)
        self.motion_rep = FakeMotionRep()
        self.on_step = on_step
        self.steps = []
        self.encode_calls = 0

    def _encode_text(self, prompts):
        self.encode_calls += 1
        return torch.zeros((1, 1)), torch.zeros((1, 1))

    def autoregressive_step(self, **kwargs):
        call = len(self.steps)
        candidate, horizon = divmod(call, 2)
        history_len = 0 if kwargs["init_history_sequence"] is None else kwargs["init_history_sequence"].shape[1]
        self.steps.append({"candidate": candidate, "horizon": horizon,
                           "history_len": history_len, "num_frames": kwargs["num_frames"],
                           "cfg_weight": kwargs["cfg_weight"]})
        assert kwargs["num_frames"] == history_len + 52
        # The sentinel is the retained history. Every generated horizon has a
        # distinct marker; incorrect crops leak -99 or repeat the first chunk.
        result = torch.full((1, kwargs["num_frames"], 414), -99.0)
        result[:, history_len:] = 10 * (candidate + 1) + horizon + 1
        if self.on_step is not None:
            self.on_step(len(self.steps))
        return result


class BackendPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp.name)
        token = cls.temp_path / "token"
        token.write_text("unit-test-token")
        cls.backend = load_backend(token)
        cls.original_cwd = Path.cwd()
        os.chdir(cls.temp_path)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.original_cwd)
        cls.temp.cleanup()

    def setUp(self):
        self.backend.CANCELLED.clear()
        self.backend.STATE.update(ready=True, features=414, error=None)
        self.model = FakeModel()
        self.backend.model = self.model
        self.seeds = []
        self.backend.seed_everything = self.seeds.append
        metrics_file = self.temp_path / "metrics.jsonl"
        if metrics_file.exists():
            metrics_file.unlink()

    def generate(self, body):
        with mock.patch.object(torch.Tensor, "to", lambda tensor, *args, **kwargs: tensor), \
             mock.patch.object(torch.cuda, "reset_peak_memory_stats"), \
             mock.patch.object(torch.cuda, "synchronize"), \
             mock.patch.object(torch.cuda, "max_memory_allocated", return_value=0), \
             mock.patch.object(torch.cuda, "max_memory_reserved", return_value=0), \
             redirect_stdout(io.StringIO()):
            payload = self.backend.generate(body)
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            return {"motion": data["motion"].copy(),
                    "positions": data["positions"].copy(),
                    "metadata": json.loads(str(data["metadata"]))}

    def test_explicit_legacy_uses_52_frame_history_and_crops_both_horizons(self):
        history = np.zeros((52, 414), dtype=np.float32)
        result = self.generate({"request_id": "legacy", "prompt": "raise both arms overhead",
                                "history": history.tolist(),
                                "generation_options": {"profile": "legacy", "seed": 77}})
        self.assertEqual(self.seeds, [77])
        self.assertEqual([step["history_len"] for step in self.model.steps], [52, 52])
        self.assertEqual([step["cfg_weight"] for step in self.model.steps], [(2.0, 2.0)] * 2)
        self.assertEqual(result["motion"].shape, (104, 414))
        np.testing.assert_array_equal(result["motion"][:52], np.full((52, 414), 11))
        np.testing.assert_array_equal(result["motion"][52:], np.full((52, 414), 12))
        self.assertEqual(result["metadata"]["seed"], 77)
        self.assertEqual(result["metadata"]["candidate_settings"][0]["profile"], "legacy")
        self.assertEqual(result["metadata"]["selection"]["chosen_index"], 0)

    def test_auto_overhead_compares_responsive_and_expressive_with_same_seed(self):
        history = np.zeros((52, 414), dtype=np.float32)
        result = self.generate({"request_id": "auto", "prompt": "A person raises both arms overhead.",
                                "history": history.tolist(), "generation_options": {"seed": 42, "pose_goal": False}})
        self.assertEqual(self.seeds, [42, 42])
        self.assertEqual([step["history_len"] for step in self.model.steps], [4, 4, 12, 12])
        self.assertEqual([step["cfg_weight"] for step in self.model.steps],
                         [(2.0, 2.0)] * 2 + [(4.0, 2.0)] * 2)
        settings = result["metadata"]["candidate_settings"]
        self.assertEqual([entry["profile"] for entry in settings], ["responsive", "expressive"])
        self.assertEqual(result["metadata"]["selection"]["chosen_index"], 1)
        self.assertEqual(result["metadata"]["selection"]["action"], "overhead")
        np.testing.assert_array_equal(result["motion"][:52], np.full((52, 414), 21))
        np.testing.assert_array_equal(result["motion"][52:], np.full((52, 414), 22))

    def test_explicit_repeat_candidates_get_recorded_distinct_seeds(self):
        result = self.generate({"request_id": "repeats", "prompt": "walk",
                                "generation_options": {"profile": "responsive", "seed": 9, "candidates": 2}})
        self.assertEqual(self.seeds, [9, 10])
        self.assertEqual([entry["seed"] for entry in result["metadata"]["candidate_settings"]], [9, 10])
        self.assertEqual([step["history_len"] for step in self.model.steps], [0, 4, 0, 4])

    def test_selected_seed_matches_selected_repeat(self):
        result = self.generate({"request_id": "selected-repeat", "prompt": "raise both arms overhead",
                                "generation_options": {"profile": "responsive", "seed": 9, "candidates": 2}})
        self.assertEqual(result["metadata"]["selection"]["chosen_index"], 1)
        self.assertEqual(result["metadata"]["seed"], 10)
        self.assertEqual(result["metadata"]["base_seed"], 9)

    def test_automatic_quality_failure_recovers_with_longer_history_and_new_seed(self):
        self.model.motion_rep.unsafe_markers = {11}
        result = self.generate({"request_id": "recovery", "prompt": "Do a backflip",
                                "history": np.zeros((52, 414)).tolist(),
                                "generation_options": {"seed": 42}})
        self.assertEqual(result["metadata"]["selection"]["chosen_index"], 1)
        self.assertIn("floor_penetration", result["metadata"]["selection"]["assessments"][0]["reasons"])
        self.assertEqual([entry["profile"] for entry in result["metadata"]["candidate_settings"]],
                         ["responsive", "legacy"])
        self.assertEqual([step["history_len"] for step in self.model.steps], [4, 4, 52, 52])
        self.assertEqual(self.seeds, [42, 43])
        self.assertEqual(result["metadata"]["seed"], 43)
        self.assertEqual(self.model.encode_calls, 1)
        np.testing.assert_array_equal(result["motion"][:52], np.full((52, 414), 21))

    def test_default_request_stops_after_first_safe_candidate(self):
        result = self.generate({"request_id": "default", "prompt": "Do a backflip"})
        self.assertEqual(result["metadata"]["selection"]["chosen_index"], 0)
        self.assertEqual(len(result["metadata"]["candidate_settings"]), 1)
        self.assertEqual(len(self.model.steps), 2)

    def test_default_request_recovers_without_supplied_options(self):
        self.model.motion_rep.unsafe_markers = {11}
        with mock.patch.object(self.backend.secrets, "randbelow", return_value=42):
            result = self.generate({"request_id": "default-recovery", "prompt": "Do a backflip"})
        self.assertEqual(result["metadata"]["selection"]["chosen_index"], 1)
        self.assertEqual(self.seeds, [42, 43])

    def test_automatic_recovery_can_use_third_candidate_and_wrap_seed(self):
        self.model.motion_rep.unsafe_markers = {11, 21}
        result = self.generate({"request_id": "third", "prompt": "Do a backflip",
                                "generation_options": {"seed": 2**32 - 1}})
        self.assertEqual(result["metadata"]["selection"]["chosen_index"], 2)
        self.assertEqual(self.seeds, [2**32 - 1, 0, 1])
        self.assertEqual(len(self.model.steps), 6)
        self.assertEqual(result["metadata"]["seed"], 1)

    def test_exhausted_quality_recovery_logs_reasons_and_settings(self):
        self.model.motion_rep.unsafe_markers = {11, 21, 31}
        with self.assertRaisesRegex(RuntimeError, "floor penetration") as failure:
            self.generate({"request_id": "exhausted", "prompt": "Do a backflip",
                           "generation_options": {"seed": 42}})
        self.assertLess(len(str(failure.exception)), 180)
        self.assertIn("No motion committed", str(failure.exception))
        self.assertNotIn("take preserved", str(failure.exception))
        rows = [json.loads(line) for line in (self.temp_path / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "rejected")
        self.assertEqual(rows[0]["request_id"], "exhausted")
        self.assertIsNone(rows[0]["selection"]["chosen_index"])
        self.assertTrue(all(not row["safe"] for row in rows[0]["selection"]["assessments"]))
        self.assertEqual([entry["seed"] for entry in rows[0]["candidate_settings"]], [42, 43, 44])
        self.assertEqual(len(self.model.steps), 6)

    def test_explicit_candidate_budgets_never_expand_on_quality_failure(self):
        for options, count in (({"profile": "legacy", "seed": 42}, 1),
                               ({"seed": 42, "candidates": 1}, 1),
                               ({"seed": 42, "candidates": 2}, 2),
                               ({"profile": "responsive", "seed": 42, "candidates": 3}, 3)):
            with self.subTest(options=options):
                self.model = FakeModel()
                self.backend.model = self.model
                self.model.motion_rep.unsafe_markers = {11, 21, 31}
                with self.assertRaises(RuntimeError):
                    self.generate({"request_id": "explicit", "prompt": "Do a backflip",
                                   "generation_options": options})
                self.assertEqual(len(self.model.steps), count * 2)
                rows = (self.temp_path / "metrics.jsonl").read_text().splitlines()
                self.assertEqual(len(json.loads(rows[-1])["candidate_settings"]), count)

    def test_cancellation_after_failed_candidate_does_not_start_recovery(self):
        self.model.motion_rep.unsafe_markers = {11}
        self.model.on_step = lambda count: self.backend.CANCELLED.add("cancel-recovery") if count == 2 else None
        with self.assertRaises(InterruptedError):
            self.generate({"request_id": "cancel-recovery", "prompt": "Do a backflip",
                           "generation_options": {"seed": 42}})
        self.assertEqual(len(self.model.steps), 2)
        self.assertEqual(self.seeds, [42])
        self.assertFalse((self.temp_path / "metrics.jsonl").exists())

    def test_cancellation_during_recovery_does_not_finish_or_write_result(self):
        self.model.motion_rep.unsafe_markers = {11}
        self.model.on_step = lambda count: self.backend.CANCELLED.add("cancel-recovery-mid") if count == 3 else None
        with self.assertRaises(InterruptedError):
            self.generate({"request_id": "cancel-recovery-mid", "prompt": "Do a backflip",
                           "generation_options": {"seed": 42}})
        self.assertEqual(len(self.model.steps), 3)
        self.assertEqual(self.seeds, [42, 43])
        self.assertFalse((self.temp_path / "metrics.jsonl").exists())

    def test_recovery_preserves_native_pose_goal(self):
        self.model.motion_rep.unsafe_markers = {11}
        with mock.patch.object(self.backend, "build_action_conditions", return_value=(None, None, {"action": "squat"})) as build:
            result = self.generate({"request_id": "pose-recovery", "prompt": "Do a squat",
                                    "history": np.zeros((52, 414)).tolist(),
                                    "generation_options": {"seed": 42}})
        self.assertEqual(result["metadata"]["pose_goal"], {"action": "squat"})
        self.assertEqual([call.args[1] for call in build.call_args_list], ["squat"] * 4)
        self.assertEqual([call.kwargs["generated_offset"] for call in build.call_args_list], [0, 52, 0, 52])
        self.assertEqual([step["history_len"] for step in self.model.steps], [12, 12, 52, 52])

    def test_recovery_preserves_root_waypoint(self):
        self.model.motion_rep.unsafe_markers = {11}
        target = {"position_xz": [0.0, 0.8], "frame": 103}
        with mock.patch.object(self.backend, "build_root_conditions", return_value=(None, None)) as build:
            result = self.generate({"request_id": "target-recovery", "prompt": "walk to the mark",
                                    "history": np.zeros((52, 414)).tolist(), "motion_target": target,
                                    "generation_options": {"seed": 42}})
        self.assertEqual(result["metadata"]["motion_target"], target)
        self.assertEqual([call.args[1] for call in build.call_args_list], [target] * 4)
        self.assertEqual([call.kwargs["history_length"] for call in build.call_args_list], [4, 4, 52, 52])

    def test_auto_stop_uses_expressive_profile(self):
        result = self.generate({"request_id": "stop", "prompt": "A person stops and stands still.",
                                "history": np.zeros((52, 414)).tolist(),
                                "generation_options": {"seed": 42}})
        self.assertEqual([step["history_len"] for step in self.model.steps], [12, 12])
        self.assertEqual(result["metadata"]["candidate_settings"][0]["profile"], "expressive")

    def test_auto_pose_goals_use_proven_profiles(self):
        for action, prompt, profile, history_frames in (
            ("overhead", "Raise both arms overhead", "responsive", 4),
            ("squat", "Do a squat", "expressive", 12),
        ):
            self.model.steps.clear()
            with mock.patch.object(self.backend, "build_action_conditions", return_value=(None, None, {"action": action})) as build:
                result = self.generate({"request_id": action, "prompt": prompt,
                                        "history": np.zeros((52, 414)).tolist(),
                                        "generation_options": {"seed": 42}})
            self.assertEqual(len(result["metadata"]["candidate_settings"]), 1)
            self.assertEqual(result["metadata"]["candidate_settings"][0]["profile"], profile)
            self.assertEqual(result["metadata"]["pose_goal"], {"action": action})
            self.assertEqual([call.kwargs["generated_offset"] for call in build.call_args_list], [0, 52])
            self.assertEqual([step["history_len"] for step in self.model.steps], [history_frames] * 2)

    def test_explicit_profile_disables_automatic_pose_goals(self):
        with mock.patch.object(self.backend, "build_action_conditions") as build:
            result = self.generate({"request_id": "text-only", "prompt": "Raise both arms overhead",
                                    "generation_options": {"profile": "legacy", "seed": 1}})
        build.assert_not_called()
        self.assertIsNone(result["metadata"]["pose_goal"])

    def test_explicit_pose_goal_and_root_waypoint_conflict(self):
        with self.assertRaisesRegex(ValueError, "either a root waypoint"):
            self.generate({"request_id": "conflict", "prompt": "Do a squat",
                           "motion_target": {"position_xz": [0., .8], "frame": 103},
                           "generation_options": {"pose_goal": True}})
        self.assertEqual(self.model.encode_calls, 0)

    def test_explicit_unsupported_pose_goal_is_rejected(self):
        for prompt in ("Walk forward", "Wave with the right hand", "Do not squat"):
            with self.assertRaisesRegex(ValueError, "unambiguous overhead raise or squat"):
                self.generate({"request_id": "unsupported", "prompt": prompt,
                               "generation_options": {"pose_goal": True}})
        self.assertEqual(self.model.encode_calls, 0)

    def test_root_target_is_scheduled_and_reported(self):
        target = {"position_xz": [0.0, 0.8], "frame": 103}
        with mock.patch.object(self.backend, "build_root_conditions", return_value=(None, None)) as build:
            result = self.generate({"request_id": "target", "prompt": "walk to the mark",
                                    "history": np.zeros((52, 414)).tolist(), "motion_target": target,
                                    "generation_options": {"seed": 42}})
        self.assertEqual([call.kwargs["generated_offset"] for call in build.call_args_list], [0, 52])
        self.assertEqual([call.kwargs["history_length"] for call in build.call_args_list], [4, 4])
        self.assertEqual(result["metadata"]["motion_target"], target)
        self.assertAlmostEqual(result["metadata"]["target_error_m"], .8)

    def test_root_target_requires_history_before_encoding(self):
        with self.assertRaisesRegex(ValueError, "requires prior actor history"):
            self.generate({"request_id": "target-no-history", "prompt": "walk",
                           "motion_target": {"position_xz": [0., .8], "frame": 103}})
        self.assertEqual(self.model.encode_calls, 0)

    def test_cancellation_between_candidates_writes_no_result(self):
        self.model.on_step = lambda count: self.backend.CANCELLED.add("cancel") if count == 2 else None
        with self.assertRaises(InterruptedError):
            self.generate({"request_id": "cancel", "prompt": "A person raises both arms overhead.",
                           "generation_options": {"seed": 3, "pose_goal": False}})
        self.assertEqual(len(self.model.steps), 2)
        self.assertFalse((self.temp_path / "metrics.jsonl").exists())

    def test_cancellation_within_candidate_skips_second_horizon_and_output(self):
        self.model.on_step = lambda count: self.backend.CANCELLED.add("cancel-mid") if count == 1 else None
        with self.assertRaises(InterruptedError):
            self.generate({"request_id": "cancel-mid", "prompt": "A person raises both arms overhead.",
                           "generation_options": {"seed": 4}})
        self.assertEqual(len(self.model.steps), 1)
        self.assertEqual(self.model.motion_rep.inverse_calls, [])
        self.assertFalse((self.temp_path / "metrics.jsonl").exists())

    def test_malformed_options_fail_before_model_or_encoder(self):
        malformed = ({"profile": "unknown"}, {"seed": True}, {"seed": -1},
                     {"candidates": 0}, {"candidates": 4}, {"mystery": 1}, [])
        for index, options in enumerate(malformed):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.generate({"request_id": f"bad-{index}", "prompt": "walk",
                               "generation_options": options})
        self.assertEqual(self.model.encode_calls, 0)
        self.assertEqual(self.model.steps, [])
        self.assertEqual(self.seeds, [])
        self.assertFalse((self.temp_path / "metrics.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
