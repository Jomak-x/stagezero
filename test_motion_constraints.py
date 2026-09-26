"""CPU checks for the optional ARDY root waypoint adapter."""

import sys
import types
import unittest
from unittest import mock

import torch

from motion_constraints import build_root_conditions, validate_motion_target


class FakeRoot2DConstraintSet:
    def __init__(self, skeleton, *, frame_indices, root_2d):
        self.skeleton = skeleton
        self.frame_indices = frame_indices
        self.root_2d = root_2d


class FakeMotionRep:
    motion_rep_dim = 414

    def __init__(self):
        self.calls = []
        self.overlap = False
        self.empty = False

    def create_conditions_from_constraints_batched(self, constraints, lengths, *, to_normalize, device):
        self.calls.append((constraints, lengths, to_normalize, device))
        total = int(lengths[0])
        observed = torch.zeros((1, total, 414), dtype=torch.float32)
        mask = torch.zeros_like(observed)
        frame = int(constraints[0][0].frame_indices[0])
        observed[:, frame, 0:2] = constraints[0][0].root_2d
        if not self.empty:
            mask[:, frame, 0:2] = 1
        if self.overlap:
            mask[:, 0, 0] = 1
        return observed, mask


class MotionConstraintTests(unittest.TestCase):
    def setUp(self):
        self.rep = FakeMotionRep()
        self.model = types.SimpleNamespace(gen_horizon_len=52, num_frames_per_token=4,
                                           skeleton=object(), motion_rep=self.rep)
        self.ardy = types.ModuleType("ardy")
        self.constraints = types.ModuleType("ardy.constraints")
        self.constraints.Root2DConstraintSet = FakeRoot2DConstraintSet
        self.modules = mock.patch.dict(sys.modules, {"ardy": self.ardy, "ardy.constraints": self.constraints})
        self.modules.start()

    def tearDown(self):
        self.modules.stop()

    def test_validation_bounds_and_types(self):
        self.assertIsNone(validate_motion_target(None, prior_root_xz=None))
        target = validate_motion_target({"position_xz": [1, 2], "frame": 51}, prior_root_xz=[0.5, 2])
        self.assertEqual(target, {"position_xz": [1.0, 2.0], "frame": 51})
        invalid = [
            ({"position_xz": [0, 0], "frame": 51}, None),
            ({"position_xz": [3.1, 0], "frame": 51}, [0, 0]),
            ({"position_xz": [float("nan"), 0], "frame": 51}, [0, 0]),
            ({"position_xz": [True, 0], "frame": 51}, [0, 0]),
            ({"position_xz": [0, 0], "frame": True}, [0, 0]),
            ({"position_xz": [0, 0], "frame": 52}, [0, 0]),
            ({"position_xz": [0, 0], "frame": 51, "extra": 1}, [0, 0]),
        ]
        for value, prior in invalid:
            with self.subTest(value=value, prior=prior), self.assertRaises(ValueError):
                validate_motion_target(value, prior_root_xz=prior)

    def test_first_horizon_target_maps_after_cropped_history(self):
        target = {"position_xz": [0.2, 0.8], "frame": 51}
        observed, mask = build_root_conditions(self.model, target, history_length=4,
                                                generated_offset=0, device="cpu")
        self.assertEqual(tuple(observed.shape), (1, 56, 414))
        self.assertEqual(tuple(mask.shape), (1, 56, 414))
        constraint = self.rep.calls[0][0][0][0]
        self.assertEqual(int(constraint.frame_indices[0]), 55)
        self.assertAlmostEqual(float(constraint.root_2d[0, 0]), 0.2)
        self.assertAlmostEqual(float(constraint.root_2d[0, 1]), 0.8)
        self.assertEqual(self.rep.calls[0][1].tolist(), [56])
        self.assertTrue(self.rep.calls[0][2])
        self.assertEqual(int(mask[:, :4].count_nonzero()), 0)

    def test_second_horizon_target_only_applies_to_second_step(self):
        target = {"position_xz": [0.0, 0.8], "frame": 103}
        self.assertEqual(build_root_conditions(self.model, target, history_length=12,
                                               generated_offset=0, device="cpu"), (None, None))
        self.assertEqual(len(self.rep.calls), 0)
        observed, mask = build_root_conditions(self.model, target, history_length=12,
                                               generated_offset=52, device="cpu")
        self.assertEqual(tuple(observed.shape), (1, 64, 414))
        self.assertEqual(int(self.rep.calls[0][0][0][0].frame_indices[0]), 63)
        self.assertEqual(int(mask[:, :12].count_nonzero()), 0)

    def test_rejects_wrong_model_or_history_and_bad_masks(self):
        target = {"position_xz": [0.0, 0.8], "frame": 51}
        for history in (3, 53, -4, True):
            with self.subTest(history=history), self.assertRaises(ValueError):
                build_root_conditions(self.model, target, history_length=history,
                                      generated_offset=0, device="cpu")
        with self.assertRaises(ValueError):
            build_root_conditions(self.model, target, history_length=4,
                                  generated_offset=1, device="cpu")
        self.model.gen_horizon_len = 48
        with self.assertRaises(ValueError):
            build_root_conditions(self.model, target, history_length=4,
                                  generated_offset=0, device="cpu")
        self.model.gen_horizon_len = 52
        self.rep.overlap = True
        with self.assertRaisesRegex(ValueError, "overlaps"):
            build_root_conditions(self.model, target, history_length=4,
                                  generated_offset=0, device="cpu")
        self.rep.overlap = False
        self.rep.empty = True
        with self.assertRaisesRegex(ValueError, "empty"):
            build_root_conditions(self.model, target, history_length=4,
                                  generated_offset=0, device="cpu")


if __name__ == "__main__":
    unittest.main()
