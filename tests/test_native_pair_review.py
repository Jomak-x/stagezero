"""Raw paired viewer invariants independent of browser rendering."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.native_pair_review import (
    FPS, LIMBS, capsule_between, handshake_contact_weights, load_native,
    mannequin_mesh, source_model,
)


def native_fixture(directory: Path) -> tuple[Path, np.ndarray]:
    expected = np.zeros((4, 2, 22, 3), dtype=np.float32)
    for frame in range(4):
        expected[frame, :, :, 0] = np.arange(22) * .04
        expected[frame, :, :, 1] = frame * .02 + .5
        expected[frame, 1, :, 2] = 1.2
    source = directory / "native.npz"
    np.savez_compressed(source, joints=expected, smoothed_joints=expected + 10,
                        metadata='{"fps":30,"prompt":"test pair"}')
    return source, expected


class NativePairReviewTests(unittest.TestCase):
    def test_raw_joints_used_even_if_smoothed_array_is_present(self) -> None:
        with TemporaryDirectory() as directory:
            source, expected = native_fixture(Path(directory))
            actual, metadata = load_native(source)
            self.assertEqual(actual.shape, (4, 2, 22, 3))
            self.assertEqual(FPS, metadata["fps"])
            np.testing.assert_array_equal(actual, expected)

    def test_limb_geometry_terminates_at_source_joint_endpoints(self) -> None:
        with TemporaryDirectory() as directory:
            source, _ = native_fixture(Path(directory))
            poses, _ = load_native(source)
            for frame in range(len(poses)):
                for actor in (0, 1):
                    joints = poses[frame, actor]
                    vertices, faces = mannequin_mesh(joints)
                    self.assertEqual(vertices.shape, (1752, 3))
                    self.assertEqual(faces.shape, (3000, 3))
                    self.assertTrue(np.isfinite(vertices).all())
                    offset = 0
                    for start, end, radius in LIMBS:
                        limb, _ = capsule_between(joints[start], joints[end], radius)
                        np.testing.assert_array_equal(limb[0], joints[start])
                        np.testing.assert_array_equal(limb[-1], joints[end])
                        np.testing.assert_array_equal(vertices[offset], joints[start])
                        np.testing.assert_array_equal(vertices[offset + len(limb) - 1], joints[end])
                        offset += len(limb)

    def test_rejects_smoothed_or_non_native_shape(self) -> None:
        with TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.npz"
            np.savez_compressed(bad, smoothed_joints=np.zeros((2, 2, 22, 3)), metadata="{}")
            with self.assertRaises(KeyError):
                load_native(bad)
            np.savez_compressed(bad, joints=np.zeros((2, 2, 27, 3)), metadata="{}")
            with self.assertRaisesRegex(ValueError, "native joints"):
                load_native(bad)
            np.savez_compressed(bad, joints=np.zeros((2, 2, 22, 3)), metadata='{"fps": 20}')
            with self.assertRaisesRegex(ValueError, "30 fps"):
                load_native(bad)

    def test_handshake_weights_stay_inside_longest_measured_run(self) -> None:
        raw = np.zeros((160, 2, 22, 3), dtype=np.float32)
        raw[:, 1, 20, 0] = 1.
        raw[40:136, 1, 20, 0] = .1
        raw[5:20, 1, 20, 0] = .1
        original = raw.copy()
        weights, report = handshake_contact_weights(raw, (20, 20))
        np.testing.assert_array_equal(raw, original)
        self.assertEqual(weights.shape, (2, 160, 2))
        self.assertEqual((report["longest_run_start_frame"],
                          report["longest_run_end_frame_exclusive"]), (40, 136))
        self.assertTrue(np.all(weights[:, :40] == 0))
        self.assertTrue(np.all(weights[:, 136:] == 0))
        self.assertTrue(np.all(weights[:, :, 1] == 0))
        self.assertEqual(weights[0, 40, 0], 0)
        self.assertEqual(weights[0, 46, 0], 1)
        self.assertEqual(weights[0, 129, 0], 1)
        self.assertEqual(weights[0, 135, 0], 0)
        np.testing.assert_array_equal(weights[0], weights[1])

    def test_handshake_rejects_unmeasured_contact_and_wrong_joints(self) -> None:
        raw = np.zeros((30, 2, 22, 3), dtype=np.float32)
        raw[:, 1, :, 0] = 2.
        with self.assertRaisesRegex(ValueError, "never come within"):
            handshake_contact_weights(raw, (20, 20))
        with self.assertRaisesRegex(ValueError, "source wrists"):
            handshake_contact_weights(raw, (19, 20))

    def test_model_label_whitelist(self) -> None:
        self.assertEqual(source_model({"model": "InterGen"}), "InterGen")
        self.assertEqual(source_model({"model": "InterMask"}), "InterMask")
        self.assertEqual(source_model({"model": "<script>alert(1)</script>"}), "Native paired model")


if __name__ == "__main__":
    unittest.main()
