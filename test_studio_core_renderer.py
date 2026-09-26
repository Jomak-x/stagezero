"""Offline checks for textured Core actors in the Studio viewport."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from viser import _messages

from realtime_clip import CanonicalClip
from studio_core_renderer import StudioCoreRenderer


ROOT = Path(__file__).resolve().parent


class FakeHandle:
    def __init__(self, props):
        self._impl = SimpleNamespace(props=props)
        self.bones = [SimpleNamespace(position=None, wxyz=None)
                      for _ in range(len(props.bone_positions))]
        self.visible = True
        self.removed = False

    def remove(self):
        self.removed = True


class FakeScene:
    def __init__(self):
        self.handles = []
        self.names = []
        self.messages = []
        self._websock_interface = SimpleNamespace(queue_message=self.messages.append)

    def add_mesh_skinned(self, name, vertices, faces, *, bone_wxyzs,
                         bone_positions, skin_weights, **kwargs):
        top4 = np.argsort(skin_weights, axis=1)[:, -4:]
        weights = np.take_along_axis(skin_weights, top4, axis=1)
        props = _messages.SkinnedMeshProps(
            vertices, faces, kwargs["color"], False, None, False,
            kwargs["side"], "standard", kwargs["cast_shadow"],
            kwargs["receive_shadow"], bone_wxyzs, bone_positions,
            top4.astype(np.uint16), weights.astype(np.float32),
        )
        handle = FakeHandle(props)
        self.names.append(name)
        self.handles.append(handle)
        return handle


class StudioCoreRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(ROOT / "assets/core-motion/martial-combo.npz", allow_pickle=False) as data:
            cls.source_positions = data["positions"][:, :80].copy()
            cls.source_rotations = data["rotations"][:, :80].copy()

    def test_distinct_textured_actors_and_exact_appended_prefix(self):
        scene = FakeScene()
        renderer = StudioCoreRenderer(SimpleNamespace(scene=scene))
        self.addCleanup(renderer.remove)
        self.assertNotEqual(renderer.characters[0].mesh_sha256,
                            renderer.characters[1].mesh_sha256)
        self.assertEqual(scene.names, ["/actor/core/0", "/actor/core/1"])
        self.assertTrue(all(not handle.visible for handle in scene.handles))
        self.assertEqual(len(scene.messages), 2)
        for message, character in zip(scene.messages, renderer.characters):
            self.assertIsInstance(message.props, _messages.SkinnedMeshProps)
            self.assertEqual(message.props.uv.shape, (len(character.vertices), 2))
            self.assertEqual(message.props.normals.shape, (len(character.vertices), 3))
            self.assertEqual(message.props.texture_png, character.texture_png)
            self.assertGreater(len(message.props.texture_png), 1000)

        positions = np.concatenate([self.source_positions,
                                    self.source_positions + [1.8, 0., 0.]], axis=0)
        rotations = np.concatenate([self.source_rotations, self.source_rotations], axis=0)
        first = CanonicalClip.from_arrays(positions[:, :40], rotations[:, :40],
                                          actor_ids=("civilian", "ranger"), source="ardy_core")
        extended = CanonicalClip.from_arrays(positions, rotations,
                                             actor_ids=first.actor_ids, source="ardy_core")
        renderer.set_clip(first)
        prefix_positions = renderer.fitted_positions.copy()
        prefix_rotations = renderer.fitted_rotations.copy()
        offsets = renderer.floor_offsets
        # The accepted native solo run uses original Core arm fitting. The
        # research paired wrist IK can contort elbows and is not the default.
        accepted = renderer.characters[0].clip_payload(
            positions[:1, :40], rotations[:1, :40], preserve_wrists=False,
            preserve_feet=True, floor_y=0., floor_offsets=[offsets[0]],
        )
        np.testing.assert_array_equal(prefix_positions[0],
                                      np.asarray(accepted["fitted_positions"][0], dtype=np.float32))
        self.assertTrue(renderer.tick(27))
        renderer.set_visible(True)
        self.assertTrue(all(handle.visible for handle in scene.handles))
        self.assertEqual(len(scene.handles[0].bones), 17)
        self.assertEqual(len(scene.handles[1].bones), 17)
        self.assertGreater(renderer.actor_root("ranger")[0] - renderer.actor_root("civilian")[0], 1.7)
        renderer.set_clip(extended)
        self.assertEqual(renderer.floor_offsets, offsets)
        np.testing.assert_array_equal(renderer.fitted_positions[:, :40], prefix_positions)
        np.testing.assert_array_equal(renderer.fitted_rotations[:, :40], prefix_rotations)
        renderer.tick(79)
        self.assertEqual(renderer.frame, 79)
        renderer.tick(27)
        np.testing.assert_array_equal(renderer.fitted_positions[:, :40], prefix_positions)
        np.testing.assert_allclose(scene.handles[0].bones[0].position,
                                   renderer.fitted_positions[0, 27, 0])

        solo = CanonicalClip.from_arrays(positions[:1, :40], rotations[:1, :40],
                                         actor_ids=("civilian",), source="ardy_core")
        renderer.set_clip(solo)
        renderer.tick(0)
        self.assertTrue(scene.handles[0].visible)
        self.assertFalse(scene.handles[1].visible)
        renderer.remove()
        self.assertTrue(all(handle.removed for handle in scene.handles))


if __name__ == "__main__":
    unittest.main()
