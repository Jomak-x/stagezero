"""Native glTF export parity and bounded generated-fit loading."""
from copy import deepcopy
from io import BytesIO
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from character_actor import GeneratedCharacterActor
from character_assets import inspect_glb
from character_compatibility import assess_character
from generated_character_rig import export_generated_character, build_generated_retargeter, _pack, _MARKER
from retargeting import _default_skeleton, neutral_source_pose, RigMappingError
from test_character_actor import _test_glb


def fixture():
    scene = trimesh.load(BytesIO(_test_glb()), file_type="glb", force="scene", process=False)
    return trimesh.exchange.gltf.export_glb(scene, include_normals=True)


def gltf_vertices(asset, pose):
    primitive = asset.document["meshes"][0]["primitives"][0]
    attributes = primitive["attributes"]
    vertices = asset.read_accessor(attributes["POSITION"])
    joints = asset.read_accessor(attributes["JOINTS_0"], normalize=False)
    weights = asset.read_accessor(attributes["WEIGHTS_0"])
    skin = asset.skins[0]
    inverse = asset.read_accessor(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    matrices = pose.world_matrices[skin["joints"]] @ inverse
    homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
    result = np.zeros_like(vertices, dtype=float)
    for slot in range(4):
        result += weights[:, slot, None] * np.einsum("nij,nj->ni", matrices[joints[:, slot]], homogeneous)[:, :3]
    return result


class GeneratedRigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skeleton = _default_skeleton()
        cls.source = fixture()
        cls.bind = neutral_source_pose(cls.skeleton)
        cls.data = export_generated_character(cls.source, cls.skeleton, *cls.bind)
        cls.asset = inspect_glb(cls.data)

    def test_native_skin_and_material_integrity(self):
        source, asset = inspect_glb(self.source), self.asset
        self.assertEqual(asset.kind, "skinned")
        self.assertTrue(assess_character(asset, skeleton=self.skeleton).motion_ready)
        for key in ("materials", "images", "textures", "samplers"):
            self.assertEqual(asset.document.get(key), source.document.get(key))
        self.assertEqual(asset.binary_chunk[:len(source.binary_chunk)], source.binary_chunk)
        self.assertEqual(asset.vertex_count, source.vertex_count)
        self.assertEqual(len(asset.skins[0]["joints"]), 17)
        np.testing.assert_allclose(asset.bounds[:, 1], [0, 1.70], atol=1e-6)

    def test_motion_matches_fitted_cpu_and_local_hierarchy(self):
        actor = GeneratedCharacterActor(None, self.source, "/test", self.skeleton, *self.bind)
        rig = build_generated_retargeter(self.asset, self.skeleton)
        poses = [self.bind]
        for angle in (35, 80, -50):
            positions, rotations = (value.copy() for value in self.bind)
            positions[0] += [.35, -.12, -.2]
            for name, axis, value in (("left_shoulder_yaw_skel", "z", angle),
                                      ("left_elbow_skel", "x", angle/2),
                                      ("left_ankle_roll_skel", "x", angle),
                                      ("right_hip_yaw_skel", "x", -angle/2),
                                      ("right_knee_skel", "x", angle/2)):
                index = self.skeleton.bone_index[name]
                rotations[index] = Rotation.from_euler(axis, value, degrees=True).as_matrix() @ rotations[index]
            poses.append((positions, rotations))
        for positions, rotations in poses:
            actor.update(positions, rotations)
            pose = rig.retarget(positions, rotations)
            np.testing.assert_allclose(gltf_vertices(self.asset, pose), actor.deform_vertices(), atol=4e-7)
            for node in self.asset.nodes:
                parent = np.eye(4) if node.parent is None else pose.world_matrices[node.parent]
                np.testing.assert_allclose(parent @ pose.local_matrices[node.index], pose.world_matrices[node.index], atol=1e-10)
            self.assertGreaterEqual(gltf_vertices(self.asset, pose)[actor._support_indices, 1].min(), -1e-6)
        first = rig.retarget(*poses[1]).world_matrices.copy()
        rig.retarget(*poses[2])
        np.testing.assert_allclose(rig.retarget(*poses[1]).world_matrices, first)

    def test_loading_fit_does_not_refit(self):
        from unittest.mock import patch
        with patch("character_actor._fit_human", side_effect=AssertionError("Refit attempted")):
            rig = build_generated_retargeter(self.asset, self.skeleton)
            self.assertEqual(rig.profile.profile_name, "generated_human_v1")
            np.testing.assert_allclose(rig.bind_pose().world_matrices[0], np.eye(4))

    def test_rejects_missing_or_modified_fit_metadata(self):
        for mutate in (
            lambda doc: doc.pop("extras"),
            lambda doc: doc.update(extras=[]),
            lambda doc: doc["skins"][0].pop("inverseBindMatrices"),
            lambda doc: doc["extras"][_MARKER].update(source_bind_positions="bad"),
            lambda doc: doc["extras"][_MARKER].update(version=True),
            lambda doc: doc["extras"][_MARKER].update(source_bind_positions=[[0, 0, 0]]),
            lambda doc: doc["extras"][_MARKER].update(source_bind_rotations=np.zeros((34, 3, 3)).tolist()),
            lambda doc: doc["extras"][_MARKER].update(extra="not allowed"),
            lambda doc: doc["nodes"][16].update(translation=[0, 0, 0]),
            lambda doc: doc["nodes"][3].update(name="ChangedHead"),
        ):
            doc = deepcopy(self.asset.document)
            mutate(doc)
            asset = inspect_glb(_pack(doc, self.asset.binary_chunk))
            with self.assertRaises(RigMappingError):
                build_generated_retargeter(asset, self.skeleton)

    def test_rejects_already_rigged_input_and_partial_bind(self):
        with self.assertRaisesRegex(ValueError, "static mesh"):
            export_generated_character(self.data, self.skeleton)
        with self.assertRaisesRegex(ValueError, "both calibration"):
            export_generated_character(self.source, self.skeleton, self.bind[0])

    def test_rejects_nonrigid_motion_without_mutating_last_pose(self):
        rig = build_generated_retargeter(self.asset, self.skeleton)
        baseline = rig.retarget(*self.bind)
        rotations = self.bind[1].copy()
        rotations[0] *= 2
        with self.assertRaises(RigMappingError):
            rig.retarget(self.bind[0], rotations)
        np.testing.assert_allclose(rig.retarget(*self.bind).world_matrices, baseline.world_matrices)


if __name__ == "__main__":
    unittest.main()
