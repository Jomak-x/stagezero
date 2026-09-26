"""Behavioral pose contracts, using two independently named humanoid GLBs."""
from __future__ import annotations

import copy
import json
import math
import struct
import unittest

import numpy as np

from character_assets import AssetValidationError, inspect_glb
from retargeting import build_retargeter, detect_rig_profile, neutral_source_pose, RigMappingError
from tests.glb_fixtures import make_glb, make_humanoid_glb, make_static_glb


X90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
Y90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
Z90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)


def changed_asset(asset, change):
    document = copy.deepcopy(asset.document)
    change(document)
    return inspect_glb(make_glb(document, asset.binary_chunk))


def mapping_for(asset, **options):
    profile = detect_rig_profile(asset)
    return {"schema_version": 1, "bones": dict(profile.bones), **options}


def with_hands(asset, scheme):
    """Extend the tiny GLB with measured forearm tails and correct inverse binds."""
    document = copy.deepcopy(asset.document)
    profile = detect_rig_profile(asset)
    matrices = [np.linalg.inv(asset.nodes[i].world_matrix) for i in document["skins"][0]["joints"]]
    for side, sign in (("left", 1), ("right", -1)):
        forearm = profile.bones[f"{side}_forearm"]
        name = f"{side}_wrist_yaw_skel" if scheme == "g1" else f"mixamorig:{side.title()}Hand"
        offset = [sign * 0.29, 0, 0]
        index = len(document["nodes"])
        document["nodes"].append({"name": name, "translation": offset})
        document["nodes"][forearm].setdefault("children", []).append(index)
        document["skins"][0]["joints"].append(index)
        local = np.eye(4)
        local[:3, 3] = offset
        matrices.append(np.linalg.inv(asset.nodes[forearm].world_matrix @ local))
    binary = asset.binary_chunk + b"\0" * (-len(asset.binary_chunk) % 4)
    data = b"".join(struct.pack("<16f", *matrix.T.reshape(16)) for matrix in matrices)
    view = len(document["bufferViews"])
    document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
    accessor = len(document["accessors"])
    document["accessors"].append({"bufferView": view, "componentType": 5126, "count": len(matrices), "type": "MAT4"})
    document["skins"][0]["inverseBindMatrices"] = accessor
    return inspect_glb(make_glb(document, binary + data))


class RetargetingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = {scheme: inspect_glb(make_humanoid_glb(scheme)) for scheme in ("g1", "mixamo")}
        # One load for the fixture suite; production receives the session skeleton.
        first = build_retargeter(cls.assets["g1"])
        cls.skeleton = first.skeleton

    def target(self, scheme="g1", mapping=None, asset=None):
        return build_retargeter(self.assets[scheme] if asset is None else asset, mapping, skeleton=self.skeleton)

    def pose_from_locals(self, retargeter, changes, translation=(0, 0, 0)):
        positions, rotations = retargeter.neutral_source_pose()
        neutral = self.skeleton.neutral_joints.numpy()
        positions[0] += np.array(translation)
        for i, name in enumerate(self.skeleton.bone_order_names):
            parent = int(self.skeleton.joint_parents[i])
            local = changes.get(name, np.eye(3))
            rotations[i] = local if parent == -1 else rotations[parent] @ local
            if parent != -1:
                positions[i] = positions[parent] + rotations[parent] @ (neutral[i] - neutral[parent])
        return positions, rotations

    def test_both_profiles_preserve_explicit_gltf_bind_separately_from_g1_neutral(self):
        for scheme, asset in self.assets.items():
            with self.subTest(scheme=scheme):
                target = self.target(scheme)
                self.assertEqual(target.profile.profile_name, scheme)
                pose = target.bind_pose()
                np.testing.assert_allclose(pose.local_matrices, [n.local_matrix for n in asset.nodes], atol=1e-10)
                np.testing.assert_allclose(pose.world_matrices, [n.world_matrix for n in asset.nodes], atol=1e-10)
                neutral = target.retarget(*target.neutral_source_pose())
                self.assertFalse(np.allclose(neutral.local_matrices, pose.local_matrices))

    def test_source_calibration_is_default_skeleton_fk_not_recording(self):
        positions, rotations = neutral_source_pose(self.skeleton)
        neutral = self.skeleton.neutral_joints.numpy()
        np.testing.assert_allclose(positions - positions[0], neutral - neutral[0], atol=1e-10)
        self.assertAlmostEqual(float(np.min(positions[:, 1])), 0)
        np.testing.assert_allclose(rotations, np.tile(np.eye(3), (34, 1, 1)), atol=1e-10)

    def test_anatomical_segments_start_at_hip_and_shoulder_not_terminal_motors(self):
        asset = with_hands(self.assets["mixamo"], "mixamo")
        target = self.target(asset=asset)
        # G1 declares the hip, foot and hand landmarks. Its shoulder is the
        # first arm joint attached to the torso, before the roll/yaw motors.
        hips = dict(zip(("right", "left"), self.skeleton.hip_joint_names))
        for changes in ({}, {"left_hip_roll_skel": X90,
                             "left_shoulder_roll_skel": Z90,
                             "left_shoulder_yaw_skel": Y90}):
            positions, rotations = self.pose_from_locals(target, changes)
            pose = target.retarget(positions, rotations)
            source_lengths, target_lengths = [], []
            for side in ("left", "right"):
                shoulder = f"{side}_elbow_skel"
                while self.skeleton.bone_parents[shoulder] != "waist_pitch_skel":
                    shoulder = self.skeleton.bone_parents[shoulder]
                foot = getattr(self.skeleton, f"{side}_foot_joint_names")[0]
                hand = getattr(self.skeleton, f"{side}_hand_joint_names")[0]
                segments = (("thigh", "shin", hips[side], f"{side}_knee_skel"),
                            ("shin", "foot", f"{side}_knee_skel", foot),
                            ("upper_arm", "forearm", shoulder, f"{side}_elbow_skel"),
                            ("forearm", "hand", f"{side}_elbow_skel", hand))
                for start, end, source_start, source_end in segments:
                    a, b = target.profile.bones[f"{side}_{start}"], target.profile.bones[f"{side}_{end}"]
                    source_segment = positions[self.skeleton.bone_index[source_end]] - positions[self.skeleton.bone_index[source_start]]
                    actual = pose.world_matrices[b, :3, 3] - pose.world_matrices[a, :3, 3]
                    expected = target.basis @ source_segment
                    with self.subTest(side=side, segment=start, changes=tuple(changes)):
                        np.testing.assert_allclose(actual / np.linalg.norm(actual), expected / np.linalg.norm(expected), atol=1e-9)
                    if not changes and start in ("thigh", "shin"):
                        source_lengths.append(np.linalg.norm(source_segment))
                        target_lengths.append(np.linalg.norm(actual))
            if not changes:
                self.assertAlmostEqual(target.root_scale, sum(target_lengths) / sum(source_lengths))

    def test_upper_and_lower_arm_segments_follow_source_from_lowered_to_raised(self):
        for scheme, base in self.assets.items():
            asset = with_hands(base, scheme)
            target = self.target(asset=asset)
            self.assertEqual(target.warnings, ())
            for angle in (0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 2):
                c, s = math.cos(angle), math.sin(angle)
                rotate = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                positions, rotations = self.pose_from_locals(target, {
                    "left_shoulder_roll_skel": rotate,
                    "right_shoulder_roll_skel": rotate.T,
                    "left_elbow_skel": X90 if angle == math.pi / 2 else np.eye(3),
                })
                posed = target.retarget(positions, rotations)
                for side in ("left", "right"):
                    for role, end in (("upper_arm", "forearm"), ("forearm", "hand")):
                        with self.subTest(scheme=scheme, angle=angle, side=side, segment=role):
                            start_role, end_role = f"{side}_{role}", f"{side}_{end}"
                            start_node, end_node = target.profile.bones[start_role], target.profile.bones[end_role]
                            target_segment = posed.world_matrices[end_node, :3, 3] - posed.world_matrices[start_node, :3, 3]
                            source_segment = target.basis @ (positions[target.source_position_indices[end_role]] - positions[target.source_position_indices[start_role]])
                            np.testing.assert_allclose(target_segment / np.linalg.norm(target_segment), source_segment / np.linalg.norm(source_segment), atol=1e-9)
                            bind_length = np.linalg.norm(asset.nodes[end_node].world_matrix[:3, 3] - asset.nodes[start_node].world_matrix[:3, 3])
                            self.assertAlmostEqual(float(np.linalg.norm(target_segment)), float(bind_length))
                            if role == "upper_arm" and angle in (0, math.pi):
                                self.assertGreater(float(target_segment[1]) * (-1 if angle == 0 else 1), 0.2)

    def test_minimal_forearm_axis_fallback_is_explicit_and_overridable(self):
        target = self.target()
        self.assertEqual(len(target.warnings), 2)
        mapping = mapping_for(self.assets["g1"], bone_axes={"left_forearm": [0, 0, 1], "right_forearm": [0, 0, 1]})
        custom = self.target(mapping=mapping)
        self.assertEqual(custom.warnings, ())
        positions, rotations = custom.neutral_source_pose()
        pose = custom.retarget(positions, rotations)
        for side in ("left", "right"):
            role = f"{side}_forearm"
            direction = custom.basis @ (positions[custom.source_position_indices[f"{side}_hand"]] - positions[custom.source_position_indices[role]])
            direction /= np.linalg.norm(direction)
            np.testing.assert_allclose(pose.world_matrices[custom.profile.bones[role], :3, :3] @ [0, 0, 1], direction, atol=1e-9)

    def test_self_consistent_bind_still_preserves_skinning_identity(self):
        for scheme, asset in self.assets.items():
            target = self.target(scheme)
            pose = target.bind_pose()
            skin = asset.document["skins"][0]
            inverse_binds = asset.read_accessor(skin["inverseBindMatrices"])
            for ordinal, index in enumerate(skin["joints"]):
                inverse_bind = inverse_binds[ordinal].reshape(4, 4).T
                np.testing.assert_allclose(pose.world_matrices[index] @ inverse_bind, np.eye(4), atol=1e-7)

    def test_shoulder_aggregates_all_three_g1_axis_joints(self):
        asset = with_hands(self.assets["g1"], "g1")
        target = self.target(asset=asset, mapping=mapping_for(asset, source_to_target_basis=np.eye(3).tolist()))
        positions, rotations = self.pose_from_locals(target, {
            "left_shoulder_pitch_skel": X90,
            "left_shoulder_roll_skel": Y90,
            "left_shoulder_yaw_skel": Z90,
        })
        pose = target.retarget(positions, rotations)
        arm = target.profile.bones["left_upper_arm"]
        elbow = target.profile.bones["left_forearm"]
        hand = target.profile.bones["left_hand"]
        neutral = target.retarget(*target.neutral_source_pose())
        # The hand inherits every motor rotation. The upper-arm direction also
        # includes the noncoincident pitch/roll/yaw origins, so its geometric
        # swing must not be equated to a terminal motor matrix alone.
        np.testing.assert_allclose(pose.world_matrices[hand, :3, :3], X90 @ Y90 @ Z90 @ neutral.world_matrices[hand, :3, :3], atol=1e-10)
        expected = positions[self.skeleton.bone_index["left_elbow_skel"]] - positions[self.skeleton.bone_index["left_shoulder_pitch_skel"]]
        actual = pose.world_matrices[elbow, :3, 3] - pose.world_matrices[arm, :3, 3]
        np.testing.assert_allclose(actual / np.linalg.norm(actual), expected / np.linalg.norm(expected), atol=1e-10)
        self.assertFalse(np.allclose(pose.world_matrices[elbow, :3, 3], target.rest_world[elbow, :3, 3]))

    def test_terminal_motor_axial_rotation_survives_anatomical_position_mapping(self):
        target = self.target(mapping=mapping_for(self.assets["g1"], source_to_target_basis=np.eye(3).tolist()))
        positions, rotations = target.neutral_source_pose()
        neutral = target.retarget(positions, rotations)
        for role, start, end, motor in (
            ("left_upper_arm", "left_shoulder_pitch_skel", "left_elbow_skel", "left_shoulder_yaw_skel"),
            ("left_thigh", "left_hip_pitch_skel", "left_knee_skel", "left_hip_yaw_skel"),
        ):
            # Pose input contains independent position and orientation channels.
            # A half-turn about an unchanged segment must remain visible even
            # though its rotation endpoint differs from its position landmark.
            axis = positions[self.skeleton.bone_index[end]] - positions[self.skeleton.bone_index[start]]
            axis /= np.linalg.norm(axis)
            half_turn = 2 * axis[:, None] @ axis[None, :] - np.eye(3)
            turned = rotations.copy()
            turned[self.skeleton.bone_index[motor]] = half_turn
            pose = target.retarget(positions, turned)
            node = target.profile.bones[role]
            np.testing.assert_allclose(pose.world_matrices[node, :3, :3], half_turn @ neutral.world_matrices[node, :3, :3], atol=1e-9)

    def test_parent_global_rotation_is_removed_when_computing_child_local(self):
        target = self.target(mapping=mapping_for(self.assets["g1"], source_to_target_basis=np.eye(3).tolist()))
        pose = target.retarget(*self.pose_from_locals(target, {
            "pelvis_skel": Y90, "waist_yaw_skel": Z90,
            "left_shoulder_roll_skel": X90, "left_elbow_skel": Z90,
        }))
        arm = target.profile.bones["left_upper_arm"]
        elbow = target.profile.bones["left_forearm"]
        neutral = target.retarget(*target.neutral_source_pose())
        elbow_bind = neutral.world_matrices[elbow, :3, :3]
        np.testing.assert_allclose(pose.world_matrices[elbow, :3, :3], Y90 @ Z90 @ X90 @ Z90 @ elbow_bind, atol=1e-10)
        expected_local = pose.world_matrices[arm, :3, :3].T @ Y90 @ Z90 @ X90 @ Z90 @ elbow_bind
        np.testing.assert_allclose(pose.local_matrices[elbow, :3, :3], expected_local, atol=1e-10)
        np.testing.assert_allclose(pose.world_matrices[arm] @ pose.local_matrices[elbow], pose.world_matrices[elbow], atol=1e-10)

    def test_root_translation_applies_once_in_target_basis_and_units(self):
        target = self.target(mapping=mapping_for(self.assets["g1"], source_to_target_basis=Y90.tolist(), root_scale=2.5))
        neutral = target.retarget(*target.neutral_source_pose())
        travel = np.array([1.0, 0.25, -2.0])
        moved = target.retarget(*self.pose_from_locals(target, {}, travel))
        offset = Y90 @ travel * 2.5
        for index in target.profile.bones.values():
            np.testing.assert_allclose(moved.world_matrices[index, :3, 3] - neutral.world_matrices[index, :3, 3], offset, atol=1e-10)
        # Sibling scene mesh does not get another copy of root travel.
        np.testing.assert_allclose(moved.local_matrices[0], neutral.local_matrices[0])

    def test_target_lengths_and_scales_survive_motion_and_different_proportions(self):
        scales = []
        for scheme in self.assets:
            target = self.target(scheme)
            scales.append(target.root_scale)
            pose = target.retarget(*self.pose_from_locals(target, {"left_shoulder_roll_skel": Z90, "left_elbow_skel": X90, "left_hip_roll_skel": X90}, (2, 1, 0)))
            for node in target.asset.nodes:
                if node.index != target.profile.bones["pelvis"]:
                    np.testing.assert_allclose(pose.local_matrices[node.index, :3, 3], node.local_matrix[:3, 3], atol=1e-10)
                if node.parent is not None:
                    expected = np.linalg.norm(node.world_matrix[:3, 3] - target.rest_world[node.parent, :3, 3])
                    actual = np.linalg.norm(pose.world_matrices[node.index, :3, 3] - pose.world_matrices[node.parent, :3, 3])
                    self.assertAlmostEqual(float(actual), float(expected))
        self.assertAlmostEqual(scales[1] / scales[0], 1.2)

    def test_unmapped_helper_follows_ancestor_and_keeps_its_local_bind(self):
        asset = self.assets["g1"]
        arm = detect_rig_profile(asset).bones["left_upper_arm"]
        def add_helper(document):
            index = len(document["nodes"])
            document["nodes"].append({"name": "sleeve_helper", "translation": [0.12, 0.08, 0.05]})
            document["nodes"][arm]["children"].append(index)
        asset = changed_asset(asset, add_helper)
        target = self.target(asset=asset)
        pose = target.retarget(*self.pose_from_locals(target, {"left_shoulder_roll_skel": Z90}))
        helper = len(asset.nodes) - 1
        np.testing.assert_allclose(pose.local_matrices[helper], asset.nodes[helper].local_matrix)
        np.testing.assert_allclose(pose.world_matrices[helper], pose.world_matrices[arm] @ asset.nodes[helper].local_matrix, atol=1e-10)
        self.assertFalse(np.allclose(pose.world_matrices[helper], asset.nodes[helper].world_matrix))

    def test_rotated_scaled_armature_and_nonidentity_joint_bind_preserved(self):
        asset = self.assets["mixamo"]
        arm = detect_rig_profile(asset).bones["left_upper_arm"]
        pelvis = detect_rig_profile(asset).bones["pelvis"]
        def wrap_armature(document):
            index = len(document["nodes"])
            document["nodes"].append({"name": "Armature", "translation": [2, 0, -3], "rotation": [0, 0.7071067811865476, 0, 0.7071067811865476], "scale": [0.5, 0.5, 0.5], "children": [pelvis]})
            document["scenes"][0]["nodes"].remove(pelvis)
            document["scenes"][0]["nodes"].append(index)
            document["nodes"][arm]["rotation"] = [0, 0, 0.3826834323650898, 0.9238795325112867]
        asset = changed_asset(asset, wrap_armature)
        target = self.target(asset=asset)
        neutral = target.retarget(*target.neutral_source_pose())
        np.testing.assert_allclose(target.bind_pose().local_matrices, [n.local_matrix for n in asset.nodes], atol=1e-10)
        pose = target.retarget(*self.pose_from_locals(target, {"left_shoulder_roll_skel": X90}, (1, 0, 0)))
        for node in asset.nodes:
            if node.index != pelvis:
                np.testing.assert_allclose(pose.local_matrices[node.index, :3, 3], node.local_matrix[:3, 3], atol=1e-10)
        np.testing.assert_allclose(pose.root_position - neutral.root_position, target.basis[:, 0] * target.root_scale, atol=1e-10)

    def test_helper_between_mapped_joints_is_part_of_parent_composition(self):
        asset = self.assets["g1"]
        bones = detect_rig_profile(asset).bones
        arm, elbow = bones["left_upper_arm"], bones["left_forearm"]
        def insert_helper(document):
            index = len(document["nodes"])
            document["nodes"].append({"name": "twist_helper", "rotation": [0, 0, 0.7071067811865476, 0.7071067811865476], "children": [elbow]})
            document["nodes"][arm]["children"] = [index]
        asset = changed_asset(asset, insert_helper)
        target = self.target(asset=asset, mapping=mapping_for(asset, source_to_target_basis=np.eye(3).tolist()))
        neutral = target.retarget(*target.neutral_source_pose())
        np.testing.assert_allclose(target.bind_pose().local_matrices, [n.local_matrix for n in asset.nodes], atol=1e-10)
        pose = target.retarget(*self.pose_from_locals(target, {"left_shoulder_roll_skel": X90, "left_elbow_skel": Y90}))
        helper = len(asset.nodes) - 1
        np.testing.assert_allclose(pose.local_matrices[helper], asset.nodes[helper].local_matrix)
        np.testing.assert_allclose(pose.world_matrices[elbow, :3, :3], X90 @ Y90 @ neutral.world_matrices[elbow, :3, :3], atol=1e-10)
        np.testing.assert_allclose(pose.world_matrices[helper] @ pose.local_matrices[elbow], pose.world_matrices[elbow], atol=1e-10)

    def test_unknown_names_require_mapping_but_explicit_json_works(self):
        asset = self.assets["g1"]
        mapping = mapping_for(asset)
        def rename(document):
            for i, node in enumerate(document["nodes"]):
                node["name"] = f"joint_{i}"
        asset = changed_asset(asset, rename)
        self.assertIsNone(detect_rig_profile(asset))
        with self.assertRaisesRegex(RigMappingError, "Mapping required"):
            self.target(asset=asset)
        mapping["bones"] = {role: f"joint_{index}" for role, index in mapping["bones"].items()}
        target = self.target(asset=asset, mapping=json.dumps(mapping))
        np.testing.assert_allclose(target.bind_pose().local_matrices, [n.local_matrix for n in asset.nodes], atol=1e-10)

    def test_mapping_rejects_duplicates_missing_roles_wrong_hierarchy_and_basis(self):
        asset = self.assets["g1"]
        mutations = [
            (lambda m: m["bones"].update(left_forearm=m["bones"]["left_upper_arm"]), "duplicate"),
            (lambda m: m["bones"].pop("left_foot"), "Missing required"),
            (lambda m: m["bones"].update(left_forearm=m["bones"]["right_forearm"], right_forearm=m["bones"]["left_forearm"]), "hierarchy"),
            (lambda m: m.update(source_to_target_basis=[[-1, 0, 0], [0, 1, 0], [0, 0, 1]]), "proper orthonormal"),
            (lambda m: m.update(root_scale=0), "root_scale"),
            (lambda m: m["bones"].update(left_forearm=0), "skin joint"),
        ]
        for mutation, error in mutations:
            with self.subTest(error=error):
                mapping = mapping_for(asset)
                mutation(mapping)
                with self.assertRaisesRegex(RigMappingError, error):
                    self.target(mapping=mapping)

    def test_static_and_independent_armatures_are_not_motion_ready(self):
        self.assertIsNone(detect_rig_profile(inspect_glb(make_static_glb())))
        asset = self.assets["g1"]
        def extra_root(document):
            index = len(document["nodes"])
            document["nodes"].append({"name": "other_pelvis"})
            document["scenes"][0]["nodes"].append(index)
            document["skins"].append({"joints": [index]})
        with self.assertRaisesRegex(AssetValidationError, "independent armatures"):
            changed_asset(asset, extra_root)

    def test_nonuniform_and_reflected_bind_and_bad_motion_are_rejected(self):
        for scale in ([1, 2, 1], [-1, 1, 1]):
            asset = changed_asset(self.assets["g1"], lambda d: d["nodes"][1].update(scale=scale))
            with self.assertRaises(RigMappingError):
                self.target(asset=asset)
        target = self.target()
        positions, rotations = target.neutral_source_pose()
        rotations[0, 0, 0] = 2
        with self.assertRaisesRegex(RigMappingError, "orthonormal"):
            target.retarget(positions, rotations)
        positions[0, 0] = float("nan")
        with self.assertRaisesRegex(RigMappingError, "finite"):
            target.retarget(positions, rotations)

    def test_invalid_mapping_json_is_a_mapping_error(self):
        for raw in ('{"bones": {}, "bones": {}}', '{', '[]', 'null'):
            with self.subTest(raw=raw), self.assertRaises(RigMappingError):
                self.target(mapping=raw)


if __name__ == "__main__":
    unittest.main()
