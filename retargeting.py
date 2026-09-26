"""Bind-calibrated G1 motion for explicitly mapped, single-humanoid GLB rigs.

Matrices use column vectors; arrays are indexed by original glTF node index.
Only pelvis translation is transferred. Target joint offsets and uniform scales
are retained, including unmapped helpers. No IK, finger animation or automatic
foot-contact correction is attempted. See rig_profiles/README.md for the contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


class RigMappingError(ValueError):
    """A rig needs a supported, unambiguous humanoid mapping/calibration."""


ROLE_PARENTS = {
    "pelvis": None, "spine": "pelvis",
    "left_upper_arm": "spine", "left_forearm": "left_upper_arm",
    "right_upper_arm": "spine", "right_forearm": "right_upper_arm",
    "left_thigh": "pelvis", "left_shin": "left_thigh", "left_foot": "left_shin",
    "right_thigh": "pelvis", "right_shin": "right_thigh", "right_foot": "right_shin",
    "left_hand": "left_forearm", "right_hand": "right_forearm", "head": "spine",
}
REQUIRED_ROLES = tuple(list(ROLE_PARENTS)[:12])
# Terminal axis joints carry the accumulated pitch/roll/yaw rotations.
SOURCE_NAMES = {
    "pelvis": "pelvis_skel", "spine": "waist_pitch_skel",
    "left_upper_arm": "left_shoulder_yaw_skel", "left_forearm": "left_elbow_skel",
    "right_upper_arm": "right_shoulder_yaw_skel", "right_forearm": "right_elbow_skel",
    "left_thigh": "left_hip_yaw_skel", "left_shin": "left_knee_skel", "left_foot": "left_ankle_roll_skel",
    "right_thigh": "right_hip_yaw_skel", "right_shin": "right_knee_skel", "right_foot": "right_ankle_roll_skel",
    "left_hand": "left_wrist_yaw_skel", "right_hand": "right_wrist_yaw_skel", "head": "waist_pitch_skel",
}
LIMB_ENDS = {
    f"{side}_{start}": f"{side}_{end}"
    for side in ("left", "right")
    for start, end in (("upper_arm", "forearm"), ("forearm", "hand"), ("thigh", "shin"), ("shin", "foot"))
}


@dataclass(frozen=True)
class RigProfile:
    profile_name: str
    bones: Mapping[str, int]
    source_to_target_basis: np.ndarray | None = None
    root_scale: float | None = None
    bone_axes: Mapping[str, np.ndarray] | None = None


@dataclass(frozen=True)
class TargetPose:
    local_matrices: np.ndarray
    world_matrices: np.ndarray
    root_position: np.ndarray


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RigMappingError("Pose and calibration values must be numeric arrays") from error


def _rotation(matrix: np.ndarray, label: str, tolerance: float = 1e-5) -> None:
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise RigMappingError(f"{label} must be a finite 3x3 rotation")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=tolerance, rtol=0) or not np.allclose(np.linalg.det(matrix), 1, atol=tolerance, rtol=0):
        raise RigMappingError(f"{label} must be a proper orthonormal rotation (no reflection)")


def _similarity(matrix: np.ndarray, label: str) -> tuple[np.ndarray, float]:
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7, rtol=0):
        raise RigMappingError(f"{label} must be a finite affine matrix")
    scale = np.linalg.norm(matrix[:3, :3], axis=0)
    if np.min(scale) <= 1e-8 or not np.allclose(scale, scale[0], atol=1e-7, rtol=1e-5):
        raise RigMappingError(f"{label} requires positive uniform scale; nonuniform/singular scales are unsupported")
    rotation = matrix[:3, :3] / scale[0]
    _rotation(rotation, label)
    return rotation, float(scale[0])


def _joint_set(asset: Any) -> set[int]:
    return {joint for skin in asset.document.get("skins", []) for joint in skin["joints"]}


def _resolve_profile(asset: Any, spec: Mapping[str, Any]) -> RigProfile:
    if type(spec.get("schema_version", 1)) is not int or spec.get("schema_version", 1) != 1:
        raise RigMappingError("Mapping schema_version must be 1")
    unexpected = set(spec) - {"schema_version", "profile_name", "bones", "source_to_target_basis", "root_scale", "bone_axes"}
    if unexpected:
        raise RigMappingError(f"Unknown mapping fields: {', '.join(sorted(unexpected))}")
    bones = spec.get("bones")
    if not isinstance(bones, Mapping):
        raise RigMappingError("Mapping must contain a bones object with role-to-node entries")
    unknown = set(bones) - set(ROLE_PARENTS)
    if unknown:
        raise RigMappingError(f"Unknown bone roles: {', '.join(sorted(unknown))}")
    missing = set(REQUIRED_ROLES) - set(bones)
    if missing:
        raise RigMappingError(f"Missing required bone roles: {', '.join(sorted(missing))}")
    names: dict[str, list[int]] = {}
    for node in asset.nodes:
        names.setdefault(node.name, []).append(node.index)
    resolved: dict[str, int] = {}
    joints = _joint_set(asset)
    for role, value in bones.items():
        if type(value) is int and 0 <= value < len(asset.nodes):
            index = value
        elif isinstance(value, str) and len(names.get(value, [])) == 1:
            index = names[value][0]
        else:
            raise RigMappingError(f"Bone role {role} must reference a unique node name or valid node index: {value!r}")
        if index not in joints:
            raise RigMappingError(f"Bone role {role} must reference a skin joint")
        resolved[role] = index
    if len(set(resolved.values())) != len(resolved):
        raise RigMappingError("Bone roles must not map to duplicate nodes")
    reverse = {index: role for role, index in resolved.items()}
    for role, index in resolved.items():
        parent = asset.nodes[index].parent
        while parent is not None and parent not in reverse:
            parent = asset.nodes[parent].parent
        expected = ROLE_PARENTS[role]
        actual = reverse.get(parent)
        if actual != expected:
            raise RigMappingError(f"Invalid humanoid hierarchy: {role} must descend from {expected or 'an unmapped root'}, found {actual}")
    pelvis = resolved["pelvis"]
    for joint in joints:
        cursor: int | None = joint
        while cursor is not None and cursor != pelvis:
            cursor = asset.nodes[cursor].parent
        if cursor is None:
            raise RigMappingError("Independent armatures or skin joints outside the mapped pelvis hierarchy are unsupported")
    basis = spec.get("source_to_target_basis")
    if basis is not None:
        basis = _array(basis)
        _rotation(basis, "source_to_target_basis")
    root_scale = spec.get("root_scale")
    if root_scale is not None and (type(root_scale) not in (int, float) or not math.isfinite(root_scale) or root_scale <= 0):
        raise RigMappingError("root_scale must be a finite positive number")
    axes = spec.get("bone_axes")
    if axes is None:
        axes = {}
    if not isinstance(axes, Mapping) or set(axes) - set(LIMB_ENDS):
        raise RigMappingError("bone_axes must map limb roles to local-space direction vectors")
    axes = {role: _unit(_array(axis), f"{role} bone axis") for role, axis in axes.items()}
    return RigProfile(str(spec.get("profile_name", "custom")), resolved, basis, root_scale, axes)


def detect_rig_profile(asset: Any) -> RigProfile | None:
    """Recognize exact G1/Mixamo names only; unknown/invalid rigs need mapping."""
    if not _joint_set(asset):
        return None
    names = {node.name for node in asset.nodes}
    directory = Path(__file__).resolve().parent / "rig_profiles"
    for profile_name in ("g1", "mixamo"):
        raw = json.loads((directory / f"{profile_name}.json").read_text())
        prefixes = ("",) if profile_name == "g1" else ("", "mixamorig:", "mixamorig")
        for prefix in prefixes:
            bones = {role: prefix + name for role, name in raw["bones"].items()}
            if not all(name in names for name in bones.values()):
                continue
            bones.update({role: prefix + name for role, name in raw.get("optional_bones", {}).items() if prefix + name in names})
            try:
                return _resolve_profile(asset, {"schema_version": 1, "profile_name": profile_name, "bones": bones})
            except RigMappingError:
                continue
    return None


def _default_skeleton() -> Any:
    try:
        from ardy.skeleton import G1Skeleton34
    except ModuleNotFoundError as error:
        if error.name != "ardy":
            raise
        sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor" / "ardy"))
        from ardy.skeleton import G1Skeleton34
    return G1Skeleton34()


def neutral_source_pose(skeleton: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """G1 default FK bind, standing with its lowest neutral joint at y=0.

    This does not sample a recorded/generated animation frame. The source
    skeleton's own neutral geometry and identity local rotations define bind.
    """
    if skeleton is None:
        skeleton = _default_skeleton()
    if len(skeleton.bone_order_names) != 34 or skeleton.bone_order_names[0] != "pelvis_skel" or not set(SOURCE_NAMES.values()).issubset(skeleton.bone_order_names):
        raise RigMappingError("Retargeting requires the ARDY G1 34-joint source skeleton")
    import torch
    neutral = _array(skeleton.neutral_joints)
    root = np.array([0, neutral[0, 1] - np.min(neutral[:, 1]), 0])
    rotations = np.tile(np.eye(3), (34, 1, 1))
    with torch.inference_mode():
        global_rotations, positions, _ = skeleton.fk(torch.tensor(rotations, dtype=torch.float64), torch.tensor(root, dtype=torch.float64))
    return _array(positions).copy(), _array(global_rotations).copy()


def _anatomical_basis(positions: Mapping[str, np.ndarray], label: str) -> np.ndarray:
    left = positions["left_thigh"] - positions["right_thigh"]
    up = (positions["left_upper_arm"] + positions["right_upper_arm"]) / 2 - positions["pelvis"]
    if np.linalg.norm(left) < 1e-6:
        raise RigMappingError(f"{label} hip rest positions cannot define a left/right axis")
    left = left / np.linalg.norm(left)
    up = up - left * np.dot(up, left)
    if np.linalg.norm(up) < 1e-6:
        raise RigMappingError(f"{label} shoulder/pelvis rest positions cannot define an up axis")
    up = up / np.linalg.norm(up)
    return np.stack([left, up, np.cross(left, up)], axis=1)


def _leg_length(positions: Mapping[str, np.ndarray]) -> float:
    lengths = []
    for side in ("left", "right"):
        lengths.append(sum(float(np.linalg.norm(positions[f"{side}_{b}"] - positions[f"{side}_{a}"])) for a, b in (("thigh", "shin"), ("shin", "foot"))))
    if min(lengths) < 1e-6:
        raise RigMappingError("Rest leg lengths must be nonzero")
    return sum(lengths) / 2


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-8:
        raise RigMappingError(f"{label} must be a finite nonzero 3D direction")
    return vector / np.linalg.norm(vector)


def _swing(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Shortest proper rotation from one unit axis to another, including 180°."""
    cosine = min(1.0, max(-1.0, float(np.dot(start, end))))
    cross = np.cross(start, end)
    sine = float(np.linalg.norm(cross))
    if sine < 1e-12:
        if cosine >= 0:
            return np.eye(3)
        reference = np.eye(3)[min(range(3), key=lambda i: abs(start[i]))]
        axis = _unit(np.cross(start, reference), "opposite-axis swing")
        return 2 * axis[:, None] @ axis[None, :] - np.eye(3)
    axis = cross / sine
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return cosine * np.eye(3) + (1 - cosine) * axis[:, None] @ axis[None, :] + sine * skew


class HumanoidRetargeter:
    def __init__(self, asset: Any, profile: RigProfile, skeleton: Any = None):
        self.asset = asset
        self.profile = profile
        self.skeleton = _default_skeleton() if skeleton is None else skeleton
        self.source_positions, self.source_rotations = neutral_source_pose(self.skeleton)
        self.source_indices = {role: self.skeleton.bone_index[name] for role, name in SOURCE_NAMES.items()}
        self.rest_local = np.stack([node.local_matrix for node in asset.nodes]).copy()
        self.rest_world = np.stack([node.world_matrix for node in asset.nodes]).copy()
        # All node paths are checked, because helpers/armature roots participate
        # in the same local-to-world matrix chain as mapped joints.
        self.local_scales = []
        self.world_rotations = []
        for index in range(len(asset.nodes)):
            _, scale = _similarity(self.rest_local[index], f"Node {index} local bind")
            rotation, _ = _similarity(self.rest_world[index], f"Node {index} world bind")
            self.local_scales.append(scale)
            self.world_rotations.append(rotation)
        source = {role: self.source_positions[index] for role, index in self.source_indices.items()}
        target = {role: self.rest_world[index, :3, 3] for role, index in profile.bones.items()}
        source_basis = _anatomical_basis(source, "Source")
        target_basis = _anatomical_basis(target, "Target")
        self.basis = target_basis @ source_basis.T if profile.source_to_target_basis is None else profile.source_to_target_basis.copy()
        self.root_scale = _leg_length(target) / _leg_length(source) if profile.root_scale is None else float(profile.root_scale)
        for side in ("left", "right"):
            if np.linalg.norm(target[f"{side}_forearm"] - target[f"{side}_upper_arm"]) < 1e-6:
                raise RigMappingError(f"{side} arm rest length must be nonzero")
        self.limb_axes = {}
        self.calibrated_rotations = list(self.world_rotations)
        warnings = []
        for role, end_role in LIMB_ENDS.items():
            index = profile.bones[role]
            rest_rotation = self.world_rotations[index]
            explicit_axis = (profile.bone_axes or {}).get(role)
            if end_role in target:
                rest_axis = _unit(target[end_role] - target[role], f"{role} rest segment")
                local_axis = rest_rotation.T @ rest_axis
                if explicit_axis is not None and not np.allclose(explicit_axis, local_axis, atol=1e-5, rtol=0):
                    raise RigMappingError(f"{role} bone_axes conflicts with its mapped child rest direction")
            elif explicit_axis is not None:
                local_axis = explicit_axis
                rest_axis = rest_rotation @ local_axis
            else:
                # A terminal forearm has no geometric tail in glTF. Minimal
                # rigs use a straight-arm bind assumption; supply a hand role
                # or explicit local axis for a bent forearm bind.
                rest_axis = _unit(target[role] - target[ROLE_PARENTS[role]], f"{role} inferred rest axis")
                local_axis = rest_rotation.T @ rest_axis
                warnings.append(f"{role}: no hand endpoint; assumes a straight-arm bind. Map the hand or provide bone_axes for a bent bind.")
            source_axis = _unit(self.basis @ (source[end_role] - source[role]), f"{role} source rest segment")
            bind_swing = _swing(rest_axis, source_axis)
            self.calibrated_rotations[index] = bind_swing @ rest_rotation
            self.limb_axes[role] = local_axis
            if end_role.endswith("_hand") and end_role in profile.bones:
                hand = profile.bones[end_role]
                self.calibrated_rotations[hand] = bind_swing @ self.world_rotations[hand]
        self.warnings = tuple(warnings)
        self._roles_by_node = {node: role for role, node in profile.bones.items()}
        remaining = set(range(len(asset.nodes)))
        order = []
        while remaining:
            ready = [i for i in sorted(remaining) if asset.nodes[i].parent is None or asset.nodes[i].parent not in remaining]
            if not ready:
                raise RigMappingError("Cyclic node hierarchy")
            order.extend(ready)
            remaining.difference_update(ready)
        self._order = tuple(order)

    def neutral_source_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.source_positions.copy(), self.source_rotations.copy()

    def bind_pose(self) -> TargetPose:
        """Untouched glTF rest matrices, distinct from retargeted G1 neutral."""
        root = self.profile.bones["pelvis"]
        return TargetPose(self.rest_local.copy(), self.rest_world.copy(), self.rest_world[root, :3, 3].copy())

    def retarget(self, global_positions: Any, global_rotations: Any) -> TargetPose:
        positions, rotations = _array(global_positions), _array(global_rotations)
        if positions.shape != (34, 3) or rotations.shape != (34, 3, 3) or not np.isfinite(positions).all() or not np.isfinite(rotations).all():
            raise RigMappingError("Source pose must contain finite positions (34,3) and global rotations (34,3,3)")
        for rotation in rotations:
            _rotation(rotation, "Source joint rotation", tolerance=1e-3)
        local = self.rest_local.copy()
        world = self.rest_world.copy()
        root_node = self.profile.bones["pelvis"]
        root_world = self.rest_world[root_node, :3, 3] + self.basis @ (positions[0] - self.source_positions[0]) * self.root_scale
        for index in self._order:
            parent = self.asset.nodes[index].parent
            parent_world = np.eye(4) if parent is None else world[parent]
            role = self._roles_by_node.get(index)
            if role is not None:
                source_index = self.source_indices[role]
                delta = self.basis @ rotations[source_index] @ self.source_rotations[source_index].T @ self.basis.T
                target_rotation = delta @ self.calibrated_rotations[index]
                if role in LIMB_ENDS:
                    source_end = self.source_indices[LIMB_ENDS[role]]
                    direction = _unit(self.basis @ (positions[source_end] - positions[source_index]), f"{role} source pose segment")
                    posed_axis = _unit(target_rotation @ self.limb_axes[role], f"{role} posed axis")
                    target_rotation = _swing(posed_axis, direction) @ target_rotation
                parent_rotation, _ = _similarity(parent_world, f"Node {index} posed parent")
                local[index, :3, :3] = parent_rotation.T @ target_rotation * self.local_scales[index]
            if index == root_node:
                local[index, :3, 3] = (np.linalg.inv(parent_world) @ np.array([*root_world, 1]))[:3]
            world[index] = parent_world @ local[index]
        return TargetPose(local, world, world[root_node, :3, 3].copy())


def _mapping_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise RigMappingError(f"Duplicate mapping JSON key: {key}")
        result[key] = value
    return result


def build_retargeter(asset: Any, mapping: Mapping[str, Any] | str | bytes | RigProfile | None = None, *, skeleton: Any = None) -> HumanoidRetargeter:
    """Validate an auto-detected or user JSON mapping and calibrate to bind."""
    if isinstance(mapping, (str, bytes)):
        try:
            mapping = json.loads(mapping, object_pairs_hook=_mapping_pairs)
        except (ValueError, UnicodeDecodeError) as error:
            raise RigMappingError(f"Invalid mapping JSON: {error}") from error
        if not isinstance(mapping, Mapping):
            raise RigMappingError("Mapping JSON must contain an object")
    if isinstance(mapping, RigProfile):
        profile = _resolve_profile(asset, {"profile_name": mapping.profile_name, "bones": mapping.bones, "source_to_target_basis": mapping.source_to_target_basis, "root_scale": mapping.root_scale, "bone_axes": mapping.bone_axes})
    elif mapping is None:
        profile = detect_rig_profile(asset)
        if profile is None:
            raise RigMappingError("Mapping required: provide the required humanoid bone roles as JSON")
    elif isinstance(mapping, Mapping):
        profile = _resolve_profile(asset, mapping)
    else:
        raise RigMappingError("Mapping must be a JSON object")
    return HumanoidRetargeter(asset, profile, skeleton)
