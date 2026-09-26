"""Descriptive body-proxy overlap with StageZero scene solids.

This is a sampled sphere-versus-oriented-box measurement in world metres,
not mesh collision, collision response, contacts, or physical safety. Custom
gate holes are used only when their trusted passage affordance is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from interaction_scene import SceneObject, local_axes, passage_for, scene_objects, is_walkable_ground
from motion_quality import G1_JOINT_NAMES


CORE27_JOINT_NAMES = (
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandEnd",
    "RightHandThumb1", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandEnd", "LeftHandThumb1", "RightUpLeg", "RightLeg", "RightFoot",
    "RightToeBase", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
)

_SPECS = {
    "core27": {
        "names": CORE27_JOINT_NAMES,
        "spheres": {"Hips": .22, "Spine2": .18, "Spine3": .17, "Head": .14,
                    "LeftShoulder": .12, "RightShoulder": .12,
                    "LeftArm": .11, "RightArm": .11,
                    "LeftForeArm": .10, "RightForeArm": .10,
                    "LeftHandEnd": .09, "RightHandEnd": .09,
                    "LeftUpLeg": .14, "RightUpLeg": .14,
                    "LeftLeg": .12, "RightLeg": .12,
                    "LeftFoot": .10, "RightFoot": .10,
                    "LeftToeBase": .09, "RightToeBase": .09},
        "midpoints": (("Hips", "Spine2", .18), ("Spine2", "Head", .15),
                      ("LeftShoulder", "LeftForeArm", .10),
                      ("RightShoulder", "RightForeArm", .10),
                      ("LeftForeArm", "LeftHandEnd", .08),
                      ("RightForeArm", "RightHandEnd", .08),
                      ("Hips", "LeftLeg", .13), ("Hips", "RightLeg", .13),
                      ("LeftLeg", "LeftFoot", .10), ("RightLeg", "RightFoot", .10)),
    },
    "g1": {
        "names": G1_JOINT_NAMES,
        "spheres": {"pelvis_skel": .19, "waist_yaw_skel": .17,
                    "waist_pitch_skel": .16,
                    "left_shoulder_pitch_skel": .12, "right_shoulder_pitch_skel": .12,
                    "left_elbow_skel": .10, "right_elbow_skel": .10,
                    "left_hand_roll_skel": .08, "right_hand_roll_skel": .08,
                    "left_hip_pitch_skel": .13, "right_hip_pitch_skel": .13,
                    "left_knee_skel": .11, "right_knee_skel": .11,
                    "left_ankle_pitch_skel": .09, "right_ankle_pitch_skel": .09,
                    "left_toe_base": .08, "right_toe_base": .08},
        "midpoints": (("pelvis_skel", "waist_pitch_skel", .16),
                      ("waist_pitch_skel", "left_shoulder_pitch_skel", .11),
                      ("waist_pitch_skel", "right_shoulder_pitch_skel", .11),
                      ("left_shoulder_pitch_skel", "left_elbow_skel", .10),
                      ("right_shoulder_pitch_skel", "right_elbow_skel", .10),
                      ("left_elbow_skel", "left_hand_roll_skel", .08),
                      ("right_elbow_skel", "right_hand_roll_skel", .08),
                      ("left_hip_pitch_skel", "left_knee_skel", .11),
                      ("right_hip_pitch_skel", "right_knee_skel", .11),
                      ("left_knee_skel", "left_ankle_pitch_skel", .09),
                      ("right_knee_skel", "right_ankle_pitch_skel", .09)),
    },
}


@dataclass(frozen=True)
class _Box:
    object_id: str
    center: tuple[float, float, float]
    half_size: tuple[float, float, float]
    yaw_degrees: float


def _box(obj: SceneObject, *, x_local: float = 0., y_local: float = 0.,
         width: float | None = None, height: float | None = None,
         depth: float | None = None) -> _Box:
    width_axis, _ = local_axes(obj.yaw_degrees)
    return _Box(obj.id, (obj.x + x_local * width_axis[0], obj.y + y_local,
                         obj.z + x_local * width_axis[1]),
                ((obj.width if width is None else width) / 2,
                 (obj.height if height is None else height) / 2,
                 (obj.depth if depth is None else depth) / 2), obj.yaw_degrees)


def _arch_boxes(obj: SceneObject) -> list[_Box]:
    boxes = []
    for sign in (-1, 1):
        boxes.append(_box(obj, x_local=sign * .405 * obj.width,
                          y_local=-.095 * obj.height,
                          width=.19 * obj.width, height=.81 * obj.height,
                          depth=.9 * obj.depth))
        boxes.append(_box(obj, x_local=sign * .385 * obj.width,
                          y_local=-.47 * obj.height,
                          width=.23 * obj.width, height=.06 * obj.height))
    boxes.append(_box(obj, y_local=.405 * obj.height,
                      width=obj.width, height=.19 * obj.height))
    return boxes


def _custom_passage_boxes(obj: SceneObject, passage) -> list[_Box]:
    width_axis, _ = local_axes(obj.yaw_degrees)
    delta = (passage.center_xz[0] - obj.x, passage.center_xz[1] - obj.z)
    offset = delta[0] * width_axis[0] + delta[1] * width_axis[1]
    left, right = offset - passage.width_m / 2, offset + passage.width_m / 2
    boxes = []
    for lo, hi in ((-obj.width / 2, left), (right, obj.width / 2)):
        if hi - lo > 1e-6:
            boxes.append(_box(obj, x_local=(lo + hi) / 2, width=hi - lo))
    return boxes


def _custom_boxes(obj: SceneObject, raw: Mapping) -> list[_Box]:
    # `passage_for` validates the trusted hole and its floor in world Y.
    passage = passage_for(obj, {obj.id: raw}, actor_height_m=0.)
    boxes = _custom_passage_boxes(obj, passage)
    floor = obj.y - obj.height / 2
    opening_floor = float(raw["floor_y_m"])
    if opening_floor - floor > 1e-6:
        boxes.append(_box(obj, y_local=(floor + opening_floor) / 2 - obj.y,
                          height=opening_floor - floor))
    top = obj.y + obj.height / 2
    opening_top = opening_floor + passage.height_m
    if top - opening_top > 1e-6:
        boxes.append(_box(obj, y_local=(top + opening_top) / 2 - obj.y,
                          height=top - opening_top))
    return boxes


def _scene_boxes(scene: Mapping, affordances: Mapping | None) -> tuple[list[SceneObject], list[_Box]]:
    objects = scene_objects(scene)
    if affordances is not None and not isinstance(affordances, Mapping):
        raise ValueError("affordances must be object-id keyed metadata")
    known = {obj.id for obj in objects}
    if affordances and set(affordances) - known:
        raise ValueError("affordances reference unknown object IDs")
    boxes = []
    for obj in objects:
        if is_walkable_ground(obj):
            continue  # Walkable floor slab; foot contact is measured elsewhere.
        if obj.kind == "arch":
            boxes.extend(_arch_boxes(obj))
        elif obj.kind == "custom" and affordances and obj.id in affordances:
            boxes.extend(_custom_boxes(obj, affordances[obj.id]))
        else:
            boxes.append(_box(obj))
    return objects, boxes


def _body_proxies(positions: np.ndarray, skeleton: str) -> tuple[np.ndarray, np.ndarray]:
    spec = _SPECS.get(skeleton)
    if spec is None:
        raise ValueError("skeleton must be 'core27' or 'g1'")
    p = np.asarray(positions)
    if p.ndim != 3 or p.shape[1:] != (len(spec["names"]), 3) or len(p) == 0:
        raise ValueError(f"positions must be (frames, {len(spec['names'])}, 3)")
    if p.dtype.kind not in "fi" or not np.isfinite(p).all():
        raise ValueError("positions must contain finite numeric world metres")
    index = {name: i for i, name in enumerate(spec["names"])}
    samples = [p[:, index[name], :] for name in spec["spheres"]]
    radii = list(spec["spheres"].values())
    for first, second, radius in spec["midpoints"]:
        samples.append((p[:, index[first], :] + p[:, index[second], :]) / 2)
        radii.append(radius)
    return np.stack(samples, axis=1).astype(np.float64), np.asarray(radii, dtype=np.float64)


def _signed_clearance(samples: np.ndarray, radii: np.ndarray, box: _Box) -> np.ndarray:
    width_axis, normal = local_axes(box.yaw_degrees)
    dx, dy, dz = (samples[..., i] - box.center[i] for i in range(3))
    local = np.stack((dx * width_axis[0] + dz * width_axis[1], dy,
                      dx * normal[0] + dz * normal[1]), axis=-1)
    offset = np.abs(local) - np.asarray(box.half_size)
    outside = np.linalg.norm(np.maximum(offset, 0), axis=-1)
    inside = np.minimum(np.max(offset, axis=-1), 0)
    return outside + inside - radii[None, :]


def scene_collision(positions: object, skeleton: str, scene: Mapping,
                    affordances: Mapping | None = None) -> dict:
    """Report sampled body-sphere overlap against oriented scene boxes.

    `collision_frames` means a sampled proxy sphere overlapped a conservative
    prop box. A zero count does not establish mesh clearance between samples.
    `minimum_clearance_m` is signed: negative means proxy penetration.
    """
    samples, radii = _body_proxies(np.asarray(positions), skeleton)
    objects, boxes = _scene_boxes(scene, affordances)
    per_object = []
    any_hit = np.zeros(len(samples), dtype=bool)
    for obj in objects:
        owned = [box for box in boxes if box.object_id == obj.id]
        if not owned:
            continue
        minimum_each_frame = np.minimum.reduce([_signed_clearance(samples, radii, box).min(axis=1)
                                                for box in owned])
        collided = minimum_each_frame < 0
        any_hit |= collided
        indices = np.flatnonzero(collided)
        per_object.append({"object_id": obj.id, "kind": obj.kind,
                           "collision_frames": int(collided.sum()),
                           "first_collision_frame": int(indices[0]) if len(indices) else None,
                           "minimum_clearance_m": float(minimum_each_frame.min()),
                           "worst_penetration_m": float(max(0., -minimum_each_frame.min()))})
    return {"version": 1, "skeleton": skeleton, "frames": len(samples),
            "proxy": "named_joint_spheres_and_segment_midpoints_vs_oriented_boxes",
            "sampled_body_points_per_frame": int(len(radii)),
            "total_collision_frames": int(any_hit.sum()),
            "per_object": per_object,
            "assumptions": {"units": "metres", "up_axis": "+Y", "scene_source": "object_metadata",
                            "object_geometry": "conservative boxes with explicit passage cutouts",
                            "mesh_or_physics_collision_verified": False}}
