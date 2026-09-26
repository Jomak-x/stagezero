"""CPU-only, descriptive quality measurements for ARDY G1 pose clips.

Input ``positions`` is ``(frames, 34, 3)`` in metres, in world coordinates.
The Y axis is up. ``rotations``, when supplied, is ``(frames, 34, 3, 3)``.
Joint order is copied from ``G1Skeleton34.bone_order_names_with_parents`` in
``vendor/ardy/ardy/skeleton/definitions.py``; importing that class would also
import PyTorch, so this module keeps the named order locally.

The measurements describe motion, not semantic prompt adherence. In particular,
toe height is only a proxy for contact: the array does not contain forces, foot
soles, terrain geometry, or the model's contact labels.
"""

from __future__ import annotations

import numpy as np


G1_JOINT_NAMES = (
    "pelvis_skel",
    "left_hip_pitch_skel", "left_hip_roll_skel", "left_hip_yaw_skel",
    "left_knee_skel", "left_ankle_pitch_skel", "left_ankle_roll_skel",
    "left_toe_base",
    "right_hip_pitch_skel", "right_hip_roll_skel", "right_hip_yaw_skel",
    "right_knee_skel", "right_ankle_pitch_skel", "right_ankle_roll_skel",
    "right_toe_base",
    "waist_yaw_skel", "waist_roll_skel", "waist_pitch_skel",
    "left_shoulder_pitch_skel", "left_shoulder_roll_skel",
    "left_shoulder_yaw_skel", "left_elbow_skel", "left_wrist_roll_skel",
    "left_wrist_pitch_skel", "left_wrist_yaw_skel", "left_hand_roll_skel",
    "right_shoulder_pitch_skel", "right_shoulder_roll_skel",
    "right_shoulder_yaw_skel", "right_elbow_skel", "right_wrist_roll_skel",
    "right_wrist_pitch_skel", "right_wrist_yaw_skel", "right_hand_roll_skel",
)
JOINT_INDEX = {name: index for index, name in enumerate(G1_JOINT_NAMES)}
ROOT = JOINT_INDEX["pelvis_skel"]
TOES = (JOINT_INDEX["left_toe_base"], JOINT_INDEX["right_toe_base"])
SHOULDERS = (JOINT_INDEX["left_shoulder_pitch_skel"], JOINT_INDEX["right_shoulder_pitch_skel"])
HANDS = (JOINT_INDEX["left_hand_roll_skel"], JOINT_INDEX["right_hand_roll_skel"])


def _positions(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[1:] != (34, 3) or not len(array):
        raise ValueError(f"{label} must have shape (frames, 34, 3) with at least one frame")
    if array.dtype.kind not in "fi":
        raise ValueError(f"{label} must contain numeric positions")
    return array


def _rotations(value: np.ndarray, frames: int, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (frames, 34, 3, 3) or array.dtype.kind not in "fi":
        raise ValueError(f"{label} must have shape ({frames}, 34, 3, 3) and numeric values")
    return array


def _rotation_metrics(rotations: np.ndarray | None) -> dict:
    if rotations is None:
        return {"rotations_checked": False, "rotations_finite": None, "rotations_valid": None,
                "invalid_rotation_count": None, "max_rotation_orthogonality_error": None,
                "max_rotation_determinant_error": None}
    finite = np.isfinite(rotations).all(axis=(-2, -1))
    invalid = int((~finite).sum())
    if finite.any():
        matrices = rotations[finite].astype(np.float64)
        orth_error = np.max(np.abs(matrices @ np.swapaxes(matrices, -1, -2) - np.eye(3)), axis=(-2, -1))
        det_error = np.abs(np.linalg.det(matrices) - 1.0)
        invalid += int(np.count_nonzero((orth_error > .02) | (det_error > .02)))
        max_orth = float(orth_error.max())
        max_det = float(det_error.max())
    else:
        max_orth = max_det = None
    return {"rotations_checked": True, "rotations_finite": bool(finite.all()),
            "rotations_valid": invalid == 0,
            "invalid_rotation_count": invalid,
            "max_rotation_orthogonality_error": max_orth,
            "max_rotation_determinant_error": max_det}


def analyze_motion(
    positions: np.ndarray,
    rotations: np.ndarray | None = None,
    *,
    fps: float = 25.0,
    prior_positions: np.ndarray | None = None,
    prior_rotations: np.ndarray | None = None,
    floor_height: float = 0.0,
    contact_height: float = .08,
    contact_vertical_speed: float = .15,
    penetration_tolerance: float = .01,
    overhead_margin: float = 0.0,
) -> dict:
    """Return JSON-safe clip metrics, with optional previous clip boundary metrics.

    Sliding samples are adjacent toe positions whose endpoints are both within
    ``contact_height`` metres of the floor and whose vertical speed is at most
    ``contact_vertical_speed`` m/s. This is a geometric proxy, not verified foot
    contact. ``overhead_fraction`` uses either hand above its same-side shoulder.
    ``root_drop_from_start_m`` is a squat cue, not a squat detector.
    """
    p = _positions(positions, "positions")
    r = None if rotations is None else _rotations(rotations, len(p), "rotations")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    for name, value in (("floor_height", floor_height), ("contact_height", contact_height),
                        ("contact_vertical_speed", contact_vertical_speed),
                        ("penetration_tolerance", penetration_tolerance),
                        ("overhead_margin", overhead_margin)):
        if not np.isfinite(value) or (name in ("contact_height", "contact_vertical_speed", "penetration_tolerance") and value < 0):
            raise ValueError(f"{name} must be finite and nonnegative where applicable")
    previous = None if prior_positions is None else _positions(prior_positions, "prior_positions")
    previous_rotations = None
    if prior_rotations is not None:
        if previous is None:
            raise ValueError("prior_rotations requires prior_positions")
        previous_rotations = _rotations(prior_rotations, len(previous), "prior_rotations")

    output = {"frames": int(len(p)), "fps": float(fps),
              "positions_finite": bool(np.isfinite(p).all()),
              "nonfinite_position_count": int(np.count_nonzero(~np.isfinite(p))),
              **_rotation_metrics(r)}
    if not output["positions_finite"]:
        output["quality_metrics_available"] = False
        output["reason"] = "Non-finite positions prevent geometric measurements"
        return output
    output["quality_metrics_available"] = True
    p = p.astype(np.float64)
    root = p[:, ROOT]
    planar_steps = np.linalg.norm(np.diff(root[:, (0, 2)], axis=0), axis=1)
    root_speeds = planar_steps * fps
    output["root_path_length_m"] = float(planar_steps.sum())
    output["root_net_displacement_m"] = float(np.linalg.norm(root[-1, (0, 2)] - root[0, (0, 2)]))
    output["root_mean_speed_mps"] = float(root_speeds.mean()) if len(root_speeds) else None
    output["root_peak_speed_mps"] = float(root_speeds.max()) if len(root_speeds) else None
    output["root_min_height_m"] = float(root[:, 1].min())
    output["root_max_height_m"] = float(root[:, 1].max())
    output["root_vertical_range_m"] = float(np.ptp(root[:, 1]))
    output["root_drop_from_start_m"] = float(max(0.0, root[0, 1] - root[:, 1].min()))

    toe_heights = p[:, TOES, 1] - floor_height
    output["toe_min_height_above_floor_m"] = float(toe_heights.min())
    output["floor_penetration_max_depth_m"] = float(max(0.0, -toe_heights.min()))
    output["floor_penetration_frames"] = int(np.count_nonzero((toe_heights < -penetration_tolerance).any(axis=1)))
    output["floor_penetration_fraction"] = float(output["floor_penetration_frames"] / len(p))

    if len(p) > 1:
        toe_delta = np.diff(p[:, TOES, :], axis=0)
        toe_speed = np.linalg.norm(toe_delta[..., (0, 2)], axis=-1) * fps
        vertical_speed = np.abs(toe_delta[..., 1]) * fps
        contact_proxy = ((toe_heights[:-1] <= contact_height) &
                         (toe_heights[1:] <= contact_height) &
                         (vertical_speed <= contact_vertical_speed))
        sliding = toe_speed[contact_proxy]
    else:
        sliding = np.empty(0)
    output["foot_contact_proxy_samples"] = int(len(sliding))
    output["foot_sliding_proxy_mean_mps"] = float(sliding.mean()) if len(sliding) else None
    output["foot_sliding_proxy_peak_mps"] = float(sliding.max()) if len(sliding) else None

    hand_above = p[:, HANDS, 1] - p[:, SHOULDERS, 1]
    output["left_hand_above_shoulder_mean_m"] = float(hand_above[:, 0].mean())
    output["right_hand_above_shoulder_mean_m"] = float(hand_above[:, 1].mean())
    output["left_hand_above_shoulder_max_m"] = float(hand_above[:, 0].max())
    output["right_hand_above_shoulder_max_m"] = float(hand_above[:, 1].max())
    output["left_hand_overhead_fraction"] = float(np.mean(hand_above[:, 0] > overhead_margin))
    output["right_hand_overhead_fraction"] = float(np.mean(hand_above[:, 1] > overhead_margin))
    output["either_hand_overhead_fraction"] = float(np.mean((hand_above > overhead_margin).any(axis=1)))

    if previous is None or not np.isfinite(previous).all():
        output["boundary_available"] = False
        output["boundary_reason"] = "No finite prior positions supplied" if previous is None else "Prior positions contain non-finite values"
    else:
        previous = previous.astype(np.float64)
        delta = p[0] - previous[-1]
        boundary_v = delta[ROOT] * fps
        output["boundary_available"] = True
        output["boundary_root_position_jump_m"] = float(np.linalg.norm(delta[ROOT]))
        output["boundary_mean_joint_position_jump_m"] = float(np.linalg.norm(delta, axis=1).mean())
        output["boundary_root_speed_mps"] = float(np.linalg.norm(boundary_v))
        output["boundary_entry_velocity_jump_mps"] = (float(np.linalg.norm(boundary_v - (previous[-1, ROOT] - previous[-2, ROOT]) * fps))
                                                       if len(previous) >= 2 else None)
        output["boundary_exit_velocity_jump_mps"] = (float(np.linalg.norm((p[1, ROOT] - p[0, ROOT]) * fps - boundary_v))
                                                      if len(p) >= 2 else None)
    output["boundary_root_rotation_jump_deg"] = None
    output["boundary_peak_joint_rotation_jump_deg"] = None
    if (output["boundary_available"] and r is not None and previous_rotations is not None
            and output["rotations_valid"] and _rotation_metrics(previous_rotations)["rotations_valid"]):
        relative = r[0] @ np.swapaxes(previous_rotations[-1], -1, -2)
        cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
        angle = np.degrees(np.arccos(cosine))
        output["boundary_root_rotation_jump_deg"] = float(angle[ROOT])
        output["boundary_peak_joint_rotation_jump_deg"] = float(angle.max())
    return output
