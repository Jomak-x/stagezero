"""Native ARDY pose goals for two reference-assisted G1 actions.

The small goal asset contains three keyframes per action, copied from prior
successful ARDY generations.  The first 52-frame horizon is generated without
these goals.  For the second horizon, the keyframes are moved into the current
actor's world position and facing direction, then passed through ARDY's own
constraint encoder.  Returned arrays are model *inputs*; this module never
edits generated motion.  A goal is a cue, not a physical or semantic guarantee.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np


MODEL_NAME = "ARDY-G1-RP-25FPS-Horizon52"
GOAL_ASSET = Path(__file__).resolve().parent / "assets" / "motion-goals.npz"
SUPPORTED_ACTIONS = ("overhead", "squat")


@lru_cache(maxsize=1)
def load_action_goal_asset() -> tuple[dict, dict]:
    """Read and validate generated reference poses without loading ARDY."""
    with np.load(GOAL_ASSET, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        arrays = {name: np.asarray(data[name], dtype=np.float32).copy()
                  for name in data.files if name != "metadata"}
    if metadata.get("schema_version") != 1 or metadata.get("model") != MODEL_NAME:
        raise ValueError("Incompatible motion goal asset")
    if set(metadata.get("actions", {})) != set(SUPPORTED_ACTIONS):
        raise ValueError("Motion goal asset lacks supported actions")
    for action in SUPPORTED_ACTIONS:
        info = metadata["actions"][action]
        count = len(info["keyframes_in_104_frame_generation"])
        expected = {
            f"{action}_positions": (count, 34, 3),
            f"{action}_rotations": (count, 34, 3, 3),
            f"{action}_anchor_root": (3,),
            f"{action}_anchor_right_hip": (3,),
            f"{action}_anchor_left_hip": (3,),
        }
        for name, shape in expected.items():
            if name not in arrays or arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError(f"Invalid motion goal array: {name}")
        frames = info["keyframes_in_104_frame_generation"]
        if not all(isinstance(frame, int) and 52 <= frame < 104 for frame in frames):
            raise ValueError(f"Invalid goal keyframes for {action}")
    return metadata, arrays


def _heading(positions: np.ndarray, right_hip: int, left_hip: int) -> float:
    """Match ARDY's hip-vector heading, atan2(diff_z, -diff_x)."""
    diff = positions[right_hip] - positions[left_hip]
    if float(np.linalg.norm(diff[[0, 2]])) < 0.05:
        raise ValueError("Hip vector is degenerate; cannot align pose goal")
    return float(np.arctan2(diff[2], -diff[0]))


def _y_rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)), dtype=np.float32)


def prepare_goal_pose(action: str, history_positions: np.ndarray, skeleton) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rigidly align a generated reference to the current actor's XZ and yaw.

    Only the *target* pose is transformed.  Y remains in the original flat-floor
    world frame so the feet retain their reference ground height.  The model
    receives the aligned poses as soft goals and generates all intermediate
    frames itself.
    """
    if action not in SUPPORTED_ACTIONS:
        raise ValueError(f"Unsupported action goal: {action}")
    history = np.asarray(history_positions, dtype=np.float32)
    if history.ndim != 3 or history.shape[1:] != (34, 3) or not len(history) or not np.isfinite(history).all():
        raise ValueError("history_positions must be finite (frames,34,3)")
    if getattr(skeleton, "name", None) != "g1skel34" or getattr(skeleton, "nbjoints", None) != 34:
        raise ValueError("Pose goals require the pinned G1 34-joint skeleton")
    root_idx = int(skeleton.root_idx)
    right_hip, left_hip = (int(x) for x in skeleton.hip_joint_idx)
    if (root_idx, right_hip, left_hip) != (0, 8, 1):
        raise ValueError("G1 skeleton joint indices differ from the goal asset")

    metadata, arrays = load_action_goal_asset()
    info = metadata["actions"][action]
    anchor_root = arrays[f"{action}_anchor_root"]
    anchor_hips = np.zeros((34, 3), dtype=np.float32)
    anchor_hips[right_hip] = arrays[f"{action}_anchor_right_hip"]
    anchor_hips[left_hip] = arrays[f"{action}_anchor_left_hip"]
    reference_yaw = _heading(anchor_hips, right_hip, left_hip)
    actor_yaw = _heading(history[-1], right_hip, left_hip)
    delta_yaw = float(np.arctan2(np.sin(actor_yaw - reference_yaw), np.cos(actor_yaw - reference_yaw)))
    rotation_y = _y_rotation(delta_yaw)
    actor_root = history[-1, root_idx]
    offset_xz = actor_root[[0, 2]] - anchor_root[[0, 2]]

    positions = arrays[f"{action}_positions"]
    rotations = arrays[f"{action}_rotations"]
    transformed_positions = (positions - anchor_root) @ rotation_y.T + anchor_root
    transformed_positions[..., 0] += offset_xz[0]
    transformed_positions[..., 2] += offset_xz[1]
    transformed_rotations = rotation_y @ rotations
    if not np.isfinite(transformed_positions).all() or not np.isfinite(transformed_rotations).all():
        raise ValueError("Aligned goal pose contains nonfinite values")
    goal_metadata = {
        "mode": "reference_assisted_native_ardy_conditioning",
        "action": action,
        "source": info["source_file"],
        "source_seed": info["source_seed"],
        "asset_sha256": hashlib.sha256(GOAL_ASSET.read_bytes()).hexdigest(),
        "reference_keyframes": list(info["keyframes_in_104_frame_generation"]),
        "constraint_kind": info["native_constraint_kind"],
        "world_xz_translation_m": [float(x) for x in offset_xz],
        "world_yaw_rotation_rad": delta_yaw,
        "current_root_xz_m": [float(x) for x in actor_root[[0, 2]]],
        "reference_root_xz_m": [float(x) for x in anchor_root[[0, 2]]],
        "reference_profile": info["profile"],
        "reference_cfg_weight": list(info["cfg_weight"]),
        "caveat": "Generated reference goals are soft ARDY inference conditions, not learned joint interaction or exact pose guarantees.",
    }
    return transformed_positions.astype(np.float32), transformed_rotations.astype(np.float32), goal_metadata


def build_action_conditions(model, action: str, current_history, generated_offset: int = 52, device: str = "cuda"):
    """Return ``(observed_motion, motion_mask, goal_metadata)`` for horizon two.

    ``current_history`` is one actor's normalized ARDY motion of shape
    ``(1,H,414)``, with ``H`` a positive multiple of four no greater than 52.
    For ``generated_offset=0`` this returns ``(None,None,None)`` so the first
    horizon remains free.  Other offsets are unsupported.
    """
    if generated_offset == 0:
        return None, None, None
    if generated_offset != 52:
        raise ValueError("Pose goals are supported only at generated offset 52")
    if action not in SUPPORTED_ACTIONS:
        raise ValueError(f"Unsupported action goal: {action}")
    if getattr(model, "skeleton", None) is None or model.skeleton.name != "g1skel34" or model.skeleton.nbjoints != 34:
        raise ValueError("Pose goals require the pinned G1 34-joint skeleton")
    if (model.gen_horizon_len != 52 or model.motion_rep.motion_rep_dim != 414
            or model.motion_rep.fps != 25):
        raise ValueError("Pose goals require the pinned ARDY G1 horizon-52 checkpoint")

    import torch
    from ardy.constraints import EndEffectorConstraintSet, FullBodyConstraintSet

    if not isinstance(current_history, torch.Tensor) or current_history.ndim != 3:
        raise ValueError("current_history must be a normalized torch tensor (1,H,414)")
    history_len = current_history.shape[1]
    if (current_history.shape[0] != 1 or current_history.shape[2] != 414
            or not 4 <= history_len <= 52 or history_len % 4 != 0
            or not bool(torch.isfinite(current_history).all())):
        raise ValueError("current_history must be finite (1,H,414), H=4..52 divisible by 4")
    device = torch.device(device)
    with torch.inference_mode():
        decoded = model.motion_rep.inverse(current_history.to(device), is_normalized=True)
        history_positions = decoded["posed_joints"][0].detach().cpu().numpy()
    positions, rotations, metadata = prepare_goal_pose(action, history_positions, model.skeleton)
    frame_indices = [history_len + frame - 52 for frame in metadata["reference_keyframes"]]
    # Official EndEffector/FullBody constructors pair frame indices with CPU
    # skeleton indices; their condition builder moves the final mask to device.
    indices = torch.tensor(frame_indices, dtype=torch.long)
    p = torch.from_numpy(positions).to(device)
    r = torch.from_numpy(rotations).to(device)
    root = p[:, model.skeleton.root_idx, :][:, (0, 2)]
    if action == "overhead":
        constraint = EndEffectorConstraintSet(
            model.skeleton, indices, p, r, root,
            joint_names=["LeftHand", "RightHand", "Hips"],
        )
    else:
        constraint = FullBodyConstraintSet(model.skeleton, indices, p, r, root)
    lengths = torch.full((1,), history_len + 52, dtype=torch.long, device=device)
    observed, mask = model.motion_rep.create_conditions_from_constraints_batched(
        [[constraint]], lengths, to_normalize=True, device=str(device),
    )
    shape = (1, history_len + 52, 414)
    if (observed.shape != shape or mask.shape != shape
            or bool(mask[:, :history_len].count_nonzero())
            or not bool(torch.isfinite(observed).all())):
        raise ValueError("Native ARDY condition tensors are invalid or overlap history")
    channels = int(mask.count_nonzero())
    if channels == 0:
        raise ValueError("Native ARDY pose goal produced an empty mask")
    metadata["window_keyframes"] = frame_indices
    metadata["conditioned_channels"] = channels
    metadata["history_frames"] = history_len
    return observed, mask, metadata
