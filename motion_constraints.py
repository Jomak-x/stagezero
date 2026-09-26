"""Optional ARDY G1 root waypoint conditioning for one 104-frame request.

The waypoint is a world-space X/Z position on the last frame of either
52-frame generation step. This is model conditioning through ARDY's official
Root2DConstraintSet, never a post-generation pose edit. A frame-103 waypoint
is applied in the second step; it does not make the first step plan ahead.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np


HORIZON_FRAMES = 52
REQUEST_FRAMES = 104
WAYPOINT_FRAMES = (51, 103)
MAX_TARGET_OFFSET_M = 3.0


def _position_xz(value, *, label):
    if (not isinstance(value, (list, tuple, np.ndarray))
            or isinstance(value, np.ndarray) and value.ndim != 1
            or len(value) != 2):
        raise ValueError(f"{label} must contain two X/Z coordinates in meters")
    if any(isinstance(coordinate, bool) or not isinstance(coordinate, Real)
           or not math.isfinite(coordinate) for coordinate in value):
        raise ValueError(f"{label} must contain finite numeric X/Z coordinates")
    return [float(value[0]), float(value[1])]


def validate_motion_target(value, *, prior_root_xz):
    """Return a normalized JSON-safe waypoint, or None when it was omitted.

    A continuation history must supply ``prior_root_xz`` in world meters.
    The 3 m bound prevents an accidental far-away world target; it is not a
    promise that ARDY can reach that destination within the allotted time.
    """
    if value is None:
        return None
    if prior_root_xz is None:
        raise ValueError("A motion target requires prior actor history")
    prior = _position_xz(prior_root_xz, label="Prior root position")
    if not isinstance(value, dict) or set(value) != {"position_xz", "frame"}:
        raise ValueError("motion_target requires position_xz and frame only")
    position = _position_xz(value["position_xz"], label="Target position")
    frame = value["frame"]
    if isinstance(frame, bool) or not isinstance(frame, int) or frame not in WAYPOINT_FRAMES:
        raise ValueError("Target frame must be 51 or 103 in the 104-frame result")
    if math.dist(position, prior) > MAX_TARGET_OFFSET_M:
        raise ValueError("Target must be within 3 m of the prior root position")
    return {"position_xz": position, "frame": frame}


def build_root_conditions(model, target, *, history_length, generated_offset, device):
    """Build (observed_motion, motion_mask) for one official 52-frame step.

    ``target`` is the result of ``validate_motion_target``. The caller passes
    its actual cropped history length and 0 or 52 generated frames so the
    frame index is translated into ARDY's history-plus-horizon window.
    Returns (None, None) when the waypoint belongs to the other step.
    """
    if target is None:
        return None, None
    if not isinstance(target, dict) or set(target) != {"position_xz", "frame"}:
        raise ValueError("Invalid validated motion target")
    position = _position_xz(target["position_xz"], label="Target position")
    frame = target["frame"]
    if isinstance(frame, bool) or not isinstance(frame, int) or frame not in WAYPOINT_FRAMES:
        raise ValueError("Target frame must be 51 or 103")
    if (isinstance(history_length, bool) or not isinstance(history_length, int)
            or history_length < 0 or history_length > HORIZON_FRAMES
            or history_length % 4):
        raise ValueError("History length must be 0–52 frames and divisible by four")
    if generated_offset not in (0, HORIZON_FRAMES):
        raise ValueError("Generated offset must be 0 or 52")
    if model.gen_horizon_len != HORIZON_FRAMES or model.num_frames_per_token != 4:
        raise ValueError("Unexpected ARDY model horizon or token size")
    if frame != generated_offset + HORIZON_FRAMES - 1:
        return None, None

    import torch
    from ardy.constraints import Root2DConstraintSet

    total_frames = history_length + HORIZON_FRAMES
    window_frame = history_length + frame - generated_offset
    constraint = Root2DConstraintSet(
        model.skeleton,
        frame_indices=torch.tensor([window_frame], dtype=torch.long, device=device),
        root_2d=torch.tensor([position], dtype=torch.float32, device=device),
    )
    lengths = torch.tensor([total_frames], dtype=torch.long, device=device)
    observed, motion_mask = model.motion_rep.create_conditions_from_constraints_batched(
        [[constraint]], lengths, to_normalize=True, device=device,
    )
    expected = (1, total_frames, model.motion_rep.motion_rep_dim)
    if observed.shape != expected or motion_mask.shape != expected:
        raise ValueError("ARDY returned invalid root condition shapes")
    if not torch.isfinite(observed).all().item():
        raise ValueError("ARDY returned nonfinite root conditions")
    if torch.count_nonzero(motion_mask[:, :history_length]).item():
        raise ValueError("Root condition overlaps the immutable history")
    if not torch.count_nonzero(motion_mask[:, history_length:]).item():
        raise ValueError("ARDY returned an empty root condition")
    return observed, motion_mask
