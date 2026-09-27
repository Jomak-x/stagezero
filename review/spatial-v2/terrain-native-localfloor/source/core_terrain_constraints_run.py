"""Explicit native ARDY root-height conditioning; never changes output poses.

The height is absolute pelvis/root Y, not terrain surface Y. The caller must
choose a support-relative pelvis clearance and verify the generated feet.
"""
from __future__ import annotations


class RootHeightConstraint:
    """Write only ARDY's supported scalar ``root_y_pos`` feature."""
    name = "root-height"

    def __init__(self, frame_indices, root_height):
        import torch
        frames = torch.as_tensor(frame_indices, dtype=torch.long)
        heights = torch.as_tensor(root_height, dtype=torch.float32, device=frames.device)
        if frames.ndim != 1 or heights.shape != frames.shape:
            raise ValueError("Root-height frames and heights must be matching vectors")
        if (frames < 0).any() or not torch.isfinite(heights).all():
            raise ValueError("Root-height targets must have nonnegative frames and finite heights")
        if len(frames.unique()) != len(frames):
            raise ValueError("Root-height frames must be unique")
        self.frame_indices, self.root_height = frames, heights

    def update_constraints(self, data_dict, index_dict):
        data_dict["root_y_pos"].append(self.root_height)
        index_dict["root_y_pos"].append(self.frame_indices)

    def crop_move(self, start, end):
        keep = (self.frame_indices >= start) & (self.frame_indices < end)
        return type(self)(self.frame_indices[keep] - start, self.root_height[keep])


def root_height_condition_hook(*, actor, generated_offset, history_length, device, **_):
    """Runtime hook for validated explicit ``root_targets[].root_height``."""
    import torch
    selected = [target for target in actor.get("root_targets", [])
                if "root_height" in target
                and generated_offset <= target["frame"] < generated_offset + 40]
    if not selected:
        return []
    frames = torch.tensor([target["frame"] - generated_offset + history_length
                           for target in selected], device=device)
    return [RootHeightConstraint(frames, [target["root_height"] for target in selected])]


def shift_vertical_coordinate_frame(motion, motion_rep, offset_y, *, is_normalized=True):
    """Rigidly translate a whole native feature window by one constant Y.

    This changes the coordinate origin, never relative poses, rotations,
    velocities or contacts. Call with -floor before generation and +floor on
    the returned window. The same origin must be used for every constraint.
    """
    import torch
    raw = motion_rep.unnormalize(motion) if is_normalized else motion.clone()
    raw = raw.clone()
    offset = torch.as_tensor(offset_y, dtype=raw.dtype, device=raw.device)
    if not torch.isfinite(offset).all() or offset.ndim > 1:
        raise ValueError('Coordinate-frame offset must be a finite scalar or batch vector')
    if offset.ndim == 1:
        if raw.ndim != 3 or len(offset) != raw.shape[0]:
            raise ValueError('Coordinate-frame batch offsets must match motion batch size')
        offset = offset[:, None]
    root = raw[..., motion_rep.slice_dict['root_pos']]
    root[..., 1] += offset
    joints = raw[..., motion_rep.slice_dict['local_joints_positions']]
    joints[..., 1::3] += offset[..., None] if offset.ndim else offset
    return motion_rep.normalize(raw) if is_normalized else raw


def translate_native_y(rep, normalized_motion, offset):
    """Batch-aware rigid Y frame change for normalized native ARDY features."""
    return shift_vertical_coordinate_frame(normalized_motion, rep, offset)
