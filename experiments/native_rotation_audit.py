#!/usr/bin/env python3
"""Read-only CPU audit of denormalized InterGen 262-channel source motion.

This is a consistency diagnostic, not a mesh/skin/contact validation. It never
loads a motion model or licensed SMPL asset. Native rotations do not contain a
root rotation. Root orientation is estimated from the three pelvis children;
one constant basis correction per actor is fitted on alternate frames, with
fixed offsets and FK evaluated on the held-out frames and the complete clip.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
WRISTS = [20, 21]


def summary(values):
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "p95": float(np.quantile(x, .95)), "max": float(x.max())}


def unit(v):
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(length < 1e-7):
        raise ValueError("Degenerate rotation column or pelvis landmark frame")
    return v / length


def decode_native_6d(values):
    """InterGen packs the first TWO COLUMNS interleaved, not two rows."""
    a, b = values[..., [0, 2, 4]], values[..., [1, 3, 5]]
    x = unit(a)
    y = unit(b - (x * b).sum(axis=-1, keepdims=True) * x)
    return np.stack((x, y, np.cross(x, y)), axis=-1)


def pelvis_frames(positions):
    # SMPL left hip is +X, spine is +Y. Columns map that basis to world.
    x = unit(positions[:, 1] - positions[:, 2])
    y = positions[:, 3] - positions[:, 0]
    y = unit(y - (x * y).sum(axis=-1, keepdims=True) * x)
    return np.stack((x, y, np.cross(x, y)), axis=-1)


def globals_from_local(local, roots):
    out = np.empty((len(local), 22, 3, 3), dtype=np.float64)
    out[:, 0] = roots
    for j in range(1, 22):
        out[:, j] = out[:, PARENTS[j]] @ local[:, j - 1]
    return out


def local_offsets(positions, global_rotations):
    bones = positions[:, 1:] - positions[:, PARENTS[1:]]
    return np.einsum("tjik,tjk->tji", global_rotations[:, PARENTS[1:]].swapaxes(-1, -2), bones)


def fixed_fk(positions, global_rotations, offsets):
    out = np.empty_like(positions)
    out[:, 0] = positions[:, 0]
    for j in range(1, 22):
        out[:, j] = out[:, PARENTS[j]] + np.einsum("tij,j->ti", global_rotations[:, PARENTS[j]], offsets[j - 1])
    return out


def audit_actor(positions, local):
    roots = pelvis_frames(positions)
    train = np.arange(0, len(positions), 2)
    held = np.arange(1, len(positions), 2)

    def residual(rotvec):
        g = globals_from_local(local[train], roots[train] @ Rotation.from_rotvec(rotvec).as_matrix())
        off = local_offsets(positions[train], g)
        # Fixed rest skeleton. No per-frame scales or individual joint correction.
        return (off - off.mean(axis=0)).ravel()

    starts = (np.zeros(3), np.array([np.pi/2, 0., 0.]), np.array([-np.pi/2, 0., 0.]), np.array([0., np.pi, 0.]))
    fits = [least_squares(residual, start, max_nfev=100, ftol=1e-9, xtol=1e-9, gtol=1e-9) for start in starts]
    fit = min(fits, key=lambda item: np.dot(item.fun, item.fun))
    g = globals_from_local(local, roots @ Rotation.from_rotvec(fit.x).as_matrix())
    offsets_per_frame = local_offsets(positions, g)
    offsets = offsets_per_frame[train].mean(axis=0)
    reconstructed = fixed_fk(positions, g, offsets)
    errors = np.linalg.norm(reconstructed - positions, axis=-1)
    offset_errors = np.linalg.norm(offsets_per_frame - offsets, axis=-1)
    lengths = np.linalg.norm(positions[:, 1:] - positions[:, PARENTS[1:]], axis=-1)
    lengths_median = np.median(lengths[train], axis=0)
    # This threshold is an explicit experimental acceptance proposal, not a
    # claim that source kinematics or any particular skin is anatomically valid.
    accepted = bool(fit.success and np.quantile(errors[held, 1:], .95) <= .05 and
                    np.quantile(errors[held][:, WRISTS], .95) <= .05 and
                    errors[:, 1:].max() <= .10)
    report = {
        "root_method": "orthonormal hips/spine frame plus fitted constant basis correction",
        "calibration_frames": train.tolist(), "held_out_frames": held.tolist(),
        "basis_correction_rotvec": Rotation.from_rotvec(fit.x).as_rotvec().tolist(), "fit_success": bool(fit.success),
        "fixed_rest_offsets_m": offsets.tolist(),
        "rest_offset_residual_m": summary(offset_errors),
        "rest_offset_residual_by_joint_m": [summary(offset_errors[:, j]) for j in range(21)],
        "source_bone_length_relative_deviation": summary(np.abs(lengths - lengths_median) / lengths_median),
        "fk_joint_error_m": summary(errors[:, 1:]),
        "held_out_fk_joint_error_m": summary(errors[held, 1:]),
        "fk_wrist_error_m": summary(errors[:, WRISTS]),
        "held_out_fk_wrist_error_m": summary(errors[held][:, WRISTS]),
        "eligible_for_native_rotation_skin_experiment": accepted,
    }
    return report, reconstructed


def audit(path):
    with np.load(path, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float64)
        if features.ndim != 3 or features.shape[1:] != (2, 262) or not 4 <= len(features) <= 1000:
            raise ValueError("Expected bounded denormalized features (4..1000, 2, 262)")
        if not np.isfinite(features).all():
            raise ValueError("Features must be finite")
        positions = features[..., :66].reshape(-1, 2, 22, 3)
        if "joints" in data and (data["joints"].shape != positions.shape or not np.allclose(data["joints"], positions, atol=1e-6)):
            raise ValueError("Stored raw joints disagree with feature positions")
        metadata = json.loads(str(data["metadata"].item())) if "metadata" in data else {}
    six = features[..., 132:258].reshape(-1, 2, 21, 6)
    local = decode_native_6d(six)
    actors, fk = [], []
    for actor in range(2):
        report, reconstructed = audit_actor(positions[:, actor], local[:, actor])
        actors.append(report)
        fk.append(reconstructed)
    fk = np.stack(fk, axis=1)
    source_gaps = np.linalg.norm(positions[:, 0][:, WRISTS, None] - positions[:, 1][:, None, WRISTS], axis=-1)
    fitted_gaps = np.linalg.norm(fk[:, 0][:, WRISTS, None] - fk[:, 1][:, None, WRISTS], axis=-1)
    return {
        "source_path": str(path.resolve()), "frames": len(features), "source_metadata": metadata,
        "feature_layout": {"positions": [0, 66], "per_frame_displacements": [66, 132], "local_rotation6d_excluding_root": [132, 258], "foot_contact": [258, 262]},
        "rotation_convention": "interleaved columns [0,2,4] and [1,3,5], Gram-Schmidt, column-vector FK",
        "sixd_input_column_dot_absolute": summary(np.abs((six[..., [0, 2, 4]] * six[..., [1, 3, 5]]).sum(axis=-1))),
        "actors": actors,
        "all_four_wrist_gap_change_m": summary(np.abs(fitted_gaps - source_gaps)),
        "acceptance_proposal": "Held-out joint and wrist p95 <=0.05m, all-frame joint max <=0.10m, both actors. Mesh anatomy/contact still require visual checks.",
        "native_axial_twist_recovery_candidate": all(a["eligible_for_native_rotation_skin_experiment"] for a in actors),
        "limitations": ["Root orientation and true SMPL shape/rest offsets absent; root is estimated from landmarks.", "Each actor gets one constant rest-basis correction and one fixed skeleton inferred from even frames.", "Position and rotation channels are separately generated and can disagree.", "No mesh, palm orientation, contact physics, anatomical joint limits or licensed SMPL model is validated.", "Failure of this inferred-root audit does not prove native rotations unusable with the original exact rest skeleton."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.source)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(args.output)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
