"""CPU-only geometric measurements for interacting ARDY characters and scene objects.

Positions are world-space metres in ``(frames, joints, xyz)`` with Y up. The
joint orders below are copied by name from ``CoreSkeleton27`` and
``G1Skeleton34`` in ``vendor/ardy/ardy/skeleton/definitions.py``. These
measurements are useful to reject obvious misses; proximity, overlap, and low
toe motion do not prove contact forces, collision-free meshes, or grasping.
"""

from __future__ import annotations

import numpy as np

from motion_quality import G1_JOINT_NAMES


CORE27_JOINT_NAMES = (
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandEnd",
    "RightHandThumb1", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandEnd", "LeftHandThumb1", "RightUpLeg", "RightLeg", "RightFoot",
    "RightToeBase", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
)

_LAYOUTS = {
    "core27": {
        "names": CORE27_JOINT_NAMES,
        "root": "Hips", "head": "Head",
        "hands": {"left": "LeftHandEnd", "right": "RightHandEnd"},
        "feet": ("LeftToeBase", "RightToeBase"),
        "body": ("Hips", "Spine", "Spine2", "Spine3", "Neck", "Head",
                 "LeftShoulder", "RightShoulder", "LeftUpLeg", "RightUpLeg"),
    },
    "g1": {
        "names": G1_JOINT_NAMES,
        "root": "pelvis_skel", "head": "waist_pitch_skel",
        "hands": {"left": "left_hand_roll_skel", "right": "right_hand_roll_skel"},
        "feet": ("left_toe_base", "right_toe_base"),
        "body": ("pelvis_skel", "waist_yaw_skel", "waist_pitch_skel",
                 "left_shoulder_pitch_skel", "right_shoulder_pitch_skel",
                 "left_hip_pitch_skel", "right_hip_pitch_skel"),
    },
}


def _layout(skeleton: str) -> dict:
    if skeleton not in _LAYOUTS:
        raise ValueError("skeleton must be 'core27' or 'g1'")
    spec = _LAYOUTS[skeleton]
    return {**spec, "index": {name: i for i, name in enumerate(spec["names"])}}


def joint_index(skeleton: str, name: str) -> int:
    """Look up a named joint in the official skeleton order."""
    try:
        return _layout(skeleton)["index"][name]
    except KeyError as exc:
        raise ValueError(f"unknown {skeleton} joint: {name}") from exc


def _positions(value: np.ndarray, skeleton: str, label: str = "positions") -> tuple[np.ndarray, dict]:
    spec = _layout(skeleton)
    p = np.asarray(value)
    shape = (len(spec["names"]), 3)
    if p.ndim != 3 or p.shape[1:] != shape or len(p) == 0:
        raise ValueError(f"{label} must have shape (frames, {shape[0]}, 3), with frames >= 1")
    if p.dtype.kind not in "fi" or not np.isfinite(p).all():
        raise ValueError(f"{label} must contain finite numeric positions")
    return p.astype(np.float64, copy=False), spec


def _vector(value, size: int, label: str) -> np.ndarray:
    v = np.asarray(value, dtype=np.float64)
    if v.shape != (size,) or not np.isfinite(v).all():
        raise ValueError(f"{label} must be a finite length-{size} vector")
    return v


def _nonnegative(value: float, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f"{label} must be finite")
    x = float(value)
    if (x <= 0) if positive else (x < 0):
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'}")
    return x


def _longest_true(mask: np.ndarray) -> int:
    current = longest = 0
    for hit in mask:
        current = current + 1 if hit else 0
        longest = max(longest, current)
    return longest


def gate_traversal(
    positions: np.ndarray,
    *,
    skeleton: str,
    center: tuple[float, float, float],
    normal_xz: tuple[float, float],
    opening_width_m: float,
    body_radius_m: float = 0.15,
    min_y: float = -np.inf,
    max_y: float = np.inf,
    direction: str = "forward",
) -> dict:
    """Check actual root plane crossing through a rotated gate opening.

    ``center`` is the middle of the gate opening; ``normal_xz`` points from
    entry to exit. ``body_radius_m`` adds a configurable margin around the
    named torso/hip/head joints. G1 lacks a head joint, so its torso top is
    ``waist_pitch_skel``; a gate height result for G1 cannot certify head
    clearance. A path that merely ends behind the gate does not count unless
    its sampled root path crosses the gate plane within the aperture.
    """
    p, spec = _positions(positions, skeleton)
    c = _vector(center, 3, "center")
    n = _vector(normal_xz, 2, "normal_xz")
    mag = float(np.linalg.norm(n))
    if mag < 1e-12:
        raise ValueError("normal_xz must be nonzero")
    n /= mag
    tangent = np.array((-n[1], n[0]))
    width = _nonnegative(opening_width_m, "opening_width_m", positive=True)
    radius = _nonnegative(body_radius_m, "body_radius_m")
    if not np.isfinite(min_y) and min_y != -np.inf:
        raise ValueError("min_y must be finite or -infinity")
    if not np.isfinite(max_y) and max_y != np.inf:
        raise ValueError("max_y must be finite or infinity")
    if min_y >= max_y:
        raise ValueError("min_y must be below max_y")
    if direction not in {"forward", "either"}:
        raise ValueError("direction must be 'forward' or 'either'")
    root = p[:, spec["index"][spec["root"]], :]
    signed = (root[:, (0, 2)] - c[[0, 2]]) @ n
    body_idx = [spec["index"][name] for name in spec["body"]]
    crossings = []
    for i in range(len(p) - 1):
        before, after = float(signed[i]), float(signed[i + 1])
        if before == 0 or before * after > 0:
            continue
        # A clip that only touches the plane and stops/turns back has not
        # traversed it. Skip any exact-zero plateau and require the far side.
        next_nonzero = i + 1
        while next_nonzero < len(p) and signed[next_nonzero] == 0:
            next_nonzero += 1
        if next_nonzero == len(p) or before * signed[next_nonzero] >= 0:
            continue
        if direction == "forward" and before >= 0:
            continue
        alpha = -before / (after - before)
        sample = p[i, body_idx] * (1 - alpha) + p[i + 1, body_idx] * alpha
        lateral = (sample[:, (0, 2)] - c[[0, 2]]) @ tangent
        max_lateral = float(np.max(np.abs(lateral)) + radius)
        y_low = float(sample[:, 1].min() - radius)
        y_high = float(sample[:, 1].max() + radius)
        clears = max_lateral <= width / 2 and y_low >= min_y and y_high <= max_y
        crossings.append({"frame_before": i, "frame_after": i + 1,
                          "fraction_after_frame": float(alpha),
                          "max_body_lateral_m": max_lateral,
                          "body_low_y_m": y_low, "body_high_y_m": y_high,
                          "clearance_proxy": bool(clears)})
    return {"traversed_proxy": any(cross["clearance_proxy"] for cross in crossings),
            "plane_crossings": crossings, "crossing_count": len(crossings),
            "opening_width_m": width, "body_radius_m": radius,
            "skeleton": skeleton, "g1_head_clearance_unchecked": skeleton == "g1"}


def goal_endpoint(positions: np.ndarray, *, skeleton: str, target_xz,
                  tolerance_m: float = 0.15) -> dict:
    """Planar distance in metres from final root to a destination point."""
    p, spec = _positions(positions, skeleton)
    target = _vector(target_xz, 2, "target_xz")
    tolerance = _nonnegative(tolerance_m, "tolerance_m")
    final = p[-1, spec["index"][spec["root"]], (0, 2)]
    error = float(np.linalg.norm(final - target))
    return {"endpoint_error_xz_m": error, "within_tolerance": error <= tolerance,
            "tolerance_m": tolerance, "final_root_xz_m": final.tolist()}


def pair_separation(positions_a: np.ndarray, positions_b: np.ndarray, *,
                    skeleton_a: str, skeleton_b: str,
                    radius_a_m: float = 0.25, radius_b_m: float = 0.25) -> dict:
    """Synchronous planar root separation and conservative disc overlap proxy."""
    a, sa = _positions(positions_a, skeleton_a, "positions_a")
    b, sb = _positions(positions_b, skeleton_b, "positions_b")
    if len(a) != len(b):
        raise ValueError("pair clips must have the same number of frames")
    threshold = _nonnegative(radius_a_m, "radius_a_m") + _nonnegative(radius_b_m, "radius_b_m")
    ar = a[:, sa["index"][sa["root"]], :]
    br = b[:, sb["index"][sb["root"]], :]
    distances = np.linalg.norm(ar[:, (0, 2)] - br[:, (0, 2)], axis=1)
    overlap = distances < threshold
    return {"min_root_separation_xz_m": float(distances.min()),
            "mean_root_separation_xz_m": float(distances.mean()),
            "root_disc_overlap_proxy_frames": int(overlap.sum()),
            "root_disc_overlap_proxy_fraction": float(overlap.mean()),
            "collision_radius_sum_m": threshold}


def hand_contact(positions_a: np.ndarray, positions_b: np.ndarray, *,
                 skeleton_a: str, skeleton_b: str,
                 hand_a: str = "right", hand_b: str = "left",
                 tolerance_m: float = 0.12, fps: float = 25.0,
                 minimum_duration_s: float = 0.12) -> dict:
    """Same-frame hand proximity; duration must be contiguous for a contact cue."""
    a, sa = _positions(positions_a, skeleton_a, "positions_a")
    b, sb = _positions(positions_b, skeleton_b, "positions_b")
    if len(a) != len(b):
        raise ValueError("pair clips must have the same number of frames")
    try:
        ia = sa["index"][sa["hands"][hand_a]]
        ib = sb["index"][sb["hands"][hand_b]]
    except KeyError as exc:
        raise ValueError("hand_a and hand_b must be 'left' or 'right'") from exc
    tolerance = _nonnegative(tolerance_m, "tolerance_m")
    rate = _nonnegative(fps, "fps", positive=True)
    duration = _nonnegative(minimum_duration_s, "minimum_duration_s")
    distances = np.linalg.norm(a[:, ia] - b[:, ib], axis=1)
    hits = distances <= tolerance
    longest = _longest_true(hits)
    return {"min_hand_distance_m": float(distances.min()),
            "near_frames": int(hits.sum()), "longest_near_run_frames": longest,
            "longest_near_run_s": longest / rate,
            "contact_proxy": longest / rate >= duration and longest > 0,
            "tolerance_m": tolerance, "minimum_duration_s": duration}


def object_contact(positions: np.ndarray, *, skeleton: str, object_xyz,
                   hand: str = "right", tolerance_m: float = 0.12,
                   fps: float = 25.0, minimum_duration_s: float = 0.12) -> dict:
    """Hand proximity to a static point or a time-varying object target."""
    p, spec = _positions(positions, skeleton)
    try:
        hand_idx = spec["index"][spec["hands"][hand]]
    except KeyError as exc:
        raise ValueError("hand must be 'left' or 'right'") from exc
    target = np.asarray(object_xyz, dtype=np.float64)
    if target.shape == (3,):
        target = np.broadcast_to(target, (len(p), 3))
    if target.shape != (len(p), 3) or not np.isfinite(target).all():
        raise ValueError("object_xyz must be a finite point (3,) or trajectory (frames, 3)")
    tolerance = _nonnegative(tolerance_m, "tolerance_m")
    rate = _nonnegative(fps, "fps", positive=True)
    duration = _nonnegative(minimum_duration_s, "minimum_duration_s")
    distances = np.linalg.norm(p[:, hand_idx] - target, axis=1)
    hits = distances <= tolerance
    longest = _longest_true(hits)
    return {"min_hand_object_distance_m": float(distances.min()),
            "near_frames": int(hits.sum()), "longest_near_run_frames": longest,
            "longest_near_run_s": longest / rate,
            "contact_proxy": longest / rate >= duration and longest > 0,
            "tolerance_m": tolerance, "minimum_duration_s": duration}


def floor_motion(positions: np.ndarray, *, skeleton: str, fps: float = 25.0,
                 floor_y: float = 0.0, contact_height_m: float = 0.08,
                 max_contact_vertical_speed_mps: float = 0.15) -> dict:
    """Toe floor penetration and lateral sliding during low, slow vertical motion."""
    p, spec = _positions(positions, skeleton)
    rate = _nonnegative(fps, "fps", positive=True)
    if not np.isfinite(floor_y):
        raise ValueError("floor_y must be finite")
    height = _nonnegative(contact_height_m, "contact_height_m")
    vertical_limit = _nonnegative(max_contact_vertical_speed_mps, "max_contact_vertical_speed_mps")
    toes = p[:, [spec["index"][name] for name in spec["feet"]]]
    relative_height = toes[..., 1] - floor_y
    min_height = float(relative_height.min())
    if len(p) < 2:
        lateral = np.empty(0)
    else:
        delta = np.diff(toes, axis=0) * rate
        contact = ((relative_height[:-1] <= height) & (relative_height[1:] <= height)
                   & (np.abs(delta[..., 1]) <= vertical_limit))
        lateral = np.linalg.norm(delta[..., (0, 2)], axis=-1)[contact]
    return {"toe_min_height_above_floor_m": min_height,
            "toe_penetration_max_depth_m": max(0.0, -min_height),
            "toe_penetration_frames": int(np.count_nonzero((relative_height < 0).any(axis=1))),
            "low_toe_samples": int(len(lateral)),
            "low_toe_slide_mean_mps": float(lateral.mean()) if len(lateral) else None,
            "low_toe_slide_peak_mps": float(lateral.max()) if len(lateral) else None}


def continuity(positions: np.ndarray, *, skeleton: str, fps: float = 25.0,
               horizon_boundaries: tuple[int, ...] = ()) -> dict:
    """Per-frame displacement and velocity changes, including selected seams.

    Boundary ``k`` is the first frame of a new horizon, so seam movement is
    between ``k-1`` and ``k``. Large values flag review but are not proof of an
    artifact; movement speed and action changes can be legitimate.
    """
    p, spec = _positions(positions, skeleton)
    rate = _nonnegative(fps, "fps", positive=True)
    boundaries = tuple(horizon_boundaries)
    if any(not isinstance(k, (int, np.integer)) or isinstance(k, bool) or not 1 <= k < len(p)
           for k in boundaries) or len(set(boundaries)) != len(boundaries):
        raise ValueError("horizon_boundaries must be unique integer frame indices in [1, frames-1]")
    root = p[:, spec["index"][spec["root"]]]
    step = np.linalg.norm(np.diff(root, axis=0), axis=1)
    joint_step = np.linalg.norm(np.diff(p, axis=0), axis=-1).mean(axis=1)
    velocity = np.diff(root, axis=0) * rate
    acceleration = np.linalg.norm(np.diff(velocity, axis=0) * rate, axis=1)
    seams = []
    for k in boundaries:
        seams.append({"frame_after": int(k),
                      "root_step_m": float(step[k - 1]),
                      "mean_joint_step_m": float(joint_step[k - 1]),
                      "root_speed_mps": float(step[k - 1] * rate),
                      "velocity_jump_mps": (float(np.linalg.norm(velocity[k] - velocity[k - 1]))
                                            if k < len(p) - 1 else None)})
    return {"root_peak_step_m": float(step.max()) if len(step) else None,
            "root_peak_speed_mps": float(step.max() * rate) if len(step) else None,
            "mean_joint_peak_step_m": float(joint_step.max()) if len(joint_step) else None,
            "root_peak_acceleration_mps2": float(acceleration.max()) if len(acceleration) else None,
            "horizon_seams": seams}
