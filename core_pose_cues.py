"""Trusted, bounded native pose cues for one staged two-person rehearsal.

Reference poses condition fresh ARDY output; this module never substitutes them
for generated frames. It has no runtime-directory, network, Torch or experiment
imports. The pinned asset contains four native frames per cue with provenance.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from realtime_clip import CanonicalClip

RECIPE = "pose_duet_v1"
ASSET = Path(__file__).parent / "assets/core-motion/duet-cues.npz"
ASSET_SHA256 = "997ae56122d16dc6572be1ddae6220aed320247bf62da1a0f5caa7ac05b6a670"
MIN_INITIAL_SEPARATION_M = 2.25
CUE_SCHEDULE = ((None, None), ("kick", "duck"), ("duck", "kick"), ("victory", "victory"))
POSE_DUET_BEATS = (
    ("Opening", "A martial artist performs sharp punches and a front kick.", "A boxer dodges punches, ducking and weaving."),
    ("Lead kick and dodge", "A martial artist performs a high front kick.", "A boxer ducks deeply under a high kick."),
    ("Reply kick and dodge", "A boxer ducks deeply under a high kick.", "A martial artist performs a high front kick."),
    ("Shared victory", "A person raises both arms high overhead in victory.", "A person raises both arms high overhead in victory."),
)


def recipe_roles(beats, actor_ids):
    """Accept the pinned cue/prompt schedule or an exact global role swap."""
    if len(beats) != 4 or any(b["seconds"] != 2 or set(b) != {"name", "seconds", "actor_prompts"} for b in beats):
        raise ValueError("pose_duet_v1 requires its four two-second beats without spatial overrides")
    pairs = [tuple(b["actor_prompts"][a] for a in actor_ids) for b in beats]
    expected = [(row[1], row[2]) for row in POSE_DUET_BEATS]
    if pairs == expected:
        return False
    if pairs == [pair[::-1] for pair in expected]:
        return True
    raise ValueError("pose_duet_v1 uses fixed reference-matched prompts; choose free choreography to edit actions")


def heading(positions):
    """Official Core heading: atan2((right-left).z, -(right-left).x)."""
    hips = np.asarray(positions)[19] - np.asarray(positions)[23]
    if np.linalg.norm(hips[[0, 2]]) < 1e-6:
        raise ValueError("Core hip positions have no reliable planar heading")
    return math.atan2(float(hips[2]), -float(hips[0]))


def _asset_payload(content):
    if hashlib.sha256(content).hexdigest() != ASSET_SHA256:
        raise ValueError("Trusted Core duet cue asset failed its SHA256 check")
    with np.load(io.BytesIO(content), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata.get("version") != 1 or metadata.get("skeleton") != "Core27" or metadata.get("fps") != 20:
            raise ValueError("Unsupported Core duet asset format")
        clips = {cue: CanonicalClip(data[cue + "_positions"][None], data[cue + "_rotations"][None],
                    20, (cue,), "ardy_core_reference", {}, data[cue + "_native_features"][None])
                 for cue in ("kick", "duck", "victory")}
    if any(clip.frames != 4 for clip in clips.values()):
        raise ValueError("Each trusted duet cue must contain four native frames")
    return clips, metadata


@lru_cache(maxsize=1)
def _library():
    return _asset_payload(ASSET.read_bytes())


def profile_metadata(window_index, *, swapped=False):
    if type(window_index) is not int or not 0 <= window_index < 4 or type(swapped) is not bool:
        raise ValueError("Invalid pose duet window or role assignment")
    cues = CUE_SCHEDULE[window_index]
    if swapped:
        cues = cues[::-1]
    # Load before any queue replacement, so a missing/corrupt asset fails early.
    _, provenance = _library()
    return {"recipe": RECIPE, "window_index": window_index, "roles_swapped": swapped,
            "cue_ids": list(cues), "conditioning_history_frames": 4,
            "asset_sha256": ASSET_SHA256, "asset_provenance": json.loads(json.dumps(provenance)),
            "conditioned_frames": [] if window_index == 0 else [36, 37, 38, 39],
            "alignment": "kick foot toward partner; duck hips toward partner; victory hips toward shared perpendicular audience",
            "root_height_preserved": True, "model_outcome_verified": False}


def _validate_profile(profile):
    if not isinstance(profile, dict):
        raise ValueError("Invalid pose cue profile")
    expected = profile_metadata(profile.get("window_index"), swapped=profile.get("roles_swapped"))
    if profile != expected:
        raise ValueError("Pose cue profile must match the trusted recipe and asset")
    return profile["window_index"], profile["cue_ids"]


def apply_profile(body, history_clip, profile):
    """Adapt one job explicitly; saved history and all committed poses stay intact."""
    index, cues = _validate_profile(profile)
    if len(body.get("actor_ids", [])) != 2 or body.get("frames") != 40:
        raise ValueError("Pose duet requires two actors and one native 40-frame job")
    result = dict(body)
    if history_clip is not None:
        if (not isinstance(history_clip, CanonicalClip) or history_clip.native_features is None
                or history_clip.actor_ids != tuple(body["actor_ids"]) or history_clip.frames < 4):
            raise ValueError("Pose duet requires matching native Core history")
        result["history"] = {"native_features": history_clip.native_features[:, -4:].tolist()}
    if index == 0:
        return result
    if history_clip is None:
        raise ValueError("A pose-cued transition requires committed native history")
    target, _ = cue_targets(history_clip, cues)
    result["stage_kind"] = "transition"
    result["target"] = target
    # The full-body target already constrains roots. The existing worker's mixed
    # root/full-body constructor rejects mixed devices; do not combine them.
    result.pop("root_targets", None)
    return result


def cue_targets(history_clip, cues):
    """Build per-role targets at actual current roots; report requested geometry."""
    if (not isinstance(history_clip, CanonicalClip) or history_clip.native_features is None
            or len(history_clip.actor_ids) != 2 or not isinstance(cues, (tuple, list))
            or len(cues) != 2 or any(c not in ("kick", "duck", "victory") for c in cues)):
        raise ValueError("Expected two native actors and two trusted cue IDs")
    clips, _ = _library()
    origins = history_clip.positions[:, -1, 0][:, [0, 2]].astype(np.float64)
    delta_pair = origins[1] - origins[0]
    if np.linalg.norm(delta_pair) < .65:
        raise ValueError("Current duet roots violate the separation proxy")
    # Stable actor order chooses the same side for both actors; a left-to-right
    # pair faces +Z for the finish. This rotates with the actual pair line.
    audience = math.atan2(-float(delta_pair[1]), float(delta_pair[0]))
    ps, rs, transforms = [], [], []
    for i, cue in enumerate(cues):
        clip = clips[cue]
        p = clip.positions[0].astype(np.float64)
        r = clip.rotations[0].astype(np.float64)
        partner = origins[1-i] - origins[i]
        partner_yaw = math.atan2(float(partner[0]), float(partner[1]))
        if cue == "kick":
            direction = p[-1, 21] - p[-1, 0]  # Raised right foot in the pinned cue.
            if np.linalg.norm(direction[[0, 2]]) < .1:
                raise ValueError("Kick cue has no reliable planar foot direction")
            reference_yaw = math.atan2(float(direction[0]), float(direction[2]))
            target_yaw = partner_yaw
        else:
            reference_yaw = heading(p[-1])
            target_yaw = audience if cue == "victory" else partner_yaw
        yaw = math.atan2(math.sin(target_yaw-reference_yaw), math.cos(target_yaw-reference_yaw))
        c, s = math.cos(yaw), math.sin(yaw)
        rotate = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
        offset = np.zeros(3)
        offset[[0, 2]] = origins[i] - (rotate @ p[0, 0])[[0, 2]]
        posed = p @ rotate.T + offset
        rotated = rotate @ r
        if np.max(np.abs(posed[:, 0][:, [0, 2]])) > 25:
            raise ValueError("Pose cue root target leaves the ±25 m scene bounds")
        ps.append(np.concatenate((np.repeat(posed[:1], 36, axis=0), posed)))
        rs.append(np.concatenate((np.repeat(rotated[:1], 36, axis=0), rotated)))
        transforms.append({"actor_id": history_clip.actor_ids[i], "cue": cue,
            "yaw_delta_radians": yaw, "translation_xyz_m": offset.tolist(),
            "requested_facing_radians": target_yaw, "facing_measure": "root_to_raised_right_foot" if cue == "kick" else "official_hip_heading"})
    return ({"positions": np.asarray(ps, np.float32).tolist(), "rotations": np.asarray(rs, np.float32).tolist()},
            {"asset_sha256": ASSET_SHA256, "transforms": transforms,
             "conditioned_frames": [36, 37, 38, 39], "model_outcome_verified": False})


def measure_cue_result(body, clip):
    """Measure actual output against requested final-four goals; never a pass gate."""
    if body.get("stage_kind") != "transition" or "target" not in body:
        return None
    if (not isinstance(clip, CanonicalClip) or clip.frames != 40
            or clip.actor_ids != tuple(body.get("actor_ids", ()))):
        raise ValueError("Cue measurements require the matching generated 40-frame clip")
    target = np.asarray(body["target"]["positions"], dtype=np.float64)
    if target.shape != clip.positions.shape or not np.isfinite(target).all():
        raise ValueError("Invalid cue measurement targets")
    per_actor = []
    for i, actor_id in enumerate(clip.actor_ids):
        actual, goal = clip.positions[i, -4:], target[i, -4:]
        errors = np.linalg.norm(actual - goal, axis=-1)
        angles = []
        for p, q in zip(actual, goal):
            delta = heading(p) - heading(q)
            angles.append(abs(math.degrees(math.atan2(math.sin(delta), math.cos(delta)))))
        per_actor.append({"actor_id": actor_id, "mean_joint_error_m": float(errors.mean()),
            "max_joint_error_m": float(errors.max()), "root_error_max_m": float(errors[:, 0].max()),
            "right_foot_error_max_m": float(errors[:, 21].max()),
            "hands_error_max_m": float(errors[:, [10, 16]].max()),
            "hip_heading_error_max_degrees": max(angles)})
    return {"conditioned_frames": [36, 37, 38, 39], "per_actor": per_actor,
            "quality_gate_applied": False, "physical_contact_verified": False}
