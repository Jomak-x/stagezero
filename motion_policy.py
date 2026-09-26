"""CPU-only, conservative candidate policy for 104-frame ARDY G1 generations.

``validate_generation_options`` accepts a JSON-like options object and returns
resolved model settings. ``select_candidate`` takes a prompt and a sequence of
``{"positions": (T,34,3), "rotations": (T,34,3,3)}`` mappings. It never edits
the arrays. Its action checks are geometric proxies, not proof of semantic
prompt adherence. When every candidate fails the quality guard, the chosen
index is ``None`` so the caller can reject generation or retain prior motion.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np

from motion_quality import HANDS, ROOT, SHOULDERS, analyze_motion


PROFILES = {
    "legacy": {"history_frames": 52, "carry_frames": 52, "cfg_weight": (2.0, 2.0)},
    "responsive": {"history_frames": 4, "carry_frames": 4, "cfg_weight": (2.0, 2.0)},
    "expressive": {"history_frames": 12, "carry_frames": 12, "cfg_weight": (4.0, 2.0)},
}


def validate_generation_options(options: Mapping | None) -> dict:
    """Validate ``profile``, ``seed``, ``candidates``, and ``pose_goal``.

    Defaults resolve to the existing backend behavior: legacy, no fixed seed,
    and one candidate. Booleans, floats, non-finite values, and unknown keys are
    rejected. Seed is a Python integer in ``[0, 2**32 - 1]``.
    """
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise ValueError("generation_options must be an object")
    unknown = set(options) - {"profile", "seed", "candidates", "pose_goal"}
    if unknown:
        raise ValueError(f"Unknown generation_options fields: {', '.join(sorted(map(str, unknown)))}")
    profile = options.get("profile", "legacy")
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError(f"profile must be one of {', '.join(PROFILES)}")
    seed = options.get("seed")
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**32 - 1):
        raise ValueError("seed must be an integer from 0 through 2**32 - 1")
    candidates = options.get("candidates", 1)
    if type(candidates) is not int or not 1 <= candidates <= 3:
        raise ValueError("candidates must be an integer from 1 through 3")
    pose_goal = options.get("pose_goal")
    if "pose_goal" in options and type(pose_goal) is not bool:
        raise ValueError("pose_goal must be a boolean")
    return {"profile": profile, **PROFILES[profile], "seed": seed, "candidates": candidates,
            "pose_goal": pose_goal}


def recognize_action(prompt: str) -> str | None:
    """Recognize only explicit supported positive instructions.

    Returns ``None`` on any negation cue, ambiguity, or generic prompt. This is
    intentionally narrow: a false positive could rank motion by the wrong cue.
    """
    if not isinstance(prompt, str):
        raise ValueError("prompt must be text")
    normalized = prompt.lower().replace("\u2019", "'").replace("\u2018", "'")
    words = re.findall(r"[a-z]+", normalized)
    if not words:
        return None
    if any(word in {"no", "not", "never", "without", "avoid", "dont", "cannot", "cant", "won't"}
           for word in words) or re.search(r"\b(?:don't|can't|won't)\b", normalized):
        return None
    text = " ".join(words)
    has_both_arms = "both" in words and any(word in words for word in ("arm", "arms", "hand", "hands"))
    overhead = "overhead" in words or re.search(r"\babove (?:the |their |his |her )?head\b", text)
    matches = []
    if has_both_arms and overhead and any(word in words for word in ("raise", "raises", "raising", "lift", "lifts", "lifting")):
        matches.append("overhead")
    if ("right" in words and any(word in words for word in ("hand", "arm"))
            and any(word in words for word in ("wave", "waves", "waving"))):
        matches.append("wave")
    if any(word in words for word in ("squat", "squats", "squatting")):
        matches.append("squat")
    has_stop = any(word in words for word in ("stop", "stops", "stopping"))
    has_still = "still" in words and any(word in words for word in ("stand", "stands", "standing"))
    if has_stop and has_still:
        matches.append("stop")
    return matches[0] if len(matches) == 1 else None


def _longest_run(flags: np.ndarray) -> int:
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def _proxy(action: str | None, positions: np.ndarray, prior_positions: np.ndarray | None, fps: float) -> dict:
    if action is None:
        return {"met": None, "score": 0.0, "description": "No supported action recognized"}
    p = np.asarray(positions, dtype=np.float64)
    if action in ("overhead", "wave"):
        margin = p[:, HANDS, 1] - p[:, SHOULDERS, 1]
        flags = (margin > .15).all(axis=1) if action == "overhead" else margin[:, 1] > .15
        run = _longest_run(flags)
        return {"met": run >= 5, "score": float(run), "longest_continuous_frames": run,
                "required_continuous_frames": 5,
                "description": "Hand height above same-side shoulder by more than 0.15 m"}
    if action == "squat":
        if prior_positions is None or len(prior_positions) < 4:
            return {"met": None, "score": 0.0, "description": "Four prior frames required for pelvis-height baseline"}
        baseline = float(np.median(prior_positions[-4:, ROOT, 1]))
        drop = baseline - float(p[:, ROOT, 1].min())
        return {"met": bool(drop >= .15), "score": float(drop), "drop_m": float(drop),
                "required_drop_m": .15,
                "description": "Minimum pelvis height relative to median of previous four frames"}
    if action == "stop":
        if len(p) < 25:
            return {"met": None, "score": 0.0, "description": "At least 25 frames required for stop check"}
        planar = p[-25:, ROOT][:, (0, 2)]
        speed = np.linalg.norm(np.diff(planar, axis=0), axis=1) * fps
        mean = float(speed.mean())
        return {"met": bool(mean <= .15), "score": -mean, "mean_final_25_root_speed_mps": mean,
                "maximum_mean_speed_mps": .15,
                "description": "Final 25-frame mean planar pelvis speed"}
    raise AssertionError(f"Unrecognized action {action}")


def _valid_prior(prior_positions, prior_rotations):
    if prior_positions is None:
        if prior_rotations is not None:
            raise ValueError("prior_rotations requires prior_positions")
        return None, None
    p = np.asarray(prior_positions)
    if p.ndim != 3 or p.shape[1:] != (34, 3) or len(p) == 0 or p.dtype.kind not in "fi" or not np.isfinite(p).all():
        raise ValueError("prior_positions must be finite (frames, 34, 3) numeric values")
    r = None if prior_rotations is None else np.asarray(prior_rotations)
    if r is not None and (r.shape != (len(p), 34, 3, 3) or r.dtype.kind not in "fi" or not np.isfinite(r).all()):
        raise ValueError("prior_rotations must be finite (frames, 34, 3, 3) numeric values")
    return p, r


def select_candidate(
    prompt: str,
    candidates: Sequence[Mapping],
    *,
    prior_positions: np.ndarray | None = None,
    prior_rotations: np.ndarray | None = None,
    fps: float = 25.0,
) -> dict:
    """Assess candidates and choose a safe index without changing any motion.

    Safety requires finite poses, valid rotation matrices, at most 0.05 m toe
    floor penetration, and mean joint displacement at the prior/generated seam
    at most 0.15 m. If a 104-frame two-horizon clip is supplied, its 52-frame
    internal seam must also satisfy that limit. Missing prior motion means the
    external seam is inapplicable. Among safe candidates, rank a recognized
    action's proxy success first, then proxy score, then lower worst seam.
    """
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or not 1 <= len(candidates) <= 3:
        raise ValueError("candidates must be a sequence of one through three pose mappings")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    prior_p, prior_r = _valid_prior(prior_positions, prior_rotations)
    action = recognize_action(prompt)
    assessments = []
    for index, candidate in enumerate(candidates):
        row = {"index": index, "safe": False, "proxy": None, "score": None, "seam_mean_joint_m": None,
               "floor_penetration_max_depth_m": None, "quality": None, "reasons": []}
        try:
            if not isinstance(candidate, Mapping) or set(candidate) != {"positions", "rotations"}:
                raise ValueError("candidate must contain positions and rotations")
            positions = np.asarray(candidate["positions"])
            rotations = np.asarray(candidate["rotations"])
            quality = analyze_motion(positions, rotations, fps=fps,
                                     prior_positions=prior_p, prior_rotations=prior_r)
            row["quality"] = quality
            if not quality["positions_finite"]:
                row["reasons"].append("nonfinite_positions")
            if not quality["rotations_valid"]:
                row["reasons"].append("invalid_rotations")
            if quality["positions_finite"]:
                floor = quality["floor_penetration_max_depth_m"]
                row["floor_penetration_max_depth_m"] = floor
                if floor > .05:
                    row["reasons"].append("floor_penetration")
                frame_steps = np.linalg.norm(np.diff(positions, axis=0), axis=2).mean(axis=1)
                row["max_frame_mean_joint_step_m"] = float(frame_steps.max()) if len(frame_steps) else 0.0
                if row["max_frame_mean_joint_step_m"] > .15:
                    row["reasons"].append("intra_clip_jump")
                seams = []
                if prior_p is not None:
                    seam = quality["boundary_mean_joint_position_jump_m"]
                    seams.append(seam)
                    if seam > .15:
                        row["reasons"].append("history_seam")
                if len(positions) == 104:
                    seam = float(np.linalg.norm(positions[52] - positions[51], axis=1).mean())
                    seams.append(seam)
                    if seam > .15:
                        row["reasons"].append("horizon_seam")
                row["seam_mean_joint_m"] = max(seams) if seams else None
                row["proxy"] = _proxy(action, positions, prior_p, float(fps))
                row["score"] = row["proxy"]["score"]
            row["safe"] = not row["reasons"]
        except (KeyError, TypeError, ValueError) as exc:
            row["reasons"].append(f"invalid_candidate: {exc}")
        assessments.append(row)
    safe = [row for row in assessments if row["safe"]]
    if safe:
        chosen = max(safe, key=lambda row: (
            row["proxy"]["met"] is True,
            row["score"],
            -(row["seam_mean_joint_m"] if row["seam_mean_joint_m"] is not None else 0.0),
            -row["index"],
        ))["index"]
    else:
        chosen = None
    return {"chosen_index": chosen, "action": action, "assessments": assessments,
            "proxy_caveat": "Geometric checks only; visual review is required for semantic motion quality."}
