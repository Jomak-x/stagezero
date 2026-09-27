"""Offline, bounded acceptance measurements for saved Stagezero G1 takes.

These geometric checks are deliberately narrow. Passing them does not establish
that a movement looks natural, follows a direction, or matches its prompt in
every detail. They require no renderer, model, GPU, or network service.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

from motion_quality import JOINT_INDEX, ROOT, SHOULDERS
from story_recovery import NEGATION, is_recovery_motion, recovered_upright
from takes import Take, decode_project, validate_take


FPS = 25
_TRAVEL = re.compile(
    r"\b(?:runs?|running|sprints?|sprinting|walks?|walking|jogs?|jogging|"
    r"skips?|skipping|sidesteps?|sidestepping|marches?|marching|"
    r"crawls?|crawling|travels?|traveling|travelling)\b", re.I)
_STATIONARY = re.compile(r"\b(?:in place|on the spot|on a treadmill)\b", re.I)
_BACKFLIP = re.compile(r"\bback[ -]?flips?\b", re.I)
_FALL = re.compile(r"\b(?:falls?|falling|fell|collapses?|collapsing)\b", re.I)
_STAND = re.compile(r"\bstand(?:s|ing)?\s+(?:still|upright)\b", re.I)
_DISTANCE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)\b", re.I)

# These tolerances are conservative geometric gates, not trained classifiers.
MAX_SEAM_ROOT_JUMP_BODY_LENGTHS = 0.5
MAX_SEAM_MEAN_JOINT_JUMP_BODY_LENGTHS = 0.75
MIN_TRAVEL_NET_BODY_LENGTHS = 0.75
MIN_EXPLICIT_DISTANCE_FRACTION = 0.75
MIN_INVERTED_DOT_UP = -0.5
MIN_INVERTED_RUN_FRAMES = 3
MAX_FALL_TERMINAL_ELEVATION_BODY_LENGTHS = 0.65


def _check(name: str, passed: bool, observed, requirement: str) -> dict:
    return {"name": name, "passed": bool(passed), "observed": observed,
            "requirement": requirement}


def _body_length(positions: np.ndarray) -> float:
    """Median two-leg length makes thresholds independent of actor scale."""
    lengths = []
    for side in ("left", "right"):
        hip, knee, ankle = (positions[:, JOINT_INDEX[f"{side}_{part}_skel"]]
                            for part in ("hip_yaw", "knee", "ankle_roll"))
        lengths.append(np.linalg.norm(hip - knee, axis=1)
                       + np.linalg.norm(knee - ankle, axis=1))
    value = float(np.median(np.mean(lengths, axis=0)))
    if not np.isfinite(value) or value <= 1e-5:
        raise ValueError("Take has no measurable leg length")
    return value


def _longest_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _segment_metrics(positions: np.ndarray, body_length: float, fps: int) -> dict:
    root = positions[:, ROOT].astype(np.float64)
    root_steps = np.diff(root, axis=0)
    planar_steps = np.linalg.norm(root_steps[:, (0, 2)], axis=1)
    local = positions.astype(np.float64) - root[:, None, :]
    local_speed = np.linalg.norm(np.diff(local, axis=0), axis=-1) * fps / body_length
    frame_energy = local_speed.mean(axis=1) if len(local_speed) else np.array([])
    torso = positions[:, SHOULDERS].mean(axis=1) - root
    torso_norm = np.linalg.norm(torso, axis=1)
    dot_up = np.divide(torso[:, 1], torso_norm, out=np.ones(len(root)), where=torso_norm > 1e-5)
    inverted = dot_up <= MIN_INVERTED_DOT_UP
    ankle_height = np.mean([positions[:, JOINT_INDEX[f"{side}_ankle_roll_skel"], 1]
                            for side in ("left", "right")], axis=0)
    elevation = root[:, 1] - ankle_height
    return {
        "root_path_length_m": float(planar_steps.sum()),
        "root_net_displacement_m": float(np.linalg.norm(root[-1, (0, 2)] - root[0, (0, 2)])),
        "root_vertical_range_m": float(np.ptp(root[:, 1])),
        "root_mean_planar_speed_mps": float(planar_steps.mean() * fps) if len(planar_steps) else 0.0,
        "pose_mean_joint_speed_body_lengths_per_second": float(frame_energy.mean()) if len(frame_energy) else 0.0,
        "pose_p90_joint_speed_body_lengths_per_second": float(np.percentile(frame_energy, 90)) if len(frame_energy) else 0.0,
        "pose_active_frame_fraction": float(np.mean(frame_energy >= .3)) if len(frame_energy) else 0.0,
        "torso_min_dot_up": float(dot_up.min()),
        "torso_inverted_fraction": float(inverted.mean()),
        "torso_longest_inverted_run_frames": _longest_run(inverted),
        "root_elevation_above_ankles_terminal_m": float(np.median(elevation[-min(5, len(elevation)):])),
    }


def analyze_take(take: Take, *, fps: int = FPS) -> dict:
    """Measure one validated take and gate only supported observable claims."""
    if type(fps) is not int or fps <= 0:
        raise ValueError("fps must be a positive integer")
    validate_take(take)
    positions = take.positions
    body_length = _body_length(positions)
    coverage = (take.segments[0]["start"] == 0 and
                take.segments[-1]["end"] == len(positions) and
                all(a["end"] == b["start"] for a, b in zip(take.segments, take.segments[1:])))
    structural_checks = [_check("segment_coverage", coverage,
                                {"covered_frames": sum(s["end"] - s["start"] for s in take.segments),
                                 "take_frames": len(positions)},
                                "Contiguous segments cover every take frame")]
    segments = []
    for index, segment in enumerate(take.segments):
        start, end = segment["start"], segment["end"]
        prompt = segment["prompt"]
        metrics = _segment_metrics(positions[start:end], body_length, fps)
        checks = []
        if start:
            seam = positions[start].astype(np.float64) - positions[start - 1]
            root_jump = float(np.linalg.norm(seam[ROOT]) / body_length)
            mean_joint_jump = float(np.linalg.norm(seam, axis=1).mean() / body_length)
            checks.extend((
                _check("root_seam_continuity", root_jump <= MAX_SEAM_ROOT_JUMP_BODY_LENGTHS,
                       root_jump, f"Root jump <= {MAX_SEAM_ROOT_JUMP_BODY_LENGTHS} body lengths"),
                _check("joint_seam_continuity", mean_joint_jump <= MAX_SEAM_MEAN_JOINT_JUMP_BODY_LENGTHS,
                       mean_joint_jump,
                       f"Mean joint jump <= {MAX_SEAM_MEAN_JOINT_JUMP_BODY_LENGTHS} body lengths"),
            ))
        if _TRAVEL.search(prompt) and not _STATIONARY.search(prompt):
            net = metrics["root_net_displacement_m"]
            required = MIN_TRAVEL_NET_BODY_LENGTHS * body_length
            distance_match = _DISTANCE.search(prompt)
            requested_distance = float(distance_match.group(1)) if distance_match else None
            if requested_distance is not None:
                required = max(required, MIN_EXPLICIT_DISTANCE_FRACTION * requested_distance)
            checks.append(_check("travel_displacement", net >= required, net,
                                 f"Planar net displacement >= {required:.3f} m"
                                 + (f" ({MIN_EXPLICIT_DISTANCE_FRACTION:.0%} of requested "
                                    f"{requested_distance:g} m)" if requested_distance is not None else
                                    f" ({MIN_TRAVEL_NET_BODY_LENGTHS} body lengths)")))
        if re.search(r'\bsidestep\w*\b', prompt, re.I) and not NEGATION.search(prompt):
            left = bool(re.search(r'\bleft\b', prompt, re.I))
            right = bool(re.search(r'\bright\b', prompt, re.I))
            if left != right:
                initial = positions[max(0,start-1)]
                lateral = (initial[JOINT_INDEX['left_hip_yaw_skel']] - initial[JOINT_INDEX['right_hip_yaw_skel']])[[0,2]]
                norm = float(np.linalg.norm(lateral))
                delta = positions[end-1,ROOT,[0,2]] - initial[ROOT,[0,2]]
                directed = float(delta @ (lateral / norm)) * (1 if left else -1) if norm > 1e-5 else 0.
                checks.append(_check('sidestep_direction', directed >= .25*body_length,
                                     directed, 'Travel at least 0.25 leg lengths toward requested body-relative side'))
        if re.search(r'\bdanc(?:es?|ing)\b', prompt, re.I) and not (NEGATION.search(prompt) or re.search(r'\b(?:slowly|pose|freeze|still)\b',prompt,re.I)):
            active = metrics['pose_active_frame_fraction']
            checks.append(_check('sustained_motion_activity', active >= .50, active,
                                 'At least half the frames exceed 0.3 leg lengths/second of articulated motion; this does not classify dance style'))
        if _FALL.search(prompt) and not NEGATION.search(prompt):
            elevation = metrics["root_elevation_above_ankles_terminal_m"]
            maximum = MAX_FALL_TERMINAL_ELEVATION_BODY_LENGTHS * body_length
            checks.append(_check("fall_terminal_low_pose", elevation <= maximum,
                                 elevation, f"Median root elevation above ankles in final five frames "
                                            f"<= {maximum:.3f} m ({MAX_FALL_TERMINAL_ELEVATION_BODY_LENGTHS} body lengths)"))
        if _BACKFLIP.search(prompt) and not NEGATION.search(prompt):
            run = metrics["torso_longest_inverted_run_frames"]
            checks.append(_check("backflip_torso_inversion", run >= MIN_INVERTED_RUN_FRAMES,
                                 run, f"Torso dot(up) <= {MIN_INVERTED_DOT_UP} for at least "
                                      f"{MIN_INVERTED_RUN_FRAMES} consecutive frames"))
            upright = recovered_upright([positions[start:end]])
            checks.append(_check("backflip_terminal_upright", upright, upright,
                                 "Final 10 frames satisfy story_recovery upright pose proxy"))
        if is_recovery_motion(prompt):
            upright = recovered_upright([positions[start:end]])
            checks.append(_check("recovery_terminal_upright", upright, upright,
                                 "Final 10 frames satisfy story_recovery upright pose proxy"))
        elif _STAND.search(prompt) and not NEGATION.search(prompt):
            upright = recovered_upright([positions[start:end]])
            checks.append(_check("standing_terminal_upright", upright, upright,
                                 "Final 10 frames satisfy story_recovery upright pose proxy"))
        segments.append({"index": index, "beat_id": segment.get("beat_id"),
                         "prompt": prompt, "start": start, "end": end,
                         "frames": end - start, "seconds": (end - start) / fps,
                         "metrics": metrics, "checks": checks})
    all_checks = structural_checks + [c for s in segments for c in s["checks"]]
    return {"take_id": take.id, "take_name": take.name, "fps": fps,
            "frames": len(positions), "seconds": len(positions) / fps,
            "body_length_m": body_length, "segment_count": len(segments),
            "structural_checks": structural_checks, "segments": segments,
            "passed": all(c["passed"] for c in all_checks),
            "failed_checks": [{"segment": s["index"], "name": c["name"]}
                              for s in segments for c in s["checks"] if not c["passed"]]
                             + [{"segment": None, "name": c["name"]}
                                for c in structural_checks if not c["passed"]],
            "interpretation": "Geometric proxies only; visual quality and full prompt adherence require review."}


def analyze_archive(content: bytes, *, take_id: str | None = None) -> dict:
    """Decode a Stagezero archive and inspect its active or named take."""
    takes, active, _, _ = decode_project(content)
    selected = active if take_id is None else take_id
    if selected not in takes:
        raise ValueError(f"Take {selected!r} not found in archive")
    return {"archive_format": "stagezero", "active_take_id": active,
            "selected_take_id": selected, "take": analyze_take(takes[selected])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Saved .stagezero.npz project")
    parser.add_argument("--take-id", help="Inspect this take instead of the active take")
    parser.add_argument("--output", type=Path, help="Also write the JSON report to this file")
    args = parser.parse_args(argv)
    try:
        report = analyze_archive(args.archive.read_bytes(), take_id=args.take_id)
        rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.write_text(rendered, encoding="utf-8")
        sys.stdout.write(rendered)
        return 0 if report["take"]["passed"] else 1
    except (OSError, ValueError, KeyError, IndexError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
