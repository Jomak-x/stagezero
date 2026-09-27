"""Reproduce and measure fresh StageZero interaction scenes with fixed saved plans.

The scene plans under review/interaction-quality/plans were generated earlier.
This runner does not call the external AI planning gateway. Every generation
calls the live Core and InterGen services for new source motion. It preserves
each attempt, including rejected source archives and the error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review" / "interaction-quality"
sys.path.insert(0, str(ROOT))

from cast_performance import CastPerformance, decode_project, encode_project  # noqa: E402


CASES = {
    "city-handshake-s42": (
        "Two people shake hands.", "city", "city-handshake", 42),
    "city-handshake-s43": (
        "Two people shake hands.", "city", "city-handshake", 43),
    "industrial-spar-s45": (
        "Two people start at x -3 z 0 and x 3 z 0, meet at x 0 z 0, then perform a controlled boxing spar with dodges.",
        "industrial", "industrial-spar", 45),
    "market-three-s42": (
        "Three people greet each other in turn. Person 1 shakes hands with person 2 and releases. Then person 2 shakes hands with person 3.",
        "market", "market-three", 42),
    "market-three-s48": (
        "Three people greet each other in turn. Person 1 shakes hands with person 2 and releases. Then person 2 shakes hands with person 3.",
        "market", "market-three", 48),
    "market-three-s49": (
        "Three people greet each other in turn. Person 1 shakes hands with person 2 and releases. Then person 2 shakes hands with person 3.",
        "market", "market-three", 49),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, q):
    return None if not len(values) else float(np.quantile(values, q))


def longest_run(mask):
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def measure(project: Path):
    clip, _, _, _ = decode_project(project.read_bytes())
    joints = clip.joints.astype(np.float64)
    fps = clip.fps
    metadata = clip.metadata
    root = joints[:, :, 0]
    steps = np.linalg.norm(np.diff(joints, axis=0), axis=-1)
    root_steps = np.linalg.norm(np.diff(root, axis=0), axis=-1)
    segments = metadata.get("segments", [])
    activities = metadata.get("segment_activity", [])
    seams = []
    authored_steps = np.zeros(clip.frames - 1, dtype=bool)
    for segment in segments[1:]:
        frame = segment["start_frame"]
        seams.append({
            "frame": frame, "into": segment["kind"],
            "max_joint_step_m": float(steps[frame - 1].max()),
            "max_root_step_m": float(root_steps[frame - 1].max()),
        })
    for segment in segments:
        if segment.get("source") == "authored_transition":
            lo, hi = segment["start_frame"], segment["end_frame_exclusive"]
            authored_steps[lo:max(lo, hi - 1)] = True

    idle = []
    for actor_index, actor_id in enumerate(clip.actor_ids):
        held = np.zeros(clip.frames - 1, dtype=bool)
        for activity in activities:
            if actor_id in activity.get("held_actor_ids", []):
                start = activity["start_frame"]
                end = activity["end_frame_exclusive"]
                held[start:max(start, end - 1)] = True
        max_step = steps[:, actor_index].max(axis=-1)
        still = max_step < .001
        idle.append({
            "actor_id": actor_id,
            "held_transition_frames": int(held.sum()),
            "held_still_fraction_under_1mm": float(still[held].mean()) if held.any() else None,
            "held_motion_fraction_over_0p1mm": float((max_step[held] > .0001).mean()) if held.any() else None,
            "held_joint_step_p95_m": quantile(max_step[held], .95),
            "longest_whole_track_still_seconds_under_1mm": longest_run(still) / fps,
            "longest_held_still_seconds_under_1mm": longest_run(still & held) / fps,
        })

    feet = []
    for actor_index, actor_id in enumerate(clip.actor_ids):
        actor = joints[:, actor_index]
        for side, ankle, toe in (("left", 7, 10), ("right", 8, 11)):
            support = np.minimum(actor[:, ankle, 1], actor[:, toe, 1])
            floor = float(np.quantile(support, .05))
            center = (actor[:, ankle] + actor[:, toe]) * .5
            horizontal_speed = np.linalg.norm(np.diff(center[:, [0, 2]], axis=0), axis=-1) * fps
            vertical_speed = np.abs(np.diff(support)) * fps
            near_floor = (support[:-1] - floor <= .06) & (support[1:] - floor <= .06)
            slow_vertical = vertical_speed <= .25
            candidate = near_floor & slow_vertical
            observed = horizontal_speed[candidate]
            authored_observed = horizontal_speed[candidate & authored_steps]
            feet.append({
                "actor_id": actor_id, "side": side,
                "candidate_support_frames": int(candidate.sum()),
                "candidate_support_fraction": float(candidate.mean()),
                "horizontal_speed_p50_m_s": quantile(observed, .50),
                "horizontal_speed_p95_m_s": quantile(observed, .95),
                "horizontal_speed_max_m_s": None if not len(observed) else float(observed.max()),
                "fraction_over_0p1m_s": None if not len(observed) else float((observed > .1).mean()),
                "authored_transition_candidate_frames": int((candidate & authored_steps).sum()),
                "authored_transition_horizontal_speed_p95_m_s": quantile(authored_observed, .95),
                "authored_transition_fraction_over_0p1m_s": None if not len(authored_observed) else float((authored_observed > .1).mean()),
            })

    contact = []
    paired_beats = [beat for beat in metadata.get("plan", {}).get("beats", [])
                    if len(beat.get("actor_ids", [])) == 2]
    paired_segments = [segment for segment in segments if segment.get("kind") == "paired_action"]
    for beat, segment in zip(paired_beats, paired_segments):
        prompt = beat["prompt"].lower()
        if "hand" not in prompt and "greet" not in prompt:
            continue
        actors = [clip.actor_ids.index(actor_id) for actor_id in beat["actor_ids"]]
        start, end = segment["start_frame"], segment["end_frame_exclusive"]
        hands = joints[start:end, actors][:, :, [20, 21]]
        distances = np.linalg.norm(hands[:, 0, :, None] - hands[:, 1, None, :], axis=-1)
        nearest = distances.min(axis=(1, 2))
        contact.append({
            "beat_id": beat["id"], "actor_ids": beat["actor_ids"],
            "window_frames": end - start,
            "nearest_wrist_min_m": float(nearest.min()),
            "nearest_wrist_median_m": float(np.median(nearest)),
            "frames_under_15cm": int((nearest < .15).sum()),
            "longest_under_15cm_seconds": longest_run(nearest < .15) / fps,
        })

    return {
        "archive_sha256": digest(project), "frames": clip.frames, "fps": fps,
        "duration_seconds": clip.frames / fps,
        "model_generated_source_records": metadata.get("sources"),
        "source_pair_frames_modified": metadata.get("source_pair_frames_modified"),
        "max_joint_step_m": float(steps.max()),
        "joint_step_p95_m": float(np.quantile(steps, .95)),
        "max_root_step_m": float(root_steps.max()),
        "seams": seams, "idle": idle, "foot_support_proxy": feet,
        "intended_hand_contact_windows": contact,
        "measurement_limits": [
            "Wrist proximity is not palm or finger contact.",
            "Support is inferred from ankle/toe height and low vertical speed, not force or exact foot planting.",
            "Stillness and frame-step metrics do not establish visual animation quality.",
            "Joint and root distances do not establish full mesh collision clearance.",
        ],
    }


def refine_archive(source: Path, output: Path, seed: int, *, feet: bool = False):
    """Create a checked same-source before/after ablation without GPU requests."""
    from cast_motion_refinement import RefinementRejected, refine_cast_motion
    from prompt_scene_builder import check_cast_geometry

    if output.exists():
        raise FileExistsError(f"Refusing to replace an ablation: {output}")
    clip, cast, scene, frame = decode_project(source.read_bytes())
    segments = clip.metadata["segments"]
    activities = [dict(record) for record in clip.metadata["segment_activity"]]
    geometry = clip.metadata.get("scene_geometry", [])
    if len(segments) != len(activities) or len(geometry) != len(segments):
        raise ValueError("Archive is missing matching segment activity and geometry provenance")
    for segment, activity, original_geometry in zip(segments, activities, geometry):
        if (segment["start_frame"] != activity["start_frame"] or
                segment["end_frame_exclusive"] != activity["end_frame_exclusive"]):
            raise ValueError("Archive segment and activity frames disagree")
        pair = original_geometry.get("intended_contact_pair", [])
        if any(actor not in activity["active_actor_ids"] for actor in pair):
            raise ValueError("Archived contact pair disagrees with active actors")
        activity["contact_actor_ids"] = pair

    output.mkdir(parents=True)
    before_path = output / "before.cast.stagezero.npz"
    after_path = output / "after.cast.stagezero.npz"
    shutil.copyfile(source, before_path)
    (output / "before-metrics.json").write_text(json.dumps(measure(before_path), indent=2) + "\n")
    try:
        refined, report = refine_cast_motion(clip.joints, clip.actor_ids, segments, activities,
                                             fps=clip.fps, seed=seed, feet=feet)
        if not np.array_equal(clip.joints[:, :, 0], refined[:, :, 0]):
            raise ValueError("Cast refinement modified root positions")
        native_frames = 0
        for segment, activity in zip(segments, activities):
            if segment["source"] == "intergen":
                lo, hi = segment["start_frame"], segment["end_frame_exclusive"]
                indices = [clip.actor_ids.index(actor) for actor in activity["active_actor_ids"]]
                native_frames += (hi - lo) * len(indices)
                if not np.array_equal(clip.joints[lo:hi, indices], refined[lo:hi, indices]):
                    raise ValueError("Refinement modified original active native pair frames")
        checked = []
        for segment, activity in zip(segments, activities):
            lo, hi = segment["start_frame"], segment["end_frame_exclusive"]
            checked.append(check_cast_geometry(refined[max(0, lo - 1):hi], scene,
                                               clip.actor_ids,
                                               contact_pair=activity["contact_actor_ids"]))
        metadata = clip.metadata
        metadata["segment_activity"] = activities
        metadata["motion_refinement"] = report
        metadata["refined_scene_geometry"] = checked
        metadata["same_source_ablation"] = {
            "baseline_sha256": digest(before_path), "seed": seed,
            "ankle_planting_enabled": feet,
            "exact_active_intergen_actor_frames_preserved": native_frames,
            "all_segment_scene_geometry_rechecked": True,
            "visual_acceptance": "unverified",
        }
        after = CastPerformance(clip.actor_ids, refined, fps=clip.fps, metadata=metadata)
        after_path.write_bytes(encode_project(after, cast, scene_document=scene, frame=frame))
        (output / "after-metrics.json").write_text(json.dumps(measure(after_path), indent=2) + "\n")
        (output / "refinement-report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"before": str(before_path), "after": str(after_path),
                          "native_actor_frames_preserved": native_frames}), flush=True)
    except RefinementRejected as exc:
        np.savez_compressed(output / "rejected-refinement-candidate.npz",
                            joints=exc.candidate_joints)
        (output / "refinement-report.json").write_text(
            json.dumps(exc.refinement_report, indent=2) + "\n")
        (output / "refine-failure.json").write_text(json.dumps({
            "type": type(exc).__name__, "error": str(exc),
            "before_archive_preserved": True,
            "rejected_candidate_preserved": True}, indent=2) + "\n")
        raise
    except Exception as exc:
        if "refined" in locals():
            np.savez_compressed(output / "rejected-refinement-candidate.npz", joints=refined)
        (output / "refine-failure.json").write_text(json.dumps({
            "type": type(exc).__name__, "error": str(exc),
            "before_archive_preserved": True,
            "rejected_candidate_preserved": "refined" in locals()}, indent=2) + "\n")
        raise


def generate(case, phase, provider, token, core_url):
    if phase != "final":
        raise ValueError("New generation is restricted to the selected final phase; baseline and after are historical evidence")
    prompt, background, plan_name, seed = CASES[case]
    scene = ROOT / "review" / "prompt-scenes" / "backgrounds" / f"{background}.json"
    plan = REVIEW / "plans" / f"{plan_name}.json"
    if not scene.is_file() or not plan.is_file():
        raise FileNotFoundError("Scene or reviewed plan is missing")
    base = REVIEW / phase / f"{case}-fixed-plan"
    output = base
    attempt = 2
    while output.exists():
        output = base.with_name(f"{base.name}-attempt-{attempt:03d}")
        attempt += 1
    command = [sys.executable, str(ROOT / "experiments" / "trial_prompt_scene.py"),
               "--prompt", prompt, "--scene", str(scene), "--plan", str(plan),
               "--config", str(provider), "--token", str(token), "--core-url", core_url,
               "--seed", str(seed), "--output", str(output)]
    code_files = ("experiments/trial_prompt_scene.py", "prompt_scene_builder.py",
                  "prompt_scene_plan.py", "cast_motion_refinement.py",
                  "cast_observer_motion.py", "paired_meetup.py",
                  "native_pair_transition.py")
    code_hashes = {name: digest(ROOT / name) for name in code_files}
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                          cwd=ROOT, text=True).strip()
    git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                             cwd=ROOT, text=True).strip())
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.log").write_text(result.stdout + result.stderr)
    provenance = {
        "case": case, "phase": phase, "prompt": prompt, "background": background,
        "seed": seed, "planning": "reviewed_saved_plan; no new external AI planning",
        "motion": "fresh Core and InterGen model requests; no replay manifest",
        "scene": str(scene.relative_to(ROOT)), "scene_sha256": digest(scene),
        "plan": str(plan.relative_to(ROOT)), "plan_sha256": digest(plan),
        "provider_config_path": str(provider), "provider_secret_contents_recorded": False,
        "code_commit": code_commit,
        "code_file_sha256_at_launch": code_hashes,
        "git_dirty_at_launch": git_dirty,
        "command": ["<existing-python>", "experiments/trial_prompt_scene.py", "--prompt", prompt,
                    "--scene", str(scene.relative_to(ROOT)), "--plan", str(plan.relative_to(ROOT)),
                    "--config", "<private-existing-config>", "--token", "<private-existing-token>",
                    "--core-url", core_url, "--seed", str(seed), "--output", str(output.relative_to(ROOT))],
        "runner_wall_seconds": elapsed, "exit_code": result.returncode,
    }
    (output / "generation-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    project = output / "scene.cast.stagezero.npz"
    if project.is_file():
        (output / "metrics.json").write_text(json.dumps(measure(project), indent=2) + "\n")
    else:
        (output / "generation-failure.json").write_text(json.dumps({
            "exit_code": result.returncode, "sources_preserved": (output / "sources").exists(),
            "log": "run.log"}, indent=2) + "\n")
    print(json.dumps({"case": case, "phase": phase, "output": str(output.relative_to(ROOT)),
                      "exit_code": result.returncode, "seconds": round(elapsed, 3)}), flush=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "after", "final"))
    parser.add_argument("--case", choices=("all", *CASES), default="all")
    parser.add_argument("--measure-only", action="store_true")
    parser.add_argument("--refine-archive", type=Path,
                        help="Offline same-source ablation from a saved cast archive")
    parser.add_argument("--refine-output", type=Path)
    parser.add_argument("--refine-seed", type=int)
    parser.add_argument("--refine-feet", action="store_true",
                        help="Opt into experimental ankle planting for offline ablation")
    parser.add_argument("--provider-config", type=Path, default=Path(
        "/Users/jakob/.codex/worktrees/motion-performance/Shellhacks/.runtime/prompt-native-provider.json"))
    parser.add_argument("--token", type=Path, default=Path(
        "/Users/jakob/Desktop/Shellhacks/.runtime/api-token"))
    parser.add_argument("--core-url", default="http://127.0.0.1:8769")
    args = parser.parse_args()
    if args.refine_archive:
        if args.phase or not args.refine_output or args.refine_seed is None:
            parser.error("--refine-archive requires --refine-output and --refine-seed, without --phase")
        refine_archive(args.refine_archive.resolve(), args.refine_output.resolve(),
                       args.refine_seed, feet=args.refine_feet)
        return
    if not args.phase:
        parser.error("--phase is required for generation or measurement")
    if args.phase != "final" and not args.measure_only:
        parser.error("Historical baseline/after phases support measurement only; use --phase final for new model generation")
    selected = CASES if args.case == "all" else {args.case: CASES[args.case]}
    failed = False
    for case in selected:
        if args.measure_only:
            base = REVIEW / args.phase / f"{case}-fixed-plan"
            paths = sorted(base.parent.glob(base.name + "*"))
            for output in paths:
                project = output / "scene.cast.stagezero.npz"
                if project.is_file():
                    (output / "metrics.json").write_text(json.dumps(measure(project), indent=2) + "\n")
                    print(json.dumps({"case": case, "project": str(project.relative_to(ROOT))}))
        else:
            failed |= bool(generate(case, args.phase, args.provider_config, args.token, args.core_url))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
