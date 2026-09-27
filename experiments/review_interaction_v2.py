"""Reproduce StageZero interaction trials and inspect grounded motion proxies.

The six cases reuse the reviewed PR26 plans and prompts exactly. Generation
requires an explicit model-lane reservation flag; measurement and same-source
ablation are offline. Every attempt gets a new directory, including failures.
These numerical diagnostics are never a substitute for complete video review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "review" / "interaction-quality"
REVIEW = ROOT / "review" / "interaction-v2"
sys.path.insert(0, str(ROOT))

from cast_performance import decode_project  # noqa: E402
from cast_motion_refinement import cast_body_clearance  # noqa: E402
from interaction_scene import is_walkable_ground, local_axes, scene_objects  # noqa: E402
from interaction_scene_collision import _body_proxies  # noqa: E402


CASES = {
    "city-handshake-s42": ("city-handshake", "city", 42, False),
    "city-handshake-s43": ("city-handshake", "city", 43, True),
    "industrial-spar-s45": ("industrial-spar", "industrial", 45, False),
    "market-three-s42": ("market-three", "market", 42, False),
    "market-three-s48": ("market-three", "market", 48, False),
    "market-three-s49": ("market-three", "market", 49, True),
}
CODE_FILES = (
    "experiments/trial_prompt_scene.py", "experiments/review_interaction_v2.py",
    "prompt_scene_builder.py", "prompt_scene_plan.py", "cast_motion_refinement.py",
    "cast_observer_motion.py", "paired_meetup.py", "native_pair_transition.py",
    "interaction_scene_collision.py", "cast_performance.py",
    "native_wait_pose.py", "cast_observer_turn.py", "cast_observer_continuation.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values: np.ndarray, q: float) -> float | None:
    return float(np.quantile(values, q)) if len(values) else None


def longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def runs(mask: np.ndarray) -> list[dict]:
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [{"start_frame": int(lo), "end_frame_exclusive": int(hi)}
            for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def selected(values: np.ndarray, mask: np.ndarray) -> dict:
    observed = values[mask & np.isfinite(values)]
    return {"frames": int(len(observed)),
            "min": float(observed.min()) if len(observed) else None,
            "p05": quantile(observed, .05), "p50": quantile(observed, .5),
            "p95": quantile(observed, .95),
            "max": float(observed.max()) if len(observed) else None}


def ground_height(scene: dict, xz: np.ndarray) -> np.ndarray:
    """Highest walkable slab under each XZ point, or NaN where unsupported."""
    points = np.asarray(xz, dtype=np.float64)
    result = np.full(points.shape[:-1], np.nan)
    for obj in scene_objects(scene):
        if not is_walkable_ground(obj):
            continue
        width, depth = local_axes(obj.yaw_degrees)
        delta = points - np.array([obj.x, obj.z])
        u = delta[..., 0] * width[0] + delta[..., 1] * width[1]
        v = delta[..., 0] * depth[0] + delta[..., 1] * depth[1]
        inside = (np.abs(u) <= obj.width / 2) & (np.abs(v) <= obj.depth / 2)
        top = obj.y + obj.height / 2
        result[inside] = np.fmax(result[inside], top)
    return result


def _source_manifest(archive: Path) -> list[dict]:
    """Read hashes from the actual archived source files, including rejects."""
    records = []
    for manifest in sorted((archive.parent / "sources").glob("scene-*/manifest.json")):
        try:
            data = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            records.append({"manifest": str(manifest.relative_to(ROOT)),
                            "sha256": digest(manifest), "parse_error": str(exc)})
            continue
        sources = []
        for record in data.get("sources", []):
            name = Path(record["path"]).name
            path = manifest.parent / name
            actual = digest(path) if path.is_file() else None
            sources.append({"name": name, "kind": record.get("source"),
                            "recorded_sha256": record.get("sha256"),
                            "actual_sha256": actual,
                            "hash_matches": actual == record.get("sha256") if actual else False})
        records.append({"manifest": str(manifest.relative_to(ROOT)),
                        "sha256": digest(manifest), "status": data.get("status"),
                        "sources": sources,
                        "all_archived_files": {path.name: digest(path)
                                               for path in sorted(manifest.parent.iterdir())
                                               if path.is_file() and path != manifest}})
    return records


def compare_archives(before: Path, after: Path) -> dict:
    """Describe exact display/source preservation without imposing root invariance."""
    new, _, _, _ = decode_project(after.read_bytes())
    with np.load(before, allow_pickle=False) as source:
        raw_joints_only = set(source.files) == {"joints"}
        if raw_joints_only:
            old_joints = np.asarray(source["joints"])
    if raw_joints_only:
        if old_joints.shape[1:] != new.joints.shape[1:]:
            raise ValueError("Raw before joints do not match the after cast skeleton and actor count")
        old = SimpleNamespace(joints=old_joints, frames=len(old_joints), actor_ids=new.actor_ids,
                              metadata={"segments": new.metadata.get("segments", []),
                                        "segment_activity": new.metadata.get("segment_activity", [])})
    else:
        old, _, _, _ = decode_project(before.read_bytes())
    aligned = old.frames == new.frames and old.actor_ids == new.actor_ids
    old_pairs = [s for s in old.metadata.get("segments", []) if s.get("source") == "intergen"]
    new_pairs = [s for s in new.metadata.get("segments", []) if s.get("source") == "intergen"]
    old_activities = old.metadata.get("segment_activity", [])
    new_activities = new.metadata.get("segment_activity", [])
    pair_checks = []
    for index, (left, right) in enumerate(zip(old_pairs, new_pairs)):
        left_activity = next((a for a in old_activities if a["start_frame"] == left["start_frame"]), None)
        right_activity = next((a for a in new_activities if a["start_frame"] == right["start_frame"]), None)
        left_ids = left_activity.get("active_actor_ids", []) if left_activity else []
        right_ids = right_activity.get("active_actor_ids", []) if right_activity else []
        comparable = left_ids == right_ids and all(aid in old.actor_ids and aid in new.actor_ids for aid in left_ids)
        if comparable:
            left_array = old.joints[left["start_frame"]:left["end_frame_exclusive"]][:, [old.actor_ids.index(a) for a in left_ids]]
            right_array = new.joints[right["start_frame"]:right["end_frame_exclusive"]][:, [new.actor_ids.index(a) for a in right_ids]]
            display_equal = bool(left_array.shape == right_array.shape and np.array_equal(left_array, right_array))
        else:
            display_equal = None
        pair_checks.append({"beat_index": index, "actor_ids_before": left_ids,
                            "actor_ids_after": right_ids,
                            "frames_before": left["end_frame_exclusive"] - left["start_frame"],
                            "frames_after": right["end_frame_exclusive"] - right["start_frame"],
                            "active_intergen_display_frames_equal": display_equal})
    old_sources = {(record.get("source"), record.get("beat_index"), Path(record["path"]).name): record.get("sha256")
                   for record in old.metadata.get("sources", []) if record.get("source") == "intergen"}
    new_sources = {(record.get("source"), record.get("beat_index"), Path(record["path"]).name): record.get("sha256")
                   for record in new.metadata.get("sources", []) if record.get("source") == "intergen"}
    roots = None
    if aligned:
        delta = np.linalg.norm(old.joints[:, :, 0] - new.joints[:, :, 0], axis=-1)
        roots = {"max_root_position_difference_m": float(delta.max()),
                 "changed_root_frames_over_1mm": int((delta > .001).sum())}
    return {"before_sha256": digest(before), "after_sha256": digest(after),
            "before_is_raw_unrefined_joints": raw_joints_only,
            "same_frame_grid_and_actor_order": aligned,
            "root_change": roots,
            "paired_segment_count_before": len(old_pairs),
            "paired_segment_count_after": len(new_pairs),
            "paired_display_checks": pair_checks,
            "raw_intergen_source_sha256_equal_by_beat": (old_sources == new_sources) if not raw_joints_only else None,
            "raw_intergen_sha256_before": {str(key): value for key, value in old_sources.items()},
            "raw_intergen_sha256_after": {str(key): value for key, value in new_sources.items()},
            "interpretation": "Root and post-turn leg changes are reported, not rejected; exact active InterGen checks require matching actor roles and source placement."}


def measure(archive: Path) -> dict:
    clip, _, scene, _ = decode_project(archive.read_bytes())
    joints = clip.joints.astype(np.float64)
    frames, fps = clip.frames, clip.fps
    segments = clip.metadata.get("segments", [])
    activities = clip.metadata.get("segment_activity", [])
    if len(segments) != len(activities):
        raise ValueError("Segment and activity provenance must align")
    for segment, activity in zip(segments, activities):
        if any(segment[key] != activity.get(key) for key in
               ("start_frame", "end_frame_exclusive")):
            raise ValueError("Segment and activity boundaries disagree")

    steps = np.linalg.norm(np.diff(joints, axis=0), axis=-1)
    roots = joints[:, :, 0]
    root_steps = np.linalg.norm(np.diff(roots, axis=0), axis=-1)
    held = np.zeros((frames, len(clip.actor_ids)), dtype=bool)
    transition = np.zeros(frames, dtype=bool)
    for segment, activity in zip(segments, activities):
        lo, hi = segment["start_frame"], segment["end_frame_exclusive"]
        if segment["source"] == "authored_transition":
            transition[lo:hi] = True
        for aid in activity.get("held_actor_ids", []):
            held[lo:hi, clip.actor_ids.index(aid)] = True

    performers = []
    for ai, aid in enumerate(clip.actor_ids):
        actor = joints[:, ai]
        root_ground = ground_height(scene, roots[:, ai][:, [0, 2]])
        held_frames = held[:, ai]
        still = np.max(steps[:, ai], axis=-1) < .001
        held_steps = held_frames[:-1] & held_frames[1:]
        feet = []
        for side, ankle, toe in (("left", 7, 10), ("right", 8, 11)):
            ankle_floor = ground_height(scene, actor[:, ankle][:, [0, 2]])
            toe_floor = ground_height(scene, actor[:, toe][:, [0, 2]])
            ankle_gap = actor[:, ankle, 1] - ankle_floor
            toe_gap = actor[:, toe, 1] - toe_floor
            ankle_speed = np.linalg.norm(np.diff(actor[:, ankle][:, [0, 2]], axis=0), axis=-1) * fps
            toe_speed = np.linalg.norm(np.diff(actor[:, toe][:, [0, 2]], axis=0), axis=-1) * fps
            vertical_speed = np.maximum(np.abs(np.diff(actor[:, ankle, 1])),
                                        np.abs(np.diff(actor[:, toe, 1]))) * fps
            supported = (np.isfinite(ankle_gap[:-1]) & np.isfinite(ankle_gap[1:]) &
                         np.isfinite(toe_gap[:-1]) & np.isfinite(toe_gap[1:]))
            toe_near = (toe_gap[:-1] >= -.04) & (toe_gap[:-1] <= .06) & (toe_gap[1:] >= -.04) & (toe_gap[1:] <= .06)
            candidate = supported & toe_near & (vertical_speed <= .25)
            slide = candidate & (toe_speed > .1)
            feet.append({
                "side": side,
                "scene_support_coverage_frames": int(np.isfinite(toe_gap).sum()),
                "toe_gap_m": selected(toe_gap, np.ones(frames, dtype=bool)),
                "ankle_gap_m": selected(ankle_gap, np.ones(frames, dtype=bool)),
                "held_toe_gap_m": selected(toe_gap, held_frames),
                "held_ankle_gap_m": selected(ankle_gap, held_frames),
                "held_toe_over_10cm_frames": int((held_frames & (toe_gap > .10)).sum()),
                "held_ankle_over_20cm_frames": int((held_frames & (ankle_gap > .20)).sum()),
                "toe_near_ground_ankle_over_20cm_frames": int((supported & toe_near & (ankle_gap[:-1] > .20)).sum()),
                "support_candidate_steps": int(candidate.sum()),
                "toe_slip_m_s": selected(toe_speed, candidate),
                "ankle_slip_m_s": selected(ankle_speed, candidate),
                "toe_sliding_steps_over_0p1m_s": int(slide.sum()),
                "longest_toe_slide_seconds": longest_run(slide) / fps,
                "held_toe_slip_m_s": selected(toe_speed, candidate & held_steps),
                "authored_transition_toe_slip_m_s": selected(toe_speed, candidate & transition[:-1] & transition[1:]),
            })
        performers.append({
            "actor_id": aid, "held_frames": int(held_frames.sum()),
            "held_spans": runs(held_frames),
            "held_still_step_fraction_under_1mm": float(still[held_steps].mean()) if held_steps.any() else None,
            "longest_held_still_seconds": longest_run(still & held_steps) / fps,
            "hip_height_above_scene_ground_m": selected(actor[:, 0, 1] - root_ground, np.ones(frames, dtype=bool)),
            "head_height_above_scene_ground_m": selected(actor[:, 15, 1] - root_ground, np.ones(frames, dtype=bool)),
            "held_hip_height_above_scene_ground_m": selected(actor[:, 0, 1] - root_ground, held_frames),
            "feet": feet,
        })

    transitions = []
    for segment in segments:
        if segment["source"] != "authored_transition":
            continue
        lo, hi = segment["start_frame"], segment["end_frame_exclusive"]
        local = steps[lo:max(lo, hi - 1)]
        boundary = sorted({step for step in (lo - 1, hi - 1) if 0 <= step < frames - 1})
        transitions.append({"start_frame": lo, "end_frame_exclusive": hi,
                            "within_max_joint_step_m": float(local.max()) if local.size else None,
                            "within_p95_joint_step_m": quantile(local.ravel(), .95),
                            "boundary_max_joint_step_m": float(steps[boundary].max()) if boundary else None,
                            "boundary_max_root_step_m": float(root_steps[boundary].max()) if boundary else None})

    clearance = cast_body_clearance(joints, clip.actor_ids, activities)
    body_proxies = [_body_proxies(joints[:, ai], "native22") for ai in range(len(clip.actor_ids))]
    bodies = []
    for (a, b), (values, unintended) in clearance.items():
        pa, ra = body_proxies[a]
        pb, rb = body_proxies[b]
        torso = (np.linalg.norm(pa[:, :4, None] - pb[:, None, :4], axis=-1)
                 - ra[None, :4, None] - rb[None, None, :4]).min(axis=(1, 2))
        intended = ~unintended
        for label, mask in (("intended", intended), ("unintended", unintended)):
            observed = values[mask]
            overlap = (values < 0) & mask
            worst = int(np.argmin(np.where(mask, values, np.inf))) if mask.any() else None
            bodies.append({"actor_ids": [clip.actor_ids[a], clip.actor_ids[b]],
                           "window": label, "frames": int(mask.sum()),
                           "minimum_proxy_clearance_m": float(observed.min()) if len(observed) else None,
                           "minimum_torso_proxy_clearance_m": float(torso[mask].min()) if mask.any() else None,
                           "worst_frame": worst, "overlap_frames": int(overlap.sum()),
                           "longest_overlap_seconds": longest_run(overlap) / fps,
                           "frames_below_minus_5cm": int(((values < -.05) & mask).sum())})

    paired = [s for s in segments if s.get("kind") == "paired_action"]
    beats = [b for b in clip.metadata.get("plan", {}).get("beats", []) if len(b.get("actor_ids", [])) == 2]
    contact = []
    for beat, segment in zip(beats, paired):
        if not any(word in beat["prompt"].lower() for word in ("hand", "greet")):
            continue
        a, b = [clip.actor_ids.index(aid) for aid in beat["actor_ids"]]
        lo, hi = segment["start_frame"], segment["end_frame_exclusive"]
        wrists = joints[lo:hi, [a, b]][:, :, [20, 21]]
        nearest = np.linalg.norm(wrists[:, 0, :, None] - wrists[:, 1, None, :], axis=-1).min(axis=(1, 2))
        near = nearest < .15
        contact.append({"beat_id": beat["id"], "actor_ids": beat["actor_ids"],
                        "window_frames": hi - lo, "nearest_wrist_min_m": float(nearest.min()),
                        "nearest_wrist_p50_m": quantile(nearest, .5),
                        "frames_under_15cm": int(near.sum()),
                        "longest_under_15cm_seconds": longest_run(near) / fps,
                        "first_under_15cm_frame": lo + int(np.flatnonzero(near)[0]) if near.any() else None})

    handoffs = []
    for first, second in zip(paired, paired[1:]):
        first_activity = activities[segments.index(first)]
        second_activity = activities[segments.index(second)]
        common = set(first_activity["active_actor_ids"]) & set(second_activity["active_actor_ids"])
        lo, hi = first["end_frame_exclusive"], second["start_frame"]
        interval = slice(lo, hi)
        handoff_clearance = []
        for (a, b), (values, _) in clearance.items():
            observed = values[interval]
            handoff_clearance.append({"actor_ids": [clip.actor_ids[a], clip.actor_ids[b]],
                                      "minimum_body_proxy_clearance_m": float(observed.min()) if len(observed) else None,
                                      "body_proxy_overlap_frames": int((observed < 0).sum())})
        shared_orientation = {}
        for aid in common:
            ai = clip.actor_ids.index(aid)
            # Shoulder axis gives an observable facing proxy without relying on root yaw metadata.
            shoulder = joints[:, ai, 14][:, [0, 2]] - joints[:, ai, 13][:, [0, 2]]
            facing = np.stack((-shoulder[:, 1], shoulder[:, 0]), axis=-1)
            unit = facing / np.maximum(np.linalg.norm(facing, axis=-1, keepdims=True), 1e-9)
            angle = float(np.degrees(np.arccos(np.clip(unit[lo - 1] @ unit[hi], -1, 1))))
            shared_orientation[aid] = angle
        handoffs.append({"from_pair": first_activity["active_actor_ids"],
                         "to_pair": second_activity["active_actor_ids"],
                         "between_frames": hi - lo,
                         "between_body_overlap_proxy": handoff_clearance,
                         "shared_actor_facing_change_degrees": shared_orientation,
                         "shared_actor_root_travel_m": {aid: float(np.linalg.norm(
                             roots[hi, clip.actor_ids.index(aid)] - roots[lo - 1, clip.actor_ids.index(aid)]))
                             for aid in sorted(common)},
                         "max_intervening_joint_step_m": float(steps[lo - 1:hi].max()) if hi >= lo else None})

    return {"archive_sha256": digest(archive), "frames": frames, "fps": fps,
            "duration_seconds": frames / fps, "performers": performers,
            "authored_transitions": transitions, "body_overlap_proxy": bodies,
            "intended_hand_contact_proxy": contact, "pair_handoffs": handoffs,
            "max_joint_step_m": float(steps.max()),
            "measurement_limits": [
                "Ankle is a heel proxy; joint coordinates are not the sole or shoe surface.",
                "Scene slab top is a support proxy; sampled XZ footprints can miss small surfaces.",
                "Toe support and sliding thresholds infer contact without forces or foot locking.",
                "Wrist proximity does not establish palm or finger contact.",
                "Body sphere overlap is sampled, not mesh collision or physical penetration.",
                "Numerical diagnostics do not establish visual animation quality."]}


def next_attempt(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    number = 1
    while (base / f"attempt-{number:03d}").exists():
        number += 1
    return base / f"attempt-{number:03d}"


def provenance(case: str, plan: Path, scene: Path, seed: int) -> dict:
    return {"case": case, "seed": seed,
            "plan": str(plan.relative_to(ROOT)), "plan_sha256": digest(plan),
            "scene": str(scene.relative_to(ROOT)), "scene_sha256": digest(scene),
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_dirty_at_launch": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
            "code_file_sha256_at_launch": {name: digest(ROOT / name) for name in CODE_FILES}}


def generate(case: str, provider: Path, token: Path, core_url: str) -> bool:
    plan_name, background, seed, held_out = CASES[case]
    plan = LEGACY / "plans" / f"{plan_name}.json"
    scene = ROOT / "review" / "prompt-scenes" / "backgrounds" / f"{background}.json"
    prompt = json.loads(plan.read_text())["prompt"]
    output = next_attempt(REVIEW / "fresh" / case)
    prior = provenance(case, plan, scene, seed)
    prior.update({"held_out": held_out, "comparison_type": "fresh_Core_and_InterGen_same_saved_plan_seed",
                  "planning": "reviewed saved PR26 plan; no external AI planning",
                  "provider_config_path": str(provider), "provider_secret_contents_recorded": False})
    command = [sys.executable, str(ROOT / "experiments" / "trial_prompt_scene.py"),
               "--prompt", prompt, "--scene", str(scene), "--plan", str(plan),
               "--config", str(provider), "--token", str(token), "--core-url", core_url,
               "--seed", str(seed), "--output", str(output)]
    start = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        returncode, log = result.returncode, result.stdout + result.stderr
    except Exception as exc:
        returncode, log = 1, f"{type(exc).__name__}: {exc}\n"
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.log").write_text(log)
    archive = output / "scene.cast.stagezero.npz"
    prior.update({"runner_wall_seconds": time.perf_counter() - start,
                  "exit_code": returncode, "source_archives": _source_manifest(archive),
                  "archive_sha256": digest(archive) if archive.is_file() else None,
                  "command": ["<python>", "experiments/trial_prompt_scene.py", "--prompt", prompt,
                              "--scene", str(scene.relative_to(ROOT)), "--plan", str(plan.relative_to(ROOT)),
                              "--config", "<private-config>", "--token", "<private-token>",
                              "--core-url", core_url, "--seed", str(seed),
                              "--output", str(output.relative_to(ROOT))]})
    (output / "generation-provenance.json").write_text(json.dumps(prior, indent=2) + "\n")
    if archive.is_file():
        try:
            (output / "metrics-v2.json").write_text(json.dumps(measure(archive), indent=2) + "\n")
        except Exception as exc:
            (output / "measurement-failure.json").write_text(json.dumps({
                "type": type(exc).__name__, "error": str(exc)}, indent=2) + "\n")
            returncode = 1
    if returncode or not archive.is_file():
        (output / "generation-failure.json").write_text(json.dumps({
            "exit_code": returncode, "archive_preserved": archive.is_file(),
            "source_directory_preserved": (output / "sources").exists()}, indent=2) + "\n")
    print(json.dumps({"case": case, "output": str(output.relative_to(ROOT)),
                      "exit_code": returncode, "archive": archive.is_file()}), flush=True)
    return returncode == 0 and archive.is_file()


def ablate(case: str) -> bool:
    from experiments.review_interaction_quality import refine_archive

    plan_name, _, seed, _ = CASES[case]
    source = LEGACY / "final" / f"{case}-fixed-plan" / "scene.cast.stagezero.npz"
    output = next_attempt(REVIEW / "ablation" / case)
    try:
        refine_archive(source, output, seed)
        before = output / "before.cast.stagezero.npz"
        after = output / "after.cast.stagezero.npz"
        (output / "before-metrics-v2.json").write_text(json.dumps(measure(before), indent=2) + "\n")
        (output / "after-metrics-v2.json").write_text(json.dumps(measure(after), indent=2) + "\n")
        (output / "v2-provenance.json").write_text(json.dumps({
            "case": case, "comparison_type": "exact_PR26_final_source_offline_refinement_only",
            "scope_limit": "Does not generate or evaluate the new Core observer turn or changed scene composition",
            "source_archive": str(source.relative_to(ROOT)), "source_sha256": digest(source),
            "source_plan": plan_name, "seed": seed,
            "before_archive_sha256": digest(before), "after_archive_sha256": digest(after),
            "code_file_sha256": {name: digest(ROOT / name) for name in CODE_FILES}}, indent=2) + "\n")
        (output / "comparison-v2.json").write_text(json.dumps(compare_archives(before, after), indent=2) + "\n")
        return True
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        (output / "v2-failure.json").write_text(json.dumps({
            "case": case, "type": type(exc).__name__, "error": str(exc),
            "source_archive_sha256": digest(source) if source.is_file() else None,
            "partial_artifacts_preserved": True}, indent=2) + "\n")
        print(json.dumps({"case": case, "output": str(output.relative_to(ROOT)),
                          "error": str(exc)}), flush=True)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", *CASES), default="all")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", action="store_true")
    mode.add_argument("--ablate", action="store_true")
    mode.add_argument("--measure-archive", type=Path)
    mode.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--model-lane-reserved", action="store_true",
                        help="Required acknowledgement before live Core and InterGen calls")
    parser.add_argument("--provider-config", type=Path)
    parser.add_argument("--token", type=Path)
    parser.add_argument("--core-url", default="http://127.0.0.1:8769")
    args = parser.parse_args()
    if args.measure_archive:
        print(json.dumps(measure(args.measure_archive.resolve()), indent=2))
        return
    if args.compare:
        print(json.dumps(compare_archives(*[path.resolve() for path in args.compare]), indent=2))
        return
    if args.generate and (not args.model_lane_reserved or not args.provider_config or not args.token):
        parser.error("--generate requires --model-lane-reserved, --provider-config and --token")
    chosen = CASES if args.case == "all" else {args.case: CASES[args.case]}
    passed = True
    for case in chosen:
        passed &= (generate(case, args.provider_config, args.token, args.core_url)
                   if args.generate else ablate(case))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
