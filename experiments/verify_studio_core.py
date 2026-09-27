#!/usr/bin/env python3
"""Bounded opt-in live verification of the independent native Core studio.

Uses an already running service. Never provisions/restarts a worker, prints a
credential, or changes generated poses. Every attempted case is reported even
when planning, transport, or precommit validation fails.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interaction_metrics import pair_separation, gate_traversal
from interaction_scene import scene_objects, passage_for, local_axes
from interaction_scene_collision import scene_collision
from realtime_client import RealtimeClient
from realtime_navigation import plan_navigation, validate_ground_path
from scene_composition import generate_recipe, validate_scene
from studio_core_session import CoreStudioSession
from studio_interaction_scene import adapt_studio_scene, recommend_placements


class BoundedClient(RealtimeClient):
    def __init__(self, *args, max_jobs=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.submitted_jobs = 0
        self.max_jobs = max_jobs

    def wait(self, body, *args, **kwargs):
        if self.submitted_jobs >= self.max_jobs:
            raise RuntimeError("Live verification reached its explicit GPU job cap")
        self.submitted_jobs += 1
        return super().wait(body, *args, **kwargs)


def measure(session):
    clip = session.timeline_clip()
    if clip is None:
        return {"frames": 0}
    adapted = adapt_studio_scene(session.scene_document)
    metrics = {"frames": clip.frames, "actor_ids": list(clip.actor_ids), "fps": clip.fps,
               "scene_objects": len(adapted["scene"]["objects"]),
               "geometry_proxy_only": True,
               "scene_collision_proxy": [scene_collision(p, "core27", adapted["scene"], adapted["affordances"])
                                         for p in clip.positions]}
    ground = []
    for actor_positions in clip.positions:
        try:
            diagnostics = validate_ground_path(adapted["scene"], actor_positions[:, 0, :][:, [0, 2]], actor_radius_m=.28)
            ground.append({"supported": True, "diagnostics": diagnostics})
        except ValueError as exc:
            ground.append({"supported": False, "error": str(exc)})
    metrics["authored_floor_coverage"] = ground
    seams = [segment["start"] for segment in session.snapshot()["segments"] if segment["start"] > 0]
    metrics["root_seams_m"] = [{"frame": f, "per_actor": np.linalg.norm(
        clip.positions[:, f, 0] - clip.positions[:, f - 1, 0], axis=-1).tolist()} for f in seams]
    metrics["maximum_root_seam_m"] = max((max(s["per_actor"]) for s in metrics["root_seams_m"]), default=0.)
    metrics["maximum_root_frame_step_m"] = (float(np.linalg.norm(np.diff(clip.positions[:, :, 0], axis=1), axis=-1).max())
                                              if clip.frames > 1 else 0.)
    if len(clip.actor_ids) == 2:
        metrics["pair_separation"] = pair_separation(clip.positions[0], clip.positions[1],
            skeleton_a="core27", skeleton_b="core27", radius_a_m=.325, radius_b_m=.325)
    return metrics


def pump(session, *, timeout):
    start = time.monotonic()
    first_commit = None
    hold_seconds = 0.
    last = start
    phases = []
    prior_frame = None
    initial_frames = session.snapshot()["total_frames"]
    while time.monotonic() - start < timeout:
        now = time.monotonic()
        state = session.tick(now=now)
        if state["phase"] not in phases:
            phases.append(state["phase"])
        if state["total_frames"] > initial_frames and first_commit is None:
            first_commit = now - start
        if state["generated"] and state["phase"] == "buffering" and state["frame"] == prior_frame:
            hold_seconds += now - last
        last, prior_frame = now, state["frame"]
        if state["failure"] or (state["queued_stages"] == 0 and state["inflight_request_id"] is None):
            break
        time.sleep(.02)
    else:
        state = session.snapshot()
        state["verification_timeout"] = True
    return {"elapsed_seconds": time.monotonic() - start, "first_new_commit_seconds": first_commit,
            "post_start_buffer_hold_seconds": hold_seconds, "observed_phases": phases,
            "state": state}


def archive_result(session, name, directory):
    if not session.snapshot()["total_frames"]:
        return {}
    path = directory / f"{name}.core.stagezero.npz"
    data = session.save_project()
    path.write_bytes(data)
    before = session.timeline_clip()
    with CoreStudioSession() as restored:
        restored.load_project(data)
        after = restored.timeline_clip()
        exact = (np.array_equal(before.positions, after.positions)
                 and np.array_equal(before.rotations, after.rotations)
                 and np.array_equal(before.native_features, after.native_features)
                 and restored.scene_document == session.scene_document)
    print(json.dumps({"saved_archive": str(path), "exact_round_trip": exact}), flush=True)
    return {"archive": str(path), "archive_bytes": len(data), "exact_save_load": exact}


def navigation_metrics(session, target_id, verb, route):
    clip = session.timeline_clip()
    if clip is None:
        return {"navigation_checks_passed": False}
    target_xz = np.array(route["waypoints"][-1]["position_xz"])
    error = float(np.linalg.norm(clip.positions[0, -1, 0, [0, 2]] - target_xz))
    result = {"final_root_target_error_m": error, "navigation_checks_passed": error <= .15}
    if verb == "go_through":
        obj = next(obj for obj in scene_objects(session.scene_document) if obj.id == target_id)
        passage = passage_for(obj, None, actor_height_m=1.65)
        normal = local_axes(passage.yaw_degrees)[1]
        floor = obj.y - obj.height / 2
        crossing = gate_traversal(clip.positions[0], skeleton="core27",
            center=(passage.center_xz[0], floor, passage.center_xz[1]), normal_xz=normal,
            opening_width_m=passage.width_m, min_y=floor, max_y=floor + passage.height_m,
            body_radius_m=.15, direction="either")
        result["gate_traversal_proxy"] = crossing
        result["navigation_checks_passed"] = result["navigation_checks_passed"] and crossing["traversed_proxy"]
    return result


def plan_main(scene):
    adapted = adapt_studio_scene(scene)
    placements = {"actor_1": recommend_placements(scene, 1)[0]}
    options = []
    for obj in adapted["objects"]:
        try:
            _, route = plan_navigation(adapted["scene"], ("actor_1",), actor_id="actor_1",
                target_id=obj["id"], verb="approach", initial_placements=placements,
                affordances=adapted["affordances"])
            options.append((route["schedule"]["horizons"], obj["id"], obj["name"], route))
        except ValueError:
            continue
    if not options:
        raise ValueError("No reachable real object in the supplied main scene")
    windows, target_id, target_name, route = min(options)
    if windows > 4:
        raise ValueError("Shortest real target route exceeds the four-horizon case budget")
    return placements, target_id, target_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8769")
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "review/studio-core/live-results.json")
    parser.add_argument("--archive-dir", type=Path, default=ROOT / ".runtime/core-integration-review")
    parser.add_argument("--scene", type=Path, default=ROOT / "review/scene-integration/live-city.json")
    parser.add_argument("--browser-archive", type=Path, default=ROOT / ".runtime/core-projects/core-1790442406815913000.core.stagezero.npz")
    parser.add_argument("--max-seconds", type=float, default=45.)
    parser.add_argument("--arch-only", action="store_true", help="Append only the corrected floor-supported arch case to the existing report")
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 45:
        parser.error("max-seconds must be in (0,45]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "service_url": args.url,
              "max_gpu_jobs": 10, "geometry_checks": "sampled body spheres / object boxes and 0.65 m pair root discs; not mesh physics",
              "attempts": [], "browser_archive_audit": None}
    if args.arch_only and args.output.exists():
        report = json.loads(args.output.read_text())
        report["additional_arch_run_max_gpu_jobs"] = 4
    previous_jobs = report.get("gpu_jobs_submitted", 0)
    previous_attempts = len(report["attempts"])
    token = args.token_file.read_text().strip()
    client = BoundedClient(args.url, token, timeout=4, job_timeout=40, max_jobs=4 if args.arch_only else 10)

    def write_report():
        report["gpu_jobs_submitted"] = previous_jobs + client.submitted_jobs
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    if not args.arch_only:
        try:
            with CoreStudioSession() as session:
                session.load(args.browser_archive.read_bytes())
                report["browser_archive_audit"] = {"source_archive": str(args.browser_archive),
                    "metrics": measure(session), **archive_result(session, "browser-two-actors", args.archive_dir)}
        except Exception as exc:
            report["browser_archive_audit"] = {"source_archive": str(args.browser_archive), "error": str(exc)}
        write_report()

    # The arch recipe receives an explicit authored placement correction so
    # its entry and exit remain on the unchanged rendered floor platform.
    cases = (["recipe-arch-go-through-floor-supported"] if args.arch_only else
             ["main-city-approach", "recipe-arch-go-through-floor-supported"])
    for name in cases:
        item = {"case": name, "status": "failed"}
        report["attempts"].append(item)
        with CoreStudioSession(client) as session:
            try:
                if name == "main-city-approach":
                    scene = json.loads(args.scene.read_text())
                    placements, target, target_name = plan_main(scene)
                    verb = "approach"
                    item["source_scene"] = str(args.scene)
                else:
                    scene = generate_recipe("Neon research lab", seed=0)
                    scene = validate_scene({**scene, "version": 3, "assets": scene.get("assets", [])})
                    target_object = next(obj for obj in scene["objects"] if obj["kind"] == "arch")
                    original_position = list(target_object["position"])
                    target_object["position"][2] = -1.5
                    scene = validate_scene(scene)
                    item["authored_placement_adjustment"] = {
                        "object_id": target_object["id"], "original_position": original_position,
                        "position": list(target_object["position"]),
                        "reason": "Keep the complete passage entry/exit route on the unchanged rendered platform; original recipe arch exited past its edge"}
                    target, target_name, verb = target_object["id"], target_object["name"], "go_through"
                    placements = {"actor_1": recommend_placements(scene, 1)[0]}
                    item["source_scene"] = "generate_recipe('Neon research lab', seed=0), validated v3 with explicit authored arch Z placement -1.5 m"
                session.start(1, scene, placements)
                item.update(target_id=target, target_name=target_name, verb=verb, initial_placements=placements)
                # Check the complete schedule before authorizing any GPU requests.
                _, planned = plan_navigation(scene, ("actor_1",), actor_id="actor_1", target_id=target,
                    verb=verb, initial_placements=placements)
                if planned["schedule"]["horizons"] > 4:
                    raise ValueError("Recipe route exceeds four-horizon case budget")
                item["route"] = session.navigate("actor_1", target, verb)
                item["playback"] = pump(session, timeout=args.max_seconds)
                state = item["playback"]["state"]
                item["status"] = "complete" if not state["failure"] and not state.get("verification_timeout") else "failed"
                item["metrics"] = measure(session)
                item.update(navigation_metrics(session, target, verb, item["route"]))
                if not all(check["supported"] for check in item["metrics"].get("authored_floor_coverage", [])):
                    item["navigation_checks_passed"] = False
                if not item["navigation_checks_passed"]:
                    item["status"] = "failed"
                item.update(archive_result(session, name, args.archive_dir))
            except Exception as exc:
                item["error"] = str(exc)
                item["metrics"] = measure(session)
                item.update(archive_result(session, name, args.archive_dir))
        write_report()
        print(json.dumps({"case": name, "status": item["status"], "failure": item.get("error") or item.get("playback", {}).get("state", {}).get("failure")}), flush=True)

    if not args.arch_only:
        item = {"case": "transport-failure-and-retry", "status": "failed"}
        report["attempts"].append(item)
        with CoreStudioSession() as session:
            try:
                # Reuse exact browser-generated native motion as the existing good
                # prefix; this case needs only one additional live Core horizon.
                session.load(args.browser_archive.read_bytes())
                prefix = session.timeline_clip()
                session.seek(prefix.frames - 1)
                broken = RealtimeClient("http://127.0.0.1:1", token, timeout=.2, job_timeout=.5)
                session.configure_client(broken)
                session.direct({aid: "Stand calmly in place and maintain your pose." for aid in prefix.actor_ids}, 2)
                item["failure_attempt"] = pump(session, timeout=3.)
                item["failed_without_changing_prefix"] = (session.snapshot()["failure"] is not None
                    and np.array_equal(prefix.positions, session.timeline_clip().positions))
                session.configure_client(client)
                item["retry_accepted"] = session.retry()
                item["retry_playback"] = pump(session, timeout=args.max_seconds)
                result = session.timeline_clip()
                item["prefix_preserved_after_retry"] = (np.array_equal(prefix.positions, result.positions[:, :prefix.frames])
                    and np.array_equal(prefix.rotations, result.rotations[:, :prefix.frames])
                    and np.array_equal(prefix.native_features, result.native_features[:, :prefix.frames]))
                item["status"] = "complete" if (item["failed_without_changing_prefix"] and item["retry_accepted"]
                    and item["prefix_preserved_after_retry"] and result.frames == prefix.frames + 40) else "failed"
                item["metrics"] = measure(session)
                item.update(archive_result(session, "transport-retry", args.archive_dir))
            except Exception as exc:
                item["error"] = str(exc)
    write_report()
    print(json.dumps({"report": str(args.output), "gpu_jobs_submitted": client.submitted_jobs,
                      "cases": [{"case": x["case"], "status": x["status"]} for x in report["attempts"]]}), flush=True)
    return 0 if all(x["status"] == "complete" for x in report["attempts"][previous_attempts:]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
