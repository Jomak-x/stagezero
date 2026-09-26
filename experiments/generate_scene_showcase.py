#!/usr/bin/env python3
"""Generate a complete measured scene through the real realtime motion service.

The research pair is fetched once before its entry transition. Its untouched
relative acting is rigidly placed in the Core world, then the exact cached
sample supplies both action and release. Every returned model chunk is
committed through RealtimeDirector before a project is saved. No joint frames
are authored as output or substituted for failed service work.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interaction_metrics import (continuity, gate_traversal, hand_contact,
                                 object_contact, pair_separation)
from interaction_scene import local_axes, passage_for, scene_objects
from interaction_scene_collision import scene_collision
from realtime_client import RealtimeClient
from realtime_clip import CanonicalClip
from realtime_director import RealtimeDirector, StageSpec
from scene_beats import FIGHT_PAIR_PROMPT, SCENARIOS, build_scene


def _job_id(scenario: str, beat_id: str) -> str:
    return f"{scenario}-{beat_id}-{uuid.uuid4().hex[:12]}"


def _stage_kind(beat: dict, actor_count: int) -> str:
    kind = beat["kind"]
    if kind == "paired":
        return "paired_action"
    if kind == "continuation" or (kind == "release" and actor_count == 1):
        return "action"
    return kind


def _stage_specs(plan: dict) -> list[StageSpec]:
    actor_count = len(plan["actor_ids"])
    stages = []
    for beat in plan["beats"]:
        metadata = {**beat["metadata"], "beat_id": beat["id"],
                    "scene_start_frame": beat["start_frame"]}
        stages.append(StageSpec(beat["prompt"], kind=_stage_kind(beat, actor_count),
                                frames=beat["frames"], source=beat["source"],
                                actor_prompts={actor_id: value["prompt"]
                                               for actor_id, value in beat["actors"].items()},
                                metadata=metadata))
    return stages


def _stack(clips: list[CanonicalClip]) -> CanonicalClip:
    if not clips or any(c.actor_ids != clips[0].actor_ids or c.source != clips[0].source
                        for c in clips):
        raise ValueError("Cannot stack mismatched model chunks")
    native = None if any(c.native_features is None for c in clips) else np.concatenate(
        [c.native_features for c in clips], axis=1)
    return CanonicalClip(np.concatenate([c.positions for c in clips], axis=1),
                         np.concatenate([c.rotations for c in clips], axis=1), 20,
                         clips[0].actor_ids, clips[0].source,
                         {"source_chunks": [c.metadata for c in clips]}, native)


def place_pair(pair: CanonicalClip, anchors_xz: np.ndarray) -> tuple[CanonicalClip, dict]:
    """Apply one rigid world transform to both people and all their joints."""
    if pair.source != "intergen" or len(pair.actor_ids) != 2:
        raise ValueError("Expected a canonical InterGen pair")
    anchors = np.asarray(anchors_xz, dtype=np.float64)
    if anchors.shape != (2, 2) or not np.isfinite(anchors).all():
        raise ValueError("Pair anchors must be two finite XZ roots")
    source = pair.positions[:, 0, 0][:, (0, 2)].astype(np.float64)
    source_delta, target_delta = source[1] - source[0], anchors[1] - anchors[0]
    if min(np.linalg.norm(source_delta), np.linalg.norm(target_delta)) < .45:
        raise ValueError("Paired source or approach anchors overlap; cannot place pair")
    source_angle = math.atan2(source_delta[1], source_delta[0])
    target_angle = math.atan2(target_delta[1], target_delta[0])
    yaw = source_angle - target_angle
    c, s = math.cos(yaw), math.sin(yaw)
    matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    midpoint_source = np.mean(source, axis=0)
    midpoint_target = np.mean(anchors, axis=0)
    translation = midpoint_target - np.array([c * midpoint_source[0] + s * midpoint_source[1],
                                               -s * midpoint_source[0] + c * midpoint_source[1]])
    positions = pair.positions.astype(np.float64) @ matrix.T
    positions[..., 0] += translation[0]
    positions[..., 2] += translation[1]
    rotations = matrix @ pair.rotations.astype(np.float64)
    placed = CanonicalClip(positions.astype(np.float32), rotations.astype(np.float32), 20,
                           pair.actor_ids, "intergen",
                           {**pair.metadata, "world_placement": {"yaw_radians": yaw,
                                                                  "translation_xz_m": translation.tolist(),
                                                                  "rigid_shared_pair": True}})
    mismatch = np.linalg.norm(placed.positions[:, 0, 0][:, (0, 2)] - anchors, axis=1)
    return placed, {"yaw_radians": yaw, "translation_xz_m": translation.tolist(),
                    "entry_root_mismatch_m": mismatch.tolist(),
                    "max_entry_root_mismatch_m": float(mismatch.max())}


def _repeat_pair_entry(pair: CanonicalClip, frames: int) -> dict:
    """Native full-body transition goal, formed from the actual pair entry pose."""
    return {"positions": np.repeat(pair.positions[:, :1], frames, axis=1).tolist(),
            "rotations": np.repeat(pair.rotations[:, :1], frames, axis=1).tolist()}


def _history(request) -> dict | None:
    clip = request.history
    if clip is None:
        return None
    if clip.native_features is not None:
        return {"native_features": clip.native_features.tolist()}
    return {"positions": clip.positions.tolist(), "rotations": clip.rotations.tolist()}


def _local_targets(beat: dict, actor_ids=None, history: CanonicalClip | None = None) -> dict:
    start = beat["start_frame"]
    result = {}
    for actor_index, actor_id in enumerate(actor_ids or beat["actors"]):
        targets = beat["actors"][actor_id]["root_targets"]
        if not targets:
            continue
        delta = np.zeros(2, dtype=np.float64)
        if history is not None:
            first = np.asarray(targets[0]["position_xz"], dtype=np.float64)
            if len(targets) > 1:
                second = np.asarray(targets[1]["position_xz"], dtype=np.float64)
                fraction = (targets[0]["frame"] - start) / (targets[1]["frame"] - targets[0]["frame"])
                planned_start = first - fraction * (second - first)
            else:
                planned_start = first
            actual_start = history.positions[actor_index, -1, 0, [0, 2]].astype(np.float64)
            delta = actual_start - planned_start
        result[actor_id] = [{**target, "frame": target["frame"] - start,
                             "position_xz": (np.asarray(target["position_xz"], dtype=np.float64) +
                                             delta).round(5).tolist()}
                            for target in targets]
    return result


class PlaybackMonitor:
    """Drive the director from a wall clock while model calls run in another thread."""

    def __init__(self, director: RealtimeDirector, *, realtime: bool):
        self.director = director
        self.realtime = realtime
        self.started = time.monotonic()
        self.sim_time = 0.0
        self.first_pose_seconds = None
        self.buffering_seconds = 0.0
        self.buffering_events = 0
        self._buffering_since = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        director.play(now=self.started if realtime else 0.)
        if realtime:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self):
        while not self._stop.wait(.02):
            self.pulse()

    def pulse(self):
        with self._lock:
            now = time.monotonic()
            snapshot = self.director.tick(now=now if self.realtime else self.sim_time)
            if snapshot["generated"] and self.first_pose_seconds is None:
                self.first_pose_seconds = now - self.started
            if snapshot["generated"] and snapshot["phase"] == "buffering":
                if self._buffering_since is None:
                    self._buffering_since = now
                    self.buffering_events += 1
            elif self._buffering_since is not None:
                self.buffering_seconds += now - self._buffering_since
                self._buffering_since = None
            return snapshot

    def advance_offline(self):
        self.sim_time += 2.0
        self.pulse()

    def finish(self, *, wait_for_playback: bool = True) -> dict:
        if self.realtime and wait_for_playback:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                snapshot = self.pulse()
                if snapshot["total_frames"] and snapshot["frame"] == snapshot["total_frames"] - 1:
                    break
                time.sleep(.02)
            else:
                raise TimeoutError("Playback did not reach the final generated frame")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        snapshot = self.pulse()
        with self._lock:
            if self._buffering_since is not None:
                self.buffering_seconds += time.monotonic() - self._buffering_since
                self._buffering_since = None
            return {"clock": "wall" if self.realtime else "simulated_offline",
                    "startup_seconds": self.first_pose_seconds,
                    "buffer_underrun_seconds": self.buffering_seconds,
                    "buffer_underrun_events": self.buffering_events,
                    "elapsed_seconds": time.monotonic() - self.started,
                    "final_playhead": snapshot["frame"]}


def _claim(director: RealtimeDirector, monitor: PlaybackMonitor, cancelled=lambda: False):
    deadline = time.monotonic() + (30 if monitor.realtime else 1)
    while time.monotonic() < deadline:
        if cancelled():
            raise RuntimeError("Scene generation cancelled")
        request = director.claim_request()
        if request is not None:
            return request
        if director.total_frames == 0:
            break
        if monitor.realtime:
            time.sleep(.02)
        else:
            monitor.advance_offline()
    raise RuntimeError("RealtimeDirector did not grant a synchronized chunk request")


def _commit(director: RealtimeDirector, clips: list[CanonicalClip], monitor: PlaybackMonitor,
            current_request=None, cancelled=lambda: False) -> None:
    for index, clip in enumerate(clips):
        if cancelled():
            raise RuntimeError("Scene generation cancelled")
        request = current_request if index == 0 and current_request is not None else _claim(
            director, monitor, cancelled)
        if clip.source == "intergen":
            clip = CanonicalClip(clip.positions, clip.rotations, clip.fps, clip.actor_ids,
                                 clip.source, {**clip.metadata, **request.metadata},
                                 clip.native_features)
        if not director.complete(request.request_id, clip):
            raise RuntimeError("RealtimeDirector rejected a model chunk")
        if monitor.realtime:
            monitor.pulse()
        else:
            monitor.advance_offline()


def _service_body(plan: dict, beat: dict, request, *, pair: CanonicalClip | None) -> dict:
    body = {"request_id": request.request_id, "stage_kind": beat["kind"],
            "frames": beat["frames"], "prompt": beat["prompt"],
            "actor_ids": plan["actor_ids"], "seed": plan["seed"],
            "actor_prompts": {aid: actor["prompt"] for aid, actor in beat["actors"].items()},
            "root_targets": _local_targets(beat, plan["actor_ids"], request.history)}
    if beat["kind"] in ("continuation", "release", "action") or request.history is not None:
        history = _history(request)
        if history is None:
            raise RuntimeError("Core continuation requires committed model history")
        body["history"] = history
        body["stage_kind"] = "continuation"
    if beat["kind"] == "transition":
        if pair is None:
            raise RuntimeError("Transition requires the prefetched paired clip")
        body["stage_kind"] = "transition"
        body["target"] = _repeat_pair_entry(pair, beat["frames"])
    if request.history is None:
        body["initial_placements"] = plan["initial_placements"]
    if beat["kind"] == "action" and "native_hand_target" in beat["metadata"]:
        target = beat["metadata"]["native_hand_target"]
        body["hand_target"] = {"source_job_id": target["source_job_id"],
                               "source_frame": target["source_frame"],
                               "hand": target["hand"],
                               "position_xyz": target["position_xyz"],
                               "frames": [frame - beat["start_frame"] for frame in target["global_frames"]]}
    return body


def _pair_clip(plan: dict, client: RealtimeClient, jobs: list[dict],
               cancelled=lambda: False) -> CanonicalClip | None:
    paired = next((beat for beat in plan["beats"] if beat["source"] == "intergen"
                   and beat["kind"] == "paired"), None)
    if paired is None:
        return None
    meta = paired["metadata"]
    body = {"request_id": _job_id(plan["name"], "prefetch_pair"), "stage_kind": "paired",
            "frames": meta["source_total_frames"], "prompt": paired["prompt"],
            "actor_ids": plan["actor_ids"], "seed": plan["seed"],
            "pair_sequence_id": meta["pair_sequence_id"], "source_start_frame": 0,
            "source_total_frames": meta["source_total_frames"]}
    started = time.perf_counter()
    arrival = []
    clips = client.wait(body, on_chunk=lambda clip: arrival.append(time.perf_counter() - started),
                        cancelled=cancelled)
    jobs.append({"beat_id": "pair_prefetch", "request": body,
                 "chunk_arrival_seconds": arrival, "elapsed_seconds": time.perf_counter() - started,
                 "service_result": client.status(body["request_id"])})
    return _stack(clips)


def _pair_slices(pair: CanonicalClip, beat: dict) -> list[CanonicalClip]:
    start = beat["metadata"]["source_start_frame"]
    stop = start + beat["frames"]
    return [pair.slice_frames(frame, frame + 40) for frame in range(start, stop, 40)]


def _measure(plan: dict, clip: CanonicalClip, placement: dict | None) -> dict:
    ids = {actor_id: i for i, actor_id in enumerate(plan["actor_ids"])}
    measured = {"shape": list(clip.positions.shape), "fps": clip.fps,
                "placement": placement, "beats": [], "seams": {}}
    boundaries = [beat["start_frame"] for beat in plan["beats"][1:]]
    for aid, index in ids.items():
        measured["seams"][aid] = continuity(clip.positions[index], skeleton="core27",
                                              fps=20, horizon_boundaries=tuple(boundaries))
    all_boundaries = sorted(set(boundaries) | set(range(40, clip.frames, 40)))
    boundary_rows = []
    for frame in all_boundaries:
        delta = np.linalg.norm(clip.positions[:, frame] - clip.positions[:, frame - 1], axis=-1)
        row = {"frame_after": frame, "beat_boundary": frame in boundaries,
               "max_root_step_m": float(delta[:, 0].max()),
               "max_actor_mean_joint_step_m": float(delta.mean(axis=1).max()),
               "max_joint_step_m": float(delta.max())}
        row["passed"] = (row["max_root_step_m"] <= .15 and
                         row["max_actor_mean_joint_step_m"] <= .15 and
                         row["max_joint_step_m"] <= .4)
        boundary_rows.append(row)
    measured["stage_boundaries"] = {
        "checks": boundary_rows,
        "thresholds_m": {"root": .15, "mean_joint": .15, "max_joint": .4},
        "passed": all(row["passed"] for row in boundary_rows),
    }
    for beat in plan["beats"]:
        start, end = beat["start_frame"], beat["end_frame"]
        item = {"beat_id": beat["id"], "start_frame": start, "end_frame": end,
                "gates": [], "passed": True}
        for gate in beat["quality_gates"]:
            metric = gate["metric"]
            value = None
            passed = False
            if metric == "gate_traversal":
                obj = next(o for o in scene_objects(plan["scene"]) if o.id == gate["object_id"])
                passage = passage_for(obj, None, actor_height_m=1.65)
                _, normal = local_axes(passage.yaw_degrees)
                value = gate_traversal(clip.positions[ids[gate["actor_id"]], start:end],
                                       skeleton="core27", center=(obj.x, obj.y, obj.z),
                                       normal_xz=normal, opening_width_m=passage.width_m,
                                       min_y=0, max_y=passage.height_m)
                passed = bool(value["traversed_proxy"])
            elif metric == "scene_collision":
                value = scene_collision(clip.positions[ids[gate["actor_id"]], start:end],
                                        "core27", plan["scene"])
                passed = value["total_collision_frames"] <= gate["max_overlap_frames"]
            elif metric == "pair_separation":
                value = pair_separation(clip.positions[0, start:end], clip.positions[1, start:end],
                                        skeleton_a="core27", skeleton_b="core27")
                passed = value["min_root_separation_xz_m"] >= gate["min_m"]
            elif metric == "continuity":
                steps = [float(np.linalg.norm(clip.positions[i, start, 0] -
                                              clip.positions[i, start - 1, 0])) for i in ids.values()]
                value = {"boundary_root_steps_m": steps}
                passed = max(steps) <= gate["max_boundary_jump_m"]
            elif metric == "hand_contact":
                candidates = [hand_contact(clip.positions[0, start:end], clip.positions[1, start:end],
                                           skeleton_a="core27", skeleton_b="core27",
                                           hand_a=left, hand_b=right, fps=20,
                                           tolerance_m=gate["tolerance_m"],
                                           minimum_duration_s=gate["minimum_duration_s"])
                              for left in ("left", "right") for right in ("left", "right")]
                value = max(candidates, key=lambda x: x["longest_near_run_frames"])
                passed = bool(value["contact_proxy"])
            elif metric == "hand_release":
                a = clip.positions[0, end - 10:end]
                b = clip.positions[1, end - 10:end]
                distances = [float(np.linalg.norm(a[:, ia] - b[:, ib], axis=1).mean())
                             for ia in (11, 17) for ib in (11, 17)]
                value = {"mean_final_hand_gap_m": min(distances)}
                passed = value["mean_final_hand_gap_m"] >= .25
            elif metric == "dodge_lateral":
                roots = clip.positions[:, start:end, 0][:, :, (0, 2)]
                opening = roots[1, 0] - roots[0, 0]
                opening /= max(float(np.linalg.norm(opening)), 1e-9)
                normal = np.array([-opening[1], opening[0]])
                lateral = (roots[1] - roots[0]) @ normal
                value = {"relative_lateral_range_m": float(np.ptp(lateral))}
                passed = value["relative_lateral_range_m"] >= gate["min_m"]
            elif metric == "block_guard":
                a = clip.positions[0, start:end]
                b = clip.positions[1, start:end]
                candidates = [np.linalg.norm(a[:, ai] - b[:, bi], axis=1)
                              for ai, bi in ((11, 9), (11, 15), (17, 9), (17, 15),
                                             (9, 11), (15, 11), (9, 17), (15, 17))]
                gaps = np.min(np.stack(candidates), axis=0)
                near = gaps <= gate["tolerance_m"]
                longest = current = 0
                for hit in near:
                    current = current + 1 if hit else 0
                    longest = max(longest, current)
                value = {"min_hand_forearm_gap_m": float(gaps.min()),
                         "longest_near_run_s": longest / 20}
                passed = value["longest_near_run_s"] >= gate["minimum_duration_s"]
            elif metric == "push_reaction":
                observed_end = min(start + beat["metadata"].get("source_total_frames", end - start),
                                   clip.frames)
                roots = clip.positions[:, start:observed_end, 0][:, :, (0, 2)]
                distance = np.linalg.norm(roots[0] - roots[1], axis=1)
                value = {"first_third_mean_root_gap_m": float(distance[:len(distance)//3].mean()),
                         "last_third_mean_root_gap_m": float(distance[-len(distance)//3:].mean())}
                passed = value["last_third_mean_root_gap_m"] > value["first_third_mean_root_gap_m"] + .15
            elif metric == "object_contact":
                xyz = next(b["metadata"]["native_hand_target"]["position_xyz"]
                           for b in plan["beats"] if b["id"] == beat["id"])
                value = object_contact(clip.positions[ids[gate["actor_id"]], start:end],
                                       skeleton="core27", object_xyz=xyz, hand=gate["hand"],
                                       fps=20, tolerance_m=gate["tolerance_m"],
                                       minimum_duration_s=gate["minimum_duration_s"])
                passed = bool(value["contact_proxy"])
            else:
                raise ValueError(f"No evaluator for quality gate {metric}")
            item["gates"].append({"requirement": gate, "measurement": value, "passed": passed})
            item["passed"] = item["passed"] and passed
        measured["beats"].append(item)
    measured["complete"] = (clip.frames == plan["total_frames"] and
                            measured["stage_boundaries"]["passed"] and
                            all(item["passed"] for item in measured["beats"]))
    return measured


def run_scene(plan: dict, client: RealtimeClient, output_dir: Path, *, realtime: bool = True,
              on_director=None, cancelled=lambda: False) -> dict:
    """Run a complete scene or raise; failed work is never labeled completed."""
    if not isinstance(plan, dict) or plan.get("name") not in SCENARIOS:
        raise ValueError("Expected a scene_beats plan")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    research = any(beat["source"] == "intergen" for beat in plan["beats"])
    director = RealtimeDirector(
        plan["actor_ids"], mode="research" if research else "production",
        project_metadata={"scene": plan["scene"], "scenario": plan["name"],
                          "seed": plan["seed"], "research_preview": research,
                          "initial_placements": plan["initial_placements"]})
    director.queue_sequence(_stage_specs(plan))
    if on_director is not None:
        on_director(director)
    monitor = PlaybackMonitor(director, realtime=realtime)
    jobs: list[dict] = []
    pair = None
    placed_pair = None
    placement = None
    prior_jobs: dict[str, str] = {}
    prior_clips: dict[str, CanonicalClip] = {}
    try:
        pair = _pair_clip(plan, client, jobs, cancelled=cancelled)
        for beat in plan["beats"]:
            if cancelled():
                raise RuntimeError("Scene generation cancelled")
            if beat["kind"] == "transition" and pair is not None:
                prior = director.timeline_clip()
                if prior is None:
                    raise RuntimeError("Paired transition requires committed approach motion")
                anchors = prior.positions[:, -1, 0][:, (0, 2)]
                placed_pair, placement = place_pair(pair, anchors)
                if placement["max_entry_root_mismatch_m"] > .6:
                    raise RuntimeError("Paired entry is too far from the generated approach anchors")
            if beat["source"] == "intergen":
                if placed_pair is None:
                    raise RuntimeError("Paired stage has no placed cached model sample")
                clips = _pair_slices(placed_pair, beat)
                _commit(director, clips, monitor, cancelled=cancelled)
                continue
            request = _claim(director, monitor, cancelled)
            if request.start_frame != beat["start_frame"]:
                raise RuntimeError("Director stage start does not match the scene plan")
            if beat["kind"] == "action" and "native_hand_target" in beat["metadata"]:
                target = beat["metadata"]["native_hand_target"]
                source_id = prior_jobs.get(target["source_beat"])
                source_clip = prior_clips.get(target["source_beat"])
                if source_id is None or source_clip is None:
                    raise RuntimeError("Native hand reference source job is missing")
                hand_xyz = source_clip.positions[0, :, 10].astype(np.float64)
                delta = hand_xyz - np.asarray(target["position_xyz"], dtype=np.float64)
                distance = np.linalg.norm(delta, axis=1)
                reachable = np.flatnonzero((distance <= .45) & (np.abs(delta[:, 1]) <= .20))
                if not len(reachable):
                    raise RuntimeError("No coherent Core hand pose reaches the console target")
                chosen = int(reachable[np.argmin(distance[reachable])])
                target["source_job_id"] = source_id
                target["source_frame"] = chosen
                target["reference_distance_m"] = float(distance[chosen])
            body = _service_body(plan, beat, request, pair=placed_pair)
            started = time.perf_counter()
            arrival = []
            committed = []

            def on_chunk(clip):
                arrival.append(time.perf_counter() - started)
                _commit(director, [clip], monitor,
                        current_request=request if not committed else None,
                        cancelled=cancelled)
                committed.append(clip)

            clips = client.wait(body, on_chunk=on_chunk, cancelled=cancelled)
            if len(clips) != beat["frames"] // 40:
                raise RuntimeError("Service omitted a synchronized chunk")
            if len(committed) != len(clips):
                raise RuntimeError("A streamed service chunk was not committed")
            service_result = client.status(body["request_id"])
            if beat["kind"] == "action" and "native_hand_target" in beat["metadata"]:
                native = (service_result.get("metadata") or {}).get("native_hand_target")
                if (not isinstance(native, dict) or
                        native.get("constraint") != "EndEffectorConstraintSet" or
                        native.get("source_job_id") != body["hand_target"]["source_job_id"]):
                    raise RuntimeError("Service did not attest native hand conditioning")
            prior_jobs[beat["id"]] = body["request_id"]
            prior_clips[beat["id"]] = _stack(clips)
            jobs.append({"beat_id": beat["id"], "request": {k: v for k, v in body.items()
                                                            if k not in ("history", "target")},
                         "chunk_arrival_seconds": arrival,
                         "elapsed_seconds": time.perf_counter() - started,
                         "service_result": service_result})
        timeline = director.timeline_clip()
        if timeline is None or timeline.frames != plan["total_frames"]:
            raise RuntimeError("Scene has an incomplete committed timeline")
        playback = monitor.finish()
        playback["pair_prefetch_seconds"] = next(
            (job["elapsed_seconds"] for job in jobs if job["beat_id"] == "pair_prefetch"), 0.)
        playback["first_pose_network_seconds"] = next(
            (job["chunk_arrival_seconds"][0] for job in jobs
             if job["beat_id"] != "pair_prefetch" and job["chunk_arrival_seconds"]), None)
        playback["thresholds"] = {"startup_seconds": 5., "buffer_underrun_seconds": .25}
        playback["realtime_passed"] = (playback["clock"] == "wall" and
                                       playback["startup_seconds"] is not None and
                                       playback["startup_seconds"] <= 5. and
                                       playback["buffer_underrun_seconds"] <= .25)
        metrics = _measure(plan, timeline, placement)
        accepted = metrics["complete"] and playback["realtime_passed"]
        status = ("complete" if accepted else
                  "quality_failed" if not metrics["complete"] else
                  "realtime_failed" if realtime else "offline_assembled")
        report = {"scenario": plan["name"], "status": status, "accepted": accepted,
                  "plan": plan, "jobs": jobs, "metrics": metrics, "playback": playback,
                  "service_health": client.health(),
                  "project_file": f"{plan['name']}.stagezero-realtime.npz",
                  "raw_file": f"{plan['name']}.raw.npz",
                  "pair_source_file": (f"{plan['name']}.pair-source.npz" if pair is not None else None)}
        (output_dir / f"{plan['name']}.stagezero-realtime.npz").write_bytes(director.save_project())
        if pair is not None:
            np.savez_compressed(output_dir / f"{plan['name']}.pair-source.npz",
                                positions=pair.positions, rotations=pair.rotations,
                                metadata=np.array(json.dumps(pair.metadata)))
        np.savez_compressed(output_dir / f"{plan['name']}.raw.npz",
                            positions=timeline.positions, rotations=timeline.rotations,
                            metadata=np.array(json.dumps({"fps": 20, "actor_ids": plan["actor_ids"],
                                                          "source": "RealtimeDirector committed model chunks"})))
        (output_dir / f"{plan['name']}.report.json").write_text(json.dumps(report, indent=2) + "\n")
        if not metrics["complete"]:
            raise RuntimeError("Generated all scene frames, but required quality gates failed; see report")
        if not playback["realtime_passed"] and realtime:
            raise RuntimeError("Generated all scene frames, but realtime playback gates failed; see report")
        return report
    except Exception as exc:
        inflight_id = director.snapshot()["inflight_request_id"]
        if inflight_id is not None:
            director.fail(inflight_id, exc)
        director.pause()
        monitor.finish(wait_for_playback=False)
        failure = {"scenario": plan["name"], "status": "failed", "error": str(exc),
                   "committed_frames": director.total_frames, "jobs": jobs, "plan": plan}
        (output_dir / f"{plan['name']}.failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        if director.total_frames:
            (output_dir / f"{plan['name']}.partial.stagezero-realtime.npz").write_bytes(
                director.save_project())
        raise


FIGHT_PROMPTS = (
    "A staged sparring sequence: feint, sidestep dodge, guarded block, light push, then release and step apart.",
    "Two stunt performers spar. One dodges a punch and blocks another. The other pushes them backward, and both step apart.",
    FIGHT_PAIR_PROMPT,
)


def screen_fight_pairs(client: RealtimeClient, output_dir: Path,
                       seeds=(37, 38, 39, 40, 41, 42, 43, 44)) -> list[dict]:
    """Measure real paired samples before another complete 30-second scene run."""
    output_dir = Path(output_dir) / "diagnostics" / "fight-screening"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for prompt_index, prompt in enumerate(FIGHT_PROMPTS):
        for seed in seeds:
            plan = build_scene("staged_fight", seed)
            for beat in plan["beats"]:
                if beat["source"] == "intergen":
                    beat["prompt"] = prompt
                    beat["metadata"]["pair_sequence_id"] = f"fight_screen_{prompt_index}_{seed}"
            jobs = []
            pair = _pair_clip(plan, client, jobs)
            if pair is None or pair.frames != 120:
                raise RuntimeError("Screening did not return a complete 120-frame pair")
            a, b = pair.positions
            roots = pair.positions[:, :, 0][:, :, (0, 2)]
            separation = np.linalg.norm(roots[0] - roots[1], axis=1)
            opening = roots[1, 0] - roots[0, 0]
            opening /= max(float(np.linalg.norm(opening)), 1e-9)
            normal = np.array([-opening[1], opening[0]])
            lateral = (roots[1, :80] - roots[0, :80]) @ normal
            guard = np.min(np.stack([np.linalg.norm(a[:80, ai] - b[:80, bi], axis=1)
                                     for ai, bi in ((11, 9), (11, 15), (17, 9), (17, 15),
                                                    (9, 11), (15, 11), (9, 17), (15, 17))]), axis=0)
            near = guard <= .22
            longest = current = 0
            for hit in near:
                current = current + 1 if hit else 0
                longest = max(longest, current)
            release = min(float(np.linalg.norm(a[-10:, ai] - b[-10:, bi], axis=1).mean())
                          for ai in (11, 17) for bi in (11, 17))
            first_gap, final_gap = float(separation[:40].mean()), float(separation[-40:].mean())
            seam_steps = []
            for frame in (40, 80):
                delta = np.linalg.norm(pair.positions[:, frame] - pair.positions[:, frame - 1], axis=-1)
                seam_steps.append({"frame": frame, "root_m": float(delta[:, 0].max()),
                                   "mean_joint_m": float(delta.mean(axis=1).max()),
                                   "max_joint_m": float(delta.max())})
            checks = {"separation": bool(separation.min() >= .42),
                      "dodge": bool(np.ptp(lateral) >= .2),
                      "block": bool(longest >= 2),
                      "push": bool(final_gap > first_gap + .15),
                      "release": bool(release >= .25),
                      "seams": all(s["root_m"] <= .15 and s["mean_joint_m"] <= .15 and
                                   s["max_joint_m"] <= .4 for s in seam_steps)}
            row = {"seed": seed, "prompt_index": prompt_index, "prompt": prompt,
                   "pair_sequence_id": f"fight_screen_{prompt_index}_{seed}",
                   "checks": checks, "passed": all(checks.values()),
                   "metrics": {"min_root_separation_m": float(separation.min()),
                               "lateral_dodge_range_m": float(np.ptp(lateral)),
                               "min_hand_forearm_gap_m": float(guard.min()),
                               "block_near_run_s": longest / 20,
                               "first_third_root_gap_m": first_gap,
                               "last_third_root_gap_m": final_gap,
                               "release_hand_gap_m": release,
                               "seams": seam_steps},
                   "service_result": jobs[0]["service_result"]}
            results.append(row)
            np.savez_compressed(output_dir / f"prompt{prompt_index}_seed{seed}.npz",
                                positions=pair.positions, rotations=pair.rotations,
                                metadata=np.array(json.dumps({"prompt": prompt, "seed": seed,
                                                              "service_result": row["service_result"]})))
            (output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({"prompt_index": prompt_index, "seed": seed,
                              "checks": checks, "metrics": row["metrics"]}), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the verified default seed for this scenario")
    parser.add_argument("--base-url", default="http://127.0.0.1:8769")
    parser.add_argument("--token-file", type=Path,
                        default=Path("/Users/jakob/Desktop/Shellhacks/.runtime/api-token"))
    parser.add_argument("--output-dir", type=Path, default=Path(".runtime/realtime-showcase"))
    parser.add_argument("--dry-run", action="store_true", help="Validate and print only the CPU scene plan")
    parser.add_argument("--offline-assembly", action="store_true",
                        help="Use a simulated playback clock for CPU-only integration tests")
    parser.add_argument("--screen-fight-pairs", action="store_true",
                        help="Measure bounded real InterGen research candidates, without a full scene")
    args = parser.parse_args()
    plan = build_scene(args.scenario, args.seed)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    token = args.token_file.read_text().strip()
    client = RealtimeClient(args.base_url, token, job_timeout=600)
    if args.screen_fight_pairs:
        if args.scenario != "staged_fight":
            parser.error("--screen-fight-pairs requires --scenario staged_fight")
        rows = screen_fight_pairs(client, args.output_dir)
        print(json.dumps({"screened": len(rows), "passed": sum(row["passed"] for row in rows),
                          "summary": str(args.output_dir / "diagnostics/fight-screening/summary.json")}))
        return
    report = run_scene(plan, client, args.output_dir, realtime=not args.offline_assembly)
    print(json.dumps({"scenario": report["scenario"], "status": report["status"],
                      "frames": report["metrics"]["shape"][1],
                      "report": str(args.output_dir / f"{args.scenario}.report.json")}))


if __name__ == "__main__":
    main()
