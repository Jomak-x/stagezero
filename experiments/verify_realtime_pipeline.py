"""Independent acceptance probe for the real buffered Core27 service.

Run only against a real service. CPU protocol fakes belong in
``tests/test_realtime_acceptance.py`` and are labelled there. This script never
substitutes synthetic motion for a missing endpoint or failed model request.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interaction_metrics import (continuity, floor_motion, gate_traversal,
                                 hand_contact, object_contact, pair_separation)
from interaction_runtime import MODEL_NAME
from interaction_scene import local_axes, passage_for, scene_objects
from interaction_scene_collision import scene_collision
from realtime_director import RealtimeDirector
from scene_beats import SCENARIOS


FPS = 20
FRAMES_PER_CHUNK = 40
JOINTS = 27
FEATURES = 330
ROOT_SEAM_LIMIT_M = .15
MEAN_JOINT_SEAM_LIMIT_M = .15
MAX_JOINT_SEAM_LIMIT_M = .4
FIRST_CHUNK_LIMIT_S = 2.0
MIN_REALTIME_FACTOR = 1.0
TERMINAL = {"completed", "complete", "failed", "cancelled", "canceled"}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo))


def timing_summary(values: list[float]) -> dict:
    return {"samples": len(values), "p50_s": percentile(values, .5),
            "p95_s": percentile(values, .95), "max_s": max(values) if values else None}


def assert_chunk(arrays: dict[str, np.ndarray], *, actors: int, frames: int = FRAMES_PER_CHUNK,
                 native_required: bool = True) -> None:
    shapes = {"positions": (actors, frames, JOINTS, 3),
              "rotations": (actors, frames, JOINTS, 3, 3)}
    if native_required or "native_features" in arrays:
        shapes["native_features"] = (actors, frames, FEATURES)
    for name, shape in shapes.items():
        if name not in arrays or arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise AssertionError(f"{name} must contain finite Core27 data of shape {shape}")
    rotations = arrays["rotations"]
    if not np.allclose(rotations @ np.swapaxes(rotations, -1, -2), np.eye(3), atol=.03):
        raise AssertionError("rotations are not orthonormal")
    if not np.allclose(np.linalg.det(rotations), 1, atol=.03):
        raise AssertionError("rotations have invalid determinants")


def load_chunk(blob: bytes, actors: int, *, native_required: bool = True) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        keys = ("positions", "rotations", "native_features") if native_required else ("positions", "rotations")
        arrays = {name: archive[name].copy() for name in keys}
        metadata = json.loads(str(archive["metadata"].item()))
    assert_chunk(arrays, actors=actors, native_required=native_required)
    if not isinstance(metadata, dict):
        raise AssertionError("chunk metadata must be an object")
    return arrays, metadata


def exact_disk_roundtrip(blob: bytes, output: Path, actors: int, *, native_required: bool = True) -> bool:
    """Compare arrays from wire bytes with arrays reopened from a saved file."""
    before, before_meta = load_chunk(blob, actors, native_required=native_required)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(blob)
    after, after_meta = load_chunk(output.read_bytes(), actors, native_required=native_required)
    return before_meta == after_meta and all(
        np.array_equal(before[name], after[name]) for name in before
    )


def seam_report(positions: np.ndarray, boundaries: tuple[int, ...],
                rotations: np.ndarray | None = None) -> dict:
    """Measure all actor seams in world metres; no smoothing hides a jump."""
    p = np.asarray(positions)
    if p.ndim != 4 or p.shape[2:] != (JOINTS, 3):
        raise AssertionError("positions must be [actors, frames, 27, 3]")
    r = None if rotations is None else np.asarray(rotations)
    if r is not None and r.shape != (p.shape[0], p.shape[1], JOINTS, 3, 3):
        raise AssertionError("rotations must match the actor/frame timeline")
    measured = []
    for actor in range(p.shape[0]):
        summary = continuity(p[actor], skeleton="core27", fps=FPS,
                             horizon_boundaries=boundaries)
        for seam in summary["horizon_seams"]:
            k = seam["frame_after"]
            joint_steps = np.linalg.norm(p[actor, k] - p[actor, k - 1], axis=-1)
            measured.append({**seam, "actor_index": actor,
                             "max_joint_step_m": float(joint_steps.max()),
                             "pass_root": seam["root_step_m"] <= ROOT_SEAM_LIMIT_M,
                             "pass_mean_joint": seam["mean_joint_step_m"] <= MEAN_JOINT_SEAM_LIMIT_M,
                             "pass_max_joint": float(joint_steps.max()) <= MAX_JOINT_SEAM_LIMIT_M})
            if r is not None:
                relative = np.swapaxes(r[actor, k - 1], -1, -2) @ r[actor, k]
                cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
                angle = np.degrees(np.arccos(cosine))
                measured[-1].update(max_joint_rotation_deg=float(angle.max()),
                                    mean_joint_rotation_deg=float(angle.mean()),
                                    root_angular_velocity_deg_s=float(angle[0] * FPS))
    return {"seams": measured,
            "passed": all(row["pass_root"] and row["pass_mean_joint"] and row["pass_max_joint"]
                          for row in measured)}


def playback_clock(arrival_offsets: list[float], *, chunk_frames: int = FRAMES_PER_CHUNK) -> dict:
    """Start playback at first chunk; count later chunks missing their deadline."""
    if not arrival_offsets:
        raise AssertionError("clock needs at least one real chunk arrival")
    if any(not math.isfinite(x) or x < 0 for x in arrival_offsets):
        raise AssertionError("arrival offsets must be finite nonnegative seconds")
    if any(b < a for a, b in zip(arrival_offsets, arrival_offsets[1:])):
        raise AssertionError("arrivals must be monotonic")
    first = arrival_offsets[0]
    deadlines = [first + i * chunk_frames / FPS for i in range(len(arrival_offsets))]
    deficits = [max(0., arrival - deadline)
                for arrival, deadline in zip(arrival_offsets, deadlines)]
    return {"first_chunk_s": first, "underruns": sum(x > 0 for x in deficits),
            "max_late_s": max(deficits), "arrival_offsets_s": arrival_offsets,
            "playback_deadlines_s": deadlines}


def immutable_prefix(before: np.ndarray, after: np.ndarray) -> bool:
    return (after.shape[0] == before.shape[0] and after.shape[1] >= before.shape[1]
            and after.shape[2:] == before.shape[2:]
            and np.array_equal(before, after[:, :before.shape[1]]))


class ServiceError(RuntimeError):
    def __init__(self, status: int, body: dict | bytes):
        super().__init__(f"realtime service returned HTTP {status}: {body}")
        self.status = status
        self.body = body


class Client:
    def __init__(self, base_url: str, token: str, timeout: float = 30.):
        if not base_url.startswith("http://127.0.0.1:") and not base_url.startswith("http://localhost:"):
            raise ValueError("probe requires a loopback service URL")
        if not token:
            raise ValueError("private API token is required")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, bytes]:
        encoded = None if body is None else json.dumps(body, allow_nan=False).encode()
        request = Request(self.base_url + path, data=encoded, method=method,
                          headers={"Authorization": "Bearer " + self.token,
                                   "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()
        except URLError as error:
            raise RuntimeError(f"realtime service unavailable: {error.reason}") from error

    @staticmethod
    def decode_json(data: bytes) -> dict:
        value = json.loads(data)
        if not isinstance(value, dict):
            raise AssertionError("service JSON response must be an object")
        return value

    def get_json(self, path: str) -> tuple[int, dict]:
        status, blob = self.request("GET", path)
        return status, self.decode_json(blob)

    def submit(self, body: dict) -> dict:
        status, blob = self.request("POST", "/v1/realtime/jobs", body)
        reply = self.decode_json(blob)
        if status != 202 or reply.get("request_id") != body["request_id"]:
            raise ServiceError(status, reply)
        return reply

    def cancel(self, request_id: str) -> tuple[int, dict]:
        status, blob = self.request("DELETE", self.job_path(request_id))
        return status, self.decode_json(blob)

    @staticmethod
    def job_path(request_id: str) -> str:
        return "/v1/realtime/jobs/" + quote(request_id, safe="")


def collect_job(client: Client, body: dict, *, poll_s: float, deadline_s: float) -> dict:
    """Collect indexed chunks as soon as exposed, with actual receive times."""
    expected = body["frames"] // FRAMES_PER_CHUNK
    actors = len(body["actor_ids"])
    t0 = time.perf_counter()
    client.submit(body)
    received: list[tuple[dict[str, np.ndarray], dict, bytes]] = []
    arrival_offsets: list[float] = []
    early_chunks = 0
    path = client.job_path(body["request_id"])
    final_status = None
    while time.perf_counter() - t0 < deadline_s:
        status_code, state = client.get_json(path)
        if status_code != 200:
            raise ServiceError(status_code, state)
        produced = state.get("produced_chunks", 0)
        if type(produced) is not int or not 0 <= produced <= expected:
            raise AssertionError("invalid produced_chunks count")
        while len(received) < produced:
            index = len(received)
            code, blob = client.request("GET", path + f"/chunks/{index}")
            if code != 200:
                raise ServiceError(code, Client.decode_json(blob))
            arrays, meta = load_chunk(blob, actors, native_required=body["stage_kind"] != "paired")
            if meta.get("request_id", body["request_id"]) != body["request_id"]:
                raise AssertionError("stale chunk request_id")
            chunk_index = meta.get("chunk_index", index)
            if chunk_index != index:
                raise AssertionError("out-of-order chunk index")
            if meta.get("fps") != FPS or meta.get("frames") != FRAMES_PER_CHUNK:
                raise AssertionError("chunk metadata does not identify a 20 fps Core horizon")
            if meta.get("start_frame") != index * FRAMES_PER_CHUNK:
                raise AssertionError("chunk start frame is discontinuous")
            if meta.get("actor_ids") != body["actor_ids"] or meta.get("stage_kind") != body["stage_kind"]:
                raise AssertionError("chunk actor IDs or stage kind changed")
            received.append((arrays, meta, blob))
            arrival_offsets.append(time.perf_counter() - t0)
            if state.get("status") == "running":
                early_chunks += 1
        final_status = state.get("status")
        if final_status in TERMINAL:
            break
        time.sleep(poll_s)
    else:
        client.cancel(body["request_id"])
        raise TimeoutError(f"job exceeded {deadline_s:g}s")
    if final_status not in {"completed", "complete"} or len(received) != expected:
        raise AssertionError(f"job ended {final_status} after {len(received)}/{expected} chunks")
    keys = ("positions", "rotations") if body["stage_kind"] == "paired" else (
        "positions", "rotations", "native_features")
    arrays = {key: np.concatenate([item[0][key] for item in received], axis=1) for key in keys}
    if arrays["positions"].shape[1] != body["frames"]:
        raise AssertionError("job output has incorrect total frame count")
    elapsed = time.perf_counter() - t0
    return {"request_id": body["request_id"], "body": body, "chunks": received,
            "arrays": arrays, "arrival_offsets_s": arrival_offsets,
            "early_chunks": early_chunks,
            "elapsed_s": elapsed, "realtime_factor": body["frames"] / FPS / elapsed,
            "clock": playback_clock(arrival_offsets), "status": state}


def _job_id(prefix: str) -> str:
    return f"validation-{prefix}-{uuid.uuid4().hex[:12]}"


def _body(prefix: str, prompt: str, seed: int, *, stage: str = "approach",
          actor_ids: list[str] | None = None, frames: int = 40, **extra) -> dict:
    return {"request_id": _job_id(prefix), "stage_kind": stage,
            "frames": frames, "prompt": prompt, "actor_ids": actor_ids or ["actor"],
            "seed": seed, **extra}


def _sanitized_job(job: dict) -> dict:
    return {"request_id": job["request_id"], "stage_kind": job["body"]["stage_kind"],
            "actors": len(job["body"]["actor_ids"]), "frames": job["body"]["frames"],
            "chunks": len(job["chunks"]), "elapsed_s": job["elapsed_s"],
            "chunks_received_before_completion": job["early_chunks"],
            "realtime_factor": job["realtime_factor"], "clock": job["clock"],
            "seams": seam_report(job["arrays"]["positions"],
                                 tuple(range(40, job["body"]["frames"], 40)),
                                 job["arrays"]["rotations"])}


def wait_terminal(client: Client, request_id: str, *, poll_s: float, deadline_s: float) -> dict:
    path = client.job_path(request_id)
    until = time.monotonic() + deadline_s
    while time.monotonic() < until:
        code, state = client.get_json(path)
        if code != 200:
            raise ServiceError(code, state)
        if state.get("status") in TERMINAL:
            return state
        time.sleep(poll_s)
    client.cancel(request_id)
    raise TimeoutError("job did not reach terminal state")


def cancellation_probe(client: Client, *, poll_s: float, deadline_s: float) -> dict:
    """Keep one real six-window job busy while cancelling a queued successor."""
    lead = _body("cancel-lead", "A person walks forward naturally.", 9101, frames=240)
    queued = _body("cancel-queued", "A person waves.", 9102)
    client.submit(lead)
    try:
        client.submit(queued)
        cancel_code, cancel_state = client.cancel(queued["request_id"])
        if cancel_code != 200:
            raise ServiceError(cancel_code, cancel_state)
        queued_final = wait_terminal(client, queued["request_id"],
                                     poll_s=poll_s, deadline_s=deadline_s)
        lead_final = wait_terminal(client, lead["request_id"],
                                   poll_s=poll_s, deadline_s=deadline_s)
        no_stale = queued_final.get("status") == "cancelled" and queued_final.get("produced_chunks") == 0
        return {"queued_cancelled_without_chunks": no_stale,
                "lead_completed": lead_final.get("status") == "complete",
                "queued_status": queued_final.get("status"),
                "queued_chunks": queued_final.get("produced_chunks"),
                "lead_status": lead_final.get("status")}
    finally:
        # A failing assertion must not leave work queued for the shared GPU.
        client.cancel(queued["request_id"])
        client.cancel(lead["request_id"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fight_screening(directory: Path) -> tuple[dict, list[dict]]:
    """Summarize a bounded candidate screen without claiming model reliability."""
    path = directory / "diagnostics/fight-screening/summary.json"
    if not path.is_file():
        return {"available": False, "passed": False, "missing": path.name}, []
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise AssertionError("fight screening summary must be a list")
    passed_rows = [row for row in rows if row.get("passed") is True]
    identities = [(row.get("seed"), row.get("prompt_index")) for row in rows]
    revisions = {row.get("service_result", {}).get("metadata", {}).get("source_revision")
                 for row in rows}
    all_real = all(row.get("service_result", {}).get("status") == "complete"
                   and row.get("service_result", {}).get("produced_chunks") == 3
                   and row.get("service_result", {}).get("metadata", {}).get("model") == "InterGen"
                   for row in rows)
    valid = (len(rows) == 24 and len(set(identities)) == 24 and len(passed_rows) == 7
             and len(revisions) == 1 and None not in revisions and all_real)
    summary = {"available": True, "passed": valid,
               "scope": "24 preselected seed/prompt candidates on one research model; not a reliability estimate",
               "candidates": len(rows), "passed_candidates": len(passed_rows),
               "passing_seed_prompt_indices":
                   [[row["seed"], row["prompt_index"]] for row in passed_rows],
               "model_source_revision": next(iter(revisions)) if len(revisions) == 1 else None,
               "sha256": sha256_file(path)}
    return summary, rows


def validate_showcase_artifacts(directory: Path) -> dict:
    """Reopen full real scene archives and independently inspect the motion.

    The scene runner's own `metrics.complete` remains visible but is never
    accepted on its own: this rechecks array identity, frame count, all seams,
    and collision/contact proxies from the stored model output.
    """
    screening, screening_rows = read_fight_screening(directory)
    entries = []
    for name in SCENARIOS:
        report_path = directory / f"{name}.report.json"
        project_path = directory / f"{name}.stagezero-realtime.npz"
        raw_path = directory / f"{name}.raw.npz"
        missing = [path.name for path in (report_path, project_path, raw_path) if not path.is_file()]
        if missing:
            entries.append({"scenario": name, "passed": False, "missing": missing})
            continue
        try:
            source_report = json.loads(report_path.read_text())
            plan = source_report["plan"]
            if source_report.get("scenario") != name or plan.get("name") != name:
                raise AssertionError("scenario identity mismatch")
            if plan.get("fps") != FPS or plan.get("total_frames") != 600:
                raise AssertionError("plan must be a full 600-frame 20 fps scene")
            director = RealtimeDirector.load_project(project_path.read_bytes())
            clip = director.timeline_clip()
            if clip is None:
                raise AssertionError("project has no committed model motion")
            with np.load(raw_path, allow_pickle=False) as raw:
                raw_positions = raw["positions"].copy()
                raw_rotations = raw["rotations"].copy()
                raw_meta = json.loads(str(raw["metadata"].item()))
            if (clip.positions.shape != (len(plan["actor_ids"]), 600, JOINTS, 3)
                    or clip.rotations.shape != (len(plan["actor_ids"]), 600, JOINTS, 3, 3)
                    or clip.actor_ids != tuple(plan["actor_ids"])):
                raise AssertionError("project has an incorrect synchronized Core27 timeline")
            if raw_meta.get("fps") != FPS or raw_meta.get("actor_ids") != plan["actor_ids"]:
                raise AssertionError("raw metadata disagrees with project timeline")
            exact = (np.array_equal(clip.positions, raw_positions)
                     and np.array_equal(clip.rotations, raw_rotations))
            if not exact:
                raise AssertionError("saved project and raw motion arrays differ")
            if len(director.segments) != 15:
                raise AssertionError("project does not preserve all fifteen committed horizons")
            playback = source_report.get("playback", {})
            realtime_measured = (playback.get("clock") == "wall"
                                 and isinstance(playback.get("startup_seconds"), (int, float))
                                 and playback["startup_seconds"] <= 5.
                                 and isinstance(playback.get("buffer_underrun_seconds"), (int, float))
                                 and playback["buffer_underrun_seconds"] <= .25
                                 and playback.get("realtime_passed") is True)
            for segment in director.segments:
                matching = [beat for beat in plan["beats"]
                            if beat["start_frame"] <= segment["start"] < beat["end_frame"]]
                if (len(matching) != 1 or segment["end"] != segment["start"] + 40
                        or segment["source"] != matching[0]["source"]
                        or segment.get("metadata", {}).get("beat_id") != matching[0]["id"]):
                    raise AssertionError("project segment provenance disagrees with the scene plan")
            expected_jobs = sum(beat["source"] == "ardy_core" for beat in plan["beats"])
            expected_jobs += any(beat["source"] == "intergen" for beat in plan["beats"])
            jobs = source_report.get("jobs", [])
            if len(jobs) != expected_jobs:
                raise AssertionError("scene report lacks a required model job")
            request_ids = [job["request"]["request_id"] for job in jobs]
            if len(set(request_ids)) != len(request_ids):
                raise AssertionError("scene reused a model request ID")
            if any(len(job["chunk_arrival_seconds"]) != job["request"]["frames"] // 40
                   for job in jobs):
                raise AssertionError("scene report omits a model chunk arrival")
            hashes = {"report_sha256": sha256_file(report_path),
                      "project_sha256": sha256_file(project_path),
                      "raw_sha256": sha256_file(raw_path)}
            pair_transform = None
            paired_beats = [beat for beat in plan["beats"] if beat["source"] == "intergen"]
            if paired_beats:
                pair_path = directory / f"{name}.pair-source.npz"
                if not pair_path.is_file():
                    raise AssertionError("original InterGen pair source archive is missing")
                hashes["pair_source_sha256"] = sha256_file(pair_path)
                with np.load(pair_path, allow_pickle=False) as pair_archive:
                    original_p = pair_archive["positions"].copy()
                    original_r = pair_archive["rotations"].copy()
                pair_start = min(beat["start_frame"] for beat in paired_beats)
                pair_end = max(beat["end_frame"] for beat in paired_beats)
                if (original_p.shape != (2, pair_end - pair_start, JOINTS, 3)
                        or original_r.shape != (2, pair_end - pair_start, JOINTS, 3, 3)):
                    raise AssertionError("original paired sample has the wrong synchronized shape")
                placement = source_report["metrics"]["placement"]
                yaw = placement["yaw_radians"]
                translation = np.asarray(placement["translation_xz_m"], dtype=np.float64)
                c, s = math.cos(yaw), math.sin(yaw)
                matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                expected_p = original_p.astype(np.float64) @ matrix.T
                expected_p[..., 0] += translation[0]
                expected_p[..., 2] += translation[1]
                expected_r = matrix @ original_r.astype(np.float64)
                observed_p = clip.positions[:, pair_start:pair_end]
                observed_r = clip.rotations[:, pair_start:pair_end]
                p_error = float(np.max(np.abs(expected_p - observed_p)))
                r_error = float(np.max(np.abs(expected_r - observed_r)))
                pair_transform = {"source_frames": int(original_p.shape[1]),
                                  "max_position_error_m": p_error,
                                  "max_rotation_matrix_error": r_error,
                                  "shared_rigid_transform_exact": p_error <= 1e-4 and r_error <= 1e-4,
                                  "license": "CC BY-NC-SA 4.0", "research_only": True}
                if not pair_transform["shared_rigid_transform_exact"]:
                    raise AssertionError("paired actors no longer share the original rigid world transform")
            selected_screened_candidate = None
            if name == "staged_fight":
                paired_prompt = next(beat["prompt"] for beat in plan["beats"]
                                     if beat["kind"] == "paired")
                matches = [row for row in screening_rows
                           if row.get("seed") == plan["seed"] and row.get("prompt") == paired_prompt]
                pair_job = next((job for job in jobs if job["beat_id"] == "pair_prefetch"), None)
                source_revision = (None if pair_job is None else
                                   pair_job.get("service_result", {}).get("metadata", {}).get("source_revision"))
                selected_screened_candidate = {
                    "seed": plan["seed"], "prompt_index": matches[0].get("prompt_index") if len(matches) == 1 else None,
                    "passed_screening": len(matches) == 1 and matches[0].get("passed") is True,
                    "same_model_source_revision": source_revision == screening.get("model_source_revision"),
                }
                selected_screened_candidate["passed"] = (screening["passed"]
                    and selected_screened_candidate["passed_screening"]
                    and selected_screened_candidate["same_model_source_revision"])
            # Re-serialize and reload the saved project to check the actual
            # on-disk arrays survive an ordinary subsequent save as well.
            reloaded = RealtimeDirector.load_project(director.save_project()).timeline_clip()
            resave_exact = (np.array_equal(reloaded.positions, clip.positions)
                            and np.array_equal(reloaded.rotations, clip.rotations)
                            and ((reloaded.native_features is None and clip.native_features is None)
                                 or (reloaded.native_features is not None and clip.native_features is not None
                                     and np.array_equal(reloaded.native_features, clip.native_features))))
            all_seams = seam_report(clip.positions, tuple(range(40, 600, 40)), clip.rotations)
            beat_boundaries = tuple(beat["start_frame"] for beat in plan["beats"][1:])
            beat_seams = seam_report(clip.positions, beat_boundaries, clip.rotations)
            collision = {actor_id: scene_collision(clip.positions[index], "core27", plan["scene"])
                         for index, actor_id in enumerate(plan["actor_ids"])}
            collision_gates = []
            for beat in plan["beats"]:
                for gate in beat["quality_gates"]:
                    if gate["metric"] == "scene_collision":
                        actor_index = plan["actor_ids"].index(gate["actor_id"])
                        measured = scene_collision(
                            clip.positions[actor_index, beat["start_frame"]:beat["end_frame"]],
                            "core27", plan["scene"])
                        collision_gates.append({"beat_id": beat["id"], "actor_id": gate["actor_id"],
                                                "frames": measured["total_collision_frames"],
                                                "limit": gate["max_overlap_frames"],
                                                "passed": measured["total_collision_frames"] <= gate["max_overlap_frames"]})
            floor = {actor_id: floor_motion(clip.positions[index], skeleton="core27", fps=FPS)
                     for index, actor_id in enumerate(plan["actor_ids"])}
            pair = None
            hand = None
            if len(plan["actor_ids"]) == 2:
                pair = pair_separation(clip.positions[0], clip.positions[1],
                                       skeleton_a="core27", skeleton_b="core27")
                focus = next((beat for beat in plan["beats"] if beat["kind"] == "paired"), None)
                if focus is not None:
                    lo, hi = focus["start_frame"], focus["end_frame"]
                    variants = [hand_contact(clip.positions[0, lo:hi], clip.positions[1, lo:hi],
                                             skeleton_a="core27", skeleton_b="core27",
                                             hand_a=a, hand_b=b, fps=FPS,
                                             tolerance_m=.15, minimum_duration_s=.35)
                                for a in ("left", "right") for b in ("left", "right")]
                    hand = max(variants, key=lambda value: value["longest_near_run_frames"])
            critical_gates = []
            ids = {actor_id: index for index, actor_id in enumerate(plan["actor_ids"])}
            for beat in plan["beats"]:
                lo, hi = beat["start_frame"], beat["end_frame"]
                for gate in beat["quality_gates"]:
                    kind = gate["metric"]
                    if kind == "gate_traversal":
                        obj = next(obj for obj in scene_objects(plan["scene"])
                                   if obj.id == gate["object_id"])
                        opening = passage_for(obj, None, actor_height_m=1.65)
                        _, normal = local_axes(opening.yaw_degrees)
                        measurement = gate_traversal(
                            clip.positions[ids[gate["actor_id"]], lo:hi], skeleton="core27",
                            center=(obj.x, obj.y, obj.z), normal_xz=normal,
                            opening_width_m=opening.width_m, min_y=0, max_y=opening.height_m)
                        passed_gate = measurement["traversed_proxy"]
                    elif kind == "object_contact":
                        target = beat["metadata"]["native_hand_target"]["position_xyz"]
                        measurement = object_contact(
                            clip.positions[ids[gate["actor_id"]], lo:hi], skeleton="core27",
                            object_xyz=target, hand=gate["hand"], fps=FPS,
                            tolerance_m=gate["tolerance_m"],
                            minimum_duration_s=gate["minimum_duration_s"])
                        passed_gate = measurement["contact_proxy"]
                    elif kind == "pair_separation":
                        measurement = pair_separation(clip.positions[0, lo:hi], clip.positions[1, lo:hi],
                                                      skeleton_a="core27", skeleton_b="core27")
                        passed_gate = measurement["min_root_separation_xz_m"] >= gate["min_m"]
                    elif kind == "hand_contact":
                        measurement = hand
                        passed_gate = hand is not None and hand["contact_proxy"]
                    elif kind == "continuity":
                        root_steps = [float(np.linalg.norm(clip.positions[i, lo, 0] -
                                                           clip.positions[i, lo - 1, 0]))
                                      for i in range(len(ids))]
                        measurement = {"boundary_root_steps_m": root_steps}
                        passed_gate = max(root_steps) <= gate["max_boundary_jump_m"]
                    elif kind == "hand_release":
                        a, b = clip.positions[0, hi - 10:hi], clip.positions[1, hi - 10:hi]
                        gaps = [float(np.linalg.norm(a[:, i] - b[:, j], axis=1).mean())
                                for i in (11, 17) for j in (11, 17)]
                        measurement = {"mean_final_hand_gap_m": min(gaps)}
                        passed_gate = measurement["mean_final_hand_gap_m"] >= .25
                    elif kind == "dodge_lateral":
                        roots = clip.positions[:, lo:hi, 0][:, :, (0, 2)]
                        opening = roots[1, 0] - roots[0, 0]
                        opening = opening / max(float(np.linalg.norm(opening)), 1e-9)
                        normal = np.array([-opening[1], opening[0]])
                        lateral = (roots[1] - roots[0]) @ normal
                        measurement = {"relative_lateral_range_m": float(np.ptp(lateral))}
                        passed_gate = measurement["relative_lateral_range_m"] >= gate["min_m"]
                    elif kind == "block_guard":
                        a, b = clip.positions[0, lo:hi], clip.positions[1, lo:hi]
                        candidate_gaps = [np.linalg.norm(a[:, i] - b[:, j], axis=1)
                                          for i, j in ((11, 9), (11, 15), (17, 9), (17, 15),
                                                       (9, 11), (15, 11), (9, 17), (15, 17))]
                        gaps = np.min(np.stack(candidate_gaps), axis=0)
                        longest = current = 0
                        for near in gaps <= gate["tolerance_m"]:
                            current = current + 1 if near else 0
                            longest = max(longest, current)
                        measurement = {"min_hand_forearm_gap_m": float(gaps.min()),
                                       "longest_near_run_s": longest / FPS}
                        passed_gate = measurement["longest_near_run_s"] >= gate["minimum_duration_s"]
                    elif kind == "push_reaction":
                        observed_end = min(lo + beat["metadata"].get("source_total_frames", hi - lo),
                                           clip.frames)
                        roots = clip.positions[:, lo:observed_end, 0][:, :, (0, 2)]
                        gaps = np.linalg.norm(roots[0] - roots[1], axis=1)
                        third = len(gaps) // 3
                        measurement = {"first_third_mean_root_gap_m": float(gaps[:third].mean()),
                                       "last_third_mean_root_gap_m": float(gaps[-third:].mean())}
                        passed_gate = (measurement["last_third_mean_root_gap_m"] >
                                       measurement["first_third_mean_root_gap_m"] + .15)
                    elif kind == "scene_collision":
                        measurement = next(item for item in collision_gates
                                           if item["beat_id"] == beat["id"] and
                                           item["actor_id"] == gate["actor_id"])
                        passed_gate = measurement["passed"]
                    else:
                        continue
                    critical_gates.append({"beat_id": beat["id"], "metric": kind,
                                           "passed": bool(passed_gate), "measurement": measurement})
            chunk_latencies = [arrival[0] for job in jobs
                               if (arrival := job.get("chunk_arrival_seconds", []))]
            job_rtfs = [job["request"]["frames"] / FPS / job["elapsed_seconds"]
                        for job in jobs
                        if job.get("request", {}).get("frames") and job.get("elapsed_seconds", 0) > 0]
            local_underruns = sum(playback_clock(job["chunk_arrival_seconds"])["underruns"]
                                  for job in jobs
                                  if job.get("chunk_arrival_seconds"))
            source_complete = (source_report.get("status") == "complete"
                               and source_report.get("accepted") is True
                               and source_report.get("metrics", {}).get("complete") is True)
            no_prop_overlap = all(item["passed"] for item in collision_gates)
            pair_clear = pair is None or pair["root_disc_overlap_proxy_frames"] == 0
            contact = hand is None or name != "gate_meet_handshake" or hand["contact_proxy"]
            critical_passed = all(item["passed"] for item in critical_gates)
            passed = (source_complete and realtime_measured and exact and resave_exact and all_seams["passed"]
                      and beat_seams["passed"] and no_prop_overlap
                      and contact and critical_passed
                      and (selected_screened_candidate is None or selected_screened_candidate["passed"]))
            entries.append({"scenario": name, "passed": bool(passed), "source_status": source_report.get("status"),
                            "frames": 600, "actors": len(plan["actor_ids"]),
                            "project_raw_exact": exact, "resave_exact": resave_exact,
                            "source_artifact_sha256": hashes,
                            "selected_screened_candidate": selected_screened_candidate,
                            "playback": {key: playback.get(key) for key in
                                         ("clock", "startup_seconds", "first_pose_network_seconds",
                                          "buffer_underrun_seconds", "buffer_underrun_events",
                                          "pair_prefetch_seconds", "realtime_passed")},
                            "realtime_measured": realtime_measured,
                            "pair_source_transform": pair_transform,
                            "all_chunk_seams": all_seams, "beat_seams": beat_seams,
                            "scene_collision": collision, "required_collision_gates": collision_gates,
                            "floor_motion": floor,
                            "pair_separation": pair, "paired_hand_gap": hand,
                            "independent_quality_gates": critical_gates,
                            "job_first_chunk": timing_summary(chunk_latencies),
                            "job_realtime_factor_p05": percentile(job_rtfs, .05),
                            "job_local_underruns": local_underruns,
                            "source_quality_complete": source_complete,
                            "independent_quality_passed": critical_passed,
                            "required_prop_clearance_proxy": no_prop_overlap,
                            "no_pair_root_overlap_proxy": pair_clear,
                            "pair_overlap_review_required": not pair_clear})
        except (AssertionError, ValueError, KeyError, OSError, TypeError, RuntimeError) as error:
            entries.append({"scenario": name, "passed": False,
                            "error": {"type": type(error).__name__, "message": str(error)[:500]}})
    return {"passed": len(entries) == 3 and all(entry["passed"] for entry in entries),
            "scenarios": entries,
            "fight_screening": screening,
            "limitations": ["Scene collision and hand contact use sampled geometric proxies; physical mesh collision and contact forces remain unverified.",
                            "Root-disc overlap during close contact is reported for review; this conservative proxy alone does not establish mesh collision.",
                            "Job local underruns omit client rendering and cross-job scheduling gaps; use a running viewer for final playback latency."]}


def release_scope() -> dict:
    return {"scope": "research-preview", "commercial_release_ready": False,
            "blocking_license": "InterGen CC BY-NC-SA 4.0",
            "reason": "The two integrated pair scenes use InterGen research motion; geometry proxies and viewer latency also require release-specific review."}


def local_source_checksums() -> dict:
    root = Path(__file__).resolve().parents[1]
    names = ("interaction_runtime.py", "motion_bridge.py", "realtime_backend.py",
             "realtime_client.py", "realtime_director.py", "realtime_clip.py",
             "scene_beats.py", "experiments/generate_scene_showcase.py",
             "experiments/verify_realtime_pipeline.py")
    return {"scope": "local checkout at report generation; not a remote deployment attestation",
            "sha256": {name: sha256_file(root / name) for name in names}}


def run_probe(args: argparse.Namespace) -> dict:
    client = Client(args.base_url, args.token_file.read_text().strip(), args.request_timeout)
    status, health = client.get_json("/health")
    if status != 200 or not health.get("ready"):
        raise ServiceError(status, health)
    if health.get("model") != MODEL_NAME:
        raise AssertionError("health does not identify the required ARDY Core model")
    if (health.get("fps"), health.get("joints"), health.get("horizon")) != (FPS, JOINTS, FRAMES_PER_CHUNK):
        raise AssertionError("health does not advertise Core27 at 20 fps and horizon 40")
    unauthorized = Client(args.base_url, "intentionally-wrong-token", args.request_timeout)
    auth_code, _ = unauthorized.request("GET", "/health")
    if auth_code != 401:
        raise AssertionError("health accepted an invalid token")
    invalid = _body("invalid", "A person walks forward.", 999, frames=41)
    invalid_code, _ = client.request("POST", "/v1/realtime/jobs", invalid)
    if invalid_code != 400:
        raise AssertionError("invalid 41-frame job was not rejected")

    prompts = ["A person walks forward naturally.", "A person turns to the left and keeps walking.",
               "A person reaches forward and then relaxes.", "A person steps sideways carefully.",
               "A person waves once while standing."]
    jobs = []
    for index in range(args.requests):
        body = _body(f"soak-{index}", prompts[index % len(prompts)], 7100 + index)
        jobs.append(collect_job(client, body, poll_s=args.poll, deadline_s=args.job_deadline))
    multi_window = collect_job(client, _body("multi-window", "A person walks steadily forward.", 9090,
                                              frames=120), poll_s=args.poll,
                               deadline_s=args.job_deadline)
    cancellation = cancellation_probe(client, poll_s=args.poll, deadline_s=args.job_deadline)
    recovery = collect_job(client, _body("after-cancel", "A person walks forward.", 9110),
                           poll_s=args.poll, deadline_s=args.job_deadline)

    # Three distinct Core compositions exercise a continuous three-stage clock.
    # The separate integrated scene runner owns the paired/object 600-frame
    # scenarios; these shorter jobs are a service and history stress check.
    definitions = [
        ("walk-turn", ["walker"], "A person walks forward, turns, and continues walking."),
        ("reach", ["reacher"], "A person approaches and reaches forward, then relaxes."),
        ("meeting", ["left", "right"], "Two people approach one another and greet each other."),
    ]
    scenes = []
    for scene_index, (label, actor_ids, prompt) in enumerate(definitions):
        stages = []
        assembled = None
        timeline_arrivals = []
        start = time.perf_counter()
        for stage_index, stage in enumerate(("approach", "continuation", "continuation")):
            extra = {}
            if stages:
                # Core history uses the last four native frames of every actor.
                extra["history"] = {"native_features": stages[-1]["arrays"]["native_features"][:, -4:].tolist()}
            body = _body(f"{label}-{stage}", prompt, 8200 + scene_index * 10 + stage_index,
                         stage=stage, actor_ids=actor_ids, **extra)
            job = collect_job(client, body, poll_s=args.poll, deadline_s=args.job_deadline)
            stages.append(job)
            timeline_arrivals.append(time.perf_counter() - start)
            current = job["arrays"]["positions"]
            before = assembled
            assembled = current.copy() if before is None else np.concatenate([before, current], axis=1)
            if before is not None and not immutable_prefix(before, assembled):
                raise AssertionError("stage altered the committed scene prefix")
        rotations = np.concatenate([job["arrays"]["rotations"] for job in stages], axis=1)
        seam = seam_report(assembled, (40, 80), rotations)
        scene = {"name": label, "actor_ids": actor_ids, "frames": int(assembled.shape[1]),
                 "stage_jobs": [_sanitized_job(job) for job in stages],
                 "continuous_clock": playback_clock(timeline_arrivals),
                 "seams": seam, "committed_prefix_exact": True}
        if len(actor_ids) == 2:
            scene["pair_separation"] = pair_separation(assembled[0], assembled[1],
                                                       skeleton_a="core27", skeleton_b="core27")
            scene["hand_gap"] = hand_contact(assembled[0], assembled[1],
                                              skeleton_a="core27", skeleton_b="core27", fps=FPS)
        scenes.append(scene)

    first_chunks = [job["clock"]["first_chunk_s"] for job in jobs]
    rtfs = [job["realtime_factor"] for job in jobs]
    roundtrip = exact_disk_roundtrip(jobs[0]["chunks"][0][2], args.output.parent / "realtime-first-chunk.npz", 1)
    all_seams = [scene["seams"]["passed"] for scene in scenes]
    # A separate functional gate is reported so a slow but correct pipeline
    # cannot silently claim the realtime requirement.
    checks = {"auth_rejects_wrong_token": True, "malformed_rejected": True,
              "soak_25_or_more": len(jobs) >= 25,
              "all_jobs_complete": len(jobs) == args.requests,
              "progressive_three_chunk_job": len(multi_window["chunks"]) == 3 and
                                               multi_window["early_chunks"] >= 1 and
                                               multi_window["clock"]["underruns"] == 0,
              "save_load_exact_arrays": roundtrip,
              "queued_cancel_discards_chunks": cancellation["queued_cancelled_without_chunks"],
              "cancelled_request_recovery": cancellation["lead_completed"] and recovery["status"]["status"] == "complete",
              "three_core_compositions": len(scenes) == 3,
              "two_actor_timeline": scenes[-1]["actor_ids"] == ["left", "right"] and scenes[-1]["frames"] == 120,
              "committed_prefix_exact": all(scene["committed_prefix_exact"] for scene in scenes),
              "core_composition_seams_within_limits": all(all_seams),
              "no_playback_underruns": all(scene["continuous_clock"]["underruns"] == 0 for scene in scenes),
              "warm_first_chunk_p95_below_2s": percentile(first_chunks, .95) <= FIRST_CHUNK_LIMIT_S,
              "warm_realtime_factor_p05_above_1": percentile(rtfs, .05) >= MIN_REALTIME_FACTOR}
    report = {"kind": "actual-gpu-service", "passed": all(checks.values()), "checks": checks,
              "thresholds": {"root_seam_m": ROOT_SEAM_LIMIT_M,
                             "mean_joint_seam_m": MEAN_JOINT_SEAM_LIMIT_M,
                             "max_joint_seam_m": MAX_JOINT_SEAM_LIMIT_M,
                             "first_chunk_p95_s": FIRST_CHUNK_LIMIT_S,
                             "realtime_factor_p05": MIN_REALTIME_FACTOR},
              "health": {key: health.get(key) for key in ("service", "model", "fps", "joints", "features", "horizon", "ready")},
              "soak": {"request_count": len(jobs), "first_chunk": timing_summary(first_chunks),
                       "completion": timing_summary([job["elapsed_s"] for job in jobs]),
                       "realtime_factor_p05": percentile(rtfs, .05),
                       "realtime_factor_p50": percentile(rtfs, .5),
                       "realtime_factor_p95": percentile(rtfs, .95),
                       "underruns": sum(job["clock"]["underruns"] for job in jobs),
                       "jobs": [_sanitized_job(job) for job in jobs]},
              "multi_window": _sanitized_job(multi_window),
              "scenes": scenes,
              "cancellation": cancellation,
              "release": release_scope(),
              "local_source": local_source_checksums(),
              "limitations": ["Proximity metrics are geometric proxies, not proof of physical contact or mesh clearance.",
                              "This probe measures service chunk arrival; browser rendering latency is separate.",
                              "The service is already warm at health check; process/model cold load is measured separately."]}
    showcase = validate_showcase_artifacts(args.showcase_dir)
    report["showcase"] = showcase
    report["checks"]["three_integrated_600_frame_scenes"] = showcase["passed"]
    report["passed"] = all(report["checks"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8769")
    parser.add_argument("--token-file", type=Path,
                        help="absolute path to the existing private API token")
    parser.add_argument("--output", type=Path, default=Path("review/realtime-validation.json"))
    parser.add_argument("--requests", type=int, default=25)
    parser.add_argument("--poll", type=float, default=.05)
    parser.add_argument("--request-timeout", type=float, default=30.)
    parser.add_argument("--job-deadline", type=float, default=120.)
    parser.add_argument("--showcase-dir", type=Path, default=Path(".runtime/realtime-showcase"))
    parser.add_argument("--showcase-only", action="store_true",
                        help="Revalidate saved full-scene artifacts without rerunning the GPU soak")
    args = parser.parse_args()
    if args.showcase_only:
        if not args.output.is_file():
            parser.error("--showcase-only requires an existing validation report")
        report = json.loads(args.output.read_text())
        showcase = validate_showcase_artifacts(args.showcase_dir)
        report["showcase"] = showcase
        report.setdefault("checks", {})["three_integrated_600_frame_scenes"] = showcase["passed"]
        report["release"] = release_scope()
        report["local_source"] = local_source_checksums()
        report["passed"] = all(report["checks"].values())
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"passed": report["passed"], "showcase": showcase["passed"]}, indent=2))
        return 0 if report["passed"] else 1
    if (args.requests < 25 or not 0 < args.poll <= 2 or args.job_deadline <= 0
            or args.token_file is None or not args.token_file.is_absolute()):
        parser.error("requires >=25 requests, poll in (0,2], positive deadline, and an absolute token-file path")
    try:
        report = run_probe(args)
    except (AssertionError, RuntimeError, TimeoutError, OSError, ValueError) as error:
        report = {"kind": "actual-gpu-service-attempt", "passed": False,
                  "checks": {"probe_completed": False},
                  "error": {"type": type(error).__name__, "message": str(error)[:500]}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2))
        return 1
    print(json.dumps({"passed": report["passed"], "checks": report["checks"],
                      "soak": {key: value for key, value in report["soak"].items() if key != "jobs"}}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
