#!/usr/bin/env python3
"""Audit a recorded live swing from its actual 60 Hz state and browser video.

The checker reads the server's published pose arrays. It does not call the
animation controller or infer contact from the source Core clips. City checks
are conservative joint-sphere versus imported building AABB proxies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Official CoreSkeleton27 parent order, with the root's -1 omitted below.
PARENTS = np.array([-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 10, 10,
                    4, 13, 14, 15, 16, 16, 0, 19, 20, 21, 0, 23, 24, 25])
TORSO_JOINTS = (0, 1, 2, 3, 4, 5, 6, 7, 13, 19, 23)
GRIP_KEYS = ("mj_right_shoulder_m", "mj_left_shoulder_m", "hero_thigh_m")
FOOT_SEGMENTS = frozenset((21, 22, 25, 26))
SEGMENT_RADII = np.array([0., .13, .13, .13, .13, .12, .11,
                          .10, .085, .075, .065, .055, .05,
                          .10, .085, .075, .065, .055, .05,
                          .11, .105, .085, .065, .11, .105, .085, .065])


def stats(values):
    if not values:
        return {"samples": 0, "max": None, "p95": None, "mean": None}
    a = np.asarray(values, dtype=float)
    return {"samples": int(len(a)), "max": float(a.max()),
            "p95": float(np.percentile(a, 95)), "mean": float(a.mean())}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sphere_box_penetration(point, radius, box):
    """Depth of a conservative sphere/AABB overlap, zero when clear."""
    p = np.asarray(point, dtype=float)
    lo, hi = np.asarray(box["min"], dtype=float), np.asarray(box["max"], dtype=float)
    outside = np.maximum(np.maximum(lo - p, p - hi), 0)
    distance = float(np.linalg.norm(outside))
    if distance > 0:
        return max(0., radius - distance)
    return radius + float(np.min(np.minimum(p - lo, hi - p)))


def segment_clearance_samples(positions):
    """Three interior samples and both endpoints per Core27 parent-child bone."""
    p = np.asarray(positions, dtype=float)
    if p.shape != (2, 27, 3):
        raise ValueError("Expected two Core27 actors")
    origins = p[:, PARENTS[1:]]
    ends = p[:, 1:]
    fractions = np.array([0., .25, .5, .75, 1.])
    samples = origins[:, :, None] * (1 - fractions[None, None, :, None]) + ends[:, :, None] * fractions[None, None, :, None]
    radii = np.broadcast_to(SEGMENT_RADII[1:][None, :, None], samples.shape[:-1])
    segments = np.broadcast_to(np.arange(1, 27)[None, :, None], samples.shape[:-1])
    return samples.reshape(-1, 3), radii.reshape(-1), segments.reshape(-1)


def box_depths(points, radii, box):
    """Vectorized conservative sphere/AABB intersection depth."""
    lo, hi = np.asarray(box["min"], dtype=float), np.asarray(box["max"], dtype=float)
    outside = np.maximum(np.maximum(lo - points, points - hi), 0)
    outside_distance = np.linalg.norm(outside, axis=-1)
    inside_depth = np.min(np.minimum(points - lo, hi - points), axis=-1)
    return np.where(outside_distance > 0, np.maximum(0, radii - outside_distance),
                    radii + inside_depth)


def action_sequence(commands):
    """Find ordered user inputs: two swing steers, carry, carry steer, land, kiss."""
    aliases = {"start": "swing", "left": "steer", "right": "steer",
               "steer_left": "steer", "steer_right": "steer", "pickup": "carry",
               "carry_mj": "carry"}
    pairs = [(aliases.get(c.get("action"), c.get("action")), c.get("id")) for c in commands]
    phases = ["swing", "steer", "steer", "carry", "steer", "land", "kiss"]
    chosen = []
    j = 0
    for action, cid in pairs:
        if j < len(phases) and action == phases[j]:
            chosen.append(cid)
            j += 1
    return {"required": phases, "observed": [p[0] for p in pairs],
            "matched_command_ids": chosen, "complete": j == len(phases)}


def validate_command_states(sequence, command_states, events):
    """Confirm commands applied in the intended airborne/carry/landing states."""
    rows = [command_states.get(cid) for cid in sequence["matched_command_ids"]]
    if not sequence["complete"] or len(rows) != 7 or not all(rows):
        return False
    start, steer_a, steer_b, carry_cmd, carry_steer, land_cmd, kiss_cmd = rows
    attached_frames = [event.get("frame") for event in events.values() if event.get("event") == "mj_attached"]
    contact_frames = [event.get("frame") for event in events.values() if event.get("event") == "landing_contact"]
    return bool(
        all(row["status"] == "applied" for row in rows)
        and start["phase"] == "swing"
        and all(row["phase"] == "swing" and not row["grounded"]
                and row["flight_amount"] > .5 and row["carry_amount"] < .05
                for row in (steer_a, steer_b))
        and carry_cmd["phase"] == "pickup" and not carry_cmd["grounded"]
        and carry_cmd["flight_amount"] > .5
        and carry_steer["phase"] == "swing" and not carry_steer["grounded"]
        and carry_steer["carry_amount"] > .98
        and land_cmd["phase"] == "landing" and land_cmd["carry_amount"] > .98
        and kiss_cmd["phase"] == "kiss" and kiss_cmd["grounded"]
        and any(isinstance(f, int) and carry_cmd["frame"] < f < carry_steer["frame"]
                for f in attached_frames)
        and any(isinstance(f, int) and land_cmd["frame"] < f < kiss_cmd["frame"]
                for f in contact_frames)
    )


def video_info(path):
    if not path.is_file():
        return {"present": False, "bytes": 0, "webm_header": False, "duration_seconds": None,
                "video_packets": 0, "max_packet_gap_seconds": None}
    size = path.stat().st_size
    with path.open("rb") as stream:
        magic = stream.read(4)
    result = {"present": True, "bytes": size,
              "webm_header": magic == bytes.fromhex("1a45dfa3"), "duration_seconds": None,
              "video_packets": 0, "max_packet_gap_seconds": None,
              "sha256": sha256_file(path)}
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path)],
                               capture_output=True, text=True, timeout=10, check=True)
        times = [float(line) for line in probe.stdout.splitlines() if line.strip()]
        if times and all(math.isfinite(t) for t in times):
            result["video_packets"] = len(times)
            result["duration_seconds"] = max(0., times[-1] - times[0])
            result["max_packet_gap_seconds"] = float(max(np.diff(times), default=0.))
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return result


def _lines(path):
    with path.open() as stream:
        yield from stream


def audit(run_dir: Path, scene: dict, *, require_video: bool = True):
    log = run_dir / "frames.jsonl"
    if not log.is_file():
        raise FileNotFoundError(log)
    report_path = run_dir / "recorded-report.json"
    if not report_path.is_file():
        report_path = run_dir / "report.json"
    saved = json.loads(report_path.read_text()) if report_path.is_file() else {}
    saved_frame_limit = saved.get("frames")
    if saved_frame_limit is not None and (type(saved_frame_limit) is not int or saved_frame_limit < 2):
        raise ValueError("Saved report has invalid frame count")
    previous = None
    initial_bones = None
    frame_count = 0
    first_frame = last_frame = first_time = last_time = None
    max_frame_gap = 0
    nonfinite = 0
    pose_steps, root_steps, bone_drifts, rotation_errors = [], [], [], []
    penetration_depths = []
    penetration_by_building = {}
    full_body_depths = []
    full_body_by_building = {}
    worst_full_body = {"metres": 0., "frame": None, "time": None,
                       "actor": None, "segment": None, "building": None,
                       "sample_point": None, "sample_radius": None}
    allowed_roof_contact_samples = 0
    web_depths = []
    web_by_building = {}
    web_frames = 0
    web_terminal_allowances = set()
    worst_web = {"metres": 0., "frame": None, "building": None,
                 "fraction_hand_to_anchor": None, "anchor_building": None}
    roof_by_building = {r["building_id"]: r for r in scene["roofs"]}
    grip_samples = {key: [] for key in GRIP_KEYS}
    active_carry_frames = 0
    phases = set()
    phase_counts = {}
    phase_starts = {}
    events = {}
    worst_joint_step = {"metres": 0., "from_frame": None, "to_frame": None,
                        "actor": None, "joint": None}
    final_positions = None
    final_phase = None
    final_landing_id = None
    observed_commands = {}
    command_states = {}
    provenance = set()
    recorded_log_hash = hashlib.sha256()
    for line_no, line in enumerate(_lines(log), 1):
        if not line.strip():
            continue
        state = json.loads(line)
        if "error" in state:
            raise ValueError(f"Simulation error at log line {line_no}: {state['error']}")
        actor_rows = state.get("actors")
        if not isinstance(actor_rows, list) or len(actor_rows) != 2:
            raise ValueError(f"Expected two actors at log line {line_no}")
        positions = np.asarray([a["positions"] for a in actor_rows], dtype=float)
        rotations = np.asarray([a["rotations"] for a in actor_rows], dtype=float)
        if positions.shape != (2, 27, 3) or rotations.shape != (2, 27, 3, 3):
            raise ValueError(f"Invalid Core27 pose shape at line {line_no}")
        frame = int(state["frame"])
        if saved_frame_limit is not None and frame > saved_frame_limit:
            break
        recorded_log_hash.update(line.encode())
        timestamp = float(state["time"])
        if not np.isfinite(positions).all() or not np.isfinite(rotations).all() or not math.isfinite(timestamp):
            nonfinite += 1
            continue
        lengths = np.linalg.norm(positions[:, 1:] - positions[:, PARENTS[1:]], axis=-1)
        if initial_bones is None:
            initial_bones = lengths
            first_frame, first_time = frame, timestamp
        bone_drifts.append(float(np.max(np.abs(lengths - initial_bones))))
        rotation_errors.append(float(np.max(np.abs(rotations @ np.swapaxes(rotations, -1, -2) - np.eye(3)))))
        if previous is not None:
            max_frame_gap = max(max_frame_gap, frame - previous["frame"])
            joint_steps = np.linalg.norm(positions - previous["positions"], axis=-1)
            pose_step = float(np.max(joint_steps))
            pose_steps.append(pose_step)
            if pose_step > worst_joint_step["metres"]:
                actor, joint = np.unravel_index(np.argmax(joint_steps), joint_steps.shape)
                worst_joint_step = {"metres": pose_step, "from_frame": previous["frame"],
                                    "to_frame": frame, "actor": ("spider", "mj")[actor],
                                    "joint": int(joint)}
            root_steps.append(float(np.max(np.linalg.norm(positions[:, 0] - previous["positions"][:, 0], axis=-1))))
        for actor in range(2):
            for joint in TORSO_JOINTS:
                for box in scene["buildings"]:
                    depth = sphere_box_penetration(positions[actor, joint], .13, box)
                    if depth > .005:
                        penetration_depths.append(depth)
                        penetration_by_building[box["id"]] = penetration_by_building.get(box["id"], 0) + 1
        chain_points, chain_radii, chain_segments = segment_clearance_samples(positions)
        for box in scene["buildings"]:
            depths = box_depths(chain_points, chain_radii, box)
            overlaps = depths > .005
            roof = roof_by_building.get(box["id"])
            if roof is not None and state.get("phase") in ("ready", "pickup", "settling", "landed", "kiss"):
                # Toe/foot segments may touch the identified load-bearing roof.
                # Only samples near its plane qualify; deep entry still counts.
                near_plane = np.abs(chain_points[:, 1] - roof["position"][1]) <= .18
                allowed = overlaps & np.isin(chain_segments, tuple(FOOT_SEGMENTS)) & near_plane
                allowed_roof_contact_samples += int(allowed.sum())
                overlaps &= ~allowed
            hits = depths[overlaps]
            if len(hits):
                full_body_depths.extend(hits.tolist())
                full_body_by_building[box["id"]] = full_body_by_building.get(box["id"], 0) + int(len(hits))
                candidate = int(np.argmax(np.where(overlaps, depths, -np.inf)))
                if float(depths[candidate]) > worst_full_body["metres"]:
                    worst_full_body = {"metres": float(depths[candidate]), "frame": frame,
                                       "time": timestamp, "actor": ("spider", "mj")[candidate // 130],
                                       "segment": int(chain_segments[candidate]), "building": box["id"],
                                       "sample_point": chain_points[candidate].tolist(),
                                       "sample_radius": float(chain_radii[candidate])}
        web_hand, web_anchor = state.get("web_hand"), state.get("web_anchor")
        if web_hand is not None and web_anchor is not None:
            start, end = np.asarray(web_hand, dtype=float), np.asarray(web_anchor, dtype=float)
            if start.shape == end.shape == (3,) and np.isfinite(start).all() and np.isfinite(end).all():
                web_frames += 1
                length = float(np.linalg.norm(end - start))
                count = max(2, int(np.ceil(length / .25)) + 1)
                fraction = np.linspace(0, 1, count)
                line = start[None] * (1 - fraction[:, None]) + end[None] * fraction[:, None]
                anchor_spec = state.get("anchor") or {}
                anchor_building = anchor_spec.get("building_id")
                terminal_allowance = float(anchor_spec.get("terminal_allowance_m", .4))
                if not math.isfinite(terminal_allowance) or not .4 <= terminal_allowance <= 1.:
                    raise ValueError(f"Invalid source-geometry terminal allowance at frame {frame}")
                web_terminal_allowances.add(terminal_allowance)
                for box in scene["buildings"]:
                    depth = box_depths(line, np.full(count, .025), box)
                    if box["id"] == anchor_building:
                        depth[length * (1 - fraction) <= terminal_allowance] = 0
                    if float(depth.max()) > worst_web["metres"]:
                        worst_index = int(np.argmax(depth))
                        worst_web = {"metres": float(depth[worst_index]), "frame": frame,
                                     "building": box["id"],
                                     "fraction_hand_to_anchor": float(fraction[worst_index]),
                                     "anchor_building": anchor_building}
                    hits = depth[depth > .005]
                    if len(hits):
                        web_depths.extend(hits.tolist())
                        web_by_building[box["id"]] = web_by_building.get(box["id"], 0) + int(len(hits))
        pose = state.get("metrics", {}).get("pose", {})
        if pose.get("active"):
            active_carry_frames += 1
            for key in GRIP_KEYS:
                value = pose.get(key)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    grip_samples[key].append(float(value))
        phase = state.get("phase")
        phases.add(phase)
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        phase_starts.setdefault(phase, frame)
        for event in state.get("events", []):
            if isinstance(event, dict):
                kind = event.get("event", event.get("kind"))
                if kind:
                    events[(kind, event.get("frame"), event.get("time"))] = event
        for event in state.get("controller_events", []):
            if isinstance(event, dict) and event.get("event"):
                events[(event["event"], event.get("frame"), event.get("time"))] = event
        for command in state.get("commands", []):
            if isinstance(command, dict) and "id" in command:
                observed_commands[command["id"]] = command
                if command.get("status") in ("applied", "rejected") and command["id"] not in command_states:
                    command_states[command["id"]] = {
                        "frame": frame, "phase": phase, "status": command["status"],
                        "carry_amount": float(state.get("carry_amount", 0)),
                        "flight_amount": float(state.get("flight_amount", 0)),
                        "grounded": bool(state.get("grounded", False)),
                    }
        source = state.get("pose_provenance", {}).get("source")
        if source:
            provenance.add(source)
        previous = {"positions": positions, "frame": frame}
        final_positions, final_phase = positions, state.get("phase")
        final_landing_id = state.get("landing_zone", final_landing_id)
        last_frame, last_time = frame, timestamp
        frame_count += 1
    if frame_count < 2:
        raise ValueError("Run needs at least two valid logged frames")
    if saved_frame_limit is not None and last_frame != saved_frame_limit:
        raise ValueError("Recorded run is missing final reported frame")
    commands = saved.get("commands", [])
    sequence = action_sequence(commands)
    stateful_sequence = validate_command_states(sequence, command_states, events)
    command_latencies = [c["visible_latency_ms"] for c in commands
                         if isinstance(c.get("visible_latency_ms"), (int, float))]
    control_latencies = [c["response_latency_ms"] for c in observed_commands.values()
                         if isinstance(c.get("response_latency_ms"), (int, float))]
    roofs = scene["roofs"]
    destination = next((r for r in roofs if r["id"] == final_landing_id), None)
    if destination is None:
        destination = min(roofs, key=lambda r: np.linalg.norm(
            np.asarray(r["position"])[[0, 2]] - final_positions[0, 0, [0, 2]]))
    roof_center = np.asarray(destination["position"], dtype=float)
    supporting_box = next((b for b in scene["buildings"]
                           if b["id"] == destination["building_id"]), None)
    roof_support = {}
    for actor, name in enumerate(("spider", "mj")):
        toe = final_positions[actor, [22, 26]]
        distances = np.linalg.norm(toe[:, [0, 2]] - roof_center[[0, 2]], axis=-1)
        if supporting_box is not None:
            lo, hi = np.asarray(supporting_box["min"]), np.asarray(supporting_box["max"])
            xz_inside = bool(np.all(toe[:, [0, 2]] >= lo[[0, 2]] - .1)
                             and np.all(toe[:, [0, 2]] <= hi[[0, 2]] + .1))
        else:
            xz_inside = bool(np.max(distances) <= destination["radius"] + .15)
        roof_support[name] = {"roof_id": destination["id"],
                              "toe_plane_errors_m": np.abs(toe[:, 1] - roof_center[1]).tolist(),
                              "toe_center_distances_xz_m": distances.tolist(),
                              "roof_radius_m": destination["radius"],
                              "within_supporting_building_aabb_xz": xz_inside,
                              "within_roof_proxy": bool(xz_inside
                                                      and np.max(np.abs(toe[:, 1] - roof_center[1])) <= .20)}
    video = video_info(run_dir / "interactive-run.webm")
    scene_provenance_match = (saved.get("scene_provenance") == scene.get("provenance")
                              if "scene_provenance" in saved else None)
    native_files = []
    native_response_seconds = []
    for path in sorted(run_dir.glob("core-command-*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            rotations = archive["rotations"]
            valid = (archive["positions"].shape == (1, 40, 27, 3)
                     and rotations.shape == (1, 40, 27, 3, 3)
                     and np.isfinite(archive["positions"]).all()
                     and np.isfinite(rotations).all()
                     and np.max(np.abs(rotations @ np.swapaxes(rotations, -1, -2) - np.eye(3))) < .01
                     and np.min(np.linalg.det(rotations)) > .99
                     and metadata.get("model") == "ARDY Core")
        native_files.append({"file": path.name, "sha256": sha256_file(path),
                             "valid_core27_window": bool(valid), "prompt": metadata.get("prompt"),
                             "command_id": metadata.get("command_id")})
        if isinstance(metadata.get("response_seconds"), (int, float)):
            native_response_seconds.append(float(metadata["response_seconds"]))
    measurements = {
        "run_id": saved.get("run_id", run_dir.name), "frames": frame_count,
        "recorded_report": report_path.name if report_path.is_file() else None,
        "recorded_report_sha256": sha256_file(report_path) if report_path.is_file() else None,
        "recorded_frame_log_prefix_sha256": recorded_log_hash.hexdigest(),
        "scene_provenance_matches_current_source": scene_provenance_match,
        "first_frame": first_frame, "last_frame": last_frame,
        "duration_seconds": float(last_time - first_time), "max_frame_index_gap": max_frame_gap,
        "nonfinite_frames": nonfinite, "phases": sorted(x for x in phases if x),
        "phase_counts": phase_counts, "phase_first_frame": phase_starts,
        "events": sorted({key[0] for key in events}),
        "sequence": sequence, "command_states_at_apply": command_states,
        "stateful_sequence_complete": stateful_sequence,
        "joint_step_m": stats(pose_steps), "worst_joint_step": worst_joint_step,
        "root_step_m": stats(root_steps),
        "bone_length_drift_m": stats(bone_drifts), "rotation_orthogonality_error": stats(rotation_errors),
        "city_aabb_torso_penetration_m": stats(penetration_depths),
        "city_aabb_penetration_samples_by_building": penetration_by_building,
        "city_aabb_full_body_penetration_m": stats(full_body_depths),
        "city_aabb_full_body_samples_by_building": full_body_by_building,
        "worst_full_body_aabb_overlap": worst_full_body,
        "allowed_roof_foot_contact_samples": allowed_roof_contact_samples,
        "web_aabb_penetration_m": stats(web_depths),
        "web_aabb_samples_by_building": web_by_building, "web_frames": web_frames,
        "web_terminal_allowances_m": sorted(web_terminal_allowances),
        "worst_web_collision": worst_web,
        "active_carry_frames": active_carry_frames,
        "active_carry_grips_m": {key: stats(values) for key, values in grip_samples.items()},
        "roof_support_final": roof_support, "final_phase": final_phase,
        "visible_command_latency_ms": stats(command_latencies),
        "server_control_latency_ms": stats(control_latencies),
        "native_core_refresh_seconds": stats(native_response_seconds),
        "latency_meaning": "Server control latency measures HTTP command receipt to next physics application. Visible latency measures server receipt to browser telemetry sent after WebGL render and canvas draw; it does not measure photons on the display. Native Core refresh time is measured separately from model request start to completed model response; physics reacts before it arrives.",
        "browser_video": video, "pose_sources": sorted(provenance),
        "recording_time_difference_seconds": (None if video["duration_seconds"] is None
                                              else float(video["duration_seconds"] - (last_time - first_time))),
        "native_refresh_files": native_files,
        "native_refreshes_reported": saved.get("metrics", {}).get("model_refreshes", 0),
        "method": "Actual logged world joint poses; five point samples per Core27 bone with segment radii; 25-cm samples along web with each source-mesh-derived terminal allowance only at its anchor building; conservative building AABBs; intended roof foot contact reported separately. These are geometric proxies, not mesh contact or force proofs.",
    }
    gates = {
        "sequence": sequence["complete"],
        "stateful_command_sequence": stateful_sequence,
        "scene_source_matches_record": scene_provenance_match is not False,
        "fixed_60hz_frames": max_frame_gap == 1 and abs(measurements["duration_seconds"] - (frame_count - 1) / 60) < .05,
        "finite_poses": nonfinite == 0,
        "joint_step_under_0_25m": max(pose_steps) < .25,
        "root_step_under_0_20m": max(root_steps) < .20,
        "bone_drift_under_1cm": max(bone_drifts) < .01,
        "no_deep_torso_aabb_penetration": not penetration_depths or max(penetration_depths) < .15,
        "no_deep_full_body_aabb_penetration": not full_body_depths or max(full_body_depths) < .15,
        "web_clears_city_aabbs": web_frames >= 30 and (not web_depths or max(web_depths) < .05),
        "active_carry_contacts": active_carry_frames >= 15 and all(
            values and max(values) < .20 for values in grip_samples.values()),
        "landing_and_kiss_phases": ("settling" in phases or "landed" in phases) and "kiss" in phases,
        "final_roof_support": all(row["within_roof_proxy"] for row in roof_support.values()),
        "browser_paint_acknowledged": len(command_latencies) >= len(sequence["matched_command_ids"]),
        "native_refreshes_recorded": (len(native_files) == len(commands)
                                      and all(f["valid_core27_window"] for f in native_files)
                                      and {f["command_id"] for f in native_files} == {c["id"] for c in commands}
                                      and saved.get("metrics", {}).get("model_refreshes", 0) >= len(native_files)),
        "continuous_browser_recording": (video["present"] and video["webm_header"]
                                        and video["bytes"] >= 100_000
                                        and video["duration_seconds"] is not None
                                        and abs(video["duration_seconds"] - measurements["duration_seconds"]) <= 2.
                                        and video["video_packets"] >= video["duration_seconds"] * 15
                                        and video["max_packet_gap_seconds"] <= .25),
    }
    if not require_video:
        gates.pop("continuous_browser_recording")
    measurements["gates"] = gates
    measurements["pass"] = all(gates.values())
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-video-gate", action="store_true")
    args = parser.parse_args()
    from swing_scene import load_swing_scene
    result = audit(args.run_dir, load_swing_scene(), require_video=not args.no_video_gate)
    output = args.output or args.run_dir / "verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": result["pass"], "gates": result["gates"],
                      "output": str(output)}, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
