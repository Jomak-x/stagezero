"""Paired free-text/control trials against the private Core interaction lab.

Run from the StageZero repository, normally through an SSH forward to the
existing Pod's loopback port 8768. ``--dry-run`` builds and validates every
request without touching the GPU or reading the API token. Each returned NPZ
contains the untouched model arrays plus scene, plan, and measured metadata.

Example:
  python experiments/run_scene_interaction_trials.py --dry-run
  python experiments/run_scene_interaction_trials.py --base-url http://127.0.0.1:8768 \
      --token-file .runtime/api-token --output-dir .runtime/scene-interaction-trials
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interaction_metrics import (
    continuity, floor_motion, gate_traversal, goal_endpoint, hand_contact, joint_index,
    pair_separation,
)
from interaction_planner import plan_action
from interaction_runtime import FPS, HORIZON, MODEL_NAME, validate_request
from interaction_scene import local_axes
from interaction_scene_collision import scene_collision
from scene_objects import make_object


DEFAULT_SEEDS = (11, 22, 33, 44, 55)
SCENES = ("straight_arch", "rotated_gate", "obstacle_arch", "paired_approach")


def _scene(*objects, version=2):
    data = {"version": version, "name": "Interaction lab", "objects": list(objects),
            "effects": [], "lighting": "neutral"}
    if version == 3:
        # Geometry for a declared custom opening. The trusted passage metadata
        # below is authoritative; the asset bounding box alone proves no hole.
        data["assets"] = [{"id": "gate-asset", "name": "Rotated gate", "parts": [
            {"shape": "box", "position": [-1.3, 1.5, 0], "size": [.2, 3, .5],
             "color": [130, 130, 130]},
            {"shape": "box", "position": [1.3, 1.5, 0], "size": [.2, 3, .5],
             "color": [130, 130, 130]},
            {"shape": "box", "position": [0, 2.95, 0], "size": [2.8, .1, .5],
             "color": [130, 130, 130]}]}]
    return data


def _gate_scene(name):
    if name in ("straight_arch", "obstacle_arch"):
        arch = make_object("arch", 0)
        arch.update(id="north-arch", name="North arch", position=[0, 1.5, 2],
                    size=[2.8, 3, .4])
        objects = [arch]
        if name == "obstacle_arch":
            crate = make_object("crate", 1)
            crate.update(id="approach-crate", name="Approach crate", position=[0, .4, .3],
                         size=[.8, .8, .8])
            objects.append(crate)
        return _scene(*objects), {}, (0.0, 0.0, -1.0), "north-arch"
    if name == "rotated_gate":
        gate = make_object("arch", 0)
        gate.update(id="red-gate", name="Red gate", kind="custom", asset="gate-asset",
                    yaw=90.0, position=[2, 1.5, 0], size=[3, 3, .5])
        affordances = {"red-gate": {"kind": "passage", "verified_open": True,
                                     "width_m": 2.2, "height_m": 2.5, "depth_m": .5,
                                     "floor_y_m": 0.0,
                                     "center_xz": [2, 0], "yaw_degrees": 90.0}}
        return _scene(gate, version=3), affordances, (-1.0, 0.0, 0.0), "red-gate"
    raise ValueError(f"unknown gate scene: {name}")


def _frames_for_plan(plan):
    last_time = plan["waypoints"][-1]["time_seconds"]
    if last_time < 120 / FPS - .2:
        return 120
    if last_time < 160 / FPS - .2:
        return 160
    if last_time < 240 / FPS - .2:
        return 240
    raise ValueError("Scene route exceeds the 240-frame interaction runtime limit")


def _root_targets(plan, frames):
    """Quantize planner seconds to Core's zero-based 20 fps output frames."""
    targets = []
    for waypoint in plan["waypoints"][1:]:
        frame = round(waypoint["time_seconds"] * FPS)
        if frame >= frames:
            raise ValueError("A waypoint lies beyond the model generation horizon")
        # The planner may emit a near-duplicate route corner in one video frame.
        # Keep the later waypoint at that frame; the manifest retains the full
        # original plan so this lossy timing decision remains inspectable.
        value = {"frame": frame, "position_xz": waypoint["position_xz"]}
        if targets and targets[-1]["frame"] == frame:
            targets[-1] = value
        else:
            targets.append(value)
    return targets


def build_trial(scene_name: str, seed: int, condition: str) -> dict:
    """Build one validated JSON request with scene and expected measurements."""
    if scene_name not in SCENES or condition not in ("text", "controlled"):
        raise ValueError("unknown scene or condition")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a uint32")
    if scene_name == "paired_approach":
        scene, affordances, plan, frames = _scene(), {}, None, 120
        starts = ((0.0, -1.5), (0.0, 1.5))
        destinations = ((0.0, -.65), (0.0, .65))
        actors = []
        for index in range(2):
            actor = {"id": f"person-{index + 1}",
                     "prompt": "A person walks toward another person, slows down, and faces them.",
                     "seed": seed + index if seed < 2**32 - 1 else seed - index,
                     "initial_position_xz": list(starts[index]),
                     "initial_yaw": 0.0 if index == 0 else math.pi}
            if condition == "controlled":
                actor["root_targets"] = [
                    {"frame": 79, "position_xz": list(destinations[index]),
                     "heading": actor["initial_yaw"]},
                    {"frame": 119, "position_xz": list(destinations[index]),
                     "heading": actor["initial_yaw"]},
                ]
            actors.append(actor)
        details = {"scene": scene, "affordances": affordances, "plans": None,
                   "pair_goal_xz": [list(v) for v in destinations]}
    else:
        scene, affordances, start_xyz, gate_id = _gate_scene(scene_name)
        plan = plan_action({"verb": "go_through", "actor_id": "person-1", "target_id": gate_id},
                           scene, start_xyz, affordances=affordances, speed_mps=1.0)
        frames = _frames_for_plan(plan)
        actor = {"id": "person-1", "prompt": f"A person walks through the {scene['objects'][0]['name']}.",
                 "seed": seed, "initial_position_xz": [start_xyz[0], start_xyz[2]],
                 "initial_yaw": 0.0 if scene_name != "rotated_gate" else math.pi / 2}
        if condition == "controlled":
            actor["root_targets"] = _root_targets(plan, frames)
        actors = [actor]
        details = {"scene": scene, "affordances": affordances, "plan": plan}
    request = {"request_id": f"scene-{scene_name}-{condition}-{seed}",
               "frames": frames, "actors": actors}
    validate_request(request)
    return {"scene_name": scene_name, "condition": condition, "seed": seed,
            "request": request, **details}


def _dense_targets(plan: dict, frames: int, *, step_frames: int = 8) -> list[dict]:
    """Piecewise-linear route samples every 0.4 s, plus a final stop/hold."""
    waypoints = plan["waypoints"]
    last_frame = round(waypoints[-1]["time_seconds"] * FPS)
    sample_frames = list(range(step_frames, last_frame, step_frames)) + [last_frame]
    sample_frames = sorted(set(sample_frames))

    def route_position(frame: int) -> np.ndarray:
        seconds = frame / FPS
        if seconds >= waypoints[-1]["time_seconds"]:
            return np.asarray(waypoints[-1]["position_xz"], dtype=float)
        for left, right in zip(waypoints, waypoints[1:]):
            if seconds <= right["time_seconds"]:
                interval = right["time_seconds"] - left["time_seconds"]
                alpha = (seconds - left["time_seconds"]) / interval if interval else 1.0
                return ((1 - alpha) * np.asarray(left["position_xz"], dtype=float)
                        + alpha * np.asarray(right["position_xz"], dtype=float))
        raise AssertionError("Unreachable route time")

    targets = []
    for frame in sample_frames:
        point = route_position(frame)
        future = route_position(min(last_frame, frame + step_frames // 2))
        tangent = future - point
        if np.linalg.norm(tangent) < 1e-6:
            tangent = point - route_position(max(0, frame - step_frames // 2))
        if np.linalg.norm(tangent) < 1e-6:
            raise ValueError("Route has no heading at a dense waypoint")
        heading = math.atan2(tangent[0], tangent[1])
        targets.append({"frame": frame, "position_xz": point.tolist(),
                        "heading": heading})
    targets.append({"frame": frames - 1,
                    "position_xz": waypoints[-1]["position_xz"],
                    "heading": targets[-1]["heading"]})
    if len(targets) > 24:
        raise ValueError("Dense route exceeds runtime waypoint cap")
    return targets


def build_trial_v2(scene_name: str, seed: int) -> dict:
    """Second-pass slower dense controls and a final model-native hold goal."""
    if scene_name not in SCENES:
        raise ValueError("unknown scene")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 2:
        raise ValueError("v2 seed must be 0..2**32-2")
    if scene_name == "paired_approach":
        scene, affordances, frames = _scene(), {}, 120
        starts = ((0.0, -1.5), (0.0, 1.5))
        ends = ((0.0, -.8), (0.0, .8))
        actors = []
        for index in range(2):
            heading = 0.0 if index == 0 else math.pi
            targets = []
            for frame in (20, 40, 60, 80, 100, 119):
                fraction = min(frame / 80.0, 1.0)
                z = starts[index][1] * (1 - fraction) + ends[index][1] * fraction
                targets.append({"frame": frame, "position_xz": [0.0, z],
                                "heading": heading})
            actors.append({"id": f"person-{index + 1}",
                           "prompt": "A person slowly approaches another person, then stops and faces them.",
                           "seed": seed + index, "initial_position_xz": list(starts[index]),
                           "initial_yaw": heading, "root_targets": targets})
        details = {"scene": scene, "affordances": affordances, "plans": None,
                   "pair_goal_xz": [list(v) for v in ends]}
    else:
        scene, affordances, start_xyz, gate_id = _gate_scene(scene_name)
        plan = plan_action({"verb": "go_through", "actor_id": "person-1", "target_id": gate_id},
                           scene, start_xyz, affordances=affordances,
                           actor_radius_m=.35 if scene_name == "obstacle_arch" else .4,
                           speed_mps=.75)
        frames = _frames_for_plan(plan)
        actor = {"id": "person-1",
                 "prompt": f"A person slowly walks through the {scene['objects'][0]['name']}, then stops and stands still.",
                 "seed": seed, "initial_position_xz": [start_xyz[0], start_xyz[2]],
                 "initial_yaw": 0.0 if scene_name != "rotated_gate" else math.pi / 2,
                 "root_targets": _dense_targets(plan, frames)}
        actors = [actor]
        details = {"scene": scene, "affordances": affordances, "plan": plan}
    request = {"request_id": f"scene-v2-{scene_name}-{seed}",
               "frames": frames, "actors": actors}
    validate_request(request)
    return {"scene_name": scene_name, "condition": "dense_hold", "seed": seed,
            "request": request, **details}


def build_pair_dense_trial(seed: int) -> dict:
    """Third-pass pair-only schedule with synchronized 0.4 s spacing."""
    trial = build_trial_v2("paired_approach", seed)
    trial["condition"] = "pair_dense"
    trial["request"]["request_id"] = f"scene-pair-dense-{seed}"
    ends = ((0.0, -.9), (0.0, .9))
    trial["pair_goal_xz"] = [list(v) for v in ends]
    for index, actor in enumerate(trial["request"]["actors"]):
        start_z = actor["initial_position_xz"][1]
        end_z = ends[index][1]
        heading = actor["initial_yaw"]
        actor["root_targets"] = []
        # Place explicit goals at each autoregressive window's last frame;
        # otherwise a target at frame 80 is invisible while frames 40–79 are
        # generated and the actor can overshoot before the next window.
        for frame in sorted({*range(8, 120, 8), 39, 79, 119}):
            progress = min(frame / 80.0, 1.0)
            actor["root_targets"].append({
                "frame": frame,
                "position_xz": [0.0, start_z * (1 - progress) + end_z * progress],
                "heading": heading,
            })
    validate_request(trial["request"])
    return trial


def _read_npz(payload: bytes, expected: dict) -> tuple[dict, dict]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files if key != "metadata"}
        if "metadata" not in archive:
            raise ValueError("Runtime returned no metadata")
        metadata = json.loads(str(archive["metadata"].item()))
    if metadata.get("model") != MODEL_NAME or metadata.get("request_id") != expected["request_id"]:
        raise ValueError("Runtime returned a different model or request")
    for index in range(len(expected["actors"])):
        for suffix, shape in (("motion", (expected["frames"], 330)),
                              ("positions", (expected["frames"], 27, 3)),
                              ("rotations", (expected["frames"], 27, 3, 3))):
            key = f"actor_{index}_{suffix}"
            if key not in arrays or arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                raise ValueError(f"Runtime returned invalid {key}")
    return arrays, metadata


def _trial_metrics(trial: dict, arrays: dict, runtime: dict) -> dict:
    request = trial["request"]
    frames = request["frames"]
    actors = []
    for index, actor in enumerate(request["actors"]):
        p = arrays[f"actor_{index}_positions"]
        result = {"id": actor["id"],
                  "floor": floor_motion(p, skeleton="core27", fps=FPS),
                  "scene_collision": scene_collision(
                      p, "core27", trial["scene"], affordances=trial["affordances"]),
                  "continuity": continuity(p, skeleton="core27", fps=FPS,
                                           horizon_boundaries=tuple(range(HORIZON, frames, HORIZON)))}
        if trial["scene_name"] != "paired_approach":
            plan = trial["plan"]
            geometry = plan["geometry"]
            gate_obj = next(obj for obj in trial["scene"]["objects"] if obj["id"] == plan["target_id"])
            _, axis = local_axes(geometry["yaw_degrees"])
            # The planner chooses the entry side from the actor's start.
            center = np.asarray([gate_obj["position"][0], gate_obj["position"][2]])
            start = np.asarray(actor["initial_position_xz"])
            if np.dot(start - center, axis) > 0:
                axis = (-axis[0], -axis[1])
            waypoint_center = next(w for w in plan["waypoints"] if w["role"] == "center")
            gate_center = (*waypoint_center["position_xz"][:1], 0.0,
                           waypoint_center["position_xz"][1])
            result["gate"] = gate_traversal(
                p, skeleton="core27", center=gate_center, normal_xz=axis,
                opening_width_m=geometry["passage_width_m"],
                body_radius_m=plan["assumptions"]["actor_radius_m"],
                min_y=-.15, max_y=geometry["passage_height_m"])
            planned_frame = round(plan["waypoints"][-1]["time_seconds"] * FPS)
            exit_xz = plan["waypoints"][-1]["position_xz"]
            result["planned_exit_frame"] = planned_frame
            result["endpoint"] = goal_endpoint(
                p[:planned_frame + 1], skeleton="core27", target_xz=exit_xz,
                tolerance_m=.3)
            result["final_exit_drift"] = goal_endpoint(
                p, skeleton="core27", target_xz=exit_xz, tolerance_m=.3)
        else:
            result["endpoint"] = goal_endpoint(
                p, skeleton="core27", target_xz=trial["pair_goal_xz"][index],
                tolerance_m=.3)
        cont, floor = result["continuity"], result["floor"]
        result["movement_quality_proxy"] = {
            "mean_joint_peak_step_le_0_15_m": cont["mean_joint_peak_step_m"] is not None
            and cont["mean_joint_peak_step_m"] <= .15,
            "root_peak_speed_le_3_mps": cont["root_peak_speed_mps"] is not None
            and cont["root_peak_speed_mps"] <= 3.0,
            "toe_penetration_le_0_05_m": floor["toe_penetration_max_depth_m"] <= .05,
        }
        actors.append(result)
    outcome = {"actors": actors, "generation_seconds": runtime.get("generation_seconds"),
               "native_conditions": runtime.get("native_conditions"),
               "post_generation_pose_edits": runtime.get("post_generation_pose_edits")}
    if len(actors) == 2:
        a, b = arrays["actor_0_positions"], arrays["actor_1_positions"]
        # Mirrors ARDY's Core heading extraction from right-minus-left hip
        # (vendor/ardy/ardy/motion_rep/tools.py:compute_heading_angle).
        facing = []
        for frame in (79, 119):
            row = {"frame": frame, "facing_error_degrees": []}
            for own, other in ((a, b), (b, a)):
                right_hip = joint_index("core27", "RightUpLeg")
                left_hip = joint_index("core27", "LeftUpLeg")
                root = joint_index("core27", "Hips")
                hip = own[frame, right_hip, (0, 2)] - own[frame, left_hip, (0, 2)]
                toward = other[frame, root, (0, 2)] - own[frame, root, (0, 2)]
                if np.linalg.norm(hip) < 1e-5 or np.linalg.norm(toward) < 1e-5:
                    row["facing_error_degrees"].append(None)
                else:
                    heading = math.atan2(hip[1], -hip[0])
                    target = math.atan2(toward[0], toward[1])
                    difference = math.atan2(math.sin(heading - target),
                                             math.cos(heading - target))
                    row["facing_error_degrees"].append(abs(math.degrees(difference)))
            facing.append(row)
        outcome["pair"] = {
            "separation": pair_separation(a, b, skeleton_a="core27", skeleton_b="core27",
                                           radius_a_m=.28, radius_b_m=.28),
            "hand_proximity": hand_contact(a, b, skeleton_a="core27", skeleton_b="core27",
                                           tolerance_m=.12, minimum_duration_s=.15, fps=FPS),
            "facing_from_hip_vector": facing,
        }
    return outcome


def _get_json(base_url: str, token: str, path: str) -> dict:
    request = Request(base_url.rstrip("/") + path, headers={"Authorization": "Bearer " + token})
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def _post_npz(base_url: str, token: str, body: dict) -> bytes:
    encoded = json.dumps(body, allow_nan=False).encode()
    request = Request(base_url.rstrip("/") + "/generate", data=encoded,
                      headers={"Authorization": "Bearer " + token,
                               "Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=180) as response:
        if response.headers.get_content_type() != "application/octet-stream":
            raise ValueError("Runtime returned an unexpected content type")
        return response.read()


def _write_trial(path: Path, arrays: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays,
                            metadata=np.array(json.dumps(metadata, allow_nan=False)))
    temporary.replace(path)


def _summary(records: list[dict]) -> dict:
    """Aggregate observed outcomes; failed/missing trials stay in the denominator."""
    groups: dict[str, dict] = {}
    for record in records:
        scene, condition = record["name"].split("__", 2)[:2]
        key = f"{scene}__{condition}"
        bucket = groups.setdefault(key, {"attempted": 0, "completed": 0,
                                         "gate_traversed": 0, "endpoint_within_tolerance": 0,
                                         "scene_collision_free": 0,
                                         "endpoint_error_m": [], "generation_seconds": [],
                                         "pair_min_separation_m": []})
        bucket["attempted"] += 1
        if record["status"] != "ok":
            continue
        bucket["completed"] += 1
        metrics = record["metrics"]
        actor = metrics["actors"][0]
        if "gate" in actor and actor["gate"]["traversed_proxy"]:
            bucket["gate_traversed"] += 1
        if actor["endpoint"]["within_tolerance"]:
            bucket["endpoint_within_tolerance"] += 1
        if all(item["scene_collision"]["total_collision_frames"] == 0
               for item in metrics["actors"]):
            bucket["scene_collision_free"] += 1
        bucket["endpoint_error_m"].append(actor["endpoint"]["endpoint_error_xz_m"])
        if metrics["generation_seconds"] is not None:
            bucket["generation_seconds"].append(metrics["generation_seconds"])
        if "pair" in metrics:
            bucket["pair_min_separation_m"].append(
                metrics["pair"]["separation"]["min_root_separation_xz_m"])
    for bucket in groups.values():
        for source, result in (("endpoint_error_m", "endpoint_error_mean_m"),
                               ("generation_seconds", "generation_mean_seconds"),
                               ("pair_min_separation_m", "pair_min_separation_mean_m")):
            values = bucket.pop(source)
            bucket[result] = float(np.mean(values)) if values else None
    return groups


def run(trials: list[dict], *, base_url: str, token: str, output_dir: Path,
        resume: bool = False) -> dict:
    """Run sequential bounded trials. No GPU call occurs before ready health."""
    health = _get_json(base_url, token, "/health")
    if not health.get("ready") or health.get("model") != MODEL_NAME:
        raise RuntimeError(f"Core interaction lab is not ready: {health}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"model": MODEL_NAME, "fps": FPS, "health": health,
              "trials": [], "errors": []}
    report_path = output_dir / "report.json"
    for trial in trials:
        name = f"{trial['scene_name']}__{trial['condition']}__seed{trial['seed']}"
        path = output_dir / f"{name}.npz"
        try:
            if resume and path.exists():
                with np.load(path, allow_pickle=False) as archive:
                    metadata = json.loads(str(archive["metadata"].item()))
                if metadata.get("request") != trial["request"]:
                    raise ValueError(f"Existing trial {name} differs from current request")
            else:
                started = time.perf_counter()
                arrays, runtime = _read_npz(_post_npz(base_url, token, trial["request"]),
                                            trial["request"])
                metrics = _trial_metrics(trial, arrays, runtime)
                metadata = {"scene_name": trial["scene_name"], "condition": trial["condition"],
                            "seed": trial["seed"], "scene": trial["scene"],
                            "affordances": trial["affordances"],
                            "request": trial["request"], "runtime": runtime, "metrics": metrics,
                            "client_round_trip_seconds": time.perf_counter() - started}
                if "plan" in trial:
                    metadata["plan"] = trial["plan"]
                if "plans" in trial:
                    metadata["plans"] = trial["plans"]
                    metadata["pair_goal_xz"] = trial["pair_goal_xz"]
                _write_trial(path, arrays, metadata)
            record = {"name": name, "path": str(path), "status": "ok",
                      "metrics": metadata["metrics"],
                      "client_round_trip_seconds": metadata["client_round_trip_seconds"]}
            report["trials"].append(record)
            print(json.dumps({"trial": name, "status": "ok", "generation_seconds":
                              record["metrics"]["generation_seconds"]}), flush=True)
        except (HTTPError, URLError, TimeoutError, ValueError, RuntimeError, OSError, KeyError) as exc:
            record = {"name": name, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            report["errors"].append(record)
            print(json.dumps(record), flush=True)
        report["summary"] = _summary(report["trials"] + report["errors"])
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8768")
    parser.add_argument("--token-file", type=Path, default=Path(".runtime/api-token"))
    parser.add_argument("--output-dir", type=Path, default=Path(".runtime/scene-interaction-trials"))
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--variant", choices=("v1", "v2", "pair_dense"), default="v1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.variant == "v1":
        trials = [build_trial(scene, seed, condition)
                  for scene in args.scenes for seed in args.seeds
                  for condition in ("text", "controlled")]
    elif args.variant == "v2":
        trials = [build_trial_v2(scene, seed) for scene in args.scenes for seed in args.seeds]
    else:
        if args.scenes != list(SCENES) and args.scenes != ["paired_approach"]:
            raise ValueError("pair_dense variant only supports paired_approach")
        trials = [build_pair_dense_trial(seed) for seed in args.seeds]
    if args.dry_run:
        print(json.dumps({"model": MODEL_NAME, "trial_count": len(trials),
                          "trials": [{"scene_name": t["scene_name"],
                                      "condition": t["condition"], "seed": t["seed"],
                                      "frames": t["request"]["frames"],
                                      "root_targets": [a.get("root_targets", [])
                                                       for a in t["request"]["actors"]],
                                      "plan": t.get("plan")} for t in trials]}, indent=2))
        return 0
    token = args.token_file.read_text().strip()
    if not token:
        raise RuntimeError("Token file is empty")
    report = run(trials, base_url=args.base_url, token=token,
                 output_dir=args.output_dir, resume=args.resume)
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
