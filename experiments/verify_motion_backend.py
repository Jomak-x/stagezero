"""Bounded real HTTP probe for an isolated StageZero motion backend.

This calls an already running backend. It neither launches a model nor changes
the saved project. The default endpoint is the candidate service on port 8767;
pass an SSH tunnel's local port with --url when running from the Mac.

Example:
    python experiments/verify_motion_backend.py \
        --source-project .runtime/projects/example.stagezero.npz \
        --output .runtime/motion-research/backend-probe
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_motion import MODEL, validate_result


MOTION_POLICY = "directed-v2"
PROMPTS = {
    "overhead": "A person raises both arms overhead.",
    "squat": "A person does a squat.",
    "wave": "A person waves with their right hand.",
    "stop": "A person stops and stands still.",
}
DEFAULT_PROJECT = (
    Path(__file__).resolve().parents[1]
    / ".runtime/projects/StageZero-two-endings-20260926-015459-6a268b.stagezero.npz"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_history(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"]))
        if manifest.get("version") != 1 or manifest.get("model") != MODEL or manifest.get("fps") != 25:
            raise ValueError("Source project has an incompatible G1 model, rate, or version")
        takes = manifest.get("takes")
        if not isinstance(takes, list) or not takes:
            raise ValueError("Source project has no takes")
        first = takes[0]
        motion = np.asarray(data[f"{first['key']}_motion"][:104], dtype=np.float32)
        positions = np.asarray(data[f"{first['key']}_positions"][:104], dtype=np.float32)
        if motion.shape != (104, 414) or not np.isfinite(motion).all():
            raise ValueError("Source first take lacks a finite 104-frame G1 motion prefix")
        if positions.shape != (104, 34, 3) or not np.isfinite(positions).all():
            raise ValueError("Source first take lacks a finite 104-frame G1 pose prefix")
        for take in takes[1:]:
            other = np.asarray(data[f"{take['key']}_motion"][:104], dtype=np.float32)
            if other.shape != motion.shape or not np.array_equal(other, motion):
                raise ValueError("Source takes disagree on their shared 104-frame prefix")
    return motion[-52:].copy(), positions[-52:].copy(), {"take_key": first["key"], "history_frames": 52}


def write_report(path: Path, report: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def post(session: requests.Session, url: str, path: str, body: dict, timeout: int) -> requests.Response:
    return session.post(url + path, json=body, timeout=(5, timeout))


def decode_generation(response: requests.Response, request_id: str, seed: int, action: str | None,
                      *, expect_pose_goal: bool = False, explicit_profile: str | None = None) -> dict:
    if response.status_code != 200:
        detail = response.text[:300]
        raise RuntimeError(f"HTTP {response.status_code}: {detail}")
    with np.load(io.BytesIO(response.content), allow_pickle=False) as data:
        result = {name: data[name].copy() for name in ("positions", "rotations", "motion")}
        result["metadata"] = json.loads(str(data["metadata"]))
    validate_result(result, request_id)
    meta = result["metadata"]
    if meta.get("frames") != 104 or meta.get("motion_policy") != MOTION_POLICY:
        raise ValueError("Unexpected frame count or motion policy")
    if meta.get("base_seed") != seed or meta.get("selection", {}).get("action") != action:
        raise ValueError("Incorrect base seed or recognized action")
    settings = meta.get("candidate_settings")
    selection = meta["selection"]
    assessments = selection.get("assessments")
    chosen = selection.get("chosen_index")
    expected_count = 2 if (action in ("overhead", "squat")
                           and not expect_pose_goal and explicit_profile is None) else 1
    if (not isinstance(settings, list) or len(settings) != expected_count
            or not isinstance(assessments, list) or len(assessments) != expected_count
            or type(chosen) is not int or not 0 <= chosen < expected_count):
        raise ValueError("Candidate count or chosen index is inconsistent")
    if settings[chosen].get("seed") != meta.get("seed") or not assessments[chosen].get("safe"):
        raise ValueError("Chosen seed does not match metadata or selected candidate is unsafe")
    if explicit_profile is not None:
        expected_profiles = [explicit_profile]
    elif expected_count == 2:
        expected_profiles = ["responsive", "expressive"]
    else:
        expected_profiles = ["expressive" if action in ("squat", "stop") else "responsive"]
    if [entry.get("profile") for entry in settings] != expected_profiles:
        raise ValueError("Automatic profile order is incorrect")
    if [entry.get("seed") for entry in settings] != [seed] * expected_count:
        raise ValueError("Paired automatic candidates did not use the requested seed")
    if any(entry.get("index") != index for index, entry in enumerate(assessments)):
        raise ValueError("Candidate assessments do not match generation order")
    goal_metadata = meta.get("pose_goal")
    if expect_pose_goal and not goal_metadata:
        raise ValueError("Expected automatic pose goal conditioning metadata")
    if not expect_pose_goal and goal_metadata:
        raise ValueError("Pose goal applied to an unconditioned request")
    return result


def case_summary(result: dict, elapsed: float, prior_positions: np.ndarray) -> dict:
    meta = result["metadata"]
    chosen = meta["selection"]["chosen_index"]
    assessment = meta["selection"]["assessments"][chosen]
    external_seam = float(np.linalg.norm(result["positions"][0] - prior_positions[-1], axis=1).mean())
    if external_seam > .15:
        raise ValueError(f"Source-to-generation seam exceeds 0.15 m: {external_seam:.4f} m")
    return {
        "status": "ok", "request_id": meta["request_id"], "seed": meta["seed"],
        "profile": meta["candidate_settings"][chosen]["profile"],
        "chosen_index": chosen, "candidate_count": len(meta["candidate_settings"]),
        "action_proxy_met": assessment["proxy"]["met"],
        "pose_goal": meta.get("pose_goal"),
        "source_seam_m": external_seam,
        "mean_seam_m": assessment["seam_mean_joint_m"],
        "floor_penetration_m": assessment["floor_penetration_max_depth_m"],
        "backend_generation_seconds": meta.get("generation_seconds"),
        "round_trip_seconds": elapsed,
    }


def save_result(path: Path, result: dict) -> None:
    np.savez_compressed(
        path, positions=result["positions"], rotations=result["rotations"],
        motion=result["motion"], metadata=np.array(json.dumps(result["metadata"])),
    )


def probe_targets(session: requests.Session, args: argparse.Namespace, history: np.ndarray,
                  prior_positions: np.ndarray, report: dict, report_path: Path) -> None:
    """Compare two native root targets against seed-matched free generations.

    The target result is reported separately from the ordinary backend gate:
    a conditioned model can be healthy while missing the requested location.
    """
    prompt = "A person walks forward."
    prior_root = prior_positions[-1, 0, [0, 2]]
    target_xz = (prior_root + np.array([0.0, 0.8])).tolist()
    section = report["target_probe"] = {
        "status": "running", "prompt": prompt, "position_xz": target_xz,
        "prior_root_xz": prior_root.tolist(), "threshold_m": 0.10,
        "cases": {}, "checks": {},
    }
    write_report(report_path, report)
    for seed in (101, 102):
        key = f"unconditioned-seed{seed}"
        request_id = f"motion-target-free-{uuid.uuid4().hex}"
        body = {"request_id": request_id, "prompt": prompt, "history": history.tolist(),
                "generation_options": {"seed": seed, "pose_goal": False}}
        started = time.perf_counter()
        try:
            response = post(session, args.url, "/generate", body, args.request_timeout)
            result = decode_generation(response, request_id, seed, None)
            meta = result["metadata"]
            if meta.get("motion_target") is not None or meta.get("target_error_m") is not None:
                raise ValueError("Unconditioned request unexpectedly reports a target")
            row = case_summary(result, time.perf_counter() - started, prior_positions)
            row["target_error_frame51_m"] = float(np.linalg.norm(result["positions"][51, 0, [0, 2]] - target_xz))
            row["target_error_frame103_m"] = float(np.linalg.norm(result["positions"][103, 0, [0, 2]] - target_xz))
            section["cases"][key] = row
            save_result(args.output / f"target-{key}.npz", result)
        except Exception as exc:
            section["cases"][key] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        write_report(report_path, report)
        print(f"target {key}: {section['cases'][key]['status']}", flush=True)
        for frame in (51, 103):
            key = f"conditioned-frame{frame}-seed{seed}"
            request_id = f"motion-target-{uuid.uuid4().hex}"
            requested_target = {"position_xz": target_xz, "frame": frame}
            body = {"request_id": request_id, "prompt": prompt, "history": history.tolist(),
                    "generation_options": {"seed": seed, "pose_goal": False}, "motion_target": requested_target}
            started = time.perf_counter()
            try:
                response = post(session, args.url, "/generate", body, args.request_timeout)
                result = decode_generation(response, request_id, seed, None)
                meta = result["metadata"]
                if meta.get("motion_target") != requested_target:
                    raise ValueError("Backend motion_target metadata differs from request")
                measured = float(np.linalg.norm(result["positions"][frame, 0, [0, 2]] - target_xz))
                reported = meta.get("target_error_m")
                if not isinstance(reported, (int, float)) or not np.isfinite(reported):
                    raise ValueError("Backend target_error_m is absent or nonfinite")
                if not np.isclose(measured, reported, atol=1e-5, rtol=0):
                    raise ValueError(f"Backend target_error_m mismatch: {reported} versus {measured}")
                row = case_summary(result, time.perf_counter() - started, prior_positions)
                row.update({"frame": frame, "target_error_m": measured,
                            "target_met": bool(measured < .10),
                            "unconditioned_error_m": section["cases"][f"unconditioned-seed{seed}"].get(
                                f"target_error_frame{frame}_m")})
                section["cases"][key] = row
                save_result(args.output / f"target-{key}.npz", result)
            except Exception as exc:
                section["cases"][key] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            write_report(report_path, report)
            print(f"target {key}: {section['cases'][key]['status']}", flush=True)

    request_id = f"motion-target-nohistory-{uuid.uuid4().hex}"
    try:
        response = post(session, args.url, "/generate", {
            "request_id": request_id, "prompt": prompt, "history": None,
            "generation_options": {"seed": 101, "pose_goal": False},
            "motion_target": {"position_xz": target_xz, "frame": 51},
        }, 15)
        section["checks"]["target_without_history"] = {
            "passed": response.status_code == 400, "http_status": response.status_code,
        }
    except Exception as exc:
        section["checks"]["target_without_history"] = {
            "passed": False, "error": f"{type(exc).__name__}: {exc}",
        }
    section["status"] = "passed" if (
        len(section["cases"]) == 6
        and all(row["status"] == "ok" for row in section["cases"].values())
        and all(row["target_met"] for key, row in section["cases"].items() if key.startswith("conditioned"))
        and all(check["passed"] for check in section["checks"].values())
    ) else "failed"
    write_report(report_path, report)


def run(args: argparse.Namespace) -> dict:
    project = args.source_project.resolve(strict=True)
    source_digest = sha256(project)
    history, prior_positions, history_info = source_history(project)
    token = args.token_file.read_text().strip()
    if not token:
        raise ValueError("API token file is empty")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report = {
        "url": args.url, "model": MODEL, "source_project": str(project),
        "source_sha256_before": source_digest, "history": history_info,
        "cases": {}, "checks": {}, "status": "running",
    }
    write_report(report_path, report)
    with requests.Session() as session:
        session.headers.update({"Authorization": "Bearer " + token})
        health = session.get(args.url + "/health", timeout=(5, 10))
        if health.status_code != 200:
            raise RuntimeError(f"Candidate backend not ready: HTTP {health.status_code} {health.text[:300]}")
        state = health.json()
        if state.get("model") != MODEL or state.get("motion_policy") != MOTION_POLICY or not state.get("ready"):
            raise ValueError("Candidate backend health does not match the directed G1 policy")
        report["health"] = {key: state.get(key) for key in ("ready", "model", "motion_policy", "gpu", "load_seconds")}
        write_report(report_path, report)
        reference = None
        # Normal run: ten goal clips, four paired no-goal controls, four
        # wave/stop controls, one explicit-profile default, and one repeat =
        # 20 model calls. --probe-targets limits goal seeds to two so its six
        # root waypoint calls still keep the invocation at 20 model calls.
        goal_seeds = (101, 102) if args.probe_targets else (101, 102, 103, 104, 105)
        planned_cases = []
        for action in ("overhead", "squat"):
            for seed in goal_seeds:
                planned_cases.append((f"{action}-seed{seed}", action, seed, True, None, None))
        for action in ("overhead", "squat"):
            for seed in (101, 102):
                planned_cases.append((f"{action}-nogoal-seed{seed}", action, seed, False, False, None))
        for action in ("wave", "stop"):
            for seed in (101, 102):
                planned_cases.append((f"{action}-seed{seed}", action, seed, False, None, None))
        planned_cases.append(("overhead-explicit-responsive-seed101", "overhead", 101,
                              False, None, "responsive"))
        report["planned_generation_calls"] = len(planned_cases) + 1 + (6 if args.probe_targets else 0)
        write_report(report_path, report)
        for key, action, seed, expect_pose_goal, pose_goal_option, profile_option in planned_cases:
            request_id = f"motion-probe-{uuid.uuid4().hex}"
            options = {"seed": seed}
            if pose_goal_option is not None:
                options["pose_goal"] = pose_goal_option
            if profile_option is not None:
                options["profile"] = profile_option
            body = {"request_id": request_id, "prompt": PROMPTS[action],
                    "history": history.tolist(), "generation_options": options}
            started = time.perf_counter()
            try:
                response = post(session, args.url, "/generate", body, args.request_timeout)
                result = decode_generation(response, request_id, seed, action,
                                           expect_pose_goal=expect_pose_goal,
                                           explicit_profile=profile_option)
                elapsed = time.perf_counter() - started
                row = case_summary(result, elapsed, prior_positions)
                row["requested_pose_goal"] = pose_goal_option
                row["requested_profile"] = profile_option
                save_result(args.output / f"{key}.npz", result)
                report["cases"][key] = row
                if action == "overhead" and seed == 101 and expect_pose_goal:
                    reference = result
            except Exception as exc:
                report["cases"][key] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            write_report(report_path, report)
            print(f"{key}: {report['cases'][key]['status']}", flush=True)

        report["paired_goal_comparison"] = {}
        for action in ("overhead", "squat"):
            for seed in (101, 102):
                goal = report["cases"][f"{action}-seed{seed}"]
                baseline = report["cases"][f"{action}-nogoal-seed{seed}"]
                report["paired_goal_comparison"][f"{action}-seed{seed}"] = {
                    "goal_proxy_met": goal.get("action_proxy_met") if goal["status"] == "ok" else None,
                    "no_goal_proxy_met": baseline.get("action_proxy_met") if baseline["status"] == "ok" else None,
                    "goal_status": goal["status"], "no_goal_status": baseline["status"],
                }
        write_report(report_path, report)

        # Repeat one full generation with a fresh request id. Timings and IDs
        # may differ; every generated sample array should be byte identical.
        if reference is not None:
            request_id = f"motion-probe-repeat-{uuid.uuid4().hex}"
            body = {"request_id": request_id, "prompt": PROMPTS["overhead"],
                    "history": history.tolist(), "generation_options": {"seed": 101}}
            try:
                response = post(session, args.url, "/generate", body, args.request_timeout)
                repeated = decode_generation(response, request_id, 101, "overhead", expect_pose_goal=True)
                equal = all(np.array_equal(reference[name], repeated[name]) for name in ("positions", "rotations", "motion"))
                report["checks"]["deterministic_repeat"] = {"passed": bool(equal), "seed": 101,
                    "arrays": ["positions", "rotations", "motion"]}
            except Exception as exc:
                report["checks"]["deterministic_repeat"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
            write_report(report_path, report)
        else:
            report["checks"]["deterministic_repeat"] = {"passed": False, "error": "Reference generation failed"}
            write_report(report_path, report)

        malformed = {"request_id": f"motion-probe-bad-{uuid.uuid4().hex}",
                     "prompt": PROMPTS["wave"], "history": history.tolist(),
                     "generation_options": {"seed": True}}
        try:
            response = post(session, args.url, "/generate", malformed, 15)
            report["checks"]["malformed_options"] = {"passed": response.status_code == 400,
                                                        "http_status": response.status_code}
        except Exception as exc:
            report["checks"]["malformed_options"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        write_report(report_path, report)

        cancel_id = f"motion-probe-cancel-{uuid.uuid4().hex}"
        try:
            cancelled = post(session, args.url, "/cancel", {"request_id": cancel_id}, 15)
            generated = post(session, args.url, "/generate", {
                "request_id": cancel_id, "prompt": PROMPTS["wave"],
                "history": history.tolist(), "generation_options": {"seed": 101},
            }, 15)
            report["checks"]["cancel_before_generate"] = {
                "passed": cancelled.status_code == 200 and generated.status_code == 409,
                "cancel_http_status": cancelled.status_code, "generate_http_status": generated.status_code,
            }
        except Exception as exc:
            report["checks"]["cancel_before_generate"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        write_report(report_path, report)

        if args.probe_targets:
            probe_targets(session, args, history, prior_positions, report, report_path)

    source_digest_after = sha256(project)
    report["source_sha256_after"] = source_digest_after
    report["checks"]["source_unchanged"] = {"passed": source_digest == source_digest_after}
    report["status"] = "passed" if (
        all(case["status"] == "ok" for case in report["cases"].values())
        and len(report["cases"]) == len(planned_cases)
        and all(check["passed"] for check in report["checks"].values())
    ) else "failed"
    write_report(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8767", help="Already running candidate backend URL")
    parser.add_argument("--token-file", type=Path, default=Path(".runtime/api-token"))
    parser.add_argument("--source-project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--output", type=Path, required=True, help="Private result directory; saves rolling report and NPZ clips")
    parser.add_argument("--probe-targets", action="store_true", help="Also test two seeds at frames 51 and 103 against seed-matched free motion")
    parser.add_argument("--request-timeout", type=int, default=300, help="Seconds for each real generation request")
    args = parser.parse_args()
    args.url = args.url.rstrip("/")
    if not args.url.startswith("http://127.0.0.1:") and not args.url.startswith("http://localhost:"):
        parser.error("Only a loopback HTTP endpoint is accepted")
    if args.request_timeout < 15 or args.request_timeout > 1800:
        parser.error("--request-timeout must be 15–1800 seconds")
    try:
        report = run(args)
    except Exception as exc:
        print(f"Probe setup failed: {type(exc).__name__}: {exc}", flush=True)
        return 2
    target_status = report.get("target_probe", {}).get("status")
    suffix = f"; target probe {target_status}" if target_status else ""
    print(f"Backend probe {report['status']}{suffix}; report: {args.output / 'report.json'}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
