"""Private, single-GPU ARDY G1 adapter. Run only on the existing Pod.

Uses the official load_model -> autoregressive_step -> motion_rep.inverse path.
No TensorRT, alternative encoder, or custom motion synthesis.
"""
import io
import json
import os
from pathlib import Path
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import psutil
import torch
from ardy.model import load_model
from ardy.tools import seed_everything
from motion_policy import PROFILES, recognize_action, select_candidate, validate_generation_options
from motion_constraints import build_root_conditions, validate_motion_target
from motion_action_goals import build_action_conditions

MODEL = "ARDY-G1-RP-25FPS-Horizon52"
TOKEN = Path(os.environ.get("STAGEZERO_TOKEN_FILE", ".runtime/api-token")).read_text().strip()
STATE = {"ready": False, "error": None, "model": MODEL, "motion_policy": "directed-v2"}
MODEL_LOCK = threading.Lock()
CANCEL_LOCK = threading.Lock()
CANCELLED = set()
model = None


def cancelled(request_id):
    with CANCEL_LOCK:
        return request_id in CANCELLED


def load():
    global model
    start = time.perf_counter()
    try:
        torch.set_num_threads(4)
        model = load_model(MODEL, device="cuda", text_encoder_mode="local")
        assert model.skeleton.name == "g1skel34" and model.skeleton.nbjoints == 34
        assert model.motion_rep.fps == 25 and model.gen_horizon_len == 52
        STATE.update(ready=True, load_seconds=time.perf_counter() - start,
                     fps=25, joints=34, token_frames=model.num_frames_per_token,
                     features=model.motion_rep.motion_rep_dim,
                     torch=torch.__version__, gpu=torch.cuda.get_device_name())
        print(json.dumps({"event": "ready", **STATE}), flush=True)
    except Exception as exc:
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        print(STATE["error"], flush=True)


def generate(body):
    request_id = body["request_id"]
    if not isinstance(body.get("prompt"), str):
        raise ValueError("Instruction must be text")
    prompt = body["prompt"].strip()
    if not prompt or len(prompt) > 500 or not isinstance(request_id, str) or len(request_id) > 100:
        raise ValueError("Provide an instruction of 1–500 characters")
    if not STATE["ready"]:
        raise RuntimeError(STATE["error"] or "Model is still loading")
    history = body.get("history")
    if history is not None:
        history = np.asarray(history, dtype=np.float32)
        if (history.ndim != 2 or history.shape[1] != STATE["features"]
                or not 4 <= len(history) <= 52 or len(history) % model.num_frames_per_token
                or not np.isfinite(history).all()):
            raise ValueError("Invalid G1 motion history")
    supplied_options = body.get("generation_options")
    options = validate_generation_options(supplied_options)
    # Explicit profiles preserve text-only comparisons unless a pose goal is
    # requested. Automatic goals use native model conditions, not pose edits.
    automatic = supplied_options is None or "profile" not in supplied_options
    action = recognize_action(prompt)
    if options["pose_goal"] is True and action not in ("overhead", "squat"):
        raise ValueError("Explicit pose goals require an unambiguous overhead raise or squat instruction")
    use_goal = automatic if options["pose_goal"] is None else options["pose_goal"]
    if body.get("motion_target") is not None:
        if options["pose_goal"] is True:
            raise ValueError("Choose either a root waypoint or a reference pose goal")
        use_goal = False
    goal_action = action if use_goal and action in ("overhead", "squat") else None
    if automatic:
        profile = "expressive" if action == "stop" or goal_action == "squat" else "responsive"
        options.update(profile=profile, **PROFILES[profile])
        if supplied_options is None or "candidates" not in supplied_options:
            options["candidates"] = 2 if goal_action is None and action in ("overhead", "squat") else 1
    base_seed = options["seed"] if options["seed"] is not None else secrets.randbelow(2**32)
    queued = time.perf_counter()
    with MODEL_LOCK, torch.inference_mode():
        if cancelled(request_id):
            raise InterruptedError("Superseded instruction")
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        prior_positions = prior_rotations = None
        if history is not None:
            previous = model.motion_rep.inverse(torch.from_numpy(history).unsqueeze(0).to("cuda"), is_normalized=True)
            prior_positions = previous["posed_joints"][0].cpu().numpy()
            prior_rotations = previous["global_rot_mats"][0].cpu().numpy()
        target = validate_motion_target(body.get("motion_target"),
            prior_root_xz=None if prior_positions is None else prior_positions[-1, 0, [0, 2]])
        # Every submitted instruction is freshly encoded; no prompt/result cache.
        encoding_started = time.perf_counter()
        feat, mask = model._encode_text([prompt])
        torch.cuda.synchronize()
        encoded = time.perf_counter()
        candidates, candidate_settings, step_times = [], [], []
        for candidate_index in range(options["candidates"]):
            if cancelled(request_id):
                raise InterruptedError("Superseded instruction")
            profile = "expressive" if automatic and goal_action is None and candidate_index == 1 else options["profile"]
            settings = PROFILES[profile]
            # Paired profile alternatives use the same seed; repeat samples
            # requested within an explicit profile use different recorded seeds.
            seed = (base_seed + (candidate_index if not automatic or goal_action else max(0, candidate_index - 1))) % 2**32
            seed_everything(seed)
            current = None if history is None else torch.from_numpy(history[-settings["history_frames"]:]).unsqueeze(0).to("cuda")
            chunks, candidate_steps, goal_metadata = [], [], None
            for generated_offset in (0, 52):
                if cancelled(request_id):
                    raise InterruptedError("Superseded instruction")
                before = time.perf_counter()
                history_len = 0 if current is None else current.shape[1]
                observed, motion_mask = build_root_conditions(model, target,
                    history_length=history_len, generated_offset=generated_offset, device="cuda")
                if goal_action is not None:
                    observed, motion_mask, goal_metadata = build_action_conditions(
                        model, goal_action, current, generated_offset=generated_offset, device="cuda")
                result = model.autoregressive_step(
                    num_frames=history_len + 52,
                    num_denoising_steps=model.diffusion.num_base_steps,
                    motion_mask=motion_mask, observed_motion=observed,
                    cfg_weight=settings["cfg_weight"], text_feat=feat, text_pad_mask=mask,
                    init_history_sequence=current,
                )
                chunks.append(result[:, history_len:history_len + 52])
                current = result[:, -settings["carry_frames"]:]
                torch.cuda.synchronize()
                candidate_steps.append(time.perf_counter() - before)
            motion = torch.cat(chunks, dim=1)
            output = model.motion_rep.inverse(motion, is_normalized=True)
            candidates.append({"positions": output["posed_joints"][0].cpu().numpy(),
                               "rotations": output["global_rot_mats"][0].cpu().numpy(),
                               "motion": motion[0].cpu().numpy()})
            candidate_settings.append({"profile": profile, "seed": seed,
                                       "history_frames": 0 if history is None else min(len(history), settings["history_frames"]),
                                       "carry_frames": settings["carry_frames"], "cfg_weight": list(settings["cfg_weight"]),
                                       "pose_goal": goal_metadata,
                                       "step_seconds": candidate_steps})
            step_times.extend(candidate_steps)
        if cancelled(request_id):
            raise InterruptedError("Superseded instruction")
        selection = select_candidate(prompt,
            [{"positions": c["positions"], "rotations": c["rotations"]} for c in candidates],
            prior_positions=prior_positions, prior_rotations=prior_rotations)
        chosen = selection["chosen_index"]
        if chosen is None:
            raise RuntimeError("Generated candidates failed motion-quality checks; no motion committed. Retry the instruction.")
        arrays = candidates[chosen]
        if any(not np.isfinite(a).all() for a in arrays.values()):
            raise RuntimeError("Model returned non-finite motion")
        torch.cuda.synchronize()
        metrics = {
            "request_id": request_id, "prompt": prompt, "model": MODEL,
            "fps": 25, "frames": len(arrays["positions"]), "history_frames": candidate_settings[chosen]["history_frames"],
            "motion_policy": "directed-v2", "seed": candidate_settings[chosen]["seed"], "base_seed": base_seed,
            "candidate_settings": candidate_settings, "selection": selection,
            "pose_goal": candidate_settings[chosen]["pose_goal"],
            "motion_target": target,
            "target_error_m": None if target is None else float(np.linalg.norm(
                arrays["positions"][target["frame"], 0, [0, 2]] - target["position_xz"])),
            "queue_seconds": start - queued, "encoding_seconds": encoded - encoding_started,
            "step_seconds": step_times, "generation_seconds": time.perf_counter() - start,
            "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "process_rss_gib": psutil.Process().memory_info().rss / 2**30,
        }
        data = io.BytesIO()
        np.savez_compressed(data, **arrays, metadata=np.array(json.dumps(metrics)))
        with Path("metrics.jsonl").open("a") as metrics_file:
            metrics_file.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
        return data.getvalue()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, status, data, content_type="application/json"):
        if not isinstance(data, bytes):
            data = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def authorized(self):
        if not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + TOKEN):
            self.reply(401, {"error": "Unauthorized"})
            return False
        return True

    def do_GET(self):
        if self.authorized():
            self.reply(200 if STATE["ready"] else 503, STATE) if self.path == "/health" else self.reply(404, {"error": "Not found"})

    def do_POST(self):
        if not self.authorized():
            return
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 2_000_000:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(size))
            if self.path == "/cancel":
                with CANCEL_LOCK:
                    if len(CANCELLED) > 1024:
                        CANCELLED.clear()
                    CANCELLED.add(body["request_id"])
                self.reply(200, {"cancelled": True})
            elif self.path == "/generate":
                self.reply(200, generate(body), "application/octet-stream")
            else:
                self.reply(404, {"error": "Not found"})
        except InterruptedError as exc:
            self.reply(409, {"error": str(exc)})
        except (ValueError, KeyError, TypeError) as exc:
            self.reply(400, {"error": str(exc)})
        except Exception as exc:
            print(f"Request failed: {type(exc).__name__}: {exc}", flush=True)
            self.reply(503, {"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    threading.Thread(target=load, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("STAGEZERO_PORT", "8765"))), Handler).serve_forever()
