"""Small playback controller: latest instruction wins, local controls never wait on GPU."""
import io
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import requests

MODEL = "ARDY-G1-RP-25FPS-Horizon52"


class Backend:
    def __init__(self, token_path, url="http://127.0.0.1:8765"):
        self.url = url
        self.token = Path(token_path).read_text().strip()

    def call(self, path, body=None, timeout=90):
        headers = {"Authorization": "Bearer " + self.token}
        try:
            response = requests.get(self.url + path, headers=headers, timeout=(3, timeout)) if body is None else requests.post(self.url + path, json=body, headers=headers, timeout=(3, timeout))
        except requests.Timeout as exc:
            raise RuntimeError("Pod took too long to respond; check the connection and retry") from exc
        except requests.ConnectionError as exc:
            raise RuntimeError("Pod connection unavailable; restart run-live.command and retry") from exc
        if response.status_code != 200:
            try:
                detail = response.json().get("error") or "Model is still loading"
            except ValueError:
                detail = f"Backend HTTP {response.status_code}"
            raise RuntimeError(detail)
        return response

    def cancel(self, request_id):
        try:
            self.call("/cancel", {"request_id": request_id}, timeout=3)
        except Exception:
            pass  # Local version checks remain authoritative even without the tunnel.

    def generate(self, request_id, prompt, history):
        response = self.call("/generate", {"request_id": request_id, "prompt": prompt,
                                          "history": None if history is None else history.tolist()})
        with np.load(io.BytesIO(response.content), allow_pickle=False) as data:
            result = {name: data[name].copy() for name in ("positions", "rotations", "motion")}
            result["metadata"] = json.loads(str(data["metadata"]))
        validate_result(result, request_id)
        return result


def validate_result(result, request_id):
    p, r, m, meta = (result[k] for k in ("positions", "rotations", "motion", "metadata"))
    if meta["request_id"] != request_id or meta["model"] != MODEL or meta["fps"] != 25:
        raise ValueError("Backend returned a different request or incompatible checkpoint")
    if p.shape != (104, 34, 3) or r.shape != (104, 34, 3, 3) or m.shape != (104, 414):
        raise ValueError("Backend returned an incompatible G1 skeleton or clip length")
    if any(not np.isfinite(a).all() for a in (p, r, m)):
        raise ValueError("Backend returned invalid motion values")
    if not np.allclose(r @ np.swapaxes(r, -1, -2), np.eye(3), atol=0.02) or not np.allclose(np.linalg.det(r), 1, atol=0.02):
        raise ValueError("Backend returned invalid joint rotations")


class MotionSession:
    def __init__(self, backend, recorded_positions, recorded_rotations, metrics_path=None):
        self.backend = backend
        self.recorded = (np.asarray(recorded_positions), np.asarray(recorded_rotations))
        self.positions, self.rotations = self.recorded
        self.motion = None
        self.fps = 60
        self.frame = 0
        self.playing = False
        self.mode = "Recorded preview"
        self.kind = "recorded"
        self.status = "Recorded playback — no AI generation"
        self.prompt = ""
        self.version = 0
        self.current_id = None
        self.pending = None
        self.busy = False
        self.resume_after_generation = True
        self.started = 0.0
        self.clip_revision = 0
        self.metrics = None
        self.needs_ack = None
        self.metrics_path = metrics_path
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    def _invalidate(self):
        self.version += 1
        if self.current_id:
            threading.Thread(target=self.backend.cancel, args=(self.current_id,), daemon=True).start()
        self.current_id = None
        self.pending = None
        self.busy = False
        self.needs_ack = None

    def set_mode(self, mode):
        with self.lock:
            self._invalidate()
            self.mode = mode
            self.positions, self.rotations = self.recorded
            self.motion = None
            self.fps = 60
            self.frame = 0
            self.playing = False
            self.kind = "recorded" if mode == "Recorded preview" else "reference"
            self.metrics = None
            self.clip_revision += 1
            self.status = "Recorded playback — no AI generation" if self.kind == "recorded" else "Ready for an instruction · holding reference pose"

    def edit_prompt(self, prompt):
        with self.lock:
            self.prompt = prompt
            if self.busy:
                self._invalidate()
                self.status = "Instruction changed · press Generate to submit"

    def submit(self, prompt):
        with self.lock:
            prompt = prompt.strip()
            if self.mode != "Live ARDY":
                return
            if not 1 <= len(prompt) <= 500:
                self.status = "Enter a movement instruction (1–500 characters)"
                return
            self._invalidate()
            self.prompt = prompt
            self.playing = False
            self.resume_after_generation = True
            history = None
            if self.motion is not None:
                n = min(52, self.frame + 1) // 4 * 4
                if n:
                    history = self.motion[self.frame + 1 - n:self.frame + 1].copy()
            request_id = str(uuid.uuid4())
            self.current_id = request_id
            self.busy = True
            self.status = "Generating next 4.16 s · holding pose; camera stays active"
            self.pending = (self.version, request_id, prompt, history, time.perf_counter())
            self.wake.set()

    def pause(self):
        with self.lock:
            self.playing = False
            self.resume_after_generation = False

    def play(self):
        with self.lock:
            if self.busy:
                self.resume_after_generation = True
            elif self.kind in ("generated", "recorded"):
                if self.frame >= len(self.positions) - 1:
                    self.frame = 0
                self.started = time.perf_counter() - self.frame / self.fps
                self.playing = True

    def reset(self):
        with self.lock:
            self._invalidate()
            self.playing = False
            self.frame = 0
            self.status = "Reset · next instruction starts a fresh take" if self.mode == "Live ARDY" else "Recorded playback — no AI generation"
            # A reset deliberately clears generated history and returns to the reference pose.
            if self.mode == "Live ARDY":
                self.positions, self.rotations = self.recorded
                self.motion = None
                self.fps = 60
                self.kind = "reference"
                self.metrics = None
                self.clip_revision += 1

    def tick(self):
        with self.lock:
            if self.playing:
                self.frame = min(int((time.perf_counter() - self.started) * self.fps), len(self.positions) - 1)
                if self.frame == len(self.positions) - 1:
                    self.playing = False
            return self.clip_revision, self.frame

    def _work(self):
        while True:
            self.wake.wait()
            with self.lock:
                job = self.pending
                self.pending = None
                self.wake.clear()
            if job is None:
                continue
            version, request_id, prompt, history, submitted = job
            try:
                result = self.backend.generate(request_id, prompt, history)
                validate_result(result, request_id)
                with self.lock:
                    if version != self.version or request_id != self.current_id:
                        continue
                    self.positions, self.rotations, self.motion = (result[k] for k in ("positions", "rotations", "motion"))
                    self.fps = 25
                    self.frame = 0
                    self.kind = "generated"
                    self.clip_revision += 1
                    self.busy = False
                    self.status = "Fresh ARDY motion · complete segment received (not streaming)"
                    self.metrics = {**result["metadata"], "command_to_received_seconds": time.perf_counter() - submitted}
                    self.needs_ack = (request_id, submitted)
                    self.started = time.perf_counter()
                    self.playing = self.resume_after_generation
            except Exception as exc:
                with self.lock:
                    if version != self.version:
                        continue
                    self.busy = False
                    self.playing = False
                    self.status = f"Generation failed · {type(exc).__name__}: {str(exc)[:200]}. Retry or use Recorded preview."

    def record_ack(self, request_id, elapsed, screenshot=None):
        with self.lock:
            if self.metrics is None or self.metrics["request_id"] != request_id:
                return
            self.metrics["command_to_browser_render_ack_seconds"] = elapsed
            if self.metrics_path:
                path = Path(self.metrics_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as file:
                    file.write(json.dumps(self.metrics) + "\n")
