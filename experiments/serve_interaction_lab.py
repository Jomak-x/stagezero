"""Private ARDY Core two-actor lab service on the existing CUDA Pod.

Run from the StageZero repository with vendor/ardy on PYTHONPATH. This
process is isolated from the live G1 backend. It binds only to loopback and
uses the existing private API token. POST /generate returns a compressed NPZ.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interaction_runtime import GenerationCancelled, InteractionRuntime, MODEL_NAME, encode_npz, validate_request


HOST = "127.0.0.1"
PORT = int(os.environ.get("STAGEZERO_INTERACTION_PORT", "8768"))
TOKEN = Path(os.environ.get("STAGEZERO_TOKEN_FILE", ".runtime/api-token")).read_text().strip()
STATE = {"ready": False, "error": None, "model": MODEL_NAME, "service": "interaction-lab-v1"}
MODEL_LOCK = threading.Lock()
CANCEL_LOCK = threading.Lock()
CANCELLED: set[str] = set()
ACTIVE: set[str] = set()
runtime: InteractionRuntime | None = None


def cancelled(request_id: str) -> bool:
    with CANCEL_LOCK:
        return request_id in CANCELLED


def load() -> None:
    global runtime
    try:
        import torch
        from ardy.model import load_model

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.set_num_threads(4)
        started = time.perf_counter()
        runtime = InteractionRuntime(load_model(MODEL_NAME, device="cuda", text_encoder_mode="local"))
        STATE.update(ready=True, fps=20, joints=27, features=330, horizon=40,
                     load_seconds=time.perf_counter() - started,
                     gpu=torch.cuda.get_device_name())
        print(json.dumps({"event": "ready", **STATE}), flush=True)
    except Exception as exc:
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"event": "load_failed", "error": STATE["error"]}), flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, status: int, value, content_type: str = "application/json") -> None:
        data = value if isinstance(value, bytes) else json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def authorized(self) -> bool:
        if not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + TOKEN):
            self.reply(401, {"error": "Unauthorized"})
            return False
        return True

    def read_json(self) -> dict:
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if not 0 < size <= 2_000_000:
            raise ValueError("Request body must be 1–2,000,000 bytes")
        value = json.loads(self.rfile.read(size))
        if not isinstance(value, dict):
            raise ValueError("Request body must be an object")
        return value

    def do_GET(self):
        if not self.authorized():
            return
        if self.path == "/health":
            self.reply(200 if STATE["ready"] else 503, STATE)
        else:
            self.reply(404, {"error": "Not found"})

    def do_POST(self):
        if not self.authorized():
            return
        try:
            body = self.read_json()
            if self.path == "/cancel":
                if set(body) != {"request_id"} or not isinstance(body["request_id"], str) or not 1 <= len(body["request_id"]) <= 100:
                    raise ValueError("Cancel requires a request_id of 1–100 characters")
                with CANCEL_LOCK:
                    active = body["request_id"] in ACTIVE
                    if active:
                        CANCELLED.add(body["request_id"])
                self.reply(200, {"cancelled": active})
                return
            if self.path != "/generate":
                self.reply(404, {"error": "Not found"})
                return
            if not STATE["ready"] or runtime is None:
                self.reply(503, {"error": STATE["error"] or "Core model is loading"})
                return
            # Reject malformed requests before queuing on the GPU.
            validate_request(body)
            request_id = body["request_id"]
            with CANCEL_LOCK:
                if request_id in ACTIVE:
                    raise ValueError("request_id is already active")
                if len(ACTIVE) >= 32:
                    self.reply(503, {"error": "Interaction lab queue is full"})
                    return
                ACTIVE.add(request_id)
            try:
                if not MODEL_LOCK.acquire(timeout=20):
                    self.reply(503, {"error": "Core model is busy"})
                    return
                try:
                    arrays, metadata = runtime.generate(body, is_cancelled=cancelled,
                                                        deadline=time.monotonic() + 90)
                finally:
                    MODEL_LOCK.release()
            finally:
                with CANCEL_LOCK:
                    ACTIVE.discard(request_id)
                    CANCELLED.discard(request_id)
            self.reply(200, encode_npz(arrays, metadata), "application/octet-stream")
        except GenerationCancelled as exc:
            self.reply(409, {"error": str(exc)})
        except TimeoutError as exc:
            self.reply(504, {"error": str(exc)})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self.reply(400, {"error": str(exc)})
        except Exception as exc:
            print(json.dumps({"event": "request_failed", "error": f"{type(exc).__name__}: {exc}"}), flush=True)
            self.reply(503, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    if not 1 <= PORT <= 65535:
        raise ValueError("Invalid interaction service port")
    threading.Thread(target=load, daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(json.dumps({"event": "listening", "host": HOST, "port": PORT}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
