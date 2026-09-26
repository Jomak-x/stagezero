"""Loopback HTTP queue for a single self-hosted character GPU worker.

The worker contract is ``python script --input image.png --output model.glb``.
The server owns its job directories and never exposes worker output or traceback text.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import threading
import time
import uuid
import warnings
import zlib

from PIL import Image


MAX_PNG_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_GLB_BYTES = 40 * 1024 * 1024
JOB_TIMEOUT = 1800
RESULT_TTL = 3600
MAX_RETAINED = 8
MAX_STAGE_LOG_BYTES = 16 * 1024
_STAGES = frozenset({"checking_gpu", "waiting_gpu_memory", "gpu_available", "loading_pipeline", "generating", "generating_phase",
                     "removing_reference_background", "baking_texture", "enhancing_front_texture",
                     "exporting", "complete"})


def _valid_png(data: bytes) -> bool:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return False
    try:
        offset = 8
        saw_idat = False
        saw_iend = False
        chunk_index = 0
        while offset + 12 <= len(data):
            length = struct.unpack_from(">I", data, offset)[0]
            end = offset + 12 + length
            if end > len(data):
                return False
            kind = data[offset + 4:offset + 8]
            if not all(65 <= char <= 90 or 97 <= char <= 122 for char in kind):
                return False
            if chunk_index == 0 and (kind != b"IHDR" or length != 13):
                return False
            if zlib.crc32(data[offset + 4:end - 4]) != struct.unpack_from(">I", data, end - 4)[0]:
                return False
            if kind == b"IDAT":
                saw_idat = True
            if kind == b"IEND":
                saw_iend = length == 0 and saw_idat and end == len(data)
                break
            offset = end
            chunk_index += 1
        if not saw_iend:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format != "PNG" or image.width * image.height > MAX_PIXELS:
                    return False
                image.load()
        return True
    except (OSError, ValueError, Image.DecompressionBombWarning, Image.DecompressionBombError):
        return False


def _valid_glb(path: Path) -> bool:
    """Check bounded GLB framing and the required glTF 2.0 asset declaration."""
    try:
        size = path.stat().st_size
        if not 20 <= size <= MAX_GLB_BYTES or size % 4:
            return False
        with path.open("rb") as stream:
            header = stream.read(20)
            magic, version, declared, json_size, kind = struct.unpack("<4sIII4s", header)
            if magic != b"glTF" or version != 2 or declared != size or kind != b"JSON" or json_size % 4:
                return False
            if json_size > size - 20:
                return False
            document = json.loads(stream.read(json_size).rstrip(b" ").decode("utf-8"))
            if not isinstance(document, dict) or document.get("asset", {}).get("version") != "2.0":
                return False
            position = 20 + json_size
            if position == size:
                return True
            if position + 8 > size:
                return False
            bin_size, bin_kind = struct.unpack("<I4s", stream.read(8))
            return bin_kind == b"BIN\0" and bin_size % 4 == 0 and position + 8 + bin_size == size
    except (OSError, ValueError, UnicodeError, TypeError, AttributeError, struct.error):
        return False


def _latest_stage(path: Path) -> str | None:
    """Read only a short log tail and publish only worker-defined stage labels."""
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            start = max(0, size - MAX_STAGE_LOG_BYTES)
            stream.seek(start)
            tail = stream.read(MAX_STAGE_LOG_BYTES)
        lines = tail.splitlines()
        if start and lines:
            lines = lines[1:]  # The first line may start in the middle of a record.
        for line in reversed(lines):
            if len(line) > 256 or not line.startswith(b'{"stage":'):
                continue
            value = json.loads(line)
            stage = value.get("stage") if isinstance(value, dict) else None
            if isinstance(stage, str) and stage in _STAGES:
                return stage
    except (OSError, ValueError, UnicodeError):
        pass
    return None


@dataclass
class Job:
    id: str
    directory: Path
    status: str = "queued"
    error: str | None = None
    process: subprocess.Popen | None = None
    created: float = field(default_factory=time.monotonic)
    finished: float | None = None


class JobManager:
    def __init__(self, *, jobs_dir: Path, worker_python: Path, worker_script: Path,
                 worker_cwd: Path | None = None, ready_file: Path | None = None,
                 timeout: float = JOB_TIMEOUT,
                 result_ttl: float = RESULT_TTL, max_retained: int = MAX_RETAINED):
        self.jobs_dir = Path(jobs_dir).resolve()
        # Do not dereference virtualenv bin/python: its symlink path selects the venv.
        self.worker_python = Path(os.path.abspath(os.fspath(worker_python)))
        self.worker_script = Path(worker_script).resolve()
        self.worker_cwd = Path(worker_cwd).resolve() if worker_cwd else self.worker_script.parent
        self.ready_file = Path(ready_file).resolve() if ready_file else None
        self.timeout = timeout
        self.result_ttl = result_ttl
        self.max_retained = max_retained
        self.lock = threading.RLock()
        self.jobs: dict[str, Job] = {}
        self.busy_id: str | None = None
        if not self.worker_python.is_file() or not os.access(self.worker_python, os.X_OK):
            raise ValueError("Worker Python is unavailable or not executable")
        if not self.worker_script.is_file():
            raise ValueError("Worker script is unavailable")
        if not self.worker_cwd.is_dir():
            raise ValueError("Worker directory is unavailable")
        if timeout <= 0 or result_ttl <= 0 or max_retained < 1:
            raise ValueError("Invalid job limits")
        self.jobs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self.jobs_dir.is_dir() or not os.access(self.jobs_dir, os.W_OK | os.X_OK):
            raise ValueError("Jobs directory is unavailable")

    def _ready(self) -> bool:
        return (self.worker_python.is_file() and os.access(self.worker_python, os.X_OK)
                and self.worker_script.is_file() and self.worker_cwd.is_dir()
                and (self.ready_file is None or self.ready_file.is_file()))

    def health(self) -> dict:
        ready = self._ready()
        with self.lock:
            result = {"ready": ready, "busy": self.busy_id is not None, "engine": "TRELLIS"}
        if not ready:
            result["error"] = "Worker unavailable"
        return result

    def _cleanup_locked(self) -> None:
        terminal = sorted((job for job in self.jobs.values() if job.finished is not None),
                          key=lambda job: job.finished or 0)
        expired = {job.id for job in terminal if time.monotonic() - (job.finished or 0) > self.result_ttl}
        expired.update(job.id for job in terminal[:max(0, len(terminal) - self.max_retained)])
        for job_id in expired:
            job = self.jobs.pop(job_id)
            shutil.rmtree(job.directory, ignore_errors=True)

    def submit(self, png: bytes) -> Job | None:
        with self.lock:
            self._cleanup_locked()
            if not self._ready():
                raise RuntimeError("Worker not ready")
            if self.busy_id is not None:
                return None
            job_id = uuid.uuid4().hex
            directory = self.jobs_dir / job_id
            directory.mkdir(mode=0o700)
            try:
                (directory / "input.png").write_bytes(png)
            except OSError:
                shutil.rmtree(directory, ignore_errors=True)
                raise
            job = Job(job_id, directory)
            self.jobs[job_id] = job
            self.busy_id = job_id
            threading.Thread(target=self._run, args=(job,), daemon=True, name=f"gpu-job-{job_id[:8]}").start()
            return job

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            self._cleanup_locked()
            return self.jobs.get(job_id)

    def snapshot(self, job_id: str) -> dict | None:
        with self.lock:
            self._cleanup_locked()
            job = self.jobs.get(job_id)
            if job is None:
                return None
            result = {"id": job.id, "status": job.status}
            if job.status == "failed":
                result["error"] = job.error or "Generation failed"
            if job.status == "generating":
                stage = _latest_stage(job.directory / "worker.log")
                if stage:
                    result["stage"] = stage
            return result

    def result(self, job_id: str) -> Path | None:
        with self.lock:
            self._cleanup_locked()
            job = self.jobs.get(job_id)
            return job.directory / "result.glb" if job and job.status == "succeeded" else None

    @staticmethod
    def _stop(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    def cancel(self, job_id: str) -> bool:
        with self.lock:
            self._cleanup_locked()
            job = self.jobs.get(job_id)
            if job is None:
                return False
            if job.status in ("queued", "generating"):
                job.status = "cancelled"
                process = job.process
            else:
                process = None
        if process is not None:
            self._stop(process)
        return True

    def _run(self, job: Job) -> None:
        try:
            with self.lock:
                if job.status == "cancelled":
                    return
                job.status = "generating"
                argv = [str(self.worker_python), str(self.worker_script), "--input",
                        str(job.directory / "input.png"), "--output", str(job.directory / "result.glb")]
                with (job.directory / "worker.log").open("wb") as log:
                    process = subprocess.Popen(argv, cwd=self.worker_cwd, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                job.process = process
            try:
                code = process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                self._stop(process)
                code = None
            with self.lock:
                if job.status == "cancelled":
                    return
                if code is None:
                    job.status, job.error = "failed", "Generation timed out"
                elif code != 0:
                    job.status = "failed"
                    job.error = ("GPU memory busy" if _latest_stage(job.directory / "worker.log") == "waiting_gpu_memory"
                                 else "Worker failed")
                elif not _valid_glb(job.directory / "result.glb"):
                    job.status, job.error = "failed", "Worker returned an invalid GLB"
                else:
                    job.status = "succeeded"
        except (OSError, ValueError):
            with self.lock:
                if job.status != "cancelled":
                    job.status, job.error = "failed", "Worker could not start"
        finally:
            with self.lock:
                job.process = None
                if job.finished is None:
                    job.finished = time.monotonic()
                if self.busy_id == job.id:
                    self.busy_id = None
                self._cleanup_locked()


class CharacterHTTPServer(ThreadingHTTPServer):
    # Concurrent clients should reach the handler and receive a deliberate 409
    # while a worker is busy, rather than being reset by the socket backlog.
    request_queue_size = 64


def make_handler(manager: JobManager, token: str):
    import hmac

    class Handler(BaseHTTPRequestHandler):
        server_version = "CharacterGPU/1"

        def log_message(self, format, *args):
            pass  # Request paths and authentication must never appear in access logs.

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def _authorized(self) -> bool:
            provided = self.headers.get("Authorization", "")
            expected = "Bearer " + token
            if not hmac.compare_digest(provided.encode("utf-8"), expected.encode("ascii")):
                self._json(401, {"error": "Unauthorized"})
                return False
            return True

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value: dict) -> None:
            self._send(status, json.dumps(value, separators=(",", ":")).encode(), "application/json")

        def _job_id(self, result: bool = False) -> str | None:
            match = re.fullmatch(r"/jobs/([a-f0-9]{32})" + (r"/result" if result else ""), self.path)
            return match.group(1) if match else None

        def do_GET(self):
            if not self._authorized():
                return
            if self.path == "/health":
                self._json(200, manager.health())
                return
            job_id = self._job_id()
            if job_id:
                snapshot = manager.snapshot(job_id)
                self._json(200, snapshot) if snapshot else self._json(404, {"error": "Job not found"})
                return
            job_id = self._job_id(result=True)
            if job_id:
                snapshot = manager.snapshot(job_id)
                if snapshot is None:
                    self._json(404, {"error": "Job not found"})
                elif snapshot["status"] != "succeeded":
                    self._json(409, {"error": "Result unavailable"})
                else:
                    path = manager.result(job_id)
                    try:
                        data = path.read_bytes() if path else b""
                    except OSError:
                        data = b""
                    if not data or len(data) > MAX_GLB_BYTES:
                        self._json(500, {"error": "Result unavailable"})
                    else:
                        self._send(200, data, "model/gltf-binary")
                return
            self._json(404, {"error": "Not found"})

        def do_POST(self):
            if not self._authorized():
                return
            if self.path != "/jobs":
                self._json(404, {"error": "Not found"})
                return
            if self.headers.get("Transfer-Encoding"):
                self._json(400, {"error": "Content-Length required"})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._json(400, {"error": "Content-Length required"})
                return
            if length < 1 or length > MAX_PNG_BYTES:
                self._json(413 if length > MAX_PNG_BYTES else 400, {"error": "Invalid image size"})
                return
            data = self.rfile.read(length)
            if len(data) != length or not _valid_png(data):
                self._json(400, {"error": "Invalid PNG"})
                return
            try:
                job = manager.submit(data)
            except RuntimeError:
                self._json(503, {"error": "Worker unavailable"})
                return
            except OSError:
                self._json(500, {"error": "Could not create job"})
                return
            if job is None:
                self._json(409, {"error": "Worker busy"})
            else:
                self._json(202, {"id": job.id, "status": "queued"})

        def do_DELETE(self):
            if not self._authorized():
                return
            job_id = self._job_id()
            if not job_id:
                self._json(404, {"error": "Not found"})
            elif manager.cancel(job_id):
                self._json(200, manager.snapshot(job_id) or {"id": job_id, "status": "cancelled"})
            else:
                self._json(404, {"error": "Job not found"})

    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--jobs-dir", required=True, type=Path)
    parser.add_argument("--worker-python", required=True, type=Path)
    parser.add_argument("--worker-script", required=True, type=Path)
    parser.add_argument("--worker-cwd", type=Path)
    parser.add_argument("--ready-file", type=Path, help="File created after the worker smoke check succeeds")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    try:
        if not args.token_file.is_file() or args.token_file.stat().st_size > 4096:
            raise ValueError("Token file is unavailable or too large")
        token = args.token_file.read_text(encoding="ascii").strip()
        if not token or any(ord(char) < 33 or ord(char) > 126 for char in token):
            raise ValueError("Token file contains an invalid token")
        manager = JobManager(jobs_dir=args.jobs_dir, worker_python=args.worker_python,
                             worker_script=args.worker_script, worker_cwd=args.worker_cwd,
                             ready_file=args.ready_file)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    server = CharacterHTTPServer(("127.0.0.1", args.port), make_handler(manager, token))
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
