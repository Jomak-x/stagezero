"""Private warm native InterGen worker; no downloads, retargeting or shared-worker changes.

One owned sampler subprocess keeps the official model resident. HTTP requests are
serialized; cancellation/timeout kills only that subprocess and discards its
result. A subsequent request reloads it. Bind only to loopback and use an SSH tunnel.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass, field
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import math
import multiprocessing
from pathlib import Path
import queue
import re
import signal
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

MAX_BYTES = 32_000_000
MAX_BODY = 4096


def validate_job(body):
    if not isinstance(body, dict) or set(body) != {'request_id', 'prompt', 'seed', 'frames'}:
        raise ValueError('Expected request_id, prompt, seed and frames')
    if not isinstance(body['request_id'], str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}', body['request_id']):
        raise ValueError('Invalid request_id')
    prompt = body['prompt']
    if (not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 500
            or any(ord(c) < 32 for c in prompt)):
        raise ValueError('Prompt must contain 1–500 printable characters')
    if type(body['seed']) is not int or not 0 <= body['seed'] < 2**32:
        raise ValueError('Seed must be an unsigned 32-bit integer')
    if type(body['frames']) is not int or not 30 <= body['frames'] <= 210:
        raise ValueError('Frames must be between 30 and 210')
    return dict(body, prompt=prompt.strip())


class InterGenSampler:
    """Load once; retain exact official text_process/decode_motion/sample outputs."""
    def __init__(self, config):
        started = time.perf_counter()
        args = SimpleNamespace(**config)
        args.repo, args.checkpoint, args.clip_cache = map(Path, (args.repo, args.checkpoint, args.clip_cache))
        args.text_only_clip, args.seed = True, 0
        spec = importlib.util.spec_from_file_location('installed_native_intergen_probe', args.probe)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Native worker requires CUDA')
        free, _ = torch.cuda.mem_get_info()
        if free < args.min_free_gpu_mib * 1024**2:
            raise RuntimeError('Insufficient free GPU memory for native worker')
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device=0)
        torch.set_num_threads(4)
        self.report = probe.inspect(args)
        self.model, self.state_key_count = probe.prepare_model(args, self.report)
        self.model = self.model.to('cuda:0').eval()
        from utils.utils import MotionNormalizer
        self.normalizer = MotionNormalizer()
        torch.cuda.synchronize()
        self.ready_seconds = time.perf_counter() - started
        self.fraction = args.cuda_memory_fraction
        self.min_free_gpu_mib = args.min_free_gpu_mib

    def sample(self, job):
        import numpy as np
        import torch
        from scipy.ndimage import gaussian_filter1d
        # forward_test is exactly text_process followed by decode_motion. Calling
        # those two methods separately permits synchronized timing without hooks.
        if torch.cuda.mem_get_info()[0] < self.min_free_gpu_mib * 1024**2:
            raise RuntimeError('Insufficient free GPU memory for native sampling')
        torch.manual_seed(job['seed'])
        torch.cuda.manual_seed_all(job['seed'])
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            batch = {'motion_lens': torch.tensor([job['frames']], dtype=torch.long, device='cuda:0'),
                     'text': [job['prompt']]}
            batch = self.model.text_process(batch)
            torch.cuda.synchronize()
            text_seconds = time.perf_counter() - started
            sample_started = time.perf_counter()
            batch.update(self.model.decode_motion(batch))
            output = batch['output']
            torch.cuda.synchronize()
            sample_seconds = time.perf_counter() - sample_started
        generation_seconds = time.perf_counter() - started
        if tuple(output.shape) != (1, job['frames'], 524):
            raise ValueError('InterGen returned incompatible native features')
        archive_started = time.perf_counter()
        features = self.normalizer.backward(output[0].reshape(job['frames'], 2, 262).cpu().numpy())
        joints = features[..., :66].reshape(job['frames'], 2, 22, 3)
        if not np.isfinite(features).all():
            raise ValueError('InterGen returned non-finite native features')
        metadata = {'model': 'InterGen', 'source_revision': self.report['source_revision'],
                    'checkpoint': self.report['checkpoint'], 'clip_file': None,
                    'text_only_clip_from_checkpoint': True, 'checkpoint_state_keys': self.state_key_count,
                    'prompt': job['prompt'], 'seed': job['seed'], 'frames': job['frames'], 'fps': 30,
                    'request_id': job['request_id'], 'feature_shape': list(features.shape),
                    'joint_shape': list(joints.shape), 'generation_seconds': generation_seconds,
                    'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
                    'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
                    'cuda_memory_fraction': self.fraction, 'missing_checkpoint_keys': 0,
                    'unexpected_checkpoint_keys': 0,
                    'license': 'CC BY-NC-SA 4.0; noncommercial research preview only',
                    'worker_timings': {'model_ready_seconds': self.ready_seconds,
                                       'text_encode_seconds': text_seconds,
                                       'sample_seconds': sample_seconds}}
        stream = io.BytesIO()
        np.savez_compressed(stream, features=features.astype(np.float32), joints=joints.astype(np.float32),
                            smoothed_joints=gaussian_filter1d(joints, sigma=1, axis=0, mode='nearest').astype(np.float32),
                            metadata=json.dumps(metadata))
        content = stream.getvalue()
        if not 0 < len(content) <= MAX_BYTES:
            raise ValueError('Native archive exceeds its size bound')
        timings = dict(metadata['worker_timings'], archive_seconds=time.perf_counter()-archive_started)
        return content, timings


def _sampler_child(connection, config):
    try:
        sampler = InterGenSampler(config)
        connection.send({'ready': True, 'model_ready_seconds': sampler.ready_seconds})
        while True:
            job = connection.recv()
            if job is None:
                return
            try:
                content, timings = sampler.sample(job)
                connection.send({'ok': True, 'timings': timings})
                connection.send_bytes(content)
            except Exception:
                connection.send({'ok': False, 'error': 'Native model sampling failed'})
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send({'ready': False, 'error': 'Native model initialization failed'})
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class ResidentSampler:
    """Own exactly one child; process isolation provides a hard cancellation bound."""
    def __init__(self, config, *, load_timeout=120, child_target=_sampler_child):
        self.config, self.load_timeout, self.child_target = config, load_timeout, child_target
        self.process = self.connection = None
        self.ready_seconds = None

    def stop(self):
        process, self.process = self.process, None
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=.75)
            if process.is_alive():
                process.kill()
                process.join(timeout=.75)
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.ready_seconds = None

    def _receive(self, deadline, cancelled, reader=None):
        # Pipe.poll alone does not bound recv/recv_bytes once a frame header has
        # arrived. A receiver thread lets this owner enforce cancellation even
        # when a sampler stalls halfway through sending an archive.
        connection = self.connection
        responses = queue.Queue(maxsize=1)
        def receive():
            try:
                responses.put((True, (reader or connection.recv)()))
            except BaseException as exc:
                responses.put((False, exc))
        threading.Thread(target=receive, daemon=True, name='native-result-reader').start()
        while True:
            if cancelled():
                raise InterruptedError('Native paired generation cancelled')
            if time.monotonic() >= deadline:
                raise TimeoutError('Native worker exceeded its time limit')
            try:
                ok, value = responses.get(timeout=.02)
                if ok:
                    return value
                raise RuntimeError('Native sampler response failed') from value
            except queue.Empty:
                if not self.process.is_alive():
                    raise RuntimeError('Native sampler stopped unexpectedly')

    def warm(self, cancelled=lambda: False, *, deadline=None):
        if self.process is not None and self.process.is_alive() and self.ready_seconds is not None:
            return
        self.stop()
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=self.child_target, args=(child, self.config), daemon=True)
        self.process.start()
        child.close()
        try:
            load_deadline = time.monotonic()+self.load_timeout
            ready = self._receive(load_deadline if deadline is None else min(deadline, load_deadline), cancelled)
            if ready.get('ready') is not True:
                raise RuntimeError('Native model initialization failed')
            self.ready_seconds = ready['model_ready_seconds']
        except BaseException:
            self.stop()
            raise

    def sample(self, job, *, deadline, cancelled):
        try:
            self.warm(cancelled, deadline=deadline)
            if cancelled():
                raise InterruptedError('Native paired generation cancelled')
            if time.monotonic() >= deadline:
                raise TimeoutError('Native worker exceeded its time limit')
            self.connection.send(job)
            result = self._receive(deadline, cancelled)
            if result.get('ok') is not True:
                raise RuntimeError('Native model sampling failed')
            # The child publishes header only after its full bounded archive is
            # ready; it then writes one frame. recv_bytes enforces payload size.
            connection = self.connection
            content = self._receive(deadline, cancelled, lambda: connection.recv_bytes(MAX_BYTES))
            if cancelled():
                raise InterruptedError('Native paired generation cancelled')
            return content, result['timings']
        except BaseException:
            self.stop()
            raise


@dataclass
class Job:
    body: dict
    status: str = 'queued'
    submitted_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    finished_at: float | None = None
    content: bytes | None = None
    timings: dict = field(default_factory=dict)
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)

    def public(self):
        return {'request_id': self.body['request_id'], 'status': self.status,
                'frames': self.body['frames'], 'seed': self.body['seed'],
                'bytes': len(self.content) if self.content else 0,
                'timings': self.timings, 'error': self.error}


class NativeJobManager:
    def __init__(self, sampler, *, max_queued=2, max_retained=8, job_timeout=90):
        if type(max_queued) is not int or not 1 <= max_queued <= 8 or max_retained < max_queued+1:
            raise ValueError('Invalid native queue bounds')
        if not math.isfinite(job_timeout) or job_timeout <= 0:
            raise ValueError('Invalid native timeout')
        self.sampler, self.job_timeout, self.max_retained = sampler, job_timeout, max_retained
        self.jobs, self.lock = OrderedDict(), threading.RLock()
        self.queue = queue.Queue(maxsize=max_queued)
        self.closed = False
        self.worker = threading.Thread(target=self._work, name='native-serial-sampler', daemon=True)
        self.worker.start()

    def submit(self, body):
        body = validate_job(body)
        with self.lock:
            if self.closed:
                raise RuntimeError('Native worker is stopping')
            if body['request_id'] in self.jobs:
                raise FileExistsError('Request ID already exists')
            if self.queue.full():
                raise OverflowError('Native queue is full')
            while len(self.jobs) >= self.max_retained:
                terminal = next((key for key, job in self.jobs.items() if job.done.is_set()), None)
                if terminal is None:
                    raise OverflowError('Native job storage is full')
                del self.jobs[terminal]
            job = Job(body)
            self.jobs[body['request_id']] = job
            self.queue.put_nowait(job)
            return job.public()

    def status(self, identifier):
        with self.lock:
            return self.jobs[identifier].public()

    def result(self, identifier):
        with self.lock:
            job = self.jobs[identifier]
            if job.status != 'complete' or job.content is None:
                raise BlockingIOError('Native result is not ready')
            return job.content

    def cancel(self, identifier):
        with self.lock:
            job = self.jobs[identifier]
            job.cancel_event.set()
            job.content = None
            if job.status != 'running':
                job.status = 'cancelled'
                job.finished_at = time.monotonic()
                job.done.set()
        job.done.wait(timeout=2)
        return self.status(identifier)

    def _work(self):
        while True:
            job = self.queue.get()
            if job is None:
                self.queue.task_done()
                return
            try:
                with self.lock:
                    if job.cancel_event.is_set():
                        continue
                    job.status, job.started_at = 'running', time.monotonic()
                content, timings = self.sampler.sample(job.body,
                    deadline=job.started_at+self.job_timeout, cancelled=job.cancel_event.is_set)
                with self.lock:
                    if job.cancel_event.is_set():
                        job.status = 'cancelled'
                    else:
                        job.content, job.status = content, 'complete'
                        job.timings = dict(timings, queue_seconds=job.started_at-job.submitted_at,
                                           worker_wall_seconds=time.monotonic()-job.started_at)
            except (InterruptedError, TimeoutError) as exc:
                with self.lock:
                    job.status = 'cancelled' if job.cancel_event.is_set() else 'failed'
                    job.error = str(exc)
            except Exception:
                with self.lock:
                    job.status, job.error = 'failed', 'Native model sampling failed'
            finally:
                with self.lock:
                    job.finished_at = time.monotonic()
                    job.done.set()
                self.queue.task_done()

    def close(self):
        with self.lock:
            self.closed = True
            for job in self.jobs.values():
                if not job.done.is_set():
                    job.cancel_event.set()
        self.queue.put(None, timeout=3)
        self.worker.join(timeout=3)
        self.sampler.stop()


def make_server(manager, token, *, host='127.0.0.1', port=8772):
    if host not in ('127.0.0.1', 'localhost'):
        raise ValueError('Native worker must bind to loopback')
    if not isinstance(token, str) or not 16 <= len(token) <= 4096 or any(ord(c) < 33 or ord(c) > 126 for c in token):
        raise ValueError('Native worker requires a private printable token of at least 16 characters')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def send(self, code, payload):
            binary = isinstance(payload, bytes)
            data = payload if binary else json.dumps(payload, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/octet-stream' if binary else 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def route(self):
            supplied = self.headers.get('Authorization', '')
            if not hmac.compare_digest(supplied.encode(), ('Bearer '+token).encode()):
                self.send(401, {'error': 'Unauthorized'})
                return
            path = urlsplit(self.path)
            parts = path.path.strip('/').split('/')
            try:
                if path.query or path.fragment:
                    raise ValueError('Unexpected query')
                if self.command == 'GET' and parts == ['health']:
                    self.send(200, {'ready': manager.sampler.ready_seconds is not None,
                                    'model': 'InterGen', 'native_fps': 30,
                                    'model_ready_seconds': manager.sampler.ready_seconds,
                                    'queue_depth': manager.queue.qsize()})
                elif self.command == 'POST' and parts == ['v1', 'native', 'jobs']:
                    if self.headers.get('Transfer-Encoding'):
                        raise ValueError('Chunked requests are unsupported')
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= MAX_BODY:
                        raise ValueError('Native request exceeds its size bound')
                    body = json.loads(self.rfile.read(length))
                    self.send(202, manager.submit(body))
                elif len(parts) in (4, 5) and parts[:3] == ['v1', 'native', 'jobs']:
                    if len(parts) == 5 and parts[4] == 'result' and self.command == 'GET':
                        self.send(200, manager.result(parts[3]))
                    elif len(parts) == 4 and self.command == 'GET':
                        self.send(200, manager.status(parts[3]))
                    elif len(parts) == 4 and self.command == 'DELETE':
                        self.send(200, manager.cancel(parts[3]))
                    else:
                        self.send(404, {'error': 'Unknown endpoint'})
                else:
                    self.send(404, {'error': 'Unknown endpoint'})
            except KeyError:
                self.send(404, {'error': 'Unknown request'})
            except (OverflowError, FileExistsError, BlockingIOError) as exc:
                self.send(409, {'error': str(exc)})
            except (ValueError, TypeError):
                self.send(400, {'error': 'Invalid native request'})
            except Exception:
                self.send(500, {'error': 'Native worker request failed'})

        do_GET = do_POST = do_DELETE = route

    return ThreadingHTTPServer((host, port), Handler)


def run_server(server, manager):
    """Warm and serve, reaping only our child on normal exit, SIGINT or SIGTERM."""
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt
    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        manager.sampler.warm()
        print(json.dumps({'event': 'native_worker_ready', 'port': server.server_port,
                          'model_ready_seconds': manager.sampler.ready_seconds}), flush=True)
        server.serve_forever(poll_interval=.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--clip-cache', required=True)
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8772)
    parser.add_argument('--cuda-memory-fraction', type=float, default=.08)
    parser.add_argument('--min-free-gpu-mib', type=int, default=6144)
    args = parser.parse_args()
    if not 0 < args.cuda_memory_fraction <= .08 or args.min_free_gpu_mib < 6144:
        parser.error('Native worker keeps the existing 0.08 memory ceiling and 6144 MiB free-memory gate')
    for name in ('probe', 'repo', 'checkpoint', 'clip_cache'):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    token = args.token_file.read_text().strip()
    config = {key: getattr(args, key) for key in ('probe', 'repo', 'checkpoint', 'clip_cache',
              'cuda_memory_fraction', 'min_free_gpu_mib')}
    sampler = ResidentSampler(config)
    manager = NativeJobManager(sampler)
    # Reserve only our explicitly selected port before allocating GPU memory.
    server = make_server(manager, token, port=args.port)
    run_server(server, manager)


if __name__ == '__main__':
    main()
