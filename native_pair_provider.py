"""Bounded remote InterGen sampling of exact native 30 fps paired motion.

The published checkpoint and probe must already be installed on the selected
host.  Each request creates one temporary remote process and one unique file;
it never contacts the older 20 fps retargeting service.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import subprocess
import threading
import time
import uuid

from native_pair_clip import MAX_BYTES, NativePairClip, load_source


_REMOTE_SAMPLE = r'''
import json, os, signal, subprocess, sys, threading
from pathlib import Path
line = bytearray()
while not line.endswith(b'\n') and len(line) <= 4096:
    part = os.read(0, 1)
    if not part:
        raise SystemExit('Missing native request')
    line.extend(part)
p = json.loads(line)
output = Path(p['output'])
if output.exists():
    raise SystemExit('Unique native output already exists')
if not Path(p['checkpoint']).is_file() or not Path(p['probe']).is_file():
    raise SystemExit('Installed InterGen assets are missing')
env = os.environ.copy()
env['PYTHONPATH'] = p['deps'] + ':' + p['clip_cache']
argv = [p['python'], p['probe'], 'sample', '--repo', p['repo'],
        '--checkpoint', p['checkpoint'], '--clip-cache', p['clip_cache'],
        '--text-only-clip', '--prompt', p['prompt'], '--seed', str(p['seed']),
        '--frames', str(p['frames']), '--cuda-memory-fraction', '0.08',
        '--output', str(output)]
child = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, start_new_session=True)
def stop_child():
    # Kill only this request's fresh process group, including stubborn workers.
    # Cancellation must have the same escalation bound as the request timeout.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    # The leader can exit while its data-loader children remain alive.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=3)
def interrupted(signum, frame):
    # Unwind communicate's waitpid lock before cleanup; waiting from a signal
    # handler can deadlock when the signal interrupted Popen.wait itself.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    raise InterruptedError('Native InterGen generation cancelled')
cancellation_started = threading.Event()
cancellation_done = threading.Event()
def cancelled():
    if os.read(0, 1) == b'':
        cancellation_started.set()
        try:
            stop_child()
        finally:
            cancellation_done.set()
try:
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    threading.Thread(target=cancelled, daemon=True).start()
    log, _ = child.communicate(timeout=p['timeout'])
except InterruptedError:
    stop_child()
    output.unlink(missing_ok=True)
    raise SystemExit('Native InterGen generation cancelled')
except subprocess.TimeoutExpired:
    stop_child()
    child.communicate(timeout=3)
    output.unlink(missing_ok=True)
    raise SystemExit('Native InterGen generation timed out')
if cancellation_started.is_set():
    cancellation_done.wait(timeout=7)
    output.unlink(missing_ok=True)
    raise SystemExit('Native InterGen generation cancelled')
if child.returncode:
    output.unlink(missing_ok=True)
    raise SystemExit('Native InterGen generation failed: ' + log[-4000:].decode('utf-8', 'replace'))
size = output.stat().st_size
if not 0 < size <= p['max_bytes']:
    output.unlink(missing_ok=True)
    raise SystemExit('Native output exceeds size bound')
print(json.dumps({'bytes': size, 'output': str(output)}), flush=True)
'''


@dataclass(frozen=True)
class NativePairConfig:
    ssh_host: str
    ssh_port: int
    known_hosts: str
    identity_file: str | None = None
    remote_python: str = '/workspace/stagezero/.venv/bin/python'
    remote_probe: str = '/workspace/stagezero/intergen-lab/intergen_probe.py'
    remote_repo: str = '/workspace/stagezero/intergen-lab/InterGen'
    remote_checkpoint: str = '/workspace/stagezero/intergen-lab/intergen.ckpt'
    remote_clip_cache: str = '/workspace/stagezero/intergen-lab/CLIP'
    remote_deps: str = '/workspace/stagezero/intergen-lab/deps'
    remote_output_dir: str = '/workspace/stagezero/intergen-lab/results-native-recovery-20260926'
    min_free_gpu_mib: int = 6144
    min_free_disk_mib: int = 64
    generation_timeout: int = 90
    transfer_timeout: int = 30

    def __post_init__(self):
        if not re.fullmatch(r'[a-zA-Z0-9_.@:-]{1,255}', self.ssh_host):
            raise ValueError('Invalid SSH host')
        if type(self.ssh_port) is not int or not 1 <= self.ssh_port <= 65535:
            raise ValueError('Invalid SSH port')
        for name in ('remote_python', 'remote_probe', 'remote_repo', 'remote_checkpoint',
                     'remote_clip_cache', 'remote_deps', 'remote_output_dir'):
            value = getattr(self, name)
            if (not isinstance(value, str) or not value.startswith('/') or
                    not re.fullmatch(r'[a-zA-Z0-9_./-]+', value) or '..' in Path(value).parts):
                raise ValueError(f'Invalid {name}')
        for name in ('known_hosts', 'identity_file'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not Path(value).is_absolute()):
                raise ValueError(f'{name} must be an absolute path')
        for name in ('min_free_gpu_mib', 'min_free_disk_mib', 'generation_timeout', 'transfer_timeout'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'Invalid {name}')

    @classmethod
    def from_env(cls):
        env = os.environ
        host = env.get('STAGEZERO_NATIVE_PAIR_SSH_HOST')
        known_hosts = env.get('STAGEZERO_NATIVE_PAIR_KNOWN_HOSTS')
        if not host or not known_hosts:
            raise ValueError('Native pair SSH host and known_hosts must be configured')
        return cls(
            ssh_host=host,
            ssh_port=int(env.get('STAGEZERO_NATIVE_PAIR_SSH_PORT', '22')),
            known_hosts=known_hosts,
            identity_file=env.get('STAGEZERO_NATIVE_PAIR_SSH_KEY') or None,
            remote_python=env.get('STAGEZERO_NATIVE_PAIR_REMOTE_PYTHON', cls.remote_python),
            remote_probe=env.get('STAGEZERO_NATIVE_PAIR_REMOTE_PROBE', cls.remote_probe),
            remote_repo=env.get('STAGEZERO_NATIVE_PAIR_REMOTE_REPO', cls.remote_repo),
            remote_checkpoint=env.get('STAGEZERO_NATIVE_PAIR_REMOTE_CHECKPOINT', cls.remote_checkpoint),
            remote_clip_cache=env.get('STAGEZERO_NATIVE_PAIR_REMOTE_CLIP_CACHE', cls.remote_clip_cache),
            remote_deps=env.get('STAGEZERO_NATIVE_PAIR_REMOTE_DEPS', cls.remote_deps),
            remote_output_dir=env.get('STAGEZERO_NATIVE_PAIR_REMOTE_OUTPUT_DIR', cls.remote_output_dir),
        )


class NativePairProvider:
    """Thread-safe serial sampler; `cancelled` is polled throughout one request."""

    def __init__(self, config: NativePairConfig | None = None):
        self.config = config if config is not None else NativePairConfig.from_env()
        self._lock = threading.Lock()
        self.last_raw_archive: bytes | None = None

    @classmethod
    def from_config(cls, path):
        source = Path(path)
        if source.stat().st_size > 16_384:
            raise ValueError('Native pair config exceeds 16 KiB')
        values = json.loads(source.read_text())
        if not isinstance(values, dict) or set(values) - set(NativePairConfig.__dataclass_fields__):
            raise ValueError('Invalid native pair configuration')
        return cls(NativePairConfig(**values))

    def _ssh(self, remote_command: str):
        c = self.config
        command = ['ssh', '-p', str(c.ssh_port), '-o', 'BatchMode=yes',
                   '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10',
                   '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2',
                   '-o', 'UserKnownHostsFile=' + c.known_hosts]
        if c.identity_file:
            command += ['-i', c.identity_file]
        return command + [c.ssh_host, remote_command]

    @staticmethod
    def _check_cancel(cancelled):
        if cancelled is not None and cancelled():
            raise RuntimeError('Native paired generation cancelled')

    def _run(self, command, *, input_bytes=None, timeout=30, max_bytes=8192, cancelled=None,
             keep_stdin=False):
        self._check_cancel(cancelled)
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=True)
        output = bytearray()
        errors = bytearray()
        selector = selectors.DefaultSelector()
        try:
            if input_bytes is not None:
                process.stdin.write(input_bytes)
                process.stdin.flush()
            if not keep_stdin:
                process.stdin.close()
            selector.register(process.stdout, selectors.EVENT_READ, output)
            selector.register(process.stderr, selectors.EVENT_READ, errors)
            deadline = time.monotonic() + timeout
            while selector.get_map() or process.poll() is None:
                self._check_cancel(cancelled)
                if time.monotonic() >= deadline:
                    raise TimeoutError('Native paired SSH operation timed out')
                for key, _ in selector.select(timeout=0.1):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    else:
                        key.data.extend(chunk)
                        if len(key.data) > max_bytes:
                            raise ValueError('Native paired SSH response exceeds size bound')
            if process.returncode:
                raise RuntimeError('Native paired SSH operation failed: ' +
                                   errors[-2000:].decode('utf-8', 'replace'))
            return bytes(output)
        except BaseException:
            if process.poll() is None:
                # Closing stdin signals the remote wrapper to terminate exactly
                # its own sample process group, with no global process kills.
                if not process.stdin.closed:
                    process.stdin.close()
                if keep_stdin and input_bytes is not None:
                    # Leave SSH connected until its wrapper acknowledges bounded
                    # remote cleanup; do not release the provider lock first.
                    try:
                        process.wait(timeout=7)
                    except subprocess.TimeoutExpired:
                        pass
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
            raise
        finally:
            selector.close()
            if not process.stdin.closed:
                process.stdin.close()
            process.stdout.close()
            process.stderr.close()

    def _health(self, cancelled):
        c = self.config
        probe = ('nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits')
        gpu = self._run(self._ssh(probe), timeout=20, cancelled=cancelled).decode().strip()
        try:
            free_gpu = min(int(line.strip()) for line in gpu.splitlines() if line.strip())
        except ValueError as exc:
            raise RuntimeError('Cannot read native generation GPU memory') from exc
        if free_gpu < c.min_free_gpu_mib:
            raise RuntimeError('Insufficient free GPU memory for native generation')
        disk_cmd = 'df -Pm ' + shlex.quote(c.remote_output_dir)
        disk = self._run(self._ssh(disk_cmd), timeout=20, cancelled=cancelled).decode().splitlines()
        try:
            free_disk = int(disk[-1].split()[3])
        except (IndexError, ValueError) as exc:
            raise RuntimeError('Cannot read native generation disk space') from exc
        if free_disk < c.min_free_disk_mib:
            raise RuntimeError('Insufficient free disk space for native generation')
        return {'free_gpu_mib': free_gpu, 'free_disk_mib': free_disk}

    def generate(self, prompt: str, seed: int, frames: int = 210, cancelled=None) -> NativePairClip:
        if (not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 500
                or any(ord(ch) < 32 for ch in prompt)):
            raise ValueError('Native pair prompt must contain 1–500 printable characters')
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError('Native pair seed must be an unsigned 32-bit integer')
        if type(frames) is not int or not 30 <= frames <= 210:
            raise ValueError('Native pair frames must be between 30 and 210')
        if cancelled is not None and not callable(cancelled):
            raise ValueError('cancelled must be callable')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('A native pair generation is already running')
        try:
            self._check_cancel(cancelled)
            health = self._health(cancelled)
            c = self.config
            output = c.remote_output_dir.rstrip('/') + '/native-pair-' + uuid.uuid4().hex + '.npz'
            payload = {'python': c.remote_python, 'probe': c.remote_probe,
                       'repo': c.remote_repo, 'checkpoint': c.remote_checkpoint,
                       'clip_cache': c.remote_clip_cache, 'deps': c.remote_deps,
                       'output': output, 'prompt': prompt.strip(), 'seed': seed,
                       'frames': frames, 'timeout': c.generation_timeout,
                       'max_bytes': MAX_BYTES}
            command = shlex.quote(c.remote_python) + ' -c ' + shlex.quote(_REMOTE_SAMPLE)
            try:
                result = self._run(self._ssh(command), input_bytes=(json.dumps(payload) + '\n').encode(),
                                   timeout=c.generation_timeout + 20, max_bytes=8192,
                                   cancelled=cancelled, keep_stdin=True)
                record = json.loads(result)
                if record.get('output') != output or not 0 < record.get('bytes', 0) <= MAX_BYTES:
                    raise ValueError('Unexpected native sample output')
                content = self._run(self._ssh('cat -- ' + shlex.quote(output)),
                                    timeout=c.transfer_timeout, max_bytes=MAX_BYTES,
                                    cancelled=cancelled)
                if len(content) != record['bytes']:
                    raise ValueError('Incomplete native pair transfer')
                clip = load_source(content)
                if (clip.frames != frames or clip.features is None
                        or clip.metadata.get('model') != 'InterGen'
                        or clip.metadata.get('prompt') != prompt.strip()
                        or clip.metadata.get('seed') != seed):
                    raise ValueError('Native InterGen returned incomplete arrays')
                metadata = dict(clip.metadata, provider='intergen_ssh_native',
                                free_gpu_before_mib=health['free_gpu_mib'],
                                free_disk_before_mib=health['free_disk_mib'])
                result = NativePairClip(clip.joints, clip.features, metadata)
                self.last_raw_archive = content
                return result
            finally:
                # This path is constructed here from a UUID; only our own file is removed.
                try:
                    self._run(self._ssh('rm -f -- ' + shlex.quote(output)), timeout=15)
                except Exception:
                    pass
        finally:
            self._lock.release()
