import json
import os
import signal
import subprocess
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

from native_pair_provider import NativePairConfig, NativePairProvider, _REMOTE_SAMPLE


ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / 'review/two-character/native-recovery/raw/assist42.npz'
PROMPT = 'One person helps another person stand up.'


def config():
    return NativePairConfig('root@example.org', 26407, '/tmp/known_hosts')


class FakeProvider(NativePairProvider):
    def __init__(self):
        super().__init__(config())
        self.calls = []
        self.output = None
        self.fail_transfer = False

    def _run(self, command, *, input_bytes=None, **kwargs):
        self.calls.append((command, input_bytes, kwargs))
        remote = command[-1]
        if remote.startswith('nvidia-smi'):
            return b'14064\n'
        if remote.startswith('df -Pm'):
            return b'Filesystem 1048576-blocks Used Available Capacity Mounted on\noverlay 30720 30001 719 98% /\n'
        if input_bytes is not None:
            payload = json.loads(input_bytes)
            self.output = payload['output']
            assert payload['prompt'] == PROMPT
            return json.dumps({'output': self.output, 'bytes': FIXTURE.stat().st_size}).encode()
        if remote.startswith('cat --'):
            return b'bad' if self.fail_transfer else FIXTURE.read_bytes()
        if remote.startswith('rm -f --'):
            return b''
        raise AssertionError(remote)


class NativePairProviderTests(unittest.TestCase):
    def test_exact_native_sample_and_cleanup(self):
        provider = FakeProvider()
        clip = provider.generate(PROMPT, 42, 210)
        self.assertEqual((clip.joints.shape, clip.features.shape),
                         ((210, 2, 22, 3), (210, 2, 262)))
        self.assertEqual(clip.fps, 30)
        self.assertEqual(clip.metadata['provider'], 'intergen_ssh_native')
        self.assertEqual(provider.last_raw_archive, FIXTURE.read_bytes())
        self.assertEqual(len([c for c in provider.calls if c[0][-1].startswith('rm -f --')]), 1)
        self.assertNotIn(PROMPT, ' '.join(str(c[0]) for c in provider.calls))

    def test_invalid_requests_never_contact_ssh(self):
        provider = FakeProvider()
        for prompt, seed, frames in [('', 42, 210), ('bad\ncommand', 42, 210),
                                     (PROMPT, True, 210), (PROMPT, -1, 210),
                                     (PROMPT, 42, 29), (PROMPT, 42, 211)]:
            with self.assertRaises(ValueError):
                provider.generate(prompt, seed, frames)
        self.assertEqual(provider.calls, [])

    def test_failure_preserves_last_good_and_cleans_own_file(self):
        provider = FakeProvider()
        provider.generate(PROMPT, 42, 210)
        previous = provider.last_raw_archive
        provider.fail_transfer = True
        with self.assertRaises(ValueError):
            provider.generate(PROMPT, 42, 210)
        self.assertIs(provider.last_raw_archive, previous)
        self.assertEqual(len([c for c in provider.calls if c[0][-1].startswith('rm -f --')]), 2)

    def test_serial_and_cancelled_request(self):
        provider = FakeProvider()
        provider._lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, 'already running'):
                provider.generate(PROMPT, 42)
        finally:
            provider._lock.release()
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            provider.generate(PROMPT, 42, cancelled=lambda: True)
        self.assertEqual(provider.calls, [])

    def test_cancel_stops_only_its_own_local_subprocess(self):
        provider = NativePairProvider(config())
        cancellation = threading.Event()
        timer = threading.Timer(0.2, cancellation.set)
        timer.start()
        start = time.monotonic()
        try:
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                provider._run([sys.executable, '-c', 'import time; time.sleep(10)'],
                              timeout=5, cancelled=cancellation.is_set, keep_stdin=True)
        finally:
            timer.cancel()
        self.assertLess(time.monotonic() - start, 2)

    def test_remote_wrapper_escalates_only_its_own_stubborn_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe = root / 'probe.py'
            pidfile = root / 'child.pid'
            output = root / 'sample.npz'
            checkpoint = root / 'checkpoint'
            checkpoint.touch()
            probe.write_text(
                'import os, signal, time\n'
                'from pathlib import Path\n'
                'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
                f'Path({str(pidfile)!r}).write_text(str(os.getpid()))\n'
                f'Path({str(output)!r}).write_bytes(b"partial")\n'
                'time.sleep(60)\n')
            payload = dict(output=str(output), checkpoint=str(checkpoint), probe=str(probe),
                           deps=str(root), clip_cache=str(root), python=sys.executable,
                           repo=str(root), prompt='Test', seed=1, frames=30,
                           timeout=30, max_bytes=32000000)
            unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
            try:
                for cancellation in ('eof', 'signal'):
                    with self.subTest(cancellation=cancellation):
                        pidfile.unlink(missing_ok=True)
                        wrapper = subprocess.Popen([sys.executable, '-c', _REMOTE_SAMPLE],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        child_pid = None
                        try:
                            wrapper.stdin.write((json.dumps(payload) + '\n').encode())
                            wrapper.stdin.flush()
                            deadline = time.monotonic() + 5
                            while not pidfile.exists() and time.monotonic() < deadline:
                                time.sleep(.01)
                            self.assertTrue(pidfile.exists(), 'Sample child never started')
                            child_pid = int(pidfile.read_text())
                            started = time.monotonic()
                            if cancellation == 'eof':
                                wrapper.stdin.close()
                                wrapper.stdin = None
                            else:
                                wrapper.send_signal(signal.SIGTERM)
                            wrapper.communicate(timeout=8)
                            self.assertLess(time.monotonic() - started, 7)
                            self.assertNotEqual(wrapper.returncode, 0)
                            with self.assertRaises(ProcessLookupError):
                                os.kill(child_pid, 0)
                            self.assertFalse(output.exists())
                            self.assertIsNone(unrelated.poll())
                        finally:
                            if wrapper.poll() is None:
                                wrapper.kill(); wrapper.wait(timeout=3)
                            if child_pid is not None:
                                try:
                                    os.kill(child_pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                            if wrapper.stdin is not None:
                                wrapper.stdin.close()
                            wrapper.stdout.close(); wrapper.stderr.close()
            finally:
                unrelated.terminate(); unrelated.wait(timeout=3)

    def test_config_rejects_shell_paths_and_unknown_keys(self):
        with self.assertRaises(ValueError):
            NativePairConfig('root@host;evil', 22, '/tmp/known_hosts')
        with self.assertRaises(ValueError):
            NativePairConfig('root@host', 22, '/tmp/known_hosts', remote_probe='/tmp/x;evil')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'ssh_host': 'root@example.org', 'ssh_port': 22,
                                        'known_hosts': '/tmp/known_hosts'}))
            self.assertIsInstance(NativePairProvider.from_config(path), NativePairProvider)
            path.write_text(json.dumps({'ssh_host': 'root@example.org', 'ssh_port': 22,
                                        'known_hosts': '/tmp/known_hosts', 'unknown': 'x'}))
            with self.assertRaises(ValueError):
                NativePairProvider.from_config(path)


if __name__ == '__main__':
    unittest.main()
