import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

from native_pair_provider import NativePairConfig, NativePairProvider


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
