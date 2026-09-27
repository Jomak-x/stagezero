"""CPU-only contract tests for warm worker transport and process ownership."""
import io
import json
import os
import selectors
import subprocess
import sys
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from native_pair_provider import NativePairConfig, NativePairProvider
from native_pair_worker import NativeJobManager, ResidentSampler, make_server, validate_job

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / 'review/two-character/native-recovery/raw/assist42.npz'
PROMPT = 'One person helps another person stand up.'
TOKEN = 'private-test-token-123456'


def body(identifier='request-1'):
    return dict(request_id=identifier, prompt=PROMPT, seed=42, frames=210)


def fixture_archive(job):
    with np.load(FIXTURE, allow_pickle=False) as source:
        metadata = dict(json.loads(source['metadata'].item()), request_id=job['request_id'],
                        prompt=job['prompt'], seed=job['seed'], frames=job['frames'])
        arrays = {key: source[key] for key in source.files if key != 'metadata'}
    result = io.BytesIO()
    np.savez_compressed(result, **arrays, metadata=json.dumps(metadata))
    return result.getvalue()


class FakeSampler:
    ready_seconds = .25
    def __init__(self, gate=None):
        self.gate = gate
        self.active = self.peak_active = self.calls = 0
        self.started = threading.Event()
        self.stopped = False

    def sample(self, job, *, deadline, cancelled):
        self.calls += 1
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        self.started.set()
        try:
            while self.gate is not None and not self.gate.is_set():
                if cancelled():
                    raise InterruptedError('Native paired generation cancelled')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Native worker exceeded its time limit')
                time.sleep(.005)
            return fixture_archive(job), dict(model_ready_seconds=.25, text_encode_seconds=.01, sample_seconds=.1)
        finally:
            self.active -= 1

    def stop(self):
        self.stopped = True


def fake_resident_child(connection, config):
    if config.get('stall_load'):
        time.sleep(30)
    connection.send({'ready': True, 'model_ready_seconds': .123})
    count = 0
    try:
        while True:
            job = connection.recv()
            count += 1
            if job['prompt'] == 'stall':
                time.sleep(30)
            connection.send({'ok': True, 'timings': {'child_pid': os.getpid(), 'calls': count}})
            if job['prompt'] == 'stall-transfer':
                time.sleep(30)
            connection.send_bytes(b'archive')
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class NativeWorkerTests(unittest.TestCase):
    def setUp(self):
        self.managers = []

    def tearDown(self):
        for manager in self.managers:
            manager.close()

    def manager(self, sampler=None, **kwargs):
        manager = NativeJobManager(sampler or FakeSampler(), **kwargs)
        self.managers.append(manager)
        return manager

    def test_strict_job_validation_and_bounds(self):
        self.assertEqual(validate_job(body()), body())
        for changes in ({'frames': True}, {'frames': 211}, {'seed': -1}, {'seed': True},
                        {'prompt': 'bad\ncommand'}, {'request_id': '../other'}, {'extra': 2}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_job(dict(body(), **changes))
        with self.assertRaises(ValueError):
            make_server(self.manager(), TOKEN, host='0.0.0.0', port=0)

    def test_serial_bounded_queue_duplicates_and_cancelled_results(self):
        gate = threading.Event()
        sampler = FakeSampler(gate)
        manager = self.manager(sampler, max_queued=1)
        manager.submit(body('one'))
        self.assertTrue(sampler.started.wait(1))
        manager.submit(body('two'))
        with self.assertRaises(FileExistsError):
            manager.submit(body('one'))
        with self.assertRaises(OverflowError):
            manager.submit(body('three'))
        self.assertEqual(manager.cancel('two')['status'], 'cancelled')
        self.assertEqual(manager.cancel('one')['status'], 'cancelled')
        gate.set()
        manager.queue.join()
        self.assertEqual(sampler.calls, 1)
        self.assertEqual(sampler.peak_active, 1)
        for identifier in ('one', 'two'):
            with self.assertRaises(BlockingIOError):
                manager.result(identifier)

    def test_completed_result_storage_is_bounded(self):
        manager = self.manager(max_queued=1, max_retained=2)
        for identifier in ('one', 'two', 'three'):
            manager.submit(body(identifier))
            manager.jobs[identifier].done.wait(2)
            self.assertEqual(manager.status(identifier)['status'], 'complete')
        self.assertEqual(list(manager.jobs), ['two', 'three'])
        self.assertGreater(len(manager.result('three')), 0)
        self.assertEqual(manager.cancel('three')['status'], 'cancelled')
        with self.assertRaises(BlockingIOError):
            manager.result('three')

    def test_resident_reuses_process_and_bounds_sample_and_transfer_cancellation(self):
        sampler = ResidentSampler({}, load_timeout=5, child_target=fake_resident_child)
        try:
            sampler.warm()
            _, one = sampler.sample(body(), deadline=time.monotonic()+2, cancelled=lambda: False)
            _, two = sampler.sample(body(), deadline=time.monotonic()+2, cancelled=lambda: False)
            self.assertEqual(one['child_pid'], two['child_pid'])
            self.assertEqual(two['calls'], 2)
            for prompt in ('stall', 'stall-transfer'):
                sampler.warm()
                old_pid = sampler.process.pid
                start = time.monotonic()
                with self.assertRaises(TimeoutError):
                    sampler.sample(dict(body(), prompt=prompt), deadline=start+.1, cancelled=lambda: False)
                self.assertLess(time.monotonic()-start, 2)
                self.assertIsNone(sampler.process)
                with self.assertRaises(ProcessLookupError):
                    os.kill(old_pid, 0)
            sampler.warm()
            event = threading.Event()
            timer = threading.Timer(.05, event.set)
            timer.start()
            try:
                with self.assertRaises(InterruptedError):
                    sampler.sample(dict(body(), prompt='stall'), deadline=time.monotonic()+5, cancelled=event.is_set)
            finally:
                timer.cancel()
        finally:
            sampler.stop()

    def test_cold_reload_is_bounded_by_request_deadline(self):
        sampler = ResidentSampler({'stall_load': True}, load_timeout=5, child_target=fake_resident_child)
        try:
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                sampler.sample(body(), deadline=started+.1, cancelled=lambda: False)
            self.assertLess(time.monotonic()-started, 2)
            self.assertIsNone(sampler.process)
        finally:
            sampler.stop()

    def test_parent_sigterm_reaps_only_owned_resident_sampler(self):
        script = """
import json
from native_pair_worker import NativeJobManager, ResidentSampler, make_server, run_server
from test_native_pair_worker import fake_resident_child, TOKEN
sampler = ResidentSampler({}, child_target=fake_resident_child)
sampler.warm()
print(json.dumps({'child_pid': sampler.process.pid}), flush=True)
manager = NativeJobManager(sampler)
run_server(make_server(manager, TOKEN, port=0), manager)
"""
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        parent = subprocess.Popen([sys.executable, '-c', script], cwd=ROOT,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        child_pid = None
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(parent.stdout, selectors.EVENT_READ)
                self.assertTrue(selector.select(timeout=5), 'Test worker did not initialize')
            child_pid = json.loads(parent.stdout.readline())['child_pid']
            ready = json.loads(parent.stdout.readline())
            self.assertEqual(ready['event'], 'native_worker_ready')
            parent.terminate()
            _, errors = parent.communicate(timeout=5)
            self.assertEqual(parent.returncode, 0, errors.decode())
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
            self.assertIsNone(unrelated.poll())
        finally:
            if parent.poll() is None:
                parent.kill(); parent.wait(timeout=3)
            parent.stdout.close(); parent.stderr.close()
            if child_pid is not None:
                try:
                    os.kill(child_pid, 9)
                except ProcessLookupError:
                    pass
            unrelated.terminate(); unrelated.wait(timeout=3)

    def test_authenticated_provider_roundtrip_preserves_raw_native_arrays(self):
        manager = self.manager()
        server = make_server(manager, TOKEN, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = 'http://127.0.0.1:'+str(server.server_port)
        try:
            with self.assertRaises(HTTPError) as denied:
                urlopen(url+'/health', timeout=2)
            self.assertEqual(denied.exception.code, 401)
            request = Request(url+'/health', headers={'Authorization':'Bearer '+TOKEN})
            with urlopen(request, timeout=2) as response:
                self.assertTrue(json.load(response)['ready'])
            with tempfile.TemporaryDirectory() as directory:
                token_path = Path(directory)/'token'
                token_path.write_text(TOKEN)
                provider = NativePairProvider(NativePairConfig('host', 22, '/tmp/known_hosts',
                    worker_url=url, worker_token_path=str(token_path)))
                provider._ssh = lambda *_: self.fail('Warm provider must not invoke SSH')
                clip = provider.generate(PROMPT, 42)
                with np.load(FIXTURE, allow_pickle=False) as source:
                    np.testing.assert_array_equal(clip.joints, source['joints'])
                    np.testing.assert_array_equal(clip.features, source['features'])
                self.assertEqual(clip.metadata['provider'], 'intergen_warm_native')
                self.assertEqual(clip.frames, 210)
                self.assertIn('transfer_seconds', clip.metadata['provider_timings'])
                self.assertIsInstance(provider.last_raw_archive, bytes)
        finally:
            server.shutdown(); server.server_close(); thread.join(2)

    def test_provider_cancellation_discards_only_its_request(self):
        sampler = FakeSampler(threading.Event())
        manager = self.manager(sampler)
        server = make_server(manager, TOKEN, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                token_path = Path(directory)/'token'; token_path.write_text(TOKEN)
                provider = NativePairProvider(NativePairConfig('host', 22, '/tmp/known_hosts',
                    worker_url='http://127.0.0.1:'+str(server.server_port), worker_token_path=str(token_path)))
                sentinel = b'previous-good-archive'
                provider.last_raw_archive = sentinel
                with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                    provider.generate(PROMPT, 42, cancelled=sampler.started.is_set)
                self.assertIs(provider.last_raw_archive, sentinel)
                self.assertEqual(len(manager.jobs), 1)
                self.assertEqual(next(iter(manager.jobs.values())).status, 'cancelled')
        finally:
            server.shutdown(); server.server_close(); thread.join(2)

    def test_provider_rejects_wrong_provenance_without_replacing_good_archive(self):
        class WrongSeed(FakeSampler):
            def sample(self, job, *, deadline, cancelled):
                return fixture_archive(dict(job, seed=43)), {}
        manager = self.manager(WrongSeed())
        server = make_server(manager, TOKEN, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                token_path = Path(directory)/'token'; token_path.write_text(TOKEN)
                provider = NativePairProvider(NativePairConfig('host', 22, '/tmp/known_hosts',
                    worker_url='http://127.0.0.1:'+str(server.server_port), worker_token_path=str(token_path)))
                provider.last_raw_archive = b'previous-good-archive'
                with self.assertRaisesRegex(ValueError, 'incompatible arrays or provenance'):
                    provider.generate(PROMPT, 42)
                self.assertEqual(provider.last_raw_archive, b'previous-good-archive')
                self.assertEqual(next(iter(manager.jobs.values())).status, 'cancelled')
        finally:
            server.shutdown(); server.server_close(); thread.join(2)

    def test_worker_config_never_allows_token_routing_outside_loopback(self):
        for url in ('https://example.com:8772', 'http://example.com:8772',
                    'http://user:password@127.0.0.1:8772', 'http://127.0.0.1:8772/path',
                    'http://127.0.0.1:8772?token=test', 'http://127.0.0.1'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                NativePairConfig('host', 22, '/tmp/known_hosts', worker_url=url, worker_token_path='/tmp/token')
        with self.assertRaises(ValueError):
            NativePairConfig('host', 22, '/tmp/known_hosts', worker_url='http://127.0.0.1:8772')


if __name__ == '__main__':
    unittest.main()
