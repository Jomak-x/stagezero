"""Regression tests for per-transfer snapshots of Viser upload callbacks."""

import asyncio
from collections import defaultdict
from concurrent.futures import Future
from contextlib import redirect_stderr
import io
from types import SimpleNamespace
import threading
import unittest

from viser import _messages
from viser._gui_api import GuiApi
from viser._viser import ViserServer

from bounded_upload import ScopedUploadLimits
from upload_events import install_upload_snapshots


class _Interface:
    def __init__(self):
        self.handlers = defaultdict(list)
        self.sent = []

    def register_handler(self, cls, callback):
        self.handlers[cls].append(callback)

    def unregister_handler(self, cls, callback):
        self.handlers[cls].remove(callback)

    def queue_message(self, message):
        self.sent.append(message)

    def dispatch(self, client_id, message):
        for callback in tuple(self.handlers[type(message)]):
            callback(client_id, message)


class _QueuedExecutor:
    """Run Viser's synchronous callbacks only when the test requests it."""

    def __init__(self):
        self.jobs = []

    def submit(self, callback, *args, **kwargs):
        future = Future()
        self.jobs.append((future, callback, args, kwargs))
        return future

    def drain(self):
        errors = []
        while self.jobs:
            future, callback, args, kwargs = self.jobs.pop(0)
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(callback(*args, **kwargs))
            except Exception as exc:
                errors.append(exc)
                future.set_exception(exc)
        return errors


def _start(component, transfer, filename, size):
    return _messages.FileTransferStartUpload(
        component, transfer, filename, "application/octet-stream", 1, size
    )


def _part(component, transfer, content):
    return _messages.FileTransferPart(component, transfer, 0, content)


class UploadEventSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        self.interface = _Interface()
        self.executor = _QueuedExecutor()
        self.server = object.__new__(ViserServer)
        self.server._connected_clients = {
            client_id: SimpleNamespace(client_id=client_id)
            for client_id in (1, 2)
        }
        self.server._client_lock = threading.Lock()
        self.server._client_disconnect_cb = []
        self.gui = object.__new__(GuiApi)
        self.gui._owner = self.server
        self.gui._event_loop = self.loop
        self.gui._thread_executor = self.executor
        self.gui._websock_interface = self.interface
        self.gui._gui_input_handle_from_uuid = {}
        self.gui._current_file_upload_states = {}
        self.server.gui = self.gui
        self.interface.register_handler(
            _messages.FileTransferStartUpload, self.gui._handle_file_transfer_start
        )
        self.interface.register_handler(
            _messages.FileTransferPart, self.gui._handle_file_transfer_part
        )
        self.limits = None

    def tearDown(self):
        if self.limits is not None:
            self.limits.close()
        pending = asyncio.all_tasks(self.loop)
        for task in pending:
            task.cancel()
        if pending:
            self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self.loop.close()

    def control(self, callback, component="upload"):
        handle = SimpleNamespace(
            _impl=SimpleNamespace(uuid=component, value=None, update_cb=[callback])
        )
        self.gui._gui_input_handle_from_uuid[component] = handle
        return handle

    def complete(self, client_id, transfer, filename, content, component="upload"):
        self.interface.dispatch(
            client_id, _start(component, transfer, filename, len(content))
        )
        self.interface.dispatch(client_id, _part(component, transfer, content))

    def assert_upload(self, observed, handle, *, client_id, transfer, filename, content):
        event, seen_filename, seen_content = observed
        self.assertIs(event.target, handle)
        self.assertIs(event.client, self.server._connected_clients[client_id])
        self.assertEqual(event.client_id, client_id)
        self.assertEqual(event.transfer_uuid, transfer)
        self.assertEqual((seen_filename, seen_content), (filename, content))
        self.assertEqual((event.file.name, event.file.content), (filename, content))

    def _check_two_clients_before_sync_drain(self, *, guarded):
        observed = []

        def on_upload(event):
            observed.append((event, event.file.name, event.file.content))

        handle = self.control(on_upload)
        if guarded:
            # ScopedUploadLimits must install snapshots before capturing its
            # native transfer callbacks; callers need no separate installer.
            errors = []
            self.limits = ScopedUploadLimits(self.server)
            self.limits.register(handle, max_bytes=100, on_error=errors.append)
        else:
            install_upload_snapshots(self.server)

        self.complete(1, "transfer-a", "alpha.bin", b"first")
        self.complete(2, "transfer-b", "beta.bin", b"second")
        self.assertEqual(self.executor.drain(), [])
        self.assertEqual(len(observed), 2)
        self.assert_upload(
            observed[0], handle, client_id=1, transfer="transfer-a",
            filename="alpha.bin", content=b"first",
        )
        self.assert_upload(
            observed[1], handle, client_id=2, transfer="transfer-b",
            filename="beta.bin", content=b"second",
        )
        self.assertIsNot(observed[0][0].file, observed[1][0].file)
        with self.assertRaises(AttributeError):
            observed[0][0].file.name = "changed.bin"
        if guarded:
            self.assertEqual(errors, [])
        self.assertEqual(self.gui._current_file_upload_states, {})

    def test_unguarded_uploads_keep_each_client_and_file_before_sync_drain(self):
        self._check_two_clients_before_sync_drain(guarded=False)

    def test_guarded_uploads_keep_each_client_and_file_before_sync_drain(self):
        self._check_two_clients_before_sync_drain(guarded=True)

    def test_same_client_successive_uploads_keep_their_own_files(self):
        observed = []

        def on_upload(event):
            observed.append((event, event.file.name, event.file.content))

        handle = self.control(on_upload)
        install_upload_snapshots(self.server)
        self.complete(1, "first-id", "first.txt", b"one")
        self.complete(1, "second-id", "second.txt", b"two")
        self.assertEqual(self.executor.drain(), [])
        self.assertEqual(len(observed), 2)
        self.assert_upload(
            observed[0], handle, client_id=1, transfer="first-id",
            filename="first.txt", content=b"one",
        )
        self.assert_upload(
            observed[1], handle, client_id=1, transfer="second-id",
            filename="second.txt", content=b"two",
        )

    def test_async_callback_keeps_its_file_after_another_upload_completes(self):
        release_first = asyncio.Event()
        started_first = asyncio.Event()
        observed = []
        callback_number = 0

        async def on_upload(event):
            nonlocal callback_number
            callback_number += 1
            if callback_number == 1:
                started_first.set()
                await release_first.wait()
            observed.append((event, event.file.name, event.file.content))

        handle = self.control(on_upload)
        install_upload_snapshots(self.server)
        self.complete(1, "held", "held.bin", b"before")
        self.loop.run_until_complete(asyncio.wait_for(started_first.wait(), 1))
        self.complete(2, "later", "later.bin", b"after")
        self.loop.run_until_complete(asyncio.sleep(0))
        self.assertEqual(len(observed), 1)
        self.assert_upload(
            observed[0], handle, client_id=2, transfer="later",
            filename="later.bin", content=b"after",
        )
        release_first.set()
        self.loop.run_until_complete(asyncio.sleep(0))
        self.assertEqual(len(observed), 2)
        self.assert_upload(
            observed[1], handle, client_id=1, transfer="held",
            filename="held.bin", content=b"before",
        )
        self.assertEqual(self.executor.jobs, [])

    def test_installation_is_idempotent(self):
        observed = []

        def on_upload(event):
            observed.append(event.transfer_uuid)

        self.control(on_upload)
        install_upload_snapshots(self.server)
        install_upload_snapshots(self.server)
        self.assertEqual(
            len(self.interface.handlers[_messages.FileTransferStartUpload]), 1
        )
        self.assertEqual(len(self.interface.handlers[_messages.FileTransferPart]), 1)
        self.complete(1, "once", "once.bin", b"data")
        self.assertEqual(self.executor.drain(), [])
        self.assertEqual(observed, ["once"])

    def test_callback_error_does_not_poison_a_later_transfer(self):
        observed = []

        def on_upload(event):
            if event.transfer_uuid == "bad":
                raise RuntimeError("callback failed")
            observed.append((event, event.file.name, event.file.content))

        handle = self.control(on_upload)
        install_upload_snapshots(self.server)
        self.complete(1, "bad", "bad.bin", b"bad")
        with redirect_stderr(io.StringIO()):
            failures = self.executor.drain()
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], RuntimeError)
        self.complete(2, "good", "good.bin", b"good")
        self.assertEqual(self.executor.drain(), [])
        self.assertEqual(len(observed), 1)
        self.assert_upload(
            observed[0], handle, client_id=2, transfer="good",
            filename="good.bin", content=b"good",
        )

    def test_snapshot_transfer_keeps_starting_client_and_handle_identity(self):
        observed = []
        handle = self.control(lambda event: observed.append(event.file.content))
        install_upload_snapshots(self.server)
        self.interface.dispatch(1, _start('upload', 'owned', 'first.bin', 4))
        self.interface.dispatch(2, _part('upload', 'owned', b'evil'))
        self.interface.dispatch(1, _part('other-control', 'owned', b'evil'))
        self.assertEqual(self.executor.jobs, [])
        self.interface.dispatch(1, _part('upload', 'owned', b'good'))
        self.assertEqual(self.executor.drain(), [])
        self.assertEqual(observed, [b'good'])

        self.interface.dispatch(1, _start('upload', 'retired', 'old.bin', 4))
        replacement = self.control(lambda event: observed.append(event.file.content))
        self.assertIsNot(replacement, handle)
        self.interface.dispatch(1, _part('upload', 'retired', b'old!'))
        self.assertNotIn('retired', self.gui._current_file_upload_states)
        self.complete(2, 'new', 'new.bin', b'new!')
        self.assertEqual(self.executor.drain(), [])
        self.assertEqual(observed, [b'good', b'new!'])

    def test_completion_does_not_wait_for_viser_connection_callback_lock(self):
        observed = []
        self.control(lambda event: observed.append(event.file.content))
        install_upload_snapshots(self.server)

        class HeldConnectionLock:
            def __enter__(self):
                raise AssertionError('Would deadlock the Viser event loop')

            def __exit__(self, *args):
                pass

        self.server._client_lock = HeldConnectionLock()
        self.complete(1, 'a', 'first.bin', b'first')
        self.assertEqual(self.executor.drain(), [])
        self.assertEqual(observed, [b'first'])


if __name__ == "__main__":
    unittest.main()
