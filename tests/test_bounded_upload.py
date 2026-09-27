"""Test accepted uploads through Viser's actual transfer callbacks."""

import asyncio
from collections import defaultdict
from types import SimpleNamespace
import threading
import unittest

import msgspec
from viser import _messages
from viser._gui_api import GuiApi
from viser._viser import ViserServer

from bounded_upload import (ScopedUploadLimits, UploadRejectedMessage,
                            acquire_scoped_upload_limits, release_scoped_upload_limits)
from character_assets import DEFAULT_LIMITS


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


def _start(component, transfer, size, parts=1):
    return _messages.FileTransferStartUpload(component, transfer, "character.glb", "model/gltf-binary", parts, size)


def _part(component, transfer, index, content):
    return _messages.FileTransferPart(component, transfer, index, content)


class ScopedUploadLimitsTest(unittest.TestCase):
    def test_character_upload_declaration_accepts_500mb_and_rejects_one_byte_more(self):
        errors = []
        file_cap = DEFAULT_LIMITS.max_file_bytes
        parts = (file_cap + 512 * 1024 - 1) // (512 * 1024)
        self.guard(self.control("glb"), errors, max_bytes=file_cap,
                   max_total_bytes=file_cap + 1024 * 1024)
        self.interface.dispatch(1, _start("glb", "at-limit", file_cap, parts=parts))
        self.assertIn("at-limit", self.gui._current_file_upload_states)
        self.assertEqual(self.gui._current_file_upload_states["at-limit"]["total_bytes"], file_cap)
        self.interface.dispatch(1, _start("glb", "over-limit", file_cap + 1, parts=parts))
        self.assertNotIn("over-limit", self.gui._current_file_upload_states)
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(self.rejections[1]), 1)

    def test_rejection_message_round_trips_through_viser_codec(self):
        message = UploadRejectedMessage("glb", "transfer", "Too large")
        encoded = msgspec.msgpack.encode(message.as_serializable_dict())
        self.assertEqual(_messages.Message.deserialize(encoded), message)

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        self.interface = _Interface()
        self.rejections = defaultdict(list)
        self.server = object.__new__(ViserServer)
        self.server._connected_clients = {
            client_id: SimpleNamespace(
                client_id=client_id,
                _websock_connection=SimpleNamespace(
                    queue_message=self.rejections[client_id].append
                ),
            )
            for client_id in (1, 2)
        }
        self.server._client_lock = threading.Lock()
        self.server._client_disconnect_cb = []
        self.gui = object.__new__(GuiApi)
        self.gui._owner = self.server
        self.gui._event_loop = self.loop
        self.gui._websock_interface = self.interface
        self.gui._gui_input_handle_from_uuid = {}
        self.gui._current_file_upload_states = {}
        self.server.gui = self.gui
        self.interface.register_handler(_messages.FileTransferStartUpload, self.gui._handle_file_transfer_start)
        self.interface.register_handler(_messages.FileTransferPart, self.gui._handle_file_transfer_part)
        self.completed = []
        self.limits = None

    def tearDown(self):
        if self.limits is not None:
            self.limits.close()
        self.loop.run_until_complete(asyncio.sleep(0))
        self.loop.close()

    def control(self, component_id):
        async def on_upload(event):
            self.completed.append((component_id, event.client_id, event.target._impl.value.content))

        handle = SimpleNamespace(_impl=SimpleNamespace(uuid=component_id, value=None, update_cb=[on_upload]))
        handle.remove = lambda: self.gui._gui_input_handle_from_uuid.pop(component_id, None)
        self.gui._gui_input_handle_from_uuid[component_id] = handle
        return handle

    def guard(self, handle, errors, *, max_bytes=100_000, **kwargs):
        self.limits = ScopedUploadLimits(self.server, **kwargs)
        self.limits.register(handle, max_bytes=max_bytes, on_error=errors.append)

    def test_scene_and_character_controls_share_one_transfer_wrapper(self):
        file_cap = DEFAULT_LIMITS.max_file_bytes
        scene = self.control('scene')
        character = self.control('character')
        scene_limits = acquire_scoped_upload_limits(self.server, gui=self.gui)
        character_limits = acquire_scoped_upload_limits(
            self.server, gui=self.gui, max_total_bytes=file_cap + 1024 * 1024)
        self.assertIs(scene_limits, character_limits)
        self.assertEqual(character_limits._max_total, file_cap + 1024 * 1024)
        self.assertEqual(len(self.interface.handlers[_messages.FileTransferStartUpload]), 1)
        self.assertEqual(len(self.interface.handlers[_messages.FileTransferPart]), 1)
        errors = []
        scene_limits.register(scene, max_bytes=1_000_000, on_error=errors.append)
        character_limits.register(character, max_bytes=file_cap, on_error=errors.append)
        self.interface.dispatch(1, _start('scene', 'too-large', 1_000_001))
        self.assertNotIn('too-large', self.gui._current_file_upload_states)
        self.interface.dispatch(1, _start('character', 'rig', 4))
        self.interface.dispatch(1, _part('character', 'rig', 0, b'data'))
        release_scoped_upload_limits(self.gui)
        self.interface.dispatch(1, _start('scene', 'valid', 4))
        self.interface.dispatch(1, _part('scene', 'valid', 0, b'json'))
        self.drain()
        self.assertEqual(self.completed, [('character', 1, b'data'), ('scene', 1, b'json')])
        self.assertEqual(len(errors), 1)
        release_scoped_upload_limits(self.gui)
        self.assertEqual(len(self.interface.handlers[_messages.FileTransferStartUpload]), 1)
        self.assertEqual(len(self.interface.handlers[_messages.FileTransferPart]), 1)

    def drain(self):
        self.loop.run_until_complete(asyncio.sleep(0))

    def test_guarded_upload_reaches_viser_once_and_other_control_is_unchanged(self):
        errors = []
        self.guard(self.control("glb"), errors)
        self.control("project")
        self.interface.dispatch(1, _start("project", "ordinary", 4))
        self.interface.dispatch(1, _part("project", "ordinary", 0, b"data"))
        self.interface.dispatch(1, _start("glb", "guarded", 65_537, parts=2))
        self.interface.dispatch(1, _part("glb", "guarded", 1, b"z"))
        self.interface.dispatch(1, _part("glb", "guarded", 0, b"a" * 65_536))
        self.drain()
        self.assertEqual(self.completed, [("project", 1, b"data"), ("glb", 1, b"a" * 65_536 + b"z")])
        self.assertEqual(errors, [])
        self.assertEqual(self.gui._current_file_upload_states, {})

    def test_unregister_removes_only_target_and_discards_its_transfer_buffers(self):
        errors = []
        first = self.control('first')
        self.guard(first, errors)
        second = self.control('second')
        self.limits.register(second, max_bytes=100_000, on_error=errors.append)
        self.interface.dispatch(1, _start('first', 'old-mapping', 65_537, parts=2))
        self.interface.dispatch(1, _part('first', 'old-mapping', 0, b'a' * 65_536))
        self.interface.dispatch(2, _start('second', 'keep-mapping', 4))
        self.limits.unregister(first, remove=True)
        self.assertNotIn('first', self.gui._gui_input_handle_from_uuid)
        self.assertNotIn('first', self.limits._controls)
        self.assertNotIn('old-mapping', self.gui._current_file_upload_states)
        self.assertIn('keep-mapping', self.gui._current_file_upload_states)
        self.interface.dispatch(1, _part('first', 'old-mapping', 1, b'z'))
        self.interface.dispatch(1, _start('first', 'stale-control', 1))
        self.interface.dispatch(1, _part('first', 'stale-control', 0, b'x'))
        self.interface.dispatch(2, _part('second', 'keep-mapping', 0, b'data'))
        self.drain()
        self.assertEqual(self.completed, [('second', 2, b'data')])
        self.assertEqual(self.gui._current_file_upload_states, {})
        self.assertEqual(errors, [])
        self.assertEqual(len(self.rejections[1]), 1)
        self.assertEqual(self.rejections[1][0].transfer_uuid, 'old-mapping')
        self.assertEqual(self.rejections[2], [])
        self.limits.unregister(first, remove=True)  # Retiring twice is harmless.

    def test_oversized_or_malformed_start_never_reaches_native_buffer(self):
        errors = []
        self.guard(self.control("glb"), errors)
        self.interface.dispatch(1, _start("glb", "oversize", 100_001))
        self.interface.dispatch(1, _part("glb", "oversize", 0, b"ignored"))
        self.interface.dispatch(1, _start("glb", "many-parts", 65_537, parts=200))
        self.interface.dispatch(1, _start("glb", "empty", 0, parts=0))
        self.assertEqual(self.gui._current_file_upload_states, {})
        self.assertEqual(self.completed, [])
        self.assertEqual(len(errors), 3)
        self.assertEqual([m.transfer_uuid for m in self.rejections[1]],
                         ["oversize", "many-parts", "empty"])
        self.assertEqual(self.rejections[2], [])

    def test_invalid_parts_abort_and_release_native_buffer(self):
        for bad_part in (
            _part("glb", "x", 0, b"duplicate"),
            _part("glb", "x", 2, b"out-of-range"),
            _part("glb", "x", 1, b"b" * (512 * 1024 + 1)),
            _part("glb", "x", 1, []),
        ):
            with self.subTest(index=bad_part.part_index, size=len(bad_part.content)):
                errors = []
                self.guard(self.control("glb"), errors, max_bytes=700_000)
                self.interface.dispatch(1, _start("glb", "x", 600_000, parts=2))
                self.interface.dispatch(1, _part("glb", "x", 0, b"a" * 100_000))
                self.interface.dispatch(1, bad_part)
                self.assertNotIn("x", self.gui._current_file_upload_states)
                self.assertEqual(self.completed, [])
                self.assertEqual(len(errors), 1)
                self.limits.close()
                self.limits = None

    def test_foreign_client_and_component_cannot_modify_guarded_transfer(self):
        errors = []
        self.guard(self.control("glb"), errors)
        self.control("project")
        self.interface.dispatch(1, _start("glb", "x", 4))
        self.interface.dispatch(2, _part("glb", "x", 0, b"evil"))
        self.interface.dispatch(1, _part("project", "x", 0, b"evil"))
        self.interface.dispatch(2, _start("project", "x", 4))
        self.interface.dispatch(1, _part("glb", "x", 0, b"good"))
        self.drain()
        self.assertEqual(self.completed, [("glb", 1, b"good")])
        self.assertEqual(errors, [])
        self.assertEqual(self.rejections[1], [])
        self.assertEqual(self.rejections[2], [])

    def test_budget_timeout_and_disconnect_release_pending_uploads(self):
        now = [0.0]
        errors = []
        self.guard(self.control("glb"), errors, max_bytes=150_000,
                   max_total_bytes=150_000, timeout_seconds=10, clock=lambda: now[0])
        self.interface.dispatch(1, _start("glb", "first", 100_000))
        self.interface.dispatch(2, _start("glb", "over-budget", 60_000))
        self.assertEqual(set(self.gui._current_file_upload_states), {"first"})
        self.assertEqual(len(errors), 1)
        self.assertEqual([m.transfer_uuid for m in self.rejections[2]], ["over-budget"])
        now[0] = 11
        self.limits.poll()
        self.assertEqual(self.gui._current_file_upload_states, {})
        self.assertEqual(len(errors), 2)
        self.assertEqual([m.transfer_uuid for m in self.rejections[1]], ["first"])
        self.interface.dispatch(2, _start("glb", "after-timeout", 70_000, parts=2))
        self.interface.dispatch(2, _part("glb", "after-timeout", 0, b"b" * 20))
        self.assertIn("after-timeout", self.gui._current_file_upload_states)
        self.limits._on_disconnect(self.server._connected_clients[2])
        self.assertEqual(self.gui._current_file_upload_states, {})
        self.interface.dispatch(1, _start("glb", "again", 4))
        self.interface.dispatch(1, _part("glb", "again", 0, b"good"))
        self.drain()
        self.assertEqual(self.completed, [("glb", 1, b"good")])

    def test_active_transfer_limit_rejects_then_recovers(self):
        errors = []
        self.guard(self.control("glb"), errors, max_bytes=200_000,
                   max_active_transfers=2, max_total_bytes=400_000)
        self.interface.dispatch(1, _start("glb", "one", 100_000, parts=2))
        self.interface.dispatch(2, _start("glb", "two", 100_000, parts=2))
        self.interface.dispatch(1, _start("glb", "three", 100_000, parts=2))
        self.assertEqual(set(self.gui._current_file_upload_states), {"one", "two"})
        self.assertEqual([m.transfer_uuid for m in self.rejections[1]], ["three"])
        self.assertEqual(len(errors), 1)
        self.limits._on_disconnect(self.server._connected_clients[2])
        self.interface.dispatch(1, _start("glb", "retry", 100_000, parts=2))
        self.assertEqual(set(self.gui._current_file_upload_states), {"one", "retry"})

    def test_close_restores_native_handlers(self):
        self.control("project")
        self.limits = ScopedUploadLimits(self.server)
        self.limits.close()
        self.assertEqual(len(self.interface.handlers[_messages.FileTransferStartUpload]), 1)
        self.assertEqual(len(self.interface.handlers[_messages.FileTransferPart]), 1)
        self.assertEqual(self.server._client_disconnect_cb, [])
        self.interface.dispatch(1, _start("project", "ordinary", 4))
        self.interface.dispatch(1, _part("project", "ordinary", 0, b"data"))
        self.drain()
        self.assertEqual(self.completed, [("project", 1, b"data")])
