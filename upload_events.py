"""Per-transfer upload events for the pinned Viser 1.0.16 GUI API.

Viser's native event points at a mutable handle. Capture the file before queued
callbacks run, while retaining the real handle for GUI updates. This adapter
does not impose file-size limits; ScopedUploadLimits remains responsible for
the controls registered with it.
"""

import asyncio
from dataclasses import dataclass
import threading
import time

from viser import _messages
from viser._gui_handles import UploadedFile
from viser._threadpool_exceptions import print_threadpool_errors
from viser._viser import ClientHandle, ViserServer


@dataclass(frozen=True)
class UploadFile:
    name: str
    content: bytes


@dataclass(frozen=True)
class UploadEvent:
    client: object
    client_id: object
    target: object
    transfer_uuid: str
    file: UploadFile


def install_upload_snapshots(server_or_gui):
    """Install once per GuiApi, before wrapping it with ScopedUploadLimits."""
    gui = getattr(server_or_gui, 'gui', server_or_gui)
    if getattr(gui, '_stagezero_upload_snapshots', None) is None:
        gui._stagezero_upload_snapshots = _UploadSnapshots(gui)
    return gui._stagezero_upload_snapshots


class _UploadSnapshots:
    def __init__(self, gui):
        self.gui = gui
        self.lock = threading.RLock()
        self.original_start = gui._handle_file_transfer_start
        interface = gui._websock_interface
        interface.unregister_handler(_messages.FileTransferStartUpload, self.original_start)
        interface.unregister_handler(_messages.FileTransferPart, gui._handle_file_transfer_part)
        # Keep the instance methods coherent: ScopedUploadLimits captures and
        # restores this pair, so closing limits cannot remove snapshot delivery.
        gui._handle_file_transfer_start = self.start
        gui._handle_file_transfer_part = self.part
        interface.register_handler(_messages.FileTransferStartUpload, self.start)
        interface.register_handler(_messages.FileTransferPart, self.part)

    def start(self, client_id, message):
        with self.lock:
            states = self.gui._current_file_upload_states
            if message.transfer_uuid in states:
                return
            handle = self.gui._gui_input_handle_from_uuid.get(message.source_component_uuid)
            if handle is None:
                return
            self.original_start(client_id, message)
            state = states.get(message.transfer_uuid)
            if state is not None:
                state['snapshot_owner'] = (client_id, message.source_component_uuid, handle)

    def part(self, client_id, message):
        with self.lock:
            gui = self.gui
            state = gui._current_file_upload_states.get(message.transfer_uuid)
            if state is None:
                return
            owner, component_id, handle = state['snapshot_owner']
            if client_id != owner or message.source_component_uuid != component_id:
                return
            if gui._gui_input_handle_from_uuid.get(component_id) is not handle:
                gui._current_file_upload_states.pop(message.transfer_uuid, None)
                return
            state['parts'][message.part_index] = message.content
            state['transferred_bytes'] += len(message.content)
            gui._websock_interface.queue_message(_messages.FileTransferPartAck(
                source_component_uuid=component_id,
                transfer_uuid=message.transfer_uuid,
                transferred_bytes=state['transferred_bytes'],
                total_bytes=state['total_bytes'],
            ))
            if state['transferred_bytes'] < state['total_bytes']:
                return
            assert state['transferred_bytes'] == state['total_bytes']
            gui._current_file_upload_states.pop(message.transfer_uuid)
            file = UploadFile(state['filename'], b''.join(
                state['parts'][i] for i in range(state['part_count'])))
            handle._impl.value = UploadedFile(name=file.name, content=file.content)
            handle._impl.update_timestamp = time.time()
            if isinstance(gui._owner, ClientHandle):
                client = gui._owner
            elif isinstance(gui._owner, ViserServer):
                # Match Viser's event-loop lookup. get_clients() takes a
                # non-reentrant lock held across async connection callbacks;
                # waiting on it here can prevent that callback from resuming.
                client = gui._owner._connected_clients.get(client_id)
            else:
                raise TypeError('Unsupported Viser GUI owner')
            if client is None:
                return
            event = UploadEvent(client, client_id, handle, message.transfer_uuid, file)
            callbacks = tuple(handle._impl.update_cb)
            for callback in callbacks:
                if asyncio.iscoroutinefunction(callback):
                    gui._event_loop.create_task(callback(event))
                else:
                    gui._thread_executor.submit(callback, event).add_done_callback(
                        print_threadpool_errors)
