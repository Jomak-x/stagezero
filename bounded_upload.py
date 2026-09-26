"""Bound Viser uploads for selected GUI controls without changing other uploads.

Viser 1.0.16 buffers upload parts in its GuiApi before invoking on_upload. This
wrapper replaces only that GuiApi's two transfer callbacks, validates selected
controls before forwarding to Viser, and delegates all other controls unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import threading
import time
from typing import Callable

from viser import _messages


_LOG = logging.getLogger(__name__)
_DEFAULT_CHUNK_BYTES = 512 * 1024
_MIN_PART_BYTES = 64 * 1024


@dataclass
class UploadRejectedMessage(_messages.Message):
    """Tell one browser tab to clear a failed scoped upload by transfer ID."""

    control_uuid: str
    transfer_uuid: str
    error: str

    def redundancy_key(self) -> str:
        return f"upload-rejected-{self.transfer_uuid}"


@dataclass
class _Control:
    handle: object
    max_bytes: int
    on_error: Callable[[str], None]


@dataclass
class _Transfer:
    client_id: object
    component_id: str
    size_bytes: int
    part_count: int
    deadline: float
    received_bytes: int = 0
    seen: set[int] = field(default_factory=set)


class ScopedUploadLimits:
    """Apply byte, part, client, and lifetime limits to registered uploads.

    Call ``poll()`` from the server's regular tick to expire idle transfers.
    Call ``close()`` during teardown to restore Viser's original callbacks.
    ``on_error`` receives a user-facing message for a rejected upload.
    """

    def __init__(
        self,
        server: object,
        *,
        max_active_transfers: int = 2,
        max_total_bytes: int = 64 * 1024 * 1024,
        max_chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
        timeout_seconds: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_active_transfers < 1 or max_total_bytes < 1 or max_chunk_bytes < 1:
            raise ValueError("Upload limits must be positive.")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Upload timeout must be positive and finite.")

        self._server = server
        self._gui = server.gui
        self._interface = self._gui._websock_interface
        self._original_start = self._gui._handle_file_transfer_start
        self._original_part = self._gui._handle_file_transfer_part
        self._controls: dict[str, _Control] = {}
        self._transfers: dict[str, _Transfer] = {}
        self._max_active = max_active_transfers
        self._max_total = max_total_bytes
        self._max_chunk = max_chunk_bytes
        self._timeout = timeout_seconds
        self._clock = clock
        self._lock = threading.RLock()
        self._closed = False

        # Viser calls all handlers for a message type. Remove the native pair
        # before registering the wrappers so no upload is processed twice.
        self._interface.unregister_handler(
            _messages.FileTransferStartUpload, self._original_start
        )
        try:
            self._interface.unregister_handler(
                _messages.FileTransferPart, self._original_part
            )
        except Exception:
            self._interface.register_handler(
                _messages.FileTransferStartUpload, self._original_start
            )
            raise
        self._interface.register_handler(
            _messages.FileTransferStartUpload, self._handle_start
        )
        self._interface.register_handler(_messages.FileTransferPart, self._handle_part)
        server.on_client_disconnect(self._on_disconnect)

    def register(
        self,
        handle: object,
        *,
        max_bytes: int,
        on_error: Callable[[str], None],
    ) -> object:
        """Limit a server GUI upload handle; return it for convenient assignment."""
        if max_bytes < 1 or max_bytes > self._max_total:
            raise ValueError("Upload size must fit the aggregate byte budget.")
        component_id = handle._impl.uuid
        with self._lock:
            if self._closed:
                raise RuntimeError("Upload limits are closed.")
            if self._gui._gui_input_handle_from_uuid.get(component_id) is not handle:
                raise ValueError("Upload handle does not belong to this server GUI.")
            self._controls[component_id] = _Control(handle, max_bytes, on_error)
        return handle

    def _reject(
        self, client_id: object, component_id: str, transfer_id: str, message: str
    ) -> None:
        control = self._controls.get(component_id)
        if control is not None:
            try:
                control.on_error(message)
            except Exception:
                _LOG.exception("Upload error callback failed")
        client = self._server.get_clients().get(client_id)
        if client is not None:
            try:
                client._websock_connection.queue_message(
                    UploadRejectedMessage(component_id, transfer_id, message)
                )
            except Exception:
                _LOG.exception("Could not send upload rejection to client")

    def _drop(self, transfer_id: str) -> _Transfer | None:
        transfer = self._transfers.pop(transfer_id, None)
        if transfer is not None:
            self._gui._current_file_upload_states.pop(transfer_id, None)
        return transfer

    def _expire_stale(self) -> list[tuple[str, _Transfer]]:
        now = self._clock()
        expired: list[tuple[str, _Transfer]] = []
        for transfer_id, transfer in tuple(self._transfers.items()):
            if transfer.deadline <= now:
                self._drop(transfer_id)
                expired.append((transfer_id, transfer))
        return expired

    def poll(self) -> None:
        """Remove uploads idle past the timeout, including Viser's part buffers."""
        with self._lock:
            if self._closed:
                return
            expired = self._expire_stale()
        for transfer_id, transfer in expired:
            self._reject(transfer.client_id, transfer.component_id, transfer_id,
                         "Upload timed out. Please try again.")

    def _handle_start(
        self, client_id: object, message: _messages.FileTransferStartUpload
    ) -> None:
        error: str | None = None
        component_id = message.source_component_uuid
        if not isinstance(component_id, str) or not isinstance(message.transfer_uuid, str):
            return
        with self._lock:
            if self._closed:
                return
            expired = self._expire_stale()
            # Never let another control reset an active guarded transfer.
            if message.transfer_uuid in self._transfers:
                if component_id in self._controls:
                    error = "An upload with this transfer ID is already active."
            elif component_id not in self._controls:
                self._original_start(client_id, message)
            else:
                control = self._controls[component_id]
                size = message.size_bytes
                count = message.part_count
                if self._gui._gui_input_handle_from_uuid.get(component_id) is not control.handle:
                    error = "Upload control is no longer available."
                elif message.transfer_uuid in self._gui._current_file_upload_states:
                    error = "An upload with this transfer ID is already active."
                elif not 1 <= len(message.transfer_uuid) <= 128:
                    error = "Upload transfer ID is invalid."
                elif (not isinstance(message.filename, str) or
                      not 1 <= len(message.filename) <= 255 or
                      not isinstance(message.mime_type, str) or
                      len(message.mime_type) > 128):
                    error = "Upload filename or MIME type is invalid."
                elif type(size) is not int or not 1 <= size <= control.max_bytes:
                    max_mib = control.max_bytes / (1024 * 1024)
                    error = f"File must be between 1 byte and {max_mib:g} MiB."
                elif type(count) is not int or not 1 <= count <= min(
                    size, (size + _MIN_PART_BYTES - 1) // _MIN_PART_BYTES
                ):
                    error = "Upload has too many or invalid parts."
                elif len(self._transfers) >= self._max_active:
                    error = "Too many uploads are active. Please try again shortly."
                elif size + sum(t.size_bytes for t in self._transfers.values()) > self._max_total:
                    error = "Upload memory budget is full. Please try again shortly."
                else:
                    self._transfers[message.transfer_uuid] = _Transfer(
                        client_id, component_id, size, count,
                        self._clock() + self._timeout,
                    )
                    try:
                        self._original_start(client_id, message)
                    except Exception:
                        self._drop(message.transfer_uuid)
                        raise
        for expired_id, expired_transfer in expired:
            self._reject(expired_transfer.client_id, expired_transfer.component_id,
                         expired_id, "Upload timed out. Please try again.")
        if error is not None:
            self._reject(client_id, component_id, message.transfer_uuid, error)

    def _handle_part(self, client_id: object, message: _messages.FileTransferPart) -> None:
        error: str | None = None
        component_id = message.source_component_uuid
        if component_id is not None and not isinstance(component_id, str):
            return
        if not isinstance(message.transfer_uuid, str):
            return
        with self._lock:
            if self._closed:
                return
            expired = self._expire_stale()
            transfer = self._transfers.get(message.transfer_uuid)
            if transfer is None:
                if component_id not in self._controls:
                    self._original_part(client_id, message)
            elif client_id != transfer.client_id or component_id != transfer.component_id:
                # A second client/control cannot alter or abort another upload.
                pass
            else:
                index = message.part_index
                valid_content = isinstance(message.content, bytes)
                length = len(message.content) if valid_content else 0
                next_bytes = transfer.received_bytes + length
                next_parts = len(transfer.seen) + 1
                if message.transfer_uuid not in self._gui._current_file_upload_states:
                    error = "Upload state was lost. Please try again."
                elif type(index) is not int or not 0 <= index < transfer.part_count:
                    error = "Upload part index is invalid."
                elif index in transfer.seen:
                    error = "Upload contains a duplicate part."
                elif not valid_content:
                    error = "Upload part content is invalid."
                elif not 1 <= length <= self._max_chunk:
                    error = "Upload part exceeds the chunk size limit."
                elif next_bytes > transfer.size_bytes:
                    error = "Upload exceeds its declared size."
                elif next_bytes == transfer.size_bytes and next_parts != transfer.part_count:
                    error = "Upload ended before all parts arrived."
                elif next_parts == transfer.part_count and next_bytes != transfer.size_bytes:
                    error = "Upload parts do not match the declared size."
                else:
                    try:
                        self._original_part(client_id, message)
                    except Exception:
                        self._drop(message.transfer_uuid)
                        raise
                    transfer.seen.add(index)
                    transfer.received_bytes = next_bytes
                    transfer.deadline = self._clock() + self._timeout
                    if next_bytes == transfer.size_bytes:
                        self._transfers.pop(message.transfer_uuid, None)
                if error is not None:
                    self._drop(message.transfer_uuid)
        for expired_id, expired_transfer in expired:
            self._reject(expired_transfer.client_id, expired_transfer.component_id,
                         expired_id, "Upload timed out. Please try again.")
        if error is not None and component_id is not None:
            self._reject(client_id, component_id, message.transfer_uuid, error)

    def _on_disconnect(self, client: object) -> None:
        client_id = client.client_id
        with self._lock:
            for transfer_id, transfer in tuple(self._transfers.items()):
                if transfer.client_id == client_id:
                    self._drop(transfer_id)

    def close(self) -> None:
        """Release pending buffers and restore Viser's original callbacks."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for transfer_id in tuple(self._transfers):
                self._drop(transfer_id)
            self._interface.unregister_handler(
                _messages.FileTransferStartUpload, self._handle_start
            )
            self._interface.unregister_handler(
                _messages.FileTransferPart, self._handle_part
            )
            self._interface.register_handler(
                _messages.FileTransferStartUpload, self._original_start
            )
            self._interface.register_handler(_messages.FileTransferPart, self._original_part)
            self._server._client_disconnect_cb.remove(self._on_disconnect)
