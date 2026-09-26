"""Material-preserving GLB actor transport for the StageZero Studio client.

The original GLB is parsed by Three.js in each tab. Only glTF node-local
matrices cross the socket after load, so Three.js retains the file's materials,
textures, skins, and inverse-bind matrices. Ordinary Viser GLB nodes are not
affected by this actor-specific protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
import time
from typing import Callable, Literal, Mapping

import numpy as np
import viser
from viser import _messages


Status = Literal["loaded", "error"]
ResultCallback = Callable[[int, str, int, Status, str | None], None]


@dataclass
class ActorGlbLoadMessage(_messages.Message):
    """Create or replace the single client-local actor scene node."""

    name: str
    asset_id: str
    revision: int
    glb_data: bytes
    scale: float
    fallback_name: str
    required_nodes: tuple[int, ...]
    ground_offset: float = 0.0
    cast_shadow: bool = True
    receive_shadow: bool = True
    props: dict[str, object] = field(default_factory=dict)

    def redundancy_key(self) -> str:
        return f"actor-glb-load-{self.name}-{self.revision}"


@dataclass
class ActorGlbPoseMessage(_messages.Message):
    """Column-major float32 local matrices keyed by glTF node index."""

    revision: int
    node_indices: np.ndarray
    local_matrices: np.ndarray

    def redundancy_key(self) -> str:
        return f"actor-glb-pose-{self.revision}"


@dataclass
class ActorGlbCommandMessage(_messages.Message):
    """Commit a loaded candidate, discard it, or restore G1."""

    revision: int
    action: Literal["commit", "reject", "g1"]

    def redundancy_key(self) -> str:
        return f"actor-glb-command-{self.revision}"


@dataclass
class ActorGlbStatusMessage(_messages.Message):
    """Browser load result. The websocket supplies the trusted client ID."""

    asset_id: str
    revision: int
    status: Status
    error: str | None = None

    def redundancy_key(self) -> str:
        return f"actor-glb-status-{self.revision}"


@dataclass
class _AssetState:
    asset_id: str
    revision: int
    glb_data: bytes
    scale: float
    required_nodes: tuple[int, ...]
    ground_offset: float
    initiator_id: int | None = None
    deadline: float = 0.0
    client_status: dict[int, Status] = field(default_factory=dict)
    sent_clients: set[int] = field(default_factory=set)
    pose: ActorGlbPoseMessage | None = None


class GlbCharacterRenderer:
    """Own the actor load/commit/pose protocol for connected Studio tabs.

    `load()` creates a pending candidate and returns its revision. The caller
    supplies its latest frame with `set_pose(..., revision=revision)`. Once the
    initiating client's `loaded` callback arrives, call `commit(revision)`.
    On `error`, call `reject(revision)`; the existing actor remains visible.
    `poll()` reports timeouts even if there is no incoming websocket traffic.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        *,
        fallback_name: str = "/actor/g1_mesh",
        on_result: ResultCallback | None = None,
        timeout_seconds: float = 15.0,
        actor_name: str = "/actor/glb_actor",
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.server = server
        self.fallback_name = fallback_name
        self.actor_name = actor_name
        self.on_result = on_result
        self.timeout_seconds = timeout_seconds
        self._lock = RLock()
        self._revision = 0
        self._active: _AssetState | None = None
        self._pending: _AssetState | None = None

        # This app-owned message type extends Viser's message base. Its server
        # and per-client queues are supported infra APIs; no dependency files
        # or runtime classes are modified.
        server._websock_server.register_handler(
            ActorGlbStatusMessage, self._handle_status
        )
        server.on_client_connect(self._client_connected)
        server.on_client_disconnect(self._client_disconnected)

    @property
    def active_revision(self) -> int | None:
        with self._lock:
            return None if self._active is None else self._active.revision

    @property
    def state(self) -> Literal["g1", "loading", "glb"]:
        with self._lock:
            if self._pending is not None:
                return "loading"
            return "glb" if self._active is not None else "g1"

    @property
    def active_asset_id(self) -> str | None:
        with self._lock:
            return None if self._active is None else self._active.asset_id

    def _send(self, client: viser.ClientHandle, message: _messages.Message) -> None:
        # ClientHandle exposes its connection privately in pinned Viser 1.0.16;
        # queue_message itself is the supported infra transport operation.
        client._websock_connection.queue_message(message)

    def _load_message(self, state: _AssetState) -> ActorGlbLoadMessage:
        return ActorGlbLoadMessage(
            name=self.actor_name,
            asset_id=state.asset_id,
            revision=state.revision,
            glb_data=state.glb_data,
            scale=state.scale,
            fallback_name=self.fallback_name,
            required_nodes=state.required_nodes,
            ground_offset=state.ground_offset,
        )

    def _send_state(self, client: viser.ClientHandle, state: _AssetState, *, committed: bool) -> None:
        self._send(client, self._load_message(state))
        state.sent_clients.add(client.client_id)
        if state.pose is not None:
            self._send(client, state.pose)
        if committed:
            self._send(client, ActorGlbCommandMessage(state.revision, "commit"))

    def _client_connected(self, client: viser.ClientHandle) -> None:
        with self._lock:
            if self._active is not None:
                self._send_state(client, self._active, committed=True)
            elif self._pending is not None:
                self._send_state(client, self._pending, committed=False)

    def _client_disconnected(self, client: viser.ClientHandle) -> None:
        failure: tuple[int, str, int, Status, str] | None = None
        with self._lock:
            if self._active is not None:
                self._active.sent_clients.discard(client.client_id)
            pending = self._pending
            if pending is not None:
                pending.sent_clients.discard(client.client_id)
            if pending is not None and pending.initiator_id == client.client_id:
                failure = (
                    client.client_id,
                    pending.asset_id,
                    pending.revision,
                    "error",
                    "Initiating tab disconnected during GLB load",
                )
                self._reject_locked(pending.revision)
        if failure is not None:
            self._notify(*failure)

    def _notify(self, client_id: int, asset_id: str, revision: int, status: Status, error: str | None) -> None:
        if self.on_result is not None:
            self.on_result(client_id, asset_id, revision, status, error)

    def _handle_status(self, client_id: int, message: ActorGlbStatusMessage) -> None:
        result: tuple[int, str, int, Status, str | None] | None = None
        with self._lock:
            state = next(
                (
                    item for item in (self._pending, self._active)
                    if item is not None
                    and item.revision == message.revision
                    and item.asset_id == message.asset_id
                ),
                None,
            )
            if state is None or client_id in state.client_status:
                return
            if message.status not in ("loaded", "error"):
                return
            state.client_status[client_id] = message.status
            result = (client_id, state.asset_id, state.revision, message.status, message.error)
        # The caller may commit/reject in this callback; never hold our lock.
        if result is not None:
            self._notify(*result)

    def load(
        self,
        asset_id: str,
        glb_data: bytes,
        initiating_client_id: int,
        *,
        scale: float = 1.0,
        ground_offset: float = 0.0,
        required_nodes: tuple[int, ...] = (),
        initial_node_matrices: np.ndarray | Mapping[int, np.ndarray] | None = None,
    ) -> int:
        if not asset_id:
            raise ValueError("asset_id is required")
        if not glb_data:
            raise ValueError("glb_data is empty")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        if not np.isfinite(ground_offset):
            raise ValueError("ground_offset must be finite")
        if any(index < 0 for index in required_nodes):
            raise ValueError("required_nodes must contain nonnegative glTF indices")
        with self._lock:
            clients = self.server.get_clients()
            if initiating_client_id not in clients:
                raise ValueError(f"initiating client {initiating_client_id} is not connected")
            if self._pending is not None:
                self._reject_locked(self._pending.revision)
            self._revision += 1
            state = _AssetState(
                asset_id=asset_id,
                revision=self._revision,
                glb_data=bytes(glb_data),
                scale=float(scale),
                required_nodes=tuple(required_nodes),
                ground_offset=float(ground_offset),
                initiator_id=initiating_client_id,
                deadline=time.monotonic() + self.timeout_seconds,
            )
            if initial_node_matrices is not None:
                state.pose = self._pose_message(state.revision, initial_node_matrices)
            self._pending = state
            for client in clients.values():
                self._send_state(client, state, committed=False)
            return state.revision

    def commit(self, revision: int) -> bool:
        with self._lock:
            state = self._pending
            if state is None or state.revision != revision:
                return False
            if state.client_status.get(state.initiator_id) != "loaded":
                return False
            self._active = state
            self._pending = None
            for client in self.server.get_clients().values():
                if client.client_id not in state.sent_clients:
                    self._send_state(client, state, committed=False)
                self._send(client, ActorGlbCommandMessage(revision, "commit"))
            return True

    def _reject_locked(self, revision: int) -> bool:
        state = self._pending
        if state is None or state.revision != revision:
            return False
        self._pending = None
        for client in self.server.get_clients().values():
            self._send(client, ActorGlbCommandMessage(revision, "reject"))
        return True

    def reject(self, revision: int) -> bool:
        with self._lock:
            return self._reject_locked(revision)

    def poll(self) -> None:
        failure: tuple[int, str, int, Status, str] | None = None
        with self._lock:
            state = self._pending
            if state is not None and time.monotonic() >= state.deadline:
                failure = (
                    state.initiator_id,
                    state.asset_id,
                    state.revision,
                    "error",
                    "GLB load timed out",
                )
                self._reject_locked(state.revision)
        if failure is not None:
            self._notify(*failure)

    @staticmethod
    def _pose_message(
        revision: int, matrices: np.ndarray | Mapping[int, np.ndarray]
    ) -> ActorGlbPoseMessage:
        if isinstance(matrices, Mapping):
            indices = np.fromiter(sorted(matrices), dtype=np.uint32)
            rows = np.stack([np.asarray(matrices[int(index)], dtype=np.float32) for index in indices]) if len(indices) else np.empty((0, 4, 4), dtype=np.float32)
        else:
            rows = np.asarray(matrices, dtype=np.float32)
            indices = np.arange(len(rows), dtype=np.uint32)
        if rows.ndim != 3 or rows.shape[1:] != (4, 4):
            raise ValueError("node matrices must have shape (N,4,4)")
        if not np.isfinite(rows).all():
            raise ValueError("node matrices must be finite")
        if len(indices) and rows.shape[0] != len(indices):
            raise ValueError("node index and matrix counts differ")
        # Three's Matrix4.fromArray consumes column-major elements. The
        # retargeter returns conventional row-major mathematical matrices.
        columns = np.ascontiguousarray(rows.transpose(0, 2, 1).reshape(-1))
        return ActorGlbPoseMessage(revision, indices, columns)

    def set_pose(
        self,
        node_matrices: np.ndarray | Mapping[int, np.ndarray],
        *,
        revision: int | None = None,
    ) -> bool:
        with self._lock:
            state = (
                self._active if revision is None else next(
                    (item for item in (self._pending, self._active) if item is not None and item.revision == revision),
                    None,
                )
            )
            if state is None:
                return False
            message = self._pose_message(state.revision, node_matrices)
            state.pose = message
            for client in self.server.get_clients().values():
                self._send(client, message)
            return True

    def restore_g1(self) -> int:
        with self._lock:
            self._revision += 1
            self._pending = None
            self._active = None
            for client in self.server.get_clients().values():
                self._send(client, ActorGlbCommandMessage(self._revision, "g1"))
            return self._revision
