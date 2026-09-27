"""Send prepared native affine motion once; browsers own the playback clock.

This is the same weighted segment-affine formula as NativeRigAsset.skin, not
quaternion skinning or a retargeted skeleton. Binary arrays are row-major,
little-endian float32 (bone indices are uint16).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from threading import Condition, RLock, Timer
import time
from typing import Callable, Literal

import numpy as np
from viser import _messages

from experiments.native_pair_rig import NativeRigActor

MAX_ACTORS = 8
MAX_FRAMES = 1000
MAX_PARTS = 64
MAX_VERTICES = 500_000
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


@dataclass
class NativePairClipMessage(_messages.Message):
    revision: int
    fps: int
    frames: int
    actors: list[dict]

    def redundancy_key(self) -> str:
        # Replacing a clip must also release old clip bytes from socket buffers.
        return 'native-pair-clip'


@dataclass
class NativePairTransportMessage(_messages.Message):
    revision: int
    sequence: int
    frame: int
    playing: bool
    enabled: bool
    capturing: bool

    def redundancy_key(self) -> str:
        return 'native-pair-transport'


@dataclass
class NativePairStatusMessage(_messages.Message):
    revision: int
    status: Literal['loaded', 'error']
    error: str | None = None

    def redundancy_key(self) -> str:
        return 'native-pair-status'


def _array(value, shape, label, dtype='<f4'):
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f'{label} has an invalid shape or dtype')
    if not np.isfinite(array).all():
        raise ValueError(f'{label} must be finite')
    if dtype == '<u2' and (np.any(array < 0) or np.any(array >= 22) or
                           np.any(array != np.floor(array))):
        raise ValueError(f'{label} must contain native bone indices in [0,21]')
    with np.errstate(over='ignore', invalid='ignore'):
        converted = np.asarray(array, dtype=dtype, order='C')
    if not np.isfinite(converted).all():
        raise ValueError(f'{label} exceeds float32 range')
    return converted


def serialize_native_actors(actors, *, frames):
    """Copy bounded prepared caches without changing actor history or geometry."""
    if type(frames) is not int or not 4 <= frames <= MAX_FRAMES:
        raise ValueError('Native local playback requires 4–1000 frames')
    if not isinstance(actors, (list, tuple)) or not 1 <= len(actors) <= MAX_ACTORS:
        raise ValueError('Native local playback requires 1–8 actors')
    outputs, names, identifiers = [], set(), set()
    total_vertices = payload_bytes = 0
    for identifier, actor in actors:
        if not isinstance(identifier, str) or not identifier or len(identifier) > 256 or identifier in identifiers:
            raise ValueError('Native actor identifiers must be unique bounded strings')
        identifiers.add(identifier)
        if not isinstance(actor, NativeRigActor) or actor._removed:
            raise ValueError('Native local playback requires live native rig actors')
        cache = actor._prepared
        if cache is None or len(cache) != frames or any(len(item) != 5 for item in cache):
            raise ValueError('Prepare the complete native clip before local playback')
        asset = actor.asset
        if not 1 <= len(asset.parts) <= MAX_PARTS or len(actor.handles) != len(asset.parts):
            raise ValueError('Native rig has an invalid part count')
        rest = _array(asset.rest, (22, 3), 'rest')
        linear = _array(np.stack([item[0] for item in cache]), (frames, 22, 3, 3), 'linear')
        targets = _array(np.stack([item[1] for item in cache]), (frames, 22, 3), 'targets')
        finger_weights = np.zeros(2) if cache[0][3] is None else np.asarray(cache[0][3])
        _array(finger_weights, (2,), 'finger weights')
        if np.any((finger_weights < 0) | (finger_weights > 1)):
            raise ValueError('Finger weights must be in [0,1]')
        hand_pose = cache[0][4]
        for item in cache:
            current = np.zeros(2) if item[3] is None else np.asarray(item[3])
            if item[4] != hand_pose or not np.array_equal(current, finger_weights):
                raise ValueError('Variable finger weights or hand poses are not supported by native local playback')
        finger = asset.finger_warps(finger_weights, hand_pose) if np.any(finger_weights) else None
        result = {'rest': rest.tobytes(), 'linear': linear.tobytes(),
                  'targets': targets.tobytes(), 'parts': []}
        payload_bytes += rest.nbytes + linear.nbytes + targets.nbytes
        for index, (part, handle) in enumerate(zip(asset.parts, actor.handles)):
            name = getattr(handle, 'name', f'{identifier}/part-{index}')
            if not isinstance(name, str) or not name or len(name) > 512 or name in names:
                raise ValueError('Native mesh names must be unique bounded strings')
            names.add(name)
            vertices = len(part['weights'])
            total_vertices += vertices
            if not 1 <= vertices <= MAX_VERTICES or total_vertices > MAX_VERTICES:
                raise ValueError('Native local playback exceeds its vertex limit')
            payload_bytes += vertices * (4 * 3 * 4 + 4 * 2 + 4 * 4)
            if payload_bytes > MAX_PAYLOAD_BYTES:
                raise ValueError('Native local playback exceeds its payload byte limit')
            bones = _array(part['bones'], (vertices, 4), 'bones', '<u2')
            weights = _array(part['weights'], (vertices, 4), 'weights')
            if np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1, atol=1e-5, rtol=0):
                raise ValueError('Native skin weights must be nonnegative and normalized')
            # Warp in float64 before packing, just as asset.skin does. Applying
            # a finger rotation to already body-deformed vertices is incorrect.
            bind_world = np.asarray(part['bind_world'])
            _array(bind_world, (vertices, 4, 3), 'bind_world')
            if finger is not None:
                nodes = np.asarray(part['skin_nodes'])
                if (nodes.shape != (vertices, 4) or not np.issubdtype(nodes.dtype, np.integer)
                        or np.any(nodes < 0) or np.any(nodes >= len(finger))):
                    raise ValueError('Native finger skin nodes are invalid')
                warps = finger[nodes]
                bind_world = np.einsum('vwij,vwj->vwi', warps[:, :, :3, :3], bind_world) + warps[:, :, :3, 3]
            bind_world = _array(bind_world, (vertices, 4, 3), 'bind_world')
            result['parts'].append({'name': name, 'bind_world': bind_world.tobytes(),
                                    'bones': bones.tobytes(), 'weights': weights.tobytes()})
        outputs.append(result)
    return outputs, payload_bytes


class NativePairPlaybackController:
    """Publish clips and explicit transport changes to current and future tabs.

    get_state returns the current session snapshot, optionally including enabled.
    Call update after session changes; ordinary playback frame ticks are ignored.
    Session seeks/restarts must increment transport_revision. Capture frame changes
    are also explicit updates even if the session has no transport revision.
    """
    def __init__(self, server, *, get_state: Callable[[], dict]):
        if not callable(get_state):
            raise ValueError('Native local playback requires a state callback')
        self.server, self.get_state = server, get_state
        self._lock = RLock()
        self._ready = Condition(self._lock)
        self._revision = self._sequence = self._payload_bytes = 0
        self._clip = None
        self._enabled = False
        self._last_key = None
        self._client_status = {}
        self._sent_clients = set()
        self._readiness_timers = {}
        server._websock_server.register_handler(NativePairStatusMessage, self._handle_status)
        server.on_client_connect(self._client_connected)
        server.on_client_disconnect(self._client_disconnected)

    @property
    def revision(self):
        with self._lock:
            return self._revision

    @property
    def payload_bytes(self):
        with self._lock:
            return self._payload_bytes

    @property
    def client_status(self):
        with self._lock:
            return {key: dict(value) for key, value in self._client_status.items()}

    def is_ready(self, client):
        """Whether this connected tab acknowledged the currently loaded clip."""
        with self._lock:
            return (self._clip is not None and
                    self.server.get_clients().get(client.client_id) is client and
                    self._client_status.get(client.client_id, {}).get('revision') == self._revision and
                    self._client_status.get(client.client_id, {}).get('status') == 'loaded')

    def require_ready(self, client, *, timeout=5.0):
        """Wait at most five seconds for this tab before allowing local capture.

        Returns the acknowledged clip revision. A replacement during the wait
        aborts capture so an export cannot silently switch to a different take.
        """
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 <= timeout <= 5:
            raise ValueError('Native playback readiness timeout must be in [0,5] seconds')
        deadline = time.monotonic() + timeout
        with self._ready:
            revision = self._revision
            while True:
                if self._clip is None or self._revision != revision:
                    raise RuntimeError('The local playback clip changed or was removed. Reload this tab and retry export.')
                if (self.server.get_clients().get(client.client_id) is not client or
                        client.client_id not in self._sent_clients):
                    raise RuntimeError('This playback tab disconnected. Reload this tab and retry export.')
                status = self._client_status.get(client.client_id, {})
                if status.get('revision') == revision:
                    if status.get('status') == 'loaded':
                        return revision
                    if status.get('status') == 'error':
                        raise RuntimeError('Local playback could not load in this tab. Reload this tab and retry export.')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError('This tab has not loaded local playback. Reload this tab to update the viewer, then retry export.')
                self._ready.wait(remaining)

    def _cancel_readiness_timer(self, client_id):
        timer = self._readiness_timers.pop(client_id, None)
        if timer is not None:
            timer.cancel()

    def _schedule_readiness_notice(self, client):
        self._cancel_readiness_timer(client.client_id)
        if not callable(getattr(client, 'add_notification', None)):
            return
        timer = Timer(5., self._readiness_notice, args=(client, self._revision))
        timer.daemon = True
        self._readiness_timers[client.client_id] = timer
        timer.start()

    def _readiness_notice(self, client, revision):
        with self._lock:
            if (self._clip is None or self._revision != revision or
                    self.server.get_clients().get(client.client_id) is not client):
                return
            self._readiness_timers.pop(client.client_id, None)
            try:
                self.require_ready(client, timeout=0)
            except RuntimeError as exc:
                client.add_notification(title='Reload for local playback', body=str(exc),
                                        color='red', auto_close=False)

    def load(self, actors, fps=30, frames=None):
        if type(fps) not in (int, float) or fps != 30:
            raise ValueError('Native local playback requires 30 fps')
        if frames is None and actors:
            prepared = actors[0][1]._prepared
            frames = None if prepared is None else len(prepared)
        outputs, size = serialize_native_actors(actors, frames=frames)
        with self._lock:
            self._revision += 1
            self._clip = NativePairClipMessage(self._revision, 30, frames, outputs)
            self._payload_bytes = size
            self._last_key = None
            self._client_status.clear()
            for client_id in tuple(self._readiness_timers):
                self._cancel_readiness_timer(client_id)
            clients = self.server.get_clients()
            self._sent_clients = set(clients)
            for client in clients.values():
                client._websock_connection.queue_message(self._clip)
                self._schedule_readiness_notice(client)
            self._ready.notify_all()
            return self._revision

    def _transport(self, state, enabled):
        frame = state.get('frame', 0)
        if type(frame) is not int or not 0 <= frame < self._clip.frames:
            raise ValueError('Native transport frame is outside the loaded clip')
        playing, capturing = bool(state.get('playing', False)), bool(state.get('capturing', False))
        self._sequence += 1
        return NativePairTransportMessage(self._revision, self._sequence, frame,
                                          playing, bool(enabled), capturing)

    def update(self, state, enabled):
        with self._lock:
            self._enabled = bool(enabled)
            if self._clip is None:
                return False
            key = (state.get('transport_revision'), bool(state.get('playing', False)),
                   self._enabled, bool(state.get('capturing', False)),
                   state.get('frame') if state.get('capturing', False) else None)
            if key == self._last_key:
                return False
            message = self._transport(state, self._enabled)
            self._last_key = key
            for client in self.server.get_clients().values():
                client._websock_connection.queue_message(message)
            return True

    def clear(self):
        """Stop clients and release the retained clip before actor removal."""
        with self._lock:
            if self._clip is not None:
                message = self._transport({'frame': 0, 'playing': False, 'capturing': False}, False)
                for client in self.server.get_clients().values():
                    client._websock_connection.queue_message(message)
            self._clip = None
            self._payload_bytes = 0
            self._enabled = False
            self._last_key = None
            self._client_status.clear()
            self._sent_clients.clear()
            for client_id in tuple(self._readiness_timers):
                self._cancel_readiness_timer(client_id)
            self._ready.notify_all()

    def _client_connected(self, client):
        while True:
            with self._lock:
                version = (self._revision, self._sequence)
            # Never acquire the session's lock under ours. Retry if a transport
            # changed during this read, rather than giving an old state a newer
            # sequence and undoing a concurrent pause, seek or clip replacement.
            state = self.get_state()
            with self._lock:
                if version != (self._revision, self._sequence):
                    continue
                self._client_status.pop(client.client_id, None)
                if self._clip is None:
                    return
                self._sent_clients.add(client.client_id)
                client._websock_connection.queue_message(self._clip)
                client._websock_connection.queue_message(
                    self._transport(state, state.get('enabled', self._enabled)))
                self._schedule_readiness_notice(client)
                self._ready.notify_all()
                return

    def _client_disconnected(self, client):
        with self._lock:
            self._sent_clients.discard(client.client_id)
            self._client_status.pop(client.client_id, None)
            self._cancel_readiness_timer(client.client_id)
            self._ready.notify_all()

    def _handle_status(self, client_id, message):
        with self._lock:
            if (self._clip is None or type(message.revision) is not int or
                    message.revision != self._revision or client_id not in self._sent_clients or
                    message.status not in ('loaded', 'error')):
                return
            self._client_status[client_id] = {'revision': message.revision, 'status': message.status,
                'error': message.error[:500] if isinstance(message.error, str) else None}
            if message.status == 'loaded':
                self._cancel_readiness_timer(client_id)
            self._ready.notify_all()
