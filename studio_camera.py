"""Camera controls for the StageZero Viser studio.

Viser's camera position setter translates ``look_at`` by the same amount.  This
module uses that behavior for pans and following, then sets an explicit target
for changes of viewpoint.  No camera writes occur during ordinary playback
unless the user has enabled follow.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock

import numpy as np
import viser


_UP = np.array((0.0, 1.0, 0.0), dtype=np.float64)
_MIN_DISTANCE = 0.35
_MAX_DISTANCE = 500.0


def _vector(value) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("camera coordinates must be finite 3D vectors")
    return vector


def _ground_root(value) -> np.ndarray:
    root = _vector(value).copy()
    root[1] = 0.0
    return root


def _camera_pair(camera) -> tuple[np.ndarray, np.ndarray]:
    return _vector(camera.position), _vector(camera.look_at)


def _set_view(camera, position, target) -> None:
    """Write both endpoints, accounting for Viser's coupled position setter."""
    camera.up_direction = _UP
    camera.position = _vector(position)
    camera.look_at = _vector(target)


def _orbit_position(position, target, yaw_degrees: float = 0.0, pitch_degrees: float = 0.0) -> np.ndarray:
    offset = position - target
    radius = float(np.linalg.norm(offset))
    if radius < _MIN_DISTANCE:
        offset = np.array((0.0, 1.5, 4.0))
        radius = float(np.linalg.norm(offset))
    yaw = np.arctan2(offset[2], offset[0]) + np.deg2rad(yaw_degrees)
    pitch = np.arcsin(np.clip(offset[1] / radius, -1.0, 1.0)) + np.deg2rad(pitch_degrees)
    pitch = np.clip(pitch, np.deg2rad(-82.0), np.deg2rad(82.0))
    horizontal = radius * np.cos(pitch)
    return target + np.array((horizontal * np.cos(yaw), radius * np.sin(pitch), horizontal * np.sin(yaw)))


def _pan_offset(position, target, horizontal: float, vertical: float) -> np.ndarray:
    forward = target - position
    distance = float(np.linalg.norm(forward))
    if distance < _MIN_DISTANCE:
        return np.zeros(3)
    forward /= distance
    right = np.cross(forward, _UP)
    right_length = float(np.linalg.norm(right))
    if right_length < 1e-6:
        right = np.array((1.0, 0.0, 0.0))
    else:
        right /= right_length
    up = np.cross(right, forward)
    return (right * horizontal + up * vertical) * max(distance * 0.09, 0.05)


class StudioCamera:
    """Create shared controls that act on the client who clicked them.

    ``root_getter`` returns the current actor root in world coordinates. The
    ``update`` method receives that same root from the render loop after a pose
    change. ``rebase`` should be called after intentional playhead jumps to keep
    follow from sweeping the camera through the scene.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        root_getter: Callable[[], object],
        *,
        default_target=(0.0, 0.9, 0.0),
        default_position=(4.0, 2.25, 4.2),
    ) -> None:
        self.server = server
        self.root_getter = root_getter
        self.default_target = _vector(default_target)
        self.default_position = _vector(default_position)
        self._last_root: dict[int, np.ndarray] = {}
        self._lock = RLock()
        self.follow = False
        self.follow_handle = None
        self.studio = None

    def _manual(self, client) -> None:
        if self.studio is not None:
            self.studio.enter_free(client)

    def _root(self) -> np.ndarray:
        root = self.root_getter()
        return np.zeros(3) if root is None else _ground_root(root)

    def _clients(self):
        return tuple(self.server.get_clients().values())

    def reset(self, client: viser.ClientHandle) -> None:
        self._manual(client)
        root = self._root()
        _set_view(client.camera, self.default_position + root, self.default_target + root)
        client.camera.fov = np.deg2rad(42.0)
        with self._lock:
            self._last_root[client.client_id] = root

    def focus(self, client: viser.ClientHandle) -> None:
        """Center the actor while retaining the current angle and distance."""
        self._manual(client)
        position, target = _camera_pair(client.camera)
        new_target = self.default_target + self._root()
        if np.linalg.norm(position - target) < _MIN_DISTANCE:
            position = target + (self.default_position - self.default_target)
        _set_view(client.camera, position + (new_target - target), new_target)

    def preset(self, client: viser.ClientHandle, name: str) -> None:
        self._manual(client)
        current_position, current_target = _camera_pair(client.camera)
        radius = float(np.linalg.norm(current_position - current_target))
        radius = float(np.clip(radius, 2.0, 30.0))
        target = self.default_target + self._root()
        # Top has a small forward component so its up vector stays well-defined.
        directions = {
            "Perspective": np.array((0.68, 0.40, 0.61)),
            "Front": np.array((0.0, 0.12, 1.0)),
            "Side": np.array((1.0, 0.12, 0.0)),
            "Top": np.array((0.0, 1.0, 0.08)),
        }
        direction = directions[name]
        direction /= np.linalg.norm(direction)
        _set_view(client.camera, target + radius * direction, target)

    def orbit(self, client: viser.ClientHandle, *, yaw: float = 0.0, pitch: float = 0.0) -> None:
        self._manual(client)
        position, target = _camera_pair(client.camera)
        _set_view(client.camera, _orbit_position(position, target, yaw, pitch), target)

    def pan(self, client: viser.ClientHandle, *, horizontal: float = 0.0, vertical: float = 0.0) -> None:
        self._manual(client)
        position, target = _camera_pair(client.camera)
        client.camera.position = position + _pan_offset(position, target, horizontal, vertical)

    def zoom(self, client: viser.ClientHandle, direction: int) -> None:
        self._manual(client)
        position, target = _camera_pair(client.camera)
        offset = position - target
        distance = float(np.linalg.norm(offset))
        if distance < 1e-8:
            return
        factor = 0.8 if direction > 0 else 1.25
        new_distance = float(np.clip(distance * factor, _MIN_DISTANCE, _MAX_DISTANCE))
        _set_view(client.camera, target + offset * (new_distance / distance), target)

    def set_follow(self, enabled: bool) -> None:
        """Toggle follow without moving any camera at the moment of enabling."""
        root = self._root()
        with self._lock:
            self.follow = enabled
            self._last_root = {client.client_id: root.copy() for client in self._clients()}
        if self.follow_handle is not None and self.follow_handle.value != enabled:
            self.follow_handle.value = enabled

    def rebase(self, root=None) -> None:
        """Keep each view still after a playhead jump or a changed motion clip."""
        current = self._root() if root is None else _ground_root(root)
        with self._lock:
            self._last_root = {client.client_id: current.copy() for client in self._clients()}

    def update(self, root) -> None:
        """Translate following cameras by actor displacement; leave others alone."""
        if root is None:
            return
        current = _ground_root(root)
        with self._lock:
            if not self.follow:
                return
            clients = self._clients()
            connected = {client.client_id for client in clients}
            for stale in self._last_root.keys() - connected:
                del self._last_root[stale]
            for client in clients:
                previous = self._last_root.get(client.client_id)
                self._last_root[client.client_id] = current.copy()
                if self.studio is not None and self.studio.is_locked(client.client_id):
                    continue
                if previous is None:
                    continue
                delta = current - previous
                if np.linalg.norm(delta) > 1e-7:
                    # Viser also translates look_at when position changes.
                    client.camera.position = _vector(client.camera.position) + delta

    def build_gui(self, gui: viser.GuiApi) -> None:
        """Add a compact Camera tab or folder inside the caller's GUI context."""
        if self.studio is not None:
            gui.add_html('<div data-stagezero-cameras></div>')
        gui.add_markdown("**Camera** · Drag or two-finger scroll to pan · Pinch to zoom · Choose Orbit or Look in the viewport · WASD to move, Q/E down/up")
        presets = gui.add_button_group("View", ("Perspective", "Front", "Side", "Top"))
        focus = gui.add_button("Focus actor", hint="Center the actor while keeping your current angle and distance")
        orbit = gui.add_button_group("Orbit", ("←", "↑", "↓", "→"))
        pan = gui.add_button_group("Pan", ("Left", "Up", "Down", "Right"))
        dolly = gui.add_button_group("Distance", ("Closer", "Farther"))
        reset = gui.add_button("Reset camera")
        self.follow_handle = gui.add_checkbox("Follow actor", initial_value=False, hint="Keep your chosen camera angle while the actor moves")

        @presets.on_click
        def _preset(event):
            if event.client is not None:
                self.preset(event.client, presets.value)

        @focus.on_click
        def _focus(event):
            if event.client is not None:
                self.focus(event.client)

        @orbit.on_click
        def _orbit(event):
            if event.client is not None:
                yaw, pitch = {"←": (-15, 0), "→": (15, 0), "↑": (0, 12), "↓": (0, -12)}[orbit.value]
                self.orbit(event.client, yaw=yaw, pitch=pitch)

        @pan.on_click
        def _pan(event):
            if event.client is not None:
                horizontal, vertical = {"Left": (-1, 0), "Right": (1, 0), "Up": (0, 1), "Down": (0, -1)}[pan.value]
                self.pan(event.client, horizontal=horizontal, vertical=vertical)

        @dolly.on_click
        def _dolly(event):
            if event.client is not None:
                self.zoom(event.client, 1 if dolly.value == "Closer" else -1)

        @reset.on_click
        def _reset(event):
            if event.client is not None:
                self.reset(event.client)

        @self.follow_handle.on_update
        def _follow(event):
            if event.client is not None:
                self.set_follow(self.follow_handle.value)
