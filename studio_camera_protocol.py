"""Project cameras and per-viewer hard-cut preview over the existing socket."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from viser import _messages, transforms

from camera_model import camera_at_frame


@dataclass
class CameraStudioStateMessage(_messages.Message):
    project_id: str
    revision: int
    take_id: str | None
    frame: int
    length: int
    fps: float
    cameras: list[dict]
    cuts: list[dict]
    mode: Literal['free', 'fixed', 'sequence']
    active_camera_id: str | None
    selected_camera_id: str | None
    busy: bool
    error: str | None


@dataclass
class CameraStudioCommandMessage(_messages.Message):
    project_id: str
    revision: int
    take_id: str | None
    action: Literal['add', 'select', 'update', 'duplicate', 'delete', 'view', 'free',
                    'sequence', 'cut_add', 'cut_update', 'cut_delete', 'cuts_clear']
    camera_id: str | None = None
    cut_id: str | None = None
    frame: int | None = None
    camera: dict | None = None

    def redundancy_key(self) -> str:
        # Commands are operations, so none may be coalesced in a socket batch.
        return str(id(self))


@dataclass
class _View:
    project_id: str
    mode: str = 'free'
    active: str | None = None
    selected: str | None = None
    editor: dict | None = None
    applied: tuple | None = None
    published: dict | None = None
    error: str | None = None


class CameraStudioController:
    """Session lock serializes commands, playback, and viewer state publication."""

    def __init__(self, server, session, camera):
        self.server, self.session, self.camera = server, session, camera
        self.views: dict[int, _View] = {}
        camera.studio = self
        server._websock_server.register_handler(CameraStudioCommandMessage, self._command)
        server.on_client_disconnect(self._disconnected)

    def is_locked(self, client_id: int) -> bool:
        view = self.views.get(client_id)
        return view is not None and view.mode != 'free'

    def _view(self, client) -> _View:
        view = self.views.get(client.client_id)
        if view is None:
            view = self.views[client.client_id] = _View(self.session.camera_project_id)
        if view.project_id != self.session.camera_project_id:
            self._free(client, view)
            view.project_id = self.session.camera_project_id
            view.selected = None
            view.error = None
            view.published = None
        return view

    def _disconnected(self, client):
        with self.session.lock:
            self.views.pop(client.client_id, None)

    def _take(self):
        return (self.session.takes.get(self.session.active_take)
                if self.session.kind == 'generated' else None)

    @staticmethod
    def _snapshot(client):
        camera = client.camera
        return dict(position=list(camera.position), look_at=list(camera.look_at),
                    up_direction=list(camera.up_direction), fov=float(camera.fov))

    def _free(self, client, view):
        if view.mode != 'free' and view.editor is not None:
            with client.atomic():
                client.camera.position = tuple(view.editor['position'])
                client.camera.look_at = tuple(view.editor['look_at'])
                client.camera.up_direction = tuple(view.editor['up_direction'])
                client.camera.fov = view.editor['fov']
        view.mode, view.active, view.applied, view.editor = 'free', None, None, None
        # Follow resumes from the current root rather than accumulating locked motion.
        self.camera.rebase()

    def enter_free(self, client):
        """Legacy manual camera actions explicitly leave the locked preview."""
        with self.session.lock:
            view = self._view(client)
            self._free(client, view)
            view.error = None
            self._publish(client, view)

    def _lock_view(self, client, view, mode, identifier=None):
        if view.mode == 'free':
            view.editor = self._snapshot(client)
        view.mode, view.active, view.applied = mode, identifier, None

    def _camera_record(self, identifier):
        record = next((item for item in self.session.cameras if item['id'] == identifier), None)
        if record is None:
            raise ValueError('Select an existing camera')
        return record

    def _command(self, client_id, message):
        client = self.server.get_clients().get(client_id)
        if client is None:
            return
        with self.session.lock:
            view = self._view(client)
            take = self._take()
            try:
                if (message.project_id != self.session.camera_project_id
                        or type(message.revision) is not int
                        or message.revision != self.session.project_revision
                        or message.take_id != (take.id if take else None)):
                    raise ValueError('Project changed. Please try again with the current camera list.')
                action = message.action
                view.error = None
                if action == 'add':
                    record = self.session.add_camera(list(client.camera.position),
                                                     list(client.camera.wxyz), float(client.camera.fov))
                    view.selected = record['id']
                elif action in ('select', 'update', 'duplicate', 'delete', 'view'):
                    record = self._camera_record(message.camera_id)
                    if action == 'select':
                        view.selected = record['id']
                    elif action == 'update':
                        if not isinstance(message.camera, dict) or not message.camera:
                            raise ValueError('No camera changes supplied')
                        if set(message.camera) - {'name', 'position', 'wxyz', 'fov'}:
                            raise ValueError('Unsupported camera field')
                        self.session.update_camera(record['id'], **message.camera)
                    elif action == 'duplicate':
                        view.selected = self.session.duplicate_camera(record['id'])['id']
                    elif action == 'delete':
                        self.session.remove_camera(record['id'])
                        if view.selected == record['id']:
                            view.selected = None
                    else:
                        self._lock_view(client, view, 'fixed', record['id'])
                elif action == 'free':
                    self._free(client, view)
                elif action == 'sequence':
                    if take is None or not take.camera_cuts:
                        raise ValueError('Add a camera cut to this take first')
                    self._lock_view(client, view, 'sequence')
                elif action in ('cut_add', 'cut_update', 'cut_delete', 'cuts_clear'):
                    if take is None:
                        raise ValueError('Select a recorded take to edit camera cuts')
                    if self.session.busy:
                        raise ValueError('Wait for motion generation before editing camera cuts')
                    if action == 'cut_add':
                        self.session.add_camera_cut(take.id, message.frame, message.camera_id)
                    elif action == 'cut_update':
                        self.session.update_camera_cut(take.id, message.cut_id,
                                                       frame=message.frame, camera_id=message.camera_id)
                    elif action == 'cut_delete':
                        self.session.remove_camera_cut(take.id, message.cut_id)
                    else:
                        self.session.clear_camera_cuts(take.id)
                else:
                    raise ValueError('Unsupported camera command')
            except (ValueError, TypeError, KeyError, AssertionError) as exc:
                view.error = str(exc) or 'Camera is not ready. Please try again.'
            self.update()

    def _apply(self, client, view):
        take = self._take()
        if view.mode == 'sequence':
            if take is None or not take.camera_cuts:
                self._free(client, view)
                return
            view.active = camera_at_frame(take.camera_cuts, self.session.frame)
        if view.mode == 'free':
            return
        record = next((item for item in self.session.cameras if item['id'] == view.active), None)
        if record is None:
            self._free(client, view)
            return
        signature = (record['id'], tuple(record['position']), tuple(record['wxyz']), record['fov'])
        # Other studio features may frame the scene or newly imported actor. A
        # fixed preview remains authoritative, but ordinary playback sends no
        # repeated camera writes when nothing has changed.
        q = np.asarray(record['wxyz'])
        actual_q = np.asarray(client.camera.wxyz)
        matches = (np.allclose(client.camera.position, record['position'], atol=1e-5)
                   and (np.allclose(actual_q, q, atol=1e-5) or np.allclose(actual_q, -q, atol=1e-5))
                   and abs(client.camera.fov - record['fov']) < 1e-6)
        if signature == view.applied and matches:
            return
        rotation = transforms.SO3(q).as_matrix()
        position = np.asarray(record['position'])
        with client.atomic():
            client.camera.position = position
            client.camera.look_at = position + rotation[:, 2] * 5.0
            client.camera.up_direction = -rotation[:, 1]
            client.camera.fov = record['fov']
        view.applied = signature

    def _publish(self, client, view):
        take = self._take()
        values = dict(project_id=self.session.camera_project_id, revision=self.session.project_revision,
                      take_id=take.id if take else None, frame=int(self.session.frame),
                      length=len(take.positions) if take else 0, fps=25.0,
                      cameras=[dict(item) for item in self.session.cameras],
                      cuts=[dict(item) for item in take.camera_cuts] if take else [],
                      mode=view.mode, active_camera_id=view.active,
                      selected_camera_id=view.selected, busy=bool(self.session.busy), error=view.error)
        if values != view.published:
            client._websock_connection.queue_message(CameraStudioStateMessage(**values))
            view.published = values

    def update(self):
        """Evaluate the current frame, including backward seeks, loops and edits."""
        with self.session.lock:
            ids = {item['id'] for item in self.session.cameras}
            for client in tuple(self.server.get_clients().values()):
                try:
                    view = self._view(client)
                    if view.selected not in ids:
                        view.selected = None
                    self._apply(client, view)
                    self._publish(client, view)
                except AssertionError:
                    # Viser camera getters are not ready until the browser's
                    # first camera packet. The next render iteration retries.
                    continue
