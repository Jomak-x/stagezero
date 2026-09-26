"""Behavioral coverage for per-client camera preview and socket commands."""

import copy
import tempfile
import unittest

import numpy as np
import viser

from directing import DirectorSession
from studio_camera import StudioCamera
from studio_camera_protocol import (CameraStudioCommandMessage,
                                    CameraStudioController,
                                    CameraStudioStateMessage)
from takes import Take


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def _quaternion(matrix):
    """Recover the camera quaternion from its forward and up vectors."""
    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        result = np.array([scale / 4, (matrix[2, 1] - matrix[1, 2]) / scale,
                           (matrix[0, 2] - matrix[2, 0]) / scale,
                           (matrix[1, 0] - matrix[0, 1]) / scale])
    else:
        axis = max(range(3), key=lambda i: matrix[i, i])
        following = (axis + 1) % 3
        last = (axis + 2) % 3
        scale = np.sqrt(1.0 + matrix[axis, axis] - matrix[following, following]
                        - matrix[last, last]) * 2
        result = np.zeros(4)
        result[axis + 1] = scale / 4
        result[0] = (matrix[last, following] - matrix[following, last]) / scale
        result[following + 1] = (matrix[following, axis] + matrix[axis, following]) / scale
        result[last + 1] = (matrix[last, axis] + matrix[axis, last]) / scale
    return result / np.linalg.norm(result)


class FakeCamera:
    """The position setter preserves Viser's coupled target translation."""

    def __init__(self, position, target, up, fov):
        self._position = np.asarray(position, dtype=float)
        self._look_at = np.asarray(target, dtype=float)
        self._up_direction = np.asarray(up, dtype=float)
        self._fov = float(fov)
        self.writes = 0

    @property
    def position(self):
        return self._position.copy()

    @position.setter
    def position(self, value):
        value = np.asarray(value, dtype=float)
        self._look_at += value - self._position
        self._position = value.copy()
        self.writes += 1

    @property
    def look_at(self):
        return self._look_at.copy()

    @look_at.setter
    def look_at(self, value):
        self._look_at = np.asarray(value, dtype=float).copy()
        self.writes += 1

    @property
    def up_direction(self):
        return self._up_direction.copy()

    @up_direction.setter
    def up_direction(self, value):
        self._up_direction = np.asarray(value, dtype=float).copy()
        self.writes += 1

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, value):
        self._fov = float(value)
        self.writes += 1

    @property
    def wxyz(self):
        forward = _unit(self._look_at - self._position)
        right = _unit(np.cross(forward, self._up_direction))
        down = np.cross(forward, right)
        return _quaternion(np.array([right, down, forward]).T)


class FakeConnection:
    def __init__(self):
        self.messages = []

    def queue_message(self, message):
        self.messages.append(message)


class FakeAtomic:
    def __init__(self, client):
        self.client = client

    def __enter__(self):
        self.client.atomic_batches += 1

    def __exit__(self, *_):
        return False


class FakeClient:
    def __init__(self, client_id, position, target, up, fov):
        self.client_id = client_id
        self.camera = FakeCamera(position, target, up, fov)
        self._websock_connection = FakeConnection()
        self.atomic_batches = 0

    def atomic(self):
        return FakeAtomic(self)

    @property
    def state(self):
        return self._websock_connection.messages[-1]


def real_camera_client(client_id, position, target, up, fov):
    """Initialize Viser's real camera handle without a websocket or browser."""
    client = FakeClient(client_id, position, target, up, fov)
    client.camera = viser.CameraHandle(client)
    state = client.camera._state
    state.position = np.asarray(position, dtype=np.float64)
    state.look_at = np.asarray(target, dtype=np.float64)
    state.up_direction = np.asarray(up, dtype=np.float64)
    state.fov = float(fov)
    state.update_timestamp = 1.0
    client.camera._update_wxyz()
    return client


class FakeServer:
    def __init__(self, clients):
        self.clients = {client.client_id: client for client in clients}
        self._websock_server = self
        self.handler = None
        self.disconnect_handler = None

    def get_clients(self):
        return self.clients

    def register_handler(self, message_type, callback):
        assert message_type is CameraStudioCommandMessage
        self.handler = callback

    def on_client_disconnect(self, callback):
        self.disconnect_handler = callback

    def disconnect(self, client):
        self.clients.pop(client.client_id)
        self.disconnect_handler(client)


class FakeGuiHandle:
    def __init__(self, value=None):
        self.value = value

    def on_click(self, callback):
        return callback

    def on_update(self, callback):
        return callback


class FakeGui:
    def __init__(self):
        self.html = []

    def add_html(self, value):
        self.html.append(value)

    def add_markdown(self, *_):
        return FakeGuiHandle()

    def add_button_group(self, _label, options):
        return FakeGuiHandle(options[0])

    def add_button(self, *_args, **_kwargs):
        return FakeGuiHandle()

    def add_checkbox(self, _label, initial_value=False, **_kwargs):
        return FakeGuiHandle(initial_value)


def _take(take_id, frames=30):
    positions = np.zeros((frames, 34, 3), dtype=np.float32)
    rotations = np.tile(np.eye(3, dtype=np.float32), (frames, 34, 1, 1))
    motion = np.zeros((frames, 414), dtype=np.float32)
    return Take(take_id, take_id, positions, rotations, motion,
                segments=[dict(start=0, end=frames, prompt='synthetic')])


class CameraRuntimeTests(unittest.TestCase):
    def setUp(self):
        recorded = np.zeros((30, 34, 3), dtype=np.float32)
        rotations = np.tile(np.eye(3, dtype=np.float32), (30, 34, 1, 1))
        self.session = DirectorSession(None, recorded, rotations)
        self.session.set_mode('Live ARDY')
        self.take = _take('take-a')
        self.session.takes[self.take.id] = self.take
        self.session.select_take(self.take.id)
        self.root = np.zeros(3)
        self.first = FakeClient(1, [4, 2, -5], [0, 1, 0], [0, 1, 0], .72)
        self.second = FakeClient(2, [-3, 4, -7], [1, 0, 0], [0, 1, 0], 1.03)
        self.server = FakeServer([self.first, self.second])
        self.camera = StudioCamera(self.server, lambda: self.root)
        self.controller = CameraStudioController(self.server, self.session, self.camera)
        self.controller.update()

    def command(self, client, action, envelope=None, **payload):
        state = client.state if envelope is None else envelope
        message = CameraStudioCommandMessage(
            project_id=state.project_id, revision=state.revision,
            take_id=state.take_id, action=action, **payload)
        self.server.handler(client.client_id, message)
        return client.state

    def assert_pose(self, client, record):
        np.testing.assert_allclose(client.camera.position, record['position'], atol=1e-6)
        expected = np.asarray(record['wxyz'])
        actual = client.camera.wxyz
        self.assertTrue(np.allclose(actual, expected, atol=1e-6)
                        or np.allclose(actual, -expected, atol=1e-6))
        self.assertAlmostEqual(client.camera.fov, record['fov'])

    def test_two_clients_keep_independent_modes_selection_and_pose(self):
        self.assertIsInstance(self.first.state, CameraStudioStateMessage)
        original_second = self.second.camera.position
        state = self.command(self.first, 'add')
        camera_id = state.selected_camera_id
        self.assertEqual(len(state.cameras), 1)
        self.assertEqual(self.second.state.mode, 'free')
        self.assertIsNone(self.second.state.selected_camera_id)
        self.command(self.second, 'select', camera_id=camera_id)
        self.assertEqual(self.second.state.mode, 'free')
        np.testing.assert_allclose(self.second.camera.position, original_second)

        fixed = self.command(self.first, 'view', camera_id=camera_id)
        self.assertEqual((fixed.mode, fixed.active_camera_id), ('fixed', camera_id))
        self.assertEqual(self.second.state.mode, 'free')
        self.assert_pose(self.first, fixed.cameras[0])
        self.assertEqual(self.first.atomic_batches, 1)

        self.camera.set_follow(True)
        self.root = np.array([2.0, 0.0, -1.0])
        self.camera.update(self.root)
        self.assert_pose(self.first, fixed.cameras[0])
        np.testing.assert_allclose(self.second.camera.position, original_second + self.root)
        writes = self.first.camera.writes
        self.controller.update()
        self.assertEqual(self.first.camera.writes, writes)
        self.first.camera.position = self.first.camera.position + [9, 0, 0]
        self.controller.update()
        self.assert_pose(self.first, fixed.cameras[0])
        self.assertEqual(self.first.atomic_batches, 2)

    def test_sequence_hard_cuts_seek_backward_loop_and_definition_edit(self):
        first = self.session.add_camera([1, 2, 3], [1, 0, 0, 0], .8)
        second = self.session.add_camera([-4, 5, 6], [0, 0, 1, 0], 1.1)
        self.session.add_camera_cut(self.take.id, 0, first['id'])
        self.session.add_camera_cut(self.take.id, 10, second['id'])
        self.controller.update()
        self.command(self.first, 'sequence')
        self.assert_pose(self.first, first)
        initial_batches = self.first.atomic_batches

        for frame, record in ((9, first), (10, second), (29, second),
                              (4, first), (0, first), (10, second)):
            self.session.seek(frame)
            self.controller.update()
            self.assertEqual(self.first.state.frame, frame)
            self.assertEqual(self.first.state.active_camera_id, record['id'])
            self.assert_pose(self.first, record)
        self.assertGreater(self.first.atomic_batches, initial_batches)
        writes = self.first.camera.writes
        self.controller.update()
        self.assertEqual(self.first.camera.writes, writes)

        updated = self.session.update_camera(second['id'], position=[8, 4, 2], fov=.93)
        self.controller.update()
        self.assert_pose(self.first, updated)
        self.assertEqual(self.first.state.mode, 'sequence')
        self.assertEqual(self.second.state.mode, 'free')

    def test_free_restores_editor_position_target_up_and_fov_after_mode_switches(self):
        original = dict(position=self.first.camera.position, look_at=self.first.camera.look_at,
                        up_direction=self.first.camera.up_direction, fov=self.first.camera.fov)
        first = self.session.add_camera([1, 2, 3], [1, 0, 0, 0], .8)
        second = self.session.add_camera([-4, 5, 6], [0, 0, 1, 0], 1.1)
        self.session.add_camera_cut(self.take.id, 0, second['id'])
        self.controller.update()
        self.command(self.first, 'view', camera_id=first['id'])
        self.command(self.first, 'sequence')
        self.assert_pose(self.first, second)
        self.command(self.first, 'free')
        self.assertEqual(self.first.state.mode, 'free')
        np.testing.assert_allclose(self.first.camera.position, original['position'])
        np.testing.assert_allclose(self.first.camera.look_at, original['look_at'])
        np.testing.assert_allclose(self.first.camera.up_direction, original['up_direction'])
        self.assertAlmostEqual(self.first.camera.fov, original['fov'])

        self.command(self.first, 'view', camera_id=first['id'])
        self.camera.orbit(self.first, yaw=15)
        self.assertEqual(self.first.state.mode, 'free')
        self.assertFalse(self.controller.is_locked(self.first.client_id))
        self.assertEqual(self.second.state.mode, 'free')

    def test_real_viser_camera_restores_free_pose_and_project_epoch(self):
        self.server.disconnect(self.first)
        self.first = real_camera_client(1, [4, 2, -5], [0, 1, 0], [0, 1, 0], .72)
        self.server.clients[1] = self.first
        self.controller.update()
        original = dict(position=self.first.camera.position.copy(),
                        look_at=self.first.camera.look_at.copy(),
                        up_direction=self.first.camera.up_direction.copy(),
                        fov=self.first.camera.fov)
        record = self.session.add_camera([1, 2, 3], [1, 0, 0, 0], .8)
        self.controller.update()

        self.command(self.first, 'view', camera_id=record['id'])
        self.assert_pose(self.first, record)
        self.command(self.first, 'free')
        self.assertEqual(self.first.state.mode, 'free')
        np.testing.assert_allclose(self.first.camera.position, original['position'])
        np.testing.assert_allclose(self.first.camera.look_at, original['look_at'])
        np.testing.assert_allclose(self.first.camera.up_direction, original['up_direction'])
        self.assertAlmostEqual(self.first.camera.fov, original['fov'])

        self.command(self.first, 'view', camera_id=record['id'])
        with tempfile.TemporaryDirectory() as directory:
            self.session.new_project(directory)
            self.controller.update()
        self.assertEqual(self.first.state.mode, 'free')
        np.testing.assert_allclose(self.first.camera.position, original['position'])
        np.testing.assert_allclose(self.first.camera.look_at, original['look_at'])
        np.testing.assert_allclose(self.first.camera.up_direction, original['up_direction'])
        self.assertAlmostEqual(self.first.camera.fov, original['fov'])

    def test_stale_project_revision_and_take_envelopes_cannot_mutate(self):
        old = self.first.state
        camera = self.session.add_camera([0, 1, 2], [1, 0, 0, 0], .75)
        self.controller.update()
        revision = self.session.project_revision
        for envelope in (old,
                         CameraStudioCommandMessage('wrong-project', revision, self.take.id, 'update'),
                         CameraStudioCommandMessage(self.session.camera_project_id,
                                                    revision, 'wrong-take', 'update')):
            with self.subTest(envelope=envelope):
                state = self.command(self.first, 'delete', envelope=envelope,
                                     camera_id=camera['id'])
                self.assertIn('Project changed', state.error)
                self.assertEqual(self.session.project_revision, revision)
                self.assertEqual(len(self.session.cameras), 1)
                self.assertEqual(state.mode, 'free')
                self.assertEqual(state.project_id, self.session.camera_project_id)

    def test_failed_commands_preserve_model_revision_and_view(self):
        camera = self.session.add_camera([0, 1, 2], [1, 0, 0, 0], .75)
        self.controller.update()
        before = copy.deepcopy(self.session.cameras)
        revision = self.session.project_revision
        errors = [('update', dict(camera_id=camera['id'], camera={'fov': 0})),
                  ('update', dict(camera_id=camera['id'], camera={'unknown': 3})),
                  ('cut_add', dict(camera_id=camera['id'], frame=30)),
                  ('cut_update', dict(cut_id='missing', frame=4)),
                  ('cut_delete', dict(cut_id='missing')),
                  ('sequence', {}), ('bogus', {})]
        for action, payload in errors:
            with self.subTest(action=action, payload=payload):
                state = self.command(self.first, action, **payload)
                self.assertTrue(state.error)
                self.assertEqual(self.session.project_revision, revision)
                self.assertEqual(self.session.cameras, before)
                self.assertEqual(self.take.camera_cuts, [])
                self.assertEqual(state.mode, 'free')

    def test_clear_delete_and_project_epoch_release_locked_views(self):
        first = self.session.add_camera([1, 2, 3], [1, 0, 0, 0], .8)
        second = self.session.add_camera([-4, 5, 6], [0, 0, 1, 0], 1.1)
        self.session.add_camera_cut(self.take.id, 0, first['id'])
        self.session.add_camera_cut(self.take.id, 12, second['id'])
        self.controller.update()
        self.command(self.first, 'sequence')
        self.command(self.second, 'view', camera_id=second['id'])

        revision = self.session.project_revision
        state = self.command(self.first, 'delete', camera_id=first['id'])
        self.assertTrue(state.error)
        self.assertEqual(self.session.project_revision, revision)
        self.command(self.first, 'cuts_clear')
        self.assertEqual(self.first.state.mode, 'free')
        self.assertEqual(self.take.camera_cuts, [])
        self.command(self.second, 'delete', camera_id=second['id'])
        self.assertEqual(self.second.state.mode, 'free')
        self.assertEqual(self.second.state.active_camera_id, None)
        self.assertEqual(len(self.session.cameras), 1)

        self.command(self.first, 'view', camera_id=first['id'])
        prior_epoch = self.first.state.project_id
        with tempfile.TemporaryDirectory() as directory:
            self.session.new_project(directory)
            self.controller.update()
        self.assertNotEqual(self.first.state.project_id, prior_epoch)
        self.assertEqual(self.first.state.mode, 'free')
        self.assertEqual(self.first.state.cameras, [])
        self.assertEqual(self.first.state.cuts, [])
        self.assertIsNone(self.first.state.take_id)
        self.assertEqual(self.second.state.mode, 'free')

    def test_switching_to_take_without_cuts_releases_sequence_and_rejects_old_take(self):
        camera = self.session.add_camera([1, 2, 3], [1, 0, 0, 0], .8)
        self.session.add_camera_cut(self.take.id, 0, camera['id'])
        self.controller.update()
        old_state = self.command(self.first, 'sequence')
        editor_position = np.array([4, 2, -5])
        replacement = _take('take-b')
        self.session.takes[replacement.id] = replacement
        self.session.select_take(replacement.id)
        self.controller.update()
        self.assertEqual(self.first.state.take_id, replacement.id)
        self.assertEqual(self.first.state.mode, 'free')
        np.testing.assert_allclose(self.first.camera.position, editor_position)
        state = self.command(self.first, 'cut_add', envelope=old_state,
                             camera_id=camera['id'], frame=0)
        self.assertIn('Project changed', state.error)
        self.assertEqual(self.take.camera_cuts[0]['camera_id'], camera['id'])
        self.assertEqual(replacement.camera_cuts, [])

    def test_busy_generation_rejects_cut_edits_but_allows_camera_update(self):
        camera = self.session.add_camera([1, 2, 3], [1, 0, 0, 0], .8)
        self.session.busy = True
        self.controller.update()
        revision = self.session.project_revision
        state = self.command(self.first, 'cut_add', camera_id=camera['id'], frame=0)
        self.assertTrue(state.error)
        self.assertEqual(self.session.project_revision, revision)
        self.assertEqual(self.take.camera_cuts, [])
        state = self.command(self.first, 'update', camera_id=camera['id'],
                             camera={'name': 'During generation'})
        self.assertIsNone(state.error)
        self.assertEqual(self.session.project_revision, revision + 1)
        self.assertEqual(state.cameras[0]['name'], 'During generation')

    def test_disconnect_discards_client_mode_and_new_connection_starts_free(self):
        record = self.session.add_camera([1, 2, 3], [1, 0, 0, 0], .8)
        self.controller.update()
        self.command(self.first, 'view', camera_id=record['id'])
        self.assertTrue(self.controller.is_locked(self.first.client_id))
        self.server.disconnect(self.first)
        self.assertFalse(self.controller.is_locked(self.first.client_id))
        replacement = FakeClient(1, [5, 6, -2], [0, 0, 0], [0, 1, 0], .7)
        self.server.clients[1] = replacement
        self.controller.update()
        self.assertEqual(replacement.state.mode, 'free')
        np.testing.assert_allclose(replacement.camera.position, [5, 6, -2])

    def test_camera_gui_exposes_exact_extension_marker(self):
        gui = FakeGui()
        self.camera.build_gui(gui)
        self.assertEqual(gui.html, ['<div data-stagezero-cameras></div>'])


if __name__ == '__main__':
    unittest.main()
