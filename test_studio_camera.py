"""Behavioral tests for StageZero's per-client camera movement."""

import unittest

import numpy as np

from studio_camera import StudioCamera


class FakeCamera:
    def __init__(self):
        self._position = np.zeros(3)
        self.look_at = np.zeros(3)
        self.up_direction = np.zeros(3)
        self.fov = 0.0
        self.writes = 0

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, new_position):
        new_position = np.asarray(new_position, dtype=np.float64)
        self.look_at = self.look_at + new_position - self._position
        self._position = new_position
        self.writes += 1


class FakeClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self.camera = FakeCamera()


class FakeServer:
    def __init__(self, clients):
        self.clients = {client.client_id: client for client in clients}

    def get_clients(self):
        return self.clients


class StudioCameraTests(unittest.TestCase):
    def setUp(self):
        self.root = np.array((0.0, 0.0, 0.0))
        self.clients = [FakeClient(1), FakeClient(2)]
        self.camera = StudioCamera(FakeServer(self.clients), lambda: self.root)
        for client in self.clients:
            self.camera.reset(client)

    def test_follow_starts_without_jump_and_preserves_manual_orbit(self):
        first, second = self.clients
        self.camera.orbit(first, yaw=25)
        first_offset = first.camera.position - first.camera.look_at
        second_position = second.camera.position.copy()
        first_writes = first.camera.writes
        self.camera.set_follow(True)
        self.camera.update(self.root)
        self.assertEqual(first.camera.writes, first_writes)
        self.root = np.array((2.0, 0.0, -1.0))
        self.camera.update(self.root)
        np.testing.assert_allclose(first.camera.position - first.camera.look_at, first_offset)
        np.testing.assert_allclose(second.camera.position, second_position + self.root)
        self.camera.set_follow(False)
        stopped_position = first.camera.position.copy()
        self.camera.update(np.array((3.0, 0.0, 0.0)))
        np.testing.assert_allclose(first.camera.position, stopped_position)

    def test_rebase_after_seek_keeps_camera_still(self):
        client = self.clients[0]
        self.camera.set_follow(True)
        before = client.camera.position.copy()
        self.root = np.array((100.0, 0.0, 0.0))
        self.camera.rebase()
        self.camera.update(self.root)
        np.testing.assert_allclose(client.camera.position, before)

    def test_focus_preserves_angle_and_preset_targets_actor(self):
        client = self.clients[0]
        self.camera.zoom(client, 1)
        old_offset = client.camera.position - client.camera.look_at
        self.root = np.array((3.0, 0.0, -4.0))
        self.camera.focus(client)
        np.testing.assert_allclose(client.camera.position - client.camera.look_at, old_offset)
        np.testing.assert_allclose(client.camera.look_at, self.camera.default_target + self.root)
        self.camera.preset(client, "Top")
        self.assertGreater(client.camera.position[1], client.camera.look_at[1])
        np.testing.assert_allclose(client.camera.look_at, self.camera.default_target + self.root)

    def test_pan_moves_position_and_target_together(self):
        client = self.clients[0]
        offset = client.camera.position - client.camera.look_at
        self.camera.pan(client, horizontal=1, vertical=1)
        np.testing.assert_allclose(client.camera.position - client.camera.look_at, offset)


if __name__ == "__main__":
    unittest.main()
