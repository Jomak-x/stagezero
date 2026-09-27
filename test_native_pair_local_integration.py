"""Local native playback keeps CPU meshes quiet and session transport explicit."""
import unittest

import numpy as np

from native_pair_playback import (NativePairClipMessage, NativePairPlaybackController,
                                 NativePairTransportMessage)
from native_pair_renderer import NativePairRenderer
from native_pair_session import NativePairSession
from test_native_pair_playback import Client, Server


class CountingMesh:
    def __init__(self, name, vertices, events):
        self.name = name
        self._vertices = vertices.copy()
        self.vertex_updates = 0
        self.visible = True
        self.removed = False
        self.events = events

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        self._vertices = value.copy()
        self.vertex_updates += 1

    def remove(self):
        self.removed = True
        self.events.append(('remove', self.name))


class CountingScene:
    def __init__(self, events):
        self.meshes, self.events = [], events

    def add_mesh_simple(self, name, vertices, faces, **kwargs):
        handle = CountingMesh(name, vertices, self.events)
        self.meshes.append(handle)
        return handle


class NativePairLocalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.session = NativePairSession(clock=lambda: self.now)
        self.addCleanup(self.session.close)
        self.session.load_reviewed_handshake()

    def renderer(self, *, local=True):
        events = []
        server = Server()
        server.scene = CountingScene(events)
        for client in server.clients.values():
            def queue(message, client=client):
                client.messages.append(message)
                if isinstance(message, NativePairTransportMessage):
                    events.append(('transport', message.enabled))
            client._websock_connection.queue_message = queue
        renderer = NativePairRenderer(server)
        if local:
            renderer.local_playback = NativePairPlaybackController(server, get_state=self.session.tick)
        self.addCleanup(renderer.remove)
        renderer.sync_cast(self.session.snapshot())
        renderer.set_clip(self.session.timeline_clip())
        return renderer, server, events

    def test_natural_local_ticks_update_camera_roots_without_mesh_vertex_or_socket_updates(self):
        renderer, server, _ = self.renderer()
        controller = renderer.local_playback
        state = self.session.play()
        controller.update(state, True)
        initial_vertices = [mesh.vertices.copy() for mesh in server.scene.meshes]
        initial_counts = [mesh.vertex_updates for mesh in server.scene.meshes]
        clip = self.session.timeline_clip()
        for frame in range(1, 61):
            self.now = 10. + frame / 30
            state = self.session.tick()
            renderer.tick(state['frame'])
            self.assertFalse(controller.update(state, True))
            for index, identifier in enumerate(state['selected_pair']):
                np.testing.assert_array_equal(renderer.actor_root(identifier), clip.joints[frame, index, 0])
        self.assertEqual([mesh.vertex_updates for mesh in server.scene.meshes], initial_counts)
        for mesh, original in zip(server.scene.meshes, initial_vertices):
            np.testing.assert_array_equal(mesh.vertices, original)
        for client in server.clients.values():
            self.assertEqual(len(client.messages), 2)
            self.assertIsInstance(client.messages[0], NativePairClipMessage)
            self.assertIsInstance(client.messages[1], NativePairTransportMessage)
        # Seeks use transport packets too, without reserializing or reskinning.
        state = self.session.seek(100)
        renderer.tick(state['frame'])
        self.assertTrue(controller.update(state, True))
        self.assertEqual(server.clients[1].messages[-1].frame, 100)
        self.assertEqual([mesh.vertex_updates for mesh in server.scene.meshes], initial_counts)

    def test_cpu_renderer_still_updates_meshes_without_local_playback(self):
        renderer, server, _ = self.renderer(local=False)
        counts = [mesh.vertex_updates for mesh in server.scene.meshes]
        renderer.tick(80)
        self.assertEqual([mesh.vertex_updates for mesh in server.scene.meshes], [count + 1 for count in counts])
        for index, identifier in enumerate(self.session.snapshot()['selected_pair']):
            np.testing.assert_array_equal(renderer.actors[identifier].joint_positions,
                                          self.session.timeline_clip().joints[80, index])

    def test_remove_stops_playback_before_mesh_removal_and_does_not_replay_removed_clip(self):
        renderer, server, events = self.renderer()
        controller = renderer.local_playback
        controller.update(self.session.play(), True)
        events.clear()
        renderer.remove()
        self.assertEqual(events[:2], [('transport', False), ('transport', False)])
        self.assertTrue(all(event[0] == 'remove' for event in events[2:]))
        self.assertTrue(all(mesh.removed for mesh in server.scene.meshes))
        self.assertEqual(controller.payload_bytes, 0)
        for client in server.clients.values():
            self.assertFalse(client.messages[-1].playing)
            self.assertFalse(client.messages[-1].enabled)
        late = Client(3)
        server.connected(late)
        self.assertEqual(late.messages, [])
        count = len(events)
        renderer.remove()
        self.assertEqual(len(events), count)

    def test_transport_revision_changes_on_explicit_commands_not_playback_ticks(self):
        baseline = self.session.snapshot()['transport_revision']
        state = self.session.play()
        self.assertGreater(state['transport_revision'], baseline)
        revision = state['transport_revision']
        for offset in (.1, .5, 1.):
            self.now = 10. + offset
            state = self.session.tick()
            self.assertEqual(state['transport_revision'], revision)
        self.assertEqual(state['frame'], 30)
        for operation in (lambda: self.session.seek(90), self.session.pause,
                          self.session.pause, self.session.play, self.session.restart):
            state = operation()
            self.assertGreater(state['transport_revision'], revision)
            revision = state['transport_revision']
        self.assertEqual(state['frame'], 0)
        self.assertTrue(state['playing'])
        self.now = 30.
        state = self.session.tick()
        self.assertFalse(state['playing'])
        self.assertEqual(state['transport_revision'], revision)
        with self.assertRaises(ValueError):
            self.session.seek(-1)
        self.assertEqual(self.session.snapshot()['transport_revision'], revision)

    def test_capture_commands_increment_transport_and_natural_ticks_leave_capture_frozen(self):
        revision = self.session.play()['transport_revision']
        state = self.session.begin_capture()
        self.assertGreater(state['transport_revision'], revision)
        self.assertTrue(state['capturing'])
        self.assertFalse(state['playing'])
        revision = state['transport_revision']
        for frame in (0, 30, 30, 209):
            state = self.session.capture_seek(frame)
            self.assertGreater(state['transport_revision'], revision)
            revision = state['transport_revision']
            self.now += 2.
            state = self.session.tick()
            self.assertEqual(state['frame'], frame)
            self.assertEqual(state['transport_revision'], revision)
        state = self.session.end_capture(frame=42)
        self.assertGreater(state['transport_revision'], revision)
        self.assertEqual(state['frame'], 42)
        self.assertFalse(state['capturing'])
        self.assertFalse(state['playing'])
        self.assertEqual(self.session.end_capture()['transport_revision'], state['transport_revision'])


if __name__ == '__main__':
    unittest.main()
