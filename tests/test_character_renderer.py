"""Actor GLB transport: revision races, per-client readiness, and poses."""
import unittest

import msgspec
import numpy as np
from viser import _messages

from character_renderer import (
    ActorGlbCommandMessage,
    ActorGlbLoadMessage,
    ActorGlbPoseMessage,
    ActorGlbStatusMessage,
    GlbCharacterRenderer,
)


class FakeConnection:
    def __init__(self):
        self.messages = []

    def queue_message(self, message):
        self.messages.append(message)


class FakeClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self._websock_connection = FakeConnection()

    @property
    def messages(self):
        return self._websock_connection.messages


class FakeSocket:
    def __init__(self):
        self.handlers = {}

    def register_handler(self, cls, handler):
        self.handlers[cls] = handler


class FakeServer:
    def __init__(self):
        self._websock_server = FakeSocket()
        self.clients = {1: FakeClient(1), 2: FakeClient(2)}
        self.connected = None
        self.disconnected = None

    def get_clients(self):
        return dict(self.clients)

    def on_client_connect(self, callback):
        self.connected = callback

    def on_client_disconnect(self, callback):
        self.disconnected = callback

    def ack(self, client_id, asset_id, revision, status="loaded", error=None):
        self._websock_server.handlers[ActorGlbStatusMessage](
            client_id, ActorGlbStatusMessage(asset_id, revision, status, error)
        )


class CharacterRendererTests(unittest.TestCase):
    def setUp(self):
        self.server = FakeServer()
        self.results = []
        self.renderer = GlbCharacterRenderer(
            self.server, on_result=lambda *result: self.results.append(result)
        )

    def test_candidate_commits_only_after_initiator_ack_and_preserves_old_on_failure(self):
        first = self.renderer.load("first", b"glb-one", 1, required_nodes=(2, 3))
        self.assertEqual(self.renderer.state, "loading")
        self.assertFalse(self.renderer.commit(first))
        self.assertIsInstance(self.server.clients[1].messages[-1], ActorGlbLoadMessage)
        self.server.ack(2, "first", first)
        self.assertFalse(self.renderer.commit(first))
        self.server.ack(1, "first", first)
        self.assertTrue(self.renderer.commit(first))
        self.assertEqual(self.renderer.active_asset_id, "first")

        second = self.renderer.load("second", b"glb-two", 1)
        self.server.ack(1, "second", second, "error", "parse failed")
        self.assertTrue(self.renderer.reject(second))
        self.assertEqual(self.renderer.active_revision, first)
        self.assertEqual(self.server.clients[2].messages[-1],
                         ActorGlbCommandMessage(second, "reject"))
        self.server.ack(1, "second", second)
        self.assertEqual(len([result for result in self.results if result[2] == second]), 1)

    def test_pose_is_column_major_and_replayed_to_late_tab(self):
        matrices = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
        matrices[2, 0, 3] = 1.5
        revision = self.renderer.load("hero", b"glb", 1, initial_node_matrices=matrices)
        pose = [message for message in self.server.clients[1].messages
                if isinstance(message, ActorGlbPoseMessage)][-1]
        self.assertEqual(pose.node_indices.tolist(), [0, 1, 2])
        self.assertAlmostEqual(pose.local_matrices[2 * 16 + 12], 1.5)
        self.server.ack(1, "hero", revision)
        self.assertTrue(self.renderer.commit(revision))
        late = FakeClient(3)
        self.server.clients[3] = late
        self.server.connected(late)
        self.assertEqual([type(message) for message in late.messages],
                         [ActorGlbLoadMessage, ActorGlbPoseMessage,
                          ActorGlbCommandMessage])

    def test_superseded_and_disconnected_initiator_do_not_replace_actor(self):
        old = self.renderer.load("old", b"old", 1)
        new = self.renderer.load("new", b"new", 1)
        self.server.ack(1, "old", old)
        self.assertFalse(self.renderer.commit(old))
        self.server.disconnected(self.server.clients.pop(1))
        self.assertEqual(self.renderer.state, "g1")
        self.assertEqual(self.results[-1][2:], (new, "error", "Initiating tab disconnected during GLB load"))
        self.assertFalse(self.renderer.commit(new))

    def test_late_tab_during_pending_load_receives_current_actor_first(self):
        first = self.renderer.load("first", b"old", 1)
        self.server.ack(1, "first", first)
        self.assertTrue(self.renderer.commit(first))
        second = self.renderer.load("second", b"new", 1)
        late = FakeClient(3)
        self.server.clients[3] = late
        self.server.connected(late)
        self.assertEqual(
            [message.asset_id for message in late.messages
             if isinstance(message, ActorGlbLoadMessage)], ["first"]
        )
        self.assertTrue(self.renderer.reject(second))
        self.assertEqual(self.renderer.active_asset_id, "first")
        self.assertEqual(
            [message.asset_id for message in late.messages
             if isinstance(message, ActorGlbLoadMessage)], ["first"]
        )

    def test_timeout_rejects_candidate_without_clearing_active_actor(self):
        first = self.renderer.load("first", b"old", 1)
        self.server.ack(1, "first", first)
        self.assertTrue(self.renderer.commit(first))
        second = self.renderer.load("second", b"new", 1)
        self.renderer._pending.deadline = 0
        self.renderer.poll()
        self.assertEqual(self.renderer.active_revision, first)
        self.assertEqual(self.renderer.state, "glb")
        self.assertEqual(self.results[-1][2:],
                         (second, "error", "GLB load timed out"))
        self.assertFalse(self.renderer.reject(second))

    def test_custom_messages_round_trip_through_pinned_viser_codec(self):
        messages = [
            ActorGlbLoadMessage("/actor/glb_actor", "id", 1, b"glb", 1.0,
                                "/actor/g1_mesh", (1, 2)),
            ActorGlbCommandMessage(1, "commit"),
            ActorGlbStatusMessage("id", 1, "loaded", None),
        ]
        for message in messages:
            with self.subTest(type=type(message).__name__):
                packed = msgspec.msgpack.encode(message.as_serializable_dict())
                decoded = _messages.Message.deserialize(packed)
                self.assertEqual(decoded, message)
        pose = ActorGlbPoseMessage(
            1, np.array([2], dtype=np.uint32), np.arange(16, dtype=np.float32)
        )
        decoded_pose = msgspec.msgpack.decode(
            msgspec.msgpack.encode(pose.as_serializable_dict())
        )
        self.assertEqual(decoded_pose["type"], "ActorGlbPoseMessage")
        self.assertEqual(len(decoded_pose["node_indices"]), 4)
        self.assertEqual(len(decoded_pose["local_matrices"]), 64)


if __name__ == "__main__":
    unittest.main()
