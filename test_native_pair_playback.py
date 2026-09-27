"""Native local playback parity, bounded binary payloads and transport races."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import msgspec
import numpy as np
from scipy.spatial.transform import Rotation
from viser import _messages

from experiments.native_pair_rig import NativeRigActor, NativeRigAsset
from native_pair_playback import (NativePairClipMessage, NativePairPlaybackController,
    NativePairStatusMessage, NativePairTransportMessage, serialize_native_actors)


class Scene:
    def add_mesh_simple(self, name, vertices, faces, **kwargs):
        return SimpleNamespace(name=name, vertices=vertices, visible=True, remove=lambda: None)


class Client:
    def __init__(self, identifier):
        self.client_id = identifier
        self.messages = []
        self._websock_connection = SimpleNamespace(queue_message=self.messages.append)


class Server:
    def __init__(self):
        self.clients = {1: Client(1), 2: Client(2)}
        self.handlers = {}
        self._websock_server = SimpleNamespace(register_handler=self.handlers.__setitem__)

    def get_clients(self):
        return dict(self.clients)

    def on_client_connect(self, callback):
        self.connected = callback

    def on_client_disconnect(self, callback):
        self.disconnected = callback

    def ack(self, identifier, revision, status='loaded', error=None):
        self.handlers[NativePairStatusMessage](identifier,
            NativePairStatusMessage(revision, status, error))


def decoded_skin(actor, frame, frames):
    rest = np.frombuffer(actor['rest'], '<f4').reshape(22, 3)
    linear = np.frombuffer(actor['linear'], '<f4').reshape(frames, 22, 3, 3)[frame]
    targets = np.frombuffer(actor['targets'], '<f4').reshape(frames, 22, 3)[frame]
    outputs = []
    for part in actor['parts']:
        bones = np.frombuffer(part['bones'], '<u2').reshape(-1, 4)
        weights = np.frombuffer(part['weights'], '<f4').reshape(-1, 4)
        bind = np.frombuffer(part['bind_world'], '<f4').reshape(-1, 4, 3)
        moved = np.einsum('vwij,vwj->vwi', linear[bones], bind - rest[bones]) + targets[bones]
        outputs.append(np.sum(moved * weights[..., None], axis=1))
    return outputs


class NativePairPlaybackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.asset = NativeRigAsset(Path(__file__).parent / 'assets/paired/Xbot.glb')

    def setUp(self):
        self.server = Server()
        self.state = {'frame': 0, 'playing': True, 'capturing': False, 'transport_revision': 1}
        self.controller = NativePairPlaybackController(self.server, get_state=lambda: dict(self.state))
        self.actor = self.make_actor()

    def make_actor(self, prefix='/native/a', frames=8, hand_pose=None):
        actor = NativeRigActor(SimpleNamespace(scene=Scene()), prefix, self.asset, (100, 150, 200))
        poses = []
        for index in range(frames):
            pose = self.asset.rest.copy()
            pose[18] += np.array([-.02, -.05, .025]) * np.sin(index / max(frames - 1, 1))
            rotation = Rotation.from_euler('y', .3 * index / frames).as_matrix()
            poses.append(pose @ rotation.T + [index * .01, 0., -.2])
        options = {} if hand_pose is None else {'hand_pose': hand_pose, 'contact_weights': np.ones((frames, 2))}
        actor.prepare_clip(np.asarray(poses), **options)
        return actor

    def load(self):
        return self.controller.load([('a', self.actor)], fps=30, frames=8)

    def test_actual_authored_mesh_matches_cpu_for_relaxed_hand_and_fist(self):
        for hand_pose in (None, 'handshake', 'fist'):
            with self.subTest(hand_pose=hand_pose):
                actor = self.make_actor(hand_pose=hand_pose)
                outputs, _ = serialize_native_actors([('a', actor)], frames=8)
                for frame in (0, 3, 7):
                    linear, targets, _, fingers, pose = actor._prepared[frame]
                    expected = self.asset.skin(linear, targets, fingers, pose)
                    actual = decoded_skin(outputs[0], frame, 8)
                    for received, original in zip(actual, expected):
                        np.testing.assert_allclose(received, original, atol=1e-6, rtol=1e-6)
                self.assertIsNone(actor.vertices)  # serialization publishes no CPU mesh frame

    def test_packet_roundtrip_sizes_and_savings_against_vertex_frames(self):
        actor = self.make_actor(frames=210)
        outputs, count = serialize_native_actors([('a', actor)], frames=210)
        vertices = sum(len(part['weights']) for part in self.asset.parts)
        expected = (22 * 3 + 210 * 22 * 12) * 4 + vertices * 72
        self.assertEqual(count, expected)
        self.assertLess(count, 210 * vertices * 3 * 4 / 10)
        packet = NativePairClipMessage(5, 30, 210, outputs)
        encoded = msgspec.msgpack.encode(packet.as_serializable_dict())
        decoded = _messages.Message.deserialize(encoded)
        self.assertEqual(decoded.revision, 5)
        self.assertEqual(decoded.actors[0]['linear'], outputs[0]['linear'])
        self.assertEqual(set(outputs[0]), {'rest', 'linear', 'targets', 'parts'})
        self.assertEqual(set(outputs[0]['parts'][0]), {'name', 'bind_world', 'bones', 'weights'})
        self.assertLess(len(encoded), count + 2000)
        self.assertEqual(packet.redundancy_key(), NativePairClipMessage(6, 30, 8, []).redundancy_key())

    def test_one_clip_and_no_thirty_hz_transport_for_natural_ticks(self):
        revision = self.load()
        self.assertEqual(revision, 1)
        self.assertTrue(self.controller.update(self.state, True))
        for frame in range(1, 8):
            self.state['frame'] = frame
            self.assertFalse(self.controller.update(self.state, True))
        for client in self.server.clients.values():
            self.assertEqual(len(client.messages), 2)
            self.assertIsInstance(client.messages[0], NativePairClipMessage)
            self.assertIsInstance(client.messages[1], NativePairTransportMessage)
        self.assertGreater(self.controller.payload_bytes, 0)

    def test_seek_restart_pause_enable_and_capture_send_authoritative_transports(self):
        self.load()
        self.controller.update(self.state, True)
        self.state.update(frame=4, transport_revision=2)
        self.assertTrue(self.controller.update(self.state, True))
        self.state.update(frame=0, transport_revision=3)
        self.assertTrue(self.controller.update(self.state, True))
        self.state['playing'] = False
        self.assertTrue(self.controller.update(self.state, True))
        self.assertTrue(self.controller.update(self.state, False))
        self.state['capturing'] = True
        self.assertTrue(self.controller.update(self.state, True))
        self.state['frame'] = 5
        self.assertTrue(self.controller.update(self.state, True))
        self.assertFalse(self.controller.update(self.state, True))
        messages = self.server.clients[1].messages[1:]
        self.assertEqual([item.sequence for item in messages], list(range(1, 8)))
        self.assertEqual(messages[-1].frame, 5)
        self.assertTrue(messages[-1].capturing)

    def test_reconnect_sends_latest_clip_and_fresh_clock_then_discards_stale_acks(self):
        revision = self.load()
        self.controller.update(self.state, True)
        self.server.ack(1, revision)
        self.assertEqual(self.controller.client_status[1]['status'], 'loaded')
        self.state['frame'] = 6  # no update; reconnect must read authoritative state
        late = self.server.clients[1] = Client(1)
        self.server.connected(late)
        self.assertEqual(len(late.messages), 2)
        self.assertEqual(late.messages[-1].frame, 6)
        self.assertEqual(late.messages[-1].sequence, 2)
        self.assertTrue(late.messages[-1].enabled)
        self.assertNotIn(1, self.controller.client_status)
        replacement = self.load()
        self.server.ack(1, revision)
        self.server.ack(999, replacement)
        self.assertEqual(self.controller.client_status, {})
        self.server.ack(1, replacement, 'error', 'x' * 900)
        self.assertEqual(len(self.controller.client_status[1]['error']), 500)
        self.server.disconnected(late)
        self.server.ack(1, replacement)
        self.assertEqual(self.controller.client_status, {})

    def test_clear_disables_and_releases_clip_without_reusing_revision_or_sequence(self):
        self.load()
        self.controller.update(self.state, True)
        self.controller.clear()
        last = self.server.clients[1].messages[-1]
        self.assertFalse(last.enabled)
        self.assertFalse(last.playing)
        self.assertEqual(last.sequence, 2)
        self.assertEqual(self.controller.payload_bytes, 0)
        late = Client(3)
        self.server.connected(late)
        self.assertEqual(late.messages, [])
        self.assertEqual(self.load(), 2)
        self.controller.update(self.state, True)
        self.assertEqual(self.server.clients[1].messages[-1].sequence, 3)

    def test_invalid_payload_keeps_last_good_clip_and_revision(self):
        self.load()
        original = self.server.clients[1].messages[-1]
        self.actor._prepared[0][0][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.load()
        self.assertEqual(self.controller.revision, 1)
        self.assertIs(self.server.clients[1].messages[-1], original)

    def test_rejects_variable_fingers_and_invalid_counts_before_send(self):
        actor = self.make_actor(hand_pose='fist')
        actor._prepared[-1][3][0] = .2
        with self.assertRaisesRegex(ValueError, 'Variable finger'):
            self.controller.load([('a', actor)], frames=8)
        with self.assertRaisesRegex(ValueError, '30 fps'):
            self.controller.load([('a', self.actor)], fps=60, frames=8)
        for frames in (0, 3, 1001, True):
            with self.assertRaises(ValueError):
                self.controller.load([('a', self.actor)], frames=frames)
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.controller.load([('a', self.actor), ('a', self.actor)], frames=8)
        with patch('native_pair_playback.MAX_PAYLOAD_BYTES', 10):
            with self.assertRaisesRegex(ValueError, 'payload byte limit'):
                self.load()
        self.assertEqual(self.controller.revision, 0)
        self.assertEqual(self.server.clients[1].messages, [])

    def test_rejects_nonfinite_float32_overflow_and_out_of_range_transport(self):
        self.actor._prepared[0][0][0, 0, 0] = 1e99
        with self.assertRaisesRegex(ValueError, 'float32'):
            self.load()
        self.actor._prepared[0][0][0, 0, 0] = 1.
        self.load()
        for frame in (-1, 8, True, float('nan')):
            with self.assertRaises(ValueError):
                self.controller.update(dict(self.state, frame=frame), True)

    def test_capture_readiness_requires_this_client_and_this_revision(self):
        revision = self.load()
        first, second = self.server.clients.values()
        self.server.ack(first.client_id, revision)
        self.assertTrue(self.controller.is_ready(first))
        self.assertEqual(self.controller.require_ready(first, timeout=0), revision)
        self.assertFalse(self.controller.is_ready(second))
        with self.assertRaisesRegex(RuntimeError, 'Reload this tab'):
            self.controller.require_ready(second, timeout=0)
        replacement = self.load()
        self.server.ack(first.client_id, revision)  # stale load is not capture ready
        self.assertFalse(self.controller.is_ready(first))
        with self.assertRaisesRegex(RuntimeError, 'Reload this tab'):
            self.controller.require_ready(first, timeout=0)
        self.server.ack(first.client_id, replacement, 'error', 'unsupported viewer')
        with self.assertRaisesRegex(RuntimeError, 'could not load'):
            self.controller.require_ready(first)
        self.server.ack(first.client_id, replacement)
        self.server.clients[first.client_id] = Client(first.client_id)
        self.assertFalse(self.controller.is_ready(first))
        with self.assertRaisesRegex(RuntimeError, 'disconnected'):
            self.controller.require_ready(first)

    def test_capture_readiness_wait_wakes_on_ack_and_rejects_a_replaced_clip(self):
        revision = self.load()
        client = self.server.clients[1]
        for replacement in (False, True):
            with self.subTest(replacement=replacement):
                waiting = Event()
                original_wait = self.controller._ready.wait
                def wait(timeout):
                    waiting.set()
                    return original_wait(timeout)
                with patch.object(self.controller._ready, 'wait', side_effect=wait):
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        result = executor.submit(self.controller.require_ready, client, timeout=1.)
                        self.assertTrue(waiting.wait(1.))
                        if replacement:
                            self.load()
                            with self.assertRaisesRegex(RuntimeError, 'changed or was removed'):
                                result.result(timeout=1.)
                        else:
                            self.server.ack(1, revision)
                            self.assertEqual(result.result(timeout=1.), revision)
                revision = self.load()

    def test_capture_readiness_timeout_is_bounded_and_never_exports_an_unloaded_tab(self):
        self.load()
        client = self.server.clients[1]
        with self.assertRaisesRegex(RuntimeError, 'not loaded local playback'):
            self.controller.require_ready(client, timeout=.001)
        for timeout in (-1, 6, float('inf'), float('nan'), True):
            with self.assertRaises(ValueError):
                self.controller.require_ready(client, timeout=timeout)
        self.controller.clear()
        with self.assertRaisesRegex(RuntimeError, 'removed'):
            self.controller.require_ready(client, timeout=0)

    def test_reconnect_retries_state_if_a_pause_is_sent_during_snapshot_read(self):
        self.load()
        self.controller.update(self.state, True)
        captured, release = Event(), Event()
        reads = []
        def get_state():
            state = dict(self.state)
            reads.append(state)
            if len(reads) == 1:
                captured.set()
                if not release.wait(2.):
                    raise AssertionError('Reconnect snapshot was not released')
            return state
        self.controller.get_state = get_state
        late = self.server.clients[1] = Client(1)
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(self.server.connected, late)
            self.assertTrue(captured.wait(1.))
            self.state.update(frame=3, playing=False, transport_revision=2)
            self.controller.update(self.state, True)
            release.set()
            result.result(timeout=1.)
        self.assertEqual(len(reads), 2)
        transports = [message for message in late.messages if isinstance(message, NativePairTransportMessage)]
        self.assertEqual([message.sequence for message in transports], [2, 3])
        self.assertTrue(all(message.frame == 3 and not message.playing for message in transports))
        self.assertEqual(sum(isinstance(message, NativePairClipMessage) for message in late.messages), 1)

    def test_unloaded_client_gets_one_reload_notice_and_loaded_or_stale_clients_do_not(self):
        client = self.server.clients[1]
        notifications = []
        client.add_notification = lambda **message: notifications.append(message)
        with patch('native_pair_playback.Timer') as timer:
            revision = self.load()
            timer.assert_called_once()
            self.assertEqual(timer.call_args.args[0], 5.)
            self.assertTrue(timer.return_value.daemon)
            self.controller._readiness_notice(client, revision)
            self.assertEqual(len(notifications), 1)
            self.assertEqual(notifications[0]['title'], 'Reload for local playback')
            self.assertFalse(notifications[0]['auto_close'])
            self.assertIn('Reload this tab', notifications[0]['body'])
            self.server.ack(1, revision)
            self.controller._readiness_notice(client, revision)
            self.assertEqual(len(notifications), 1)
            self.load()
            self.controller._readiness_notice(client, revision)
            self.assertEqual(len(notifications), 1)
            self.controller.clear()
            timer.return_value.cancel.assert_called()

    def test_director_registers_native_status_before_the_socket_server_starts(self):
        # A fresh interpreter matters: this test module itself imports the
        # protocol, which would hide a late-import startup race in this process.
        script = textwrap.dedent('''
            import sys
            import msgspec
            from viser import _messages
            import director_viewer

            class ReachedServerStartup(Exception):
                pass

            def check_server_start(**kwargs):
                # First websocket deserialization freezes Viser's cached type
                # registry. A returning tab can acknowledge immediately here.
                message = _messages.Message.deserialize(msgspec.msgpack.encode({
                    'type': 'NativePairStatusMessage', 'revision': 1,
                    'status': 'loaded', 'error': None,
                }))
                assert type(message).__name__ == 'NativePairStatusMessage'
                assert message.revision == 1 and message.status == 'loaded'
                raise ReachedServerStartup

            director_viewer.create_studio_server = check_server_start
            director_viewer.load_recording = lambda path: (None, None, None)
            sys.argv = ['director_viewer.py']
            try:
                director_viewer.main()
            except ReachedServerStartup:
                pass
            else:
                raise AssertionError('Director did not reach server startup')
        ''')
        completed = subprocess.run([sys.executable, '-c', script],
            cwd=Path(__file__).parent, capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == '__main__':
    unittest.main()
