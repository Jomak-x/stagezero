"""Main-studio archive dispatch and exclusive browser playback regression tests."""
import io
import threading
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cast_performance import cast_from_performance, encode_project
from cast_performance_session import CastPerformanceSession
from cast_performance_renderer import CastPerformanceRenderer
from native_pair_clip import NativePairClip
from native_pair_session import NativePairSession
from native_pair_renderer import NativePairRenderer
from native_pair_playback import NativePairClipMessage, NativePairTransportMessage
from studio_cast_runtime import NativePlaybackRouter, decode_native_project, set_cast_camera_view
from test_cast_performance import performance, repack
from test_native_pair_local_integration import CountingScene
from test_native_pair_playback import Client, Server


class CastCameraTests(unittest.TestCase):
    def test_frame_camera_resets_roll_after_coupled_endpoint_writes_in_one_transaction(self):
        events = []
        class Camera:
            def __setattr__(self, key, value):
                events.append(key)
                object.__setattr__(self, key, value)
                if key == 'position':
                    object.__setattr__(self, 'look_at', np.array(value) + (1., 0., 0.))
                if key in ('position', 'look_at'):
                    object.__setattr__(self, 'up_direction', (1., 1., 0.))
        @contextmanager
        def atomic():
            events.append('begin')
            yield
            events.append('end')
        client = SimpleNamespace(camera=Camera(), atomic=atomic)
        set_cast_camera_view(client, (6., 4., 6.), (0., 1., 0.), .7)
        self.assertEqual(events, ['begin', 'position', 'look_at', 'up_direction', 'fov', 'end'])
        self.assertEqual(client.camera.look_at, (0., 1., 0.))
        self.assertEqual(client.camera.up_direction, (0., 1., 0.))
        self.assertEqual(client.camera.position, (6., 4., 6.))


class ArchiveDispatchTests(unittest.TestCase):
    def test_cast_archive_dispatch_preserves_one_two_three_actor_tracks_and_frame(self):
        for count in (1, 2, 3):
            clip = performance(count)
            content = encode_project(clip, cast_from_performance(clip), frame=6)
            mode, decoded = decode_native_project(content)
            self.assertEqual(mode, 'cast')
            self.assertEqual(decoded[0].joints.tobytes(), clip.joints.tobytes())
            self.assertEqual(decoded[0].actor_ids, clip.actor_ids)
            self.assertEqual(decoded[3], 6)

    def test_pair_archive_remains_exact_and_separate(self):
        session = NativePairSession()
        self.addCleanup(session.close)
        session.load_source(NativePairClip(performance(2).joints))
        session.seek(5)
        mode, decoded = decode_native_project(session.save())
        self.assertEqual(mode, 'paired')
        self.assertEqual(decoded[0].joints.tobytes(), session.timeline_clip().joints.tobytes())
        self.assertEqual(decoded[4], 5)

    def test_schema_does_not_bypass_full_validation(self):
        clip = performance(3)
        content = encode_project(clip, cast_from_performance(clip))
        for bad in (b'invalid', repack(content, document=lambda doc: dict(doc, frame=99)),
                    repack(content, document=lambda doc: dict(doc, schema='unknown'))):
            with self.assertRaises(ValueError):
                decode_native_project(bad)
        stream = io.BytesIO()
        np.savez_compressed(stream, joints=clip.joints, metadata=np.array({'schema': 'cast'}, dtype=object))
        with self.assertRaises(ValueError):
            decode_native_project(stream.getvalue())


class SharedPlaybackTests(unittest.TestCase):
    def setUp(self):
        self.server = Server()
        self.server.scene = CountingScene([])
        self.paired = NativePairSession()
        self.cast = CastPerformanceSession()
        self.addCleanup(self.paired.close)
        self.addCleanup(self.cast.close)
        self.paired.load_source(NativePairClip(performance(2).joints))
        self.cast.load_performance(performance(3))
        self.pair_renderer = NativePairRenderer(self.server)
        self.cast_renderer = CastPerformanceRenderer(self.server)
        self.addCleanup(self.pair_renderer.remove)
        self.addCleanup(self.cast_renderer.remove)
        self.router = NativePlaybackRouter(self.server,
            {'paired': (self.paired, self.pair_renderer), 'cast': (self.cast, self.cast_renderer)},
            lock=threading.RLock())

    def test_switch_back_republishes_retained_clip_with_one_protocol_controller(self):
        handler = self.server.connected
        self.router.select('paired')
        pair_clip = self.paired.timeline_clip()
        self.paired.seek(5)
        self.router.refresh()
        self.router.select('cast')
        self.assertIsNone(self.pair_renderer.local_playback)
        self.assertIs(self.cast_renderer.local_playback, self.router.controller)
        self.assertEqual(len(self.server.clients[1].messages[-2].actors), 3)
        self.cast.seek(6)
        self.router.refresh()
        self.router.select('paired')
        self.assertIsNone(self.cast_renderer.local_playback)
        self.assertIs(self.paired.timeline_clip(), pair_clip)
        messages = self.server.clients[1].messages
        self.assertIsInstance(messages[-2], NativePairClipMessage)
        self.assertEqual(len(messages[-2].actors), 2)
        self.assertIsInstance(messages[-1], NativePairTransportMessage)
        self.assertEqual(messages[-1].frame, 5)
        self.assertIs(self.server.connected, handler)
        self.assertEqual(len(self.server.handlers), 1)
        self.assertEqual(self.cast.snapshot()['frame'], 6)

    def test_inactive_renderer_cannot_replace_selected_clip_and_reconnect_routes_cast(self):
        self.router.select('cast')
        revision = self.router.controller.revision
        self.paired.add_actor('Sideline')
        self.pair_renderer.sync_cast(self.paired.snapshot())
        self.assertEqual(self.router.controller.revision, revision)
        self.cast.seek(4)
        self.router.refresh()
        late = Client(3)
        self.server.clients[3] = late
        self.server.connected(late)
        self.assertEqual(len(late.messages[0].actors), 3)
        self.assertEqual(late.messages[1].frame, 4)
        self.assertTrue(late.messages[1].enabled)
        self.router.select(None)
        later = Client(4)
        self.server.clients[4] = later
        self.server.connected(later)
        self.assertEqual(later.messages, [])
        self.assertEqual(self.router.controller.payload_bytes, 0)

    def test_completed_take_actor_count_replacement_and_capture_keep_exact_frame(self):
        self.router.select('cast')
        self.cast.load_performance(performance(1, frames=4))
        self.cast.seek(3)
        self.router.refresh()
        self.assertEqual(len(self.server.clients[1].messages[-2].actors), 1)
        self.assertEqual(self.server.clients[1].messages[-1].frame, 3)
        self.cast.begin_capture()
        self.cast.capture_seek(1)
        self.router.refresh()
        self.assertTrue(self.server.clients[1].messages[-1].capturing)
        self.assertEqual(self.server.clients[1].messages[-1].frame, 1)
        self.cast.end_capture(3)
        self.router.refresh()
        self.assertFalse(self.server.clients[1].messages[-1].capturing)
        self.assertEqual(self.server.clients[1].messages[-1].frame, 3)


if __name__ == '__main__':
    unittest.main()
