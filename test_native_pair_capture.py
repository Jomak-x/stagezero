"""Capture failure must release the native session lease and restore transport."""
from pathlib import Path
import tempfile
import threading
import io
import numpy as np
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from native_pair_capture import capture_pair
from native_pair_clip import NativePairClip
from native_pair_session import NativePairSession


class FakeRenderer:
    def sync_cast(self, snapshot):
        pass

    def set_clip(self, clip):
        pass

    def tick(self, frame):
        pass


class FakeEncoder:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0


class NativePairCaptureTests(unittest.TestCase):
    def setUp(self):
        self.session = NativePairSession()
        self.session.load_reviewed_handshake()
        self.session.seek(42)
        self.addCleanup(self.session.close)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def test_encoder_start_failure_releases_lease(self):
        with patch('native_pair_capture.subprocess.Popen', side_effect=OSError('ffmpeg unavailable')):
            with self.assertRaisesRegex(OSError, 'ffmpeg unavailable'):
                capture_pair(self.session, FakeRenderer(), object(), Path(self.folder.name) / 'export')
        state = self.session.snapshot()
        self.assertFalse(state['capturing'])
        self.assertEqual(state['frame'], 42)
        self.session.play()

    def test_transport_blocked_during_capture_and_released_after_failure(self):
        encoder = FakeEncoder()

        def inspect_frame(*args, **kwargs):
            self.assertTrue(self.session.snapshot()['capturing'])
            with self.assertRaisesRegex(RuntimeError, 'export is running'):
                self.session.seek(1)
            with self.assertRaisesRegex(RuntimeError, 'export is running'):
                self.session.add_actor('Unexpected')
            raise RuntimeError('render failed')

        with patch('native_pair_capture.subprocess.Popen', return_value=encoder), \
             patch('native_pair_capture.get_render_with_timeout', side_effect=inspect_frame):
            with self.assertRaisesRegex(RuntimeError, 'render failed'):
                capture_pair(self.session, FakeRenderer(), object(), Path(self.folder.name) / 'export')
        self.assertTrue(encoder.terminated)
        self.assertFalse(self.session.snapshot()['capturing'])
        self.assertEqual(self.session.snapshot()['frame'], 42)
        self.assertTrue(self.session.add_actor('After export').startswith('actor_'))

    def test_capture_acquisition_waits_for_live_scene_transaction(self):
        entered = threading.Event()
        mutex = threading.Lock()

        class RenderLock:
            def __enter__(self):
                entered.set()
                mutex.acquire()
            def __exit__(self, *args):
                mutex.release()

        errors = []
        def capture():
            try:
                capture_pair(self.session, FakeRenderer(), object(),
                             Path(self.folder.name)/'atomic', render_lock=RenderLock())
            except Exception as exc:
                errors.append(exc)

        mutex.acquire()
        with patch('native_pair_capture._capture_locked', return_value='done'):
            worker = threading.Thread(target=capture)
            worker.start()
            try:
                self.assertTrue(entered.wait(2), 'Capture must acquire the render transaction lock')
                self.assertFalse(self.session.snapshot()['capturing'])
                # This must remain legal until the live transaction releases its lock.
                self.session.update_scene(self.session.scene_document)
            finally:
                mutex.release()
                worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertFalse(self.session.snapshot()['capturing'])

    def test_every_frame_updates_scene_under_lock_before_flush_and_render(self):
        self.session.load_source(NativePairClip(self.session.timeline_clip().joints[:4]))
        self.session.seek(2)
        mutex = threading.Lock()
        events = []
        renderer = FakeRenderer()
        renderer.tick = lambda frame: events.append(('actor', frame))
        encoder = FakeEncoder()
        encoder.stdin = io.BytesIO()
        encoder.poll = lambda: 0

        def scene(frame, state):
            self.assertTrue(mutex.locked())
            self.assertEqual(self.session.snapshot()['frame'], frame)
            self.assertEqual(events[-1], ('actor', frame))
            events.append(('scene', frame))

        def flush():
            self.assertTrue(mutex.locked())
            self.assertEqual(events[-1][0], 'scene')
            events.append(('flush', self.session.snapshot()['frame']))

        def render(*args, **kwargs):
            frame = self.session.snapshot()['frame']
            self.assertTrue(mutex.locked())
            self.assertEqual(events[-3:], [('actor', frame), ('scene', frame), ('flush', frame)])
            events.append(('render', frame))
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        with patch('native_pair_capture.subprocess.Popen', return_value=encoder), \
             patch('native_pair_capture.get_render_with_timeout', side_effect=render):
            capture_pair(self.session, renderer, object(), Path(self.folder.name)/'ordered',
                         render_lock=mutex, render_frame=scene, flush=flush)
        self.assertEqual([frame for event, frame in events if event == 'render'], list(range(4)))
        self.assertEqual(self.session.snapshot()['frame'], 2)
        self.assertFalse(self.session.snapshot()['capturing'])
        self.assertFalse(mutex.locked())

    def test_scene_callback_failure_releases_capture_and_restores_transport(self):
        mutex = threading.Lock()
        encoder = FakeEncoder()
        def failed_scene(*args):
            self.assertTrue(mutex.locked())
            raise RuntimeError('scene update failed')
        with patch('native_pair_capture.subprocess.Popen', return_value=encoder), \
             patch('native_pair_capture.get_render_with_timeout') as render:
            with self.assertRaisesRegex(RuntimeError, 'scene update failed'):
                capture_pair(self.session, FakeRenderer(), object(), Path(self.folder.name)/'failed-scene',
                             render_lock=mutex, render_frame=failed_scene)
        render.assert_not_called()
        self.assertFalse(mutex.locked())
        self.assertFalse(self.session.snapshot()['capturing'])
        self.assertEqual(self.session.snapshot()['frame'], 42)

    def test_local_capture_requires_current_revision_for_the_exporting_client(self):
        client = SimpleNamespace(client_id=7)
        events = []
        # This tab acknowledged the previous take; another tab loaded the new one.
        acknowledgements = {7: 6, 8: 7}
        playback = SimpleNamespace(revision=6)
        renderer = FakeRenderer()
        renderer.local_playback = playback

        def load_clip(clip):
            self.assertIs(clip, self.session.timeline_clip())
            playback.revision = 7
            events.append(('load', playback.revision))

        def require_ready(requested_client):
            self.assertIs(requested_client, client)
            self.assertEqual(events, [('load', 7)])
            events.append(('ready', requested_client.client_id, playback.revision))
            if acknowledgements.get(requested_client.client_id) != playback.revision:
                raise RuntimeError('Reload this browser before exporting the current take')

        renderer.set_clip = load_clip
        playback.require_ready = require_ready
        with patch('native_pair_capture.subprocess.Popen') as encoder, \
             patch('native_pair_capture.get_render_with_timeout') as render:
            with self.assertRaisesRegex(RuntimeError, 'Reload this browser'):
                capture_pair(self.session, renderer, client, Path(self.folder.name)/'unready')
        encoder.assert_not_called()
        render.assert_not_called()
        self.assertEqual(events, [('load', 7), ('ready', 7, 7)])
        self.assertFalse((Path(self.folder.name)/'unready'/'scene.native-pair.stagezero.npz').exists())
        self.assertFalse(self.session.snapshot()['capturing'])
        self.assertEqual(self.session.snapshot()['frame'], 42)
        self.session.play()  # A rejected export must not retain its capture lease.

    def test_local_capture_publishes_exact_frames_without_optional_scene_callback(self):
        self.session.load_source(NativePairClip(self.session.timeline_clip().joints[:4]))
        self.session.seek(2)
        client = SimpleNamespace(client_id=7)
        mutex = threading.Lock()
        events = []
        commands = []
        renderer = FakeRenderer()
        renderer.set_clip = lambda clip: events.append(('load', clip.frames))
        renderer.tick = lambda frame: events.append(('actor', frame))

        def require_ready(requested_client):
            self.assertIs(requested_client, client)
            self.assertEqual(events, [('load', 4)])
            events.append(('ready', client.client_id))

        def update(state, enabled):
            self.assertTrue(mutex.locked())
            self.assertTrue(enabled)
            self.assertTrue(state['capturing'])
            self.assertFalse(state['playing'])
            self.assertEqual(events[-1], ('actor', state['frame']))
            commands.append(state)
            events.append(('transport', state['frame']))

        renderer.local_playback = SimpleNamespace(require_ready=require_ready, update=update)
        encoder = FakeEncoder()
        encoder.stdin = io.BytesIO()
        encoder.poll = lambda: 0

        def flush():
            self.assertTrue(mutex.locked())
            frame = self.session.snapshot()['frame']
            self.assertEqual(events[-1], ('transport', frame))
            events.append(('flush', frame))

        def render(requested_client, **kwargs):
            self.assertIs(requested_client, client)
            self.assertTrue(mutex.locked())
            frame = self.session.snapshot()['frame']
            self.assertEqual(events[-3:], [('actor', frame), ('transport', frame), ('flush', frame)])
            events.append(('render', frame))
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        with patch('native_pair_capture.subprocess.Popen', return_value=encoder), \
             patch('native_pair_capture.get_render_with_timeout', side_effect=render):
            # Deliberately omit render_frame: transport is mandatory capture work.
            capture_pair(self.session, renderer, client, Path(self.folder.name)/'local',
                         render_lock=mutex, flush=flush)
        self.assertEqual([state['frame'] for state in commands], list(range(4)))
        revisions = [state['transport_revision'] for state in commands]
        self.assertTrue(all(left < right for left, right in zip(revisions, revisions[1:])))
        self.assertEqual([frame for event, frame in events if event == 'render'], list(range(4)))
        self.assertFalse(mutex.locked())
        self.assertFalse(self.session.snapshot()['capturing'])
        self.assertEqual(self.session.snapshot()['frame'], 2)

    def test_camera_fit_includes_placed_full_trajectory_and_cast(self):
        from director_viewer import native_cast_camera_view
        clip = NativePairClip(self.session.timeline_clip().joints[:4])
        joints = clip.joints.copy()
        joints[0, ..., 0] -= 8
        joints[-1, ..., 2] += 9
        clip = NativePairClip(joints)
        placement = {'x': 14., 'z': -7., 'yaw_degrees': 90.}
        roots = np.array([[5., 1., -18.], [19., 1., 1.]])
        position, target, fov = native_cast_camera_view(clip, placement, roots, aspect=.8)
        world = joints.reshape(-1, 3) @ np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
        world += [14., 0., -7.]
        points = np.concatenate([world, roots + [0., -1.1, 0.], roots + [0., 1.3, 0.]])
        forward = target-position; forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0., 1., 0.]); right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        relative = points-position
        depth = relative @ forward
        self.assertTrue(np.all(depth > 0))
        self.assertTrue(np.all(np.abs(relative @ right) < depth*np.tan(fov/2)*.8))
        self.assertTrue(np.all(np.abs(relative @ up) < depth*np.tan(fov/2)))


if __name__ == '__main__':
    unittest.main()
