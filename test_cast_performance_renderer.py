"""Every cast track remains in source world coordinates with local playback."""
import unittest

import numpy as np

from cast_performance import CastPerformance, cast_from_performance
from cast_performance_renderer import CastPerformanceRenderer
from native_pair_clip import NativePairClip
from native_pair_playback import NativePairClipMessage, NativePairPlaybackController
from test_cast_performance import performance
from test_native_pair_local_integration import CountingScene
from test_native_pair_playback import Client, Server


class CastPerformanceRendererTests(unittest.TestCase):
    def renderer(self, clip, *, local=False):
        server = Server()
        server.scene = CountingScene([])
        renderer = CastPerformanceRenderer(server)
        self.addCleanup(renderer.remove)
        state = {'frame': 0, 'playing': True, 'capturing': False, 'transport_revision': 1}
        if local:
            renderer.local_playback = NativePairPlaybackController(server, get_state=lambda: dict(state))
        renderer.sync_cast({'cast': cast_from_performance(clip), 'actor_ids': list(clip.actor_ids)})
        renderer.set_clip(clip)
        return renderer, server, state

    def test_one_two_and_three_tracks_render_exact_world_positions_without_retargeting(self):
        for count in (1, 2, 3):
            with self.subTest(count=count):
                clip = performance(count, offset=7.)
                renderer, _, _ = self.renderer(clip)
                renderer.tick(5)
                for index, identifier in enumerate(clip.actor_ids):
                    np.testing.assert_array_equal(renderer.actors[identifier].joint_positions, clip.joints[5, index])
                    np.testing.assert_array_equal(renderer.actor_root(identifier), clip.joints[5, index, 0])
                np.testing.assert_array_equal(renderer.actor_root(), clip.joints[5, :, 0].mean(axis=0))

    def test_local_playback_for_each_cast_size_sends_one_packet_and_no_tick_vertex_updates(self):
        for count in (1, 2, 3):
            with self.subTest(count=count):
                clip = performance(count)
                renderer, server, state = self.renderer(clip, local=True)
                controller = renderer.local_playback
                controller.update(state, True)
                counts = [mesh.vertex_updates for mesh in server.scene.meshes]
                for frame in range(1, clip.frames):
                    state['frame'] = frame
                    renderer.tick(frame)
                    self.assertFalse(controller.update(state, True))
                self.assertEqual([mesh.vertex_updates for mesh in server.scene.meshes], counts)
                packet = server.clients[1].messages[0]
                self.assertIsInstance(packet, NativePairClipMessage)
                self.assertEqual(len(packet.actors), count)
                self.assertEqual(len(server.clients[1].messages), 2)
                for index, identifier in enumerate(clip.actor_ids):
                    targets = np.frombuffer(packet.actors[index]['targets'], '<f4').reshape(clip.frames, 22, 3)
                    np.testing.assert_array_equal(targets, clip.joints[:, index].astype(np.float32))
                    np.testing.assert_array_equal(renderer.actor_root(identifier), clip.joints[-1, index, 0])

    def test_replacing_three_actors_with_one_releases_old_meshes_and_cached_motion(self):
        renderer, server, state = self.renderer(performance(3), local=True)
        renderer.local_playback.update(state, True)
        removed = [handle for identifier in ('performer_2', 'performer_3')
                   for handle in renderer.actors[identifier].handles]
        replacement = performance(1, offset=-4.)
        renderer.sync_cast({'cast': cast_from_performance(replacement), 'actor_ids': list(replacement.actor_ids)})
        self.assertTrue(all(handle.removed for handle in removed))
        self.assertEqual(renderer.local_playback.payload_bytes, 0)
        renderer.set_clip(replacement)
        renderer.local_playback.update(state, True)
        self.assertEqual(set(renderer.actors), {'performer_1'})
        packet = [message for message in server.clients[1].messages if isinstance(message, NativePairClipMessage)][-1]
        self.assertEqual(len(packet.actors), 1)
        renderer.remove()
        self.assertFalse(server.clients[1].messages[-1].enabled)
        self.assertEqual(renderer.local_playback.payload_bytes, 0)
        late = Client(3)
        server.connected(late)
        self.assertEqual(late.messages, [])

    def test_failed_pose_preparation_preserves_previous_actor_caches(self):
        original = performance(2)
        renderer, _, _ = self.renderer(original)
        previous = {identifier: actor._prepared for identifier, actor in renderer.actors.items()}
        invalid = original.joints.copy()
        invalid[:, 1] = 0  # Second track fails after the first actor was prepared.
        with self.assertRaises(ValueError):
            renderer.set_clip(CastPerformance(original.actor_ids, invalid))
        for identifier, actor in renderer.actors.items():
            self.assertIs(actor._prepared, previous[identifier])
        renderer.tick(6)
        for index, identifier in enumerate(original.actor_ids):
            np.testing.assert_array_equal(renderer.actors[identifier].joint_positions, original.joints[6, index])

    def test_renderer_rejects_pair_format_and_track_identity_mismatch(self):
        clip = performance(2)
        renderer, _, _ = self.renderer(clip)
        with self.assertRaises(ValueError):
            renderer.set_clip(NativePairClip(clip.joints))
        with self.assertRaisesRegex(ValueError, 'exact performance cast'):
            renderer.set_clip(CastPerformance(tuple(reversed(clip.actor_ids)), clip.joints))
        renderer.set_visible(True)
        self.assertTrue(all(actor.visible for actor in renderer.actors.values()))


if __name__ == '__main__':
    unittest.main()
