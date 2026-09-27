"""Camera visibility is descriptive ray/bounds geometry, not a physics claim."""
import unittest

import numpy as np

from director_viewer import native_cast_camera_view
from interaction_scene import SceneObject
from paired_scene import EMPTY_SCENE
from prompt_scene_camera import (_blocked_rays, _cast_blocked_rays, _framing_corners, _sample_frames,
                                 MAX_SAMPLED_FRAMES, ORBIT_DEGREES, VISIBILITY_JOINTS,
                                 prompt_scene_camera_view)
from test_cast_performance import performance


class PromptSceneCameraTests(unittest.TestCase):
    def scene(self, camera, target, *, beyond=False):
        point = target + (target - camera) * .5 if beyond else (camera + target) / 2
        return dict(EMPTY_SCENE, objects=[{'id': 'lamp', 'name': 'Lamp', 'kind': 'lamp',
            'position': [float(point[0]), 2., float(point[2])], 'size': [.6, 4., .6],
            'color': [200, 190, 170], 'yaw': 0.}])

    def baseline(self, clip, aspect=16/9):
        return native_cast_camera_view(clip, {'x': 0., 'z': 0., 'yaw_degrees': 0.},
                                       clip.joints[0, :, 0], aspect=aspect)

    def test_empty_scene_preserves_existing_fit_exactly(self):
        clip = performance(3)
        expected = self.baseline(clip)
        actual = prompt_scene_camera_view(clip, EMPTY_SCENE)
        for original, chosen in zip(expected, actual):
            np.testing.assert_array_equal(chosen, original)

    def test_lamp_blocking_start_moves_orbit_without_changing_fov_center_or_elevation(self):
        clip = performance(1)
        original, center, fov = self.baseline(clip)
        scene = self.scene(original, clip.joints[0, 0, 9])
        chosen, target, selected_fov = prompt_scene_camera_view(clip, scene)
        self.assertFalse(np.allclose(chosen, original))
        np.testing.assert_array_equal(target, center)
        self.assertEqual(selected_fov, fov)
        first_direction, chosen_direction = original - center, chosen - center
        self.assertAlmostEqual(first_direction[1] / np.linalg.norm(first_direction),
                               chosen_direction[1] / np.linalg.norm(chosen_direction), places=12)
        from interaction_scene import scene_objects
        rays = clip.joints[0, 0, [0, 9, 15]]
        self.assertTrue(_blocked_rays(original, rays, scene_objects(scene)).any())
        self.assertFalse(_blocked_rays(chosen, rays, scene_objects(scene)).any())
        self.assertGreaterEqual(np.linalg.norm(chosen - center), np.linalg.norm(original - center) - 1e-12)

    def test_objects_behind_actor_and_floor_do_not_change_clear_original_view(self):
        clip = performance(1)
        original, center, fov = self.baseline(clip)
        scene = self.scene(original, clip.joints[0, 0, 9], beyond=True)
        scene['objects'].append({'id': 'floor', 'name': 'Floor', 'kind': 'platform',
            'position': [0., -.1, 0.], 'size': [20., .2, 20.], 'color': [100, 100, 100], 'yaw': 0.})
        chosen, _, _ = prompt_scene_camera_view(clip, scene)
        np.testing.assert_array_equal(chosen, original)

    def test_rotated_thin_bounds_use_authored_yaw_and_segment_endpoints(self):
        box = SceneObject('wall', 'Wall', 'custom', 0., 1., 0., 4., 2., .1, 90.)
        camera = np.array([2., 1., 0.])
        targets = np.array([[-2., 1., 0.], [3., 1., 0.], [2., 1., 4.]])
        np.testing.assert_array_equal(_blocked_rays(camera, targets, [box]), [True, False, False])
        # Parallel rays outside the box must not hit through division by zero.
        np.testing.assert_array_equal(_blocked_rays(np.array([2., 4., 0.]), targets + [0, 3, 0], [box]),
                                      [False, False, False])

    def test_late_foot_obstruction_changes_view_even_when_every_torso_ray_is_clear(self):
        from cast_performance import CastPerformance
        from interaction_scene import scene_objects
        source = performance(1, frames=40)
        joints = source.joints.copy()
        joints[..., 2] += np.linspace(0., 4., source.frames)[:, None, None]
        clip = CastPerformance(source.actor_ids, joints)
        original, center, fov = self.baseline(clip)
        # A small foreground prop crosses a late toe ray while staying below
        # every hip/chest/head sight line. Torso-only scoring misses this case.
        blocker = original * .15 + clip.joints[-1, 0, 10] * .85
        scene = dict(EMPTY_SCENE, objects=[{'id': 'foreground', 'name': 'Low foreground prop',
            'kind': 'crate', 'position': blocker.tolist(), 'size': [.3, .2, .3],
            'color': [160, 140, 100], 'yaw': 0.}])
        objects = scene_objects(scene)
        upper = clip.joints[:, :, [0, 9, 15], :].reshape(-1, 3)
        feet = clip.joints[-1, :, [7, 8, 10, 11], :].reshape(-1, 3)
        self.assertFalse(_blocked_rays(original, upper, objects).any())
        self.assertTrue(_blocked_rays(original, feet, objects).any())
        camera, chosen_center, chosen_fov = prompt_scene_camera_view(clip, scene)
        self.assertFalse(np.allclose(camera, original))
        self.assertFalse(_blocked_rays(camera, upper, objects).any())
        self.assertFalse(_blocked_rays(camera, feet, objects).any())
        all_rays = clip.joints[:, :, VISIBILITY_JOINTS, :].reshape(-1, 3)
        self.assertFalse(_blocked_rays(camera, all_rays, objects).any())
        np.testing.assert_array_equal(chosen_center, center)
        self.assertEqual(chosen_fov, fov)

    def test_orbit_keeps_whole_wide_cast_inside_portrait_frustum(self):
        clip = performance(3)
        aspect = .65
        original, center, fov = self.baseline(clip, aspect)
        scene = self.scene(original, clip.joints[0, 1, 9])
        camera, center, fov = prompt_scene_camera_view(clip, scene, aspect=aspect)
        direction = camera - center
        direction /= np.linalg.norm(direction)
        right = np.cross([0., 1., 0.], direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        corners = _framing_corners(clip, clip.joints[0, :, 0], center)
        depth = np.linalg.norm(camera - center) - corners @ direction
        self.assertTrue(np.all(depth > 0))
        self.assertTrue(np.all(np.abs(corners @ right) <= depth * np.tan(fov / 2) * aspect))
        self.assertTrue(np.all(np.abs(corners @ up) <= depth * np.tan(fov / 2)))

    def test_sampling_is_bounded_and_always_includes_start_end_and_current_frame(self):
        samples = _sample_frames(1000, 391)
        self.assertLessEqual(len(samples), MAX_SAMPLED_FRAMES)
        self.assertTrue({0, 391, 999} <= set(samples))
        clip = performance(1)
        with self.assertRaises(ValueError):
            prompt_scene_camera_view(clip, EMPTY_SCENE, frame=clip.frames)

    def test_cast_occlusion_excludes_self_and_actors_behind_targets_at_each_frame(self):
        solo = performance(1, frames=4)
        camera = np.array([0., 1.3, 5.])
        self.assertFalse(_cast_blocked_rays(camera, solo.joints).any())
        joints = np.repeat(solo.joints, 2, axis=1)
        joints[:2, 1, :, 2] += 2.
        joints[2:, 1, :, 2] -= 2.
        blocked = _cast_blocked_rays(camera, joints)
        self.assertTrue(blocked[:2, 0, :2].any())
        self.assertFalse(blocked[:2, 1].any())
        self.assertTrue(blocked[2:, 1, :2].any())
        self.assertFalse(blocked[2:, 0].any())

    def test_shared_view_avoids_foreground_cast_without_changing_actor_tracks(self):
        from cast_performance import CastPerformance
        source = performance(1)
        joints = np.repeat(source.joints, 2, axis=1)
        direction = np.array([.95, 0., .8])
        direction /= np.linalg.norm(direction)
        joints[:, 1] += direction * 1.3
        clip = CastPerformance(('rear', 'front'), joints)
        before = clip.joints.tobytes()
        original, center, fov = self.baseline(clip)
        original_blocked = _cast_blocked_rays(original, clip.joints)
        self.assertTrue(original_blocked.any())
        camera, chosen_center, chosen_fov = prompt_scene_camera_view(clip, EMPTY_SCENE)
        self.assertLess(_cast_blocked_rays(camera, clip.joints).sum(), original_blocked.sum())
        np.testing.assert_array_equal(chosen_center, center)
        self.assertEqual(chosen_fov, fov)
        self.assertEqual(clip.joints.tobytes(), before)
        self.assertEqual(len(ORBIT_DEGREES), 36)


if __name__ == '__main__':
    unittest.main()
