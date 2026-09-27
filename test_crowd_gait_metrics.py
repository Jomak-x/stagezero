import unittest

import numpy as np

from crowd_gait_metrics import evaluate, foot_metrics, rendered_feet, source_loop_metrics


class CrowdGaitMetricTests(unittest.TestCase):
    def inputs(self):
        manifest = {'totalFrames': 6, 'clips': [
            {'id': 'walk', 'frames': 4, 'offset': 0, 'fps': 30., 'strideMeters': 1.},
            {'id': 'idle', 'frames': 2, 'offset': 4, 'fps': 30., 'strideMeters': 0.}]}
        poses = np.zeros((6, 22, 3))
        trajectory = {'dt': .5, 'frames': [
            {'t': 0., 'people': [[0., 0., 0., 1.]]},
            {'t': .5, 'people': [[0., .24, 0., 1.]]},
            {'t': 1., 'people': [[0., .48, 0., 1.]]}]}
        return manifest, poses, trajectory

    def test_sliding_is_not_excluded_by_horizontal_speed(self):
        times = np.arange(11)*.1
        feet = np.zeros((11, 1, 4, 3))
        feet[..., 0] = times[:, None, None]*2
        result = foot_metrics(times, feet, np.ones((11, 1)))
        self.assertAlmostEqual(result['proxy_world_xz_speed_mps']['mean'], 2.)
        self.assertEqual(result['proxy_world_xz_speed_mps']['samples'], 40)
        self.assertAlmostEqual(result['proxy_bout_endpoint_drift_m']['max'], 2.)

    def test_airborne_feet_are_excluded_and_absence_is_null(self):
        feet = np.ones((3, 1, 4, 3))
        result = foot_metrics(np.arange(3)*.1, feet, np.ones((3, 1)))
        self.assertIsNone(result['proxy_world_xz_speed_mps']['mean'])
        self.assertEqual(result['proxy_world_xz_speed_mps']['samples'], 0)

    def test_phase_scales_stride_with_actor_xz_size(self):
        manifest, poses, trajectory = self.inputs()
        # Actor 0 at distance .24: .24 / (.96*1) * 4 = frame 1 exactly.
        poses[:4, :, 0] = np.arange(4)[:, None]
        times, feet, _ = rendered_feet(manifest, poses, trajectory, sample_fps=2.)
        self.assertEqual(times.tolist(), [0., .5, 1.])
        self.assertAlmostEqual(feet[1, 0, 0, 0], .96)
        self.assertAlmostEqual(feet[1, 0, 0, 2], .24)

    def test_yaw_shortest_arc_and_idle_selection(self):
        manifest, poses, trajectory = self.inputs()
        poses[4:, :, 2] = 1.
        trajectory['frames'][0]['people'][0][2:] = [np.pi-.1, 0.]
        trajectory['frames'][1]['people'][0][2:] = [-np.pi+.1, 0.]
        trajectory['frames'][2]['people'][0][2:] = [-np.pi+.1, 0.]
        _, feet, blend = rendered_feet(manifest, poses, trajectory, sample_fps=4.)
        self.assertAlmostEqual(feet[1, 0, 0, 0], 0., places=7)
        self.assertAlmostEqual(feet[1, 0, 0, 2], -.96+.12)
        self.assertEqual(blend.max(), 0.)

    def test_source_loop_discontinuity_is_not_hidden(self):
        manifest, poses, _ = self.inputs()
        poses[3, :, 0] = .3
        result = source_loop_metrics(manifest, poses)
        self.assertAlmostEqual(result[0]['loop_endpoint_discontinuity_rms_m'], .3)
        self.assertAlmostEqual(result[0]['loop_edge_speed_at_nominal_fps_mps']['max'], 9.)

    def test_gait_uses_style_and_excludes_start_stop_clips(self):
        manifest, poses, trajectory = self.inputs()
        manifest['clips'][0]['id'] = 'relaxed-walk'
        manifest['clips'].insert(0, dict(manifest['clips'][0], id='start', type='start'))
        manifest['clips'].append(dict(manifest['clips'][1], id='brisk-walk', offset=2, frames=2))
        poses[2:4, :, 0] = 3.
        trajectory['agents'] = [{'gait': 'brisk'}]
        _, feet, _ = rendered_feet(manifest, poses, trajectory, sample_fps=2.)
        self.assertAlmostEqual(feet[0, 0, 0, 0], 3*.96)
        trajectory['agents'][0]['gait'] = 'casual'
        _, feet, _ = rendered_feet(manifest, poses, trajectory, sample_fps=2.)
        self.assertEqual(feet[0, 0, 0, 0], 0.)

    def test_world_still_feet_and_transition_subsets(self):
        manifest, poses, trajectory = self.inputs()
        for frame in trajectory['frames']:
            frame['people'][0] = [0., 0., 0., .2]
        result = evaluate(manifest, poses, trajectory, sample_fps=4.)
        self.assertEqual(result['metrics']['proxy_world_xz_speed_mps']['max'], 0.)
        self.assertEqual(result['metrics']['transition_proxy_world_xz_speed_mps']['samples'], 16)

    def test_abrupt_speed_drop_has_bounded_deterministic_blend(self):
        manifest, poses, trajectory = self.inputs()
        trajectory['frames'][1]['people'][0][3] = 0.
        trajectory['frames'][2]['people'][0][3] = 0.
        _, _, blend = rendered_feet(manifest, poses, trajectory, sample_fps=4.)
        np.testing.assert_allclose(blend[:, 0], [1., .55, .1, .05, 0.], atol=1e-12)

    def test_invalid_input_fails(self):
        manifest, poses, trajectory = self.inputs()
        poses[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            rendered_feet(manifest, poses, trajectory)


if __name__ == '__main__':
    unittest.main()
