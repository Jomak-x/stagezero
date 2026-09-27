"""CPU contracts for rigid initial observer staging before scene validation."""
import inspect
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from cast_motion_refinement import refine_cast_motion
from experiments.native_pair_rig import NativeRigAsset
from native_pair_clip import NativePairClip
from paired_meetup import _heading
from prompt_scene_builder import check_cast_geometry, select_initial_staging
from test_native_pair_rig import fixture_glb
from test_prompt_scene_builder import SCENE


ACTIVE = ['actor_1', 'actor_2']
IDLE = 'actor_3'


class InitialObserverStagingTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        source = Path(folder.name) / 'fixture.glb'
        source.write_bytes(fixture_glb()[0])
        rest = NativeRigAsset(source).rest
        self.pair = NativePairClip(
            np.repeat(np.stack([rest, rest + [3, 0, 0]])[None], 30, axis=0),
            metadata={'model': 'InterGen'})
        self.idle_pose = rest.copy() + [6, 0, 0]
        self.plan = {'actors': [{'id': aid, 'start': None} for aid in (*ACTIVE, IDLE)]}
        self.placement = {'starts': {
            ACTIVE[0]: {'x': 0., 'z': 0.}, ACTIVE[1]: {'x': 3., 'z': 0.},
            IDLE: {'x': 6., 'z': 0.}},
            'meeting': {'x': 1.5, 'z': 0., 'yaw_degrees': 0.}, 'target_id': None}

    def stage(self, **kwargs):
        return select_initial_staging(self.pair, SCENE, self.plan, self.placement,
                                      {IDLE: self.idle_pose}, ACTIVE, **kwargs)

    def test_automatic_observer_faces_meeting_and_preserves_source_anatomy(self):
        source_pair = self.pair.joints.copy()
        source_idle = self.idle_pose.copy()
        original_placement = {aid: dict(start) for aid, start in self.placement['starts'].items()}
        with patch('prompt_scene_builder.check_cast_geometry', wraps=check_cast_geometry) as geometry:
            selected, poses, planning_scene, proxies = self.stage()
        staged = poses[IDLE]
        target = selected['meeting']
        direction = np.array([target['x'] - staged[0, 0], target['z'] - staged[0, 2]])
        expected_heading = math.atan2(direction[0], direction[1])
        heading_error = math.atan2(math.sin(_heading(staged)-expected_heading),
                                   math.cos(_heading(staged)-expected_heading))
        self.assertAlmostEqual(heading_error, 0., places=10)
        np.testing.assert_allclose(staged[0, [0, 2]],
                                   [selected['starts'][IDLE]['x'], selected['starts'][IDLE]['z']])
        np.testing.assert_allclose(staged[:, 1], source_idle[:, 1], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(staged-staged[0], axis=1),
                                   np.linalg.norm(source_idle-source_idle[0], axis=1), atol=1e-12)
        self.assertIn(IDLE, selected['initial_source_staging']['observer_orientation'])
        self.assertTrue(any(np.array_equal(call.args[0][:, 2],
                                           np.repeat(staged[None], self.pair.frames, axis=0))
                            for call in geometry.call_args_list))
        self.assertEqual(proxies[0]['root_xz'], staged[0, [0, 2]].tolist())
        self.assertEqual(len(planning_scene['objects']), 1)
        self.assertEqual(SCENE['objects'], [])
        np.testing.assert_array_equal(self.pair.joints, source_pair)
        np.testing.assert_array_equal(self.idle_pose, source_idle)
        self.assertEqual(self.placement['starts'], original_placement)

    def test_explicit_yaw_prevents_automatic_observer_rotation(self):
        self.placement['starts'][IDLE]['yaw_degrees'] = 55.
        self.plan['actors'][-1]['start'] = {'x': 6., 'z': 0.}
        self.plan['actors'][-1]['start_yaw_degrees'] = 55.
        yaw = math.radians(55.)
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
        origin = self.idle_pose[0].copy()
        self.idle_pose = (self.idle_pose-origin) @ rotation.T + origin
        self.assertAlmostEqual(math.degrees(_heading(self.idle_pose)), 55.)
        selected, poses, _, _ = self.stage()
        np.testing.assert_array_equal(poses[IDLE], self.idle_pose)
        self.assertEqual(selected['starts'][IDLE], self.placement['starts'][IDLE])
        self.assertIn(IDLE, selected['initial_source_staging']['explicit_starts_preserved'])
        self.assertNotIn(IDLE, selected['initial_source_staging']['observer_orientation'])

    def test_refinement_defaults_to_no_foot_planting(self):
        self.assertIs(inspect.signature(refine_cast_motion).parameters['feet'].default, False)


if __name__ == '__main__':
    unittest.main()
