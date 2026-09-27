"""Native terrain precommit checks against authored support and saved failures."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from core_terrain_validation import validate_terrain_clip
from motion_bridge import _layout
from scene_objects import make_object
from studio_core_session import CoreStudioSession


ROOT = Path(__file__).resolve().parent
SAVED = ROOT / 'review/spatial-v2/terrain-native-localfloor'


def _scene(*objects):
    return {'version': 2, 'name': 'Terrain validation', 'objects': list(objects),
            'effects': [], 'lighting': 'neutral'}


def _platform(floor=2.):
    obj = make_object('platform', 0)
    obj['size'] = [8., .3, 8.]
    obj['position'] = [0., floor - .15, 0.]
    return obj


class TerrainValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names, _, cls.neutral = _layout()
        indices = {name: index for index, name in enumerate(names)}
        toes = [indices['LeftToeBase'], indices['RightToeBase']]
        cls.clearance = float(-np.mean(cls.neutral[toes, 1]))

    def neutral_clip(self, floor=2., frames=1, *, x=0., z=0.):
        positions = np.broadcast_to(self.neutral, (1, frames, 27, 3)).copy()
        positions[..., 0] += x
        positions[..., 1] += floor + self.clearance
        positions[..., 2] += z
        rotations = np.broadcast_to(np.eye(3), (1, frames, 27, 3, 3)).copy()
        return SimpleNamespace(positions=positions.astype(np.float32), rotations=rotations,
                               frames=frames, fps=20, actor_ids=('actor_1',))

    def test_neutral_elevated_floor_passes_without_pose_changes(self):
        scene = _scene(_platform())
        motion = self.neutral_clip(frames=3)
        original_scene, original_positions = copy.deepcopy(scene), motion.positions.copy()
        original_rotations = motion.rotations.copy()
        result = validate_terrain_clip(motion, scene)
        self.assertTrue(result['presentation_support_checked'])
        self.assertFalse(result['raw_native_support_checked'])
        self.assertFalse(result['poses_modified'])
        self.assertGreaterEqual(result['minimum_sampled_sole_clearance_m'], -.01)
        self.assertEqual(scene, original_scene)
        np.testing.assert_array_equal(motion.positions, original_positions)
        np.testing.assert_array_equal(motion.rotations, original_rotations)

    def test_wrong_floor_void_and_body_blocker_reject(self):
        blocked = make_object('wall', 1)
        blocked['position'] = [0., 3., 0.]
        blocked['size'] = [1., 2., .2]
        void = _platform()
        void['position'][0] = 5.
        cases = (
            ('wrong floor', _scene(_platform(0.)), 'no support'),
            ('void', _scene(void), 'no support'),
            ('body blocker', _scene(_platform(), blocked), 'overlaps'),
        )
        for label, scene, message in cases:
            with self.subTest(label=label):
                motion = self.neutral_clip()
                original_scene, original_positions = copy.deepcopy(scene), motion.positions.copy()
                with self.assertRaisesRegex(ValueError, message):
                    validate_terrain_clip(motion, scene)
                self.assertEqual(scene, original_scene)
                np.testing.assert_array_equal(motion.positions, original_positions)

    def test_saved_native_stair_seeds_reject_without_mutation(self):
        scene = json.loads((SAVED / 'scene.json').read_text())
        original_scene = copy.deepcopy(scene)
        for seed in (33, 11):
            with self.subTest(seed=seed):
                path = SAVED / f'height_sparse__seed{seed}__with_prefix.core.npz'
                with np.load(path, allow_pickle=False) as archive:
                    positions = np.concatenate([archive[f'c{i}_positions'] for i in range(1, 5)], axis=1)
                    rotations = np.concatenate([archive[f'c{i}_rotations'] for i in range(1, 5)], axis=1)
                    history = SimpleNamespace(positions=archive['c0_positions'].copy(),
                                              rotations=archive['c0_rotations'].copy(),
                                              frames=40, fps=20, actor_ids=('actor_1',))
                motion = SimpleNamespace(positions=positions, rotations=rotations,
                                         frames=160, fps=20, actor_ids=('actor_1',))
                original_positions, original_rotations = positions.copy(), rotations.copy()
                original_history = history.positions.copy()
                with self.assertRaisesRegex(ValueError, 'raw native sole penetrates terrain'):
                    validate_terrain_clip(motion, scene, history=history)
                np.testing.assert_array_equal(motion.positions, original_positions)
                np.testing.assert_array_equal(motion.rotations, original_rotations)
                np.testing.assert_array_equal(history.positions, original_history)
                self.assertEqual(scene, original_scene)

    def test_elevated_gate_replays_committed_trigger_after_seek(self):
        gate = make_object('door', 0)
        gate['size'] = [1.6, 3., .12]
        gate['position'] = [0., 3.5, 0.]
        scene = _scene(_platform(), gate)
        history = self.neutral_clip(frames=20, z=-.6)
        crossing = self.neutral_clip(frames=2, z=0.)
        self.assertTrue(validate_terrain_clip(crossing, scene, history=history,
                                              enabled=True, start_frame=0,
                                              terrain_start_frame=0)['presentation_support_checked'])
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            validate_terrain_clip(crossing, scene, history=history, enabled=True,
                                  start_frame=0, terrain_start_frame=21)
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            validate_terrain_clip(crossing, scene, history=history, enabled=True,
                                  start_frame=21, terrain_start_frame=0)

    def test_default_session_geometry_stays_flat(self):
        session = SimpleNamespace(_director=SimpleNamespace(project_metadata={'studio_core': {}},
                                                             total_frames=0),
                                  scene_reactions_enabled=False)
        self.assertEqual(CoreStudioSession._reaction_options(session),
                         {'enabled': False, 'terrain': False,
                          'terrain_start_frame': 0, 'start_frame': 0})
        self.assertIsNone(CoreStudioSession._check_geometry(self.neutral_clip(floor=0.),
                                                             _scene()))


if __name__ == '__main__':
    unittest.main()
