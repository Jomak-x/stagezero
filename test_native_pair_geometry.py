"""Native world-pair scene checks against the actual recovered source."""
import json
from pathlib import Path
import unittest

import numpy as np

from native_pair_geometry import check_native_pair_geometry
from scene_objects import make_object


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'review/two-character/native-recovery/originals/handshake_seed42.npz'
CITY = ROOT / 'review/scene-integration/live-city.json'


def _source():
    with np.load(SOURCE, allow_pickle=False) as archive:
        return archive['joints'].copy()


def _city():
    return json.loads(CITY.read_text())


class NativePairGeometryTests(unittest.TestCase):
    def test_original_handshake_passes_authored_city_without_partner_gate(self):
        result = check_native_pair_geometry(_source(), _city())
        self.assertEqual(result['frames'], 210)
        self.assertEqual(result['skeleton'], 'native22')
        self.assertTrue(all(actor['ground']['continuous_support_verified'] for actor in result['actors']))
        self.assertTrue(all(actor['collision']['total_collision_frames'] == 0 for actor in result['actors']))
        self.assertFalse(result['physical_contact_verified'])

    def test_scene_solid_blocks_pair_without_modifying_source(self):
        joints = _source()
        before = joints.tobytes()
        scene = _city()
        blocker = make_object('crate', 0)
        blocker['position'] = [float(v) for v in joints[0, 0, 0]]
        scene['objects'].append(blocker)
        with self.assertRaisesRegex(ValueError, 'scene solid crate-0'):
            check_native_pair_geometry(joints, scene)
        self.assertEqual(joints.tobytes(), before)

    def test_unsupported_authored_floor_rejects_root_path(self):
        floor = make_object('platform', 0)
        floor['position'] = [10., -.05, 0.]
        floor['size'] = [6., .1, 6.]
        scene = {'version': 2, 'name': 'Remote floor', 'objects': [floor],
                 'effects': [], 'lighting': 'neutral'}
        with self.assertRaisesRegex(ValueError, 'continuous authored floor'):
            check_native_pair_geometry(_source(), scene)

    def test_partner_overlap_is_not_a_scene_solid(self):
        joints = _source()
        joints[:, 1] = joints[:, 0]
        scene = {'version': 2, 'name': 'Empty stage', 'objects': [],
                 'effects': [], 'lighting': 'neutral'}
        self.assertEqual(check_native_pair_geometry(joints, scene)['frames'], 210)

    def test_wrong_native_shape_rejected_before_scene(self):
        with self.assertRaisesRegex(ValueError, 'world joints'):
            check_native_pair_geometry(np.zeros((4, 2, 27, 3)), _city())


if __name__ == '__main__':
    unittest.main()
