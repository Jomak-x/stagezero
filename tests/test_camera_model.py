"""Pure camera validation and deterministic hard-cut evaluation."""
import unittest

from camera_model import (camera_at_frame, copy_camera_cuts, validate_camera,
                          validate_camera_cuts, validate_cameras)


def camera(camera_id='camera-a'):
    return dict(id=camera_id, name='Wide', position=[1, 2, 3], wxyz=[2, 0, 0, 0], fov=1.0)


def cut(frame, camera_id='camera-a'):
    return dict(id=f'cut-{frame}', frame=frame, camera_id=camera_id)


class CameraModelTests(unittest.TestCase):
    def test_camera_normalizes_orientation_and_does_not_alias_input(self):
        source = camera()
        result = validate_camera(source)
        self.assertEqual(result['wxyz'], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(result['position'], [1.0, 2.0, 3.0])
        result['position'][0] = 9
        self.assertEqual(source['position'][0], 1)
        self.assertEqual(source['wxyz'][0], 2)

    def test_camera_rejects_malformed_pose_and_intrinsics(self):
        invalid = [
            ('position', [0, 0]), ('position', [0, 0, float('nan')]),
            ('position', [0, 0, float('inf')]), ('position', [0, 0, 10_001]),
            ('position', [0, 0, '3']), ('position', [0, 0, True]),
            ('position', [0, 0, 10 ** 1000]),
            ('wxyz', [0, 0, 0, 0]), ('wxyz', [1, 0, 0]),
            ('wxyz', [1, 0, 0, float('inf')]), ('wxyz', [1e-12, 0, 0, 0]),
            ('fov', 0), ('fov', 3.2), ('fov', float('nan')), ('fov', True),
            ('name', ''), ('name', ' '), ('name', 'x' * 81), ('id', ''),
        ]
        for field, value in invalid:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_camera(dict(camera(), **{field: value}))
        with self.assertRaises(ValueError):
            validate_cameras([camera(), camera()])

    def test_cut_boundaries_include_exact_frame_seek_backward_and_loop(self):
        cuts = validate_camera_cuts([cut(0), cut(10, 'camera-b'), cut(20)], 30,
                                    {'camera-a', 'camera-b'})
        self.assertEqual([camera_at_frame(cuts, frame) for frame in (0, 9, 10, 19, 20, 29, 7, 0)],
                         ['camera-a', 'camera-a', 'camera-b', 'camera-b',
                          'camera-a', 'camera-a', 'camera-a', 'camera-a'])
        self.assertIsNone(camera_at_frame([], 0))
        self.assertIsNone(camera_at_frame(cuts, -1))

    def test_cut_validation_rejects_noncanonical_or_dangling_timeline(self):
        invalid = [[cut(1)], [cut(0), cut(0)], [cut(0), cut(20)],
                   [cut(0), cut(10), cut(5)], [cut(0), dict(cut(10), id='cut-0')],
                   [dict(cut(0), frame=False)], [dict(cut(0), frame=0.0)],
                   [cut(0, 'missing')]]
        for cuts in invalid:
            with self.subTest(cuts=cuts), self.assertRaises(ValueError):
                validate_camera_cuts(cuts, 20, {'camera-a'})

    def test_copy_truncates_exclusive_boundary_without_shifting_frames(self):
        source = [cut(0), cut(4), cut(8)]
        copied = copy_camera_cuts(source, 8)
        self.assertEqual([entry['frame'] for entry in copied], [0, 4])
        copied[0]['camera_id'] = 'camera-b'
        self.assertEqual(source[0]['camera_id'], 'camera-a')
