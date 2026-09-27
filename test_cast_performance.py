"""Independent cast archive bounds and exact immutable display coordinates."""
from functools import lru_cache
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from cast_performance import (CastPerformance, FORMAT, cast_from_performance,
                              decode_project, encode_project)
from experiments.native_pair_rig import NativeRigAsset
from native_pair_clip import NativePairClip, decode_project as decode_pair, encode_project as encode_pair
from paired_scene import EMPTY_SCENE


@lru_cache(maxsize=1)
def rest_pose():
    return NativeRigAsset(Path(__file__).parent / 'assets/paired/Xbot.glb').rest


def performance(count=3, frames=8, offset=0.):
    ids = tuple(f'performer_{index + 1}' for index in range(count))
    joints = np.tile(rest_pose(), (frames, count, 1, 1))
    for index in range(count):
        joints[:, index, :, 0] += index * 3. + offset
        joints[:, index, :, 2] += np.arange(frames)[:, None] * .01
    return CastPerformance(ids, joints, metadata={
        'model': 'Structural test fixture', 'provenance': {'source': 'authored rest translations'},
        'plan': {'actors': [{'id': identifier, 'name': f'Performer {index + 1}'} for index, identifier in enumerate(ids)]},
        'segments': [{'label': 'Fixture motion', 'start_frame': 0, 'end_frame_exclusive': frames,
                      'frames': frames, 'source': 'test fixture', 'kind': 'authored_test'}]})


def repack(content, *, document=None, joints=None):
    with np.load(io.BytesIO(content), allow_pickle=False) as archive:
        original = json.loads(archive['metadata'].item())
        coordinates = archive['joints'].copy()
    if document is not None:
        original = document(original)
    stream = io.BytesIO()
    np.savez_compressed(stream, joints=coordinates if joints is None else joints,
                        metadata=np.array(json.dumps(original)))
    return stream.getvalue()


class CastPerformanceTests(unittest.TestCase):
    def test_one_two_three_actor_roundtrips_preserve_exact_bytes_ids_and_scene(self):
        scene = dict(EMPTY_SCENE, name='Saved cast stage')
        for count in (1, 2, 3):
            with self.subTest(count=count):
                clip = performance(count)
                cast = cast_from_performance(clip)
                restored, loaded_cast, loaded_scene, frame = decode_project(encode_project(clip, cast, scene, 5))
                self.assertEqual(restored.actor_ids, clip.actor_ids)
                self.assertEqual(restored.joints.dtype, clip.joints.dtype)
                self.assertEqual(restored.joints.tobytes(), clip.joints.tobytes())
                self.assertEqual(restored.metadata, clip.metadata)
                self.assertEqual(loaded_cast, cast)
                self.assertEqual(loaded_scene, scene)
                self.assertEqual(frame, 5)

    def test_arrays_and_nested_metadata_have_no_mutable_shared_owner(self):
        source = np.zeros((4, 1, 22, 3), dtype=np.float32)
        metadata = {'provenance': {'source': ['unchanged']}}
        clip = CastPerformance(['solo'], source, metadata=metadata)
        source[:] = 9
        metadata['provenance']['source'].append('external mutation')
        clip.metadata['provenance']['source'].append('returned copy mutation')
        self.assertEqual(clip.metadata, {'provenance': {'source': ['unchanged']}})
        self.assertTrue(np.all(clip.joints == 0))
        with self.assertRaises(ValueError):
            clip.joints.flags.writeable = True
        self.assertEqual(clip.actor_ids, ('solo',))

    def test_rejects_wrong_shapes_counts_ids_timing_and_nonfinite_coordinates(self):
        for ids, array in (((), np.zeros((4, 0, 22, 3))),
                (('a',) * 2, np.zeros((4, 2, 22, 3))),
                (('a',), np.zeros((4, 2, 22, 3))),
                (('a',), np.zeros((1001, 1, 22, 3))),
                (('a',), np.zeros((3, 1, 22, 3))),
                (('a',), np.zeros((4, 1, 27, 3))),
                (('a',), np.zeros((4, 1, 22, 3), dtype=np.int32)),
                (('a',), np.full((4, 1, 22, 3), np.inf)),
                (('a', 'b', 'c', 'd'), np.zeros((4, 4, 22, 3)))):
            with self.assertRaises(ValueError):
                CastPerformance(ids, array)
        with self.assertRaises(ValueError):
            CastPerformance(('a',), np.zeros((4, 1, 22, 3)), fps=20)
        clip = performance()
        with self.assertRaisesRegex(ValueError, 'contiguously'):
            CastPerformance(clip.actor_ids, clip.joints,
                metadata={'segments': [dict(clip.metadata['segments'][0], start_frame=1)]})

    def test_pair_and_cast_archives_are_not_interchangeable(self):
        clip = performance(2)
        cast = cast_from_performance(clip)
        cast_archive = encode_project(clip, cast)
        pair_archive = encode_pair(NativePairClip(clip.joints), cast, clip.actor_ids)
        with self.assertRaises(ValueError):
            decode_pair(cast_archive)
        with self.assertRaises(ValueError):
            decode_project(pair_archive)
        with self.assertRaises(ValueError):
            encode_project(NativePairClip(clip.joints), cast)

    def test_archive_rejects_tampered_identity_playhead_schema_and_extra_members(self):
        clip = performance()
        content = encode_project(clip, cast_from_performance(clip))
        changes = [lambda doc: dict(doc, format='stagezero_native_pair'),
                   lambda doc: dict(doc, frame=clip.frames),
                   lambda doc: dict(doc, actor_ids=list(reversed(doc['actor_ids']))),
                   lambda doc: dict(doc, fps=20), lambda doc: dict(doc, unexpected=True)]
        for change in changes:
            with self.assertRaises(ValueError):
                decode_project(repack(content, document=change))
        expanded = io.BytesIO(content)
        with ZipFile(expanded, 'a') as zipped:
            zipped.writestr('features.npy', b'not part of this format')
        with self.assertRaises(ValueError):
            decode_project(expanded.getvalue())
        with self.assertRaises(ValueError):
            decode_project(repack(content, joints=np.full_like(clip.joints, np.nan)))

    def test_huge_declared_array_is_rejected_before_numpy_load_allocates(self):
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {'shape': (10**9, 3, 22, 3),
            'fortran_order': False, 'descr': '<f8'})
        metadata = io.BytesIO()
        np.save(metadata, np.array('{}'))
        archive = io.BytesIO()
        with ZipFile(archive, 'w') as zipped:
            zipped.writestr('joints.npy', header.getvalue())
            zipped.writestr('metadata.npy', metadata.getvalue())
        with patch('cast_performance.np.load', side_effect=AssertionError('Array loading was too early')):
            with self.assertRaisesRegex(ValueError, 'shape or dtype'):
                decode_project(archive.getvalue())


if __name__ == '__main__':
    unittest.main()
