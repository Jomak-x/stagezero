"""Rejected authored candidates retain exact provenance and cannot publish."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cast_motion_refinement import RefinementRejected
from experiments.native_pair_rig import NativeRigAsset
from prompt_scene_builder import PromptSceneBuilder
from test_native_pair_rig import fixture_glb
from test_prompt_scene_builder import CoreDouble, SCENE


class CastRefinementRejectionTests(unittest.TestCase):
    def test_builder_archives_both_compositions_and_never_publishes_rejected_candidate(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rig = root / 'fixture.glb'
            rig.write_bytes(fixture_glb()[0])
            client = CoreDouble(NativeRigAsset(rig).rest)
            plan = {'title': 'Rejection provenance',
                    'actors': [{'id': 'actor_1', 'name': 'One'}],
                    'beats': [{'id': 'beat-1', 'actor_ids': ['actor_1'],
                               'prompt': 'Stand.', 'seconds': 2}]}
            staging = {'starts': {'actor_1': {'x': 0., 'z': 0.}},
                       'meeting': {'x': 0., 'z': 0.}, 'target_id': None}
            report = {'version': 1, 'body_clearance': [{'worsened_overlap_frames': 3}],
                      'visual_acceptance': 'unverified'}
            captured = {}
            progress = []

            def reject(joints, *args, **kwargs):
                candidate = joints.copy()
                candidate[:, 0, 20, 0] += .05
                captured['unrefined'] = joints.copy()
                captured['candidate'] = candidate.copy()
                raise RefinementRejected(candidate, report)

            builder = PromptSceneBuilder('Stand.', SimpleNamespace(plan=lambda *args, **kwargs: plan),
                                         None, client, root / 'sources')
            with patch('prompt_scene_plan.auto_place', return_value=staging), \
                 patch('cast_motion_refinement.refine_cast_motion', side_effect=reject) as refinement, \
                 patch('cast_performance.CastPerformance') as published:
                with self.assertRaises(RefinementRejected):
                    builder(SCENE, on_progress=progress.append)
            refinement.assert_called_once()
            published.assert_not_called()
            self.assertFalse(any(event['phase'] == 'complete' for event in progress))

            folders = list((root / 'sources').iterdir())
            self.assertEqual(len(folders), 1)
            manifest = json.loads((folders[0] / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'rejected')
            self.assertIn('worsens unintended sampled body overlap', manifest['error'])
            self.assertEqual(manifest['motion_refinement'], report)
            self.assertNotIn('frames', manifest)
            self.assertEqual(Path(manifest['unrefined_composition']).parent, folders[0])
            self.assertEqual(Path(manifest['refined_composition']).parent, folders[0])
            with np.load(manifest['unrefined_composition']) as saved:
                np.testing.assert_array_equal(saved['joints'], captured['unrefined'])
            with np.load(manifest['refined_composition']) as saved:
                np.testing.assert_array_equal(saved['joints'], captured['candidate'])
            self.assertFalse(np.array_equal(captured['candidate'], captured['unrefined']))


if __name__ == '__main__':
    unittest.main()
