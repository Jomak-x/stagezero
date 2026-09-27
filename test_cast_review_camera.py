"""Matched-camera evidence must use identical finite projection inputs."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments.capture_cast_review import read_cases, set_camera


class MatchedReviewCameraTests(unittest.TestCase):
    def test_fixed_camera_bypasses_automatic_orientation_and_zoom(self):
        fixed={'position':[4,3,-7], 'look_at':[0,.8,-1], 'fov_radians':.73}
        client=SimpleNamespace(camera=SimpleNamespace())
        with patch('experiments.capture_cast_review.prompt_scene_camera_view',side_effect=AssertionError('must not reframe')):
            report=set_camera(client,None,{},'closer',.25,fixed)
        self.assertEqual(client.camera.position,(4.,3.,-7.))
        self.assertEqual(client.camera.look_at,(0.,.8,-1.))
        self.assertEqual(client.camera.fov,.73)
        self.assertEqual(client.camera.up_direction,(0,1,0))
        self.assertEqual(report['mode'],'fixed_comparison_view')

    def test_manifest_rejects_nonfinite_or_degenerate_comparison_camera(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'take.npz').write_bytes(b'archive placeholder')
            for position, fov in (([0,float('nan'),1],.7),([0,0,0],.7),([0,1,2],4.)):
                (root/'cases.json').write_text(json.dumps([{'archive':'take.npz','output_dir':'new','fixed_camera':{
                    'position':position,'look_at':[0,0,0],'fov_radians':fov}}]))
                with self.assertRaisesRegex(ValueError,'fixed_camera'):
                    read_cases(root/'cases.json')

if __name__=='__main__': unittest.main()
