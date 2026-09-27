import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.native_pair_rig import NativeRigAsset
from test_native_pair_rig import fixture_glb
from cast_motion_refinement import refine_cast_motion, cast_body_clearance


class RefinementTests(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'rig.glb'; path.write_bytes(fixture_glb()[0])
            pose = NativeRigAsset(path).rest
        self.points = np.repeat(np.stack([pose, pose+[2,0,0], pose+[4,0,0]])[None], 120, axis=0)
        self.ids = ('a','b','c')
        self.segments = [dict(label='native pair',source='intergen',kind='paired_action',start_frame=0,end_frame_exclusive=120,frames=120)]
        self.activities = [dict(start_frame=0,end_frame_exclusive=120,active_actor_ids=['a','b'],contact_actor_ids=['a','b'])]

    def test_native_contact_exact_and_idle_feet_exact_with_live_upper_body(self):
        out, report = refine_cast_motion(self.points,self.ids,self.segments,self.activities)
        np.testing.assert_array_equal(out[:,:2], self.points[:,:2])
        np.testing.assert_array_equal(out[:,2,[0,1,2,4,5,7,8,10,11]],self.points[:,2,[0,1,2,4,5,7,8,10,11]])
        self.assertGreater(np.max(abs(out[:,2,20]-self.points[:,2,20])),.01)
        self.assertFalse(report['source_pair_frames_modified'])
        self.assertEqual(report['body_clearance'][0]['unintended_frames'],0)

    def test_body_proxy_detects_overlap_beyond_root_disc(self):
        self.points[:,2,20] = self.points[:,0,9]
        values=cast_body_clearance(self.points,self.ids,self.activities)
        self.assertTrue((values[(0,2)][0]<0).all())
        self.assertTrue(values[(0,2)][1].all())

    def test_misaligned_activity_rejected_not_silently_zipped(self):
        with self.assertRaises(ValueError):
            refine_cast_motion(self.points,self.ids,self.segments,[])
        self.activities[0]['end_frame_exclusive']=119
        with self.assertRaises(ValueError):
            refine_cast_motion(self.points,self.ids,self.segments,self.activities)

if __name__ == '__main__': unittest.main()
