"""Motion contract tests: source identity, reversible pair placement, finite skin."""
import json
from pathlib import Path
import unittest
import numpy as np
from crowd_demo_motion_export import root_center, reconstruct, PAIR_SOURCE
from crowd_motion_library import REPO_ROOT, sha256
from native_pair_clip import load_source

class MotionExportTests(unittest.TestCase):
 def test_reversible_root_center_preserves_common_pair(self):
  source=load_source(REPO_ROOT/PAIR_SOURCE).joints
  origin=source[0,:,0][:,[0,2]].mean(axis=0)
  recovered=[]
  for role in range(2):
   p,track=root_center(source[:,role],origin_xz=origin)
   self.assertTrue(np.allclose(p[:,0][:,[0,2]],0,atol=1e-12))
   r=reconstruct(p,track);r[:,:,0]+=origin[0];r[:,:,2]+=origin[1]
   np.testing.assert_allclose(r,source[:,role],atol=1e-7);recovered.append(r)
  q=np.stack(recovered,axis=1)
  np.testing.assert_allclose(q[:,0]-q[:,1],source[:,0]-source[:,1],atol=2e-7)
 def test_invalid_root_inputs_rejected(self):
  for invalid in (np.zeros((3,27,3)),np.full((3,22,3),np.nan)):
   with self.assertRaises(ValueError):root_center(invalid)
  with self.assertRaises(ValueError):root_center(np.zeros((3,22,3)),origin_xz=[np.inf,0])
 def test_exported_atlas_rig_and_clip_contract(self):
  path=REPO_ROOT/'review/crowd-demos/assets';m=json.loads((path/'manifest.json').read_text())
  aff=np.fromfile(path/'affine.bin',dtype='<f4').reshape(-1,22,3,4)
  poses=np.fromfile(path/'poses.bin',dtype='<f4').reshape(-1,22,3)
  self.assertEqual(len(aff),m['totalFrames']);self.assertEqual(len(poses),len(aff));self.assertTrue(np.isfinite(aff).all())
  # Source solver's translations must match exact exported joint positions.
  np.testing.assert_allclose(aff[:,:,:,3],poses,atol=1e-6)
  self.assertEqual(m['rig']['asset_sha256'],sha256(REPO_ROOT/'assets/paired/Xbot.glb'))
  expected_offset=0
  for c in m['clips']:
   self.assertEqual(c['offset'],expected_offset);expected_offset+=c['frames']
   self.assertEqual(np.asarray(c['rootXZ']).shape,(c['frames'],2));self.assertTrue(np.isfinite(c['rootXZ']).all())
   self.assertAlmostEqual(c['duration'],(c['frames']-1)/c['fps'])
   np.testing.assert_allclose(c['startPose'],poses[c['offset']],atol=1e-6)
   np.testing.assert_allclose(c['endPose'],poses[expected_offset-1],atol=1e-6)
  for p in m['parts']:
   bones=np.asarray(p['bones']);weights=np.asarray(p['weights']).reshape(-1,4)
   self.assertTrue(((bones>=0)&(bones<22)).all());np.testing.assert_allclose(weights.sum(axis=1),1,atol=1e-6)
 def test_exported_pair_reconstructs_native_body(self):
  path=REPO_ROOT/'review/crowd-demos/assets';m=json.loads((path/'manifest.json').read_text());p=np.fromfile(path/'poses.bin',dtype='<f4').reshape(-1,22,3)
  pair=m['pairs'][0];original=load_source(REPO_ROOT/PAIR_SOURCE).joints
  for role,index in enumerate(pair['clipIndices']):
   c=m['clips'][index];q=reconstruct(p[c['offset']:c['offset']+c['frames']],c['rootXZ'])
   q[:,:,0]+=pair['sourceOriginXZ'][0];q[:,:,2]+=pair['sourceOriginXZ'][1];q[:,:,1]+=c['floorReferenceY']
   np.testing.assert_allclose(q,original[:,role],atol=3e-7)
if __name__=='__main__':unittest.main()
