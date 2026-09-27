import copy
import json
from pathlib import Path
import unittest
import numpy as np
from crowd_demo_cues import apply_motion_cues, sample_root
ROOT=Path(__file__).resolve().parent

def fixture():
    agents=[{'id':i,'radius':.28,'schedule':[{'start':0.,'end':30.,'task':'watch'}]} for i in range(3)]
    frames=[]
    for t in np.arange(301)/10:
        frames.append({'t':float(t),'people':[[-.65,0.,np.pi/2,0.,1,0.],[.65,0.,-np.pi/2,0.,1,0.],[8.,8.,0.,0.,1,0.]]})
    return {'environment':'crossing','duration':30.,'dt':.1,'agents':agents,'frames':frames,'events':[{'kind':'greeting','id':'test','actor_ids':[0,1],'start_s':12.,'end_s':20.,'clip_start_s':14.,'clip_end_s':18.,'anchor_xz':[0.,0.],'actor_anchors_xz':[[-.65,0],[.65,0]],'shared_yaw':0.,'reserved_radius_m':2.15}]}

class DemoCuesTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.manifest=json.loads((ROOT/'review/crowd-demos/assets/manifest.json').read_text())
 def test_native_track_inside_reserved_pocket_and_unchanged_outsiders(self):
  original=fixture();saved=copy.deepcopy(original);d=apply_motion_cues(original,self.manifest)
  self.assertEqual(original,saved)
  a=np.array([f['people'] for f in d['frames']]);native=(np.array([f['t'] for f in d['frames']])>=14)&(np.array([f['t'] for f in d['frames']])<=17.9666667)
  for role in (0,1):
   c=self.manifest['clips'][6+role]
   for frame in np.flatnonzero(native):
    p=d['frames'][frame]['people'][role];np.testing.assert_allclose(p[:2],sample_root(c,frame*.1-14),atol=1e-12);self.assertEqual(p[6],6+role);self.assertEqual(p[2],0)
  self.assertLess(np.linalg.norm(np.diff(a[:,:2,:2],axis=0),axis=2).max(),.2)
  self.assertGreater(np.linalg.norm(a[:,0,:2]-a[:,1,:2],axis=1).min(),.7)
  np.testing.assert_array_equal(a[:,2,:2],np.tile([8.,8.],(301,1)))
  for t in (0,12,20,30):np.testing.assert_allclose(a[round(t*10),:2,:2],[[-.65,0],[.65,0]])
 def test_gestures_only_during_stationary_dwell(self):
  d=fixture();d['events']=[]
  for f in d['frames']:f['people'][1][3]=.2;f['people'][2][3]=.5
  output=apply_motion_cues(d,self.manifest)
  self.assertTrue(any(f['people'][0][6]>=0 for f in output['frames']))
  self.assertTrue(all(f['people'][1][6]==-1 and f['people'][2][6]==-1 for f in output['frames']))
  for f in output['frames']:
   p=f['people'][0]
   if p[6]>=0:self.assertLessEqual(p[7],self.manifest['clips'][p[6]]['duration'])
 def test_common_rotation_preserves_source_pair_spacing(self):
  d=fixture();d['events'][0]['shared_yaw']=1.1;out=apply_motion_cues(d,self.manifest)
  for t in (14.,15.,16.,17.9):
   p=out['frames'][round(t*10)]['people'];actual=np.linalg.norm(np.array(p[0][:2])-p[1][:2]);expected=np.linalg.norm(sample_root(self.manifest['clips'][6],t-14)-sample_root(self.manifest['clips'][7],t-14));self.assertAlmostEqual(actual,expected)
 def test_fail_closed_on_small_pocket_or_reapplication(self):
  d=fixture();d['events'][0]['reserved_radius_m']=.5
  with self.assertRaises(ValueError):apply_motion_cues(d,self.manifest)
  d=apply_motion_cues(fixture(),self.manifest)
  with self.assertRaises(ValueError):apply_motion_cues(d,self.manifest)
if __name__=='__main__':unittest.main()

class FlashmobCuesTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.manifest=json.loads((ROOT/'review/crowd-demos/assets/manifest.json').read_text())
 def dance_fixture(self):
  slots=[[x*2.6,z*2.6] for z in range(4) for x in range(6)]
  return {'environment':'city','duration':70.,'dt':.1,'agents':[{'id':i+6,'radius':.28,'schedule':[]} for i in range(24)],'frames':[{'t':i/10,'people':[[*slot,0.,0.,1,0.] for slot in slots]} for i in range(701)],'events':[{'kind':'flashmob','actor_ids':list(range(6,30)),'dance_start_s':40.,'dance_end_s':65.,'shared_yaw':0.,'slot_xz':slots,'reserved_radius_m':1.2}]}
 def test_exact_synchronized_25_seconds_and_continuous_roots(self):
  d=self.dance_fixture();out=apply_motion_cues(d,self.manifest);c=self.manifest['clips'][10]
  for f in (400,415,520,649,650):
   p=out['frames'][f]['people'];self.assertTrue(all(x[6]==10 for x in p));self.assertTrue(all(x[7]==p[0][7] for x in p))
   for i,row in enumerate(p):np.testing.assert_allclose(np.array(row[:2])-d['events'][0]['slot_xz'][i],sample_root(c,f/10-40),atol=1e-12)
  for f in (399,400,650,651):
   np.testing.assert_allclose(np.array([p[:2] for p in out['frames'][f]['people']]),d['events'][0]['slot_xz'],atol=1e-12)
  self.assertTrue(all(a['unit_scale'] for a in out['agents']))
 def test_display_floor_lift_has_bounded_continuous_entry_and_exit(self):
  c=self.manifest['clips'][10];lift=np.asarray(c['groundLiftY']);g=c['groundCorrection']
  self.assertEqual(lift.shape,(751,));self.assertTrue(np.isfinite(lift).all());self.assertGreaterEqual(lift.min(),0.)
  self.assertLessEqual(np.abs(np.diff(lift)).max(),.25/30+1e-12);self.assertGreaterEqual(g['minMeshYAfter'],.0019)
  out=apply_motion_cues(self.dance_fixture(),self.manifest);y=np.array([f['people'][0][8] for f in out['frames']])
  self.assertEqual(y[390],0.);self.assertEqual(y[660],0.);self.assertAlmostEqual(y[400],lift[0]);self.assertAlmostEqual(y[650],lift[-1]);self.assertLessEqual(np.abs(np.diff(y)).max(),.025001)
 def test_floor_correction_is_applied_once_as_instance_y(self):
  c=self.manifest['clips'][10];f=int(np.argmax(c['groundLiftY']));lift=c['groundLiftY'][f]
  a=np.fromfile(ROOT/'review/crowd-demos/assets/affine.bin',dtype='<f4').reshape(-1,22,3,4)[c['offset']+f];minimum=float('inf')
  for part in self.manifest['parts']:
   bind=np.asarray(part['localBind']).reshape(-1,4,3);bones=np.asarray(part['bones']).reshape(-1,4);weights=np.asarray(part['weights']).reshape(-1,4);rows=a[bones,1,:]
   y=(((rows[:,:,:3]*bind).sum(axis=-1)+rows[:,:,3])*weights).sum(axis=1);minimum=min(minimum,float(y.min()))
  self.assertAlmostEqual(minimum,c['groundCorrection']['minMeshYBefore'],places=6)
  self.assertAlmostEqual(minimum+lift,.002,places=6)
  self.assertGreater(minimum+2*lift,.09)  # A second application would visibly hover.
 def test_formation_clearance_and_unsettled_actor_fail_closed(self):
  d=self.dance_fixture();d['events'][0]['slot_xz'][1]=[1.,0.]
  with self.assertRaises(ValueError):apply_motion_cues(d,self.manifest)
  d=self.dance_fixture();d['frames'][402]['people'][2][0]+=.1
  with self.assertRaises(ValueError):apply_motion_cues(d,self.manifest)
 def test_refresh_keeps_pair_roots_and_removes_ordinary_waves(self):
  from crowd_demo_cues import refresh_existing_cues
  d=apply_motion_cues(fixture(),self.manifest)
  for f in d['frames']:f['people'][2][6:]=[3,.5,0.]
  before=np.array([f['people'] for f in d['frames']]);out=refresh_existing_cues(d,self.manifest);after=np.array([f['people'] for f in out['frames']])
  np.testing.assert_array_equal(before[:,:,:6],after[:,:,:6]);np.testing.assert_array_equal(before[:,:,:][np.isin(before[:,:,6],[6,7])],after[:,:,:][np.isin(before[:,:,6],[6,7])])
  self.assertTrue(np.all(after[:,2,6]==-1))
