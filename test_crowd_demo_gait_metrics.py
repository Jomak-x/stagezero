import unittest
import numpy as np
from crowd_demo_gait_metrics import rendered_demo_feet
class DemoFeetTest(unittest.TestCase):
 def test_pair_uses_unit_scale_world_y_and_preserved_pose(self):
  poses=np.zeros((12,22,3));poses[:,:,0]=.1;poses[:,:,1]=.04
  clips=[dict(id='walk-relaxed',type='walk',offset=0,frames=2,fps=10,strideMeters=1),dict(id='idle',type='idle',offset=2,frames=2,fps=10,strideMeters=0),dict(id='greeting-role-0',type='pair',offset=4,frames=8,fps=10,strideMeters=0)]
  poses[4:,:,0]=.4;poses[4:,:,1]=.08
  m={'totalFrames':12,'clips':clips};t={'dt':.1,'agents':[{'gait':'relaxed'}],'frames':[{'t':i*.1,'people':[[2,3,0,0,1,0,2,.3+i*.1,.5]]}for i in range(3)]}
  _,feet,_=rendered_demo_feet(m,poses,t,10)
  self.assertTrue(np.allclose(feet[0,0,:,0],2.4));self.assertTrue(np.allclose(feet[0,0,:,1],.58));self.assertTrue(np.allclose(feet[0,0,:,2],3))
 def test_formation_actor_preserves_native_width_instead_of_random_height(self):
  poses=np.zeros((12,22,3));poses[:,:,0]=.1;poses[:,:,1]=.04
  poses[4:,:,0]=.4;poses[4:,:,1]=.8
  clips=[dict(id='walk-relaxed',type='walk',offset=0,frames=2,fps=10,strideMeters=1),dict(id='idle',type='idle',offset=2,frames=2,fps=10,strideMeters=0),dict(id='dance',type='dance',offset=4,frames=8,fps=10,strideMeters=0)]
  people=[[2,3,0,0,1,0,2,.35,0] for _ in range(7)]
  agents=[{'gait':'relaxed','unit_scale':i==6} for i in range(7)]
  t={'dt':.1,'agents':agents,'frames':[{'t':i*.1,'people':people} for i in range(3)]}
  _,feet,_=rendered_demo_feet({'totalFrames':12,'clips':clips},poses,t,10)
  self.assertTrue(np.allclose(feet[0,6,:,0],2.4));self.assertTrue(np.allclose(feet[0,6,:,1],.8))
if __name__=='__main__':unittest.main()
