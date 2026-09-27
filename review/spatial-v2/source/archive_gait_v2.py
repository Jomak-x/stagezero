from pathlib import Path
import sys,numpy as np
ROOT=Path('/Users/jakob/.codex/worktrees/background-demo-finish/Shellhacks');sys.path.insert(0,str(ROOT))
from realtime_clip import CanonicalClip
from studio_core_session import CoreStudioSession
from scene_composition import validate_scene
for name in ('dense-heading-065','dense-xz-120'):
 folder=ROOT/'review/spatial-v2/gait-ablation';a=np.load(folder/(name+'.npz'))
 scene=validate_scene({'version':3,'assets':[],'effects':[],'lighting':'neutral','name':'Native gait comparison','objects':[],'camera':{'position':[4,2.5,7.5],'look_at':[0,1,2.4]}})
 with CoreStudioSession() as session:
  session.start(1,scene,{'actor_1':{'position_xz':[0,0],'yaw':0}})
  d=session._director
  d.submit_instruction('A person walks forward naturally.',kind='approach',frames=80)
  for f in (0,40):
   req=d.claim_request();assert req
   clip=CanonicalClip(a['positions'][:,f:f+40],a['rotations'][:,f:f+40],20,('actor_1',),'ardy_core',{'experiment':name},a['native_features'][:,f:f+40])
   assert d.complete(req.request_id,clip)
  (folder/(name+'.core.stagezero.npz')).write_bytes(session.save_project())
