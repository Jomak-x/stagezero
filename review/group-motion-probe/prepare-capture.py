from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import numpy as np,json
from cast_performance import CastPerformance,decode_project,encode_project
from prompt_scene_camera import prompt_scene_camera_view
R=Path(__file__).resolve().parents[2];D=R/'review/group-motion-probe'
cases=[]
for case in json.loads((D/'cases.json').read_text()):
 p=D/'fresh'/case['case']/'attempt-001/scene.cast.stagezero.npz'
 if not p.exists():continue
 clip,cast,scene,_=decode_project(p.read_bytes());pos,look,fov=prompt_scene_camera_view(clip,scene)
 cam={'position':pos.tolist(),'look_at':look.tolist(),'fov_radians':float(fov)}
 if case['case']=='wave-three-s71':
  frames=clip.frames;n=len(clip.actor_ids);j=np.repeat(clip.joints[0:1],frames*n,axis=0)
  for actor in range(n):
   j[actor*frames:(actor+1)*frames,actor]=clip.joints[:,actor]
   j[(actor+1)*frames:,actor]=clip.joints[-1,actor]
  metadata={'model':'Exact-source sequential control','fps':30,'frames':len(j),'comparison':'Same independently generated sources scheduled sequentially with explicit holds; not an actual production planner run','segments':[{'label':'Same-source sequential timing control','source':'ardy_core','kind':'timing_control','start_frame':0,'end_frame_exclusive':len(j),'frames':len(j)}]}
  baseline=D/'baselines/wave-three-s71-sequential.cast.stagezero.npz';baseline.write_bytes(encode_project(CastPerformance(clip.actor_ids,j,metadata=metadata),cast,scene))
  cases.append({'archive':str(baseline.relative_to(D)),'output_dir':'video/wave-sequential-control','fixed_camera':cam})
 cases.append({'archive':str(p.relative_to(D)),'output_dir':'video/'+case['case'],'fixed_camera':cam})
(D/'capture.json').write_text(json.dumps(cases,indent=2)+'\n');print('Capturecases',len(cases))
