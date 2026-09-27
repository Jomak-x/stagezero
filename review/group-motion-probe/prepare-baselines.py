from pathlib import Path
from copy import deepcopy
import json,hashlib,numpy as np
from cast_performance import CastPerformance,decode_project,encode_project
from prompt_scene_builder import check_cast_geometry
R=Path(__file__).resolve().parents[2];out=R/'review/group-motion-probe/baselines';out.mkdir(exist_ok=True)
for name in ['city-handshake-s42','industrial-spar-s45']:
 p=R/'review/interaction-v2/fresh'/name/'attempt-001/scene.cast.stagezero.npz';data=p.read_bytes();c,cast,scene,_=decode_project(data)
 j=np.concatenate([c.joints,np.repeat(c.joints[0:1,0:1],c.frames,axis=0)],axis=1)
 target=np.array([0.,0.,-6.]);delta=target-j[0,2,0];delta[1]=0;j[:,2]+=delta
 ids=(*c.actor_ids,'actor_3');m=deepcopy(c.metadata);m.update(actor_ids=list(ids),title='Original pair with a stationary third observer',third_probe_baseline={'source':str(p.relative_to(R)),'sha256':hashlib.sha256(data).hexdigest(),'pair_payload_exact':True,'third_pose':'first real pose of actor1 held at separate supported start; authored baseline only'},source_pair_frames_modified=False)
 for a in m['segment_activity']:a['held_actor_ids']=[*a.get('held_actor_ids',[]),'actor_3']
 if 'plan' in m:m['plan']['actors'].append({'id':'actor_3','name':'Observer','start':{'x':0.,'z':-6.}});m['plan']['actor_count']=3
 for a in m['segment_activity']:check_cast_geometry(j[a['start_frame']:a['end_frame_exclusive']],scene,ids,contact_pair=a.get('contact_actor_ids',[]))
 new=CastPerformance(ids,j,metadata=m);cast.append({'id':'actor_3','name':'Observer','color':[182,129,242]})
 (out/(name+'.cast.stagezero.npz')).write_bytes(encode_project(new,cast,scene_document=scene))
 print(name,'baseline pair exact',new.joints[:,:2].tobytes()==c.joints.tobytes())
