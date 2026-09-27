import sys, json, uuid, math, time
from pathlib import Path
import numpy as np
ROOT=Path('/Users/jakob/.codex/worktrees/background-demo-finish/Shellhacks');sys.path.insert(0,str(ROOT))
from realtime_client import RealtimeClient
from ardy.skeleton import CoreSkeleton27
out=ROOT/'review/spatial-v2/gait-ablation';out.mkdir(parents=True,exist_ok=True)
c=RealtimeClient('http://127.0.0.1:8769',Path('/Users/jakob/Desktop/Shellhacks/.runtime/api-token').read_text())
print(json.dumps(c.health()),flush=True)
s=CoreSkeleton27(); names=s.bone_names if hasattr(s,'bone_names') else list(s.bone_index)
idx=s.bone_index
print(idx,flush=True)
def run(name, targets, histlen=4, speed=.65, prefix=None):
    chunks=[];calls=[]
    for w in range(2):
        body=dict(request_id='gait-v2-'+uuid.uuid4().hex,stage_kind='approach' if not chunks and prefix is None else 'continuation',frames=40,prompt='A person walks forward naturally.',actor_ids=['actor_1'],seed=33)
        previous=chunks[-1] if chunks else prefix
        if previous is not None: body['history']={'native_features':previous.native_features[:,-histlen:].tolist()}
        else:body['initial_placements']={'actor_1':{'position_xz':[0.,0.], 'yaw':0.}}
        if targets is not None:
            body['root_targets']={'actor_1':[dict(frame=f,position_xz=[0.,speed*(w*40+f+1)/20],**({'heading':0.} if targets=='heading' else {})) for f in ((39,) if targets=='sparse' else (7,15,23,31,39))]}
        t=time.monotonic();chunks+=c.wait(body);calls.append(time.monotonic()-t)
    p=np.concatenate([x.positions for x in chunks],axis=1)[0];r=np.concatenate([x.rotations for x in chunks],axis=1)[0];n=np.concatenate([x.native_features for x in chunks],axis=1)[0]
    np.savez_compressed(out/(name+'.npz'),positions=p[None],rotations=r[None],native_features=n[None])
    torso=p[:,idx['Head']]-p[:,idx['Hips']];lean=np.degrees(np.arctan2(np.linalg.norm(torso[:,[0,2]],axis=-1),torso[:,1]));feet=p[:,[idx['LeftFoot'],idx['RightFoot']]]
    speedfeet=np.linalg.norm(np.diff(feet[:,:,[0,2]],axis=0),axis=-1)*20;low=feet[:-1,:,1]<np.percentile(feet[:,:,1],10)+.04
    row=dict(name=name,seconds=sum(calls),lean_median=float(np.median(lean)),lean_max=float(lean.max()),root_y_min=float(p[:,0,1].min()),root_y_median=float(np.median(p[:,0,1])),distance=float(np.linalg.norm(p[-1,0,[0,2]]-p[0,0,[0,2]])),end=p[-1,0].tolist(),foot_low_speed=float(np.median(speedfeet[low])) if low.any() else None)
    print(json.dumps(row),flush=True);return row
rows=[]
for name,tgt,hist,speed in [('free',None,4,.65),('dense-heading-065','heading',4,.65),('dense-xz-065','xz',4,.65),('sparse-xz-065','sparse',4,.65),('dense-heading-120','heading',4,1.2),('dense-xz-120','xz',4,1.2),('sparse-xz-120','sparse',4,1.2),('dense-xz-120-history40','xz',40,1.2)]:
    rows.append(run(name,tgt,hist,speed))
(out/'report.json').write_text(json.dumps(rows,indent=2))
