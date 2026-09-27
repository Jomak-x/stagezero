"""Check saved crowd roots against the actual adapted background geometry.

Conservative body cylinders vs oriented primitive bounds; not skin collisions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def check(scene,trajectory,radius=.29,height=2.1):
    positions=np.array([[person[:2] for person in frame['people']] for frame in trajectory['frames']])
    hits=[];minimum=float('inf');tested=0
    for box in scene['boxes']:
        cx,cy,cz=box['center'];sx,sy,sz=box['size']
        if cy+sy/2<.25 or cy-sy/2>height:continue
        yaw=box.get('yawRadians',0);c=np.cos(yaw);s=np.sin(yaw)
        dx=positions[...,0]-cx;dz=positions[...,1]-cz
        lx=c*dx-s*dz;lz=s*dx+c*dz
        separation=np.hypot(np.maximum(abs(lx)-sx/2,0),np.maximum(abs(lz)-sz/2,0))-radius
        minimum=min(minimum,float(separation.min()));tested+=1
        n=int((separation<0).sum())
        if n:hits.append({'origin':box['origin'],'actor_samples':n,'minimum_clearance_m':float(separation.min())})
    # Whole rectangular foundation: linear interpolation of roots stays inside
    # this convex footprint, so the radius inset proves continuous root support.
    foundations=[b for b in scene['boxes'] if b.get('origin',{}).get('label')=='asphalt foundation']
    support=False;floor_y=None
    for b in foundations:
        cx,cy,cz=b['center'];sx,sy,sz=b['size'];floor_y=cy+sy/2
        if abs(floor_y)<=.011 and abs(b.get('yawRadians',0))<1e-8:
            support=bool((abs(positions[...,0]-cx)+radius<=sx/2).all() and (abs(positions[...,1]-cz)+radius<=sz/2).all())
            if support:break
    return {'actors':len(trajectory['agents']),'sample_frames':len(positions),'body_proxy_radius_m':radius,'body_proxy_y_range_m':[.25,height],'tested_solid_primitive_bounds':tested,'solid_contact_actor_samples':sum(x['actor_samples'] for x in hits),'contacts':hits,'minimum_solid_proxy_clearance_m':minimum if tested else None,'continuous_root_disc_support_on_foundation':support,'foundation_top_y_m':floor_y,'limitations':['Oriented box bounds conservatively represent every generated primitive; not triangle mesh contact.','Root support within1.1cm of root plane is not sole grounding or physical foot contact.','Body checks sample saved15Hz roots; a moving limb can reach beyond the cylinder.']}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review',type=Path,default=Path('review/crowd-crossing'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    scene_path=a.review/'assets/crossing_scene.json';scene=json.loads(scene_path.read_text());report={'schema':'stagezero.crowd-scene-proxy.v1','scene_sha256':hashlib.sha256(scene_path.read_bytes()).hexdigest(),'results':[]}
    for n in [16,32,64,100]:
        path=a.review/f'trajectories-{n}.json';item=check(scene,json.loads(path.read_text()));item['trajectory_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();report['results'].append(item)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:x[k] for k in ['actors','solid_contact_actor_samples','continuous_root_disc_support_on_foundation']} for x in report['results']]))
if __name__=='__main__':main()
