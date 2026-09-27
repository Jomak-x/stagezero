"""Native22 foot proxies matching DemoCrowdRenderer's saved action cues.

Not shoe contact, physics, or a guarantee of visual foot planting.
"""
from pathlib import Path
import argparse,json
import numpy as np
from crowd_gait_metrics import rendered_feet,foot_metrics,source_loop_metrics,FEET

def rendered_demo_feet(manifest,poses,trajectory,sample_fps=30):
    poses=np.asarray(poses,dtype=float)
    times,feet,blend=rendered_feet(manifest,poses,trajectory,sample_fps=sample_fps)
    data=np.array([f['people'] for f in trajectory['frames']],dtype=float)
    ts=np.array([f['t'] for f in trajectory['frames']]);ix=np.minimum(len(ts)-2,np.floor(times/trajectory['dt']).astype(int));alpha=np.clip((times-ts[ix])/trajectory['dt'],0,1)
    a,b=data[ix],data[ix+1];root=a[:,:,:2]*(1-alpha[:,None,None])+b[:,:,:2]*alpha[:,None,None]
    delta=b[:,:,2]-a[:,:,2];yaw=a[:,:,2]+np.arctan2(np.sin(delta),np.cos(delta))*alpha[:,None]
    idle=next((c for c in manifest['clips'] if c['id']=='idle'),manifest['clips'][0])
    for actor in range(len(trajectory['agents'])):
        if data.shape[2]>=8:
            pair_indices=[j for j,c in enumerate(manifest['clips']) if c.get('type')=='pair']
            boundary=np.isin(a[:,actor,6],pair_indices)!=np.isin(b[:,actor,6],pair_indices)
            yaw[boundary,actor]=a[boundary,actor,2]
        c,s=np.cos(yaw[:,actor]),np.sin(yaw[:,actor])
        unit_scale=actor<6 or bool(trajectory['agents'][actor].get('unit_scale'))
        if unit_scale:
            x=feet[:,actor,:,0]-root[:,actor,0,None];z=feet[:,actor,:,2]-root[:,actor,1,None]
            baseyaw=a[:,actor,2]+np.arctan2(np.sin(delta[:,actor]),np.cos(delta[:,actor]))*alpha
            bc,bs=np.cos(baseyaw),np.sin(baseyaw)
            scale=.96+(actor*7%9)/100
            lx=(x*bc[:,None]-z*bs[:,None])/scale;lz=(x*bs[:,None]+z*bc[:,None])/scale
            feet[:,actor,:,0]=lx*c[:,None]+lz*s[:,None]+root[:,actor,0,None]
            feet[:,actor,:,2]=-lx*s[:,None]+lz*c[:,None]+root[:,actor,1,None]
            feet[:,actor,:,1]/=.94+(actor*17%13)/100
        if data.shape[2]>=8:
            for ci,clip in enumerate(manifest['clips']):
                mask=a[:,actor,6]==ci
                if not mask.any():continue
                seconds=a[:,actor,7]+np.where(b[:,actor,6]==ci,(b[:,actor,7]-a[:,actor,7])*alpha,0)
                ph=np.clip(seconds*clip['fps'],0,clip['frames']-1);left=np.floor(ph).astype(int);weight=(ph-left)[:,None,None]
                target=poses[clip['offset']+left][:,FEET]*(1-weight)+poses[clip['offset']+np.minimum(left+1,clip['frames']-1)][:,FEET]*weight
                duration=(clip['frames']-1)/clip['fps']
                neutral=np.where(seconds<duration/2,clip['offset'],clip['offset']+clip['frames']-1) if 'greeting-role' in clip['id'] else idle['offset']+(np.floor(times*idle['fps']+actor*7).astype(int)%idle['frames'])
                w=np.clip(np.minimum(seconds,duration-seconds)/.25,0,1);w=w*w*(3-2*w)
                local=poses[neutral][:,FEET]*(1-w[:,None,None])+target*w[:,None,None]
                if not unit_scale:local*=np.array([.96+(actor*7%9)/100,.94+(actor*17%13)/100,.96+(actor*7%9)/100])
                converted=np.empty_like(local);converted[:,:,0]=local[:,:,0]*c[:,None]+local[:,:,2]*s[:,None]+root[:,actor,0,None];converted[:,:,2]=-local[:,:,0]*s[:,None]+local[:,:,2]*c[:,None]+root[:,actor,1,None];converted[:,:,1]=local[:,:,1]
                feet[mask,actor]=converted[mask]
        if data.shape[2]>=9:feet[:,actor,:,1]+=(a[:,actor,8]*(1-alpha)+b[:,actor,8]*alpha)[:,None]
    return times,feet,blend

def evaluate(manifest,poses,trajectory,sample_fps=30):
    times,feet,blend=rendered_demo_feet(manifest,poses,trajectory,sample_fps)
    return {'schema':'stagezero.demo-gait-proxies.v1','scene':trajectory.get('environment',trajectory.get('scene')),'actors':len(trajectory['agents']),'sample_fps':sample_fps,'metrics':foot_metrics(times,feet,blend),'source_loops':source_loop_metrics(manifest,poses),'limitations':['Endpoint foot proxies, not deformed sole contacts or physical ground forces.','Matches explicit action cue interpolation, pair endpoint blends, shared pair yaw and unit scale for featured pairs and flagged formation actors.','Gesture and pair clips are one-shot; their endpoint differences are not played as looping seams.','No threshold here establishes visual acceptance.']}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,default=Path('review/crowd-demos/assets'));p.add_argument('--trajectory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads((a.assets/'manifest.json').read_text());poses=np.fromfile(a.assets/'poses.bin',dtype='<f4').reshape(m['totalFrames'],22,3);t=json.loads(a.trajectory.read_text());result=evaluate(m,poses,t);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'actors':result['actors'],'foot_proxy_drift':result['metrics']['proxy_bout_endpoint_drift_m']}))
