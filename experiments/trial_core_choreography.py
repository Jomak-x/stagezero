"""Run a bounded, serial real-Core choreography trial; preserve failures and exact output."""
from pathlib import Path
import argparse, json, sys, time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from studio_core_session import CoreStudioSession
from experiments.verify_studio_core import BoundedClient, pump, measure, archive_result
from core_choreography import choreography_preset

def performance_metrics(session):
    clip=session.timeline_clip()
    if clip is None:return {}
    p=clip.positions.astype(float); r=clip.rotations.astype(float)
    seams=[s['start'] for s in session.snapshot()['segments'] if s['start']>0]
    details=[]
    for f in seams:
        steps=np.linalg.norm(p[:,f]-p[:,f-1],axis=-1)
        rel=r[:,f] @ np.swapaxes(r[:,f-1],-1,-2)
        angles=np.degrees(np.arccos(np.clip((np.trace(rel,axis1=-2,axis2=-1)-1)/2,-1,1)))
        details.append({'frame':f,'joint_step_mean_m':steps.mean(axis=-1).tolist(),'joint_step_max_m':steps.max(axis=-1).tolist(),'joint_rotation_max_degrees':angles.max(axis=-1).tolist()})
    distance=np.linalg.norm(p[0,:,:,None,:]-p[1,:,None,:,:],axis=-1)
    foot=[]
    for a in range(2):
        q=p[a][:,[22,26],:]; floor=np.percentile(q[:,:,1],5,axis=0)
        low=(q[:-1,:,1]<floor+.05)&(q[1:,:,1]<floor+.05)
        speed=np.linalg.norm(np.diff(q[:,:,[0,2]],axis=0),axis=-1)*20
        foot.append({'low_foot_sample_count':int(low.sum()),'low_foot_speed_median_mps':float(np.median(speed[low])) if low.any() else None,'low_foot_speed_p95_mps':float(np.percentile(speed[low],95)) if low.any() else None,'note':'Height-based proxy, not inferred physical contact'})
    return {'horizon_boundaries':details,'minimum_cross_actor_joint_distance_m':float(distance.min()),'cross_actor_joint_pairs_under_12cm_frames':int((distance.min(axis=(1,2))<.12).sum()),'foot_slide_proxy':foot}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preset',default='feint_dodge');ap.add_argument('--plan',type=Path)
    ap.add_argument('--finish-reference',type=Path);ap.add_argument('--seed-stride',type=int,default=0);ap.add_argument('--history-frames',type=int,choices=[4,12,40],default=40);ap.add_argument('--seed',type=int,default=6201);ap.add_argument('--name',required=True)
    ap.add_argument('--token-file',type=Path,required=True);ap.add_argument('--url',default='http://127.0.0.1:8769')
    ap.add_argument('--scene',type=Path,default=ROOT/'review/scene-integration/live-city.json')
    ap.add_argument('--output',type=Path,default=ROOT/'review/two-character')
    ap.add_argument('--separation',type=float,default=1.8);ap.add_argument('--facing',choices=['front','each-other'],default='each-other')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    plan=json.loads(args.plan.read_text()) if args.plan else choreography_preset(args.preset,seed=args.seed)
    scene=json.loads(args.scene.read_text())
    placements={'actor_1':{'position_xz':[-args.separation/2,0.],'yaw':np.pi/2 if args.facing=='each-other' else 0.},'actor_2':{'position_xz':[args.separation/2,0.],'yaw':-np.pi/2 if args.facing=='each-other' else 0.}}
    client=BoundedClient(args.url,args.token_file.read_text(),timeout=10,job_timeout=90,max_jobs=8)
    reference_clip=None
    if args.finish_reference:
        with CoreStudioSession() as reference_session:
            reference_session.load_project(args.finish_reference.read_bytes());reference_clip=reference_session.timeline_clip()
    original_wait=client.wait
    def window_wait(body,*a,**kw):
        body={**body,'seed':(body['seed']+client.submitted_jobs*args.seed_stride)%(2**31)}
        if body.get('history',{}).get('native_features') is not None:
            body={**body,'history':{'native_features':[row[-args.history_frames:] for row in body['history']['native_features']]}}
        if reference_clip is not None and client.submitted_jobs == sum(b['seconds'] for b in plan['beats'])//2-1:
            from experiments.reference_finish import make_finish_target
            target,provenance=make_finish_target(reference_clip,session.timeline_clip())
            body={**body,'stage_kind':'transition','target':target}
            body.pop('root_targets',None)  # Full-body already constrains roots; avoid mixed-device constraints in existing worker.
            report['finish_reference']=provenance
            report['finish_reference']['archive']=str(args.finish_reference)
        return original_wait(body,*a,**kw)
    client.wait=window_wait
    report={'seed_stride':args.seed_stride,'conditioning_history_frames':args.history_frames,'name':args.name,'plan':plan,'scene_source':str(args.scene),'placements':placements,'status':'failed','visual_review':'pending','no_postprocess':True}
    (args.output/(args.name+'.plan.json')).write_text(json.dumps(plan,indent=2))
    with CoreStudioSession(client) as session:
        try:
            report['health']=client.health();session.start(2,scene,placements)
            report['compiled']=session.choreograph(plan)
            report['playback']=pump(session,timeout=180)
            report['status']='failed' if report['playback']['state'].get('failure') or report['playback']['state'].get('verification_timeout') else 'complete'
        except Exception as exc:report['error']=str(exc)
        finally:
            report['metrics']=measure(session);report['performance_metrics']=performance_metrics(session)
            report.update(archive_result(session,args.name,args.output))
            report['gpu_jobs']=client.submitted_jobs
            (args.output/(args.name+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:report.get(k) for k in ['name','status','error','gpu_jobs','archive']}),flush=True)
if __name__=='__main__':main()
