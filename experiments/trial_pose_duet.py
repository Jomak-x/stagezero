"""Bounded native duet using reviewed generated pose cues; never splices output."""
import sys,json,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from studio_core_session import CoreStudioSession
from experiments.reference_finish import make_finish_target
from experiments.verify_studio_core import BoundedClient,pump,measure,archive_result
from experiments.trial_core_choreography import performance_metrics

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--token-file',type=Path,required=True);ap.add_argument('--name',default='pose-duet-v1');ap.add_argument('--seed',type=int,default=6201);args=ap.parse_args()
 out=ROOT/'review/two-character';refs={}
 for key,name in [('kick','athletic-h4'),('duck','feint-ai-v2-h4'),('victory','athletic-h4-stride')]:
  with CoreStudioSession() as s:s.load_project((out/(name+'.core.stagezero.npz')).read_bytes());refs[key]=s.timeline_clip()
 cues=[None,[('kick',1,[107,108,109,110]),('duck',0,[63,64,65,66])],[('duck',0,[63,64,65,66]),('kick',1,[107,108,109,110])],[('victory',1,[236,237,238,239])]*2]
 prompts=[('A martial artist performs sharp punches and a front kick.','A boxer dodges punches, ducking and weaving.'),('A martial artist performs a high front kick.','A boxer ducks deeply under a high kick.'),('A boxer ducks deeply under a high kick.','A martial artist performs a high front kick.'),('A person raises both arms high overhead in victory.','A person raises both arms high overhead in victory.')]
 plan={'version':1,'name':'Pose-cued sparring duet','seed':args.seed,'beats':[{'name':n,'seconds':2,'actor_prompts':dict(zip(('actor_1','actor_2'),p))} for n,p in zip(['Opening','Lead kick and dodge','Reply kick and dodge','Shared victory'],prompts)]}
 report={'name':args.name,'plan':plan,'status':'failed','visual_review':'pending','conditioning_history_frames':4,'seed_policy':'base plus window index','reference_cues':[],'no_postprocess':True}
 client=BoundedClient('http://127.0.0.1:8769',args.token_file.read_text(),max_jobs=4,job_timeout=90)
 wait=client.wait
 def run(body,*a,**kw):
  index=client.submitted_jobs;body={**body,'seed':args.seed+index}
  if 'history' in body:body['history']={'native_features':[row[-4:] for row in body['history']['native_features']]}
  if cues[index]:
   targets=[];prov=[]
   for role,(key,actor,frames) in enumerate(cues[index]):
    target,meta=make_finish_target(refs[key],session.timeline_clip(),source_actor=actor,source_frames=frames)
    targets.append({k:v[role] for k,v in target.items()});prov.append({'role':role,'cue':key,**meta})
   body['stage_kind']='transition';body['target']={k:[t[k] for t in targets] for k in ('positions','rotations')};report['reference_cues'].append({'window':index,'provenance':prov})
  return wait(body,*a,**kw)
 client.wait=run
 with CoreStudioSession(client) as session:
  try:
   session.start(2,json.loads((ROOT/'review/scene-integration/live-city.json').read_text()),{'actor_1':{'position_xz':[-.9,0],'yaw':1.5707963267948966},'actor_2':{'position_xz':[.9,0],'yaw':-1.5707963267948966}})
   session.choreograph(plan);report['playback']=pump(session,timeout=120);report['status']='failed' if report['playback']['state']['failure'] else 'complete'
  except Exception as exc:report['error']=str(exc)
  finally:
   report['metrics']=measure(session);report['performance_metrics']=performance_metrics(session);report.update(archive_result(session,args.name,out));report['gpu_jobs']=client.submitted_jobs;(out/(args.name+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False))
 print(json.dumps({k:report.get(k) for k in ['name','status','error','gpu_jobs']}))
if __name__=='__main__':main()
