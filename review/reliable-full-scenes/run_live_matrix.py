"""Repeat live acceptance prompts; saves evidence without provisioning pods."""
import argparse,json,time
from pathlib import Path
import numpy as np
from live_motion import Backend
from story_jobs import StoryJobQueue
from story_planning import StoryPlanner
from takes import encode_project
from scene_acceptance import analyze_take
CASES={
'exact': 'Make the character sprint quickly forward for 20 meters then stop and fall. Get up again and start dancing and at the end doing a backflip.',
'long-variety': 'Create one continuous performance: walk forward for eight seconds, turn left for two seconds, jog forward for eight seconds, stop for two seconds, wave for four seconds, sidestep left for six seconds, sidestep right for six seconds, squat for four seconds, stand upright for two seconds, shadowbox for eight seconds, dance for ten seconds, bow for three seconds, salute for three seconds, walk backward for six seconds.',
'long-recovery': 'Make one continuous performance: jog forward for twelve seconds, stop, fall to the ground, get up fully, wave for six seconds, crouch briefly, stand upright, sidestep left for eight seconds, dance for twelve seconds, do a backflip, salute for four seconds, walk backward for eight seconds, and bow.',
'double-recovery': 'Run forward, stop, fall down, get back up, dance, fall to the floor again, get up fully again, wave, jump, land, turn around, and walk away.',
}
CASES['long-auto']=CASES['long-variety'].replace('stop for two seconds','stop').replace('stand upright for two seconds','stand upright')
class RecordingBackend(Backend):
 def __init__(self,path,token_path,url): super().__init__(token_path,url);self.path=path;self.calls=[]
 def generate(self,request_id,prompt,history,**kwargs):
  row={'prompt':prompt,'history_frames':0 if history is None else len(history),'request_id':request_id,**kwargs};self.calls.append(row)
  try:
   result=super().generate(request_id,prompt,history,**kwargs)
   row['status']='ok';row['metadata']=result['metadata']
   np.savez_compressed(self.path/f'chunk-{len(self.calls):03d}.npz',positions=result['positions'],rotations=result['rotations'],motion=result['motion'])
   return result
  except Exception as e:row.update(status='failed',error=str(e));raise
  finally:(self.path/'calls.json').write_text(json.dumps(self.calls,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--cases',default='exact,long-auto,long-recovery,double-recovery');p.add_argument('--repeats',type=int,default=2);p.add_argument('--root',default='.runtime/reliability-matrix');p.add_argument('--plans-from');p.add_argument('--token-path',default='.runtime/api-token');p.add_argument('--url',default='http://127.0.0.1:8765');a=p.parse_args()
 root=Path(a.root);root.mkdir(exist_ok=True);summary=[]
 for name in a.cases.split(','):
  case=root/name;case.mkdir(exist_ok=True)
  try:
   plan=json.loads((Path(a.plans_from)/name/'plan.json').read_text()) if a.plans_from else StoryPlanner().plan(CASES[name]);(case/'plan.json').write_text(json.dumps(plan,indent=2));print('PLAN',name,len(plan['beats']),sum(b['seconds'] for b in plan['beats']),flush=True)
  except Exception as e:summary.append({'case':name,'phase':'planning','error':str(e)});print('PLAN FAILED',name,str(e),flush=True);continue
  for repeat in range(a.repeats):
   out=case/f'run-{repeat+1}';out.mkdir(exist_ok=True);backend=RecordingBackend(out,a.token_path,a.url);queue=StoryJobQueue([backend]);identifier=queue.submit(plan,automatic=True);started=time.monotonic()
   try:
    while time.monotonic()-started<600:
     snap=queue.snapshot(identifier)
     if snap['status'] in ('completed','failed','cancelled'):break
     time.sleep(.1)
    row={'case':name,'run':repeat+1,'elapsed':round(time.monotonic()-started,2),'snapshot':snap,'calls':len(backend.calls),'rejections':sum(c['status']=='failed' for c in backend.calls)}
    if snap['status']=='completed':
     take=queue.result(identifier);(out/'scene.stagezero.npz').write_bytes(encode_project({take.id:take},take.id,0,{'gate':{'position':[0.,0.,1.5],'radius':.55,'enabled':True}}));row.update(seconds=len(take.motion)/25,movements=len(take.segments)); acceptance=analyze_take(take); (out/'acceptance.json').write_text(json.dumps(acceptance,indent=2)); row['acceptance_passed']=acceptance['passed']; row['failed_checks']=acceptance['failed_checks']
    summary.append(row);(root/'summary.json').write_text(json.dumps(summary,indent=2));print('RESULT',json.dumps(row),flush=True)
   finally:queue.cancel(identifier);queue.close()
 (root/'summary.json').write_text(json.dumps(summary,indent=2))
 return 0 if summary and all(row.get('acceptance_passed') for row in summary) else 1
if __name__=='__main__':raise SystemExit(main())
