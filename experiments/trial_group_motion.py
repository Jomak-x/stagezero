"""Run optional frozen group probes on an explicitly reserved existing Core lane."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from realtime_client import RealtimeClient
from cast_performance import encode_project,cast_from_performance
from independent_group_motion import generate_independent_tracks

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',default='all');p.add_argument('--token',type=Path,required=True);p.add_argument('--core-url',default='http://127.0.0.1:8769');p.add_argument('--model-lane-reserved',action='store_true');a=p.parse_args()
 if not a.model_lane_reserved:p.error('Reserve shared Core lane first')
 review=ROOT/'review/group-motion-probe';cases=json.loads((review/'cases.json').read_text());cases=[c for c in cases if a.case in ('all',c['case'])]
 if not cases:p.error('Unknown case')
 client=RealtimeClient(a.core_url,a.token.read_text(),job_timeout=60)
 for case in cases:
  parent=review/'fresh'/case['case'];parent.mkdir(parents=True,exist_ok=True);number=1
  while (parent/f'attempt-{number:03d}').exists():number+=1
  out=parent/f'attempt-{number:03d}';out.mkdir()
  scene_path=ROOT/'review/prompt-scenes/backgrounds'/f'{case["background"]}.json';scene=json.loads(scene_path.read_text())
  record={'case':case,'status':'generating','scene_sha256':digest(scene_path),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'code_sha256':digest(ROOT/'independent_group_motion.py'),'scope':'independent Core motion; explicit prompts/layout, no external planner, no joint contact model'}
  (out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n');start=time.monotonic();print('START '+case['case'],flush=True)
  try:
   r=generate_independent_tracks(client,scene,actors=case['actors'],seconds=case['seconds'],seed=case['seed'],output_root=out/'sources')
   if r['performance'] is not None:
    target=out/'scene.cast.stagezero.npz';target.write_bytes(encode_project(r['performance'],cast_from_performance(r['performance']),scene_document=scene))
   else:target=r['research_path']
   record.update(status='geometry_accepted_visual_unreviewed',archive=str(target.relative_to(ROOT)),archive_sha256=digest(target),manifest=str(r['manifest'].relative_to(ROOT)))
  except Exception as exc:record.update(status='rejected',error=f'{type(exc).__name__}: {exc}')
  record['wall_seconds']=time.monotonic()-start;(out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k in ('status','error','archive','wall_seconds')}),flush=True)
if __name__=='__main__':main()
