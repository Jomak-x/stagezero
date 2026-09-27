"""Capture bounded fresh Core gestures and reproducibly build the demo atlas."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import uuid
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from realtime_client import RealtimeClient
from crowd_demo_motion_export import build_demo_library, build_flashmob_library

PROMPTS = {
 'wave': 'A person standing in place gives a friendly clear wave with their right hand raised beside their head, then lowers the hand. Both feet stay grounded. Face forward.',
 'look': 'A person standing in place calmly looks to their left then back forward, shifting weight naturally with relaxed arms. Both feet stay grounded.',
 'listen': 'A person standing in place attentively listens to a friend, nodding subtly and shifting weight naturally, arms relaxed beside the body. Both feet stay grounded. Face forward.',
}
def write(path, value):
 path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def generate(output, token_file, url):
 output.mkdir(parents=True,exist_ok=False)
 client=RealtimeClient(url,token_file.read_text().strip(),timeout=8,job_timeout=100)
 report={'started_at':datetime.now(timezone.utc).isoformat(),'health_before':client.health(),'jobs':[],'status':'running'}
 write(output/'report.json',report)
 try:
  for index,(name,prompt) in enumerate(PROMPTS.items()):
   body={'request_id':'crowd-demo-'+name+'-'+uuid.uuid4().hex,'stage_kind':'approach','frames':40,'prompt':prompt,'actor_ids':['actor'],'actor_prompts':{'actor':prompt},'seed':941+index,'initial_placements':{'actor':{'position_xz':[0.,0.],'yaw':0.}},'root_targets':{'actor':[{'frame':f,'position_xz':[0.,0.],'heading':0.} for f in (0,7,15,23,31,39)]}}
   write(output/f'{name}-request.json',body)
   job={'name':name,'request_id':body['request_id'],'status':'submitted'};report['jobs'].append(job);write(output/'report.json',report)
   start=time.monotonic();clip=client.wait(body)[0]
   job['generation_seconds']=time.monotonic()-start
   np.savez_compressed(output/f'{name}-core.npz',positions=clip.positions,rotations=clip.rotations,native_features=clip.native_features,metadata=np.array(json.dumps(clip.metadata)))
   job['status']='complete';write(output/'report.json',report)
  report['status']='complete'
 except Exception as exc:
  report['status']='failed';report['error']=type(exc).__name__+': '+str(exc);raise
 finally:
  report['finished_at']=datetime.now(timezone.utc).isoformat()
  try: report['health_after']=client.health()
  except Exception as exc: report['health_after_error']=str(exc)
  write(output/'report.json',report)
 return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--generate',action='store_true');p.add_argument('--flashmob',action='store_true');p.add_argument('--url',default='http://127.0.0.1:8769');p.add_argument('--token-file',type=Path);p.add_argument('--sources',type=Path,default=ROOT/'review/crowd-demos/motions/fresh-core');p.add_argument('--output',type=Path,default=ROOT/'review/crowd-demos/assets');a=p.parse_args()
 if a.generate:
  if not a.token_file: p.error('--token-file required for generation')
  generate(a.sources,a.token_file,a.url)
 if a.flashmob:
  import tempfile
  with tempfile.TemporaryDirectory(prefix='crowd-demo-base-') as temporary:
   base=Path(temporary);build_demo_library(base,fresh_sources=a.sources)
   result=build_flashmob_library(a.output,base=base)
 else:
  result=build_demo_library(a.output,fresh_sources=a.sources)
 print(json.dumps(result,indent=2))
if __name__=='__main__': main()
