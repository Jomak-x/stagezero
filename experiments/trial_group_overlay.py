"""Optional requested third-actor action over exact saved pair tracks."""
from pathlib import Path
import argparse,json,time,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cast_performance import decode_project
from independent_group_motion import overlay_third_track
from realtime_client import RealtimeClient

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--token',type=Path,required=True);p.add_argument('--model-lane-reserved',action='store_true');a=p.parse_args()
 if not a.model_lane_reserved:p.error('Reserve existing Core lane first')
 r=ROOT/'review/group-motion-probe';client=RealtimeClient('http://127.0.0.1:8769',a.token.read_text(),job_timeout=60)
 cases=[('city-handshake-s42','A person stands in place and waves hello with one hand several times. Friendly upright posture, feet stay near the starting spot.',81,0),('industrial-spar-s45','A person celebrates happily in place, raising both arms and cheering, then lowering their arms. Stay near the starting spot.',82,201)]
 for name,prompt,seed,start in cases:
  source=r/'baselines'/(name+'.cast.stagezero.npz');data=source.read_bytes();clip,_,_,_=decode_project(data)
  parent=r/'overlays'/name;parent.mkdir(parents=True,exist_ok=True);n=1
  while (parent/f'attempt-{n:03d}').exists():n+=1
  out=parent/f'attempt-{n:03d}';out.mkdir();t=time.monotonic();record={'case':name,'seed':seed,'prompt':prompt,'start_frame':start,'end_frame':clip.frames,'source_sha256':hashlib.sha256(data).hexdigest(),'code_sha256':hashlib.sha256((ROOT/'independent_group_motion.py').read_bytes()).hexdigest()}
  try:
   result=overlay_third_track(client,data,prompt=prompt,start_frame=start,end_frame=clip.frames,seed=seed,output_root=out/'sources')
   target=out/'scene.cast.stagezero.npz';target.write_bytes(result['project_bytes']);reloaded,_,_,_=decode_project(target.read_bytes())
   record.update(status=result['status'],fallback=result['fallback'],reason=result.get('reason'),pair_payload_exact=reloaded.joints[:,:2].tobytes()==clip.joints[:,:2].tobytes(),third_outside_interval_exact=reloaded.joints[:start,2].tobytes()==clip.joints[:start,2].tobytes(),manifest=str(result['manifest'].relative_to(ROOT)))
  except Exception as e:record.update(status='error',error=str(e))
  record['wall_seconds']=time.monotonic()-t;(out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':main()
