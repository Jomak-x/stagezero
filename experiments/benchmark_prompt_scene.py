"""Real-service sequential intent-cache benchmark; inspect outputs before acceptance."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prompt_scene_plan import ScenePromptPlanner
from prompt_scene_builder import PromptSceneBuilder
from native_pair_provider import NativePairProvider
from realtime_client import RealtimeClient
from cast_performance import encode_project,cast_from_performance
import argparse
parser=argparse.ArgumentParser(description='Generate two fresh motion variations with one shared intent planner; never reuse motion output.')
parser.add_argument('--scene',type=Path,required=True)
parser.add_argument('--config',type=Path,required=True)
parser.add_argument('--token',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--prompt',default='Two people greet each other with a handshake.')
parser.add_argument('--core-url',default='http://127.0.0.1:8769')
parser.add_argument('--first-seed',type=int,default=47)
args=parser.parse_args()
planner=ScenePromptPlanner()
provider=NativePairProvider.from_config(args.config)
core=RealtimeClient(args.core_url,args.token.read_text())
scene=json.loads(args.scene.read_text())
out=args.output
out.mkdir(exist_ok=False)
results=[]
for n,seed in enumerate([args.first_seed,args.first_seed+1]):
 d=out/str(n+1);d.mkdir()
 start=time.monotonic()
 try:
  clip=PromptSceneBuilder(args.prompt,planner,provider,core,d/'sources',seed)(scene)
  (d/'scene.cast.stagezero.npz').write_bytes(encode_project(clip,cast_from_performance(clip),scene_document=scene))
  (d/'result.json').write_text(json.dumps(clip.metadata,indent=2))
  results.append({'attempt':n+1,'seed':seed,'wall_seconds':time.monotonic()-start,'planning_seconds':clip.metadata['plan']['planning_seconds'],'planner_cache_hit':clip.metadata['plan']['planner_cache_hit'],'frames':clip.frames,'status':'complete'})
 except Exception as e:results.append({'attempt':n+1,'seed':seed,'wall_seconds':time.monotonic()-start,'status':'rejected','error':str(e)})
 (out/'results.json').write_text(json.dumps(results,indent=2))
 print(json.dumps(results[-1]),flush=True)
