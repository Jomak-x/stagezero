from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import json,hashlib,tempfile,numpy as np
from cast_performance import decode_project
from prompt_scene_builder import PromptSceneBuilder
from experiments.trial_prompt_scene import ReplayPairProvider
D=Path('review/group-prompt-integration');reports=[]
for p in sorted(D.glob('*/attempt-001/scene.cast.stagezero.npz')):
 old,cast,scene,_=decode_project(p.read_bytes());folder=next((p.parent/'sources').glob('scene-*'));manifest=json.loads((folder/'manifest.json').read_text());clips=[]
 for source in manifest['sources']:
  if source['source']!='ardy_core':continue
  source_path=Path(source['path'])
  if not source_path.exists():source_path=folder/source_path.name
  with np.load(source_path) as a:
   meta=json.loads(a['metadata'].item());clips.append((meta['request'],SimpleNamespace(actor_ids=tuple(meta['actor_ids']),fps=meta['fps'],frames=a['positions'].shape[1],positions=a['positions'].copy(),native_features=a['native_features'].copy(),rotations=a['rotations'].copy())))
 class Replay:
  def __init__(self):self.index=0
  def wait(self,request,cancelled):
   expected,clip=clips[self.index];self.index+=1
   strip=lambda r:{k:v for k,v in r.items() if k!='request_id'}
   assert strip(request)==strip(expected), (p.parent.parent.name,self.index,'request changed')
   return [clip]
 core=Replay();provider=ReplayPairProvider(folder/'manifest.json') if any(s['source']=='intergen' for s in manifest['sources']) else None
 planner=SimpleNamespace(plan=lambda *a,**kw:deepcopy(manifest['plan']))
 with tempfile.TemporaryDirectory(prefix='final-group-replay-') as temp:
  new=PromptSceneBuilder(manifest['prompt'],planner,provider,core,temp,manifest['seed'])(scene)
  assert core.index==len(clips)
  assert old.joints.tobytes()==new.joints.tobytes(),p
 reports.append({'case':p.parent.parent.name,'core_requests_exact':core.index,'all_display_arrays_byte_identical':True,'final_code_sha256':hashlib.sha256(Path('prompt_scene_builder.py').read_bytes()).hexdigest()})
 print(reports[-1],flush=True)
(D/'final-replay.json').write_text(json.dumps(reports,indent=2)+'\n')
