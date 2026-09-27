import json
from pathlib import Path
import numpy as np
from native_pair_clip import load_source
from realtime_clip import CanonicalClip
from paired_meetup import build_meetup
base=Path('review/two-character/city-meetup')
scene=json.loads(Path('review/scene-integration/live-city.json').read_text())
class Replay:
 def __init__(self, folder): self.files=sorted(folder.glob('approach-*.core.npz')); self.i=0
 def wait(self, request, **kwargs):
  with np.load(self.files[self.i], allow_pickle=False) as a:
   saved=json.loads(str(a['metadata']))['request']
   assert saved['root_targets']==request['root_targets']
   clip=CanonicalClip(a['positions'],a['rotations'],20,tuple(request['actor_ids']),'ardy_core',{},a['native_features'])
  self.i+=1;return [clip]
results=[]
folders=[base/'handshake-ui-sources']+[next((base/name/'sources').iterdir()) for name in ['sparring-live','crossed-sparring-live','embrace-live','standing-hug-live']]
for folder in folders:
 r=json.loads((folder/'request.json').read_text());client=Replay(folder)
 try:
  out=build_meetup(load_source(folder/'source-pair.npz'), client, scene, actor_ids=r['actor_ids'], starts=r['starts'], meeting=r['meeting'], seed=r['seed'])
  old=load_source(folder/'composed.npz')
  exact=np.array_equal(old.joints,out['clip'].joints)
  assert exact
  results.append({'source_folder':str(folder),'accepted_by_current_mechanical_gates':True,'exact_replayed_joints':exact,'entry_root_height_gaps_m':out['metadata']['entry_root_height_gaps_m'],'core_windows':client.i})
 except ValueError as exc:
  assert folder.parent.parent.name=='embrace-live',str(exc)
  results.append({'source_folder':str(folder),'accepted_by_current_mechanical_gates':False,'rejection':str(exc),'core_windows':client.i})
print(json.dumps(results,indent=2))
(base/'replay-guards.json').write_text(json.dumps(results,indent=2)+'\n')
