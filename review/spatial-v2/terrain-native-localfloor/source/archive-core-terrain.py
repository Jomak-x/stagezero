import json,numpy as np,torch
from pathlib import Path
from ardy.skeleton import CoreSkeleton27
from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep
from realtime_clip import CanonicalClip
from realtime_director import RealtimeDirector
from studio_core_session import CoreStudioSession
base=Path('review/spatial-v2/terrain-native-localfloor')
scene=json.loads((base/'scene.json').read_text());source_scene=json.loads(json.dumps(scene))
for obj in scene['objects']: obj.pop('handle_height',None)
report=json.loads((base/'report.json').read_text())
rep=ArdyMotionRep(CoreSkeleton27().to('cpu'),20,stats_path='/private/tmp/core-terrain-native-v2-results/core-motion-stats')
for p in [base,Path('review/spatial-v2/terrain-native-v2')]:
 prefix=np.load(p/'prefix.npz')['motion'];out=rep.inverse(torch.from_numpy(prefix)[None],is_normalized=True)
 np.savez_compressed(p/'prefix_decoded.npz',motion=prefix,positions=out['posed_joints'][0].numpy(),rotations=out['global_rot_mats'][0].numpy())
 print(p,'prefix last root',out['posed_joints'][0,-1,0].tolist())
prefix=np.load(base/'prefix_decoded.npz')
for seed in [11,33]:
 name=f'height_sparse__seed{seed}';data=np.load(base/(name+'.npz'));case=next(c for c in report['cases'] if c['name']==name)
 decoded=rep.inverse(torch.from_numpy(data['motion'])[None],is_normalized=True)
 assert np.allclose(decoded['posed_joints'][0].numpy(),data['positions'],atol=1e-6)
 for with_prefix in [False,True]:
  pos=data['positions'];rot=data['rotations'];native=data['motion']
  if with_prefix:pos=np.concatenate([prefix['positions'],pos]);rot=np.concatenate([prefix['rotations'],rot]);native=np.concatenate([prefix['motion'],native])
  session=CoreStudioSession();placements={'actor_1':{'position_xz':pos[0,0,[0,2]].tolist(),'yaw':float(np.pi)}}
  meta=session._metadata(scene,placements)
  meta['studio_core'].update(terrain_navigation_version=1,source_scene=source_scene,terrain_native_experiment={'variant':'height_sparse','seed':seed,'model':'ARDY-Core-RP-20FPS-Horizon40','coordinate_frame':'Rigid constant support Y per 40-frame window; world restoration before FK; no relative pose or joint rotation edits','raw_native_passed':False,'metrics':case['metrics'],'prefix_frames':40 if with_prefix else 0})
  d=RealtimeDirector(('actor_1',),target_buffer_frames=240,max_buffer_frames=240,project_metadata=meta)
  d.submit_instruction('A person walks up a short flight of shallow stairs.',frames=len(pos),metadata={'terrain_navigation_version':1,'source_scene':scene,'seed':seed,'native_trial':name})
  for start in range(0,len(pos),40):
   req=d.claim_request();assert req
   clip=CanonicalClip.from_arrays(pos[None,start:start+40],rot[None,start:start+40],actor_ids=('actor_1',),source='ardy_core',native_features=native[None,start:start+40],metadata={'native_terrain_trial':name,'raw_native_passed':False})
   assert d.complete(req.request_id,clip)
  path=base/(name+('__with_prefix' if with_prefix else '')+'.core.npz');path.write_bytes(d.save_project())
  loaded=CoreStudioSession();loaded.load(path.read_bytes());saved=loaded.timeline_clip()
  assert np.array_equal(saved.positions,pos[None]);assert np.array_equal(saved.rotations,rot[None]);assert np.array_equal(saved.native_features,native[None])
  print('Archive exact roundtrip',path,len(pos))
