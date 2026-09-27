from pathlib import Path
import json,numpy as np
root=Path('review/group-sequence');reports=[]
for p in sorted(root.glob('**/group-sequence-*/manifest.json')):
 if '/bridge-replay/' in str(p):continue
 d=json.loads(p.read_text());report={'source_manifest':str(p.relative_to(root)),'seed':d['seed'],'status':d['status'],'error':d.get('error'),'frames':d.get('frames'),'fresh_action_stages':d.get('fresh_action_stages',False),'maximum_seam_mean_joint_m':max([s['mean_joint_jump_m'] for s in d['seams']],default=0.),'maximum_route_target_error_m':max([s['maximum_target_error_m'] for s in d['route_measurements']],default=0.),'stages':[]}
 for idx,s in enumerate(d['stages']):
  stage={'id':s['id'],'kind':s['kind'],'actors':{}}
  for aid in d['actor_ids']:
   track=p.parent/f'stage-{idx:02d}-{aid}.npz'
   if not track.exists():continue
   a=np.load(track)['joints'];v=a[:,15]-a[:,0];rootxz=a[:,0][:,[0,2]]
   foot=a[:,[7,8,10,11]];floor=np.quantile(foot[:,:,1],.05);delta=np.diff(foot,axis=0)*30
   near=(foot[:-1,:,1]<floor+.05)&(np.abs(delta[:,:,1])<.15)
   speed=np.linalg.norm(delta[:,:,[0,2]],axis=-1)[near]
   stage['actors'][aid]={'frames':len(a),'inverted_torso_frames_proxy':int((v[:,1]<0).sum()),'pelvis_height_min_max_m':[float(a[:,0,1].min()),float(a[:,0,1].max())],'maximum_root_excursion_from_stage_start_m':float(np.linalg.norm(rootxz-rootxz[0],axis=-1).max()),'relative_joint_mean_speed_m_s':float(np.linalg.norm(np.diff(a-a[:,:1],axis=0),axis=-1).mean()*30),'near_floor_slow_vertical_foot_speed_m_s_p50_p95':np.quantile(speed,[.5,.95]).tolist() if len(speed) else None}
  report['stages'].append(stage)
 reports.append(report)
(root/'measurements.json').write_text(json.dumps({'limitations':'Geometry, torso inversion and near-floor foot velocity are proxies, not proof of semantic action, airborne flip, planted feet or animation quality. Visual review is authoritative. Native22 display at30fps.','cases':reports},indent=2)+'\n')
for r in reports:print(r['source_manifest'].split('/')[0],r['status'],r['frames'],[(s['id'],[a['inverted_torso_frames_proxy'] for a in s['actors'].values()]) for s in r['stages']])
