"""GPU affine demo atlas, retaining exact common placement for native pairs.

Core gesture labels describe prompts, not guaranteed semantic completion.
Pair root removal is reversible using rootXZ; never rotate roles separately.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from crowd_motion_library import (REPO_ROOT, OUTPUT_FPS, build_library, read_core_actor,
                                  resample, solve_affines, sha256, repo_path, request_evidence)
from experiments.native_pair_rig import NativeRigAsset
from native_pair_clip import load_source

PAIR_SOURCE = 'review/interaction-quality/after/market-three-s48-fixed-plan/sources/scene-e8470498b43c45f39ed5f0c4b3614fa1/pair-01.npz'

def root_center(poses, *, origin_xz=(0.,0.), floor_y=0.):
    """Remove only per-frame root XZ, preserving world facing and root height."""
    p=np.array(poses,dtype=np.float64,copy=True)
    if p.ndim!=3 or p.shape[1:]!=(22,3) or not len(p) or not np.isfinite(p).all():
        raise ValueError('Expected finite native22 poses[T,22,3]')
    origin=np.asarray(origin_xz,dtype=float)
    if origin.shape!=(2,) or not np.isfinite(origin).all() or not np.isfinite(floor_y):
        raise ValueError('Expected finite XZ origin and floor')
    tracks=p[:,0][:,[0,2]]-origin
    p[:,:,0]-=p[:,0,0].copy()[:,None]
    p[:,:,2]-=p[:,0,2].copy()[:,None]
    p[:,:,1]-=floor_y
    return p, tracks

def reconstruct(poses,tracks):
    p=np.array(poses,copy=True);p[:,:,0]+=np.asarray(tracks)[:,0,None];p[:,:,2]+=np.asarray(tracks)[:,1,None];return p

def build_demo_library(output:Path, *, fresh_sources:Path):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    specs=json.loads((REPO_ROOT/'review/crowd-crossing/generation/accepted-sources.json').read_text())
    specs=[s for s in specs if s['type'] in ('walk','idle')]
    rig_path=REPO_ROOT/'assets/paired/Xbot.glb'
    build_library(specs,output,rig_path=rig_path)
    manifest=json.loads((output/'manifest.json').read_text())
    poses=list(np.fromfile(output/'poses.bin',dtype='<f4').reshape(-1,22,3))
    transforms=list(np.fromfile(output/'affine.bin',dtype='<f4').reshape(-1,22,3,4))
    asset=NativeRigAsset(rig_path)
    for clip,spec in zip(manifest['clips'],specs):
        p=np.asarray(poses[clip['offset']:clip['offset']+clip['frames']])
        paths=[REPO_ROOT/part for part in spec.get('paths',[spec.get('path')])]
        raw=np.concatenate([read_core_actor(path,int(spec['actor']))[0] for path in paths])
        start=int(spec.get('start_frame',0));end=int(spec.get('end_frame_inclusive',len(raw)-1))
        raw=resample(raw[start:end+1]);heading=clip['calibration']['source_heading_rad']
        c,s=np.cos(-heading),np.sin(-heading);rotation=np.array([[c,s],[-s,c]])
        tracks=(raw[:,0][:,[0,2]]-raw[0,0,[0,2]])@rotation.T
        clip.update(duration=(clip['frames']-1)/clip['fps'],loop=True,startPose=p[0].tolist(),endPose=p[-1].tolist(),rootXZ=tracks.tolist(),sourceRootOriginXZ=raw[0,0,[0,2]].tolist(),rootTrackUsage='source travel retained; navigation supplies crowd locomotion')
    def append(identifier,p,tracks,kind,source,**extra):
        p=np.asarray(p,dtype='<f4');affine=solve_affines(asset,p)
        if not np.isfinite(affine).all():raise ValueError('Nonfinite affine')
        clip={'id':identifier,'type':kind,'fps':OUTPUT_FPS,'frames':len(p),'offset':len(poses),'strideMeters':0.,'speed':0.,'duration':(len(p)-1)/OUTPUT_FPS,'loop':False,'rootXZ':np.asarray(tracks).tolist(),'startPose':p[0].tolist(),'endPose':p[-1].tolist(),'source':source,'calibration':{'loop_seam_rms_m':float(np.sqrt(np.mean((p[-1]-p[0])**2))),'max_root_residual_m':float(np.abs(p[:,0][:,[0,2]]).max())},**extra}
        manifest['clips'].append(clip);poses.extend(p);transforms.extend(affine)
        return clip
    for name in ('wave','look','listen'):
        path=fresh_sources/f'{name}-core.npz'
        if not path.is_file():continue
        raw,meta=read_core_actor(path,0);sampled=resample(raw)
        floor=float(np.percentile(sampled[:,[7,8,10,11],1],3));p,tracks=root_center(sampled,floor_y=floor)
        source={'model':'ARDY Core','files':[{'path':repo_path(path),'sha256':sha256(path)}],'actor_index':0,'source_fps':20,'source_frames':len(raw),'requests':[request_evidence(path,meta)],'fresh_for_crowd_demo':True,'review':'Generated prompt intent; numeric pose validation; full crowd visual review required'}
        source['observed_action']={'wave':'Raised hand greeting gesture; hand has not fully lowered by sample end','look':'Very subtle upright idle; leftward look is not visually established','listen':'Both hands rise while gesturing; conversational gesture, not quiet listening'}[name]
        append(name,p,tracks,'idle' if name=='look' else 'gesture',source,loop=name=='look',rootTrackUsage='retained for provenance; stationary navigation uses centered body',floorReferenceY=floor,semanticLimit='Prompt label; no classifier verifies gesture meaning')
    pair_path=REPO_ROOT/PAIR_SOURCE;pair=load_source(pair_path);q=np.asarray(pair.joints,dtype=float)
    origin=q[0,:,0][:,[0,2]].mean(axis=0)
    # One common floor offset preserves all interactor distances, including vertical.
    floor=float(np.percentile(q[:,:,[7,8,10,11],1],3))
    indices=[]
    across=q[:,:,1]-q[:,:,2]
    facing=np.arctan2(-across[:,:,2],across[:,:,0])
    for actor in range(2):
        p,tracks=root_center(q[:,actor],origin_xz=origin,floor_y=floor)
        source={'model':'InterGen','files':[{'path':PAIR_SOURCE,'sha256':sha256(pair_path)}],'actor_index':actor,'source_fps':30,'source_frames':len(q),'request_id':pair.metadata.get('request_id'),'prompt':pair.metadata.get('prompt'),'seed':pair.metadata.get('seed'),'generation_seconds':pair.metadata.get('generation_seconds'),'source_revision':pair.metadata.get('source_revision'),'review':'Reused market-three-s48 reviewed source; wrist proximity supports greeting, fingers use authored rest pose'}
        indices.append(len(manifest['clips']))
        append(f'greeting-role-{actor}',p,tracks,'pair',source,pairId='greeting',pairRole=actor,startFacingYaw=float(facing[0,actor]),endFacingYaw=float(facing[-1,actor]),rootTrackUsage='required: shared event yaw rotates this rootXZ about event anchor; both role instance yaws equal shared event yaw',floorReferenceY=floor)
    hands=q[:,:,[20,21]];nearest=np.linalg.norm(hands[:,0,:,None]-hands[:,1,None,:],axis=-1).min(axis=(1,2))
    manifest.update(schema='stagezero_crowd_demo_motion_v1',totalFrames=len(poses),transformShape=[len(poses),22,3,4],pairs=[{'id':'greeting','clipIndices':indices,'duration':(len(q)-1)/30,'fps':30,'frames':len(q),'sourceOriginXZ':origin.tolist(),'startFacingYaws':facing[0].tolist(),'endFacingYaws':facing[-1].tolist(),'startRootXZ':[manifest['clips'][i]['rootXZ'][0] for i in indices],'endRootXZ':[manifest['clips'][i]['rootXZ'][-1] for i in indices],'instanceYaw':'shared event yaw for both roles; do not separately face actors','instanceScale':[1,1,1],'nearestWristMinMeters':float(nearest.min()),'framesWristUnder15cm':int((nearest<.15).sum()),'nativeBodyPoseUnmodified':True}])
    manifest['rig']['core27_used']=True
    manifest['sourceSchemas']=['Core27 semantic subset to native22','InterGen native22 unchanged body positions']
    manifest['limitations']+=['Pair roles must share event rotation and unit scale to retain source interaction spacing.','Gesture prompts are intentions; short generated samples can underperform wave/look/listen semantics.','Root-centered pair reconstruction preserves body positions, not palm or finger contact.','Core idle and gesture seams are repeated samples, not newly generated long actions.']
    np.asarray(poses,dtype='<f4').tofile(output/'poses.bin');np.asarray(transforms,dtype='<f4').tofile(output/'affine.bin')
    (output/'manifest.json').write_text(json.dumps(manifest,separators=(',',':'),allow_nan=False))
    summary={'clips':[{'index':i,'id':c['id'],'type':c['type'],'frames':c['frames'],'duration':c['duration']} for i,c in enumerate(manifest['clips'])],'totalFrames':len(poses),'rig_sha256':asset.sha256,'pairs':manifest['pairs'],'files':{n:{'bytes':(output/n).stat().st_size,'sha256':sha256(output/n)} for n in ('manifest.json','affine.bin','poses.bin')}}
    (output/'motion-summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary


def build_flashmob_library(output: Path, *, base: Path | None = None) -> dict:
    """Stage a 25 s synchronized routine from reviewed Core dance sources.

    This never mutates the published base. Short authored bridges are listed by
    frame alongside original generated spans. Original root motion is retained.
    """
    import shutil
    base=Path(base or REPO_ROOT/'review/crowd-demos/assets');output=Path(output)
    if output.resolve()==base.resolve():raise ValueError('Flashmob export must use a staging directory')
    output.mkdir(parents=True,exist_ok=True)
    for name in ('manifest.json','affine.bin','poses.bin'):shutil.copyfile(base/name,output/name)
    m=json.loads((output/'manifest.json').read_text())
    if any(c['id']=='flashmob-routine' for c in m['clips']):raise ValueError('Base already contains flashmob')
    poses=list(np.fromfile(output/'poses.bin',dtype='<f4').reshape(-1,22,3));affines=list(np.fromfile(output/'affine.bin',dtype='<f4').reshape(-1,22,3,4))
    rig=NativeRigAsset(REPO_ROOT/'assets/paired/Xbot.glb');phrases=[];phrase_sources=[];phrase_indices=[]
    specs=[('dance-knee-step','dance-three-s73',3,'Alternating knee lifts and rhythmic steps with relaxed arm swing'),('dance-sway','dance-three-s74',0,'Small rhythmic weight shifts and arm movement')]
    def append(identifier,p,track,source,**extra):
        p=np.asarray(p,dtype='<f4');a=solve_affines(rig,p)
        if not np.isfinite(a).all():raise ValueError('Nonfinite dance affine')
        xz=p[:,:,[0,2]];world=xz+np.asarray(track)[:,None,:]
        c={'id':identifier,'type':'dance','fps':30.,'frames':len(p),'duration':(len(p)-1)/30.,'offset':len(poses),'speed':0.,'strideMeters':0.,'loop':False,'rootXZ':np.asarray(track).tolist(),'startPose':p[0].tolist(),'endPose':p[-1].tolist(),'source':source,'rootTrackUsage':'required local slot offset; same phase/yaw/unit scale for every dancer','bodyBoundsXZ':{'rootCenteredMin':xz.min(axis=(0,1)).tolist(),'rootCenteredMax':xz.max(axis=(0,1)).tolist(),'withRootMin':world.min(axis=(0,1)).tolist(),'withRootMax':world.max(axis=(0,1)).tolist(),'rootCenteredRadius':float(np.linalg.norm(xz,axis=-1).max()),'rootExcursion':float(np.linalg.norm(track,axis=-1).max()),'skinMarginMeters':.15},**extra}
        i=len(m['clips']);m['clips'].append(c);poses.extend(p);affines.extend(a);return i
    for identifier,case,first,observed in specs:
        directory=next((REPO_ROOT/'review/group-motion-probe/fresh'/case).glob('*/sources/*'))
        paths=[directory/f'core-{i:03d}.npz' for i in range(first,first+3)]
        parts=[read_core_actor(path,0) for path in paths];q=resample(np.concatenate([x[0] for x in parts]))
        floor=float(np.percentile(q[:,[7,8,10,11],1],3));p,track=root_center(q,origin_xz=q[0,0,[0,2]],floor_y=floor)
        source={'model':'ARDY Core','files':[{'path':repo_path(path),'sha256':sha256(path)} for path in paths],'requests':[part[1].get('request',{}) for part in parts],'source_fps':20,'source_frames':120,'actor_index':0,'observed_action':next(s[3] for s in specs if s[0]==identifier),'review_video':f'review/group-motion-probe/video/{case}/playback.mp4','review':'Existing mesh video/contact sheet and fresh skeleton pose sheet reviewed; simple in-place dance, not complex choreography','floorReferenceY':floor}
        phrase_indices.append(append(identifier,p,track,source));phrases.append((p[:161],track[:161]));phrase_sources.append(source)
    neutral=np.asarray(poses[m['clips'][2]['offset']],dtype=float);neutral[:,0]-=neutral[0,0];neutral[:,2]-=neutral[0,2]
    routine=[neutral];roots=[np.zeros(2)];spans=[]
    def bridge(p,r,n,label):
        lo=len(routine);a=routine[-1].copy();b=roots[-1].copy()
        for i in range(1,n+1):
            u=i/n;u=u*u*(3-2*u);routine.append(a*(1-u)+p*u);roots.append(b*(1-u)+r*u)
        spans.append({'kind':'authored_pose_root_bridge','start_frame':lo,'end_frame_exclusive':len(routine),'label':label})
    for sequence,which in enumerate((0,1,0,1)):
        p,r=phrases[which];bridge(p[0],r[0],18,'settle into phrase' if sequence==0 else 'change dance phrase')
        lo=len(routine);routine.extend(p[1:]);roots.extend(r[1:]);spans.append({'kind':'generated_source','start_frame':lo,'end_frame_exclusive':len(routine),'source_clip_index':phrase_indices[which],'source_window_inclusive':[1,160]})
    bridge(neutral,np.zeros(2),38,'settle and return to formation slot')
    assert len(routine)==751
    index=append('flashmob-routine',routine,roots,{'model':'ARDY Core sources with authored assembly','phrases':phrase_sources,'authored_spans':spans,'generated_source_intervals':640,'authored_bridge_intervals':110,'review':'Synchronized repeated step/sway phrases; original source root motion retained; transitions disclosed'},authoredSpans=spans,unitScale=True)
    # Measure the actual authored shoe/skin vertices, not only foot joints.
    # A rate-bounded upper envelope supplies an explicit display-only rootY track.
    routine_clip=m['clips'][index];minimum_y=[]
    skin_parts=[(np.asarray(v['localBind']).reshape(-1,4,3),np.asarray(v['bones']).reshape(-1,4),np.asarray(v['weights']).reshape(-1,4)) for v in m['parts']]
    for f in range(routine_clip['frames']):
        matrix=np.asarray(affines[routine_clip['offset']+f]);lowest=float('inf')
        for bind,bones,weight in skin_parts:
            rows=matrix[bones,1,:];y=((rows[:,:,:3]*bind).sum(axis=-1)+rows[:,:,3])*weight
            lowest=min(lowest,float(y.sum(axis=1).min()))
        minimum_y.append(lowest)
    required=np.maximum(0.,-np.asarray(minimum_y)+.002)
    def envelope(values):
        values=np.asarray(values).copy();step=.25/30
        for i in range(1,len(values)):values[i]=max(values[i],values[i-1]-step)
        for i in range(len(values)-2,-1,-1):values[i]=max(values[i],values[i+1]-step)
        return values
    lift=envelope(required)
    smooth=np.convolve(np.pad(lift,(2,2),mode='edge'),np.array([1,2,3,2,1])/9,mode='valid')
    lift=envelope(np.maximum(required,smooth))
    routine_clip['groundLiftY']=lift.tolist()
    routine_clip['groundCorrection']={'method':'Authored whole-body vertical display translation from actual full-skinned-mesh minimum sole Y; smoothed rate-bounded upper envelope, no pose or XZ edits','minMeshYBefore':float(min(minimum_y)),'minMeshYAfter':float(np.min(np.asarray(minimum_y)+lift)),'maxSoleClearanceMeters':float(np.max(np.asarray(minimum_y)+lift)),'p95SoleClearanceMeters':float(np.percentile(np.asarray(minimum_y)+lift,95)),'maxLiftMeters':float(lift.max()),'startLiftMeters':float(lift[0]),'endLiftMeters':float(lift[-1]),'maxVerticalStepMeters':float(np.abs(np.diff(lift)).max()),'maxVelocityMetersPerSecond':float(np.abs(np.diff(lift)).max()*30),'safetyMarginMeters':.002,'appliedBy':'Person[8], separate from original pose/affine and rootXZ','entryExit':'Smooth 1s whole-body lift ramp during settled approach before dance and descent after dance'}
    m['totalFrames']=len(poses);m['transformShape']=[len(poses),22,3,4]
    m['flashmob']={'clipIndex':index,'duration':25.,'frames':751,'fps':30.,'phraseClipIndices':phrase_indices,'unitScale':True,'synchronized':True,'rootStartXZ':roots[0].tolist(),'rootEndXZ':roots[-1].tolist(),'recommendedSpacingMeters':2.6,'reservedSlotRadiusMeters':1.2,'bodyBoundsXZ':m['clips'][index]['bodyBoundsXZ'],'groundCorrection':routine_clip['groundCorrection'],'assembly':'Four generated 5.333 s source spans with 0.6 s entry/inter-phrase bridges and 1.267 s settled exit; source roots preserved inside formation slots'}
    m['limitations']+=['Flashmob shares generated dance phrases across all dancers in synchronized formation; it is not a jointly generated 24-person model output.','Routine uses authored interpolation between generated phrases and neutral bookends; bridge intervals are disclosed.','Native source foot motion is retained through rootXZ; no contact or foot-lock solver is claimed.']
    np.asarray(poses,dtype='<f4').tofile(output/'poses.bin');np.asarray(affines,dtype='<f4').tofile(output/'affine.bin')
    (output/'manifest.json').write_text(json.dumps(m,separators=(',',':'),allow_nan=False))
    summary={'flashmob':m['flashmob'],'totalFrames':len(poses),'source_phrases':[{'index':i,'id':m['clips'][i]['id'],'bodyBoundsXZ':m['clips'][i]['bodyBoundsXZ']} for i in phrase_indices],'files':{n:{'bytes':(output/n).stat().st_size,'sha256':sha256(output/n)} for n in ('manifest.json','affine.bin','poses.bin')}}
    (output/'motion-summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
