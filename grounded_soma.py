"""Explicit SOMA77 motion to Core27 *layout* adapter for generated characters.

The output retains SOMA body proportions, root path, endpoint trajectories and
joint motion. It is not a refit to official Core27 bone lengths. One extra spine
point is a named midpoint, and Core HandEnd uses SOMA Middle2. Raw source clips
are never rewritten. Sampling uses local SO(3) interpolation and native FK.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from motion_bridge import _layout, _fk_numpy, _swing


DIRECT_MAP={
    'Hips':'Hips','Spine':'Spine1','Spine1':'Spine2','Spine3':'Chest',
    'Neck':'Neck2','Head':'Head',
    **{side+part:side+part for side in ('Left','Right')
       for part in ('Shoulder','Arm','ForeArm','Hand','HandThumb1','Foot','ToeBase')},
    **{side+'HandEnd':side+'HandMiddle2' for side in ('Left','Right')},
    **{side+'UpLeg':side+'Leg' for side in ('Left','Right')},
    **{side+'Leg':side+'Shin' for side in ('Left','Right')},
}
_AXES={
    'Spine3':('Spine3','Head','Chest','Head'),
    **{side+part:(side+part,side+child,side+source,side+source_child)
       for side in ('Left','Right')
       for part,child,source,source_child in (
           ('Shoulder','Arm','Shoulder','Arm'),('Arm','ForeArm','Arm','ForeArm'),
           ('ForeArm','Hand','ForeArm','Hand'),('UpLeg','Leg','Leg','Shin'),
           ('Leg','Foot','Shin','Foot'),('Foot','ToeBase','Foot','ToeBase'))},
}


def _skeleton(skeleton):
    if isinstance(skeleton,(str,Path)):
        skeleton=json.loads(Path(skeleton).read_text())
    names=list(skeleton['names']);parents=np.asarray(skeleton['parents'],dtype=int)
    rest=np.asarray(skeleton['rest'],dtype=float)
    if len(names)!=77 or len(set(names))!=77 or parents.shape!=(77,) or rest.shape!=(77,3):
        raise ValueError('Expected named SOMA77 hierarchy and rest positions')
    if names[0]!='Hips' or parents[0]!=-1 or any(p<0 or p>=j for j,p in enumerate(parents[1:],1)):
        raise ValueError('SOMA hierarchy must be topologically sorted with Hips as root')
    missing=set(DIRECT_MAP.values())-set(names)
    if missing or not np.isfinite(rest).all():
        raise ValueError(f'Missing or invalid named SOMA anatomy: {sorted(missing)}')
    return names,parents,rest


def soma77_to_core27(positions,rotations,skeleton,*,source_fps=30.,target_fps=20.,metadata=None):
    """Return positions[A,T,27,3], rotations[A,T,27,3,3], conversion metadata.

    Source rotations must be global matrices in the standard SOMA T-pose
    convention: source rest offsets transformed by parent rotations reproduce
    source joint positions. This is checked rather than silently guessed.
    """
    names,parents,rest=_skeleton(skeleton)
    src={name:i for i,name in enumerate(names)}
    p=np.asarray(positions,dtype=float);r=np.asarray(rotations,dtype=float)
    if p.ndim==3:p,r=p[None],r[None]
    if p.ndim!=4 or p.shape[2:]!=(77,3) or r.shape!=p.shape[:-1]+(3,3) or p.shape[1]<2:
        raise ValueError('Expected native SOMA positions[A,T>=2,77,3] and matching global rotations')
    if not np.isfinite(p).all() or not np.isfinite(r).all():raise ValueError('Native motion must be finite')
    if min(source_fps,target_fps)<=0 or not np.isfinite([source_fps,target_fps]).all():raise ValueError('Frame rates must be positive and finite')
    if np.max(np.abs(r.swapaxes(-1,-2)@r-np.eye(3)))>.01 or np.min(np.linalg.det(r))<.99:
        raise ValueError('Native rotations must be proper SO(3) matrices')
    offsets=rest-rest[np.maximum(parents,0)]
    expected=p[:,:,parents[1:]]+np.einsum('atbij,bj->atbi',r[:,:,parents[1:]],offsets[1:])
    fk_error=float(np.max(np.linalg.norm(expected-p[:,:,1:],axis=-1)))
    if fk_error>.005:
        raise ValueError(f'Unverified SOMA rotation/rest convention: native FK error {fk_error:.6f}m')
    old_times=np.arange(p.shape[1])/float(source_fps)
    new_times=np.arange(int(np.floor(old_times[-1]*target_fps+1e-8))+1)/float(target_fps)
    local=r.copy()
    local[:,:,1:]=r[:,:,parents[1:]].swapaxes(-1,-2)@r[:,:,1:]
    sampled_local=np.empty((len(p),len(new_times),77,3,3))
    roots=np.empty((len(p),len(new_times),3))
    for actor in range(len(p)):
        for joint in range(77):
            sampled_local[actor,:,joint]=Slerp(old_times,Rotation.from_matrix(local[actor,:,joint]))(new_times).as_matrix()
        for axis in range(3):roots[actor,:,axis]=np.interp(new_times,old_times,p[actor,:,0,axis])
    sampled_p,sampled_r=_fk_numpy(sampled_local,roots,parents,offsets)
    core_names,_,core_rest=_layout();core={name:i for i,name in enumerate(core_names)}
    output_p=np.empty((len(p),len(new_times),27,3));output_r=np.empty((len(p),len(new_times),27,3,3))
    for target,source in DIRECT_MAP.items():
        output_p[:,:,core[target]]=sampled_p[:,:,src[source]]
        output_r[:,:,core[target]]=sampled_r[:,:,src[source]]
    # Core has four spine links while SOMA has three; this extra diagnostic
    # point is explicitly derived from named adjacent source links.
    output_p[:,:,core['Spine2']]=(sampled_p[:,:,src['Spine2']]+sampled_p[:,:,src['Chest']])*.5
    for actor in range(len(p)):
        for frame in range(len(new_times)):
            a=sampled_r[actor,frame,src['Spine2']];b=sampled_r[actor,frame,src['Chest']]
            output_r[actor,frame,core['Spine2']]=a@Rotation.from_rotvec(Rotation.from_matrix(a.T@b).as_rotvec()*.5).as_matrix()
    # Re-express standard SOMA joint axes as Core-style T-pose axes while
    # preserving source global twist. Long-segment directions remain native.
    for target,(ca,cb,sa,sb) in _AXES.items():
        correction=_swing(core_rest[core[cb]]-core_rest[core[ca]],rest[src[sb]]-rest[src[sa]])
        output_r[:,:,core[target]]=output_r[:,:,core[target]]@correction
    for side in ('Left','Right'):
        correction=_swing(core_rest[core[side+'Hand']]-core_rest[core[side+'ForeArm']],
            rest[src[side+'Hand']]-rest[src[side+'ForeArm']])
        output_r[:,:,core[side+'Hand']]=output_r[:,:,core[side+'Hand']]@correction
    info=dict(metadata or {})
    info['original_source']=info.get('original_source',info.get('source','native SOMA motion'))
    info['source']=f'Kimodo RP v1.1 — named SOMA→Core27 retarget, {target_fps:g}fps' if info.get('model')=='kimodo-soma-rp-v1.1' else f'named SOMA→Core27 retarget, {target_fps:g}fps'
    info['fps']=float(target_fps)
    info['soma_conversion']={
        'adapter':'SOMA77 native proportions in Core27 layout',
        'source_fps':float(source_fps),'target_fps':float(target_fps),
        'source_frames':p.shape[1],'target_frames':len(new_times),
        'source_duration_seconds':float(old_times[-1]),'target_duration_seconds':float(new_times[-1]),
        'source_fk_error_max_m':fk_error,'mapping':DIRECT_MAP,
        'derived_joint':{'Spine2':'midpoint of SOMA Spine2 and Chest'},
        'compressed_chains':{'Neck':['Chest','Neck1','Neck2'],
            'LeftHandEnd':['LeftHand','LeftHandMiddle1','LeftHandMiddle2'],
            'RightHandEnd':['RightHand','RightHandMiddle1','RightHandMiddle2']},
        'sampling':'uniform timestamps; local SO(3) SLERP; native fixed-length FK; linear root translation',
        'world_transform':'none; Y-up metres, native global root path retained',
        'rotation_conversion':'named anatomical rest-axis alignment from standard SOMA T-pose to Core-style axes',
        'limitations':['Uses native SOMA proportions, not canonical Core27 bone lengths.',
            'Compressed neck/finger chains can change endpoint distance during native articulation; this is not bone stretching.',
            'Core27 layout omits detailed fingers, jaw and eyes; raw SOMA archive retains them.']}
    return output_p,output_r,info


def convert_file(source_path,output_path,skeleton_path,*,target_fps=20.):
    """Write a separate converted archive; refuse to overwrite the native file."""
    source_path,output_path=Path(source_path),Path(output_path)
    if source_path.resolve()==output_path.resolve():raise ValueError('Raw native archive must remain untouched')
    digest=hashlib.sha256(source_path.read_bytes()).hexdigest()
    with np.load(source_path,allow_pickle=False) as data:
        metadata=json.loads(str(data['metadata'])) if 'metadata' in data else {}
        p,r,m=soma77_to_core27(data['positions'],data['rotations'],skeleton_path,
            source_fps=float(metadata.get('fps',30)),target_fps=target_fps,metadata=metadata)
    m['soma_conversion'].update({'raw_source_file':source_path.name,'raw_source_sha256':digest})
    np.savez_compressed(output_path,positions=p,rotations=r,metadata=json.dumps(m))
    return {'path':str(output_path),'positions_shape':list(p.shape),'metadata':m}
