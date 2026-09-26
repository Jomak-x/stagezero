"""Descriptive diagnostics for unmodified native Core27 / InterGen22 motions.

Arrays are metres, +Y up, [frames, actors, joints, xyz]. Metrics never certify
semantic quality, physical contact, skin clearance, or a successful interaction.
No pose, root, floor, or timing correction is applied to the input samples.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

SCHEMAS = {
    'core27': {
        'joints': 27, 'feet': {'right': 22, 'left': 26},
        'hands': {'right': 10, 'left': 16},
        'parents': [-1,0,1,2,3,4,5,4,7,8,9,10,10,4,13,14,15,16,16,0,19,20,21,0,23,24,25],
        'angles': {'right_elbow': (8,9,10), 'left_elbow': (14,15,16),
                   'right_knee': (19,20,21), 'left_knee': (23,24,25),
                   'spine': (0,2,5), 'right_hip': (4,19,20), 'left_hip': (4,23,24),
                   'right_shoulder': (4,8,9), 'left_shoulder': (4,14,15)},
        'capsules': [('torso',0,4,.15),('upper_torso',4,5,.12),('head',6,6,.11),
                     ('right_upper_arm',8,9,.065),('right_forearm',9,10,.045),
                     ('left_upper_arm',14,15,.065),('left_forearm',15,16,.045),
                     ('right_thigh',19,20,.095),('right_calf',20,21,.065),
                     ('left_thigh',23,24,.095),('left_calf',24,25,.065)],
    },
    'intergen22': {
        'joints': 22, 'feet': {'left': 10, 'right': 11},
        'hands': {'left': 20, 'right': 21},
        'parents': [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19],
        'angles': {'left_elbow': (16,18,20), 'right_elbow': (17,19,21),
                   'left_knee': (1,4,7), 'right_knee': (2,5,8),
                   'spine': (0,6,12), 'left_hip': (9,1,4), 'right_hip': (9,2,5),
                   'left_shoulder': (9,16,18), 'right_shoulder': (9,17,19)},
        'capsules': [('torso',0,9,.15),('upper_torso',9,12,.12),('head',15,15,.11),
                     ('left_upper_arm',16,18,.065),('left_forearm',18,20,.045),
                     ('right_upper_arm',17,19,.065),('right_forearm',19,21,.045),
                     ('left_thigh',1,4,.095),('left_calf',4,7,.065),
                     ('right_thigh',2,5,.095),('right_calf',5,8,.065)],
    },
}


def stats(values):
    a = np.asarray(values, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'samples': 0, 'min': None, 'median': None, 'mean': None, 'p95': None, 'max': None}
    return {'samples': int(len(a)), 'min': float(a.min()), 'median': float(np.median(a)),
            'mean': float(a.mean()), 'p95': float(np.percentile(a,95)), 'max': float(a.max())}


def runs(mask):
    """Half-open true runs, including a final run that reaches the array end."""
    m = np.asarray(mask, dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[False,m,False]))
    return list(zip(edges[::2].tolist(),edges[1::2].tolist()))


def _array(positions, schema):
    if schema not in SCHEMAS:
        raise ValueError('schema must be core27 or intergen22')
    p = np.asarray(positions, dtype=float)
    if p.ndim == 3:
        p = p[:,None]
    if p.ndim != 4 or p.shape[2:] != (SCHEMAS[schema]['joints'],3) or min(p.shape[:2])<1:
        raise ValueError('Expected nonempty [frames, actors, schema joints, 3] positions')
    if not np.isfinite(p).all():
        raise ValueError('Native positions contain nonfinite values')
    return p


def segment_distance(a,b,c,d):
    """Exact Euclidean closest distance for broadcast segments, including points.

    The constrained quadratic minimum lies inside both segments or on one of
    their four boundaries. Evaluate those candidates instead of an unstable
    near-parallel divide followed by alternating clamps.
    """
    a,b,c,d = np.broadcast_arrays(*[np.asarray(x,dtype=float) for x in (a,b,c,d)])
    u,v,w = b-a,d-c,a-c
    aa,bb,cc = np.sum(u*u,-1),np.sum(u*v,-1),np.sum(v*v,-1)
    dd,ee = np.sum(u*w,-1),np.sum(v*w,-1)
    def point_segment(point,start,delta,length):
        t=np.clip(np.sum((point-start)*delta,-1)/np.maximum(length,1e-20),0,1)
        return np.linalg.norm(point-(start+t[...,None]*delta),axis=-1)
    candidates=[point_segment(a,c,v,cc),point_segment(b,c,v,cc),
                point_segment(c,a,u,aa),point_segment(d,a,u,aa)]
    denominator=aa*cc-bb*bb
    nonparallel=denominator>1e-12*np.maximum(aa*cc,1e-20)
    safe=np.where(nonparallel,denominator,1.)
    s,t=(bb*ee-cc*dd)/safe,(aa*ee-bb*dd)/safe
    valid=nonparallel & (s>=0) & (s<=1) & (t>=0) & (t<=1)
    inner=np.linalg.norm(w+s[...,None]*u-t[...,None]*v,axis=-1)
    candidates.append(np.where(valid,inner,np.inf))
    return np.min(np.stack(candidates),axis=0)


def _foot_metrics(p, spec, fps, floor, height, vertical_speed):
    output={}
    for side,index in spec['feet'].items():
        foot=p[:,index]
        h=foot[:,1]-floor
        delta=np.diff(foot,axis=0)
        planar=np.linalg.norm(delta[:,[0,2]],axis=1)*fps
        vertical=np.abs(delta[:,1])*fps
        contact=((h[:-1]>=-.02)&(h[:-1]<=height)&(h[1:]>=-.02)&(h[1:]<=height)&(vertical<=vertical_speed))
        episodes=[]
        for start,stop in runs(contact):
            points=foot[start:stop+1][:,[0,2]]
            episodes.append({'first_frame':start,'last_frame':stop,'duration_seconds':(stop-start)/fps,
                             'path_drift_m':float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
                             'net_drift_m':float(np.linalg.norm(points[-1]-points[0]))})
        output[side]={'joint_index':index,'height_above_declared_floor_m':stats(h),
                      'below_floor_frames':int(np.count_nonzero(h<-.02)),
                      'floor_proximity_frames':int(np.count_nonzero((h>=-.02)&(h<=height))),
                      'inferred_contact_intervals':int(contact.sum()),
                      'inferred_contact_seconds':float(contact.sum()/fps),
                      'horizontal_speed_during_inferred_contact_m_s':stats(planar[contact]),
                      'all_horizontal_speed_m_s':stats(planar),'episodes':episodes}
    return output


def _articulation(p,spec,fps):
    output={}
    for name,(a,b,c) in spec['angles'].items():
        first,second=p[:,a]-p[:,b],p[:,c]-p[:,b]
        lengths=np.linalg.norm(first,axis=-1)*np.linalg.norm(second,axis=-1)
        valid=lengths>1e-10
        angles=np.degrees(np.arccos(np.clip(np.sum(first*second,axis=-1)/np.maximum(lengths,1e-10),-1,1)))
        angles=np.where(valid,angles,np.nan)
        output[name]={'angle_degrees':stats(angles),
                      'range_degrees':float(np.nanmax(angles)-np.nanmin(angles)) if valid.any() else None,
                      'angular_speed_degrees_s':stats(np.abs(np.diff(angles))*fps),
                      'degenerate_frames':int((~valid).sum())}
    parents=np.asarray(spec['parents'][1:])
    lengths=np.linalg.norm(p[:,1:]-p[:,parents],axis=-1)
    means=np.mean(lengths,axis=0)
    return {'internal_joint_angles':output,
            'bone_length_temporal_range_m':stats(np.ptp(lengths,axis=0)),
            'bone_length_coefficient_of_variation':stats(np.std(lengths,axis=0)/np.maximum(means,1e-8)),
            'root_relative_joint_speed_m_s':stats(np.linalg.norm(np.diff(p-p[:,:1],axis=0),axis=-1)*fps),
            'interpretation':'Internal angles ignore global rigid translation/rotation. Root-relative speed also contains whole-body turns; neither measures prompt adherence.'}


def _pair_metrics(a,b,spec,fps):
    contacts={}
    for (side_a,ja),(side_b,jb) in itertools.product(spec['hands'].items(),repeat=2):
        distance=np.linalg.norm(a[:,ja]-b[:,jb],axis=-1)
        row={'wrist_distance_m':stats(distance),'closest_frame':int(np.argmin(distance)),'thresholds':{}}
        for threshold in (.10,.15,.20):
            mask=distance<=threshold
            episodes=runs(mask)
            row['thresholds'][str(threshold)]={'frames':int(mask.sum()),'seconds':float(mask.sum()/fps),
                                               'longest_same_pair_seconds':max((end-start for start,end in episodes),default=0)/fps}
        contacts[f'{side_a}_to_{side_b}']=row
    overlaps=[]
    overall=np.zeros(len(a))
    for name_a,ja,ka,ra in spec['capsules']:
        for name_b,jb,kb,rb in spec['capsules']:
            distance=segment_distance(a[:,ja],a[:,ka],b[:,jb],b[:,kb])
            depth=np.maximum(0.,ra+rb-distance)
            overall=np.maximum(overall,depth)
            if np.any(depth>0):
                worst=int(np.argmax(depth))
                overlaps.append({'actor_a_region':name_a,'actor_b_region':name_b,
                                 'radius_sum_m':ra+rb,'overlap_frames':int(np.count_nonzero(depth>0)),
                                 'max_overlap_depth_m':float(depth[worst]),'worst_frame':worst})
    overlaps.sort(key=lambda row:row['max_overlap_depth_m'],reverse=True)
    return {'root_distance_m':stats(np.linalg.norm(a[:,0]-b[:,0],axis=-1)),
            'wrist_proximity':contacts,
            'capsule_overlap':{'frames_with_any_overlap':int(np.count_nonzero(overall>0)),
                               'max_depth_per_frame_m':stats(overall),'per_region_pairs':overlaps,
                               'interpretation':'Approximate anatomical capsules, not skinned meshes. Hugging and intentional touch may register overlap; wrist contact does not establish a grasp.'}}


def analyze_native(positions, *, schema, fps, floor_height=0., contact_height=.08, contact_vertical_speed=.15):
    p=_array(positions,schema)
    if not np.isfinite(fps) or fps<=0:
        raise ValueError('fps must be finite and positive')
    if not np.isfinite([floor_height,contact_height,contact_vertical_speed]).all() or min(contact_height,contact_vertical_speed)<0:
        raise ValueError('Floor and contact thresholds must be finite, thresholds nonnegative')
    spec=SCHEMAS[schema]
    actors=[]
    for i in range(p.shape[1]):
        actor=p[:,i]
        root=actor[:,0]
        actors.append({'actor':i,'root_height_m':stats(root[:,1]),
                       'root_path_length_m':float(np.linalg.norm(np.diff(root,axis=0),axis=-1).sum()),
                       'root_net_displacement_m':float(np.linalg.norm(root[-1]-root[0])),
                       'root_speed_m_s':stats(np.linalg.norm(np.diff(root,axis=0),axis=-1)*fps),
                       'joint_step_m':stats(np.linalg.norm(np.diff(actor,axis=0),axis=-1)),
                       'feet':_foot_metrics(actor,spec,fps,floor_height,contact_height,contact_vertical_speed),
                       'articulation':_articulation(actor,spec,fps)})
    pairs=[]
    for a,b in itertools.combinations(range(p.shape[1]),2):
        pairs.append({'actor_indices':[a,b],**_pair_metrics(p[:,a],p[:,b],spec,fps)})
    return {'schema':schema,'frames':len(p),'actors_count':p.shape[1],'fps':float(fps),
            'sample_duration_seconds':len(p)/fps,'time_between_first_and_last_frame_seconds':(len(p)-1)/fps,
            'actors':actors,'pairs':pairs,
            'assumptions':{'floor_height_m':floor_height,'floor_is_declared_not_estimated':True,
                           'contact_height_m':contact_height,'contact_vertical_speed_m_s':contact_vertical_speed,
                           'contact_lower_height_m':-.02,'horizontal_speed_used_to_infer_contact':False,
                           'capsules':[{'name':n,'joints':[a,b],'radius_m':r} for n,a,b,r in spec['capsules']],
                           'input_corrections':'none',
                           'limitations':['Toe/wrist centres do not locate soles, palms, or forces.',
                                          'Articulation magnitude is not naturalness or semantic accuracy.',
                                          'Capsules cannot certify actual mesh penetration or intentional contact.',
                                          'Visual review is required; no aggregate quality pass is emitted.']}}


def analyze_skinned_feet(rig, positions, rotations, *, fps, floor_height=0., contact_height=.025,
                          contact_vertical_speed=.15, maximum_vertices_per_foot=128, fitted=None):
    """Measure stable sole vertices on the exact fitted Human17 browser skin.

    The rig's retarget and deform_vertices methods are used without correction.
    Sole membership is fixed in bind space using foot weights >= .5 and the
    lowest 15 mm of that foot. Contact uses height/vertical speed, never planar
    speed. The result describes this approximate skin fit, not native joints.
    """
    p=np.asarray(positions,dtype=float);r=np.asarray(rotations,dtype=float)
    if p.ndim!=3 or p.shape[1:]!=(27,3) or r.shape!=p.shape[:-1]+(3,3) or not len(p):
        raise ValueError('Skinned-foot diagnostics require one Core27 actor')
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not np.isfinite(fps) or fps<=0:
        raise ValueError('Finite poses and positive fps required')
    if not np.isfinite([floor_height,contact_height,contact_vertical_speed]).all() or min(contact_height,contact_vertical_speed)<0:
        raise ValueError('Invalid skinned-foot thresholds')
    if maximum_vertices_per_foot<1: raise ValueError('At least one sole vertex is required')
    if fitted is not None:
        fp=np.asarray(fitted['positions'],dtype=float);fr=np.asarray(fitted['rotations'],dtype=float)
        if fp.shape!=(len(p),17,3) or fr.shape!=(len(p),17,3,3) or not np.isfinite(fp).all() or not np.isfinite(fr).all():
            raise ValueError('Provided fitted transforms must match [frames,17,...]')
    selected={};tracks={}
    for side,bone in [('left',11),('right',14)]:
        candidates=np.flatnonzero(np.asarray(rig.weights)[:,bone]>=.5)
        if not len(candidates): raise ValueError(f'Fitted mesh has no {side} foot-weighted vertices')
        lower=float(np.asarray(rig.vertices)[candidates,1].min())
        candidates=candidates[np.asarray(rig.vertices)[candidates,1]<=lower+.015]
        if len(candidates)>maximum_vertices_per_foot:
            candidates=candidates[np.linspace(0,len(candidates)-1,maximum_vertices_per_foot,dtype=int)]
        selected[side]=candidates
        tracks[side]=[]
    for frame,(pose,rotation) in enumerate(zip(p,r)):
        fit=rig.retarget(pose,rotation) if fitted is None else {'positions':fp[frame],'rotations':fr[frame]}
        vertices=rig.deform_vertices(fit['positions'],fit['rotations'])
        for side,indices in selected.items(): tracks[side].append(vertices[indices])
    output={}
    for side,track in tracks.items():
        v=np.asarray(track);heights=v[...,1]-floor_height
        delta=np.diff(v,axis=0)
        contact=(heights[:-1]>=-.01)&(heights[:-1]<=contact_height)&(heights[1:]>=-.01)&(heights[1:]<=contact_height)&(np.abs(delta[...,1])*fps<=contact_vertical_speed)
        speed=np.linalg.norm(delta[...,[0,2]],axis=-1)*fps
        output[side]={'sole_vertices_sampled':len(selected[side]),'sole_height_above_floor_m':stats(heights),
                      'lowest_sole_vertex_each_frame_m':stats(heights.min(axis=1)),
                      'frames_with_sole_below_minus_1cm':int(np.count_nonzero((heights<-.01).any(axis=1))),
                      'frames_all_sampled_sole_above_5cm':int(np.count_nonzero((heights>.05).all(axis=1))),
                      'inferred_contact_vertex_intervals':int(contact.sum()),
                      'contact_vertex_horizontal_speed_m_s':stats(speed[contact]),
                      'all_sole_vertex_horizontal_speed_m_s':stats(speed)}
    both_air=np.logical_and.reduce([np.asarray(track)[...,1].min(axis=1)-floor_height>.05 for track in tracks.values()])
    return {'feet':output,'frames_both_soles_above_5cm':int(both_air.sum()),
            'fitted_pose_input':'provided fitted transforms' if fitted is not None else 'rig default retarget',
            'fps':float(fps),'floor_height_m':floor_height,
            'contact_height_m':contact_height,'contact_vertical_speed_m_s':contact_vertical_speed,
            'rig_provenance':getattr(rig,'provenance',{}),
            'interpretation':'Exact current LBS sole samples with approximate fitted rig; no foot locking, root correction, or mesh quality pass.'}


def analyze_fitted_pair(rigs,positions,rotations,*,fps,maximum_hand_vertices=64,fitted=None):
    """Compare raw wrists, fitted wrists and sampled rendered hand surfaces.

    Sampled vertex gap is not a triangle distance or penetration test. It can
    reveal a lost grasp caused by changed rig proportions but cannot certify it.
    """
    p=_array(positions,'core27');r=np.asarray(rotations,dtype=float)
    if p.shape[1]!=2 or len(rigs)!=2 or r.shape!=p.shape[:-1]+(3,3):
        raise ValueError('Two Core actors and matching global rotations are required')
    if not np.isfinite(r).all() or not np.isfinite(fps) or fps<=0 or maximum_hand_vertices<1:
        raise ValueError('Finite rotations, positive fps and vertex sample count required')
    if fitted is not None:
        fp=np.asarray(fitted['positions'],dtype=float);fr=np.asarray(fitted['rotations'],dtype=float)
        if fp.shape!=(2,len(p),17,3) or fr.shape!=(2,len(p),17,3,3) or not np.isfinite(fp).all() or not np.isfinite(fr).all():
            raise ValueError('Provided pair fits must have [actors,frames,17,...] axes')
    selected=[]
    for rig in rigs:
        sides={}
        for side,bone in [('left',5),('right',8)]:
            ids=np.flatnonzero(np.asarray(rig.weights)[:,bone]>=.5)
            if not len(ids): raise ValueError(f'No {side} hand vertices in fitted rig')
            _,unique=np.unique(np.round(np.asarray(rig.vertices)[ids],6),axis=0,return_index=True)
            ids=ids[np.sort(unique)]
            if len(ids)>maximum_hand_vertices: ids=ids[np.linspace(0,len(ids)-1,maximum_hand_vertices,dtype=int)]
            sides[side]=ids
        selected.append(sides)
    wrist_tracks=[[],[]];hand_tracks=[{'left':[],'right':[]} for _ in rigs]
    for frame in range(len(p)):
        for actor,rig in enumerate(rigs):
            fit=rig.retarget(p[frame,actor],r[frame,actor]) if fitted is None else {'positions':fp[actor,frame],'rotations':fr[actor,frame]}
            wrist_tracks[actor].append(fit['positions'])
            vertices=rig.deform_vertices(fit['positions'],fit['rotations'])
            for side,ids in selected[actor].items():hand_tracks[actor][side].append(vertices[ids])
    wrist_positions=np.asarray(wrist_tracks)
    output={}
    hands={'left':(16,5),'right':(10,8)}
    for (sa,(ja,fa)),(sb,(jb,fb)) in itertools.product(hands.items(),repeat=2):
        raw=np.linalg.norm(p[:,0,ja]-p[:,1,jb],axis=-1)
        fit=np.linalg.norm(wrist_positions[0,:,fa]-wrist_positions[1,:,fb],axis=-1)
        av,bv=np.asarray(hand_tracks[0][sa]),np.asarray(hand_tracks[1][sb])
        gap=np.min(np.linalg.norm(av[:,:,None]-bv[:,None,:],axis=-1),axis=(1,2))
        output[f'{sa}_to_{sb}']={'raw_wrist_gap_m':stats(raw),'fitted_wrist_gap_m':stats(fit),
                                'sampled_hand_vertex_gap_m':stats(gap),
                                'closest_sampled_hand_frame':int(np.argmin(gap)),
                                'frames_sampled_hand_gap_under_5cm':int(np.count_nonzero(gap<.05)),
                                'longest_sampled_hand_gap_under_5cm_seconds':max((b-a for a,b in runs(gap<.05)),default=0)/fps}
    return {'hand_pairs':output,'fitted_pose_input':'provided fitted transforms' if fitted is not None else 'rig default retarget',
            'fps':float(fps),'rig_provenance':[rig.provenance for rig in rigs],
            'interpretation':'Same-frame native wrist, fitted wrist and sampled exact LBS hand geometry. Surface sampling does not establish a grasp or detect mesh penetration.'}


def seam_diagnostics(previous,next_clip,*,schema,fps,window_frames=10):
    """Describe an unblended boundary and its best single shared planar alignment.

    A shared rigid transform preserves incoming body motion and pair spacing.
    It does not cure a pose mismatch, foot slide, velocity jump or timing change.
    This function reports an alignment; it does not modify either clip.
    """
    a,b=_array(previous,schema),_array(next_clip,schema)
    if a.shape[1:]!=b.shape[1:] or min(len(a),len(b))<2 or fps<=0 or not np.isfinite(fps):
        raise ValueError('Seam clips need matching actors/joints, two frames and valid fps')
    if not isinstance(window_frames,int) or window_frames<1:
        raise ValueError('window_frames must be a positive integer')
    rows=[]
    for left in range(max(1,len(a)-window_frames),len(a)):
        for right in range(min(window_frames,len(b)-1)):
            # Explicit indexing preserves [actor,joint,xy] order under NumPy.
            source=b[right][...,[0,2]].reshape(-1,2);target=a[left][...,[0,2]].reshape(-1,2)
            sc,tc=source.mean(0),target.mean(0)
            u,_,vt=np.linalg.svd((source-sc).T@(target-tc))
            correction=np.eye(2);correction[-1,-1]=np.linalg.det(u@vt)
            rotation=u@correction@vt
            translation=tc-sc@rotation
            aligned=b[right].copy();aligned[...,[0,2]]=b[right][...,[0,2]]@rotation+translation
            delta=aligned-a[left]
            incoming=(b[right+1]-b[right])*fps
            incoming[...,[0,2]]=incoming[...,[0,2]]@rotation
            outgoing=(a[left]-a[left-1])*fps
            rows.append({'previous_frame':left,'next_frame':right,
                         'raw_joint_step_m':stats(np.linalg.norm(b[right]-a[left],axis=-1)),
                         'shared_planar_alignment':{'rotation_xz':rotation.tolist(),'translation_xz_m':translation.tolist()},
                         'aligned_joint_step_m':stats(np.linalg.norm(delta,axis=-1)),
                         'aligned_velocity_jump_m_s':stats(np.linalg.norm(incoming-outgoing,axis=-1)),
                         'aligned_root_step_m':np.linalg.norm(delta[:,0],axis=-1).tolist()})
    rows.sort(key=lambda r:r['aligned_joint_step_m']['mean'])
    return {'candidate_count':len(rows),'best_geometric_candidates':rows[:5],
            'interpretation':'Ranks pose distance only; no interpolation, IK, root transport, independent actor alignment, or semantic success claim. Clip time skipped/waited is explicit in frame indices.'}


def load_archive(path,*,fps=None):
    path=Path(path)
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata'].item())) if 'metadata' in data else {}
        if 'joints' in data:
            p=data['joints'].copy();schema='intergen22'
        elif 'positions' in data:
            p=data['positions'].copy();schema='core27' if p.shape[-2]==27 else None
            if schema is None: raise ValueError(f'Unsupported native joint count in {path}')
            if p.ndim==4:
                if meta.get('array_order')=='frames_actors_joints_xyz': pass
                elif p.shape[0]<=4 and p.shape[1]>4: p=p.transpose(1,0,2,3)
                elif p.shape[1]<=4 and p.shape[0]>4: pass
                else: raise ValueError('Ambiguous Core actor/frame axes; metadata array_order is required')
        else: raise ValueError(f'No raw native joints or positions array in {path}')
    rate=fps if fps is not None else meta.get('fps')
    if rate is None: raise ValueError(f'No declared fps in {path}; pass --fps')
    return _array(p,schema),schema,float(rate),meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives',type=Path,nargs='+')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--fps',type=float)
    parser.add_argument('--floor',type=float,default=0.)
    args=parser.parse_args()
    rows=[]
    for path in args.archives:
        p,schema,fps,meta=load_archive(path,fps=args.fps)
        rows.append({'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'metadata':meta,'diagnostics':analyze_native(p,schema=schema,fps=fps,floor_height=args.floor)})
    content=json.dumps({'kind':'native motion diagnostics, not a quality certificate','clips':rows},indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(content)
    else: print(content,end='')

if __name__=='__main__': main()
