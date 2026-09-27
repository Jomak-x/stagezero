"""Apply generated animation cues inside pre-reserved crowd meeting pockets.

This transforms trajectories before final navigation validation/export. Pair roots
come from the native source; entry/exit paths are disclosed authored blends.
Unrelated actor paths never change, and gestures never replace moving gaits.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import numpy as np


def _smooth(value):
    x=float(np.clip(value,0.,1.));return x*x*(3.-2.*x)


def _angle(a,b,u):
    return a+math.atan2(math.sin(b-a),math.cos(b-a))*u


def _rotate(point,yaw):
    x,z=point;c,s=math.cos(yaw),math.sin(yaw)
    return np.array([c*x+s*z,-s*x+c*z])


def sample_root(clip,seconds):
    track=np.asarray(clip['rootXZ'],dtype=float)
    if track.shape!=(clip['frames'],2) or not np.isfinite(track).all():
        raise ValueError('Invalid source root track')
    f=float(np.clip(seconds*clip['fps'],0,len(track)-1));lo=int(f);hi=min(lo+1,len(track)-1)
    return track[lo]*(1-f+lo)+track[hi]*(f-lo)


def apply_motion_cues(trajectory,manifest):
    """Return a copy; caller MUST run crowd_demo_navigation.validate before save."""
    if trajectory.get('motion_cues'):
        raise ValueError('Motion cues already applied; start from original navigation')
    result=copy.deepcopy(trajectory);frames=result['frames'];times=np.array([f['t'] for f in frames],dtype=float)
    if len(times)<2 or not np.all(np.diff(times)>0):raise ValueError('Strictly increasing frame times required')
    for frame in frames:
        for p in frame['people']:
            if len(p)<6:raise ValueError('Trajectory requires distance field')
            p[6:]=[-1,0.,0.]
    id_to_index={a['id']:i for i,a in enumerate(result['agents'])}
    if len(id_to_index)!=len(result['agents']):raise ValueError('Duplicate actor IDs')
    clips=manifest['clips'];pair=next(p for p in manifest['pairs'] if p['id']=='greeting')
    occupied=set();pair_events=0
    for ev in result.get('events',[]):
        if ev.get('kind')!='greeting':continue
        actors=[id_to_index[i] for i in ev['actor_ids']]
        if len(actors)!=2 or any(i in occupied for i in actors):raise ValueError('Overlapping pair role ownership')
        occupied.update(actors)
        start=float(ev['start_s']);end=float(ev['end_s']);clip_start=float(ev['clip_start_s']);clip_end=clip_start+float(pair['duration'])
        if not times[0]<=start<clip_start<clip_end<end<=times[-1]:raise ValueError('Pair source does not fit reserved event')
        yaw=float(ev.get('shared_yaw',0.));anchor=np.asarray(ev['anchor_xz'],dtype=float)
        radius=float(ev['reserved_radius_m'])
        if anchor.shape!=(2,) or not np.isfinite(anchor).all() or radius<=0:raise ValueError('Invalid meeting pocket')
        for role,actor in enumerate(actors):
            c=clips[pair['clipIndices'][role]]
            first=np.asarray(ev['actor_anchors_xz'][role],dtype=float)
            native_start=anchor+_rotate(sample_root(c,0),yaw)
            native_end=anchor+_rotate(sample_root(c,c['duration']),yaw)
            initial_frame=int(np.argmin(abs(times-start)));exit_frame=int(np.argmin(abs(times-end)))
            entering_yaw=frames[initial_frame]['people'][actor][2];leaving_yaw=frames[exit_frame]['people'][actor][2]
            facing_start=yaw+pair['startFacingYaws'][role];facing_end=yaw+pair['endFacingYaws'][role]
            for index,t in enumerate(times):
                if not start<=t<=end:continue
                p=frames[index]['people'][actor]
                if t<clip_start:
                    u=_smooth((t-start)/(clip_start-start));position=first*(1-u)+native_start*u
                    p[2]=_angle(entering_yaw,facing_start,u)
                elif t<=clip_end:
                    clip_time=float(t-clip_start);position=anchor+_rotate(sample_root(c,clip_time),yaw)
                    p[2]=yaw;p[6]=pair['clipIndices'][role];p[7]=clip_time
                else:
                    u=_smooth((t-clip_end)/(end-clip_end));position=native_end*(1-u)+first*u
                    p[2]=_angle(facing_end,leaving_yaw,u)
                if np.linalg.norm(position-anchor)+result['agents'][actor]['radius']>radius:
                    raise ValueError('Pair root leaves reserved meeting pocket')
                p[0:2]=[float(position[0]),float(position[1])];p[4]=3
            result['agents'][actor]['unit_scale']=True
        ev.update(clip_end_s=clip_end,motion='InterGen native pair roots and body; authored entry and exit placement',pair_id='greeting',clip_indices=pair['clipIndices'],start_facing_yaws=[yaw+x for x in pair['startFacingYaws']],end_facing_yaws=[yaw+x for x in pair['endFacingYaws']])
        pair_events+=1
    # Recompute distance and speed only for roots we modified, retaining all others.
    for actor in occupied:
        pos=np.array([f['people'][actor][:2] for f in frames]);steps=np.linalg.norm(np.diff(pos,axis=0),axis=1)
        distance=np.r_[0,np.cumsum(steps)]
        speed=np.r_[steps/np.diff(times),0.]
        for index,frame in enumerate(frames):
            frame['people'][actor][3]=float(speed[index]);frame['people'][actor][5]=float(distance[index])
    lookup={c['id']:i for i,c in enumerate(clips)};dwell_events=[]
    for actor,agent in enumerate(result['agents']):
        for task_index,task in enumerate(agent.get('schedule',[])):
            if task['task'] not in ('watch','browse'):continue
            begin=float(task['start'])+.6+((actor*3+task_index)%5)*.12
            end=float(task['end'])-.55
            cue_number=0
            while begin+1.9333333334<=end:
                choices=('look',)
                name=choices[(actor+task_index+cue_number)%len(choices)];clip_index=lookup[name];clip=clips[clip_index]
                finish=begin+clip['duration'];indices=np.flatnonzero((times>=begin)&(times<=finish))
                # Check adjacent samples too: no gesture may bleed into navigation.
                support=np.flatnonzero((times>=begin-result['dt'])&(times<=finish+result['dt']))
                if len(indices) and all(frames[i]['people'][actor][3]<=.025 and frames[i]['people'][actor][6]<0 for i in support):
                    for i in indices:
                        p=frames[i]['people'][actor];p[6]=clip_index;p[7]=min(float(clip['duration']),max(0.,float(times[i]-begin)))
                    dwell_events.append({'actor_id':agent['id'],'clip_index':clip_index,'clip_id':name,'start_s':begin,'end_s':finish,'task_index':task_index})
                begin=finish+.55+(actor%3)*.15;cue_number+=1
    result['motion_cues']={'schema':'crowd_demo_cues_v1','pair_events':pair_events,'dwell_gesture_events':len(dwell_events),'dwell_cues':dwell_events,'atlas':'assets/manifest.json','validation_required':'crowd_demo_navigation.validate on this transformed trajectory before export','pair_placement':'common source rotation, root tracks preserved; smooth authored 12–14s entry and source-end–20s exit; conservative reserved pockets','gesture_policy':'ordinary stationary watch/browse use subtle look/idle only; no random waves or conversation gestures'}
    _apply_flashmob(result,manifest)
    result.pop('metrics',None)
    return result


def _apply_flashmob(result,manifest):
    """Add source-root dance only inside navigator-owned reserved slots."""
    events=[e for e in result.get('events',[]) if e.get('kind')=='flashmob']
    if not events:return
    if result.get('motion_cues',{}).get('flashmob_applied'):
        raise ValueError('Flashmob roots already applied')
    spec=manifest.get('flashmob')
    if not spec:raise ValueError('Flashmob event requires staged dance atlas')
    clip=manifest['clips'][spec['clipIndex']];frames=result['frames'];times=np.array([f['t'] for f in frames])
    ids={a['id']:i for i,a in enumerate(result['agents'])};dancers=set()
    state_names=result.setdefault('state_names',['walking','watching','browsing','greeting'])
    if 'dancing' not in state_names:state_names.append('dancing')
    dance_state=state_names.index('dancing')
    for ev in events:
        start=float(ev['dance_start_s']);end=float(ev['dance_end_s']);yaw=float(ev.get('shared_yaw',0.))
        if abs(end-start-clip['duration'])>1e-6:raise ValueError('Dance window must match exact 25 second routine')
        slots=np.asarray(ev['slot_xz'],dtype=float);actors=[ids[i] for i in ev['actor_ids']]
        if len(actors)<24 or slots.shape!=(len(actors),2) or not np.isfinite(slots).all():raise ValueError('Flashmob requires at least 24 finite formation slots')
        separation=np.linalg.norm(slots[:,None]-slots[None,:],axis=-1);np.fill_diagonal(separation,np.inf)
        if separation.min()<spec['recommendedSpacingMeters']-1e-5:raise ValueError('Formation slots violate reviewed full body clearance')
        radius=float(ev.get('reserved_radius_m',0.))
        if radius<spec['reservedSlotRadiusMeters']:raise ValueError('Formation pocket too small for retained source roots')
        samples=np.flatnonzero((times>=start)&(times<=end))
        if len(samples)<2 or abs(times[samples[0]]-start)>1e-6 or abs(times[samples[-1]]-end)>1e-6:raise ValueError('Frame grid must include dance start and end')
        for role,actor in enumerate(actors):
            if actor in dancers:raise ValueError('Duplicate dancer assignment')
            dancers.add(actor);agent=result['agents'][actor];agent['unit_scale']=True;agent['crew']='flashmob';agent['dance_role']=role
            for i in samples:
                p=frames[i]['people'][actor]
                if np.linalg.norm(np.asarray(p[:2])-slots[role])>.025:raise ValueError('Dancer must settle at reserved slot before source root application')
                elapsed=float(times[i]-start);offset=_rotate(sample_root(clip,elapsed),yaw)
                if np.linalg.norm(offset)+clip['bodyBoundsXZ']['rootCenteredRadius']+.15>radius:raise ValueError('Dance body exceeds reserved slot bound')
                root=slots[role]+offset;p[0:2]=[float(root[0]),float(root[1])];p[2]=yaw;p[4]=dance_state;p[6:]=[spec['clipIndex'],elapsed,float(np.interp(elapsed*clip['fps'],np.arange(clip['frames']),clip.get('groundLiftY',np.zeros(clip['frames']))))]
            # Keep instance height continuous at clip boundaries: lift while the
            # crew settles and lower gradually during the first departure second.
            ground=clip.get('groundLiftY')
            if ground:
                for i,t in enumerate(times):
                    if start-1<=t<start:frames[i]['people'][actor][8]=float(ground[0])*_smooth(t-(start-1))
                    elif end<t<=end+1:frames[i]['people'][actor][8]=float(ground[-1])*(1-_smooth(t-end))
        ev.update(clip_index=spec['clipIndex'],motion='Synchronized reviewed Core step/sway phrases, retained source roots, disclosed authored bridges',unit_scale=True,source_root_track=True)
    for actor in dancers:
        p=np.array([f['people'][actor][:2] for f in frames]);steps=np.linalg.norm(np.diff(p,axis=0),axis=-1);distance=np.r_[0,np.cumsum(steps)];speed=np.r_[steps/np.diff(times),0.]
        for i,frame in enumerate(frames):frame['people'][actor][3]=float(speed[i]);frame['people'][actor][5]=float(distance[i])
    result.setdefault('motion_cues',{})['flashmob_applied']={'actor_ids':[result['agents'][i]['id'] for i in sorted(dancers)],'clip_index':spec['clipIndex'],'duration':25.,'unit_scale':True,'root_tracks_retained':True}


def refresh_existing_cues(trajectory,manifest):
    """Remove ordinary gestures without reapplying any existing paired roots.

    City actor 63 is the explicitly authored cafe service role and is retained.
    Root geometry is untouched unless a new reserved flashmob event is present.
    """
    result=copy.deepcopy(trajectory);preserve={63} if result.get('environment')=='city' else set()
    cleared=0
    for frame in result['frames']:
        for actor,p in enumerate(frame['people']):
            if result['agents'][actor]['id'] in preserve:continue
            if len(p)>6 and p[6]>=0:
                kind=manifest['clips'][int(p[6])]['type']
                if kind in ('idle','gesture'):
                    p[6:]=[-1,0.,0.];cleared+=1
    info=result.setdefault('motion_cues',{});info['ordinary_gesture_policy']='ordinary watch/browse use subtle base idle; no random waving or conversation gestures';info['gesture_policy']=info['ordinary_gesture_policy'];info['ordinary_gesture_samples_cleared']=cleared;info['dwell_cues']=[];info['dwell_gesture_events']=0
    _apply_flashmob(result,manifest);result.pop('metrics',None)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--manifest',type=Path,default=Path('review/crowd-demos/assets/manifest.json'));p.add_argument('--output',type=Path,required=True);p.add_argument('--refresh-existing',action='store_true');a=p.parse_args()
    from crowd_demo_navigation import validate
    operation=refresh_existing_cues if a.refresh_existing else apply_motion_cues
    data=operation(json.loads(a.source.read_text()),json.loads(a.manifest.read_text()))
    data['metrics']=validate(data)
    a.output.write_text(json.dumps(data,separators=(',',':'),allow_nan=False))
    print(json.dumps({'metrics':data['metrics'],'pair_events':data['motion_cues']['pair_events'],'dwell_gesture_events':data['motion_cues']['dwell_gesture_events']},indent=2))
if __name__=='__main__':main()
