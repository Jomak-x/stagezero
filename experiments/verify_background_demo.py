"""Bounded live native Core spatial-command review; no service provisioning.

All candidates and failures are retained. Automatic checkpoint has an authored
opening in its wall, explicitly distinct from the closed Industrial yard preset.
The temple case covers its ground courtyard, not its stairs or ravine bridge.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from realtime_client import RealtimeClient
from scene_composition import generate_recipe, validate_scene
from scene_objects import make_object
from studio_core_session import CoreStudioSession
from core_scene_reactions import object_states


def cases():
    yard=generate_recipe('Industrial yard',seed=4)
    yard['name']='Industrial checkpoint — authored doorway'
    wall=next(o for o in yard['objects'] if o['kind']=='wall')
    wall['size'][0]=3.
    wall['position'][0]=-2.5
    right={**wall,'id':'checkpoint-wall-right','position':[2.5,*wall['position'][1:]]}
    yard['objects'].append(right)
    floor=make_object('platform',99, [0,-.05,0]);floor['size']=[10,.1,10]
    yard['objects'].append(floor)
    door=next(o for o in yard['objects'] if o['kind']=='door')
    door['name']='Checkpoint door'
    return {
        'door':(yard, {'position_xz':[0,0], 'yaw':math.pi},
                'open Checkpoint door then go through Checkpoint door'),
        'grove':(generate_recipe('Enchanted grove',seed=4), {'position_xz':[0,1.],'yaw':0.},
                 'walk 1 metre forward then walk 1 metre left'),
        'temple-courtyard':(generate_recipe('Jungle temple',seed=4), {'position_xz':[0,1.5],'yaw':math.pi},
                 'walk 1 metre forward then walk 1 metre left'),
    }


class RecordingClient(RealtimeClient):
    def __init__(self,*args,directory,seed,**kwargs):
        super().__init__(*args,**kwargs)
        self.directory=directory;self.seed=seed;self.calls=[];self.clips=[]
    def wait(self,body,**kwargs):
        if len(self.calls)>=18: raise RuntimeError('Bounded review job cap reached')
        body={**body,'seed':self.seed}
        index=len(self.calls)+1
        row={'index':index,'seed':self.seed,'prompt':body['prompt'],'root_targets':body.get('root_targets')}
        self.calls.append(row)
        start=time.monotonic()
        try:
            clips=super().wait(body,**kwargs)
            row['seconds']=time.monotonic()-start
            self.clips.extend(clips)
            for c in clips:
                np.savez_compressed(self.directory/f'raw-{index:02d}.npz',positions=c.positions,
                    rotations=c.rotations,native_features=c.native_features)
            return clips
        except Exception as e:
            row['error']=str(e);raise


def run(args):
    if args.sparse_targets:
        import realtime_navigation
        realtime_navigation.SAMPLE_FRAMES = (39,)
    if args.sparse_walk:
        import core_spatial_commands
        original = core_spatial_commands.plan_navigation
        def comparison_plan(*a, **kw):
            stages, route = original(*a, **kw)
            from dataclasses import replace
            for index, stage in enumerate(stages):
                if stage.metadata.get('navigation', {}).get('phase') == 'walk':
                    stage.metadata['root_targets'] = {key: [g for g in goals if g['frame'] == 39]
                        for key, goals in stage.metadata['root_targets'].items()}
                    prompt = 'A person walks forward at an easy pace and stops.'
                    stages[index] = replace(stage, prompt=prompt, actor_prompts={key: prompt for key in stage.actor_prompts})
            return stages, route
        core_spatial_commands.plan_navigation = comparison_plan
    scene,placement,prompt=cases()[args.case]
    scene=validate_scene(scene)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'scene.json').write_text(json.dumps(scene,indent=2))
    client=RecordingClient(args.url,args.token_file.read_text(),directory=out,seed=args.seed,timeout=15,job_timeout=90)
    report={'case':args.case,'seed':args.seed,'sparse_targets':args.sparse_targets,'sparse_walk':args.sparse_walk,'prompt':prompt,'source':'ARDY-Core-RP-20FPS-Horizon40',
            'post_pose_edits':False,'temple_scope':'ground courtyard only; stairs and raised bridge unsupported'}
    start=time.monotonic()
    with CoreStudioSession(client) as session:
        session.start(1,scene,{'actor_1':placement})
        try:
            session.spatial_commands('actor_1',prompt)
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                state=session.tick()
                commands=state.get('spatial_commands') or {}
                if state['failure'] or commands.get('status') not in ('running',None):break
                time.sleep(.02)
            else: raise TimeoutError('Review exceeded 180 seconds')
            report['state']=session.snapshot()
        except Exception as e:
            report['error']=str(e)
            report['state']=session.snapshot()
        report['wall_seconds']=time.monotonic()-start
        report['calls']=client.calls
        report['inference_wait_seconds']=sum(c.get('seconds',0) for c in client.calls)
        clip=session.timeline_clip()
        if clip is not None:
            data=session.save_project();(out/'motion.core.stagezero.npz').write_bytes(data)
            report['frames']=clip.frames;report['duration_seconds']=clip.frames/clip.fps
            report['max_root_step_m']=float(np.linalg.norm(np.diff(clip.positions[:,:,0],axis=1),axis=-1).max())
            accepted=[];remaining=clip.frames
            for raw in client.clips:
                if remaining<=0:break
                accepted.append(raw);remaining-=raw.frames
            report['native_positions_exact']=np.array_equal(clip.positions,np.concatenate([c.positions for c in accepted],axis=1))
            report['native_rotations_exact']=np.array_equal(clip.rotations,np.concatenate([c.rotations for c in accepted],axis=1))
            report['native_features_exact']=np.array_equal(clip.native_features,np.concatenate([c.native_features for c in accepted],axis=1))
            report['final_objects']=object_states(scene,clip,enabled=report['state'].get('scene_reactions_enabled',False),start_frame=report['state'].get('scene_reactions_start_frame',0))
            with CoreStudioSession() as restored:
                restored.load_project(data)
                c=restored.timeline_clip()
                report['exact_save_load']=all(np.array_equal(getattr(clip,k),getattr(c,k)) for k in ('positions','rotations','native_features'))
        (out/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:report.get(k) for k in ('case','error','frames','duration_seconds','wall_seconds','inference_wait_seconds','native_positions_exact','exact_save_load')}))
        print(json.dumps(report['state'].get('spatial_commands')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=tuple(cases()),required=True)
    p.add_argument('--seed',type=int,default=33)
    p.add_argument('--sparse-walk',action='store_true',help='Diagnostic only: endpoint walking with dense turns/stops')
    p.add_argument('--sparse-targets',action='store_true',help='Diagnostic only: compare one endpoint constraint per window')
    p.add_argument('--url',default='http://127.0.0.1:8769')
    p.add_argument('--token-file',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
