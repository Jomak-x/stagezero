"""Bounded AI crowd intent. One planning call, deterministic navigation thereafter."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import time
from object_generation import GatewayGenerator
KEYS={'title','speed_min_mps','speed_max_mps','diagonal_fraction','group_fraction','side_weights','style_weights'}
def validate_crowd_intent(raw):
    if not isinstance(raw,dict) or set(raw)!=KEYS:raise ValueError('Invalid crowd intent keys')
    if not isinstance(raw['title'],str) or not 1<=len(raw['title'])<=100:raise ValueError('Invalid title')
    out=dict(raw)
    for k,lo,hi in [('speed_min_mps',.65,1.2),('speed_max_mps',1.2,1.65),('diagonal_fraction',.2,.5),('group_fraction',0,.35)]:
        v=raw[k]
        if type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi:raise ValueError('Invalid '+k)
    if out['speed_min_mps']>=out['speed_max_mps']:raise ValueError('Speed range must increase')
    for k,names in [('side_weights',{'north','east','south','west'}),('style_weights',{'casual','brisk','relaxed'})]:
        vals=raw[k]
        if not isinstance(vals,dict) or set(vals)!=names or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in vals.values()):raise ValueError('Invalid '+k)
        out[k]={n:v/sum(vals.values()) for n,v in vals.items()}
    return out

def plan_crowd(prompt,gateway,*,count=64,seed=42):
    if type(count) is not int or count not in (16,32,64,100) or type(seed) is not int or not 0<=seed<2**32:raise ValueError('Invalid count or seed')
    system=('Return JSON for a pedestrian scramble crossing. Keys exactly title,speed_min_mps,speed_max_mps,diagonal_fraction,group_fraction,side_weights,style_weights. Choose min speed .65–1.2, max speed 1.2–1.65 metres/second, diagonal fraction .2–.5, group fraction 0–.35. side_weights has positive numeric north,east,south,west weights; style_weights has positive casual,brisk,relaxed weights. Local navigation enforces authored geometry and signal phases. Plan diverse walking and small groups; no code or physical guarantees.')
    start=time.perf_counter();raw=gateway.request_json(system,prompt,max_tokens=800,timeout_seconds=45);intent=validate_crowd_intent(raw)
    config={k:v for k,v in intent.items() if k!='title'};config.update(count=count,seed=seed,duration_s=120,sample_hz=15)
    return dict(schema='stagezero_crowd_plan_v1',prompt=prompt,intent=intent,config=config,raw_plan=raw,planner_model=gateway.model,planning_seconds=time.perf_counter()-start,planner_calls=1)

def load_private_config(path):
    allowed={'STAGEZERO_OBJECT_API_BASE','STAGEZERO_OBJECT_MODEL','STAGEZERO_OBJECT_API_KEY','NEON_AI_GATEWAY_BASE_URL','NEON_AI_GATEWAY_TOKEN','STAGEZERO_SCENE_LAYOUT_MODEL'}
    p=Path(path)
    if p.stat().st_size>16384:raise ValueError('Configuration too large')
    for line in p.read_text().splitlines():
        parts=shlex.split(line,comments=True)
        if parts and parts[0]=='export':parts=parts[1:]
        if len(parts)==1 and '=' in parts[0]:
            k,v=parts[0].split('=',1)
            if k in allowed:os.environ.setdefault(k,v)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--private-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    load_private_config(a.private_config)
    result=plan_crowd('64 pedestrians at a Shibuya-style scramble crossing: opposing and diagonal streams, varied walking speeds and gaits, small groups, wait at curbs for green, cross and settle on the opposite sidewalk.',GatewayGenerator.from_env(stage='layout'))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'planning_seconds':result['planning_seconds'],'model':result['planner_model']}))
