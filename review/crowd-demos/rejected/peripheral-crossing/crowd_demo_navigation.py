"""Deterministic offline, space-time reserved pedestrian task programs.

Plans disc centres against exact conservative authored footprints and checks swept
pair separation. This is curated offline navigation, not a physics/LLM simulator.
"""
from __future__ import annotations
import argparse,heapq,json,math,random,time
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np
from crowd_demo_layout import make_layout
CELL=.65
TICK=.5
RADIUS=.28
STATE_NAMES=['walking','watching','browsing','greeting','signal_wait','yielding']
@dataclass(frozen=True)
class DemoConfig:
    environment:str='crossing'
    count:int=112
    seed:int=20260927
    duration:float=120.
    dt:float=.1
    def __post_init__(self):
        if self.environment not in ('crossing','city','station') or not 6<=self.count<=128:raise ValueError('invalid scene/count')
        if self.duration<25 or self.dt not in (.1,.05):raise ValueError('duration >=25 and dt .1 or .05 required')

def swept_distance(a,b,c,d):
    x=a[0]-c[0];z=a[1]-c[1];vx=b[0]-a[0]-d[0]+c[0];vz=b[1]-a[1]-d[1]+c[1]
    q=vx*vx+vz*vz;t=max(0,min(1,-(x*vx+z*vz)/q)) if q>1e-12 else 0
    return math.hypot(x+t*vx,z+t*vz)

def phase(t):
    s=t%54
    return 'walk' if 10<=s<46 else ('clearance' if 46<=s<54 else 'red')

class Planner:
    def __init__(self,cfg):
        self.cfg=cfg;self.layout=make_layout(cfg.environment);self.steps=round(cfg.duration/TICK);self.rng=random.Random(cfg.seed)
        bounds=self.layout.walkable;lo=[math.floor(min(getattr(r,k) for r in bounds)/CELL) for k in ('x0','z0')];hi=[math.ceil(max(getattr(r,k) for r in bounds)/CELL) for k in ('x1','z1')]
        ij=np.array([(i,j) for i in range(lo[0],hi[0]+1) for j in range(lo[1],hi[1]+1)])
        xy=ij*CELL;x,z=xy.T;valid=np.zeros(len(x),bool)
        for r in bounds:valid|=(x>=r.x0+.08)&(x<=r.x1-.08)&(z>=r.z0+.08)&(z<=r.z1-.08)
        for r in self.layout.obstacles:
            dx=np.maximum(np.maximum(r.x0-x,0),x-r.x1);dz=np.maximum(np.maximum(r.z0-z,0),z-r.z1);valid&=dx*dx+dz*dz>(RADIUS+.055)**2
        self.nodes={tuple(map(int,p)):tuple(q) for p,q in zip(ij[valid],xy[valid])};self.keys=list(self.nodes);self.edges={};self.reservations=[{} for _ in range(self.steps+1)];self.starts=[]
        self.meetings=[self.nearest(m) for m in self.layout.meetings];self.pockets=[self.nodes[m] for m in self.meetings]
    def nearest(self,xy,allowed=None):
        keys=self.keys if allowed is None else allowed
        return min(keys,key=lambda n:(self.nodes[n][0]-xy[0])**2+(self.nodes[n][1]-xy[1])**2)
    def neighbors(self,node):
        if node in self.edges:return self.edges[node]
        a=self.nodes[node];out=[node]
        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                if not(di or dj):continue
                n=node[0]+di,node[1]+dj
                if n in self.nodes:
                    b=self.nodes[n]
                    if all(self.layout.valid(a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,RADIUS+.035) for t in [.25,.5,.75]):out.append(n)
        self.edges[node]=out;return out
    def choose_starts(self):
        choices=[]
        for n in self.keys:
            x,z=self.nodes[n]
            if self.cfg.environment=='crossing':ok=abs(x)>13.5 and abs(z)>13.5 and max(abs(x),abs(z))<28
            elif self.cfg.environment=='city':ok=(-65<x<60 and -12<z<12) or (-7<x<7 and -44<z<-20)
            else:ok=(-44<x<44 and 9<z<31) or (-24<x<24 and -55<z<-37)
            if ok and all(math.dist((x,z),p)>3 for p in self.pockets):choices.append(n)
        self.rng.shuffle(choices)
        for i in range(self.cfg.count):
            if i<6:
                cx,cz=self.pockets[i//2];desired=(cx+(-10.4 if i%2==0 else 10.4),cz) if not (self.cfg.environment=='crossing' and i//2==2) else (cx,cz+(-10.4 if i%2==0 else 10.4))
                candidates=sorted(self.keys,key=lambda n:math.dist(self.nodes[n],desired))
            else:candidates=choices
            n=next((n for n in candidates if all(math.dist(self.nodes[n],self.nodes[s])>1.0 for s in self.starts)),None)
            if n is None:raise RuntimeError('unable to place population')
            self.starts.append(n)
        self.start_points=np.array([self.nodes[n] for n in self.starts]);self.start_blockers={};self.pocket_blockers={}
    def safe(self,a,b,t,actor):
        p,q=self.nodes[a],self.nodes[b]
        # Reserve the initial bodies until each actor has received a full plan.
        edge=(a,b)
        if edge not in self.start_blockers:
            self.start_blockers[edge]=tuple(j for j,c in enumerate(self.start_points) if swept_distance(p,q,c,c)<2*RADIUS+.05)
            self.pocket_blockers[edge]=any(swept_distance(p,q,c,c)<2.15 for c in self.pockets)
        if any(j>actor for j in self.start_blockers[edge]):return False
        if actor>=6 and self.pocket_blockers[edge]:return False
        if self.cfg.environment=='crossing':
            road=lambda c:abs(c[0])<12.76 or abs(c[1])<12.76
            if not road(p) and road(q) and phase(t*TICK)!='walk':return False
        bins=self.reservations[t]
        seen=set()
        for xx in range(min(a[0],b[0])-2,max(a[0],b[0])+3):
            for zz in range(min(a[1],b[1])-2,max(a[1],b[1])+3):
                for index,c,d in bins.get((xx,zz),[]):
                    if index in seen:continue
                    seen.add(index)
                    if swept_distance(p,q,c,d)<2*RADIUS+.055:return False
        return True
    def reserve(self,path,actor):
        for t,(a,b) in enumerate(zip(path,path[1:])):
            p,q=self.nodes[a],self.nodes[b]
            self.reservations[t].setdefault(a,[]).append((actor,p,q))
    def route(self,start,goal,t0,actor,dwell=0,end_by=None):
        final=self.steps if end_by is None else min(end_by,self.steps)
        h=lambda n:max(abs(n[0]-goal[0]),abs(n[1]-goal[1]))
        initial=(start,t0);heap=[(h(start),0,start,t0)];cost={initial:0};prev={};visited=0
        while heap and visited<1200:
            _,g,node,t=heapq.heappop(heap);key=node,t
            if g>cost.get(key,1e9):continue
            visited+=1
            if node==goal and t+dwell<=final and all(self.safe(node,node,k,actor) for k in range(t,t+dwell)):
                path=[node]
                while key!=initial:key=prev[key];path.append(key[0])
                path.reverse();return path+[node]*dwell
            if t>=final-dwell:continue
            ns=self.neighbors(node)
            for n in sorted(ns,key=h):
                if not self.safe(node,n,t,actor):continue
                ng=g+(1.03 if n==node else 1)+(0.025 if n[0]!=node[0] and n[1]!=node[1] else 0)
                nxt=n,t+1
                if ng<cost.get(nxt,1e9):cost[nxt]=ng;prev[nxt]=key;heapq.heappush(heap,(ng+1.4*h(n),ng,n,t+1))
        return None
    def continuation(self,start,t0,actor):
        # Search a finite safe continuation when the requested endpoint is occupied.
        # Revisiting cells is penalized, so this produces a detour rather than a freeze.
        path=[start];visits={start:1};tried={};budget=12000
        while path and budget:
            t=t0+len(path)-1
            if t==self.steps:return path
            n=path[-1];key=(n,t);used=tried.setdefault(key,set())
            opts=[q for q in self.neighbors(n) if q not in used and self.safe(n,q,t,actor)]
            budget-=1
            if opts:
                q=min(opts,key=lambda q:(q==n,visits.get(q,0),-math.dist(self.nodes[q],self.nodes[start]),q))
                used.add(q);path.append(q);visits[q]=visits.get(q,0)+1
            else:
                old=path.pop();visits[old]-=1
        return None
    def build(self):
        self.choose_starts();paths=[];agents=[];events=[]
        goals=[(label,self.nearest((x,z))) for label,x,z in self.layout.destinations]
        for k,c in enumerate(self.pockets):
            roles=[self.nearest((c[0]+dx,c[1])) for dx in [-.65,.65]]
            events.append({'id':f'{self.cfg.environment}-greeting-{k}','kind':'greeting','actor_ids':[2*k,2*k+1],'start_s':12.,'end_s':20.,'clip_start_s':14.,'clip_end_s':18.,'anchor_xz':list(c),'actor_anchors_xz':[list(self.nodes[n]) for n in roles],'actor_yaws':[math.pi/2,-math.pi/2],'shared_yaw':0.,'reserved_radius_m':2.15,'motion':'pending-native-pair-integration'})
        for actor in range(self.cfg.count):
            path=[self.starts[actor]];schedule=[];attempts=0;task=0
            if actor<6:
                ev=events[actor//2];goal=self.nearest(ev['actor_anchors_xz'][actor%2]);route=self.route(path[-1],goal,0,actor,end_by=24)
                if route is None:raise RuntimeError(f'featured approach failed {actor}')
                arrival=(len(route)-1)*TICK
                if not all(self.safe(goal,goal,t,actor) for t in range(len(route)-1,40)):raise RuntimeError('featured pocket unavailable')
                path=route+[goal]*(41-len(route));schedule=[{'start':0,'end':arrival,'task':'approach_friend','label':'Walk toward a friend','target':list(self.nodes[goal])},{'start':arrival,'end':20,'task':'greeting','label':'Greet a friend, then continue','target':list(self.nodes[goal])}]
            while len(path)<=self.steps:
                t0=len(path)-1;remaining=self.steps-t0
                if remaining<30:
                    tail=self.continuation(path[-1],t0,actor)
                    if tail:
                        path+=tail[1:];schedule.append({'start':t0*TICK,'end':self.cfg.duration,'task':'walk','label':'Continue along an open neighbourhood route','target':list(self.nodes[path[-1]])});break
                candidates=[g for g in goals if 5<math.dist(self.nodes[path[-1]],self.nodes[g[1]])<max(12,min(38,remaining*.55))]
                # Interior visitors repeatedly use real doors around the middle of the film.
                if self.cfg.environment=='city' and actor in (6,7,8,9) and t0*TICK<58:
                    wanted='cafe-order' if actor%2==0 else 'books-browse';candidates=[g for g in goals if g[0]==wanted]+candidates
                else:self.rng.shuffle(candidates)
                # Short local walks guarantee a bounded final task without inventing a teleport.
                if not candidates or attempts>1:
                    candidates +=[('neighbourhood',n) for n in self.rng.sample(self.keys,min(100,len(self.keys))) if 2<math.dist(self.nodes[n],self.nodes[path[-1]])<12]
                chosen=None
                for label,goal in candidates[:8]:
                    if label not in ('cafe-order','cafe-door','books-browse','books-door'):
                        x,z=self.nodes[goal];goal=self.nearest((x+self.rng.uniform(-2,2),z+self.rng.uniform(-1,1)))
                    dwell=min(self.rng.randrange(12,23),max(0,remaining-max(abs(goal[0]-path[-1][0]),abs(goal[1]-path[-1][1]))))
                    route=self.route(path[-1],goal,t0,actor,dwell=dwell)
                    if route is not None:chosen=label,goal,route,dwell;break
                if chosen is None:
                    attempts+=1
                    if attempts>1:
                        tail=self.continuation(path[-1],t0,actor)
                        if tail:
                            path+=tail[1:];schedule.append({'start':t0*TICK,'end':self.cfg.duration,'task':'walk','label':'Take an open detour and continue exploring','target':list(self.nodes[path[-1]])});break
                        repaired=False
                        for rewind in range(max(0,t0-20),-1,-20):
                            if actor<6 and rewind<40:continue
                            tail=self.continuation(path[rewind],rewind,actor)
                            if tail:
                                path=path[:rewind]+tail;schedule=[dict(s,end=min(s['end'],rewind*TICK)) for s in schedule if s['start']<rewind*TICK]
                                schedule.append({'start':rewind*TICK,'end':self.cfg.duration,'task':'walk','label':'Continue via an available pedestrian detour','target':list(self.nodes[path[-1]])});repaired=True;break
                        if repaired:break
                        raise RuntimeError(f'actor {actor} cannot continue at {t0*TICK}s')
                    continue
                label,goal,route,dwell=chosen;attempts=0;arrival=t0+len(route)-1-dwell;end=t0+len(route)-1
                schedule +=[{'start':t0*TICK,'end':arrival*TICK,'task':'walk','label':f'Walk to {label.replace("-"," ")}','target':list(self.nodes[goal])}]
                if dwell:schedule +=[{'start':arrival*TICK,'end':end*TICK,'task':'browse' if label.startswith(('cafe','book')) else 'watch','label':{'cafe-order':'Order coffee and look around','books-browse':'Browse the book shelves'}.get(label,'Pause, look around, then continue'),'target':list(self.nodes[goal])}]
                path+=route[1:];task+=1
            if len(path)!=self.steps+1:raise RuntimeError('wrong path length')
            self.reserve(path,actor);paths.append(path)
            personality=['observant visitor','purposeful commuter','relaxed neighbour','curious shopper','sociable friend','quiet reader'][actor%6]
            agents.append({'id':actor,'radius':RADIUS,'gait':['casual','brisk','relaxed'][actor%3],'color_index':(actor*7+actor//9)%16,'personality':personality,'prompt':f'You are an {personality}. Complete your listed errands, notice your surroundings during short pauses, respect doors and personal space, and continue to the next destination.','schedule':schedule,'task_program':{'repeating':True,'dwell_range_s':[6,11],'completed_routes':task},'start_xz':list(self.nodes[path[0]])})
            if actor%16==0:print(f'{self.cfg.environment}: planned {actor+1}/{self.cfg.count}',flush=True)
        positions=np.array([[self.nodes[paths[a][t]] for a in range(self.cfg.count)] for t in range(self.steps+1)])
        times=np.arange(round(self.cfg.duration/self.cfg.dt)+1)*self.cfg.dt;frames=[];distance=np.zeros(self.cfg.count);yaw=np.zeros(self.cfg.count);previous=positions[0]
        for t in times:
            k=min(self.steps-1,int(t/TICK));alpha=min(1,(t-k*TICK)/TICK);p=positions[k]*(1-alpha)+positions[k+1]*alpha
            velocity=(positions[k+1]-positions[k])/TICK;speed=np.linalg.norm(velocity,axis=1);distance+=np.linalg.norm(p-previous,axis=1);previous=p
            moving=speed>.01;target=np.arctan2(velocity[:,0],velocity[:,1]);delta=np.arctan2(np.sin(target-yaw),np.cos(target-yaw));yaw[moving]+=np.clip(delta[moving],-2.8*self.cfg.dt,2.8*self.cfg.dt)
            rows=[]
            for a in range(self.cfg.count):
                state=0 if moving[a] else 1
                sched=next((s for s in agents[a]['schedule'] if s['start']<=t<s['end']),None)
                if not moving[a] and sched:
                    state={'browse':2,'greeting':3,'walk':4 if self.cfg.environment=='crossing' and phase(t)!='walk' else 5}.get(sched['task'],1)
                if a<6 and 12<=t<=20:
                    desired=events[a//2]['actor_yaws'][a%2];d=math.atan2(math.sin(desired-yaw[a]),math.cos(desired-yaw[a]));yaw[a]+=max(-.28,min(.28,d))
                rows.append([round(p[a,0],4),round(p[a,1],4),round(float(yaw[a]),4),round(float(speed[a]),4),state,round(float(distance[a]),4)])
            frames.append({'t':round(float(t),4),'people':rows})
        result={'schema':'stagezero.crowd.v1','environment':self.cfg.environment,'units':'meters','up_axis':'Y','dt':self.cfg.dt,'duration':self.cfg.duration,'seed':self.cfg.seed,'config':asdict(self.cfg),'state_names':STATE_NAMES,'agents':agents,'frames':frames,'events':events,'layout':self.layout.to_dict(),'navigation_method':'Deterministic prioritized space-time A* with swept-disc reservations and repeated bounded tasks','traffic':({'period_s':54,'red_s':10,'walk_s':36,'clearance_s':8} if self.cfg.environment=='crossing' else None)}
        normalize_task_legs(result);result['metrics']=measure(result,self.layout);return result

def normalize_task_legs(data):
    """Expose actual arriving waypoints for long reserved detours in the inspector."""
    for i,agent in enumerate(data['agents']):
        out=[]
        for task in agent['schedule']:
            if task['task']=='walk' and task['end']-task['start']>24:
                start=task['start']
                while start<task['end']-1e-6:
                    end=min(task['end'],start+16);f=min(len(data['frames'])-1,round(end/data['dt']))
                    out.append(dict(task,start=start,end=end,label='Continue to the next route waypoint',target=data['frames'][f]['people'][i][:2]));start=end
            else:out.append(task)
        agent['schedule']=out
    return data

def apply_city_attendant(data,manifest):
    """A real stationary work role; no pretend travel and no offscreen disappearance."""
    if data['environment']!='city' or len(data['agents'])<64:return data
    index=next(i for i,a in enumerate(data['agents']) if a['id']==63);agent=data['agents'][index]
    agent.update(personality='attentive cafe attendant',prompt='Work behind the Koma Coffee counter. Watch for customers, acknowledge their order, and return to an attentive idle between customers.',stationary_work_role=True)
    agent['schedule']=[{'start':float(t),'end':float(min(t+10,120)),'task':'serve','label':'Serve coffee and acknowledge customers at the counter','target':[-29.,-23.53]} for t in range(0,120,10)]
    agent['task_program']={'repeating':True,'stationary_work_role':True,'dwell_range_s':[10,10],'completed_routes':0};agent['start_xz']=[-29.,-23.53]
    lookup={c['id']:i for i,c in enumerate(manifest['clips'])}
    for f in data['frames']:
        t=f['t'];local=t%10;clip=-1;ctime=0.
        if 2<=local<=3.9:clip=lookup['listen' if int(t//10)%3 else 'wave'];ctime=local-2
        elif 6<=local<=7.9:clip=lookup['look'];ctime=local-6
        f['people'][index]=[-29.,-23.53,0.,0.,1,0.,clip,ctime,0.]
    if 'motion_cues' in data:
        cues=data['motion_cues'];cues['dwell_cues']=[c for c in cues['dwell_cues'] if c['actor_id']!=63];cues['dwell_gesture_events']=len(cues['dwell_cues']);cues['stationary_staff']={'actor_id':63,'role':'cafe attendant','gesture_policy':'Generated listen/wave and look cues during repeated ten-second service tasks; physically present throughout.'}
    return data

def measure(data,layout=None):
    if not data.get('frames') or any(len(f['people'])!=len(data['agents']) for f in data['frames']):raise ValueError('Every actor must be physically present in every frame')
    layout=layout or make_layout(data['environment']);a=np.array([f['people'] for f in data['frames']]);p=a[:,:,:2];
    if not np.isfinite(a).all():raise ValueError('Nonfinite trajectory')
    r=np.array([x['radius'] for x in data['agents']]);mins=[];overlaps=0;static=0;outside=0;maxpenetration=0.
    for frame in p:
        inside=np.zeros(len(frame),bool)
        for rect in layout.walkable:inside|=(frame[:,0]>=rect.x0)&(frame[:,0]<=rect.x1)&(frame[:,1]>=rect.z0)&(frame[:,1]<=rect.z1)
        outside+=int(np.count_nonzero(~inside))
        d=np.linalg.norm(frame[:,None]-frame[None,:],axis=2);np.fill_diagonal(d,np.inf);mins.append(float(d.min()));penetration=r[:,None]+r[None,:]-d;overlaps+=int(np.count_nonzero(np.triu(penetration>1e-3,1)));maxpenetration=max(maxpenetration,float(penetration.max()))
        for ob in layout.obstacles:
            dx=np.maximum(np.maximum(ob.x0-frame[:,0],0),frame[:,0]-ob.x1);dz=np.maximum(np.maximum(ob.z0-frame[:,1],0),frame[:,1]-ob.z1);static+=int(np.count_nonzero(dx*dx+dz*dz<(r-.001)**2))
    steps=np.linalg.norm(np.diff(p,axis=0),axis=2);dt=data['dt'];run=np.zeros(p.shape[1]);maxrun=np.zeros_like(run);deadlocks=0
    for f,movement in enumerate(steps):
        active=np.isin(a[f,:,4],[0,5]);run=np.where((movement<.003)&active,run+dt,0);maxrun=np.maximum(maxrun,run)
    # Swept-pair check catches collisions hidden between cached samples.
    swept=0;sweptmin=math.inf
    for f in range(len(p)-1):
        rel=p[f,:,None]-p[f,None,:];vel=(p[f+1]-p[f]);dv=vel[:,None]-vel[None,:];q=np.sum(dv*dv,axis=2);u=np.clip(-np.sum(rel*dv,axis=2)/np.maximum(q,1e-12),0,1);d=np.linalg.norm(rel+dv*u[:,:,None],axis=2);np.fill_diagonal(d,np.inf);sweptmin=min(sweptmin,float(d.min()));swept+=int(np.count_nonzero(np.triu(d<r[:,None]+r[None,:]-.001,1)))
    interior=sum(bool(np.any((p[:,i,1]<-17.1)&(p[:,i,1]>-23.7)&(p[:,i,0]>-34.5)&(p[:,i,0]<-11.5))) for i in range(p.shape[1])) if data['environment']=='city' else 0
    return {'actor_count':p.shape[1],'duration_s':data['duration'],'frames':len(p),'pair_overlap_samples':overlaps,'swept_pair_overlap_samples':swept,'min_pair_distance_m':round(min(mins),4),'min_swept_pair_distance_m':round(sweptmin,4),'max_proxy_penetration_m':round(max(0,maxpenetration),4),'static_collision_samples':static,'outside_walkable_samples':outside,'max_root_acceleration_mps2':round(float(np.linalg.norm(np.diff(np.diff(p,axis=0)/dt,axis=0),axis=2).max()/dt),3),'heading_interpretation':'Navigation yaw eased at2.8rad/s; explicit native pair common-yaw is a source coordinate transform, not body heading','root_motion_interpretation':'Piecewise linear reserved-grid motion; acceleration is discontinuous at route vertices and dwell boundaries','max_root_step_m':round(float(steps.max()),4),'max_root_speed_mps':round(float(steps.max()/dt),4),'root_jump_samples':int(np.count_nonzero(steps>2*dt)),'max_unscheduled_stall_s':round(float(maxrun.max()),3),'deadlocked_actor_count':int(np.count_nonzero(maxrun>12)),'min_actor_travel_m':round(float(steps.sum(axis=0).min()),3),'min_mobile_actor_travel_m':round(float(min(steps[:,i].sum() for i,agent in enumerate(data['agents']) if not agent.get('stationary_work_role'))),3),'stationary_work_actor_ids':[agent['id'] for agent in data['agents'] if agent.get('stationary_work_role')],'median_actor_travel_m':round(float(np.median(steps.sum(axis=0))),3),'interior_visitors':interior,'full_body_collision':'not measured; disc proxy only','foot_sliding':'not measured by root navigation','validation_scope':'All frames and continuous linear pair motion between cached samples; authored ground obstacles'}

def validate(data):
    ids=[a['id'] for a in data['agents']]
    if len(set(ids))!=len(ids):raise ValueError('Actor IDs must be stable and unique')
    times=np.array([f['t'] for f in data['frames']])
    if len(times)!=round(data['duration']/data['dt'])+1 or not np.allclose(times,np.arange(len(times))*data['dt'],atol=1e-6):raise ValueError('Trajectory must cover every shared-clock sample')
    m=measure(data)
    for key in ['pair_overlap_samples','swept_pair_overlap_samples','static_collision_samples','outside_walkable_samples','root_jump_samples','deadlocked_actor_count']:
        if m[key]:raise ValueError(f'{key}: {m[key]}')
    if m['min_mobile_actor_travel_m']<10:raise ValueError('actor did not sustain movement')
    return m

def generate(cfg=DemoConfig()):return Planner(cfg).build()
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--environment',choices=['crossing','city','station'],default='crossing');ap.add_argument('--count',type=int,default=112);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();d=generate(DemoConfig(args.environment,args.count));validate(d);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(d,separators=(',',':')));print(json.dumps(d['metrics'],indent=2))
