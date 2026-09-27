"""Deterministic offline, space-time reserved pedestrian task programs.

Plans disc centres against exact conservative authored footprints and checks swept
pair separation. This is curated offline navigation, not a physics/LLM simulator.
"""
from __future__ import annotations
import argparse,heapq,json,math,random,time
from collections import deque
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
        # Crossing objectives use the two painted diagonal corridors. The old broad
        # sidewalk rectangles also span unpainted road arms; they are not crossing links.
        if cfg.environment=='crossing':
            road=(abs(x)<12.76)|(abs(z)<12.76)
            painted=(abs(x-z)<2.03*math.sqrt(2))|(abs(x+z)<2.03*math.sqrt(2))
            valid &= ~road|painted
        self.nodes={tuple(map(int,p)):tuple(q) for p,q in zip(ij[valid],xy[valid])};self.keys=list(self.nodes);self.edges={};self.reservations=[{} for _ in range(self.steps+1)];self.starts=[]
        self.heuristics={}
        self.formation=[self.nearest((x,z)) for z in [-57.85,-55.25,-52.65,-50.05] for x in [-54.6,-52,-49.4,-46.8,-44.2,-41.6]] if cfg.environment=='city' else []
        if cfg.environment=='city':
            self.nodes[(-10063,10000)]=(-29.,-23.53);self.keys=list(self.nodes);self.edges[(-10063,10000)]=[(-10063,10000)]
        self.audience=[self.nearest((-65.65+(i%2)*1.3,-59.15+(i//2)*1.3)) for i in range(12)] if cfg.environment=='city' else []
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
            elif self.cfg.environment=='city':ok=(-65<x<65 and (12.7<z<15.7 or -15.7<z<-12.7))
            else:ok=(-44<x<44 and 9<z<31) or (-24<x<24 and -55<z<-37)
            if ok and all(math.dist((x,z),p)>3 for p in self.pockets):choices.append(n)
        self.rng.shuffle(choices)
        for i in range(self.cfg.count):
            if self.cfg.environment=='city' and i==63:
                candidates=[(-10063,10000)]
            elif self.cfg.environment=='city' and 6<=i<30:
                target=self.nodes[self.formation[i-6]]
                pool=[n for n in self.keys if -66<self.nodes[n][0]<-39 and -65<self.nodes[n][1]<-46 and not (-55.8<self.nodes[n][0]<-40.4 and -59.05<self.nodes[n][1]<-48.85) and all(math.dist(self.nodes[n],self.nodes[q])>1.0 for q in self.audience) and all(math.dist(self.nodes[n],q)>2.4 for q in self.pockets)]
                candidates=sorted(pool,key=lambda n:abs(math.dist(self.nodes[n],target)-8)+(i%3)*abs(self.nodes[n][1]+61)*.03)
            elif self.cfg.environment=='city' and 30<=i<38:
                candidates=sorted(choices,key=lambda n:math.dist(self.nodes[n],(-30+(i-30)*2,-14)))
            elif self.cfg.environment=='city' and 38<=i<50:
                candidates=sorted([n for n in self.keys if -65<self.nodes[n][0]<-39 and -49<self.nodes[n][1]<-46],key=lambda n:math.dist(self.nodes[n],(-62+(i-38)*1.4,-47)))
            elif i<6:
                cx,cz=self.pockets[i//2];desired=(cx+(-10.4 if i%2==0 else 10.4),cz) if not (self.cfg.environment=='crossing' and i//2==2) else (cx,cz+(-10.4 if i%2==0 else 10.4))
                candidates=sorted(self.keys,key=lambda n:math.dist(self.nodes[n],desired))
            else:candidates=choices
            n=next((n for n in candidates if all(math.dist(self.nodes[n],self.nodes[s])>1.0 for s in self.starts)),None)
            if n is None:raise RuntimeError('unable to place population')
            self.starts.append(n)
        self.start_points=np.array([self.nodes[n] for n in self.starts]);self.start_blockers={};self.pocket_blockers={};self.formation_blockers={};self.audience_blockers={}
    def safe(self,a,b,t,actor):
        p,q=self.nodes[a],self.nodes[b]
        # Reserve the initial bodies until each actor has received a full plan.
        edge=(a,b)
        if edge not in self.start_blockers:
            self.start_blockers[edge]=tuple(j for j,c in enumerate(self.start_points) if swept_distance(p,q,c,c)<2*RADIUS+.05)
            self.pocket_blockers[edge]=any(swept_distance(p,q,c,c)<2.15 for c in self.pockets)
            self.formation_blockers[edge]=tuple(j+6 for j,n in enumerate(self.formation) if swept_distance(p,q,self.nodes[n],self.nodes[n])<2*RADIUS+.055)
            self.audience_blockers[edge]=tuple(j+38 for j,n in enumerate(self.audience) if swept_distance(p,q,self.nodes[n],self.nodes[n])<2*RADIUS+.055)
        if any(j>actor for j in self.start_blockers[edge]):return False
        if actor>=6 and self.pocket_blockers[edge] and (self.cfg.environment!='city' or 22<=t<=40):return False
        if self.cfg.environment=='city' and t>=76 and any(j>actor for j in self.audience_blockers[edge]):return False
        if self.cfg.environment=='city' and 6<=actor<30 and t>=76:
            if any(j>actor for j in self.formation_blockers[edge]):return False
        if self.cfg.environment=='city' and not 6<=actor<30:
            if -55.8<q[0]<-40.4 and -59.05<q[1]<-48.85:return False
        if self.cfg.environment=='crossing':
            road=lambda c:abs(c[0])<12.76 or abs(c[1])<12.76
            if road(q) and phase((t+1)*TICK)=='red':return False
            if not road(p) and road(q):
                if phase(t*TICK)!='walk':return False
                # Reserve enough clearance time to reach the far corner even when
                # the final task target is farther along its sidewalk.
                goal=getattr(self,'current_goal',None)
                if goal is not None:
                    gx,gz=self.nodes[goal];exit_point=(math.copysign(13.65,gx),math.copysign(13.65,gz))
                    crossing_ticks=max(abs(q[0]-exit_point[0]),abs(q[1]-exit_point[1]))/CELL
                    if t*TICK%54+crossing_ticks*TICK+3>53.5:return False
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
    def route(self,start,goal,t0,actor,dwell=0,end_by=None,partial=False,hold_until=None):
        self.current_goal=goal
        final=self.steps if end_by is None else min(end_by,self.steps)
        h=lambda n:max(abs(n[0]-goal[0]),abs(n[1]-goal[1]))
        if self.cfg.environment=='city':
            if goal not in self.heuristics:
                dist={goal:0};queue=deque([goal])
                while queue:
                    node=queue.popleft()
                    for n in self.neighbors(node):
                        if n not in dist:dist[n]=dist[node]+1;queue.append(n)
                self.heuristics[goal]=dist
            h=lambda n:self.heuristics[goal].get(n,10000)
        initial=(start,t0);heap=[(h(start),0,start,t0)];cost={initial:0};prev={};visited=0
        while heap and visited<(10000 if self.cfg.environment=='crossing' else (6000 if end_by is not None else 2000) if self.cfg.environment=='city' else 1200):
            _,g,node,t=heapq.heappop(heap);key=node,t
            if g>cost.get(key,1e9):continue
            visited+=1
            if (node==goal and t+dwell<=final and all(self.safe(node,node,k,actor) for k in range(t,max(t+dwell,hold_until or 0)))) or (partial and t==final):
                path=[node]
                while key!=initial:key=prev[key];path.append(key[0])
                path.reverse();return path+[node]* (dwell if node==goal and t+dwell<=final else 0)
            if t>=final-(0 if partial else dwell):continue
            ns=self.neighbors(node)
            for n in sorted(ns,key=h):
                if not self.safe(node,n,t,actor):continue
                ng=g+(1.03 if n==node else 1)+(0.025 if n[0]!=node[0] and n[1]!=node[1] else 0)
                nxt=n,t+1
                if ng<cost.get(nxt,1e9):cost[nxt]=ng;prev[nxt]=key;heapq.heappush(heap,(ng+1.4*h(n),ng,n,t+1))
        return None
    def continuation(self,start,t0,actor,goal=None):
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
                q=min(opts,key=lambda q:((self.heuristics.get(goal,{}).get(q,math.dist(self.nodes[q],self.nodes[goal]))+visits.get(q,0)*2 if goal is not None else 0),q==n,visits.get(q,0),-math.dist(self.nodes[q],self.nodes[start]),q))
                used.add(q);path.append(q);visits[q]=visits.get(q,0)+1
            else:
                old=path.pop();visits[old]-=1
        return None
    def build(self):
        self.choose_starts()
        if self.cfg.environment=='city':return self.build_city()
        paths=[];agents=[];events=[]
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
                if self.cfg.environment=='crossing':
                    x,z=self.nodes[path[-1]]
                    candidates=[g for g in goals if self.nodes[g[1]][0]*x<0 and self.nodes[g[1]][1]*z<0]
                else:
                    candidates=[g for g in goals if 5<math.dist(self.nodes[path[-1]],self.nodes[g[1]])<max(12,min(38,remaining*.55))]
                # Interior visitors repeatedly use real doors around the middle of the film.
                if self.cfg.environment=='city' and actor in (6,7,8,9) and t0*TICK<58:
                    wanted='cafe-order' if actor%2==0 else 'books-browse';candidates=[g for g in goals if g[0]==wanted]+candidates
                else:self.rng.shuffle(candidates)
                # Short local walks guarantee a bounded final task without inventing a teleport.
                if (not candidates or attempts>1) and self.cfg.environment!='crossing':
                    candidates +=[('neighbourhood',n) for n in self.rng.sample(self.keys,min(100,len(self.keys))) if 2<math.dist(self.nodes[n],self.nodes[path[-1]])<12]
                chosen=None
                for label,goal in candidates[:8]:
                    if label not in ('cafe-order','cafe-door','books-browse','books-door'):
                        x,z=self.nodes[goal];goal=self.nearest((x+self.rng.uniform(-2,2),z+self.rng.uniform(-1,1)))
                    dwell=min(self.rng.randrange(12,23),max(0,remaining-max(abs(goal[0]-path[-1][0]),abs(goal[1]-path[-1][1]))))
                    route=self.route(path[-1],goal,t0,actor,dwell=dwell)
                    if route is not None:chosen=label,goal,route,dwell;break
                if chosen is None:
                    if self.cfg.environment=='crossing' and remaining>=24 and all(self.safe(path[-1],path[-1],tick,actor) for tick in range(t0,min(t0+12,self.steps))):
                        queued=min(12,remaining);path +=[path[-1]]*queued;schedule.append({'start':t0*TICK,'end':(t0+queued)*TICK,'task':'signal_wait','label':'Wait at the corner for a safe scramble crossing','target':list(self.nodes[path[-1]])});continue
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
        return self.export(paths,agents,events)
    def build_city(self):
        paths=[];agents=[];events=[]
        for k,c in enumerate(self.pockets):
            roles=[self.nearest((c[0]+dx,c[1])) for dx in [-.65,.65]]
            events.append({'id':f'city-greeting-{k}','kind':'greeting','actor_ids':[2*k,2*k+1],'start_s':12.,'end_s':20.,'clip_start_s':14.,'clip_end_s':18.,'anchor_xz':list(c),'actor_anchors_xz':[list(self.nodes[n]) for n in roles],'actor_yaws':[math.pi/2,-math.pi/2],'shared_yaw':0.,'reserved_radius_m':2.15})
        crew=list(range(6,min(30,self.cfg.count)))
        events.append({'id':'city-courtyard-flashmob','kind':'flashmob','actor_ids':crew,'slot_xz':[list(self.nodes[self.formation[a-6]]) for a in crew],'anchor_xz':[-48.1,-53.95],'center_xz':[-48.1,-53.95],'arrival_start_s':18.,'settle_by_s':38.,'dance_start_s':40.,'dance_end_s':65.,'start_s':40.,'end_s':65.,'disperse_start_s':65.,'shared_yaw':0.,'reserved_radius_m':1.2,'min_spacing_m':2.6,'outsider_exclusion_xz':[-55.8,-59.05,-40.4,-48.85]})
        for actor in range(self.cfg.count):
            path=[self.starts[actor]];schedule=[];task=0
            def leg(xz,label,dwell=12,until=None,task_name='watch',partial=False,end_by=None):
                nonlocal path,task
                t0=len(path)-1;goal=self.nearest(xz);route=self.route(path[-1],goal,t0,actor,dwell=dwell,end_by=end_by,partial=partial,hold_until=round(until/TICK) if until is not None else None)
                if route is None:return False
                actual_dwell=min(dwell,max(0,len(route)-1)) if route[-1]==goal and len(route)+t0-1<self.steps else 0
                arrival=t0+len(route)-1-actual_dwell
                if until is not None:
                    end=round(until/TICK)
                    if arrival>end or not all(self.safe(goal,goal,k,actor) for k in range(arrival,end)):return False
                    route=route[:arrival-t0+1]+[goal]*(end-arrival)
                end=t0+len(route)-1
                schedule.append({'start':t0*TICK,'end':arrival*TICK,'task':'walk','label':label,'target':list(self.nodes[goal])})
                if end>arrival:schedule.append({'start':arrival*TICK,'end':end*TICK,'task':task_name,'label':label.replace('Walk to ','').replace('Approach ','')+' — pause for the scheduled activity','target':list(self.nodes[goal])})
                path+=route[1:];task+=1;return True
            if actor==63:
                path=[path[0]]*(self.steps+1);schedule=[{'start':0,'end':120,'task':'serve','label':'Attend the café counter and acknowledge visitors','target':[-29,-23.53]}]
            elif actor<6:
                goal=events[actor//2]['actor_anchors_xz'][actor%2]
                if not leg(goal,'Approach a friend on the pedestrian pavement',0,20,'greeting',end_by=24):raise RuntimeError(f'city pair {actor} cannot meet')
            elif actor in crew:
                delay=18+(actor%4)*1.5;ticks=round(delay/TICK)
                if not all(self.safe(path[0],path[0],k,actor) for k in range(ticks)):raise RuntimeError('crew waiting blocked')
                path +=[path[0]]*ticks;schedule.append({'start':0,'end':delay,'task':'watch','label':'Wait with the flashmob crew for the formation cue','target':list(self.nodes[path[0]])})
                slot=self.nodes[self.formation[actor-6]]
                if not leg(slot,'Walk to the assigned courtyard dance mark',0,65,'watch',end_by=76):raise RuntimeError(f'crew {actor} cannot settle')
                last=schedule.pop();schedule +=[dict(last,end=40,label='Settle on the dance mark and face the music rig'),{'start':40.,'end':65.,'task':'dance','label':'Perform the synchronized courtyard flashmob routine','target':list(slot)}]
            elif 38<=actor<50:
                slot=self.nodes[self.audience[actor-38]]
                if not leg(slot,'Walk to the courtyard audience line',0,70,'watch',end_by=76):raise RuntimeError(f'audience {actor} cannot arrive')
                schedule[-1]['label']='Watch the coordinated courtyard dance'
            # Each role follows named destinations; final incomplete errands keep
            # their intended endpoint instead of inventing a random walking tail.
            cycle=0
            while len(path)<=self.steps:
                t0=len(path)-1;x,z=self.nodes[path[-1]];remaining=self.steps-t0
                if actor in crew or 38<=actor<50 or actor in (4,5):
                    options=[((-37.7,-38 if cycle%2==0 else -15),'Leave the courtyard via the alley' if cycle%2==0 else 'Walk to the north sidewalk after the courtyard event'),((-63,-47),'Walk to the courtyard west exit'),((-38,-48),'Walk to the courtyard east exit')]
                elif 30<=actor<38:
                    order=[((-29,-21.45),'Walk to the café counter to order coffee'),((-29,-14.3),'Walk out through the café doorway'),((-19 if actor%2==0 else -15,-21.45),'Walk to the bookshop shelves to browse'),((-17,-14.3),'Walk out through the bookshop doorway')]
                    options=[order[(cycle+(actor-30)%4)%4]]
                    options +=[order[(cycle+1+(actor-30)%4)%4]]
                else:
                    side=-1 if z<0 else 1
                    # Alternate an errand along the same pavement with an actual
                    # crossing to the other side through one of the painted zebras.
                    gx=(-35 if x>0 else 35)+(actor%7)*1.3
                    options=[((gx,side*14.3 if cycle%2==0 else -side*14.3),'Walk to the next high-street errand using sidewalks and painted crossings')]
                done=False
                for goal,label in options:
                    if leg(goal,label,min(12,max(0,remaining-5)),task_name='browse' if 'shelves' in label or 'counter' in label else 'watch',partial=True):done=True;break
                if not done:
                    # A blocked route waits for a bounded retry at its existing
                    # safe position. The queue is explicit in the saved task log.
                    hold=min(12,remaining)
                    if hold and all(self.safe(path[-1],path[-1],k,actor) for k in range(t0,t0+hold)):
                        path +=[path[-1]]*hold;schedule.append({'start':t0*TICK,'end':(t0+hold)*TICK,'task':'signal_wait','label':'Queue for space on the planned pedestrian route','target':list(self.nodes[path[-1]])})
                    else:
                        tail=self.continuation(path[-1],t0,actor,self.nearest(options[0][0]))
                        if tail:
                            path+=tail[1:];schedule.append({'start':t0*TICK,'end':self.cfg.duration,'task':'walk','label':options[0][1]+' while yielding to other pedestrians','target':list(options[0][0])})
                        else:
                            repaired=False
                            floor=130 if actor in crew else 140 if 38<=actor<50 else 40 if actor<6 else 0
                            for rewind in range(t0-8,floor-1,-8):
                                tail=self.continuation(path[rewind],rewind,actor,self.nearest(options[0][0]))
                                if tail:
                                    path=path[:rewind]+tail;schedule=[dict(item,end=min(item['end'],rewind*TICK)) for item in schedule if item['start']<rewind*TICK]
                                    schedule.append({'start':rewind*TICK,'end':self.cfg.duration,'task':'walk','label':options[0][1]+' via a pedestrian yielding detour','target':list(options[0][0])});repaired=True;break
                            if not repaired:raise RuntimeError(f'city actor {actor} blocked at {t0*TICK}')
                else:cycle+=1
            self.reserve(path,actor);paths.append(path)
            role='flashmob performer' if actor in crew else 'courtyard audience member' if 38<=actor<50 else 'café and bookshop customer' if 30<=actor<38 else 'café attendant' if actor==63 else 'high-street pedestrian with scheduled errands'
            agent={'id':actor,'radius':RADIUS,'gait':['casual','brisk','relaxed'][actor%3],'color_index':(actor*7+actor//9)%16,'personality':['attentive','relaxed','purposeful','sociable'][actor%4]+' '+role,'prompt':f'You are a {role}. Follow your named pedestrian destinations, use painted crossings and shop doors, respect the courtyard formation reservation, and continue your listed activity.','schedule':schedule,'task_program':{'repeating':False,'role':role,'completed_routes':task},'start_xz':list(self.nodes[path[0]])}
            if actor in crew:agent['unit_scale']=True
            if actor==63:agent['stationary_work_role']='café attendant'
            agents.append(agent)
            if actor%8==0:print(f'city: planned {actor+1}/{self.cfg.count}',flush=True)
        data=self.export(paths,agents,events);data['layout']['route_policy']='Pedestrian sidewalks and two painted zebras only; side-street carriageway excluded; courtyard formation reserved for24performers40–65s.';return data
    def export(self,paths,agents,events):
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
                    state={'browse':2,'greeting':3,'signal_wait':4,'walk':4 if self.cfg.environment=='crossing' and phase(t)!='walk' else 5}.get(sched['task'],1)
                if a<6 and 12<=t<=20:
                    desired=events[a//2]['actor_yaws'][a%2];d=math.atan2(math.sin(desired-yaw[a]),math.cos(desired-yaw[a]));yaw[a]+=max(-.28,min(.28,d))
                if self.cfg.environment=='city' and 6<=a<30 and not moving[a] and 36<=t<=65:
                    d=math.atan2(math.sin(-yaw[a]),math.cos(-yaw[a]));yaw[a]+=max(-2.8*self.cfg.dt,min(2.8*self.cfg.dt,d))
                if self.cfg.environment=='city' and 38<=a<50 and not moving[a] and 35<=t<=70:yaw[a]=math.atan2(-48.1-p[a,0],-53.95-p[a,1])
                rows.append([round(p[a,0],4),round(p[a,1],4),round(float(yaw[a]),4),round(float(speed[a]),4),state,round(float(distance[a]),4)])
            frames.append({'t':round(float(t),4),'people':rows})
        result={'schema':'stagezero.crowd.v1','environment':self.cfg.environment,'units':'meters','up_axis':'Y','dt':self.cfg.dt,'duration':self.cfg.duration,'seed':self.cfg.seed,'config':asdict(self.cfg),'state_names':STATE_NAMES,'agents':agents,'frames':frames,'events':events,'layout':self.layout.to_dict(),'navigation_method':'Deterministic prioritized space-time A* with swept-disc reservations and repeated bounded tasks','traffic':({'period_s':54,'red_s':10,'walk_s':36,'clearance_s':8} if self.cfg.environment=='crossing' else None)}
        if self.cfg.environment=='crossing':result['layout']['route_policy']='Opposite-corner destinations through the two4.06m painted diagonal corridors; no new road entry without predicted clearance, and no reserved road occupancy during WAIT.'
        finalize_crossing_curb_start(result);ease_city_formation_facing(result);normalize_task_legs(result);result['metrics']=measure(result,self.layout);return result

def ease_city_formation_facing(data):
    """Reconstruct the settled turn before the native shared-yaw dance begins."""
    if data['environment']!='city':return data
    for event in data.get('events',[]):
        if event.get('kind')!='flashmob':continue
        for actor in event['actor_ids']:
            for i,frame in enumerate(data['frames']):
                if not 36<=frame['t']<40:continue
                previous=data['frames'][i-1]['people'][actor][2];p=frame['people'][actor];n=data['frames'][i+1]['people'][actor]
                target=math.atan2(n[0]-p[0],n[1]-p[1]) if math.dist(n[:2],p[:2])>.001 else event.get('shared_yaw',0.)
                delta=math.atan2(math.sin(target-previous),math.cos(target-previous));p[2]=round(previous+max(-2.8*data['dt'],min(2.8*data['dt'],delta)),4)
    return data

def finalize_crossing_curb_start(data):
    """Place an initially approaching featured actor on its first safe curb sample.

    Hold there until its reserved route joins; never jump a root during playback.
    This deterministic placement correction is always followed by full validation.
    """
    if data['environment']!='crossing':return data
    road=lambda p:abs(p[0])<12.76 or abs(p[1])<12.76
    corrected=[]
    for actor,agent in enumerate(data['agents']):
        if not road(data['frames'][0]['people'][actor]):continue
        joins=[i for i,f in enumerate(data['frames']) if f['t']<=1 and abs(f['t']/TICK-round(f['t']/TICK))<1e-6 and not road(f['people'][actor])]
        if not joins:raise ValueError('Initial crossing actor lacks a safe continuous curb join')
        join=joins[0];position=data['frames'][join]['people'][actor][:2]
        for f in data['frames'][:join]:f['people'][actor][:2]=position
        agent['start_xz']=list(position);pos=np.array([f['people'][actor][:2] for f in data['frames']]);steps=np.linalg.norm(np.diff(pos,axis=0),axis=1);distance=np.r_[0,np.cumsum(steps)];speed=np.r_[steps/data['dt'],0]
        for i,f in enumerate(data['frames']):f['people'][actor][3]=round(float(speed[i]),4);f['people'][actor][5]=round(float(distance[i]),4)
        corrected.append({'actor_id':agent['id'],'hold_until_s':data['frames'][join]['t'],'position_xz':list(position)})
    if corrected:data['initial_curb_placements']=corrected
    return data

def apply_crossing_queue_cues(data,manifest):
    """Reuse the existing generated gesture authoring for stationary queue tasks."""
    if data['environment']!='crossing' or data.get('motion_cues',{}).get('queue_cues_applied'):return data
    import copy
    from crowd_demo_cues import apply_motion_cues
    pending=copy.deepcopy(data);pending.pop('motion_cues',None);pending['events']=[]
    for agent in pending['agents']:
        agent['schedule']=[dict(task,task='watch') for task in agent['schedule'] if task['task']=='signal_wait']
    authored=apply_motion_cues(pending,manifest)
    for old,new in zip(data['frames'],authored['frames']):
        for a,b in zip(old['people'],new['people']):
            if b[6]>=0 and a[6]<0:a[6:9]=b[6:9]
    cues=data['motion_cues'];cues['queue_cues_applied']=True;cues['queue_cues']=authored['motion_cues']['dwell_cues'];cues['queue_gesture_events']=len(cues['queue_cues'])
    return data

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
    agent['schedule']=[{'start':float(t),'end':float(min(t+10,120)),'task':'serve','label':'Attend the café counter and acknowledge nearby customers','target':[-29.,-23.53]} for t in range(0,120,10)]
    agent['task_program']={'repeating':True,'stationary_work_role':True,'dwell_range_s':[10,10],'completed_routes':0};agent['start_xz']=[-29.,-23.53]
    lookup={c['id']:i for i,c in enumerate(manifest['clips'])}
    for f in data['frames']:
        t=f['t'];local=t%10;clip=-1;ctime=0.
        nearby=any(j!=index and abs(row[0]+29)<3 and -22.2<row[1]<-18.5 for j,row in enumerate(f['people']))
        if nearby and 2<=local<=3.9:clip=lookup['listen'];ctime=local-2
        elif 6<=local<=7.9:clip=lookup['look'];ctime=local-6
        f['people'][index]=[-29.,-23.53,0.,0.,1,0.,clip,ctime,0.]
    if 'motion_cues' in data:
        cues=data['motion_cues'];cues['dwell_cues']=[c for c in cues['dwell_cues'] if c['actor_id']!=63];cues['dwell_gesture_events']=len(cues['dwell_cues']);cues['stationary_staff']={'actor_id':63,'role':'cafe attendant','gesture_policy':'Generated listening acknowledgements only with a nearby counter customer, plus subtle look cues during repeated ten-second service tasks; physically present throughout.'}
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
    steps=np.linalg.norm(np.diff(p,axis=0),axis=2);dt=data['dt'];stationary_run=np.zeros(p.shape[1]);stationary_max=np.zeros(p.shape[1]);queue_run=np.zeros(p.shape[1]);queue_max=np.zeros(p.shape[1])
    for f,movement in enumerate(steps):
        still=movement<.003;stationary_run=np.where(still,stationary_run+dt,0);stationary_max=np.maximum(stationary_max,stationary_run);queue_run=np.where(still&(a[f,:,4]==4),queue_run+dt,0);queue_max=np.maximum(queue_max,queue_run)
    run=np.zeros(p.shape[1]);maxrun=np.zeros_like(run);deadlocks=0
    for f,movement in enumerate(steps):
        active=np.isin(a[f,:,4],[0,5]);run=np.where((movement<.003)&active,run+dt,0);maxrun=np.maximum(maxrun,run)
    # Swept-pair check catches collisions hidden between cached samples.
    swept=0;sweptmin=math.inf
    for f in range(len(p)-1):
        rel=p[f,:,None]-p[f,None,:];vel=(p[f+1]-p[f]);dv=vel[:,None]-vel[None,:];q=np.sum(dv*dv,axis=2);u=np.clip(-np.sum(rel*dv,axis=2)/np.maximum(q,1e-12),0,1);d=np.linalg.norm(rel+dv*u[:,:,None],axis=2);np.fill_diagonal(d,np.inf);sweptmin=min(sweptmin,float(d.min()));swept+=int(np.count_nonzero(np.triu(d<r[:,None]+r[None,:]-.001,1)))
    interior=sum(bool(np.any((p[:,i,1]<-17.1)&(p[:,i,1]>-23.7)&(p[:,i,0]>-34.5)&(p[:,i,0]<-11.5))) for i in range(p.shape[1])) if data['environment']=='city' else 0
    result={'actor_count':p.shape[1],'duration_s':data['duration'],'frames':len(p),'pair_overlap_samples':overlaps,'swept_pair_overlap_samples':swept,'min_pair_distance_m':round(min(mins),4),'min_swept_pair_distance_m':round(sweptmin,4),'max_proxy_penetration_m':round(max(0,maxpenetration),4),'static_collision_samples':static,'outside_walkable_samples':outside,'max_root_acceleration_mps2':round(float(np.linalg.norm(np.diff(np.diff(p,axis=0)/dt,axis=0),axis=2).max()/dt),3),'heading_interpretation':'Navigation yaw eased at2.8rad/s; explicit native pair common-yaw is a source coordinate transform, not body heading','root_motion_interpretation':'Piecewise linear reserved-grid motion; acceleration is discontinuous at route vertices and dwell boundaries','max_root_step_m':round(float(steps.max()),4),'max_root_speed_mps':round(float(steps.max()/dt),4),'root_jump_samples':int(np.count_nonzero(steps>2*dt)),'max_unscheduled_stall_s':round(float(maxrun.max()),3),'deadlocked_actor_count':int(np.count_nonzero(maxrun>12)),'max_continuous_stationary_s':round(float(stationary_max.max()),3),'max_signal_queue_wait_s':round(float(queue_max.max()),3),'actors_stationary_over_30s':int(np.count_nonzero(stationary_max>30)),'stall_scope':'Deadlock count covers unscheduled walking/yielding stalls only. Signal queues, scheduled dwell and staff work are measured separately; zero deadlocks does not mean zero long pauses.','min_actor_travel_m':round(float(steps.sum(axis=0).min()),3),'min_mobile_actor_travel_m':round(float(min(steps[:,i].sum() for i,agent in enumerate(data['agents']) if not agent.get('stationary_work_role'))),3),'stationary_work_actor_ids':[agent['id'] for agent in data['agents'] if agent.get('stationary_work_role')],'median_actor_travel_m':round(float(np.median(steps.sum(axis=0))),3),'interior_visitors':interior,'full_body_collision':'not measured; disc proxy only','foot_sliding':'not measured by root navigation','validation_scope':'All frames and continuous linear pair motion between cached samples; authored ground obstacles'}
    if data['environment']=='city':
        road=np.abs(p[:,:,1])<12
        painted=((p[:,:,0]>=-8.5)&(p[:,:,0]<=-4.5))|((p[:,:,0]>=5)&(p[:,:,0]<=9))
        result['unpainted_road_actor_samples']=int((road&~painted).sum())
        result['actors_using_painted_crossings']=int(np.any(road&painted,axis=0).sum())
        result['actors_completing_street_crossing']=int(np.any((p[:,:,1]*p[0,:,1]<0)&(np.abs(p[:,:,1])>12),axis=0).sum())
        for event in data.get('events',[]):
            if event.get('kind')!='flashmob':continue
            ids={agent['id']:i for i,agent in enumerate(data['agents'])};dancers=[ids[i] for i in event['actor_ids']];outsiders=[i for i in range(len(ids)) if i not in dancers]
            ts=np.array([f['t'] for f in data['frames']]);during=(ts>=event['dance_start_s'])&(ts<=event['dance_end_s']);lo_x,lo_z,hi_x,hi_z=event['outsider_exclusion_xz'];outside_positions=p[during][:,outsiders]
            result['flashmob_actor_count']=len(dancers)
            result['outsider_formation_samples']=int(((outside_positions[:,:,0]>lo_x)&(outside_positions[:,:,0]<hi_x)&(outside_positions[:,:,1]>lo_z)&(outside_positions[:,:,1]<hi_z)).sum())
            result['max_dance_slot_root_offset_m']=round(float(np.linalg.norm(p[during][:,dancers]-np.array(event['slot_xz']),axis=2).max()),4)
            result['flashmob_window_s']=[event['dance_start_s'],event['dance_end_s']]
    if data['environment']=='crossing':
        ts=np.array([f['t'] for f in data['frames']]);green=(ts%54>=10)&(ts%54<46);wait=ts%54<10
        central=(np.abs(p[:,:,0])<12.76)&(np.abs(p[:,:,1])<12.76);road=(np.abs(p[:,:,0])<12.76)|(np.abs(p[:,:,1])<12.76)
        occupancy=central.sum(axis=1);green_counts=np.where(green,occupancy,-1);peak=int(np.argmax(green_counts));first=np.flatnonzero(green&(occupancy>=40))
        entries=(~road[:-1])&road[1:];entry_phase=ts[1:]%54
        result.update(peak_green_central_actors=int(occupancy[peak]),peak_green_central_time_s=float(ts[peak]),first_40_central_time_s=float(ts[first[0]]) if len(first) else None,road_actor_samples_during_wait=int(road[wait].sum()),road_entry_samples_outside_cross=int(entries[(entry_phase<10)|(entry_phase>=46)].sum()),central_region='abs(x)<12.76 and abs(z)<12.76 metres; excludes outer arterial arms',mean_green_central_actors=round(float(occupancy[green].mean()),3),actors_visiting_central_10m_square=int(np.any((np.abs(p[:,:,0])<5)&(np.abs(p[:,:,1])<5),axis=0).sum()),actors_reaching_opposite_corner=int(np.any((p[:,:,0]*p[0,:,0]<0)&(p[:,:,1]*p[0,:,1]<0)&(np.abs(p[:,:,0])>12.76)&(np.abs(p[:,:,1])>12.76),axis=0).sum()))
    return result

def validate(data):
    ids=[a['id'] for a in data['agents']]
    if len(set(ids))!=len(ids):raise ValueError('Actor IDs must be stable and unique')
    times=np.array([f['t'] for f in data['frames']])
    if len(times)!=round(data['duration']/data['dt'])+1 or not np.allclose(times,np.arange(len(times))*data['dt'],atol=1e-6):raise ValueError('Trajectory must cover every shared-clock sample')
    m=measure(data)
    for key in ['pair_overlap_samples','swept_pair_overlap_samples','static_collision_samples','outside_walkable_samples','root_jump_samples','deadlocked_actor_count']:
        if m[key]:raise ValueError(f'{key}: {m[key]}')
    if m['min_mobile_actor_travel_m']<10:raise ValueError('actor did not sustain movement')
    if data['environment']=='city':
        if m['unpainted_road_actor_samples']:raise ValueError('City pedestrian left the painted crossing')
        if m.get('outsider_formation_samples'):raise ValueError('Outside actor entered reserved dance formation')
        if m.get('max_dance_slot_root_offset_m',0)>.42:raise ValueError('Dance root left its reviewed slot envelope')
    if data['environment']=='crossing':
        if m['road_actor_samples_during_wait']:raise ValueError('Crossing occupies roadway during WAIT')
        if m['road_entry_samples_outside_cross']:raise ValueError('Crossing enters roadway outside CROSS')
        if m['peak_green_central_actors']<min(40,len(data['agents'])*.35):raise ValueError('Crossing does not create a busy central scramble')
        if m['actors_reaching_opposite_corner']<len(data['agents'])*.8:raise ValueError('Too many actors failed to complete an opposite-corner crossing')
    return m

def generate(cfg=DemoConfig()):return Planner(cfg).build()
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--environment',choices=['crossing','city','station'],default='crossing');ap.add_argument('--count',type=int,default=112);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();d=generate(DemoConfig(args.environment,args.count));validate(d);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(d,separators=(',',':')));print(json.dumps(d['metrics'],indent=2))
