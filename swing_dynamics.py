"""Fixed 60 Hz hybrid web dynamics: authored lift/steering, gravity and ropes.

Root motion is continuous and swept against conservative imported mesh AABBs.
This is a playable hybrid controller, not a rigid-body or learned animation model.
"""
from __future__ import annotations
import copy
import math
import time
from dataclasses import dataclass

DT = 1 / 60

def add(a,b): return tuple(x+y for x,y in zip(a,b))
def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def mul(a,s): return tuple(x*s for x in a)
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def norm(a): return math.sqrt(dot(a,a))
def unit(a): return mul(a,1/max(norm(a),1e-12))
def limit(a,m): return mul(a,min(1,m/max(norm(a),1e-12)))
def mix(a,b,t): return add(mul(a,1-t),mul(b,t))
def smooth(t):
    t=max(0.,min(1.,t)); return t*t*(3-2*t)
def vector(v):
    if not isinstance(v,(tuple,list)) or len(v)!=3 or not all(isinstance(x,(float,int)) and math.isfinite(x) for x in v):
        raise ValueError('Expected three finite coordinates')
    return tuple(float(x) for x in v)
def rotate(v,yaw):
    x,y,z=v; return (math.cos(yaw)*x+math.sin(yaw)*z,y,-math.sin(yaw)*x+math.cos(yaw)*z)
def segment_hits_box(start,end,box):
    low,high=0.,1.
    for i in range(3):
        delta=end[i]-start[i]
        if abs(delta)<1e-12:
            if not box['min'][i]<=start[i]<=box['max'][i]: return False
        else:
            a=(box['min'][i]-start[i])/delta; b=(box['max'][i]-start[i])/delta
            low=max(low,min(a,b)); high=min(high,max(a,b))
            if low>high: return False
    return high>=0 and low<=1

def sphere_box_clearance(p,r,b):
    lo,hi=b['min'],b['max']
    outside=tuple(max(lo[i]-p[i],0,p[i]-hi[i]) for i in range(3))
    if norm(outside)>0: return norm(outside)-r
    return -min(min(p[i]-lo[i],hi[i]-p[i]) for i in range(3))-r

@dataclass
class Command:
    id:int
    kind:str
    submitted_at:float
    received_at:float
    target:tuple|None=None
    direction:tuple|None=None
    applied_at:float|None=None
    applied_frame:int|None=None
    status:str='queued'
    reason:str|None=None

class SwingController:
    """Commands apply at next tick; timestamp and clock share monotonic seconds.

    Latency is measured receipt-to-control-update; simulation time is separate.
    Single-thread ownership required. step() is 1/60 s; advance() accumulates time.
    """
    carry_offset=(0.,.22,-.29)
    hip_height=.96

    def __init__(self,scene,*,clock=time.monotonic):
        self.scene=copy.deepcopy(scene)
        self.buildings=self.scene['buildings']; self.anchors=self.scene['anchors']
        self.landing_zones=self.scene['landing_zones']
        if not self.anchors or not self.landing_zones: raise ValueError('Scene needs anchors and landing zones')
        for box in self.buildings:
            box['min'],box['max']=vector(box['min']),vector(box['max'])
            if any(a>=b for a,b in zip(box['min'],box['max'])): raise ValueError('Invalid building bounds')
        self.root=vector(self.scene['spawn']); self.velocity=(0.,0.,0.)
        self.mj_root=vector(self.scene['mj_spawn']); self.mj_pickup=self.mj_root
        self.clock=clock; self.frame=0; self.sim_time=0.; self.accumulator=0.
        self.yaw=math.pi; self.mj_yaw=math.pi; self.phase='ready'; self.carrying=False
        self.anchor=None; self.rope_length=0.; self.direction=(0.,0.,-1.)
        self.target=None; self.route=[]; self.commands=[]; self.pending=[]; self.events=[]
        self.clearance_min=float('inf'); self.collision_corrections=0
        self.max_step=0.; self.max_speed=0.; self.anchor_since=-100.
        self.flight_height=float(self.scene.get('flight_height',4.5))
        self.pickup_yaw=self.yaw; self.attach_origin=self.mj_root; self.attach_time=0.
        self.settle_origin=None; self.settle_time=0.; self.kiss_time=0.
        self.roam_z_min=min(b['min'][2] for b in self.buildings)+1.2
        self.roam_z_max=max(self.root[2]+1.,max(b['max'][2] for b in self.buildings))
        self.auto_turn_since=-100.
        self.hand_local=(.22,.98,.10)
        self.web_validation_frame=-1
        self.web_validated_frames=0; self.web_visible_frames=0; self.web_occlusion_releases=0
        self._refresh_clearance()
        if self.clearance_min < -1e-6: raise ValueError(f'Spawn intersects imported city: {self.clearance_min:.3f}m')

    def command(self,kind,*,target=None,direction=None,timestamp=None):
        kind={'start':'swing','pickup':'carry','carry_mj':'carry'}.get(kind,kind)
        if kind not in {'swing','steer','left','right','carry','land','kiss'}: raise ValueError('Unknown command')
        now=self.clock(); stamp=now if timestamp is None else float(timestamp)
        if not math.isfinite(stamp): raise ValueError('Timestamp must be finite')
        cmd=Command(len(self.commands)+1,kind,stamp,now,None if target is None else vector(target),None if direction is None else vector(direction))
        self.commands.append(cmd); self.pending.append(cmd); return cmd.id

    def _event(self,name,**details):
        self.events.append({'frame':self.frame,'time':self.sim_time,'event':name,**details})

    def _apply(self,cmd):
        cmd.applied_at=self.clock(); cmd.applied_frame=self.frame; cmd.status='applied'
        if cmd.kind=='kiss':
            if self.phase=='kiss': return
            if self.phase not in {'landed','kiss'} or not self.carrying:
                cmd.status='rejected'; cmd.reason='Land safely with MJ before kissing'
            else:
                self.phase='kiss'; self.kiss_time=self.sim_time; self._event('kiss_started')
            return
        if cmd.kind=='carry':
            if self.phase=='pickup': return
            if self.carrying:
                cmd.status='rejected'; cmd.reason='MJ is already attached'; return
            self.phase='pickup'; self.pickup_yaw=self.yaw
            pickup=sub(self.mj_root,rotate((self.carry_offset[0],0.,self.carry_offset[2]),self.pickup_yaw))
            pickup=(pickup[0],pickup[1]+.025,pickup[2])
            self._route_to(pickup)
            self._event('pickup_requested'); return
        if cmd.kind=='land':
            if self.phase in {'landing','settling','landed','kiss'}: return
            zone=next((z for z in self.landing_zones if cmd.target is None and z['id']==self.scene.get('landing_roof_id')),None)
            if zone is None: zone=min(self.landing_zones,key=lambda x:norm(sub(vector(x['position']),cmd.target or self.root)))
            self.phase='landing'; self.landing_zone=zone['id']
            destination=list(zone['position'])
            if not zone.get('hip_position',False):
                top=next((b['max'][1] for b in self.buildings if b['id']==zone.get('building_id')),destination[1])
                destination[1]=max(destination[1],top)+self.hip_height
            self._route_to(vector(destination)); self._event('landing_requested',zone=zone['id']); return
        if self.phase in {'settling','landed','kiss'}:
            cmd.status='rejected'; cmd.reason='Scene has landed; reset to begin another flight'; return
        if self.phase in {'pickup','landing'}: self.route=[]
        if cmd.direction is not None: d=(cmd.direction[0],0.,cmd.direction[2])
        elif cmd.target is not None:
            d=sub(cmd.target,self.root); d=(d[0],0.,d[2])
        elif cmd.kind in {'left','right'}: d=rotate(self.direction,-.65 if cmd.kind=='left' else .65)
        else: d=self.direction
        if norm(d)>1e-8: self.direction=unit(d)
        self.phase='swing'; self.target=cmd.target
        self._select_anchor(force=True)
        self._event('steering_applied',command=cmd.id,direction=list(self.direction))

    def _route_to(self,destination):
        # Conservative up/across/down route clears actual imported roof extents.
        top=max((b['max'][1] for b in self.buildings),default=0.)
        altitude=max(top+2.8,self.root[1],destination[1]+2.5)
        self.route=[(self.root[0],altitude,self.root[2]),(destination[0],altitude,destination[2]),destination]
        self.anchor=None; self.target=destination

    def _select_anchor(self,force=False,hand=None):
        if self.phase!='swing' or (not force and self.sim_time-self.anchor_since<2.4): return
        hand=self._estimated_hand() if hand is None else hand
        candidates=[]
        for item in self.anchors:
            p=vector(item['position']); d=sub(p,self.root)
            if p[1]>self.root[1]+.5 and 2<norm(d)<28 and dot((d[0],0.,d[2]),self.direction)>-.5 and self.web_clear(hand,p,item.get('building_id'),item.get('terminal_allowance_m',.4)):
                candidates.append((norm(d)-4*dot(unit((d[0],0,d[2])),self.direction),item))
        if not candidates: self.anchor=None; return
        self.anchor=min(candidates,key=lambda x:x[0])[1]
        self.rope_length=norm(sub(self.root,vector(self.anchor['position'])))
        self.anchor_since=self.sim_time; self._event('web_attached',anchor=self.anchor['id'])

    def _estimated_hand(self):
        return add(self.root,rotate(self.hand_local,self.yaw))

    def validate_web(self,hand):
        """Validate the exact rendered wrist after pose generation, before publish.

        Call once per frame, then refresh snapshot(). Occluded tethers are released
        and only an anchor visible from this exact wrist may be selected. The next
        physics tick uses this measured local offset for candidate visibility.
        """
        hand=vector(tuple(hand))
        self.hand_local=rotate(sub(hand,self.root),-self.yaw)
        first_validation=self.web_validation_frame!=self.frame
        self.web_validation_frame=self.frame
        if self.anchor is not None and not self.web_clear(hand,vector(self.anchor['position']),self.anchor.get('building_id'),self.anchor.get('terminal_allowance_m',.4)):
            previous=self.anchor['id']; self.anchor=None; self.anchor_since=-100.
            self.web_occlusion_releases+=1
            self._event('web_released',anchor=previous,reason='rendered_hand_segment_occluded')
            self._select_anchor(force=True,hand=hand)
        if self.phase=='swing' and first_validation:
            self.web_validated_frames+=1
            self.web_visible_frames+=self.anchor is not None
        return self.anchor is not None

    def web_clear(self,start,end,anchor_building=None,terminal_allowance=.4):
        for source_box in self.buildings:
            box={'id':source_box['id'],'min':[v-.025 for v in source_box['min']],
                 'max':[v+.025 for v in source_box['max']]}
            if box['id']==anchor_building:
                # Only the terminal allowance measured from the actual source
                # vertex may enter its own conservative decorative-trim AABB.
                distance=norm(sub(end,start))
                clipped=mix(start,end,max(0.,1-max(.4,terminal_allowance)/max(distance,1e-9)))
                if segment_hits_box(start,clipped,box): return False
            elif segment_hits_box(start,end,box): return False
        return True

    def proxy_spheres(self,root=None,yaw=None):
        root=self.root if root is None else root; yaw=self.yaw if yaw is None else yaw
        spheres=[(add(root,(0.,y,0.)),.45,'spider') for y in (-.5,0.,.5)]
        spheres += [(add(root,rotate((x,.3,0.),yaw)),.3,'spider-arm') for x in (-.5,.5)]
        spheres.append((add(root,rotate(self.hand_local,yaw)),.3,'web-hand'))
        if self.carrying:
            p=add(root,rotate(self.carry_offset,yaw))
            spheres += [(add(p,rotate(v,yaw)),.38,'mj') for v in [(0,0,0),(0,.45,0),(-.35,-.15,.2),(.35,-.15,.2)]]
        return spheres

    def clearance(self,root=None,yaw=None):
        return min((sphere_box_clearance(p,r,b) for p,r,_ in self.proxy_spheres(root,yaw) for b in self.buildings),default=1000.)

    def _refresh_clearance(self):
        c=self.clearance(); self.clearance_min=min(self.clearance_min,c); return c

    def _move_safely(self,displacement):
        count=max(1,math.ceil(norm(displacement)/.04)); increment=mul(displacement,1/count)
        for _ in range(count):
            candidate=add(self.root,increment)
            if candidate[1]>=self.hip_height and self.clearance(candidate)>=-1e-8:
                self.root=candidate; continue
            self.collision_corrections+=1
            for axis in sorted(range(3),key=lambda i:-abs(increment[i])):
                step=[0.,0.,0.]; step[axis]=increment[axis]; candidate=add(self.root,step)
                if candidate[1]>=self.hip_height and self.clearance(candidate)>=-1e-8: self.root=candidate
                else:
                    vel=list(self.velocity); vel[axis]=0.; self.velocity=tuple(vel)

    def _avoidance(self):
        force=(0.,0.,0.)
        for b in self.buildings:
            if self.root[1]-.95>b['max'][1]+.1: continue
            nearest=(max(b['min'][0],min(b['max'][0],self.root[0])),self.root[1],max(b['min'][2],min(b['max'][2],self.root[2])))
            delta=sub(self.root,nearest); distance=norm(delta); margin=2.3 if self.carrying else 1.8
            if 0<distance<margin+1.5: force=add(force,mul(unit(delta),10*max(0.,margin+1.5-distance)))
        return limit(force,18.)

    def step(self,dt=DT):
        if abs(dt-DT)>1e-10: raise ValueError('step requires fixed 1/60 s; use advance(seconds)')
        self.frame+=1; self.sim_time=self.frame*DT
        for cmd in self.pending: self._apply(cmd)
        self.pending.clear(); before=self.root
        if self.phase in {'ready','landed','kiss'}: self.velocity=(0.,0.,0.)
        elif self.phase=='settling':
            self.velocity=(0.,0.,0.)
            t=smooth((self.sim_time-self.settle_time)/1.4)
            endpoint=add(self.root,rotate((.035,-.0576,.42),self.yaw))
            self.mj_root=add(mix(self.settle_origin,endpoint,t),rotate((.7*math.sin(math.pi*t),.12*math.sin(math.pi*t),0),self.yaw))
            if t>=1: self.phase='landed'; self._event('landed',speed=0.)
        else:
            if self.phase=='swing':
                self._select_anchor()
                if self.anchor is not None and not self.web_clear(self._estimated_hand(),vector(self.anchor['position']),self.anchor.get('building_id'),self.anchor.get('terminal_allowance_m',.4)):
                    self.anchor=None; self.anchor_since=-100.; self._select_anchor(force=True)
                if self.sim_time-self.auto_turn_since>1.5 and ((self.root[2]<self.roam_z_min+3.5 and self.direction[2]<0) or (self.root[2]>self.roam_z_max-3.5 and self.direction[2]>0)):
                    self.direction=unit((-self.root[0]*.15,0.,-self.direction[2]))
                    self.auto_turn_since=self.sim_time; self._select_anchor(force=True); self._event('city_boundary_turn')
                desired=mul(self.direction,4.2 if self.carrying else 5.)
                if abs(self.root[0])>1.2:
                    desired=(-min(3.5,abs(self.root[0])*1.8)*math.copysign(1.,self.root[0]),desired[1],desired[2])
                a=mul(sub(desired,self.velocity),2.1)
                height=self.flight_height+1.1*math.sin(self.sim_time*1.5)
                if abs(self.root[0])>2.:
                    nearby=[b['max'][1]+1.4 for b in self.buildings if b['min'][0]-1<self.root[0]<b['max'][0]+1 and b['min'][2]-1<self.root[2]<b['max'][2]+1]
                    if nearby: height=max(height,max(nearby))
                a=(a[0],9.81+4*(height-self.root[1])-2.8*self.velocity[1],a[2])
                a=add(a,self._avoidance())
            else:
                while len(self.route)>1 and norm(sub(self.route[0],self.root))<.25 and norm(self.velocity)<1.4: self.route.pop(0)
                delta=sub(self.route[0],self.root); desired=limit(mul(delta,1.6),4.5)
                a=add(mul(sub(desired,self.velocity),3.5),(0.,9.81,0.))
            a=add(limit(a,25.),(0.,-9.81,0.))
            self.velocity=limit(add(self.velocity,mul(a,DT)),9.)
            if self.anchor is not None:
                anchor=vector(self.anchor['position']); relative=sub(add(self.root,mul(self.velocity,DT)),anchor)
                if norm(relative)>self.rope_length:
                    constrained=add(anchor,mul(unit(relative),self.rope_length))
                    self.velocity=mul(sub(constrained,self.root),1/DT)
            horizontal=(self.velocity[0],0.,self.velocity[2])
            desired_yaw=self.pickup_yaw if self.phase=='pickup' else math.atan2(horizontal[0],horizontal[2])
            if norm(horizontal)>.05 or self.phase=='pickup':
                diff=(desired_yaw-self.yaw+math.pi)%(2*math.pi)-math.pi
                candidate_yaw=self.yaw+max(-DT*2.5,min(DT*2.5,diff))
                if self.clearance(yaw=candidate_yaw)>=-1e-8: self.yaw=candidate_yaw
            self._move_safely(mul(self.velocity,DT))
            if self.phase=='pickup' and len(self.route)==1 and norm(sub(self.target,self.root))<.08 and norm(self.velocity)<.2:
                self.carrying=True
                if self.clearance()<-1e-8: self.carrying=False
                else:
                    self.attach_origin=self.mj_root; self.attach_time=self.sim_time
                    self.phase='swing'; self._select_anchor(force=True)
                    self._event('mj_attached',attachment_error=norm(sub(self.mj_root,add(self.root,rotate(self.carry_offset,self.yaw)))))
            if self.phase=='landing' and len(self.route)==1 and norm(sub(self.target,self.root))<.06 and norm(self.velocity)<.16:
                self.phase='settling'; self.anchor=None
                self.settle_origin=self.mj_root; self.settle_time=self.sim_time
                self._event('landing_contact',speed=norm(self.velocity),target_error=norm(sub(self.target,self.root)))
        if self.carrying and self.phase not in {'settling','landed','kiss'}:
            t=smooth((self.sim_time-self.attach_time)/.45)
            self.mj_root=mix(self.attach_origin,add(self.root,rotate(self.carry_offset,self.yaw)),t)
        desired_mj_yaw=self.pickup_yaw if not self.carrying else self.yaw
        if self.phase in {'settling','landed','kiss'}:
            desired_mj_yaw=self.yaw+math.pi*smooth((self.sim_time-self.settle_time)/1.4)
        yaw_delta=(desired_mj_yaw-self.mj_yaw+math.pi)%(2*math.pi)-math.pi
        self.mj_yaw+=max(-2.5*DT,min(2.5*DT,yaw_delta))
        self.max_step=max(self.max_step,norm(sub(self.root,before)))
        self.max_speed=max(self.max_speed,norm(self.velocity)); self._refresh_clearance()
        return self.snapshot()

    def advance(self,seconds):
        if not math.isfinite(seconds) or seconds<0: raise ValueError('Elapsed time must be finite/nonnegative')
        self.accumulator+=seconds
        while self.accumulator+1e-10>=DT: self.step(); self.accumulator-=DT
        return self.snapshot()

    def _flight_amount(self):
        if self.phase in {'ready','landed','kiss'}: return 0.
        if self.phase=='settling': return 0.
        if self.phase=='landing' and self.route and len(self.route)==1:
            return smooth((self.root[1]-self.target[1])/.8)
        return smooth((self.root[1]-self.hip_height)/1.5)

    def snapshot(self):
        rows=[]
        for c in self.commands:
            row=vars(c).copy(); row['response_latency_ms']=None if c.applied_at is None else max(0.,1000*(c.applied_at-c.received_at)); rows.append(row)
        carry=smooth((self.sim_time-self.attach_time)/.45) if self.carrying else 0.
        if self.phase in {'settling','landed','kiss'}: carry=1-smooth((self.sim_time-self.settle_time)/1.4)
        return {'frame':self.frame,'time':self.sim_time,'root':list(self.root),'velocity':list(self.velocity),'yaw':self.yaw,
                'phase':self.phase,'carrying':self.carrying,'mj_root':list(self.mj_root),'mj_pickup':list(self.mj_pickup),
                'anchor':copy.deepcopy(self.anchor),'rope_length':self.rope_length,'carry_offset':list(self.carry_offset),
                'carry_amount':carry,'kiss_amount':smooth((self.sim_time-self.kiss_time)/1.2) if self.phase=='kiss' else 0.,
                'flight_amount':self._flight_amount(),'grounded':self.phase in {'ready','settling','landed','kiss'},
                'mj_yaw':self.mj_yaw,
                'route':[list(x) for x in self.route],'commands':rows,'events':copy.deepcopy(self.events),
                'metrics':{'minimum_proxy_clearance_m':self.clearance_min,'collision_corrections':self.collision_corrections,
                           'max_root_step_m':self.max_step,'max_speed_m_s':self.max_speed,
                           'web_validated_swing_frames':self.web_validated_frames,'web_visible_swing_frames':self.web_visible_frames,
                           'web_occlusion_releases':self.web_occlusion_releases,
                           'web_tether_coverage':self.web_visible_frames/self.web_validated_frames if self.web_validated_frames else None,
                           'web_exact_hand_validated_frame':self.web_validation_frame,
                           'collision_model':'swept conservative actor/carry spheres vs imported building AABBs',
                           'physics_model':'fixed 60 Hz hybrid authored lift, steering and unilateral rope constraint'}}
