"""Portable attachment points for planning future scene interactions.

Targets describe locations, not executable behaviors or a physics simulation.
Coordinates stay attached when their owning prop moves, rotates or resizes.
"""
import math
import re

TARGET_KINDS=('swing_anchor','landing','climb','vault')
MAX_TARGETS=128


def validate_targets(value,objects):
    if not isinstance(value,list) or len(value)>MAX_TARGETS:
        raise ValueError('Scene targets must be a list of at most 128 entries')
    refs={o['id'] for o in objects};seen=set();result=[]
    for target in value:
        if not isinstance(target,dict) or set(target)!={'id','name','object_id','kind','local_position'}:
            raise ValueError('Interaction target has missing or unsupported fields')
        identifier=target['id'];name=target['name']
        if not isinstance(identifier,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',identifier) or identifier in seen:
            raise ValueError('Invalid or duplicate target ID')
        if not isinstance(name,str) or not 1<=len(name.strip())<=80 or any(ord(c)<32 for c in name):
            raise ValueError('Target name must be printable text')
        if not isinstance(target['object_id'],str) or target['object_id'] not in refs:
            raise ValueError('Target references a missing prop')
        if target['kind'] not in TARGET_KINDS:raise ValueError('Unsupported interaction target kind')
        point=target['local_position']
        if not isinstance(point,(list,tuple)) or len(point)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>.5 for v in point):
            raise ValueError('Target position must be inside normalized prop bounds')
        seen.add(identifier)
        result.append(dict(target,name=name.strip(),local_position=[float(v) for v in point]))
    return result


def resolve_targets(targets,objects):
    lookup={o['id']:o for o in objects};result=[]
    for target in targets:
        obj=lookup[target['object_id']]
        x,y,z=[v*s for v,s in zip(target['local_position'],obj['size'])]
        angle=math.radians(obj.get('yaw',0));c,s=math.cos(angle),math.sin(angle)
        offset=(c*x+s*z,y,-s*x+c*z)
        result.append(dict(target,position=[p+d for p,d in zip(obj['position'],offset)]))
    return result


class TargetLayer:
    def __init__(self,server):
        self.server=server;self.handles={};self.signature=None

    def update(self,targets,objects):
        points=resolve_targets(targets,objects)
        signature=tuple((t['id'],t['kind'],tuple(t['position'])) for t in points)
        if signature==self.signature:return
        self.signature=signature
        active={t['id'] for t in points}
        for identifier in set(self.handles)-active:self.handles.pop(identifier).remove()
        colors={'swing_anchor':(74,211,245),'landing':(102,229,143),'climb':(247,186,77),'vault':(225,124,210)}
        for target in points:
            if target['id'] not in self.handles:
                self.handles[target['id']]=self.server.scene.add_icosphere('/interaction-targets/'+target['id'],radius=.20,color=colors[target['kind']])
            self.handles[target['id']].color=colors[target['kind']]
            self.handles[target['id']].position=tuple(target['position'])
