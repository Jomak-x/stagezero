"""Conservative ground footprints mirrored from authored environment geometry."""
from dataclasses import dataclass, asdict
import math
@dataclass(frozen=True)
class Rect:
    id:str; x0:float; z0:float; x1:float; z1:float
    def clear(self,x,z,radius=0.):
        dx=max(self.x0-x,0,x-self.x1);dz=max(self.z0-z,0,z-self.z1)
        return dx*dx+dz*dz>=radius*radius and not(self.x0<x<self.x1 and self.z0<z<self.z1)
@dataclass
class Layout:
    kind:str; walkable:list; obstacles:list; destinations:list; meetings:list
    def valid(self,x,z,radius=.3):
        return any(r.x0<=x<=r.x1 and r.z0<=z<=r.z1 for r in self.walkable) and all(r.clear(x,z,radius) for r in self.obstacles)
    def to_dict(self):
        return {'source':['studio_client/src/environments/EnvironmentScene.ts','studio_client/src/environments/StreetDressing.ts'],'kind':self.kind,'units':'metres','ground_y':0,'crossing_horizontal_scale':.58 if self.kind=='crossing' else 1,'walkable':[asdict(r) for r in self.walkable],'obstacles':[asdict(r) for r in self.obstacles],'doorways':([{'id':'cafe','x':-29,'z':-16.85,'width':2.3},{'id':'books','x':-17,'z':-16.85,'width':2.3}] if self.kind=='city' else []),'limitations':['Conservative axis-aligned ground footprints; proxy discs do not certify full-body contact or foot sliding.','Walkable joins are flat; decorative station steps excluded.']}
def make_layout(kind):
    obs=[]
    def box(name,x,z,w,d,angle=0,scale=1):
        a=(abs(math.cos(angle))*w+abs(math.sin(angle))*d)*scale/2;b=(abs(math.sin(angle))*w+abs(math.cos(angle))*d)*scale/2
        obs.append(Rect(name,x*scale-a,z*scale-b,x*scale+a,z*scale+b))
    def disc(name,x,z,r,scale=1):box(name,x,z,2*r,2*r,scale=scale)
    if kind=='crossing':
        s=.58;walk=[Rect('scramble',-24*s,-24*s,24*s,24*s)]
        for sign in [-1,1]:
            a,b=(-30*s,-22*s) if sign<0 else (22*s,30*s)
            walk +=[Rect('sidewalk-ew',-40,a,40,b),Rect('sidewalk-ns',a,-40,b,40)]
        for sx in [-1,1]:
            for sz in [-1,1]:
                box('corner-building',sx*43,sz*42,24,20,scale=s)
                for i,w in enumerate([16,20,15,21]):
                    box('street-building',sx*(65+i*19.5),sz*(38+i%2*2),w,21,scale=s);box('side-building',sx*39,sz*(67+i*20),18,22,math.pi/2,scale=s)
                disc('signal',sx*23.5,sz*25.3,.27,s);disc('lamp',sx*26.5,sz*49,.20,s);disc('tree-planter',sx*27.7,sz*65,1.05,s);box('bench',sx*28,sz*60,2.2,.8,math.pi/2,scale=s)
                for i in range(3):disc('bollard',sx*(25+i*2),sz*24.2,.13,s)
        for sz in [-1,1]:
            for x in [-43,43]:
                for dx in [-4,4]:box('flower-planter',x+dx,sz*29.7,1.5,.68,scale=s)
        for sx in [-1,1]:
            for x in [53.5,55]:box('vending',sx*x,-29,1.05,.68,scale=s)
        dest=[('corner',sx*x,sz*z) for sx in [-1,1] for sz in [-1,1] for x,z in [(15.8,16),(22,15.5),(15.5,23),(15.5,30)]];meets=[(-23,15.5),(23,-15.5),(-15.5,-24)]
    elif kind=='city':
        walk=[Rect('north-sidewalk',-83,-16,83,-12.3),Rect('south-sidewalk',-83,12.3,83,16),Rect('west-crosswalk',-8.2,-12.6,-4.8,12.6),Rect('east-crosswalk',5.3,-12.6,8.7,12.6),Rect('cafe',-34.5,-24,-23.5,-16),Rect('books',-22.5,-24,-11.5,-16),Rect('alley',-40,-66,-35,-12),Rect('courtyard',-67,-66,-29,-46)]
        for x,z,w,d in [(-51,-26,22,18),(-73,-25,18,16),(21,-26,19,18),(44,-26,22,18),(69,-27,24,20),(-49,-80,35,16)]:box('building',x,z,w,d)
        for i in range(6):box('south-building',-67+i*26,26+i%2,23,17)
        for sign in [-1,1]:
            for i in range(3):box('side-building',sign*22,-52-i*22,19,18,math.pi/2)
        for x in [-29,-17]:
            box('shop-rear',x,-24,12,.18)
            for side in [-1,1]:
                box('shop-side',x+side*5.87,-20.5,.18,7);box('shop-front-glass',x+side*(1.15+4.85/2),-16.85,4.85,.10);box('door-frame',x+side*1.15,-16.85,.07,.09)
        box('cafe-counter',-29,-22.7,11,1.05)
        for side in [-1,1]:
            for row in range(2):
                x,z=-29+side*3.36,-18.8-row*2;disc('cafe-table',x,z,.55)
                for dz in [-.72,.72]:box('cafe-chair',x,z+dz,.44,.85)
            box('book-shelf',-17+side*5.45,-21.1,.90,4.8)
        box('books-back-shelf',-17,-23.5,11,.85);box('book-display',-17,-20.1,1.2,1.8)
        for x in range(-70,76,24):
            disc('lamp',x,-13.3,.16);disc('lamp',x+7,13.3,.16);disc('tree',x+8,-14.3,.8);box('bench',x+4.8,-14.4,2.2,.8)
        box('menu',-33,-15.5,.9,.25)
        for i in range(5):box('bike-rack',31+i*1.15+.32,-14,.75,.12)
        for x in [-78,-58,55,79]:box('flower-planter',x,-15.8,1.7,.68)
        for x in [79,80.2]:box('vending',x,-15.9,1.05,.68)
        for x in [-62,-35]:disc('courtyard-tree',x,-56,.95);box('courtyard-bench',x+2.6,-56,2.2,.8,math.pi/2)
        for x,z in [(-58,-62),(-52,-63)]:box('courtyard-tables-chairs',x,z,2.6,1.3)
        for x in [-64,-45]:box('courtyard-planter',x,-63,1.8,.68)
        box('flashmob-speaker',-57.5,-59.5,1,.7)
        dest=[('west-sidewalk',-55,-13),('east-sidewalk',50,14),('cafe-order',-29,-21.25),('books-browse',-19,-21),('courtyard-edge',-61,-49)];meets=[(-25,-14),(23,14),(-61,-49)]
    elif kind=='station':
        walk=[Rect('plaza',-56,-3,56,35),Rect('level-approach',-9,-37,9,32),Rect('concourse',-39,-61,39,-32),Rect('bus-crossing',-3,31,3,54),Rect('bus-walk',-53,53,53,61)]
        box('hall-back',0,-63,82,.6)
        for x in range(-40,41,5):
            box('hall-column',x,-32,.2,.25);box('hall-column',x,-62,.2,.25)
            if abs(x)>9:box('hall-glass',x+2.5,-31.98,4.8,.08)
        for side in [-1,1]:
            box('hall-side',side*40.5,-47,.5,32);box('terrace-and-stairs',side*41,-9.5,15,18.5)
            for i in range(5):box('ticket-gate',side*(12+i*3),-45,.55,1.3)
            for i in range(4):box('ticket-machine',side*31,-56+i*2,1.4,.8)
            for i in range(3):disc('tree',side*(32+i*10),15,.95)
            for i in range(4):box('bench',side*(19+i*10),17,2.2,.8)
            for x,z in [(side*13,4),(side*52,-3)]:disc('lamp',x,z,.2)
            box('kiosk-counter',side*22,4.3,5,4.3,side*-.25)
        disc('clock',11,9,.14)
        for x in [-34,-27,27,34]:box('flower-planter',x,9,2,.68)
        for x in [-53,53]:box('vending',x,3,1.05,.68);box('vending',x+1.2,3,1.05,.68)
        for x in [-49,49]:box('cafe-table-chairs',x,23,2.6,1.3)
        for i in range(-2,3):
            box('bus-shelter-glass',i*20,56.2,6,.1);box('bus-bench',i*20,56,2.2,.8)
            for dx in [-3,3]:box('bus-post',i*20+dx,56,.1,.1)
        for side in [-1,1]:
            for i in range(6):disc('bollard',side*(6+i*8),37,.13)
        dest=[('plaza',x,z) for x in [-40,-20,0,20,40] for z in [12,27]]+[('concourse',x,-50) for x in [-20,0,20]]+[('arrival',0,-20),('kiosk-queue',-22,8),('bus',0,59)];meets=[(-12,26),(14,24),(-5,-42)]
    else:raise ValueError(f'unknown environment: {kind}')
    return Layout(kind,walk,obs,dest,meets)
