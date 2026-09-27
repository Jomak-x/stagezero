"""Dense, human-scale market and workshop staging with reusable generated props."""
import re
from scene_environments import _part as part, _asset as asset, _object as obj, make_city
from scene_composition import validate_scene


def _box(identifier, name, color):
    return asset(identifier,name,[part('box',(0,0,0),(1,1,1),color)])


def _defaults():
    return [
        _box('set-ground','Paving',(169,155,131)),
        _box('set-wall','Workshop concrete',(139,153,155)),
        _box('set-beam','Steel structural beam',(68,82,88)),
        asset('set-stall','Striped produce market stall',[
            part('box',(0,-.25,0),(.96,.36,.72),(143,101,62)),
            part('box',(-.43,.04,-.3),(.055,.81,.055),(83,69,52),repeat=((2,1,2),(.86,0,.6))),
            part('box',(0,.425,0),(1,.12,1),(184,76,65)),
            part('box',(-.4,.489,0),(.1,.014,.99),(235,213,164),repeat=((5,1,1),(.2,0,0))),
            part('sphere',(-.3,-.03,.08),(.12,.12,.12),(204,134,57),repeat=((5,1,2),(.15,0,.16))),
        ]),
        asset('set-fountain','Tiered town fountain',[
            part('cylinder',(0,-.38,0),(1,.24,1),(171,167,153)),
            part('cylinder',(0,-.245,0),(.85,.025,.85),(87,156,173)),
            part('cylinder',(0,-.02,0),(.18,.43,.18),(190,186,166)),
            part('cylinder',(0,.20,0),(.57,.1,.57),(178,177,158)),
            part('sphere',(0,.38,0),(.18,.23,.18),(126,177,181)),
        ]),
        asset('set-crate','Timber produce crate',[
            part('box',(0,0,0),(.96,.96,.96),(123,87,56)),
            part('box',(-.32,0,.486),(.12,.97,.02),(171,131,83),repeat=((3,1,1),(.32,0,0))),
            part('box',(0,-.33,.496),(.98,.06,.008),(91,71,51),repeat=((1,3,1),(0,.33,0))),
        ]),
        asset('set-rack','Warehouse storage rack',[
            part('box',(-.46,0,-.38),(.05,1,.05),(75,94,104),repeat=((2,1,2),(.92,0,.76))),
            part('box',(0,-.44,0),(1,.04,.85),(169,134,70),repeat=((1,4,1),(0,.29,0))),
            part('box',(-.28,-.31,0),(.2,.2,.61),(153,115,74),repeat=((3,3,1),(.28,.29,0))),
        ]),
        asset('set-bench','Industrial workbench',[
            part('box',(0,.40,0),(1,.16,1),(154,130,91)),
            part('box',(-.4,-.05,-.4),(.1,.9,.1),(72,93,101),repeat=((2,1,2),(.8,0,.8))),
            part('box',(0,-.20,0),(.8,.07,.8),(84,105,108)),
        ]),
        asset('set-machine','Industrial milling machine',[
            part('box',(0,-.38,0),(.9,.24,.9),(76,99,106)),
            part('box',(0,-.02,-.25),(.38,.72,.35),(80,131,137)),
            part('box',(0,.30,0),(.4,.26,.8),(92,148,147)),
            part('box',(0,-.02,.18),(.85,.08,.45),(174,181,173)),
            part('cylinder',(0,.10,.26),(.12,.18,.12),(67,76,80)),
        ]),
        asset('set-pendant','Industrial pendant',[
            part('cylinder',(0,.22,0),(.025,.54,.025),(59,66,66)),
            part('cone',(0,-.15,0),(.85,.4,.85),(193,170,108)),
            part('sphere',(0,-.34,0),(.2,.18,.2),(250,231,167)),
        ]),
    ]


class _Builder:
    def __init__(self, generated):
        self.lookup={a['id']:a for a in _defaults()+make_city()['assets']}
        self.generated=generated or []
        self.lookup.update({a['id']:a for a in self.generated})
        self.objects=[]

    def variants(self, pattern, fallback):
        return [a['id'] for a in self.generated if re.search(pattern,a['name'],re.I)] or [fallback]

    def role(self, pattern, fallback):
        return self.variants(pattern,fallback)[0]

    def place(self, identifier, x,z, size, *, y=None, yaw=0, name=None):
        self.objects.append(obj(f'set-{len(self.objects)}',name or self.lookup[identifier]['name'],identifier,
                                (x,size[1]/2 if y is None else y,z),size,yaw=yaw))

    def finish(self,name,camera,lighting='neutral'):
        refs={o['asset'] for o in self.objects}
        return validate_scene({'version':3,'name':name,'assets':[self.lookup[i] for i in sorted(refs)],
            'objects':self.objects,'effects':[],'lighting':lighting,'camera':camera})


def make_market(seed=0, generated=None):
    b=_Builder(generated)
    stalls=b.variants(r'stall|booth|vendor cart|kiosk','set-stall')
    fountain=b.role(r'fountain','set-fountain')
    crate=b.role(r'crate|basket','set-crate')
    tree=b.role(r'\btree\b|\bplanter\b','street-tree')
    bench=b.role(r'bench','city-bench')
    lamp=b.role(r'lantern|lamp|light','street-lamp')
    b.place('set-ground',0,-6,(26,.12,30),y=-.07)
    # Continuous enclosing architecture, with a clear entry and central promenade.
    for x in (-10,-6,-2,2,6,10):
        b.place('brownstone' if int(x)%3 else 'masonry',x,-18,(3.9,6+(int(x)+seed)%3,3.5))
    for side in (-1,1):
        for i,z in enumerate((-1,-6,-11)):
            stall=stalls[(i+(0 if side<0 else 3))%len(stalls)]
            b.place(stall,side*6.6,z,(3.7,2.65,2.7),yaw=90 if side<0 else -90)
            b.place(crate,side*4.95,z-1.7,(.6,.58,.6))
        for z in (2,-13):
            b.place(tree,side*9.7,z,(1.7,3.3,1.7))
            b.place(lamp,side*3.7,z,(.5,3.1,.5))
        b.place(bench,side*3.8,-9,(1.75,.9,.65),yaw=90 if side<0 else -90)
    b.place(fountain,0,-13,(3.2,2.2,3.2))
    pergola=next((a['id'] for a in b.generated if re.search(r'pergola|freestanding.*canopy',a['name'],re.I)),None)
    if pergola:b.place(pergola,-10.1,-7,(3.5,3.1,4))
    return b.finish('Market square',{'position':[3.8,5.3,13],'look_at':[0,1.5,-6]},'warm')


def make_workshop(seed=0, generated=None):
    b=_Builder(generated)
    rack=b.role(r'rack|shelv|shelf','set-rack')
    workbench=b.role(r'workbench|work bench|tool bench|table','set-bench')
    machine=b.role(r'machine|lathe|mill|drill|compressor','set-machine')
    crate=b.role(r'stacked|shipping|pallet','set-crate')
    lamp=b.role(r'pendant|light|lamp','set-pendant')
    b.place('set-ground',0,-5,(20,.12,24),y=-.07)
    b.place('set-wall',0,-16.7,(20,6,.2))
    for side in (-1,1):
        b.place('set-wall',side*9.9,-9,(.2,6,15.4))
        for z in (-2,-8,-14):
            b.place('set-beam',side*9.55,z,(.24,6,.24))
            b.place(rack,side*8,z,(3.8,3.7,1.35),yaw=90 if side<0 else -90)
            b.place(workbench,side*4.5,z,(2.4,1.0,1.2),yaw=90 if side<0 else -90)
        for z in (-5,-11):
            b.place(machine,side*5,z,(1.55,1.8,1.5),yaw=90 if side<0 else -90)
            b.place(crate,side*7.2,z,(.85,.85,.85))
        for z in (-1,-7,-13):
            b.place(lamp,side*3.8,z,(.8,1.1,.8),y=5.35)
    for pattern,x,z,size in [(r'gantry|crane|hoist',0,-13,(6.2,4.4,2.0)),
                              (r'rolling|utility cart',2.8,-8,(1.1,1.15,.75)),
                              (r'control cabinet',-4,-15.8,(1.0,1.8,.55))]:
        item=next((a['id'] for a in b.generated if re.search(pattern,a['name'],re.I)),None)
        if item:b.place(item,x,z,size)
    # Rear portal and roof trusses leave the camera-facing roof/front open.
    for z in (-3,-10,-16):
        b.place('set-beam',0,z,(19.2,.2,.2),y=5.9)
    return b.finish('Warehouse workshop',{'position':[5,6,13],'look_at':[0,1.6,-6]},'neutral')
