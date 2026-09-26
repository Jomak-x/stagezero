"""Architectural staging with inferred roles for freshly generated scene props."""
import copy
import re
from scene_environments import make_city, make_room
from scene_composition import validate_scene


def environment_kind(prompt):
    text = prompt.lower()
    if re.search(r'\b(market|marketplace|bazaar)\b', text): return 'market'
    if re.search(r'\b(warehouse|workshop|factory)\b', text): return 'workshop'
    if re.search(r'\b(residential|neighbou?rhood|suburb|houses|housing)\b', text): return 'residential'
    if re.search(r'\b(city|urban|downtown|skyline|street|boulevard|town)\b', text): return 'city'
    if re.search(r'\b(room|apartment|interior|bedroom|office|studio|lounge|kitchen|workshop)\b', text): return 'room'
    return None


def compose_architectural_scene(prompt, assets):
    kind = environment_kind(prompt)
    if kind is None: return None
    if kind in ('market','workshop'):
        from scene_sets import make_market, make_workshop
        return (make_market if kind == 'market' else make_workshop)(generated=assets)
    if kind == 'residential':
        from scene_environments import make_residential
        scene = make_residential()
    else:
        scene = make_city() if kind == 'city' else make_room()
    originals = {a['id']: a for a in scene['assets']}
    used = {}
    def replace(obj, asset, size=None):
        obj['asset'], obj['name'] = asset['id'], asset['name']
        if size:
            obj['size'] = list(size)
            obj['position'][1] = size[1]/2
        used[asset['id']] = copy.deepcopy(asset)
    if kind in ('city','residential'):
        stores = [a for a in assets if re.search(r'store|shop|cafe|kiosk',a['name'],re.I)]
        buildings = [a for a in assets if a not in stores and re.search(r'building|tower|block|facade|skyscraper|highrise|high-rise|skyline|office|hotel|townhouse|house|cottage|bungalow',a['name'],re.I)]
        index = 0
        # A full generated building must not be stacked above another full building.
        # The starter's cafe+office stack is only for its authored modular assets.
        if buildings:
            scene['objects'] = [o for o in scene['objects'] if o['name'] != 'Right upper stories']
        for obj in scene['objects']:
            if obj['asset'] in ('masonry','office','brownstone','house-clapboard','house-brick') and buildings:
                replace(obj,buildings[index % len(buildings)]); index += 1
            elif obj['asset'] == 'storefront' and stores:
                replace(obj,stores[0])
            elif obj['asset'] in ('street-lamp','street-tree','city-bench','mailbox'):
                pattern={'street-lamp':r'light|lamp','street-tree':r'\btree\b|\bplanter\b','city-bench':r'bench','mailbox':r'mailbox'}[obj['asset']]
                candidate=next((a for a in assets if re.search(pattern,a['name'],re.I) and a not in buildings and a not in stores),None)
                if candidate:replace(obj,candidate)
        if kind == 'residential':
            from scene_environments import _object
            # A corner cafe and benches make a neighborhood more than a row
            # of houses. Reserve these sites outside the road and house plots.
            extras=[]
            if stores:extras.append((stores[0],-8.3,3,(3.25,2.7,2.9),90))
            bench=next((a for a in assets if re.search(r'\bbench\b',a['name'],re.I)),None)
            if bench:
                extras.extend((bench,side*5.2,1.0,(1.4,.9,.6),90 if side<0 else -90) for side in (-1,1))
            for a,x,z,size,yaw in extras:
                scene['objects'].append(_object(f'residential-extra-{len(scene["objects"])}',a['name'],a['id'],(x,size[1]/2,z),size,yaw=yaw))
                used[a['id']]=copy.deepcopy(a)
    else:
        # Match distinct roles, keeping camera, architecture and clear actor area deterministic.
        roles = [
            ('bookshelf',r'bookshelf|shelving|bookcase|shelf',None),
            ('sofa',r'sofa|couch|settee',None),
            ('coffee-table',r'coffee table|low table',None),
            ('coffee-table',r'desk|workbench|drafting|drawing table|work table',(1.5,.78,.75)),
            ('floor-lamp',r'lamp|light|pendant',None),
            ('indoor-plant',r'plant',None),
            ('wall-art',r'picture|artwork|painting',None),
        ]
        if re.search(r'kitchen',prompt,re.I):
            scene['objects'] = [o for o in scene['objects'] if o['asset'] not in ('sofa','coffee-table')]
            roles.insert(0,('bookshelf',r'fridge|refrigerator|tall cabinet',(.85,1.95,.72)))
        if re.search(r'bedroom',prompt,re.I):
            roles.insert(0,('sofa',r'bed(?!side)',(2.25,.8,1.55)))
        assigned, occupied = set(), set()
        for slot,pattern,size in roles:
            if slot in occupied: continue
            candidates=[a for a in assets if a['id'] not in assigned and re.search(pattern,a['name'],re.I)
                        and not re.search(r'whole room|room shell|whole interior',a['name'],re.I)]
            obj=next((o for o in scene['objects'] if o['asset']==slot),None)
            if candidates and obj:
                replace(obj,candidates[0],size)
                if slot=='floor-lamp' and re.search(r'pendant|ceiling',candidates[0]['name'],re.I):
                    obj['size']=[.6,.65,.6];obj['position']=[1.55,2.65,-1.65]
                assigned.add(candidates[0]['id']); occupied.add(slot)
        # Additional recognizable furniture can be placed without inventing more generator settings.
        extras=[(r'counter|island',(2.35,.9,.8),(0,-3.15)),
                (r'oven|stove',(.75,.9,.7),(-2.5,-3.15)),
                (r'chair|stool',(.65,.9,.65),(1.55,-.45))]
        for pattern,size,(x,z) in extras:
            candidate=next((a for a in assets if a['id'] not in assigned and re.search(pattern,a['name'],re.I)),None)
            if candidate is None: continue
            obj={'id':f'room-extra-{len(assigned)}','name':candidate['name'],'kind':'custom',
                 'asset':candidate['id'],'position':[x,size[1]/2,z],'size':list(size),
                 'color':[255,255,255],'interaction':{'action':'none','trigger':'none','radius':0}}
            furniture=[o for o in scene['objects'] if o['asset'] not in ('floorboards','rear-wall','side-wall','woven-rug','wall-art')]
            if any(abs(o['position'][0]-x)<(o['size'][0]+size[0])/2+.08 and abs(o['position'][2]-z)<(o['size'][2]+size[2])/2+.08 for o in furniture): continue
            if re.search(r'chair|stool',candidate['name'],re.I):obj['yaw']=180
            scene['objects'].append(obj);used[candidate['id']]=copy.deepcopy(candidate);assigned.add(candidate['id'])
    if not used:return None
    referenced={o['asset'] for o in scene['objects']}
    scene['assets']=[a for identifier,a in originals.items() if identifier in referenced]+list(used.values())
    scene['name']={'city':'Generated city blocks','residential':'Generated neighborhood','room':'Generated interior set'}[kind]
    return validate_scene(scene)
