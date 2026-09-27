"""Versioned mockup scene documents and deterministic, no-network set dressing."""
import copy
import math
import re
from scene_objects import KINDS, make_object, validate_objects
from scene_effects import make_effect, validate_effects

MAX_SCENE_TRIANGLES = 250_000
MAX_SCENE_FILE_BYTES = 1_000_000

LIGHTING = ('neutral', 'warm', 'moonlight', 'neon', 'sunset')
PRESETS = ('Rooftop swing district', 'Harbor chase', 'Jungle temple', 'City boulevard', 'Residential neighborhood', 'Market square', 'Warehouse workshop', 'Designed apartment', 'Neon research lab', 'Enchanted grove', 'Cozy living room', 'Industrial yard', 'Winter plaza', 'Traversable temple', 'Industrial switchback')
COLORS = {'red': [218, 70, 72], 'blue': [58, 120, 210], 'teal': [47, 177, 159],
          'green': [80, 150, 87], 'purple': [153, 88, 212], 'yellow': [232, 190, 70],
          'orange': [223, 129, 59], 'white': [220, 226, 231], 'black': [36, 43, 54]}


def validate_scene(value):
    if not isinstance(value, dict):
        raise ValueError('Scene must be an object')
    version = value.get('version')
    required = {'version', 'name', 'objects', 'effects', 'lighting'}
    optional = set()
    if type(version) is not int or version not in (2, 3):
        raise ValueError('Unsupported scene version')
    if version == 3:
        required.add('assets')
        optional.update(('camera','targets'))
    if not required <= set(value) or set(value) - required - optional:
        raise ValueError('Scene has missing or unsupported fields')
    name = value['name']
    if not isinstance(name, str) or not name.strip() or len(name) > 80 or any(ord(c) < 32 for c in name):
        raise ValueError('Scene name must contain 1–80 printable characters')
    if not isinstance(value['lighting'], str) or value['lighting'] not in LIGHTING:
        raise ValueError('Unknown lighting preset')
    result = {'version': version, 'name': name.strip(), 'objects': validate_objects(value['objects']),
              'effects': validate_effects(value['effects']), 'lighting': value['lighting']}
    from asset_geometry import validate_assets, expanded_count, triangle_count
    assets = validate_assets(value.get('assets', []))
    lookup = {a['id']: a for a in assets}
    budget = 0
    triangles = 0
    for obj in result['objects']:
        if obj['kind'] == 'custom':
            if obj['asset'] not in lookup:
                raise ValueError('Custom prop references a missing asset')
            budget += expanded_count(lookup[obj['asset']])
            triangles += triangle_count(lookup[obj['asset']])
    if budget > 12000:
        raise ValueError('Scene exceeds 12000 generated shape instances')
    if triangles > MAX_SCENE_TRIANGLES:
        raise ValueError('Scene exceeds 250000 generated triangles; use fewer curved details or instances')
    if version == 3:
        result['assets'] = assets
    if 'camera' in value:
        camera = value['camera']
        if not isinstance(camera, dict) or set(camera) != {'position', 'look_at'}:
            raise ValueError('Camera requires position and look_at')
        clean = {}
        for key in ('position', 'look_at'):
            v = camera[key]
            if not isinstance(v, (list, tuple)) or len(v) != 3 or any(type(x) not in (float, int) or not math.isfinite(x) or abs(x) > 200 for x in v):
                raise ValueError('Invalid scene camera coordinates')
            clean[key] = [float(x) for x in v]
        if sum((a-b)**2 for a,b in zip(clean['position'], clean['look_at'])) < .01:
            raise ValueError('Scene camera must be separated from target')
        result['camera'] = clean
    if 'targets' in value:
        from scene_targets import validate_targets
        result['targets']=validate_targets(value['targets'],result['objects'])
    return result


def encode_scene(value):
    """Portable compact export using the same document budget as imports."""
    import json
    payload=(json.dumps(validate_scene(value),separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')
    if len(payload)>MAX_SCENE_FILE_BYTES:
        raise ValueError('Scene file exceeds 1 MB')
    return payload


def make_preset(name, seed=0):
    if name not in PRESETS:
        raise ValueError('Unknown scene preset')
    if type(seed) is not int or not 0 <= seed <= 1_000_000:
        raise ValueError('Seed must be an integer from 0 to 1000000')
    if name == 'Traversable temple':
        from traversal_kit import traversable_temple_scene
        return validate_scene(traversable_temple_scene())
    if name == 'Industrial switchback':
        from switchback_traversal import industrial_switchback_scene
        return validate_scene(industrial_switchback_scene())
    if name in ('Rooftop swing district','Harbor chase','Jungle temple'):
        from cinematic_scenes import make_cinematic
        return make_cinematic(name,seed)
    if name == 'Residential neighborhood':
        from scene_environments import make_residential
        return validate_scene(make_residential(seed))
    if name in ('Market square','Warehouse workshop'):
        from scene_sets import make_market,make_workshop
        return (make_market if name == 'Market square' else make_workshop)(seed)
    if name in ('City boulevard', 'Designed apartment'):
        from scene_environments import make_city, make_room
        return validate_scene((make_city if name == 'City boulevard' else make_room)(seed))
    objects, effects = [], []

    def prop(kind, x, z, size=None, color=None, y=None):
        obj = make_object(kind, len(objects))
        if size: obj['size'] = list(size)
        obj['position'] = [x, obj['size'][1] / 2 if y is None else y, z]
        if color: obj['color'] = list(color)
        objects.append(obj)

    def effect(kind, position, size=None, color=None):
        fx = make_effect(kind, len(effects), position=position)
        fx['seed'] = seed + len(effects)
        if size: fx['size'] = list(size)
        if color: fx['color'] = list(color)
        effects.append(fx)

    if name == 'Neon research lab':
        lighting = 'neon'
        prop('platform', 0, 0, (7, .12, 6), (37, 49, 70), y=-.06)
        prop('arch', 0, 2.6, (2.7, 3.2, .5), (60, 80, 115))
        for x in (-2.5, 2.5):
            prop('pillar', x, 2.3, (.5, 2.9, .5), (64, 91, 125))
            prop('console', x, .8, (1.1, 1.15, .7), (49, 75, 105))
            prop('crate', x, -1.8, (.65, .65, .65), (84, 105, 128))
        prop('ball', .55, .2, (.23, .23, .23), (84, 225, 247), y=.85)
        effect('portal', (0, 1.6, 2.62), (2, 2.8, .5), (66, 218, 255))
        effect('sparks', (2.5, 1.5, .8), (.8, 1.2, .8), (92, 211, 255))
    elif name == 'Enchanted grove':
        lighting = 'moonlight'
        for x, z in [(-3, -1), (-2.8, 2.4), (2.8, 2.4), (3, -1), (0, 3.8)]:
            prop('tree', x, z, (1.8, 3.5, 1.8), (52, 110, 93))
        for x, z in [(-1.8, .4), (1.9, -.6), (-1, 2.7), (1.5, 2.9)]:
            prop('rock', x, z, (.7, .5, .6), (100, 115, 127))
            prop('plant', x + .5, z - .5, (.6, .8, .6), (70, 165, 124))
        prop('arch', 0, 2.8, (2.4, 2.8, .6), (107, 113, 130))
        effect('fireflies', (0, 1.6, .6), (6, 3, 5), (160, 255, 150))
        effect('portal', (0, 1.4, 2.8), (1.7, 2.4, .4), (174, 116, 255))
    elif name == 'Cozy living room':
        lighting = 'warm'
        prop('platform', 0, 0, (7, .1, 6), (116, 87, 66), y=-.05)
        prop('wall', 0, 3, (7, 2.8, .15), (151, 135, 120))
        prop('sofa', -1.7, 1.5, (2.4, 1, 1), (85, 135, 140))
        prop('table', 0, 1.2, (1.1, .55, .8), (147, 100, 61))
        prop('chair', 1.8, 1.4, (.8, 1, .8), (205, 151, 85))
        prop('lamp', -3, 1.8, (.5, 1.8, .5), (238, 196, 110))
        prop('plant', 2.6, 2.3, (.8, 1.5, .8), (65, 142, 93))
        prop('ball', .6, -.5, (.3, .3, .3), (219, 106, 89))
    elif name == 'Industrial yard':
        lighting = 'sunset'
        prop('wall', 0, 3, (8, 3, .25), (96, 103, 107))
        prop('door', 0, 2.7, (1.6, 2.5, .12), (179, 117, 59))
        for i, (x, z) in enumerate([(-2.4, 1.2), (-3.2, 2), (2.4, 2)]):
            prop('crate', x, z, (1, 1, 1), (131, 101, 66))
            prop('barrel', x + .5, z - 1.2, (.6, .95, .6), (75, 109, 126))
        prop('console', 1.5, 2.5, (.7, 1.2, .55), (76, 82, 89))
        effect('sparks', (2.5, 1.1, 1.8), (1.2, 2, 1.2), (255, 180, 70))
        effect('smoke', (-2.8, 1.5, 2), (1.3, 3, 1.3), (116, 127, 141))
        effect('rain', (0, 3, 0), (8, 6, 7), (137, 173, 210))
    else:
        lighting = 'moonlight'
        prop('platform', 0, 0, (8, .1, 7), (185, 206, 220), y=-.05)
        for x in (-2.8, 2.8):
            prop('tree', x, 2, (1.7, 3.2, 1.7), (107, 153, 151))
            prop('lamp', x, -.5, (.4, 2.2, .4), (235, 191, 114))
            prop('chair', x, -1.8, (.7, .9, .7), (118, 98, 86))
        prop('arch', 0, 3, (2.7, 3, .6), (180, 203, 218))
        effect('snow', (0, 3, 0), (8, 6, 7), (231, 242, 255))
    # The studio starts on the +Z side of the actor. Put backdrops behind it.
    for item in objects + effects:
        item['position'][2] *= -1
    return validate_scene({'version': 2, 'name': name, 'objects': objects, 'effects': effects, 'lighting': lighting})


def generate_recipe(prompt, seed=0):
    """Transparent keyword recipes plus counted catalog props; not language-model inference."""
    text = prompt.lower()
    matches = [('Traversable temple', ('traversable temple',)),
               ('Industrial switchback', ('industrial switchback',)),
               ('Rooftop swing district', ('spider-man','spider man','swing district','rooftop city','swinging city')),
               ('Harbor chase', ('dockyard','harbor chase','container port')),
               ('Jungle temple', ('jungle temple','temple ruin','ancient temple')),
               ('Residential neighborhood', ('residential','neighborhood','neighbourhood','suburb','houses')),
               ('Market square', ('market','bazaar')),
               ('Warehouse workshop', ('warehouse','workshop','factory')),('City boulevard', ('city', 'urban', 'downtown', 'skyline', 'street', 'boulevard', 'town')),
               ('Designed apartment', ('room', 'apartment', 'interior', 'bedroom', 'living room', 'lounge')),
               ('Neon research lab', ('neon', 'sci-fi', 'scifi', 'laboratory', 'research lab', 'spaceship')),
               ('Enchanted grove', ('forest', 'grove', 'enchanted', 'fantasy')),
               ('Cozy living room', ('cozy', 'living room', 'lounge')),
               ('Industrial yard', ('industrial', 'warehouse', 'factory')),
               ('Winter plaza', ('winter', 'snowy', 'plaza'))]
    preset = next((name for name, words in matches if any(word in text for word in words)), None)
    if preset:
        scene = make_preset(preset, seed)
    else:
        objects = []
        numbers = {'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6}
        pattern = r'(?:(\d+|a|an|one|two|three|four|five|six)\s+)?(?:(red|blue|teal|green|purple|yellow|orange|white|black)\s+)?(' + '|'.join(sorted(KINDS, key=len, reverse=True)) + r')(?:s)?\b'
        for match in re.finditer(r'\b' + pattern, text):
            count_text, color, kind = match.groups()
            count = numbers.get(count_text, int(count_text) if count_text and count_text.isdigit() else 1)
            if count > 12 or len(objects) + count > 40:
                raise ValueError('Recipe request exceeds 12 copies per kind or 40 total props')
            for _ in range(count):
                obj = make_object(kind, len(objects))
                i = len(objects)
                obj['position'][0] = (i % 5 - 2) * 1.6
                obj['position'][2] = 1.8 + (i // 5) * 1.6
                if color: obj['color'] = COLORS[color].copy()
                objects.append(obj)
        scene = {'version': 2, 'name': 'Custom recipe', 'objects': objects, 'effects': [], 'lighting': 'neutral'}
    aliases = {'rain': r'\brain\b|\brainy\b', 'snow': r'\bsnow\b|\bsnowing\b',
               'fireflies': r'\bfireflies\b', 'sparks': r'\bsparks\b', 'smoke': r'\bsmoke\b', 'portal': r'\bportal\b'}
    present = {fx['kind'] for fx in scene['effects']}
    for kind, pattern in aliases.items():
        if re.search(pattern, text) and kind not in present:
            fx = make_effect(kind, len(scene['effects']))
            fx['seed'] = seed + len(scene['effects'])
            scene['effects'].append(fx)
    if not scene['objects'] and not scene['effects']:
        raise ValueError('Name a preset, supported prop or effect; try “enchanted forest with fireflies”')
    return validate_scene(scene)
