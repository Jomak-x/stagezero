"""Versioned mockup scene documents and deterministic, no-network set dressing."""
import copy
import math
import re
from scene_objects import KINDS, make_object, validate_objects
from scene_effects import make_effect, validate_effects

LIGHTING = ('neutral', 'warm', 'moonlight', 'neon', 'sunset')
PRESETS = ('Neon research lab', 'Enchanted grove', 'Cozy living room', 'Industrial yard', 'Winter plaza')
COLORS = {'red': [218, 70, 72], 'blue': [58, 120, 210], 'teal': [47, 177, 159],
          'green': [80, 150, 87], 'purple': [153, 88, 212], 'yellow': [232, 190, 70],
          'orange': [223, 129, 59], 'white': [220, 226, 231], 'black': [36, 43, 54]}


def validate_scene(value):
    if not isinstance(value, dict) or set(value) != {'version', 'name', 'objects', 'effects', 'lighting'}:
        raise ValueError('Scene requires version, name, objects, effects and lighting')
    if type(value['version']) is not int or value['version'] != 2:
        raise ValueError('Unsupported scene version')
    name = value['name']
    if not isinstance(name, str) or not name.strip() or len(name) > 80 or any(ord(c) < 32 for c in name):
        raise ValueError('Scene name must contain 1–80 printable characters')
    if not isinstance(value['lighting'], str) or value['lighting'] not in LIGHTING:
        raise ValueError('Unknown lighting preset')
    return {'version': 2, 'name': name.strip(), 'objects': validate_objects(value['objects']),
            'effects': validate_effects(value['effects']), 'lighting': value['lighting']}


def make_preset(name, seed=0):
    if name not in PRESETS:
        raise ValueError('Unknown scene preset')
    if type(seed) is not int or not 0 <= seed <= 1_000_000:
        raise ValueError('Seed must be an integer from 0 to 1000000')
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
    matches = [('Neon research lab', ('neon', 'sci-fi', 'scifi', 'laboratory', 'research lab', 'spaceship')),
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
