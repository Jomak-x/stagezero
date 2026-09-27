"""Generate complete procedural scenes using recipes, Neon or local Ollama."""
import argparse
import json
import os
from pathlib import Path
import requests
from scene_objects import KINDS, make_object
from scene_effects import EFFECT_KINDS, make_effect
from scene_composition import validate_scene, generate_recipe, LIGHTING
from object_generation import GatewayGenerator, gateway_config, validate_prompt, MAX_RESPONSE_BYTES


def validate_generated_scene(doc):
    scene = validate_scene(doc)
    if not scene['objects'] and not scene['effects']:
        raise ValueError('Generated scene is empty; include at least one prop or effect')
    return scene


def scene_system_prompt():
    examples = [make_object(kind, i) for i, kind in enumerate(KINDS)]
    effects = [make_effect(kind, i) for i, kind in enumerate(('rain', 'snow', 'fireflies', 'sparks', 'smoke', 'portal'))]
    return ('Design an attractive stylized 3D mockup scene for a 1.3-meter humanoid actor at the origin. '
            'Return JSON only: {"version":2,"name":"short title","objects":[...],"effects":[...],"lighting":"neon"}. '
            'Use 6–18 props and 0–3 effects. Catalog examples define exact keys, supported kinds and matching actions. '
            'No code, URLs or arbitrary assets. Coordinates are meters, Y up; size is width,height,depth and position is CENTER. '
            'The camera is on +Z looking toward origin. Keep a clear 2-meter acting area around origin; put tall scenery behind at negative Z and beside it. '
            'Place bottoms at y=0 except platforms may have tops at y=0 and balls may be at hand height. '
            'Object coordinates within +/-20, dimensions .05–12, RGB integer bytes. Ball dimensions equal and touch radius .04. '
            'Static objects must have action none, trigger none, radius 0. Interactive types use the example action and trigger. '
            'Door/lamp/console proximity radius .4–1.2, touch radii <=.15. Use unique safe IDs, coherent palette, '
            'good scale, negative space, layered composition, deliberate focal point. Avoid overlapping solid props. '
            'Effects are decorative deterministic particles, not physics. Use effect schema exactly as shown, bounded '
            'intensity and integer seeds. Lighting must be one of '+json.dumps(LIGHTING)+'. '
            'OBJECT CATALOG: '+json.dumps(examples)+'. EFFECT CATALOG: '+json.dumps(effects))


class SceneGenerator:
    def __init__(self, gateway=None):
        self.gateway = gateway or GatewayGenerator.from_env()

    def generate(self, prompt):
        prompt = validate_prompt(prompt)
        system = scene_system_prompt()
        doc = self.gateway.request_json(system, prompt, max_tokens=8000)
        try:
            return validate_generated_scene(doc)
        except ValueError as exc:
            # One bounded schema-repair attempt; never retry transport failures.
            repair = system + ' Your previous response failed validation: ' + str(exc) + '. Use ONLY catalog kinds and exact field shapes. Return a corrected complete scene.'
            return validate_generated_scene(self.gateway.request_json(repair, prompt, max_tokens=8000))


class LocalSceneGenerator:
    """Optional installed Ollama service; never installs or downloads a model."""
    def __init__(self, model=None, transport=None):
        self.model = model or os.environ.get('STAGEZERO_LOCAL_MODEL', 'qwen3:4b')
        self.transport = transport or requests

    def generate(self, prompt):
        prompt = validate_prompt(prompt)
        position = {'type': 'array', 'items': {'type': 'number', 'minimum': -20, 'maximum': 20},
                    'minItems': 3, 'maxItems': 3}
        color = {'type': 'array', 'items': {'type': 'integer', 'minimum': 0, 'maximum': 255},
                 'minItems': 3, 'maxItems': 3}
        def item_schema(kinds, size_minimum, size_maximum, *, effect=False):
            properties = {'kind': {'type': 'string', 'enum': list(kinds)}, 'position': position,
                          'size': {'type': 'array', 'items': {'type': 'number', 'minimum': size_minimum,
                                                             'maximum': size_maximum}, 'minItems': 3, 'maxItems': 3},
                          'color': color}
            if effect:
                properties['intensity'] = {'type': 'number', 'minimum': 0, 'maximum': 1}
            return {'type': 'object', 'properties': properties, 'required': list(properties),
                    'additionalProperties': False}
        schema = {'type': 'object', 'properties': {
            'name': {'type': 'string', 'minLength': 1, 'maxLength': 80},
            'lighting': {'type': 'string', 'enum': list(LIGHTING)},
            'objects': {'type': 'array', 'items': item_schema(KINDS, .05, 12),
                        'minItems': 1, 'maxItems': 18},
            'effects': {'type': 'array', 'items': item_schema(EFFECT_KINDS, .1, 8, effect=True),
                        'maxItems': 3}},
            'required': ['name', 'lighting', 'objects', 'effects'], 'additionalProperties': False}
        catalog = ', '.join(f'{kind} {values["size"]}' for kind, values in KINDS.items())
        effect_catalog = ', '.join(f'{kind} {values["size"]}' for kind, values in EFFECT_KINDS.items())
        system = ('Plan one coherent stylized scene for a 1.3-meter actor at the origin. '
                  'Choose 6–10 props relevant to the request and 0–3 effects; never list the entire catalog. '
                  'Catalog kinds and typical sizes (width,height,depth): ' + catalog + '. '
                  'Effect kinds and typical sizes: ' + effect_catalog + '. '
                  'Coordinates are meters, Y up; position is the center. Keep a clear 1-meter radius around the actor. '
                  'Put tall backdrops at negative Z, smaller props to the sides around X=±3, and vary positions. '
                  'Place object bottoms at Y=0 except platforms may have tops at Y=0 and balls may be at hand height. '
                  'Colors are RGB integer bytes from 0 to 255, not normalized values: teal [40,180,180], '
                  'magenta [200,50,170], dark gray [35,40,50]. '
                  'Choose a coherent palette and focal point. Return only the compact JSON plan in the required schema.')
        try:
            with self.transport.post('http://127.0.0.1:11434/api/chat', json={
                'model': self.model, 'stream': False, 'format': schema, 'think': False,
                'messages': [{'role': 'system', 'content': system},
                             {'role': 'user', 'content': prompt}],
                'options': {'temperature': .4, 'num_predict': 3500}},
                timeout=(3, 300), stream=True, allow_redirects=False) as response:
                if response.status_code != 200:
                    raise ValueError('Local model unavailable; start Ollama and install the selected model')
                body = bytearray()
                for chunk in response.iter_content(8192):
                    body.extend(chunk)
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise ValueError('Local model response is too large')
            plan = json.loads(json.loads(body)['message']['content'])
            return _expand_local_scene_plan(plan)
        except requests.ReadTimeout:
            raise ValueError('Local model took too long to generate a scene; try again after it has loaded') from None
        except requests.RequestException:
            raise ValueError('Cannot reach local Ollama; start it or choose Recipes / AI gateway') from None
        except (KeyError, TypeError, json.JSONDecodeError):
            raise ValueError('Local model returned invalid scene JSON; current scene preserved') from None


def _expand_local_scene_plan(plan):
    if not isinstance(plan, dict) or set(plan) != {'name', 'lighting', 'objects', 'effects'}:
        raise ValueError('Local model returned an invalid scene plan')
    if not isinstance(plan['objects'], list) or not 1 <= len(plan['objects']) <= 18:
        raise ValueError('Local model must choose 1–18 props')
    if not isinstance(plan['effects'], list) or len(plan['effects']) > 3:
        raise ValueError('Local model must choose at most three effects')
    objects = []
    for index, item in enumerate(plan['objects']):
        if not isinstance(item, dict) or set(item) != {'kind', 'position', 'size', 'color'}:
            raise ValueError(f'Local model prop {index} has missing or unknown fields')
        obj = make_object(item['kind'], index)
        obj.update({key: item[key] for key in ('position', 'size', 'color')})
        objects.append(obj)
    effects = []
    for index, item in enumerate(plan['effects']):
        if not isinstance(item, dict) or set(item) != {'kind', 'position', 'size', 'color', 'intensity'}:
            raise ValueError(f'Local model effect {index} has missing or unknown fields')
        effect = make_effect(item['kind'], index)
        effect.update({key: item[key] for key in ('position', 'size', 'color', 'intensity')})
        effects.append(effect)
    return validate_generated_scene({'version': 2, 'name': plan['name'], 'lighting': plan['lighting'],
                                     'objects': objects, 'effects': effects})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt')
    parser.add_argument('--source', choices=('recipe', 'gateway', 'local'), default='recipe')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.source == 'recipe':
            scene = generate_recipe(validate_prompt(args.prompt), args.seed)
        else:
            from adaptive_scene_generation import AdaptiveSceneGenerator
            scene = (AdaptiveSceneGenerator(progress=print) if args.source == 'gateway' else LocalSceneGenerator()).generate(args.prompt)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(scene, indent=2)+'\n')
        print(f"Saved {scene['name']}: {len(scene['objects'])} props, {len(scene['effects'])} effects to {args.output}")
    except (ValueError, OSError) as exc:
        parser.exit(1, str(exc)+'\n')


if __name__ == '__main__':
    main()
