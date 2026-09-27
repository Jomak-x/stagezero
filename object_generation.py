"""Text to validated functional props. No renderer or model dependency required."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
from urllib.parse import urlparse

import requests
from scene_objects import make_object, validate_objects

MAX_RESPONSE_BYTES = 100_000
DEFAULT_SCENE_MODEL = 'gpt-5-6-sol'
DEFAULT_ASSET_MODEL = 'gpt-6-astra'


def validate_prompt(prompt):
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 2000:
        raise ValueError('Describe the objects in 1–2000 characters')
    return prompt.strip()


def generate_local(prompt):
    """Explicit offline recipe matching, not AI inference or arbitrary text understanding."""
    prompt = validate_prompt(prompt).lower()
    aliases = {'door': r'\b(doors?|gates?)\b', 'lamp': r'\b(lamps?|lights?)\b',
               'ball': r'\b(balls?)\b', 'chair': r'\b(chairs?|seats?)\b'}
    kinds = [kind for kind, pattern in aliases.items() if re.search(pattern, prompt)]
    if not kinds:
        raise ValueError('Offline recipes support door, lamp, ball and chair; name at least one')
    objects = []
    for i, kind in enumerate(kinds):
        obj = make_object(kind, i)
        obj['position'][0] = (i - (len(kinds) - 1) / 2) * 1.6
        obj['position'][2] = 1.5
        objects.append(obj)
    return validate_objects(objects)


class GatewayGenerator:
    """Adapter for a supplied OpenAI-compatible /chat/completions gateway.

    This protocol is configurable, not a claim about Neon's actual API.
    Credentials stay in the Python process, never in scene files or browser UI.
    """
    def __init__(self, base_url, model, api_key, transport=None):
        url = urlparse(base_url)
        if url.scheme != 'https' or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError('Gateway base URL must be HTTPS without credentials, query or fragment')
        if not model or not api_key:
            raise ValueError('Gateway model and API key are required')
        self.url = base_url.rstrip('/') + '/chat/completions'
        self.model, self._api_key = model, api_key
        self.transport = transport or requests

    @classmethod
    def from_env(cls, stage=None):
        config = gateway_config()
        base = config.get('STAGEZERO_OBJECT_API_BASE')
        use_neon_defaults = not base and bool(config.get('NEON_AI_GATEWAY_BASE_URL'))
        if not base and config.get('NEON_AI_GATEWAY_BASE_URL'):
            base = config['NEON_AI_GATEWAY_BASE_URL'].rstrip('/') + '/v1'
        model = config.get('STAGEZERO_OBJECT_MODEL')
        if stage == 'assets':
            model = config.get('STAGEZERO_SCENE_ASSET_MODEL') or (DEFAULT_ASSET_MODEL if use_neon_defaults else model)
        elif stage == 'layout':
            model = config.get('STAGEZERO_SCENE_LAYOUT_MODEL') or (DEFAULT_SCENE_MODEL if use_neon_defaults else model)
        elif stage is not None:
            raise ValueError('Unknown gateway generation stage')
        else:
            model = model or (DEFAULT_SCENE_MODEL if use_neon_defaults else None)
        token = config.get('STAGEZERO_OBJECT_API_KEY') or config.get('NEON_AI_GATEWAY_TOKEN')
        if not base or not model or not token:
            raise ValueError('Configure gateway URL, model and token in .runtime/objects.env or the environment')
        return cls(base, model, token)

    def generate(self, prompt):
        prompt = validate_prompt(prompt)
        examples = [make_object(kind, i) for i, kind in enumerate(('door', 'lamp', 'ball', 'chair'))]
        system = ('Return only a JSON object with a single objects array of 1 to 12 functional props. '
                  'Use exactly the fields and kind/action combinations in the examples. '
                  'No code, URLs or external assets. Coordinates are meters, +Y up; position is center. '
                  'Keep props on the floor (center y = size y / 2), except balls may be within hand reach. '
                  'Place props within 4 meters of the origin and avoid overlapping footprints. '
                  'Use unique simple IDs. Size components 0.05 to 3 meters, RGB integers 0 to 255, '
                  'Use touch radius 0.01 to 0.15 meters or proximity radius 0.1 to 1.5 meters. '
                  'Ball size must have equal components. A chair is a proximity cue; it cannot constrain motion. '
                  'Examples: ' + json.dumps(examples))
        doc = self.request_json(system, prompt)
        if not isinstance(doc, dict) or set(doc) != {'objects'} or not doc['objects']:
            raise ValueError('Expected a nonempty objects array')
        return validate_objects(doc['objects'])

    def request_json(self, system, prompt, max_tokens=3000, timeout_seconds=45):
        prompt = validate_prompt(prompt)
        try:
            with self.transport.post(self.url, headers={'Authorization': 'Bearer ' + self._api_key},
                                     json={'model': self.model, 'messages': [
                                         {'role': 'system', 'content': system},
                                         {'role': 'user', 'content': prompt}],
                                           'response_format': {'type': 'json_object'},
                                           'max_tokens': max_tokens}, timeout=(10, timeout_seconds),
                                     stream=True, allow_redirects=False) as response:
                if response.status_code != 200:
                    raise ValueError(f'Object gateway returned HTTP {response.status_code}; existing scene preserved')
                body = bytearray()
                for chunk in response.iter_content(8192):
                    body.extend(chunk)
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise ValueError('Object gateway response is too large')
            envelope = json.loads(body)
            doc = json.loads(envelope['choices'][0]['message']['content'])
            return doc
        except requests.RequestException:
            raise ValueError('Object gateway connection failed; check configuration and retry') from None
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            raise ValueError('Object gateway returned an invalid JSON scene') from None


def gateway_config():
    """Read only recognized values from the private file, never execute shell code.

    Process environment takes precedence. AWS credentials are never loaded here.
    """
    names = ('STAGEZERO_OBJECT_API_BASE', 'STAGEZERO_OBJECT_MODEL', 'STAGEZERO_OBJECT_API_KEY',
             'NEON_AI_GATEWAY_BASE_URL', 'NEON_AI_GATEWAY_TOKEN', 'STAGEZERO_LOCAL_MODEL',
             'STAGEZERO_SCENE_ASSET_MODEL', 'STAGEZERO_SCENE_LAYOUT_MODEL', 'STAGEZERO_STORY_MODEL',
             'STAGEZERO_CHARACTER_IMAGE_MODEL', 'STAGEZERO_CHARACTER_DESIGN_MODEL')
    values = {}
    path = Path(__file__).resolve().parent / '.runtime' / 'objects.env'
    if path.is_file():
        if path.stat().st_size > 16_384:
            raise ValueError('Object configuration file is too large')
        try:
            for line in path.read_text().splitlines():
                parts = shlex.split(line, comments=True)
                if parts and parts[0] == 'export':
                    parts = parts[1:]
                if len(parts) == 1 and '=' in parts[0]:
                    key, value = parts[0].split('=', 1)
                    if key in names:
                        values[key] = value
        except ValueError:
            raise ValueError('Invalid quoting in .runtime/objects.env') from None
    values.update({key: os.environ[key] for key in names if key in os.environ})
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt')
    parser.add_argument('--gateway', action='store_true', help='Use configured AI gateway instead of offline recipes')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        objects = GatewayGenerator.from_env().generate(args.prompt) if args.gateway else generate_local(args.prompt)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'version': 1, 'objects': objects}, indent=2) + '\n')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')
    print(f'Created {len(objects)} functional props: {args.output}')


if __name__ == '__main__':
    main()
