"""Small, deterministic grammar for studio voice directions.

The parser only returns native session operations. Spoken text is never code.
"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class VoiceRoute:
    target: str
    prompt: str
    take_id: str = ''
    edit_mode: str = ''
    at_frame: int | None = None


def _normal(text):
    return ' '.join(text.casefold().split())


_SPOKEN_NUMBERS = dict(zip(
    'one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen'.split(),
    range(1, 17)))
_ACTION_NUMBER = r'(\d+|' + '|'.join(_SPOKEN_NUMBERS) + ')'


def route_voice_command(text, target, takes):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Enter a direction to generate')
    text = text.strip()
    if target in ('full_scene', 'single_action'):
        prompt = text
        route = VoiceRoute(target, prompt)
    elif target == 'auto':
        scene = re.fullmatch(r'generate\s+(?:a\s+)?(?:full\s+)?scene\s*[.:;,—-]?\s+(.+)', text, re.I | re.S)
        short = re.fullmatch(r'generate\s+(?:a\s+)?short\s*(?:action)?\s*[.:;,—-]?\s+(.+)', text, re.I | re.S)
        if scene:
            route = VoiceRoute('full_scene', scene.group(1).strip())
        elif short:
            route = VoiceRoute('single_action', short.group(1).strip())
        elif text.casefold().startswith('edit '):
            route = _edit_route(text, takes)
        else:
            raise ValueError('Start with “generate scene”, “generate a short”, or “edit [name] to…”')
    else:
        raise ValueError('Choose Auto, Single action, or Full scene')
    limit = 2000 if route.target == 'full_scene' else 500
    if not 1 <= len(route.prompt) <= limit:
        raise ValueError(f'Enter a direction between 1 and {limit} characters')
    return route


def _edit_route(text, takes):
    body = re.sub(r'^edit\s+', '', text, count=1, flags=re.I)
    splits = list(re.finditer(r'\s+(?:to|into|as|so\s+that)\s+|\s*:\s*', body, re.I))
    if not splits:
        raise ValueError('Say “edit [take or action name] to [new motion]”')
    matches, failures = [], []
    for split in splits:
        name, prompt = body[:split.start()].strip(), body[split.end():].strip()
        try:
            matches.append(_resolve_edit(name, prompt, takes))
        except ValueError as exc:
            failures.append(exc)
    if len(matches) > 1:
        raise ValueError('Edit direction has multiple possible targets; quote the saved name')
    if matches:
        return matches[0]
    raise failures[0]


def _resolve_edit(name, prompt, takes):
    name = name.strip().strip('"\'“”')
    if not name or not prompt:
        raise ValueError('Name a saved take or action and describe its replacement')

    scene = re.fullmatch(r'scene\s+(.+)', name, re.I | re.S)
    if scene:
        take_name = scene.group(1).strip().strip('"\'“”')
        matches = [take for take in takes.values() if _normal(take.name) == _normal(take_name)]
        if len(matches) != 1:
            raise ValueError('Scene name is missing or ambiguous; use an exact saved take name')
        return VoiceRoute('full_scene', prompt, matches[0].id, 'scene')

    action = re.fullmatch(r'action\s+' + _ACTION_NUMBER + r'\s+(?:in|of)\s+(.+)', name, re.I)
    reverse = re.fullmatch(r'(.+?)\s+action\s+' + _ACTION_NUMBER, name, re.I)
    if action or reverse:
        spoken = (action or reverse).group(1 if action else 2).casefold()
        number = _SPOKEN_NUMBERS[spoken] if spoken in _SPOKEN_NUMBERS else int(spoken)
        take_name = (action or reverse).group(2 if action else 1).strip().strip('"\'“”')
        matches = [take for take in takes.values() if _normal(take.name) == _normal(take_name)]
        if len(matches) != 1:
            raise ValueError('Take name is missing or ambiguous; use an exact saved take name')
        take = matches[0]
        if not 1 <= number <= len(take.segments):
            raise ValueError('Action number is not in that take')
        return VoiceRoute('single_action', prompt, take.id, 'action', take.segments[number - 1]['start'])

    take_matches = [take for take in takes.values() if _normal(take.name) == _normal(name)]
    action_matches = [(take, segment) for take in takes.values() for segment in take.segments
                      if _normal(str(segment.get('prompt', ''))) == _normal(name)]
    if len(take_matches) + len(action_matches) > 1:
        raise ValueError('Name matches multiple takes or actions; use “action N in [take name]”')
    if take_matches:
        return VoiceRoute('single_action', prompt, take_matches[0].id, 'replace', 0)
    if action_matches:
        take, segment = action_matches[0]
        return VoiceRoute('single_action', prompt, take.id, 'action', segment['start'])
    raise ValueError('No saved take or action has that name')
