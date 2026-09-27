"""Fresh user prompt -> production planner -> real workers -> saved cast evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cast_performance import encode_project, cast_from_performance
from native_pair_provider import NativePairProvider
from prompt_scene_builder import PromptSceneBuilder
from prompt_scene_plan import ScenePromptPlanner
from realtime_client import RealtimeClient

CASES = [
    ('celebrate-city-91', 'Three people celebrate happily together at the same time for six seconds, each raising both arms and cheering in place. They start at (0,-5), (0,0), and (0,5). No touching.', 'city', 91),
    ('wave-market-92', 'Three people wave hello at the same time for six seconds, standing apart at (-3,0), (0,-2), and (3,0). Each stays in place and waves one hand independently.', 'market', 92),
    ('dance-industrial-93', 'Three people dance independently at the same time for six seconds, each doing small rhythmic steps in place at (-3,0), (0,-2), and (3,0). No touching.', 'industrial', 93),
    ('pair-wave-city-94', 'Two people meet and shake hands for four seconds while a third person waves hello independently during the handshake. The third person stays apart and does not touch them.', 'city', 94),
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='all')
    parser.add_argument('--token', type=Path, required=True)
    parser.add_argument('--provider-config', type=Path, required=True)
    parser.add_argument('--gateway-env', type=Path, required=True)
    parser.add_argument('--core-url', default='http://127.0.0.1:8769')
    parser.add_argument('--model-lane-reserved', action='store_true')
    args = parser.parse_args()
    if not args.model_lane_reserved:
        parser.error('Reserve shared model lanes before running')
    selected = [c for c in CASES if args.case in ('all', c[0])]
    if not selected:
        parser.error('Unknown case')
    allowed = {'NEON_AI_GATEWAY_BASE_URL', 'NEON_AI_GATEWAY_TOKEN', 'STAGEZERO_OBJECT_API_BASE', 'STAGEZERO_OBJECT_API_KEY', 'STAGEZERO_OBJECT_MODEL', 'STAGEZERO_STORY_MODEL'}
    for line in args.gateway_env.read_text().splitlines():
        words = shlex.split(line, comments=True)
        if words and words[0] == 'export': words = words[1:]
        if len(words) == 1 and '=' in words[0]:
            k, v = words[0].split('=', 1)
            if k in allowed: os.environ[k] = v
    provider = NativePairProvider.from_config(args.provider_config)
    core = RealtimeClient(args.core_url, args.token.read_text(), job_timeout=60)
    root = ROOT / 'review/group-prompt-integration'
    for name, prompt, background, seed in selected:
        parent = root / name
        number = 1
        while (parent / f'attempt-{number:03d}').exists(): number += 1
        out = parent / f'attempt-{number:03d}'
        out.mkdir(parents=True)
        scene = json.loads((ROOT / 'review/prompt-scenes/backgrounds' / (background + '.json')).read_text())
        record = dict(prompt=prompt, background=background, seed=seed, fresh_planning=True, fresh_motion=True, status='running')
        start = time.monotonic()
        planner = ScenePromptPlanner()
        print('START', name, flush=True)
        try:
            result = PromptSceneBuilder(prompt, planner, provider, core, out / 'sources', seed)(scene)
            data = encode_project(result, cast_from_performance(result), scene_document=scene)
            (out / 'scene.cast.stagezero.npz').write_bytes(data)
            (out / 'result.json').write_text(json.dumps(result.metadata, indent=2) + '\n')
            record.update(status='generated_visual_unreviewed', frames=result.frames, archive_sha256=hashlib.sha256(data).hexdigest(), concurrency_fallback=result.metadata.get('concurrency_fallback'))
        except Exception as exc:
            record.update(status='rejected', error=str(exc))
        record['wall_seconds'] = time.monotonic() - start
        (out / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record), flush=True)
if __name__ == '__main__': main()
