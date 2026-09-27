"""Generate and archive a real prompt scene on existing configured services.

Run only after coordinating shared GPU access. Does not provision or restart
services. A supplied plan is replayed exactly; otherwise the AI planner runs.
"""
import argparse
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cast_performance import encode_project, cast_from_performance
from native_pair_provider import NativePairProvider
from native_pair_clip import load_source
from prompt_scene_builder import PromptSceneBuilder
from prompt_scene_plan import ScenePromptPlanner, validate_plan
from realtime_client import RealtimeClient


class ReplayPairProvider:
    """Replay exact archived native sources; Core still runs on the real service."""
    replay = True

    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        self.seed = self.manifest['seed']
        self.sources = {}
        self.last_raw_archive = None
        for record in self.manifest['sources']:
            if record['source'] != 'intergen':
                continue
            path = Path(record['path'])
            if not path.is_file():
                path = self.manifest_path.parent/path.name
            if not path.is_file():
                raise ValueError('Archived native replay source is missing: '+record['path'])
            index = record.get('beat_index')
            if index is None:
                match = re.fullmatch(r'pair-(\d+)\.npz', path.name)
                if match is None:
                    raise ValueError('Archived pair source is missing its beat index')
                index = int(match.group(1))
            if index in self.sources:
                raise ValueError('Duplicate archived pair source for one beat')
            self.sources[index] = (path, record)

    def generate(self, prompt, seed, frames, cancelled):
        if cancelled():
            raise RuntimeError('Native replay cancelled')
        index = (seed-self.seed) % 2**32
        if index not in self.sources:
            raise ValueError('Native replay has no archived source for this beat/seed')
        path, record = self.sources[index]
        raw = path.read_bytes()
        clip = load_source(raw)
        if clip.frames != frames or record.get('source_prompt', clip.metadata.get('prompt')) != prompt:
            raise ValueError('Native replay prompt or duration differs from the archived source')
        self.last_raw_archive = raw
        return clip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--scene', type=Path, required=True)
    planning = parser.add_mutually_exclusive_group()
    planning.add_argument('--plan', type=Path)
    planning.add_argument('--replay-manifest', type=Path, help='Reuse archived plan and exact InterGen sources; generate real Core approaches again')
    parser.add_argument('--token', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--core-url', default='http://127.0.0.1:8769')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text())
    planner = ScenePromptPlanner()
    provider = None
    if args.replay_manifest:
        if args.config:
            parser.error('--config is unnecessary with --replay-manifest')
        provider = ReplayPairProvider(args.replay_manifest)
        if provider.seed != args.seed:
            parser.error('--seed must match the archived manifest for exact native replay')
        plan = validate_plan(provider.manifest['plan'], scene, args.prompt)
        planner = SimpleNamespace(plan=lambda *a, **kw: plan)
    elif args.plan:
        plan = validate_plan(json.loads(args.plan.read_text()), scene, args.prompt)
        planner = SimpleNamespace(plan=lambda *a, **kw: plan)
    if args.config:
        provider = NativePairProvider.from_config(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    builder = PromptSceneBuilder(args.prompt, planner, provider,
        RealtimeClient(args.core_url, args.token.read_text()), args.output/'sources', args.seed)
    result = builder(scene, on_progress=lambda value: print(json.dumps(value), flush=True))
    target = args.output/'scene.cast.stagezero.npz'
    target.write_bytes(encode_project(result, cast_from_performance(result), scene_document=scene))
    (args.output/'result.json').write_text(json.dumps(result.metadata, indent=2)+'\n')
    print(json.dumps({'project': str(target), 'actors': len(result.actor_ids),
                      'frames': result.frames, 'seconds': result.frames/result.fps}), flush=True)


if __name__ == '__main__':
    main()
