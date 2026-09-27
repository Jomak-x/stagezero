"""Generate complete single-actor takes from one or more story prompts.

Examples:
    .venv/bin/python story_generate.py 'walk, then wave' --output scene.stagezero.npz
    .venv/bin/python story_generate.py 'walk' 'turn' --output batch/ --backend http://127.0.0.1:8765 --backend http://127.0.0.1:8766
"""

import argparse
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

from live_motion import Backend
from object_generation import GatewayGenerator, gateway_config
from scene_composition import validate_scene
from story_jobs import StoryJobQueue
from story_planning import StoryPlanner
from takes import encode_project


DEFAULT_TOKEN = Path(__file__).resolve().parent / '.runtime/api-token'
DEFAULT_BACKEND = 'http://127.0.0.1:8765'
MAX_BATCH = 32
MAX_SCENE_BYTES = 1_000_000


def build_parser():
    parser = argparse.ArgumentParser(description='Plan and generate complete single-actor scenes.')
    parser.add_argument('prompts', nargs='+', metavar='PROMPT', help='One full-scene request per prompt')
    parser.add_argument('--output', required=True, type=Path,
                        help='Output file for one prompt, or directory for multiple prompts')
    parser.add_argument('--scene', type=Path, help='Optional scene JSON for planning and saved project')
    parser.add_argument('--plan-only', action='store_true', help='Write validated plans without calling a motion pod')
    parser.add_argument('--backend', action='append', metavar='URL',
                        help='Existing loopback pod URL; repeat for parallel lanes')
    parser.add_argument('--token-file', type=Path, default=DEFAULT_TOKEN,
                        help='Pod bearer token file (default: .runtime/api-token)')
    parser.add_argument('--model', help='Override the gateway model used for story planning')
    parser.add_argument('--seconds', type=float, default=60,
                        help='Total scene duration per prompt, 0.16–120 seconds (default: 60)')
    parser.add_argument('--timeout', type=float, default=1800,
                        help='Maximum seconds to wait for all motion jobs (default: 1800)')
    return parser


def _scene(path):
    if path is None:
        return {'gate': {'position': [0., 0., 1.5], 'radius': .55, 'enabled': True}}
    if path.stat().st_size > MAX_SCENE_BYTES:
        raise ValueError('Scene JSON is too large')
    document = json.loads(path.read_text())
    document = validate_scene(document)
    document.pop('version')
    document['gate'] = {'position': [0., 0., 1.5], 'radius': .55,
                        'enabled': 'assets' not in document}
    return document


def _planning_context(scene):
    summary = {'name': str(scene.get('name', 'Current scene'))[:100],
               'objects': [], 'targets': []}
    if 'actor_start' in scene:
        summary['actor_start'] = scene['actor_start']
    for field in ('objects', 'targets'):
        for item in scene.get(field, []):
            if not isinstance(item, dict):
                continue
            summary[field].append({key: item[key] for key in
                                   ('id', 'name', 'kind', 'position', 'size', 'object_id')
                                   if key in item})
            if len(json.dumps(summary, ensure_ascii=False)) > 3500:
                summary[field].pop()
                summary['inventory_truncated'] = True
                return summary
    return summary


def _backend_urls(urls):
    checked = []
    for url in urls or [DEFAULT_BACKEND]:
        try:
            parsed = urlparse(url)
            port = parsed.port
        except ValueError as exc:
            raise ValueError('Backend URLs must be existing HTTP loopback tunnels') from exc
        if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost')
                or not port or parsed.username or parsed.password or parsed.query
                or parsed.fragment or parsed.path not in ('', '/')):
            raise ValueError('Backend URLs must be existing HTTP loopback tunnels')
        canonical = url.rstrip('/')
        if canonical not in checked:
            checked.append(canonical)
    return checked


def _paths(output, count, index, plan_only):
    if count > 1:
        stem = f'scene-{index + 1:02d}'
        return output / f'{stem}.plan.json', output / f'{stem}.stagezero.npz'
    if plan_only:
        return output, None
    return output.with_name(output.name + '.plan.json'), output


def _atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args, planner=None, backend_factory=Backend, queue_factory=StoryJobQueue):
    """Run a bounded batch; injected factories keep CLI tests offline."""
    from story_planning import fit_story_duration, validate_story_seconds
    validate_story_seconds(args.seconds)
    if not 1 <= len(args.prompts) <= MAX_BATCH:
        raise ValueError(f'Provide 1–{MAX_BATCH} story prompts')
    if not 0 < args.timeout < float('inf'):
        raise ValueError('Timeout must be a positive finite number')
    if len(args.prompts) > 1:
        if args.output.exists() and not args.output.is_dir():
            raise ValueError('Multiple prompts require an output directory')
        args.output.mkdir(parents=True, exist_ok=True)
    elif args.output.exists() and args.output.is_dir():
        raise ValueError('One prompt requires an output file')
    scene = _scene(args.scene)
    if planner is None and args.model:
        config = gateway_config()
        base = config.get('STAGEZERO_OBJECT_API_BASE')
        if not base and config.get('NEON_AI_GATEWAY_BASE_URL'):
            base = config['NEON_AI_GATEWAY_BASE_URL'].rstrip('/') + '/v1'
        token = config.get('STAGEZERO_OBJECT_API_KEY') or config.get('NEON_AI_GATEWAY_TOKEN')
        if not base or not token:
            raise ValueError('Configure gateway URL and token before story planning')
        planner = StoryPlanner(gateway=GatewayGenerator(base, args.model, token))
    else:
        planner = planner or StoryPlanner()
        if args.model:
            if not hasattr(planner, 'gateway'):
                raise ValueError('Planner does not support model override')
            planner.gateway.model = args.model
    context = _planning_context(scene) if args.scene else None
    plans = [fit_story_duration(planner.plan(prompt, context=context, seconds=args.seconds), args.seconds)
             for prompt in args.prompts]
    paths = [_paths(args.output, len(plans), index, args.plan_only)
             for index in range(len(plans))]
    for plan, (plan_path, _) in zip(plans, paths):
        _atomic_write(plan_path, (json.dumps(plan, indent=2, ensure_ascii=False,
                                           allow_nan=False) + '\n').encode())
        print(f'Plan: {plan_path}')
    if args.plan_only:
        return paths

    urls = _backend_urls(args.backend)
    if not args.token_file.is_file() or not args.token_file.read_text().strip():
        raise ValueError('Pod token file is missing or empty')
    backends = [backend_factory(args.token_file, url=url) for url in urls]
    queue = queue_factory(backends, max_pending=MAX_BATCH, max_history=MAX_BATCH)
    identifiers = []
    try:
        identifiers = [queue.submit(plan) for plan in plans]
        deadline = time.monotonic() + args.timeout
        remaining = set(identifiers)
        while remaining:
            for job_id in tuple(remaining):
                status = queue.snapshot(job_id)['status']
                if status == 'completed':
                    remaining.remove(job_id)
                elif status in ('failed', 'cancelled'):
                    raise RuntimeError(f'Story generation {status}; check the pod connection and retry')
            if remaining:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Story generation exceeded the requested timeout')
                time.sleep(.1)
        for job_id, (_, archive_path) in zip(identifiers, paths):
            take = queue.result(job_id)
            _atomic_write(archive_path, encode_project({take.id: take}, take.id, 0, scene))
            queue.release(job_id)
            print(f'Take: {archive_path} ({len(take.motion) / 25:.2f} s)')
        return paths
    finally:
        for job_id in identifiers:
            queue.cancel(job_id)
        queue.close()


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print('Story generation cancelled', file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
