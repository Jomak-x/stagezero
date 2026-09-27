"""Reproduce bounded Core group sequences from a saved plan and real background."""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cast_performance import encode_project, cast_from_performance
from group_scene_placement import group_sequence_placement
from group_scene_sequence import build_group_sequence
from prompt_scene_plan import validate_plan
from realtime_client import RealtimeClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--policy', choices=('continuous', 'fresh'), default='fresh')
    parser.add_argument('--core-url', default='http://127.0.0.1:8769')
    parser.add_argument('--token-file', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene = json.loads(args.scene.read_text())
    plan = validate_plan(json.loads(args.plan.read_text()), scene)
    report = {'seed':args.seed,'policy':args.policy,'plan':plan,'scene':scene,
              'visual_acceptance':'unverified','physical_contact_verified':False}
    started = time.monotonic()
    try:
        placement = group_sequence_placement(plan, scene)
        report['placement'] = placement
        client = RealtimeClient(args.core_url, args.token_file.read_text().strip(), job_timeout=90)
        result = build_group_sequence(client, scene, plan, placement['starts'], placement['targets'],
            seed=args.seed, output_root=args.output/'sources', fresh_action_stages=args.policy=='fresh')
        performance = result['performance']
        (args.output/'scene.cast.stagezero.npz').write_bytes(encode_project(
            performance, cast_from_performance(performance), scene_document=scene))
        report.update(status='generated_unreviewed',source_manifest=str(result['manifest']))
    except Exception as exc:
        report.update(status='rejected',error=str(exc))
        raise
    finally:
        report['wall_seconds'] = time.monotonic()-started
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('scene','plan','placement')}))

if __name__ == '__main__':
    main()
