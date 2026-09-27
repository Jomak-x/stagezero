"""Reproduce a complete paired city direction on existing configured workers."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paired_direction import PairedSceneBuilder
from native_pair_provider import NativePairProvider
from native_pair_clip import encode_project
from realtime_client import RealtimeClient


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--token', type=Path, required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--core-url', default='http://127.0.0.1:8769')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    request = json.loads(a.request.read_text()); scene = json.loads(a.scene.read_text())
    provider = NativePairProvider.from_config(a.config) if a.config else None
    a.output.mkdir(parents=True, exist_ok=False)
    builder = PairedSceneBuilder(request, provider, RealtimeClient(a.core_url, a.token.read_text()), a.output/'sources')
    result = builder(scene, on_progress=lambda message: print(message, flush=True))
    cast = [{'id':aid,'name':f'Actor {i+1}','color':color} for i,(aid,color) in enumerate(zip(request['actor_ids'],[[52,209,220],[250,178,78]]))]
    (a.output/'scene.native-pair.stagezero.npz').write_bytes(encode_project(result['clip'], cast, request['actor_ids'], scene, 0, result['placement']))
    (a.output/'result.json').write_text(json.dumps(result['clip'].metadata, indent=2)+'\n')
    print(f"Saved {result['clip'].frames} frames: {a.output}", flush=True)

if __name__ == '__main__':
    main()
