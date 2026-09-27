"""Run exactly one installed-checkpoint native InterGen research sample."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_pair_provider import NativePairProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prompt', default='Two people box with each other.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--frames', type=int, default=210)
    parser.add_argument('--name', default='box42')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9_-]{1,48}', args.name):
        raise SystemExit('Sample name must be lowercase letters, digits, _ or -')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit('Output directory must be empty; existing evidence is preserved')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provider = NativePairProvider.from_config(args.config)
    start = time.monotonic()
    clip = provider.generate(args.prompt, args.seed, args.frames)
    content = provider.last_raw_archive
    if content is None:
        raise RuntimeError('Provider did not retain the native source archive')
    raw_path = args.output_dir / (args.name + '.native.npz')
    with raw_path.open('xb') as output:
        output.write(content)
    provenance = {
        'version': 1, 'source': 'InterGen published research checkpoint',
        'license': 'CC BY-NC-SA 4.0; noncommercial research preview only',
        'prompt': args.prompt, 'seed': args.seed, 'frames': clip.frames, 'fps': clip.fps,
        'joints_shape': list(clip.joints.shape), 'features_shape': list(clip.features.shape),
        'raw_file': raw_path.name, 'sha256': hashlib.sha256(content).hexdigest(),
        'bytes': len(content), 'wall_seconds': round(time.monotonic() - start, 3),
        'metadata': clip.metadata,
    }
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'raw': str(raw_path), 'provenance': str(args.output_dir / 'provenance.json'),
                      'sha256': provenance['sha256'], 'frames': clip.frames, 'fps': clip.fps}))


if __name__ == '__main__':
    main()
