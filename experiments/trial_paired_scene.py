"""One bounded paired InterGen research job on an already running private service.

Saves retargeted Core27 / 20 fps canonical arrays, never a native Core project
or fabricated ARDY history. Does not load models, provision Pods, or retry jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
import uuid
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from realtime_client import RealtimeClient
from realtime_clip import CanonicalClip

MAX_ARCHIVE_BYTES = 4_000_000
ARTIFACT_KIND = 'paired_intergen_research_canonical'


def request_body(prompt, seed, frames):
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 500:
        raise ValueError('Paired prompt must contain 1–500 characters')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Paired seed must be a uint32')
    if type(frames) is not int or frames not in (40, 80, 120):
        raise ValueError('Paired frames must be 40, 80, or 120')
    identifier = uuid.uuid4().hex
    return {'request_id': 'paired-trial-' + identifier, 'stage_kind': 'paired',
            'frames': frames, 'prompt': prompt.strip(), 'actor_ids': ['actor_1', 'actor_2'],
            'seed': seed, 'pair_sequence_id': 'paired-sequence-' + identifier,
            'source_start_frame': 0, 'source_total_frames': frames}


def join_chunks(chunks, body):
    if not chunks:
        raise ValueError('No complete paired chunks were returned')
    if len(chunks) > body['frames'] // 40:
        raise ValueError('Service returned too many paired chunks')
    for index, clip in enumerate(chunks):
        if (clip.source != 'intergen' or clip.native_features is not None or clip.frames != 40
                or clip.fps != 20 or clip.actor_ids != tuple(body['actor_ids'])
                or clip.metadata.get('request_id') != body['request_id']
                or clip.metadata.get('chunk_index') != index
                or clip.metadata.get('stage_kind') != 'paired'):
            raise ValueError('Unexpected source, timing, or actors in paired research output')
    metadata = {'version': 1, 'artifact_kind': ARTIFACT_KIND, 'research_only': True,
                'source': 'intergen', 'fps': 20, 'actor_ids': body['actor_ids'],
                'request': body, 'complete': len(chunks) * 40 == body['frames'],
                'source_format': 'InterGen paired joints at 30 fps retargeted by the service to Core27 at 20 fps',
                'native_features_available': False, 'physical_contact_verified': False,
                'chunks': [dict(clip.metadata) for clip in chunks]}
    return CanonicalClip(np.concatenate([c.positions for c in chunks], axis=1),
                         np.concatenate([c.rotations for c in chunks], axis=1),
                         20, tuple(body['actor_ids']), 'intergen', metadata)


def save_research_clip(path, clip):
    if clip.source != 'intergen' or clip.native_features is not None:
        raise ValueError('Only non-native InterGen research arrays can use this archive')
    encoded = json.dumps(dict(clip.metadata), allow_nan=False)
    with Path(path).open('xb') as handle:
        np.savez_compressed(handle, positions=clip.positions, rotations=clip.rotations,
                            metadata=np.array(encoded))


def load_research_clip(path):
    """Read our explicit research format with bounded archive and array sizes."""
    path = Path(path)
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError('Research canonical archive exceeds its size limit')
    content = path.read_bytes()
    with ZipFile(io.BytesIO(content)) as archive:
        infos = archive.infolist()
        expected = {'positions.npy', 'rotations.npy', 'metadata.npy'}
        if (len(infos) != 3 or {entry.filename for entry in infos} != expected
                or sum(entry.file_size for entry in infos) > MAX_ARCHIVE_BYTES):
            raise ValueError('Invalid research canonical archive members')
        for entry in infos:
            with archive.open(entry) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    raise ValueError('Unsupported research array format')
            if entry.filename == 'metadata.npy':
                valid = shape == () and dtype.kind in 'US' and dtype.itemsize <= 100_000
            else:
                tail = (27, 3) if entry.filename == 'positions.npy' else (27, 3, 3)
                valid = (len(shape) == len(tail) + 2 and shape[0] == 2
                         and shape[1] in (40, 80, 120) and shape[2:] == tail
                         and dtype.kind == 'f' and dtype.itemsize <= 8)
            if not valid:
                raise ValueError('Invalid research array shape or dtype')
    with np.load(io.BytesIO(content), allow_pickle=False) as archive:
        metadata = json.loads(archive['metadata'].item())
        if (not isinstance(metadata, dict) or metadata.get('version') != 1
                or metadata.get('artifact_kind') != ARTIFACT_KIND or metadata.get('source') != 'intergen'
                or metadata.get('research_only') is not True or metadata.get('fps') != 20
                or metadata.get('native_features_available') is not False
                or metadata.get('actor_ids') != ['actor_1', 'actor_2']):
            raise ValueError('Archive is not an explicitly marked paired InterGen research clip')
        clip = CanonicalClip(archive['positions'], archive['rotations'], 20,
                             tuple(metadata['actor_ids']), 'intergen', metadata)
    return clip, content


def run_trial(args, *, client=None):
    body = request_body(args.prompt, args.seed, args.frames)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', args.name):
        raise ValueError('Trial name must be 1–80 letters, digits, hyphens, or underscores')
    if not 0 < args.job_timeout <= 600:
        raise ValueError('Job timeout must be in (0,600] seconds')
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / (args.name + '.json')
    clip_path = args.output / (args.name + '.intergen.canonical.npz')
    if report_path.exists() or clip_path.exists():
        raise FileExistsError('Paired trial output already exists; choose a new name')
    token = args.token_file.read_text().strip() if client is None else ''
    client = client or RealtimeClient(args.url, token, timeout=10, job_timeout=args.job_timeout)
    report = {'version': 1, 'name': args.name, 'status': 'failed', 'source': 'intergen',
              'research_only': True, 'artifact_kind': ARTIFACT_KIND, 'request': body,
              'service_url': args.url, 'max_gpu_jobs': 1, 'submitted_jobs': 0,
              'license': 'CC BY-NC-SA 4.0 (InterGen research checkpoint)',
              'native_core_project': False, 'physical_contact_verified': False,
              'scene_conditioned': False, 'visual_review': 'pending'}
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    chunks = []
    started = time.monotonic()

    def receive(clip):
        candidate = chunks + [clip]
        join_chunks(candidate, body)
        chunks.append(clip)
        report.setdefault('first_complete_chunk_seconds', time.monotonic() - started)

    try:
        health = client.health()
        report['health'] = health
        if health.get('pair_research_enabled') is not True:
            raise RuntimeError('Existing service does not expose paired InterGen research; no job submitted')
        report['submitted_jobs'] = 1
        returned = client.wait(body, on_chunk=receive)
        if len(returned) * 40 != body['frames'] or len(chunks) != len(returned):
            raise RuntimeError('Paired job did not return the complete requested sequence')
        report['status'] = 'complete'
        # Preserve service retargeting/model provenance, without another generation.
        try:
            report['service_result'] = client.status(body['request_id'])
        except Exception:
            report['service_result_unavailable'] = True
    except Exception as exc:
        detail = str(exc)
        report['error'] = detail.replace(token, '[redacted]') if token else detail
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        if chunks:
            clip = join_chunks(chunks, body)
            save_research_clip(clip_path, clip)
            restored, content = load_research_clip(clip_path)
            report.update({'canonical_clip': str(clip_path.resolve()), 'frames': clip.frames,
                           'fps': clip.fps, 'duration_seconds': clip.frames / clip.fps,
                           'canonical_sha256': hashlib.sha256(content).hexdigest(),
                           'exact_save_load': bool(np.array_equal(clip.positions, restored.positions)
                                                  and np.array_equal(clip.rotations, restored.rotations)),
                           'partial_output': clip.frames != body['frames']})
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report.get(k) for k in ('name', 'status', 'error', 'frames', 'canonical_clip')},
                     allow_nan=False), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--seed', type=int, default=7301)
    parser.add_argument('--frames', type=int, choices=(40, 80, 120), default=120)
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8769')
    parser.add_argument('--name', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job-timeout', type=float, default=180)
    args = parser.parse_args(argv)
    try:
        report = run_trial(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')
    if report['status'] != 'complete':
        parser.exit(1)


if __name__ == '__main__':
    main()
