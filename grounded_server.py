"""Local visual review of native model motion on generated humans and scenes."""
from __future__ import annotations
import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
import uuid
from urllib.parse import urlparse, parse_qs
import numpy as np

ROOT = Path(__file__).resolve().parent


def read_clip(path):
    with np.load(path, allow_pickle=False) as data:
        positions = data['positions'].astype(np.float32)
        rotations = data['rotations'].astype(np.float32)
        meta = json.loads(data['metadata'].item()) if 'metadata' in data else {}
    if positions.ndim == 3:
        positions, rotations = positions[None], rotations[None]
    if positions.shape[2:] != (27, 3) or rotations.shape != positions.shape[:-1] + (3, 3):
        raise ValueError('Expected canonical Core27 arrays')
    if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
        raise ValueError('Nonfinite motion')
    return positions, rotations, meta


class Review:
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.cache = {}
        self.rigs = []
        self.jobs = {}
        self.generated = {}
        self.output = ROOT / '.runtime' / 'grounded-review' / time.strftime('%Y%m%d-%H%M%S')
        self.output.mkdir(parents=True, exist_ok=True)

    def clips(self):
        found = dict(self.generated)
        for directory in self.args.clips:
            for path in sorted(Path(directory).rglob('*.npz')):
                if 'diagnostics' in path.parts or any(x in path.name for x in ('stagezero', 'source', 'skin')):
                    continue
                try:
                    p, r, meta = read_clip(path)
                except (ValueError, KeyError, OSError):
                    continue
                key = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
                found[key] = (path, {'id': key, 'label': meta.get('label', path.stem.replace('.raw', '').replace('_', ' ').title()),
                    'prompt': meta.get('prompt', ''), 'source': meta.get('source', 'native_model'),
                    'frames': int(p.shape[1]), 'actors': int(p.shape[0]), 'fps': 20,
                    'featured': path.stem == 'core_martial_combo_8s',
                    'continuation_supported': p.shape[0] == 1 and 'soma_conversion' not in meta,
                    'metrics': meta.get('metrics', {})})
        return found

    def continue_motion(self, body):
        from realtime_client import RealtimeClient
        key, prompt = body.get('clip_id'), body.get('prompt')
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 500:
            raise ValueError('Enter a motion prompt between 1 and 500 characters')
        entries = self.clips()
        if key not in entries:
            raise ValueError('Unknown source clip')
        if entries[key][1]['actors'] != 1:
            raise ValueError('Paired samples are replay-only: independent Core continuation does not preserve contact')
        if not entries[key][1].get('continuation_supported', True):
            raise ValueError('Kimodo samples are replay-only: SOMA proportions are incompatible with Core history')
        with self.lock:
            if any(job['status'] == 'running' for job in self.jobs.values()):
                raise ValueError('A continuation is already generating')
            if len(self.jobs) >= 32:
                raise ValueError('Review session limit reached; restart before more generation')
            job_id = 'grounded-' + uuid.uuid4().hex[:16]
            self.jobs[job_id] = {'id': job_id, 'status': 'running', 'source_clip_id': key, 'prompt': prompt}
        path, desc = entries[key]
        def work():
            started = time.perf_counter()
            next_id = None
            try:
                p, r, meta = read_clip(path)
                source_frames = p.shape[1]
                commit_frame = body.get('commit_frame', source_frames)
                if type(commit_frame) is not int or not 40 <= commit_frame <= source_frames or commit_frame % 4:
                    raise ValueError('commit_frame must be a four-frame boundary within the source, at least 40')
                p, r = p[:, :commit_frame], r[:, :commit_frame]
                if p.shape[1] < 40 or p.shape[1] > 2400:
                    raise ValueError('Continuation needs 40–2400 source frames')
                token = self.args.token_file.read_text().strip()
                client = RealtimeClient(self.args.realtime_url, token, job_timeout=60)
                history = {'positions': p[:, -40:].tolist(), 'rotations': r[:, -40:].tolist()}
                native_prefix = None
                with np.load(path, allow_pickle=False) as data:
                    if 'native_features' in data:
                        native = data['native_features']
                        if native.ndim == 2:
                            native = native[None]
                        if native.shape == (len(p), source_frames, 330):
                            history = {'native_features': native[:, commit_frame-40:commit_frame].tolist()}
                            native_prefix = native[:, :commit_frame]
                actor_ids = list(meta.get('actor_ids', ['actor' + str(i) for i in range(len(p))]))
                generation_seed = 1701 + len(self.jobs)
                chunks = client.wait({'request_id': job_id, 'stage_kind': 'continuation', 'frames': 80,
                    'actor_ids': actor_ids, 'prompt': prompt, 'seed': generation_seed, 'history': history})
                new_p = np.concatenate([c.positions for c in chunks], axis=1)
                new_r = np.concatenate([c.rotations for c in chunks], axis=1)
                boundary = np.linalg.norm(new_p[:, 0] - p[:, -1], axis=-1)
                combined_p = np.concatenate([p, new_p], axis=1)
                combined_r = np.concatenate([r, new_r], axis=1)
                elapsed = time.perf_counter() - started
                parent_context = {k: v for k, v in meta.items() if k not in
                    ('request_id', 'frames', 'stage_kind', 'client_round_trip_seconds',
                     'chunk_metadata', 'service_result', 'metrics')}
                next_meta = {**parent_context, 'source': 'native continuation experiment', 'actor_ids': actor_ids,
                    'request_id': job_id, 'parent_request_id': meta.get('request_id'),
                    'frames': int(combined_p.shape[1]), 'fps': 20, 'stage_kind': 'continuation',
                    'model': 'ardy_core',
                    'chunk_metadata': [getattr(c, 'metadata', {}) for c in chunks],
                    'prompt': prompt, 'seed': generation_seed, 'parent_seed': meta.get('seed'),
                    'parent_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'parent_clip_id': key,
                    'preserved_prefix_frames': p.shape[1], 'replaced_unplayed_frames': source_frames-commit_frame, 'generation_seconds': elapsed,
                    'boundary_max_joint_step_m': float(boundary.max()),
                    'boundary_max_root_step_m': float(boundary[:,0].max()),
                    'continuation_model': 'ARDY Core; actors generated independently',
                    'retarget_floor_offsets': self.clip(key)['metadata']['retarget_floor_offsets'],
                    'label': 'Live direction: ' + prompt[:72]}
                target = self.output / (job_id + '.npz')
                extra = {}
                if native_prefix is not None and all(getattr(c, 'native_features', None) is not None for c in chunks):
                    extra['native_features'] = np.concatenate([native_prefix] + [c.native_features for c in chunks], axis=1)
                np.savez_compressed(target, positions=combined_p, rotations=combined_r,
                    metadata=json.dumps(next_meta, allow_nan=False), **extra)
                next_id = hashlib.sha256(str(target.resolve()).encode()).hexdigest()[:12]
                with self.lock:
                    self.generated[next_id] = (target, {'id': next_id, 'label': next_meta['label'],
                        'prompt': prompt, 'source': next_meta['source'], 'frames': combined_p.shape[1],
                        'fps': 20, 'actors': len(p), 'metrics': {'generation_seconds': elapsed,
                        'boundary_max_joint_step_m': float(boundary.max())}})
                # Finish retargeting before saying the result is ready to play.
                self.clip(next_id)
                with self.lock:
                    self.jobs[job_id].update(status='complete', clip_id=next_id,
                        generation_seconds=elapsed, ready_seconds=time.perf_counter()-started,
                        preserved_prefix_frames=p.shape[1])
            except Exception as exc:
                with self.lock:
                    if next_id is not None:
                        self.generated.pop(next_id, None)
                        self.cache = {k:v for k,v in self.cache.items() if k[0] != next_id}
                    self.jobs[job_id].update(status='failed', error=str(exc))
            (self.output / (job_id + '.json')).write_text(json.dumps(self.jobs[job_id], indent=2))
        threading.Thread(target=work, daemon=True).start()
        return dict(self.jobs[job_id])

    def rig(self, actor=0):
        # Adapter import is intentionally lazy: the research rig is independently testable.
        from grounded_character import GroundedCharacter
        with self.lock:
            while len(self.rigs) <= actor:
                i = len(self.rigs)
                self.rigs.append(GroundedCharacter(self.args.characters[min(i, len(self.args.characters)-1)]))
            return self.rigs[actor]

    def clip(self, key):
        entries = self.clips()
        if key not in entries:
            raise ValueError('Unknown clip')
        path, desc = entries[key]
        stamp = (key, path.stat().st_mtime_ns)
        with self.lock:
            if stamp in self.cache:
                return self.cache[stamp]
            p, r, meta = read_clip(path)
            fitted_p, fitted_r, fit_meta = [], [], []
            floor_offsets = meta.get('retarget_floor_offsets')
            retarget_options = meta.get('retarget_options', {})
            parent = None
            prefix = meta.get('preserved_prefix_frames', 0)
            parent_id = meta.get('parent_clip_id')
            if (parent_id in entries and parent_id != key and type(prefix) is int
                    and 0 < prefix < p.shape[1] and floor_offsets is not None):
                parent_entry = entries[parent_id]
                # Reuse only a verified immutable parent, with identical fit settings.
                if hashlib.sha256(parent_entry[0].read_bytes()).hexdigest() == meta.get('parent_file_sha256'):
                    candidate = self.clip(parent_id)
                    if (candidate['frames'] >= prefix and candidate['actors'] == len(p)
                            and candidate['metadata'].get('retarget_options', {}) == retarget_options
                            and candidate['metadata']['retarget_floor_offsets'] == floor_offsets):
                        parent = candidate
            start = prefix if parent is not None else 0
            for actor in range(p.shape[0]):
                fit = self.rig(actor).clip_payload(p[actor:actor+1, start:], r[actor:actor+1, start:],
                    preserve_wrists=(len(p) == 2 or bool(retarget_options.get('preserve_wrists', False))),
                    wrist_target_space=retarget_options.get('wrist_target_space', 'retargeted_root'),
                    preserve_feet=bool(retarget_options.get('preserve_feet', False)), floor_y=-.01,
                    floor_offsets=None if floor_offsets is None else [floor_offsets[actor]])
                provenance = fit['character_provenance']
                provenance['evaluated_frame_range'] = [start, int(p.shape[1])]
                if parent is not None:
                    provenance['prefix_reused_from_sha256'] = meta['parent_file_sha256']
                fit_meta.append(provenance)
                fitted_p.append((parent['fitted_positions'][actor][:start] if parent else []) + fit['fitted_positions'][0])
                fitted_r.append((parent['fitted_rotations'][actor][:start] if parent else []) + fit['fitted_rotations'][0])
            result = {**desc, 'positions': p.tolist(), 'rotations': r.tolist(),
                'fitted_positions': fitted_p, 'fitted_rotations': fitted_r,
                'metadata': {**meta, 'retarget_provenance': fit_meta,
                    'retarget_floor_offsets': [item['floor_offsets'][0] for item in fit_meta], 'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'playback': 'saved native model output; independent clips are not live continuations'}}
            while len(self.cache) >= 4:
                self.cache.pop(next(iter(self.cache)))
            self.cache[stamp] = result
            return result


    def clip_tail(self, key, start):
        full = self.clip(key)
        if type(start) is not int or not 0 <= start < full['frames']:
            raise ValueError('start must be a frame within the clip')
        result = {**full, 'start_frame': start,
                  'source_clip_id': full['metadata'].get('parent_clip_id')}
        for field in ('positions', 'rotations', 'fitted_positions', 'fitted_rotations'):
            result[field] = [actor[start:] for actor in full[field]]
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=2361)
    parser.add_argument('--token-file', type=Path, default=ROOT / '.runtime/api-token')
    parser.add_argument('--realtime-url', default='http://127.0.0.1:8769')
    parser.add_argument('--clips', action='append', default=[])
    parser.add_argument('--characters', nargs='+', default=[str(ROOT / 'grounded_assets/characters/civilian.glb'), str(ROOT / 'grounded_assets/characters/ranger.glb')])
    parser.add_argument('--three-dir', type=Path, default=ROOT / 'studio_client/node_modules/three')
    args = parser.parse_args()
    if not args.clips:
        args.clips = [str(ROOT / 'grounded_assets/clips')]
    review = Review(args)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *values):
            pass
        def send(self, body, status=200, content_type='application/json'):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, allow_nan=False, separators=(',', ':')).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            import mimetypes
            url = urlparse(self.path)
            query = parse_qs(url.query)
            try:
                if url.path == '/api/grounded/job':
                    job = review.jobs.get(query.get('id', [''])[0])
                    return self.send(job or {'error': 'Unknown job'}, 200 if job else 404)
                if url.path == '/api/grounded/clips':
                    return self.send([item[1] for item in review.clips().values()])
                if url.path == '/api/grounded/clip':
                    key = query.get('id', [''])[0]
                    return self.send(review.clip_tail(key, int(query['start'][0])) if 'start' in query else review.clip(key))
                if url.path == '/api/grounded/rig':
                    actor = int(query.get('actor', ['0'])[0])
                    if actor not in (0, 1):
                        raise ValueError('Invalid actor')
                    return self.send(review.rig(actor).rig_payload())
                if url.path == '/api/grounded/scene':
                    from grounded_scene import scene_payload
                    return self.send(scene_payload())
                base = args.three_dir if url.path.startswith('/three/') else ROOT / 'grounded_web'
                relative = url.path[len('/three/'):] if url.path.startswith('/three/') else url.path.lstrip('/') or 'index.html'
                target = (base / relative).resolve()
                if not target.is_relative_to(base.resolve()) or not target.is_file():
                    return self.send({'error': 'Not found'}, 404)
                return self.send(target.read_bytes(), content_type=mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
            except Exception as exc:
                self.send({'error': str(exc)}, 400)
        def do_POST(self):
            origin = self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{args.port}', f'http://localhost:{args.port}'):
                return self.send({'error': 'Cross-origin writes are not allowed'}, 403)
            if self.path == '/api/grounded/telemetry':
                try:
                    count = int(self.headers.get('Content-Length', '0'))
                    if not 0 < count <= 1000000:
                        raise ValueError('Invalid telemetry size')
                    report = json.loads(self.rfile.read(count))
                    if not isinstance(report, dict):
                        raise ValueError('Expected telemetry object')
                    path = review.output / ('telemetry-' + str(time.time_ns()) + '.json')
                    path.write_text(json.dumps(report, indent=2, allow_nan=False))
                    return self.send({'saved': str(path)})
                except Exception as exc:
                    return self.send({'error': str(exc)}, 400)
            if self.path == '/api/grounded/continue':
                try:
                    count = int(self.headers.get('Content-Length', '0'))
                    if not 0 < count <= 8192:
                        raise ValueError('Invalid request size')
                    return self.send(review.continue_motion(json.loads(self.rfile.read(count))), 202)
                except Exception as exc:
                    return self.send({'error': str(exc)}, 400)
            if self.path != '/api/grounded/recording':
                return self.send({'error': 'Not found'}, 404)
            count = int(self.headers.get('Content-Length', '0'))
            if not 0 < count < 250_000_000:
                return self.send({'error': 'Invalid recording size'}, 413)
            path = review.output / ('interactive-review-' + str(time.time_ns()) + '.webm')
            with review.lock:
                used = sum(p.stat().st_size for p in review.output.glob('*.webm'))
                if used + count > 500_000_000:
                    return self.send({'error': 'Recording session limit reached'}, 413)
                raw = self.rfile.read(count)
                if len(raw) != count:
                    return self.send({'error': 'Incomplete recording'}, 400)
                path.write_bytes(raw)
            self.send({'saved': str(path), 'bytes': path.stat().st_size})
    print(f'Grounded native motion review: http://127.0.0.1:{args.port}/', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()

if __name__ == '__main__':
    main()
