"""Bounded HTTP transport for the warm realtime motion service."""
from __future__ import annotations
import io
import json
import math
import time
from urllib.error import HTTPError
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zipfile import BadZipFile, ZipFile
import numpy as np
from realtime_clip import CanonicalClip


MAX_RESPONSE_BYTES = 16_000_000
HORIZON = 40


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the service bearer token to a redirect destination.
        return None


class MotionServiceError(RuntimeError):
    """An explicit HTTP rejection, distinct from an uncertain network failure."""
    def __init__(self, status, detail):
        self.status = status
        super().__init__(f'Motion service HTTP {status}: {detail}')


class RealtimeClient:
    def __init__(self, url: str, token: str, *, timeout: float = 15, job_timeout: float = 180):
        parsed = urlparse(url)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError('Expected an HTTP service URL without embedded credentials, query or fragment')
        if not token.strip() or not all(math.isfinite(x) and x > 0 for x in (timeout, job_timeout)):
            raise ValueError('Token and positive finite time limits are required')
        self.url, self.token = url.rstrip('/'), token.strip()
        self.timeout, self.job_timeout = timeout, job_timeout

    def _request(self, method, path, body=None, *, deadline=None, timeout=None):
        limit = self.timeout if timeout is None else min(self.timeout, timeout)
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Motion job exceeded the client time limit')
            limit = min(limit, remaining)
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        request = Request(self.url + path, data=data, method=method,
                          headers={'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json'})
        try:
            with build_opener(_NoRedirect()).open(request, timeout=limit) as response:
                parts = []
                size = 0
                while True:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError('Motion job exceeded the client time limit')
                    part = response.read1(min(65536, MAX_RESPONSE_BYTES + 1 - size))
                    if not part:
                        break
                    parts.append(part)
                    size += len(part)
                    if size > MAX_RESPONSE_BYTES:
                        raise RuntimeError('Service response exceeded the size limit')
                payload = b''.join(parts)
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError('Motion job exceeded the client time limit')
                return payload, response.headers.get('Content-Type', '')
        except HTTPError as exc:
            raw = exc.read(2000)
            try:
                detail = json.loads(raw).get('error', 'Request failed')
            except (ValueError, AttributeError):
                detail = 'Request failed'
            raise MotionServiceError(exc.code, detail) from None

    def health(self):
        return json.loads(self._request('GET', '/health')[0])

    def submit(self, body, *, deadline=None):
        return json.loads(self._request('POST', '/v1/realtime/jobs', body, deadline=deadline)[0])

    def status(self, request_id, *, deadline=None):
        return json.loads(self._request('GET', '/v1/realtime/jobs/' + quote(request_id, safe=''), deadline=deadline)[0])

    def cancel(self, request_id, *, timeout=None):
        return json.loads(self._request('DELETE', '/v1/realtime/jobs/' + quote(request_id, safe=''), timeout=timeout)[0])

    def chunk(self, request_id, index, *, deadline=None):
        if type(index) is not int or index < 0:
            raise ValueError('Chunk index must be a nonnegative integer')
        data, content_type = self._request('GET', '/v1/realtime/jobs/' + quote(request_id, safe='') + f'/chunks/{index}', deadline=deadline)
        if content_type.split(';')[0].strip() != 'application/octet-stream':
            raise RuntimeError('Motion chunk is not ready')
        try:
            # Cap expanded data too: a small compressed archive can otherwise
            # allocate gigabytes before CanonicalClip checks its array shapes.
            with ZipFile(io.BytesIO(data)) as zipped:
                infos = zipped.infolist()
                required = {'positions.npy', 'rotations.npy', 'metadata.npy'}
                names = {entry.filename for entry in infos}
                if (len(names) != len(infos) or not required <= names
                        or names - required - {'native_features.npy'}
                        or sum(entry.file_size for entry in infos) > MAX_RESPONSE_BYTES):
                    raise ValueError('Invalid or oversized motion chunk archive')
                for entry in infos:
                    with zipped.open(entry) as member:
                        version = np.lib.format.read_magic(member)
                        if version == (1, 0):
                            shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                        elif version == (2, 0):
                            shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                        else:
                            raise ValueError('Unsupported motion array format')
                    if entry.filename == 'metadata.npy':
                        valid = shape == () and dtype.kind in 'US' and dtype.itemsize <= 1_000_000
                    else:
                        tail = {'positions.npy': (HORIZON, 27, 3),
                                'rotations.npy': (HORIZON, 27, 3, 3),
                                'native_features.npy': (HORIZON, 330)}[entry.filename]
                        valid = (len(shape) == len(tail) + 1 and shape[0] in (1, 2)
                                 and shape[1:] == tail and dtype.kind in 'fi' and dtype.itemsize <= 8)
                    if not valid:
                        raise ValueError('Invalid motion array shape or dtype')
            with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                metadata = json.loads(archive['metadata'].item())
                if metadata['request_id'] != request_id or metadata['chunk_index'] != index:
                    raise ValueError('Service returned a different motion chunk')
                if (metadata['stage_kind'] not in ('paired', 'approach', 'transition', 'continuation')
                        or metadata['frames'] != HORIZON or metadata['start_frame'] != index * HORIZON):
                    raise ValueError('Service returned incompatible motion chunk timing or stage')
                source = 'intergen' if metadata['stage_kind'] == 'paired' else 'ardy_core'
                if source == 'ardy_core' and 'native_features' not in archive:
                    raise ValueError('Core motion chunk is missing native history features')
                clip = CanonicalClip(archive['positions'], archive['rotations'], metadata['fps'],
                                     tuple(metadata['actor_ids']), source, metadata,
                                     archive['native_features'] if 'native_features' in archive else None)
                if clip.frames != HORIZON:
                    raise ValueError('Service returned an incomplete motion horizon')
                return clip
        except (BadZipFile, KeyError, TypeError, AttributeError) as exc:
            raise ValueError('Invalid motion chunk payload') from exc

    def wait(self, body, on_chunk=None, *, cancelled=lambda: False):
        request_id = body['request_id']
        deadline = time.monotonic() + self.job_timeout
        if cancelled():
            raise RuntimeError('Generation cancelled')
        clips = []
        accepted = False
        submission_attempted = False
        try:
            submission_attempted = True
            result = self.submit(body, deadline=deadline)
            accepted = True
            if result.get('request_id') != request_id:
                raise RuntimeError('Service acknowledged a different motion job')
            while True:
                if cancelled():
                    raise RuntimeError('Generation cancelled')
                state = self.status(request_id, deadline=deadline)
                indices = state.get('available_chunks')
                expected_chunks = body['frames'] // HORIZON
                if (state.get('request_id') != request_id or state.get('stage_kind') != body['stage_kind']
                        or state.get('total_chunks') != expected_chunks
                        or state.get('status') not in ('queued', 'running', 'complete', 'failed', 'cancelled')
                        or not isinstance(indices, list)
                        or any(type(i) is not int or not 0 <= i < expected_chunks for i in indices)
                        or indices != sorted(set(indices))):
                    raise RuntimeError('Service returned an incompatible motion job status')
                for index in indices:
                    if index < len(clips):
                        continue
                    if index != len(clips):
                        raise RuntimeError('Service skipped a synchronized motion chunk')
                    if cancelled():
                        raise RuntimeError('Generation cancelled')
                    clip = self.chunk(request_id, index, deadline=deadline)
                    if (clip.actor_ids != tuple(body['actor_ids'])
                            or clip.metadata['stage_kind'] != body['stage_kind']):
                        raise RuntimeError('Service returned different actors or motion stage')
                    clips.append(clip)
                    if on_chunk is not None:
                        on_chunk(clip)
                if state['status'] == 'complete':
                    if len(clips) != expected_chunks:
                        raise RuntimeError('Service completed without all motion chunks')
                    return clips
                if state['status'] in ('failed', 'cancelled'):
                    raise RuntimeError(state.get('error') or state['status'])
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Motion job exceeded the client time limit')
                time.sleep(min(.025, remaining))
        except Exception as exc:
            # A timeout during POST can mean the GPU job was accepted but its
            # acknowledgement was lost. Cancel that ID too. An explicit rejected
            # POST (e.g. duplicate ID) must not cancel an existing unrelated job.
            rejected = isinstance(exc, MotionServiceError) and not accepted
            if submission_attempted and not rejected:
                try:
                    self.cancel(request_id, timeout=min(self.timeout, 1.0))
                except Exception:
                    pass
            raise


def request_body(request, *, seed=33):
    """Translate a director horizon to a Core service job, preserving native history."""
    mapping = {'approach': 'approach', 'action': 'approach', 'exit': 'continuation',
               'release': 'continuation', 'transition': 'transition'}
    if request.source != 'ardy_core':
        raise ValueError('Paired scenes require the coordinated scene runner and a cached full pair sample')
    kind = mapping[request.stage_kind]
    if request.history is not None and kind == 'approach':
        kind = 'continuation'
    body = {'request_id': request.request_id, 'stage_kind': kind,
            'frames': request.frames, 'prompt': request.prompt,
            'actor_ids': list(request.actor_ids), 'seed': request.metadata.get('seed', seed),
            'actor_prompts': request.actor_prompts or {}}
    if request.history is not None:
        clip = request.history
        if clip.actor_ids != tuple(request.actor_ids):
            raise ValueError('History actor order must match the request actor IDs')
        body['history'] = ({'native_features': clip.native_features.tolist()} if clip.native_features is not None
                           else {'positions': clip.positions.tolist(), 'rotations': clip.rotations.tolist()})
    for name in ('root_targets', 'target'):
        if name in request.metadata:
            body[name] = request.metadata[name]
    # Placements are cold-start instructions. Reapplying them to native
    # history translates/rotates its world-space prefix in the Core runtime.
    if request.history is None and 'initial_placements' in request.metadata:
        body['initial_placements'] = request.metadata['initial_placements']
    if request.history is None and len(request.actor_ids) == 2 and 'initial_placements' not in body:
        body['initial_placements'] = {actor_id: {'position_xz': [x, 0.0]}
                                      for actor_id, x in zip(request.actor_ids, (-1.2, 1.2))}
    if "pose_cue_profile" in request.metadata:
        from core_pose_cues import apply_profile
        body = apply_profile(body, request.history, request.metadata["pose_cue_profile"])
    return body
