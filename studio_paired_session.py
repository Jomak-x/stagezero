"""Independent, explicitly research-only paired InterGen Studio session.

One bounded remote job produces a complete pair. No chunk replaces the current
clip, backdrop, or playhead until the entire requested sample validates.
"""
from __future__ import annotations

import math
import threading
import time

from paired_scene import (ACTOR_IDS, EMPTY_SCENE, LICENSE, SCHEMA, check_scene_geometry, decode_project,
                          encode_project, join_chunks, json_copy, request_body,
                          scene_copy, validate_chunk)


class PairedStudioSession:
    def __init__(self, client=None, *, clock=time.monotonic):
        self._lock = threading.RLock()
        self._client = client
        self._clock = clock
        self._active = False
        self._closed = False
        self._clip = None
        self._scene = scene_copy(EMPTY_SCENE)
        self._frame = 0
        self._playing = False
        self._play_started = None
        self._revision = 0
        self._epoch = 0
        self._pending = None
        self._busy = False
        self._failure = None
        self._thread = None

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError('Paired research session is closed')

    def _require_active(self):
        self._ensure_open()
        if not self._active:
            raise RuntimeError('Activate paired research first')

    @property
    def active(self):
        with self._lock:
            return self._active

    @property
    def available(self):
        with self._lock:
            return self._client is not None and not self._closed

    @property
    def revision(self):
        with self._lock:
            return self._revision

    @property
    def scene_document(self):
        with self._lock:
            return json_copy(self._scene)

    def activate(self, active=True):
        if type(active) is not bool:
            raise ValueError('Paired activation must be a boolean')
        if not active:
            return self.deactivate()
        with self._lock:
            self._ensure_open()
            self._active = True
        return self.snapshot()

    set_active = activate

    def _invalidate(self):
        self._epoch += 1
        self._pending = None

    def deactivate(self):
        with self._lock:
            self._invalidate()
            self._active = False
            self._playing = False
            self._play_started = None
        return self.snapshot()

    stop = deactivate

    def configure_client(self, client):
        with self._lock:
            self._ensure_open()
            self._invalidate()
            self._client = client
        return self.snapshot()

    def cancel(self):
        with self._lock:
            self._ensure_open()
            self._invalidate()
            self._failure = None
        return self.snapshot()

    def _cancelled(self, epoch):
        with self._lock:
            return self._closed or not self._active or self._epoch != epoch

    def generate(self, prompt, seed, frames=120, scene_document=None):
        body = request_body(prompt, seed, frames)
        with self._lock:
            self._require_active()
            if self._client is None:
                raise RuntimeError('Paired research service is not configured')
            if self._busy:
                raise RuntimeError('A paired request is still running or cancelling; wait before starting another')
            scene = scene_copy(self._scene if scene_document is None else scene_document)
            self._epoch += 1
            epoch = self._epoch
            self._failure = None
            self._busy = True
            self._pending = {'request_id': body['request_id'], 'prompt': body['prompt'],
                             'frames': frames, 'chunks': 0}
            client = self._client
            self._thread = threading.Thread(target=self._worker, args=(client, body, scene, epoch),
                                            name='paired-research-studio', daemon=True)
            self._thread.start()
        return self.snapshot()

    def _worker(self, client, body, scene, epoch):
        observed = []

        def receive(clip):
            if self._cancelled(epoch):
                raise RuntimeError('Paired generation cancelled')
            validate_chunk(clip, body, len(observed))
            observed.append(clip)
            with self._lock:
                if self._epoch == epoch and self._pending is not None:
                    self._pending['chunks'] = len(observed)

        try:
            health = client.health()
            if not isinstance(health, dict) or health.get('pair_research_enabled') is not True:
                raise RuntimeError('The existing service does not expose paired InterGen research')
            if self._cancelled(epoch):
                return
            chunks = client.wait(body, on_chunk=receive, cancelled=lambda: self._cancelled(epoch))
            if self._cancelled(epoch):
                return
            if len(observed) != len(chunks):
                raise ValueError('Paired service returned inconsistent completed chunks')
            # The HTTP client returns the same validated chunks delivered to the
            # callback. Validate the result again before any visible mutation.
            service_metadata = {}
            try:
                status = client.status(body['request_id'])
                if isinstance(status, dict) and status.get('request_id') == body['request_id']:
                    service_metadata = status.get('metadata') or {}
            except Exception:
                pass  # Model provenance remains in the request and chunk records.
            clip = join_chunks(chunks, body, service_metadata=service_metadata)
            check_scene_geometry(clip, scene)
            with self._lock:
                if self._cancelled(epoch):
                    return
                self._clip, self._scene = clip, scene
                self._frame = 0
                self._playing = True
                self._play_started = self._clock()
                self._revision += 1
                self._failure = None
        except Exception as exc:
            detail = str(exc)
            token = getattr(client, 'token', None)
            if isinstance(token, str) and token:
                detail = detail.replace(token, '[redacted]')
            with self._lock:
                if self._epoch == epoch and not self._closed:
                    self._failure = detail[:500]
        finally:
            with self._lock:
                if self._epoch == epoch:
                    self._pending = None
                self._busy = False

    def timeline_clip(self):
        with self._lock:
            return self._clip

    def frame_pose(self, frame=None):
        with self._lock:
            if self._clip is None:
                return None
            frame = self._frame if frame is None else frame
            if type(frame) is not int or not 0 <= frame < self._clip.frames:
                raise ValueError('Paired frame is outside the clip')
            return self._clip.positions[:, frame], self._clip.rotations[:, frame]

    def play(self):
        with self._lock:
            self._require_active()
            if self._clip is None:
                raise RuntimeError('Generate or load paired research motion first')
            if self._frame == self._clip.frames - 1:
                self._frame = 0
            self._playing = True
            self._play_started = self._clock() - self._frame / 20
        return self.snapshot()

    def pause(self):
        with self._lock:
            self._ensure_open()
            self._tick(self._clock())
            self._playing = False
            self._play_started = None
        return self.snapshot()

    def restart(self):
        with self._lock:
            self._require_active()
            if self._clip is None:
                raise RuntimeError('Generate or load paired research motion first')
            self._frame = 0
            self._playing = True
            self._play_started = self._clock()
        return self.snapshot()

    def seek(self, frame):
        with self._lock:
            self._ensure_open()
            if self._clip is None or type(frame) is not int or not 0 <= frame < self._clip.frames:
                raise ValueError('Paired frame is outside the clip')
            self._frame = frame
            if self._playing:
                self._play_started = self._clock() - frame / 20
        return self.snapshot()

    def _tick(self, now):
        if self._active and self._playing and self._clip is not None:
            self._frame = min(self._clip.frames - 1, max(0, int((now - self._play_started) * 20 + 1e-7)))
            if self._frame == self._clip.frames - 1:
                self._playing = False
                self._play_started = None

    def tick(self, now=None):
        now = self._clock() if now is None else now
        if type(now) not in (int, float) or not math.isfinite(now):
            raise ValueError('Paired playback clock must be finite')
        with self._lock:
            self._tick(now)
        return self.snapshot()

    def snapshot(self):
        with self._lock:
            frames = self._clip.frames if self._clip is not None else 0
            if not self._active:
                phase, status = 'inactive', 'Paired InterGen research is inactive.'
            elif self._busy:
                phase = 'generating' if self._pending is not None else 'cancelling'
                status = ('Generating a complete paired research sample; current motion is preserved.'
                          if self._pending is not None else 'Cancelling paired research; late output will be ignored.')
            elif self._failure:
                phase, status = 'generation_failed', 'Paired research failed: ' + self._failure
            elif self._playing:
                phase, status = 'playing', 'Playing paired InterGen research; contact physics is not verified.'
            elif frames:
                phase, status = 'paused', 'Paired InterGen research paused.'
            else:
                phase, status = 'ready', 'Describe a paired research interaction or open a saved research project.'
            pending = self._pending or {}
            segments = []
            if self._clip is not None:
                request = self._clip.metadata['request']
                segments = [{'start': 0, 'end': frames, 'prompt': request['prompt'], 'source': 'intergen',
                             'kind': 'paired_action', 'request_id': request['request_id'],
                             'metadata': {'research_only': True, 'license': LICENSE}}]
            return {'active': self._active, 'available': self.available, 'initialized': frames > 0,
                    'phase': phase, 'status': status, 'actor_ids': list(ACTOR_IDS), 'total_frames': frames,
                    'frame': self._frame, 'playing': self._playing, 'revision': self._revision,
                    'epoch': self._epoch, 'segments': segments, 'fps': 20, 'research_only': True,
                    'source': 'intergen', 'schema': SCHEMA, 'mode': 'research', 'license': LICENSE,
                    'pending': self._pending is not None, 'busy': self._busy,
                    'pending_chunks': pending.get('chunks', 0), 'requested_frames': pending.get('frames', 0),
                    'request_id': pending.get('request_id'), 'inflight_request_id': pending.get('request_id'),
                    'failure': self._failure, 'generated': frames > 0, 'buffer_frames': max(0, frames - self._frame),
                    'physical_contact_verified': False, 'scene_conditioned': False,
                    'geometry_check': 'sampled scene solids, continuous authored floor support, and 0.35 m torso-centre proxy; hand contact and mesh physics unverified'}

    def save(self):
        with self._lock:
            self._ensure_open()
            if self._clip is None:
                raise ValueError('No complete paired research motion to save')
            return encode_project(self._clip, self._scene, self._frame)

    save_project = save

    def load(self, content):
        clip, scene, frame = decode_project(content)
        with self._lock:
            self._ensure_open()
            self._invalidate()
            self._clip, self._scene, self._frame = clip, scene, frame
            self._active = True
            self._playing = False
            self._play_started = None
            self._failure = None
            self._revision += 1
        return self.snapshot()

    load_project = load

    def close(self, timeout=1.):
        with self._lock:
            if self._closed:
                return
            self._invalidate()
            self._active = False
            self._playing = False
            self._closed = True
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0., min(float(timeout), 2.)))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


StudioPairedSession = PairedStudioSession
