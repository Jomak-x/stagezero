"""Atomic composed-cast publication, exact 30 fps transport, and capture leases."""
from __future__ import annotations

import math
import threading
import time
import uuid

from cast_performance import (CastPerformance, FPS, SCHEMA, cast_from_performance,
                              decode_project, encode_project)
from paired_scene import EMPTY_SCENE, json_copy, scene_copy


class CastPerformanceSession:
    def __init__(self, *, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.RLock()
        self._active = self._closed = self._busy = self._playing = self._capturing = False
        self._clip = None
        self._cast = []
        self._scene = scene_copy(EMPTY_SCENE)
        self._frame = self._revision = self._transport_revision = self._epoch = 0
        self._play_started = None
        self._pending = self._failure = self._progress = self._thread = self._last_build = None

    @property
    def active(self):
        with self._lock:
            return self._active

    @property
    def available(self):
        with self._lock:
            return not self._closed

    @property
    def revision(self):
        with self._lock:
            return self._revision

    @property
    def scene_document(self):
        with self._lock:
            return json_copy(self._scene)

    def _open(self):
        if self._closed:
            raise RuntimeError('Cast performance session is closed')
        if self._capturing:
            raise RuntimeError('Playback export is running; wait before changing the performance')

    def _require_active(self):
        self._open()
        if not self._active:
            raise RuntimeError('Activate the cast performance first')

    def _invalidate(self, *, clear_retry=False):
        self._epoch += 1
        self._pending = self._progress = None
        if clear_retry:
            self._last_build = None

    def activate(self, active=True):
        if type(active) is not bool:
            raise ValueError('Cast activation must be a boolean')
        if not active:
            return self.deactivate()
        with self._lock:
            self._open()
            self._active = True
        return self.snapshot()

    set_active = activate

    def deactivate(self):
        with self._lock:
            self._open()
            self._invalidate()
            self._active = self._playing = False
            self._play_started = None
            self._transport_revision += 1
        return self.snapshot()

    stop = deactivate

    def update_scene(self, scene_document):
        scene = scene_copy(scene_document)
        with self._lock:
            self._open()
            if scene != self._scene:
                self._invalidate(clear_retry=True)
                self._scene = scene
                self._revision += 1
        return self.snapshot()

    def build_performance(self, builder, scene_document=None, *, request=None):
        """Run builder(scene, cancelled=..., on_progress=...) on one owned thread."""
        if not callable(builder):
            raise ValueError('Cast performance builder must be callable')
        request = json_copy(request)
        with self._lock:
            self._open()
            if self._busy:
                raise RuntimeError('A cast performance request is still running or cancelling')
            scene = scene_copy(self._scene if scene_document is None else scene_document)
            self._invalidate()
            epoch = self._epoch
            self._last_build = (builder, scene, request)
            self._pending = {'operation': 'performance', 'request_id': 'cast-' + uuid.uuid4().hex,
                             'request': request}
            self._active = self._busy = True
            self._failure = None
            self._thread = threading.Thread(target=self._build_worker,
                args=(builder, scene, epoch), daemon=True, name='cast-performance-builder')
            self._thread.start()
        return self.snapshot()

    def _cancelled(self, epoch):
        with self._lock:
            return self._closed or not self._active or self._epoch != epoch

    def _build_worker(self, builder, scene, epoch):
        def progress(value):
            value = json_copy(value, limit=8192)
            with self._lock:
                if not self._cancelled(epoch):
                    self._progress = value
        try:
            clip = builder(json_copy(scene), cancelled=lambda: self._cancelled(epoch),
                           on_progress=progress)
            if not isinstance(clip, CastPerformance):
                raise ValueError('Builder must return a complete CastPerformance')
            cast = cast_from_performance(clip)
            with self._lock:
                if not self._cancelled(epoch):
                    self._clip, self._cast, self._scene = clip, cast, scene
                    self._frame = 0
                    self._playing = True
                    self._play_started = self._clock()
                    self._revision += 1
                    self._transport_revision += 1
        except Exception as exc:
            detail = str(exc)
            token = getattr(builder, 'token', None)
            if isinstance(token, str) and token:
                detail = detail.replace(token, '[redacted]')
            with self._lock:
                if self._epoch == epoch and not self._closed:
                    self._failure = detail[:500]
        finally:
            with self._lock:
                if self._epoch == epoch:
                    self._pending = self._progress = None
                self._busy = False

    def cancel(self):
        with self._lock:
            self._open()
            self._invalidate()
            self._failure = None
        return self.snapshot()

    def retry(self):
        with self._lock:
            self._open()
            if self._last_build is None:
                raise ValueError('No cast performance request to retry')
            builder, scene, request = self._last_build
            return self.build_performance(builder, scene_document=scene, request=request)

    def load_performance(self, clip, scene_document=None):
        if not isinstance(clip, CastPerformance):
            raise ValueError('Expected a separate CastPerformance')
        cast = cast_from_performance(clip)
        with self._lock:
            self._open()
            scene = scene_copy(self._scene if scene_document is None else scene_document)
            self._publish_loaded(clip, cast, scene, 0)
        return self.snapshot()

    def _publish_loaded(self, clip, cast, scene, frame):
        self._invalidate(clear_retry=True)
        self._clip, self._cast, self._scene, self._frame = clip, cast, scene, frame
        self._active = True
        self._playing = False
        self._failure = self._play_started = None
        self._revision += 1
        self._transport_revision += 1

    def timeline_clip(self):
        with self._lock:
            return self._clip

    def frame_pose(self, frame=None):
        with self._lock:
            if self._clip is None:
                return None
            index = self._frame if frame is None else frame
            if type(index) is not int or not 0 <= index < self._clip.frames:
                raise ValueError('Cast frame is outside the performance')
            return self._clip.joints[index]

    def play(self):
        with self._lock:
            self._require_active()
            if self._clip is None:
                raise ValueError('Build or load a cast performance first')
            if self._frame == self._clip.frames - 1:
                self._frame = 0
            self._playing = True
            self._play_started = self._clock() - self._frame / FPS
            self._transport_revision += 1
        return self.snapshot()

    def pause(self):
        with self._lock:
            self._open()
            self._tick(self._clock())
            self._playing = False
            self._play_started = None
            self._transport_revision += 1
        return self.snapshot()

    def restart(self):
        self.seek(0)
        return self.play()

    def seek(self, frame):
        with self._lock:
            self._open()
            if self._clip is None or type(frame) is not int or not 0 <= frame < self._clip.frames:
                raise ValueError('Cast frame is outside the performance')
            self._frame = frame
            if self._playing:
                self._play_started = self._clock() - frame / FPS
            self._transport_revision += 1
        return self.snapshot()

    def _tick(self, now):
        if not self._capturing and self._active and self._playing and self._clip is not None:
            self._frame = min(self._clip.frames - 1, max(0, int((now - self._play_started) * FPS + 1e-7)))
            if self._frame == self._clip.frames - 1:
                self._playing = False
                self._play_started = None

    def tick(self, now=None):
        now = self._clock() if now is None else now
        if type(now) not in (int, float) or not math.isfinite(now):
            raise ValueError('Cast playback clock must be finite')
        with self._lock:
            self._tick(now)
            return self.snapshot()

    def begin_capture(self):
        with self._lock:
            self._require_active()
            if self._busy or self._clip is None:
                raise ValueError('Finish building and load a complete performance before export')
            self._tick(self._clock())
            self._playing = False
            self._play_started = None
            self._capturing = True
            self._transport_revision += 1
        return self.snapshot()

    def capture_seek(self, frame):
        with self._lock:
            if not self._capturing or self._closed:
                raise RuntimeError('No active cast performance export')
            if type(frame) is not int or not 0 <= frame < self._clip.frames:
                raise ValueError('Cast capture frame is outside the performance')
            self._frame = frame
            self._transport_revision += 1
        return self.snapshot()

    def end_capture(self, frame=0):
        with self._lock:
            if not self._capturing:
                return self.snapshot()
            if type(frame) is not int:
                raise ValueError('Cast capture playhead must be an integer')
            self._capturing = False
            self._frame = max(0, min(frame, self._clip.frames - 1))
            self._playing = False
            self._play_started = None
            self._transport_revision += 1
        return self.snapshot()

    def save(self):
        with self._lock:
            if self._closed:
                raise RuntimeError('Cast performance session is closed')
            return encode_project(self._clip, self._cast, self._scene, self._frame)

    save_project = save

    def load(self, content):
        clip, cast, scene, frame = decode_project(content)
        with self._lock:
            self._open()
            self._publish_loaded(clip, cast, scene, frame)
        return self.snapshot()

    load_project = load

    def snapshot(self):
        with self._lock:
            clip = self._clip
            frames = 0 if clip is None else clip.frames
            phase = ('inactive' if not self._active else
                     'generating' if self._busy and self._pending else 'cancelling' if self._busy else
                     'generation_failed' if self._failure else 'playing' if self._playing else
                     'paused' if frames else 'ready')
            statuses = {'inactive': 'Cast performance inactive.',
                'generating': 'Building the cast performance; current take retained.',
                'cancelling': 'Cancelling; late output will be ignored.',
                'generation_failed': f'Performance failed: {self._failure}',
                'playing': 'Playing the cast performance at 30 fps.',
                'paused': 'Cast performance paused.', 'ready': 'Describe a performance for one to three actors.'}
            metadata = {} if clip is None else clip.metadata
            segments = None if clip is None else clip.segments
            if segments is None:
                segments = [] if clip is None else [{'start': 0, 'end': frames,
                    'prompt': metadata.get('prompt', 'Cast performance'), 'source': clip.source, 'kind': 'performance'}]
            pending = self._pending or {}
            ids = [] if clip is None else list(clip.actor_ids)
            return {'active': self._active, 'available': not self._closed, 'initialized': bool(frames),
                'busy': self._busy, 'pending': self._pending is not None, 'capturing': self._capturing,
                'phase': phase, 'status': statuses[phase], 'failure': self._failure,
                'progress': json_copy(self._progress), 'request_id': pending.get('request_id'),
                'pending_operation': pending.get('operation'), 'frame': self._frame,
                'total_frames': frames, 'fps': FPS, 'playing': self._playing,
                'transport_revision': self._transport_revision, 'revision': self._revision, 'epoch': self._epoch,
                'cast': json_copy(self._cast), 'actor_ids': ids,
                'selected_pair': ids[:2] if len(ids) >= 2 else [],
                'placement': {'x': 0., 'z': 0., 'yaw_degrees': 0.},
                'source': 'Composed cast performance' if clip is None else clip.source,
                'schema': SCHEMA, 'generated': bool(frames), 'composed': True,
                'buffer_frames': max(0, frames - self._frame), 'segments': segments,
                'hand_pose': metadata.get('render_hand_pose', 'relaxed'),
                'physical_contact_verified': False, 'native_features_available': False}

    def close(self, timeout=1.):
        with self._lock:
            self._invalidate(clear_retry=True)
            self._active = self._playing = self._capturing = False
            self._closed = True
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(max(0., min(float(timeout), 2.)))
