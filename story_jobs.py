"""Bounded asynchronous queue for complete, editable G1 scene takes.

Each backend tunnel owns one worker and one whole scene at a time. Chunks from
one scene retain their motion history in order, including across action beats.
"""

from collections import deque
import copy
from dataclasses import dataclass
import math
import re
import threading
import uuid

import numpy as np

from duration_planning import CHUNK_FRAMES, FPS
from live_motion import validate_result
from story_planning import validate_story_plan
from takes import MAX_FRAMES, Take, validate_take


_QUALITY_FAILURE = re.compile(
    r'^(?:RuntimeError: )?Generated candidates failed motion-quality checks(?: \(([a-z_]+)\))?'
    r'(?:; no motion committed\. Retry the instruction\.)?$', re.I)


class _MotionQualityFailure(RuntimeError):
    def __init__(self, reason):
        self.reason = reason if reason in ('intra_clip_jump', 'foot_slide', 'joint_jump') else None
        super().__init__('Motion quality checks rejected the generated movement')


@dataclass
class _Job:
    id: str
    plan: dict
    total_frames: int
    total_chunks: int
    status: str = 'queued'
    completed_frames: int = 0
    completed_chunks: int = 0
    completed_beats: int = 0
    current_beat: int | None = None
    request_id: str | None = None
    backend: object = None
    error: str | None = None
    take: Take | None = None
    released: bool = False


class StoryJobQueue:
    def __init__(self, backends, max_pending=8, max_history=32):
        if isinstance(backends, dict):
            backends = backends.values()
        lanes = {}
        for backend in backends:
            if not callable(getattr(backend, 'generate', None)):
                raise TypeError('Each backend must provide generate')
            url = getattr(backend, 'url', None)
            key = ('url', str(url).rstrip('/')) if url is not None else ('object', id(backend))
            lanes.setdefault(key, backend)
        if not lanes or type(max_pending) is not int or max_pending < 1:
            raise ValueError('Provide a backend and a positive queue capacity')
        if type(max_history) is not int or max_history < 1:
            raise ValueError('Job history capacity must be positive')
        self.max_pending = max_pending
        self.max_history = max_history
        self._jobs = {}
        self._pending = deque()
        self._closed = False
        self._condition = threading.Condition()
        self._workers = [threading.Thread(target=self._work, args=(backend,),
                         name=f'story-job-{index}', daemon=True)
                         for index, backend in enumerate(lanes.values())]
        for worker in self._workers:
            worker.start()

    def submit(self, plan, scene=None):
        canonical = validate_story_plan(plan)
        frames = [round(beat['seconds'] * FPS) for beat in canonical['beats']]
        total = sum(frames)
        if total > min(MAX_FRAMES, 120 * FPS):
            raise ValueError('Story exceeds the 120-second take length')
        chunks = sum(math.ceil(count / CHUNK_FRAMES) for count in frames)
        identifier = str(uuid.uuid4())
        job = _Job(identifier, canonical, total, chunks)
        with self._condition:
            if self._closed:
                raise RuntimeError('Story queue is closed')
            if len(self._pending) >= self.max_pending:
                raise ValueError('Story queue is full; wait for a queued scene to start')
            while len(self._jobs) >= self.max_history:
                oldest = next((key for key, prior in self._jobs.items()
                               if prior.status in ('completed', 'failed', 'cancelled')
                               and prior.backend is None), None)
                if oldest is None:
                    raise ValueError('Story history is full; wait for an active scene')
                del self._jobs[oldest]
            self._jobs[identifier] = job
            self._pending.append(identifier)
            self._condition.notify()
        return identifier

    def snapshot(self, identifier):
        with self._condition:
            job = self._jobs[identifier]
            return {'id': job.id, 'status': job.status, 'error': job.error,
                    'title': job.plan['title'] if job.plan else None,
                    'result_available': job.take is not None and not job.released,
                    'progress': {'completed_beats': job.completed_beats,
                                 'total_beats': len(job.plan['beats']) if job.plan else 0,
                                 'completed_chunks': job.completed_chunks,
                                 'total_chunks': job.total_chunks,
                                 'completed_frames': job.completed_frames,
                                 'total_frames': job.total_frames,
                                 'current_beat': job.current_beat,
                                 'fraction': job.completed_frames / job.total_frames}}

    def result(self, identifier):
        with self._condition:
            job = self._jobs[identifier]
            if job.status != 'completed' or job.take is None or job.released:
                raise RuntimeError(f'Story result is {job.status}')
            return copy.deepcopy(job.take)

    def release(self, identifier):
        with self._condition:
            job = self._jobs[identifier]
            if job.status != 'completed' or job.take is None:
                return False
            job.take = None
            job.released = True
            return True

    def cancel(self, identifier):
        with self._condition:
            job = self._jobs[identifier]
            if job.status not in ('queued', 'running'):
                return False
            job.status = 'cancelled'
            if identifier in self._pending:
                self._pending.remove(identifier)
            backend, request_id = job.backend, job.request_id
            self._condition.notify_all()
        if backend is not None and request_id is not None:
            mark = getattr(backend, 'mark_cancelled', None)
            if callable(mark):
                mark(request_id)
            cancel = getattr(backend, 'cancel', None)
            if callable(cancel):
                threading.Thread(target=self._safe_cancel, args=(cancel, request_id), daemon=True).start()
        return True

    @staticmethod
    def _safe_cancel(cancel, request_id):
        try:
            cancel(request_id)
        except Exception:
            pass

    def close(self):
        with self._condition:
            if self._closed:
                return
            self._closed = True
            ids = [job.id for job in self._jobs.values() if job.status in ('queued', 'running')]
        for identifier in ids:
            self.cancel(identifier)
        with self._condition:
            self._condition.notify_all()

    def _work(self, backend):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or bool(self._pending))
                if self._closed:
                    return
                job = self._jobs[self._pending.popleft()]
                if job.status != 'queued':
                    continue
                job.status = 'running'
                job.backend = backend
            try:
                take = self._generate(job, backend)
                with self._condition:
                    if job.status == 'running':
                        job.take = take
                        job.status = 'completed'
                        job.current_beat = None
            except Exception as exc:
                # A backend error may contain request bodies or credentials.
                with self._condition:
                    if job.status == 'running':
                        job.status = 'failed'
                        number = (job.current_beat or 0) + 1
                        if isinstance(exc, _MotionQualityFailure):
                            detail = ('motion discontinuity' if exc.reason == 'intra_clip_jump'
                                      else 'motion quality')
                            job.error = (f'Action {number} failed {detail} checks after 3 attempts. '
                                         'Try a simpler movement description and retry.')
                        else:
                            job.error = f'Generation failed on action {number}; check the backend and retry'
                        job.current_beat = None
            finally:
                with self._condition:
                    job.request_id = None
                    job.backend = None
                    self._condition.notify_all()

    def _generate(self, job, backend):
        arrays = {key: [] for key in ('positions', 'rotations', 'motion')}
        segments = []
        offset = 0
        for beat_index, beat in enumerate(job.plan['beats']):
            frames = round(beat['seconds'] * FPS)
            with self._condition:
                if job.status != 'running':
                    return None
                job.current_beat = beat_index
            generation_seconds = 0.0
            remaining = frames
            while remaining:
                history = self._history(arrays['motion'])
                for attempt in range(3):
                    with self._condition:
                        if job.status != 'running':
                            return None
                        request_id = str(uuid.uuid4())
                        job.request_id = request_id
                    try:
                        result = backend.generate(request_id, beat['prompt'], history)
                        validate_result(result, request_id)
                        break
                    except RuntimeError as exc:
                        match = _QUALITY_FAILURE.fullmatch(str(exc))
                        if match is None:
                            raise
                        with self._condition:
                            if job.status != 'running':
                                return None
                        if attempt == 2:
                            raise _MotionQualityFailure(match.group(1)) from None
                used = min(CHUNK_FRAMES, remaining)
                with self._condition:
                    if job.status != 'running':
                        return None
                    for key in arrays:
                        arrays[key].append(np.array(result[key][:used], copy=True))
                    generation_seconds += float(result['metadata'].get('generation_seconds', 0))
                    remaining -= used
                    job.request_id = None
                    job.completed_chunks += 1
                    job.completed_frames += used
            segments.append({'start': offset, 'end': offset + frames,
                             'prompt': beat['prompt'], 'beat_id': beat['id'],
                             'generation_seconds': generation_seconds})
            offset += frames
            with self._condition:
                if job.status != 'running':
                    return None
                job.completed_beats += 1
        take = Take(job.id, job.plan['title'],
                    *(np.concatenate(arrays[key], axis=0) for key in ('positions', 'rotations', 'motion')),
                    segments=segments)
        validate_take(take)
        return take

    @staticmethod
    def _history(motion_parts):
        if not motion_parts:
            return None
        parts = []
        count = 0
        for part in reversed(motion_parts):
            parts.append(part)
            count += len(part)
            if count >= 52:
                break
        history = np.concatenate(parts[::-1], axis=0)[-52:]
        count = len(history) // 4 * 4
        return history[-count:].copy() if count else None
