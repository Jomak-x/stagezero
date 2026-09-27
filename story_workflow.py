"""Plan one scene prompt and publish its complete generated take atomically."""

import copy
from collections import deque
import json
import os
from pathlib import Path
import threading
import uuid
from urllib.parse import urlparse

from live_motion import Backend
from story_jobs import StoryJobQueue
from story_planning import StoryPlanner, fit_story_duration, validate_story_seconds
from takes import MAX_TAKES, MAX_TOTAL_FRAMES, validate_take


_backend_locks = {}
_backend_locks_guard = threading.Lock()


class SerializedBackend:
    """Share a pod lane with ordinary action generation without delaying cancel."""

    def __init__(self, backend):
        self.backend = backend
        self.url = getattr(backend, 'url', None)
        self._cancel_lock = threading.Lock()
        self._cancelled = set()
        self._cancel_order = deque()
        key = ('url', str(self.url).rstrip('/')) if self.url is not None else ('object', id(backend))
        with _backend_locks_guard:
            self._lane = _backend_locks.setdefault(key, threading.Lock())

    def generate(self, request_id, prompt, history):
        with self._lane:
            with self._cancel_lock:
                cancelled = request_id in self._cancelled
                self._cancelled.discard(request_id)
            if cancelled:
                raise RuntimeError('Generation cancelled before backend call')
            try:
                return self.backend.generate(request_id, prompt, history)
            finally:
                with self._cancel_lock:
                    self._cancelled.discard(request_id)

    def mark_cancelled(self, request_id):
        with self._cancel_lock:
            if request_id not in self._cancelled:
                self._cancelled.add(request_id)
                self._cancel_order.append(request_id)
                while len(self._cancel_order) > 512:
                    self._cancelled.discard(self._cancel_order.popleft())

    def cancel(self, request_id):
        self.mark_cancelled(request_id)
        cancel = getattr(self.backend, 'cancel', None)
        if callable(cancel):
            return cancel(request_id)


def planning_context(scene):
    """Provide a bounded scene inventory without sending geometry to the model."""
    result = {'name': str(scene.get('name', 'Current scene'))[:100],
              'objects': [], 'targets': []}
    if 'actor_start' in scene:
        result['actor_start'] = scene['actor_start']
    for kind in ('objects', 'targets'):
        for item in scene.get(kind, []):
            if not isinstance(item, dict):
                continue
            summary = {key: item[key] for key in
                       ('id', 'name', 'kind', 'position', 'size', 'object_id') if key in item}
            result[kind].append(summary)
            if len(json.dumps(result, ensure_ascii=False, sort_keys=True)) > 3600:
                result[kind].pop()
                result['inventory_truncated'] = True
                return result
    return result


def configured_backends(primary):
    """Use existing loopback tunnels only; never provision a service from UI."""
    urls = [url.strip() for url in os.environ.get('STAGEZERO_STORY_BACKENDS', '').split(',') if url.strip()]
    if not urls:
        return [primary]
    token = Path(__file__).resolve().parent / '.runtime/api-token'
    backends = []
    for url in urls:
        try:
            parsed = urlparse(url)
            port = parsed.port
        except ValueError as exc:
            raise ValueError('Story backends must be existing HTTP loopback tunnels') from exc
        if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost')
                or not port or parsed.username or parsed.password or parsed.query
                or parsed.fragment or parsed.path not in ('', '/')):
            raise ValueError('Story backends must be existing HTTP loopback tunnels')
        backends.append(SerializedBackend(Backend(token, url.rstrip('/'))))
    return backends


class StoryWorkflow:
    def __init__(self, session, planner=None, queue=None):
        self.session = session
        self.planner = planner
        self.queue = queue
        self.lock = threading.RLock()
        self.jobs = {}
        self.closed = False
        with session.lock:
            if not isinstance(session.backend, SerializedBackend):
                session.backend = SerializedBackend(session.backend)

    def submit(self, prompt, seconds=None):
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 2000:
            raise ValueError('Describe the full scene in 1–2000 characters')
        # Validate before starting a planning thread or initializing a pod lane.
        if seconds is not None:
            validate_story_seconds(seconds)
        with self.session.lock:
            scene = copy.deepcopy(self.session.scene)
            identity = self.session.scene
            revision = self.session.project_revision
        with self.lock:
            if self.closed:
                raise RuntimeError('Story workflow is closed')
            for prior in self.jobs.values():
                if prior['queue_id'] and prior['status'] in ('queued', 'running'):
                    prior['status'] = self.queue.snapshot(prior['queue_id'])['status']
            if sum(job['status'] in ('planning', 'queued', 'running') for job in self.jobs.values()) >= 8:
                raise ValueError('Eight scenes are already pending; wait or cancel one')
            if len(self.jobs) >= 32:
                raise ValueError('Scene history is full; restart the studio after saving your project')
            if self.queue is None:
                self.queue = StoryJobQueue(configured_backends(self.session.backend))
            identifier = str(uuid.uuid4())
            self.jobs[identifier] = {'id': identifier, 'prompt': prompt.strip(), 'seconds': seconds,
                                     'status': 'planning', 'plan': None, 'queue_id': None,
                                     'error': None, 'loaded': False, 'take_id': None,
                                     'scene': scene, 'identity': identity, 'revision': revision}
        threading.Thread(target=self._plan, args=(identifier,), name=f'story-plan-{identifier}', daemon=True).start()
        return identifier

    def _plan(self, identifier):
        with self.lock:
            job = self.jobs[identifier]
            prompt, seconds, scene = job['prompt'], job['seconds'], job['scene']
        try:
            planner = self.planner or StoryPlanner()
            plan = planner.plan(prompt, context=planning_context(scene), seconds=seconds)
            # Apply the same timing and source-request checks to custom planners.
            plan = fit_story_duration(plan, seconds, expected_prompt=prompt)
            with self.lock:
                if job['status'] == 'cancelled' or self.closed:
                    return
                job['queue_id'] = self.queue.submit(plan, scene=scene)
                job['plan'] = plan
                job['status'] = 'queued'
        except Exception as exc:
            with self.lock:
                if job['status'] != 'cancelled':
                    job['status'] = 'failed'
                    job['error'] = (str(exc)[:240] if isinstance(exc, ValueError)
                                    else 'Scene planning failed; check the gateway connection and retry')

    def snapshot(self, identifier):
        with self.lock:
            job = self.jobs[identifier]
            result = {key: copy.deepcopy(value) for key, value in job.items()
                      if key not in ('scene', 'identity')}
            if job['queue_id']:
                queued = self.queue.snapshot(job['queue_id'])
                result.update({key: value for key, value in queued.items() if key != 'id'})
                job['status'] = result['status']
            else:
                result['progress'] = None
            return result

    def cancel(self, identifier):
        with self.lock:
            job = self.jobs[identifier]
            if job['queue_id']:
                return self.queue.cancel(job['queue_id'])
            if job['status'] == 'planning':
                job['status'] = 'cancelled'
                return True
            return False

    def load(self, identifier, automatic=False):
        """Install only a complete take into the exact project scene it belongs to."""
        with self.lock:
            job = self.jobs[identifier]
            if job['loaded']:
                return True
            if job['queue_id'] is None:
                if automatic:
                    return False
                raise ValueError('Wait for this scene to finish planning and generating')
            queued = self.queue.snapshot(job['queue_id'])
            if queued['status'] != 'completed':
                if automatic:
                    return False
                raise ValueError('Wait for this scene to finish generating')
            take = self.queue.result(job['queue_id'])
            validate_take(take)
            session = self.session
            with session.lock:
                if automatic and session.project_revision != job['revision']:
                    return False
                if session.busy:
                    if automatic:
                        return False
                    raise ValueError('Wait for the current action generation before loading this scene')
                if session.scene is not job['identity'] or session.scene != job['scene']:
                    if automatic:
                        return False
                    raise ValueError('The project or background changed; this result belongs to the original scene')
                if (len(session.takes) >= MAX_TAKES or
                        sum(len(item.motion) for item in session.takes.values()) + len(take.motion) > MAX_TOTAL_FRAMES):
                    if automatic:
                        return False
                    raise ValueError('Project motion budget is full; save or remove a take first')
                session._record_gate_events(take)
                session.takes[take.id] = take
                session.mode = 'Live ARDY'
                session._select(take.id, 0)
                session.project_revision += 1
                session.project_status = 'Full scene ready · Save project stores all actions'
                session.status = 'Full scene ready · press Play to review or select an action to edit'
                job['loaded'] = True
                job['take_id'] = take.id
                self.queue.release(job['queue_id'])
                return True

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            for job in self.jobs.values():
                if job['status'] == 'planning':
                    job['status'] = 'cancelled'
            if self.queue is not None:
                self.queue.close()
