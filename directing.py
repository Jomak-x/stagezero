"""One actor, multiple recorded takes. Inference remains in the existing worker."""
from pathlib import Path
import re
import time
import uuid
import numpy as np
from live_motion import MotionSession, validate_result
from takes import Take, encode_project, decode_project, MAX_TAKES, MAX_FRAMES, MAX_TOTAL_FRAMES
from duration_planning import CHUNK_FRAMES, plan_duration


class DirectorSession(MotionSession):
    def __init__(self, *args, **kwargs):
        self.takes = {}
        self.active_take = None
        self.edit_context = None
        self.planned_frames = CHUNK_FRAMES
        self.planned_seconds = CHUNK_FRAMES / 25
        self.duration_label = 'Auto · text-length estimate'
        self.project_status = ''
        self.project_revision = 0
        self.playback_speed = 1.0
        self.loop_playback = False
        self._clock_revision = -1
        self.scene = {'gate': {'position': [0., 0., 1.5], 'radius': .55, 'enabled': True}}
        super().__init__(*args, **kwargs)

    def set_mode(self, mode):
        with self.lock:
            super().set_mode(mode)
            if mode == 'Live ARDY' and self.active_take in self.takes:
                self._select(self.active_take, 0)

    def new_take(self):
        with self.lock:
            if len(self.takes) >= MAX_TAKES:
                self.status = 'Project has 12 takes; save it and start a new project.'
                return
            super().reset()
            self.active_take = None
            self.edit_context = None
            self.status = 'New take · enter an instruction to begin'

    def reset(self):
        """Rewind safely. Unlike New take, this never discards recorded motion."""
        with self.lock:
            self._invalidate()
            self.playing = False
            self.frame = 0
            self.clip_revision += 1
            self.status = 'Rewound · recorded motion preserved'

    def seek(self, frame):
        with self.lock:
            self._invalidate()
            self.playing = False
            self.frame = max(0, min(int(frame), len(self.positions) - 1))
            self.status = ('Paused recorded preview · no inference' if self.mode == 'Recorded preview'
                           else 'Paused at playhead · generating here branches if a future exists')

    def set_playback_speed(self, speed):
        """Change replay rate while keeping the current playhead continuous."""
        speed = float(speed)
        if not np.isfinite(speed) or not .25 <= speed <= 2.0:
            raise ValueError('Playback speed must be between 0.25 and 2.0')
        with self.lock:
            if self.playing:
                self.tick()
            self.playback_speed = speed
            if self.playing:
                self.started = time.perf_counter() - self.frame / (self.fps * speed)
            self._clock_revision = self.clip_revision

    def set_loop(self, enabled):
        with self.lock:
            self.loop_playback = bool(enabled)

    def tick(self):
        with self.lock:
            if self.playing:
                now = time.perf_counter()
                if self._clock_revision != self.clip_revision:
                    # MotionSession starts fresh generated clips at native rate.
                    self.started = now - self.frame / (self.fps * self.playback_speed)
                    self._clock_revision = self.clip_revision
                elapsed_frames = max(0, int((now - self.started) * self.fps * self.playback_speed))
                length = len(self.positions)
                if self.loop_playback and length > 1:
                    self.frame = elapsed_frames % length
                else:
                    self.frame = min(elapsed_frames, length - 1)
                    if self.frame == length - 1:
                        self.playing = False
            return self.clip_revision, self.frame

    def play(self):
        with self.lock:
            super().play()
            if not self.busy:
                if self.playing:
                    self.started = time.perf_counter() - self.frame / (self.fps * self.playback_speed)
                    self._clock_revision = self.clip_revision
                self.status = 'Replaying stored motion · no inference'

    def rename_active_take(self, name):
        with self.lock:
            t = self.takes.get(self.active_take) if self.mode == 'Live ARDY' else None
            if t is None:
                self.status = 'Select a take to rename'
                return False
            name = str(name).strip()
            if not 1 <= len(name) <= 80:
                self.status = 'Take name must be 1–80 characters'
                return False
            if any(other.id != t.id and other.name == name for other in self.takes.values()):
                self.status = 'A take already has that name'
                return False
            if t.name != name:
                t.name = name
                self.project_revision += 1
                self.project_status = 'Unsaved changes · Save project stores every take'
            self.status = f'Renamed take · {name}'
            return True

    def _copy_name(self, source, suffix):
        base = source.name[:80 - len(suffix)].rstrip()
        candidate = base + suffix
        names = {t.name for t in self.takes.values()}
        number = 2
        while candidate in names:
            indexed = f'{suffix} {number}'
            candidate = source.name[:80 - len(indexed)].rstrip() + indexed
            number += 1
        return candidate

    def _new_take_name(self, alternate=False):
        number = len(self.takes) + 1
        names = {t.name for t in self.takes.values()}
        while True:
            name = f'Take {number}' + (' · alternate' if alternate else '')
            if name not in names:
                return name
            number += 1

    def _can_add_take(self, length):
        if len(self.takes) >= MAX_TAKES:
            self.status = 'Take limit reached; save the project before starting another.'
            return False
        if sum(len(t.positions) for t in self.takes.values()) + length > MAX_TOTAL_FRAMES:
            self.status = 'Motion budget reached; save this project before continuing.'
            return False
        return True

    def _add_edited_take(self, source, stop, suffix):
        segments, events = source.prefix(stop)
        copied = [getattr(source, key)[:stop].copy() for key in ('positions', 'rotations', 'motion')]
        new = Take(str(uuid.uuid4()), self._copy_name(source, suffix), *copied,
                   segments=segments, parent=source.id, branch_frame=stop - 1, events=events)
        self.takes[new.id] = new
        self._select(new.id, min(self.frame, stop - 1))
        self.project_revision += 1
        self.project_status = 'Unsaved changes · Save project stores every take'
        return new

    def duplicate_active_take(self):
        with self.lock:
            source = self.takes.get(self.active_take) if self.mode == 'Live ARDY' else None
            if source is None:
                self.status = 'Select a take to duplicate'
                return None
            if not self._can_add_take(len(source.positions)):
                return None
            new = self._add_edited_take(source, len(source.positions), ' copy')
            self.status = f'Duplicated take · {new.name}'
            return new

    def trim_after_playhead(self):
        """Create an alternate ending at the playhead; keep the source intact."""
        with self.lock:
            source = self.takes.get(self.active_take) if self.mode == 'Live ARDY' else None
            if source is None:
                self.status = 'Select a take to trim'
                return None
            stop = self.frame + 1
            if stop < 4:
                self.status = 'Move the playhead to at least 0.12 s to trim'
                return None
            if stop >= len(source.positions):
                self.status = 'Playhead is already at the end of this take'
                return None
            if not self._can_add_take(stop):
                return None
            new = self._add_edited_take(source, stop, ' · trimmed')
            self.status = f'Trimmed alternate · original preserved as {source.name}'
            return new

    def _jump_action(self, direction):
        with self.lock:
            take = self.takes.get(self.active_take) if self.mode == 'Live ARDY' else None
            if take is None or not take.segments:
                return False
            starts = [s['start'] for s in take.segments]
            if direction < 0:
                targets = [start for start in starts if start < self.frame]
                target = targets[-1] if targets else starts[0]
            else:
                targets = [start for start in starts if start > self.frame]
                target = targets[0] if targets else starts[-1]
            self.seek(target)
            return True

    def previous_action(self):
        return self._jump_action(-1)

    def next_action(self):
        return self._jump_action(1)

    def _select(self, take_id, frame=0):
        t = self.takes[take_id]
        self._invalidate()
        self.active_take = take_id
        self.positions, self.rotations, self.motion = t.positions, t.rotations, t.motion
        self.kind, self.fps = 'generated', 25
        self.frame = min(frame, len(t.positions) - 1)
        self.playing = False
        self.metrics = None
        self.clip_revision += 1
        self.status = 'Recorded generated take · replay uses stored motion, no inference'

    def select_take(self, take_id):
        with self.lock:
            if take_id in self.takes and self.mode == 'Live ARDY':
                self._select(take_id)

    def submit(self, prompt, seconds=None, edit_mode=None, at_frame=None):
        with self.lock:
            if self.mode != 'Live ARDY':
                return
            prompt = str(prompt).strip()
            if not 1 <= len(prompt) <= 500:
                self.status = 'Enter a movement instruction (1–500 characters)'
                return
            try:
                plan = plan_duration(prompt, seconds)
            except ValueError as exc:
                self.status = str(exc)
                return
            t = self.takes.get(self.active_take)
            if edit_mode is None:
                stop = self.frame + 1 if t is not None else 0
            elif edit_mode == 'new':
                t, stop = None, 0
            elif edit_mode == 'extend':
                if t is None:
                    self.status = 'Select a take to extend'
                    return
                stop = len(t.positions)
            elif edit_mode == 'replace':
                if t is None:
                    self.status = 'Select a take to replace its ending'
                    return
                if type(at_frame) is not int or not 0 <= at_frame <= len(t.positions):
                    self.status = 'Replacement time must be within the selected take'
                    return
                stop = at_frame
            else:
                self.status = 'Choose New, Extend, or Replace'
                return
            branch = t is not None and stop < len(t.positions)
            if t is not None and 0 < stop < 4:
                self.status = 'Move the playhead to at least 0.12 s for continuation, or choose New take.'
                return
            if (t is None or branch) and len(self.takes) >= MAX_TAKES:
                self.status = 'Take limit reached; save the project before starting another.'
                return
            total = sum(len(x.positions) for x in self.takes.values())
            growth = stop + plan.frames if t is None or branch else plan.frames
            if stop + plan.frames > MAX_FRAMES or total + growth > MAX_TOTAL_FRAMES:
                self.status = 'Motion budget reached; save this project before continuing.'
                return
            super().submit(prompt)
            if self.busy:
                self.edit_context = (t, stop, branch)
                self.planned_frames = plan.frames
                self.planned_seconds = plan.seconds
                self.duration_label = plan.label
                version, request_id, submitted_prompt, _, submitted = self.pending
                history = None
                if t is not None and stop:
                    count = min(52, stop) // 4 * 4
                    history = t.motion[stop - count:stop].copy() if count else None
                self.pending = (version, request_id, submitted_prompt, history, submitted, plan.frames)
                action = 'alternate ending · original preserved' if branch else 'new take' if t is None else 'extension'
                self.status = f'Generating {plan.seconds:.2f} s {action} · 0/{(plan.frames + CHUNK_FRAMES - 1) // CHUNK_FRAMES} chunks'

    def _work(self):
        """Build every backend chunk privately; install only a complete take."""
        while True:
            self.wake.wait()
            with self.lock:
                job = self.pending
                self.pending = None
                self.wake.clear()
            if job is None:
                continue
            version, request_id, prompt, history, submitted, frames = job
            count = (frames + CHUNK_FRAMES - 1) // CHUNK_FRAMES
            parts = []
            generation_seconds = 0.0
            try:
                for index in range(count):
                    with self.lock:
                        if version != self.version or request_id != self.current_id:
                            break
                    result = self.backend.generate(request_id, prompt, history)
                    validate_result(result, request_id)
                    parts.append(result)
                    generation_seconds += float(result['metadata']['generation_seconds'])
                    with self.lock:
                        if version != self.version or request_id != self.current_id:
                            break
                        if index + 1 < count:
                            history = result['motion'][-52:].copy()
                            request_id = str(uuid.uuid4())
                            self.current_id = request_id
                            self.status = f'Generating {frames / 25:.2f} s · {index + 1}/{count} chunks received; holding pose'
                else:
                    trim = frames - (count - 1) * CHUNK_FRAMES
                    assembled = {key: np.concatenate([part[key] if i < count - 1 else part[key][:trim]
                                                       for i, part in enumerate(parts)], axis=0)
                                 for key in ('positions', 'rotations', 'motion')}
                    assembled['metadata'] = dict(parts[-1]['metadata'], generation_seconds=generation_seconds)
                    with self.lock:
                        if version != self.version or request_id != self.current_id:
                            continue
                        first_frame = self._install_result(assembled)
                        self.fps = 25
                        self.frame = first_frame
                        self.kind = 'generated'
                        self.clip_revision += 1
                        self.busy = False
                        self.status = f'Generated {frames / 25:.2f} s · {count} complete chunk' + ('s' if count != 1 else '')
                        self.metrics = {**assembled['metadata'], 'command_to_received_seconds': time.perf_counter() - submitted}
                        self.needs_ack = (request_id, submitted)
                        self.started = time.perf_counter() - self.frame / self.fps
                        self.playing = self.resume_after_generation
            except Exception as exc:
                with self.lock:
                    if version != self.version:
                        continue
                    self.busy = False
                    self.playing = False
                    self.status = f'Generation failed · {type(exc).__name__}: {str(exc)[:200]}. Original take preserved; retry.'

    def _install_result(self, result):
        t, stop, branch = self.edit_context
        arrays = [result[k] for k in ('positions', 'rotations', 'motion')]
        segments, events = ([], []) if t is None else t.prefix(stop)
        if t is not None:
            arrays = [np.concatenate([getattr(t, k)[:stop], a], axis=0) for k, a in zip(('positions', 'rotations', 'motion'), arrays)]
        meta = result['metadata']
        segments.append(dict(start=stop, end=stop + len(result['motion']), prompt=self.prompt,
                             request_id=meta['request_id'], generation_seconds=meta['generation_seconds']))
        take_id = str(uuid.uuid4()) if t is None or branch else t.id
        name = self._new_take_name(branch) if t is None or branch else t.name
        new = Take(take_id, name, *arrays, segments=segments,
                   parent=t.id if branch and stop else (t.parent if t and not branch else None),
                   branch_frame=stop - 1 if branch and stop else (t.branch_frame if t and not branch else None), events=events)
        self.takes[take_id] = new
        self.active_take = take_id
        self.positions, self.rotations, self.motion = arrays
        self._record_gate_events(new)
        self.project_revision += 1
        self.project_status = 'Unsaved changes · Save project stores every take'
        return stop

    def _record_gate_events(self, take):
        gate = self.scene['gate']
        if not gate['enabled'] or any(e['type'] == 'gate_open' for e in take.events):
            return
        center = np.asarray(gate['position'])[[0, 2]]
        distances = np.linalg.norm(take.positions[:, 0][:, [0, 2]] - center, axis=1)
        hits = np.flatnonzero(distances <= gate['radius'])
        if len(hits):
            take.events.append({'type': 'gate_open', 'frame': int(hits[0])})

    def gate_open(self):
        t = self.takes.get(self.active_take)
        return self.mode == 'Live ARDY' and t is not None and any(e['type'] == 'gate_open' and e['frame'] <= self.frame for e in t.events)

    def current_action(self):
        t = self.takes.get(self.active_take)
        if self.mode != 'Live ARDY' or t is None:
            return ''
        return next((s['prompt'] for s in t.segments if s['start'] <= self.frame < s['end']), '')

    def save_project(self, directory, name):
        with self.lock:
            if not self.takes:
                raise ValueError('Generate a take before saving')
            active = self.active_take or next(iter(self.takes))
            frame = self.frame if self.mode == 'Live ARDY' and self.active_take else 0
            data = encode_project(self.takes, active, frame, self.scene)
            saved_revision = self.project_revision
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        name = re.sub(r'[^a-zA-Z0-9_-]+', '-', name).strip('-')[:50] or 'performance'
        path = folder / f'{name}-{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:6]}.stagezero.npz'
        temp = path.with_suffix('.tmp')
        try:
            temp.write_bytes(data)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        with self.lock:
            if self.project_revision == saved_revision:
                self.project_status = f'Saved {path.name} · all takes and motion included'
            else:
                self.project_status = f'Saved snapshot {path.name} · newer changes remain unsaved'
        return path, data

    def new_project(self, directory):
        with self.lock:
            backup = self.save_project(directory, 'automatic-backup')[0] if self.takes else None
            self._invalidate()
            self.takes = {}
            self.project_revision += 1
            self.active_take = None
            self.mode = 'Live ARDY'
            super().reset()
            self.scene = {'gate': {'position': [0., 0., 1.5], 'radius': .55, 'enabled': True}}
            self.project_status = f'Previous project backed up: {backup.name}' if backup else 'New project'

    def load_project(self, data):
        takes, active, frame, scene = decode_project(data)
        gate = scene.get('gate')
        if not isinstance(gate, dict) or not isinstance(gate.get('enabled'), bool):
            raise ValueError('Invalid gate state')
        p = np.asarray(gate.get('position'), dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all() or np.abs(p).max() > 100 or not isinstance(gate.get('radius'), (int, float)) or not .1 <= gate['radius'] <= 3:
            raise ValueError('Invalid gate geometry')
        with self.lock:
            self._invalidate()
            self.takes, self.scene = takes, scene
            self.project_revision += 1
            self.mode = 'Live ARDY'
            self._select(active, frame)
            self.project_status = 'Loaded stored motion · no regeneration'
