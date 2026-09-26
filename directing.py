"""One actor, multiple recorded takes. Inference remains in the existing worker."""
from pathlib import Path
import re
import time
import uuid
import numpy as np
from live_motion import MotionSession
from takes import Take, encode_project, decode_project, MAX_TAKES, MAX_FRAMES, MAX_TOTAL_FRAMES


class DirectorSession(MotionSession):
    def __init__(self, *args, **kwargs):
        self.takes = {}
        self.active_take = None
        self.edit_context = None
        self.project_status = ''
        self.project_revision = 0
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
            self.status = 'Paused at playhead · generating here branches if a future exists'

    def play(self):
        with self.lock:
            super().play()
            if not self.busy:
                self.status = 'Replaying stored motion · no inference'

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

    def submit(self, prompt):
        with self.lock:
            if self.mode != 'Live ARDY':
                return
            t = self.takes.get(self.active_take)
            stop = self.frame + 1 if t is not None else 0
            branch = t is not None and stop < len(t.positions)
            if t is not None and stop < 4:
                self.status = 'Move the playhead to at least 0.12 s for continuation, or choose New take.'
                return
            if (t is None or branch) and len(self.takes) >= MAX_TAKES:
                self.status = 'Take limit reached; save the project before starting another.'
                return
            total = sum(len(x.positions) for x in self.takes.values())
            growth = stop + 104 if t is None or branch else 104
            if stop + 104 > MAX_FRAMES or total + growth > MAX_TOTAL_FRAMES:
                self.status = 'Motion budget reached; save this project before continuing.'
                return
            super().submit(prompt)
            if self.busy:
                self.edit_context = (t, stop, branch)
                self.status = 'Generating an alternate ending · original preserved' if branch else 'Generating and recording next 4.16 s · holding pose'

    def _install_result(self, result):
        t, stop, branch = self.edit_context
        arrays = [result[k] for k in ('positions', 'rotations', 'motion')]
        segments, events = ([], []) if t is None else t.prefix(stop)
        if t is not None:
            arrays = [np.concatenate([getattr(t, k)[:stop], a], axis=0) for k, a in zip(('positions', 'rotations', 'motion'), arrays)]
        meta = result['metadata']
        segments.append(dict(start=stop, end=stop + 104, prompt=self.prompt,
                             request_id=meta['request_id'], generation_seconds=meta['generation_seconds']))
        take_id = str(uuid.uuid4()) if t is None or branch else t.id
        name = f'Take {len(self.takes) + 1}' if t is None else (f'Take {len(self.takes) + 1} · alternate' if branch else t.name)
        new = Take(take_id, name, *arrays, segments=segments,
                   parent=t.id if branch else (t.parent if t else None),
                   branch_frame=stop - 1 if branch else (t.branch_frame if t else None), events=events)
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
