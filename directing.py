"""One actor, multiple recorded takes. Inference remains in the existing worker."""
from pathlib import Path
import re
import time
import uuid
import numpy as np
from live_motion import MotionSession, validate_result
from takes import Take, encode_project, decode_project, validate_take, MAX_TAKES, MAX_FRAMES, MAX_TOTAL_FRAMES
from duration_planning import CHUNK_FRAMES, plan_duration
from camera_model import (copy_camera_cuts, validate_camera,
                          validate_camera_cuts, validate_cameras)


class DirectorSession(MotionSession):
    def __init__(self, *args, **kwargs):
        self.takes = {}
        self.active_take = None
        self._removed_take = None
        self._undo_action_edit = None
        self.edit_context = None
        self.planned_frames = CHUNK_FRAMES
        self.planned_seconds = CHUNK_FRAMES / 25
        self.duration_label = 'Auto · text-length estimate'
        self.project_status = ''
        self.project_revision = 0
        self.action_edit_revision = 0
        self.cameras = []
        self.camera_project_id = str(uuid.uuid4())
        self.playback_speed = 1.0
        self.loop_playback = False
        self._clock_revision = -1
        self.scene = {'gate': {'position': [0., 0., 1.5], 'radius': .55, 'enabled': True}}
        super().__init__(*args, **kwargs)

    def _camera_changed(self, status):
        self.project_revision += 1
        self.project_status = 'Unsaved changes · Save project stores cameras and every take'
        self.status = status

    def _camera(self, camera_id):
        camera = next((camera for camera in self.cameras if camera['id'] == camera_id), None)
        if camera is None:
            raise ValueError('Camera no longer exists')
        return camera

    def _new_camera_name(self, base='Camera'):
        names = {camera['name'] for camera in self.cameras}
        number = 1
        while True:
            suffix = f' {number}'
            name = base[:80 - len(suffix)].rstrip() + suffix
            if name not in names:
                return name
            number += 1

    def add_camera(self, position, wxyz, fov):
        with self.lock:
            camera = validate_camera(dict(id=str(uuid.uuid4()), name=self._new_camera_name(),
                                          position=position, wxyz=wxyz, fov=fov))
            cameras = validate_cameras(self.cameras + [camera])
            self.cameras = cameras
            self._camera_changed(f'Added camera · {camera["name"]}')
            return validate_camera(camera)

    def update_camera(self, camera_id, **fields):
        with self.lock:
            current = self._camera(camera_id)
            if set(fields) - {'name', 'position', 'wxyz', 'fov'}:
                raise ValueError('Invalid camera fields')
            camera = validate_camera({**current, **fields})
            if camera != current:
                self.cameras = [camera if item['id'] == camera_id else item for item in self.cameras]
                self._camera_changed(f'Updated camera · {camera["name"]}')
            return validate_camera(camera)

    def duplicate_camera(self, camera_id):
        with self.lock:
            source = self._camera(camera_id)
            camera = validate_camera(dict(source, id=str(uuid.uuid4()),
                                          name=self._new_camera_name(source['name'] + ' copy')))
            self.cameras = validate_cameras(self.cameras + [camera])
            self._camera_changed(f'Duplicated camera · {camera["name"]}')
            return validate_camera(camera)

    def remove_camera(self, camera_id):
        with self.lock:
            camera = self._camera(camera_id)
            def uses_camera(take):
                return any(cut['camera_id'] == camera_id for cut in take.camera_cuts)
            usages = [f'take “{name}”' for name in dict.fromkeys(
                take.name for take in self.takes.values() if uses_camera(take))]
            guidance = ['Change or clear those camera cuts first.'] if usages else []
            removed = self._removed_take[1] if self._removed_take is not None else None
            if removed is not None and uses_camera(removed):
                usages.append(f'removed take “{removed.name}”')
                guidance.append('Restore the removed take with Undo to change or clear its cuts.')
            undo = self._undo_action_edit
            # A superseded action snapshot can never be restored, unless its
            # edited take is itself waiting in the take-removal Undo slot.
            if (undo is not None and (self.takes.get(undo[1].id) is undo[1] or removed is undo[1])
                    and uses_camera(undo[0])):
                usages.append(f'action-edit Undo snapshot for “{undo[0].name}”')
                guidance.append('Undo the action edit to change or clear that snapshot’s cuts.')
            if usages:
                raise ValueError('Camera is used by ' + ', '.join(usages) + '. ' + ' '.join(guidance))
            self.cameras = [item for item in self.cameras if item['id'] != camera_id]
            self._camera_changed(f'Deleted camera · {camera["name"]}')

    def _camera_cut_take(self, take_id):
        if self.busy:
            raise ValueError('Wait for generation to finish before editing camera cuts')
        take = self.takes.get(take_id) if isinstance(take_id, str) else None
        if take is None:
            raise ValueError('Select a saved take before editing camera cuts')
        return take

    def _set_camera_cuts(self, take, cuts, status):
        cuts = validate_camera_cuts(cuts, len(take.positions), {camera['id'] for camera in self.cameras})
        if cuts != take.camera_cuts:
            take.camera_cuts = cuts
            # A later cut edit supersedes the action snapshot, as a later rename does.
            if self._undo_action_edit is not None and self._undo_action_edit[1] is take:
                self._undo_action_edit = None
            self._camera_changed(status)

    def add_camera_cut(self, take_id, frame, camera_id):
        with self.lock:
            take = self._camera_cut_take(take_id)
            self._camera(camera_id)
            if type(frame) is not int or not 0 <= frame < len(take.positions):
                raise ValueError('Camera cut frame must be within the take')
            cuts = copy_camera_cuts(take.camera_cuts, len(take.positions))
            cut = next((item for item in cuts if item['frame'] == frame), None)
            if cut is None:
                if not cuts and frame != 0:
                    cuts.append(dict(id=str(uuid.uuid4()), frame=0, camera_id=camera_id))
                cut = dict(id=str(uuid.uuid4()), frame=frame, camera_id=camera_id)
                cuts.append(cut)
            else:
                cut['camera_id'] = camera_id
            self._set_camera_cuts(take, sorted(cuts, key=lambda item: item['frame']), 'Updated camera cuts')
            return dict(cut)

    def update_camera_cut(self, take_id, cut_id, *, frame=None, camera_id=None):
        with self.lock:
            take = self._camera_cut_take(take_id)
            current = next((cut for cut in take.camera_cuts if cut['id'] == cut_id), None)
            if current is None:
                raise ValueError('Camera cut no longer exists')
            frame = current['frame'] if frame is None else frame
            camera_id = current['camera_id'] if camera_id is None else camera_id
            self._camera(camera_id)
            if type(frame) is not int or not 0 <= frame < len(take.positions):
                raise ValueError('Camera cut frame must be within the take')
            if current['frame'] == 0 and frame != 0:
                raise ValueError('The first camera cut must remain at frame 0')
            updated = dict(current, frame=frame, camera_id=camera_id)
            cuts = [dict(cut) for cut in take.camera_cuts
                    if cut['id'] != cut_id and cut['frame'] != frame] + [updated]
            self._set_camera_cuts(take, sorted(cuts, key=lambda item: item['frame']), 'Updated camera cut')
            return dict(updated)

    def remove_camera_cut(self, take_id, cut_id):
        with self.lock:
            take = self._camera_cut_take(take_id)
            cut = next((cut for cut in take.camera_cuts if cut['id'] == cut_id), None)
            if cut is None:
                raise ValueError('Camera cut no longer exists')
            if cut['frame'] == 0:
                raise ValueError('The first camera cut must remain at frame 0; use Clear cuts to remove the sequence')
            self._set_camera_cuts(take, [dict(item) for item in take.camera_cuts if item['id'] != cut_id],
                                  'Deleted camera cut')

    def clear_camera_cuts(self, take_id):
        with self.lock:
            take = self._camera_cut_take(take_id)
            self._set_camera_cuts(take, [], 'Cleared camera cuts')

    def set_mode(self, mode):
        with self.lock:
            super().set_mode(mode)
            if mode == 'Live ARDY' and self.active_take in self.takes:
                self._select(self.active_take, 0)
            elif mode == 'Live ARDY':
                self._hold_reference_pose()

    def _hold_reference_pose(self):
        """Show a still reference frame for an empty draft, never the demo clip."""
        self.positions = self.recorded[0][:1].copy()
        self.rotations = self.recorded[1][:1].copy()
        self.motion = None
        self.fps = 60
        self.frame = 0
        self.playing = False
        self.kind = 'reference'
        self.metrics = None
        self.clip_revision += 1

    def new_take(self):
        with self.lock:
            if self.busy:
                self.status = 'Wait for generation to finish before starting a new take'
                return False
            if len(self.takes) >= MAX_TAKES:
                self.status = 'Project has 12 takes; save it and start a new project.'
                return False
            self.mode = 'Live ARDY'
            super().reset()
            self.active_take = None
            self.edit_context = None
            self.prompt = ''
            self._hold_reference_pose()
            self.status = 'New take · enter an instruction to begin'
            return True

    @property
    def can_undo_take_removal(self):
        with self.lock:
            return self._removed_take is not None

    def remove_active_take(self):
        """Remove the selected saved take, retaining one in-memory undo step."""
        with self.lock:
            if self.busy:
                self.status = 'Wait for generation to finish before removing a take'
                return False
            take = self.takes.get(self.active_take) if self.mode == 'Live ARDY' else None
            if take is None:
                self.status = 'Select a take to remove'
                return False
            removed_id = take.id
            index = list(self.takes).index(removed_id)
            frame = self.frame
            child_links = [(child.id, child.branch_frame) for child in self.takes.values()
                           if child.parent == removed_id]
            for child_id, _ in child_links:
                child = self.takes[child_id]
                child.parent = None
                child.branch_frame = None
            del self.takes[removed_id]
            self._removed_take = (index, take, frame, child_links)
            if self.takes:
                self._select(next(reversed(self.takes)), 0)
            else:
                self._invalidate()
                self.active_take = None
                self.prompt = ''
                self._hold_reference_pose()
            self.edit_context = None
            self.project_revision += 1
            self.project_status = 'Unsaved changes · Save project stores every take'
            self.status = f'Removed take · {take.name} · Undo is available'
            return True

    def undo_remove_take(self):
        """Restore the most recently removed take and its direct child links."""
        with self.lock:
            if self.busy:
                self.status = 'Wait for generation to finish before undoing removal'
                return False
            removed = self._removed_take
            if removed is None:
                self.status = 'No take removal to undo'
                return False
            index, take, frame, child_links = removed
            if take.id in self.takes or not self._can_add_take(len(take.positions)):
                return False
            entries = list(self.takes.items())
            entries.insert(min(index, len(entries)), (take.id, take))
            self.takes = dict(entries)
            for child_id, branch_frame in child_links:
                child = self.takes.get(child_id)
                if child is not None and child.parent is None:
                    child.parent = take.id
                    child.branch_frame = branch_frame
            self._removed_take = None
            self.mode = 'Live ARDY'
            self._select(take.id, frame)
            self.project_revision += 1
            self.project_status = 'Unsaved changes · Save project stores every take'
            self.status = f'Restored take · {take.name}'
            return True

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
            if not self.character_motion_enabled:
                return
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
                if self._undo_action_edit is not None and self._undo_action_edit[1] is t:
                    self._undo_action_edit = None
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
                   segments=segments, parent=source.id, branch_frame=stop - 1, events=events,
                   camera_cuts=copy_camera_cuts(source.camera_cuts, stop))
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
            if not self.character_motion_enabled:
                self.status = 'Select a motion-ready character before generating motion'
                return
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

    @property
    def can_undo_action_edit(self):
        with self.lock:
            undo = self._undo_action_edit
            return undo is not None and self.takes.get(undo[0].id) is undo[1]

    def undo_action_edit(self):
        """Restore the take snapshot from the most recent completed action edit."""
        with self.lock:
            if self.busy:
                self.status = 'Wait for generation to finish before undoing an action edit'
                return False
            undo = self._undo_action_edit
            if undo is None or self.takes.get(undo[0].id) is not undo[1]:
                self.status = 'No action edit to undo'
                return False
            original, edited, frame, stop, detached = undo
            total = (sum(len(t.positions) for t in self.takes.values())
                     - len(edited.positions) + len(original.positions))
            if total > MAX_TOTAL_FRAMES:
                self.status = 'Cannot undo edit: project motion budget exceeded. Remove another take and try again.'
                return False
            for child in self.takes.values():
                if child.parent == original.id and child.branch_frame >= stop:
                    child.parent = None
                    child.branch_frame = None
            self.takes[original.id] = original
            for child, branch_frame in detached:
                if self.takes.get(child.id) is child and child.parent is None:
                    child.parent = original.id
                    child.branch_frame = branch_frame
            self._undo_action_edit = None
            self.mode = 'Live ARDY'
            self._select(original.id, frame)
            self.project_revision += 1
            self.project_status = 'Unsaved changes · Save project stores every take'
            self.status = 'Restored actions · most recent edit undone'
            return True

    def submit_action_edit(self, prompt, segment_index, operation, seconds=None):
        """Regenerate an action and its suffix within the selected take, atomically."""
        with self.lock:
            if not self.character_motion_enabled:
                self.status = 'Select a motion-ready character before generating motion'
                return False
            if self.busy:
                self.status = 'Wait for generation to finish before editing an action'
                return False
            source = self.takes.get(self.active_take) if self.mode == 'Live ARDY' else None
            if source is None:
                self.status = 'Select a take to edit an action'
                return False
            if type(segment_index) is not int or not 0 <= segment_index < len(source.segments):
                self.status = 'Select an action on the current timeline'
                return False
            if operation not in ('replace', 'insert_before', 'insert_after'):
                self.status = 'Choose Replace, Insert before, or Insert after'
                return False
            prompt = str(prompt).strip()
            if not 1 <= len(prompt) <= 500:
                self.status = 'Enter a movement instruction (1–500 characters)'
                return False
            selected = source.segments[segment_index]
            if operation == 'replace':
                stop = selected['start']
                suffix = source.segments[segment_index + 1:]
                requested_seconds = seconds
            elif operation == 'insert_before':
                stop = selected['start']
                suffix = source.segments[segment_index:]
                requested_seconds = seconds
            else:
                stop = selected['end']
                suffix = source.segments[segment_index + 1:]
                requested_seconds = seconds
            if operation == 'replace' and seconds is None:
                new_frames = selected['end'] - selected['start']
            else:
                try:
                    new_frames = plan_duration(prompt, requested_seconds).frames
                except ValueError as exc:
                    self.status = str(exc)
                    return False
            actions = [(prompt, new_frames)] + [(s['prompt'], s['end'] - s['start']) for s in suffix]
            final_length = stop + sum(frames for _, frames in actions)
            total = sum(len(t.positions) for t in self.takes.values()) - len(source.positions) + final_length
            if final_length > MAX_FRAMES or total > MAX_TOTAL_FRAMES:
                self.status = 'Motion budget reached; shorten an action before continuing.'
                return False
            self._invalidate()
            self.prompt = prompt
            self.playing = False
            self.resume_after_generation = True
            request_id = str(uuid.uuid4())
            self.current_id = request_id
            self.busy = True
            submitted = time.perf_counter()
            self.pending = dict(kind='action_edit', version=self.version, request_id=request_id,
                                source=source, stop=stop, actions=actions, frame=self.frame,
                                submitted=submitted)
            self.wake.set()
            chunks = sum((frames + CHUNK_FRAMES - 1) // CHUNK_FRAMES for _, frames in actions)
            self.status = f'Regenerating {len(actions)} action' + ('s' if len(actions) != 1 else '') + f' · 0/{chunks} chunks'
            return True

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
            if isinstance(job, dict) and job.get('kind') == 'action_edit':
                self._work_action_edit(job)
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

    def _work_action_edit(self, job):
        """Build all replacement segments before changing any visible take state."""
        source, stop = job['source'], job['stop']
        version, request_id = job['version'], job['request_id']
        actions = job['actions']
        total_chunks = sum((frames + CHUNK_FRAMES - 1) // CHUNK_FRAMES for _, frames in actions)
        completed_chunks = 0
        generated = {key: [] for key in ('positions', 'rotations', 'motion')}
        segments, events = source.prefix(stop)
        motion_so_far = source.motion[max(0, stop - 52):stop].copy()
        generation_seconds = 0.0
        final_meta = None
        failure_context = ''
        try:
            for action_number, (prompt, frames) in enumerate(actions, start=len(segments) + 1):
                count = (frames + CHUNK_FRAMES - 1) // CHUNK_FRAMES
                action_start = stop + sum(len(part) for part in generated['motion'])
                action_seconds = 0.0
                for index in range(count):
                    with self.lock:
                        if version != self.version or request_id != self.current_id:
                            return
                    history_count = min(52, len(motion_so_far)) // 4 * 4
                    history = motion_so_far[-history_count:].copy() if history_count else None
                    failure_context = f'action {action_number} "{prompt[:60]}", chunk {index + 1}/{count} · '
                    result = self.backend.generate(request_id, prompt, history)
                    validate_result(result, request_id)
                    final_meta = result['metadata']
                    seconds = float(final_meta['generation_seconds'])
                    action_seconds += seconds
                    generation_seconds += seconds
                    trim = min(CHUNK_FRAMES, frames - index * CHUNK_FRAMES)
                    for key in generated:
                        generated[key].append(result[key][:trim])
                    motion_so_far = np.concatenate((motion_so_far, result['motion'][:trim]), axis=0)[-52:]
                    completed_chunks += 1
                    with self.lock:
                        if version != self.version or request_id != self.current_id:
                            return
                        if completed_chunks < total_chunks:
                            request_id = str(uuid.uuid4())
                            self.current_id = request_id
                            self.status = f'Regenerating actions · {completed_chunks}/{total_chunks} chunks received; holding pose'
                    failure_context = ''
                segments.append(dict(start=action_start, end=action_start + frames, prompt=prompt,
                                     request_id=final_meta['request_id'], generation_seconds=action_seconds))
            arrays = [np.concatenate([getattr(source, key)[:stop]] + generated[key], axis=0)
                      for key in ('positions', 'rotations', 'motion')]
            inherited_prefix_changed = source.branch_frame is not None and stop <= source.branch_frame
            edited = Take(source.id, source.name, *arrays, segments=segments,
                          parent=None if inherited_prefix_changed else source.parent,
                          branch_frame=None if inherited_prefix_changed else source.branch_frame, events=events,
                          camera_cuts=copy_camera_cuts(source.camera_cuts, len(arrays[0])))
            self._record_gate_events(edited)
            validate_take(edited)
            with self.lock:
                if version != self.version or request_id != self.current_id or self.takes.get(source.id) is not source:
                    return
                detached = []
                for child in self.takes.values():
                    if child.parent == source.id and child.branch_frame >= stop:
                        detached.append((child, child.branch_frame))
                        child.parent = None
                        child.branch_frame = None
                self.takes[source.id] = edited
                self._undo_action_edit = (source, edited, job['frame'], stop, detached)
                self.active_take = source.id
                self.positions, self.rotations, self.motion = edited.positions, edited.rotations, edited.motion
                self.fps, self.kind = 25, 'generated'
                self.frame = stop
                self.clip_revision += 1
                self.busy = False
                self.status = f'Regenerated {len(actions)} action' + ('s' if len(actions) != 1 else '') + ' · Undo is available'
                self.metrics = {**final_meta, 'generation_seconds': generation_seconds,
                                'command_to_received_seconds': time.perf_counter() - job['submitted']}
                self.needs_ack = (request_id, job['submitted'])
                self.started = time.perf_counter() - self.frame / self.fps
                self.playing = self.resume_after_generation
                self.project_revision += 1
                self.action_edit_revision += 1
                self.project_status = 'Unsaved changes · Save project stores every take'
        except Exception as exc:
            with self.lock:
                if version != self.version:
                    return
                self.busy = False
                self.playing = False
                self.status = f'Action edit failed · {failure_context}{type(exc).__name__}: {str(exc)[:200]}. Original take preserved; retry.'

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
                   branch_frame=stop - 1 if branch and stop else (t.branch_frame if t and not branch else None), events=events,
                   camera_cuts=[] if t is None else copy_camera_cuts(t.camera_cuts, len(arrays[0])))
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
            active = self.active_take or next(iter(self.takes), None)
            frame = self.frame if self.mode == 'Live ARDY' and self.active_take else 0
            data = encode_project(self.takes, active, frame, self.scene, self.cameras)
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
            backup = self.save_project(directory, 'automatic-backup')[0] if self.project_revision else None
            self._invalidate()
            self.takes = {}
            self.cameras = []
            self.camera_project_id = str(uuid.uuid4())
            self._removed_take = None
            self._undo_action_edit = None
            self.project_revision += 1
            self.active_take = None
            self.edit_context = None
            self.prompt = ''
            self.mode = 'Live ARDY'
            super().reset()
            self._hold_reference_pose()
            self.scene = {'gate': {'position': [0., 0., 1.5], 'radius': .55, 'enabled': True}}
            self.project_status = f'Previous project backed up: {backup.name}' if backup else 'New project'

    def load_project(self, data):
        takes, active, frame, scene, cameras = decode_project(data, include_cameras=True)
        gate = scene.get('gate')
        if not isinstance(gate, dict) or not isinstance(gate.get('enabled'), bool):
            raise ValueError('Invalid gate state')
        p = np.asarray(gate.get('position'), dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all() or np.abs(p).max() > 100 or not isinstance(gate.get('radius'), (int, float)) or not .1 <= gate['radius'] <= 3:
            raise ValueError('Invalid gate geometry')
        with self.lock:
            self._invalidate()
            self.takes, self.scene = takes, scene
            self.cameras = cameras
            self.camera_project_id = str(uuid.uuid4())
            self._removed_take = None
            self._undo_action_edit = None
            self.edit_context = None
            self.project_revision += 1
            self.mode = 'Live ARDY'
            if active is None:
                self.active_take = None
                self.prompt = ''
                self._hold_reference_pose()
                self.status = 'Empty project · enter an instruction to begin'
                self.project_status = 'Loaded empty project'
            else:
                self._select(active, frame)
                self.project_status = 'Loaded stored motion · no regeneration'
