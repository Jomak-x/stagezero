"""Transactional sequence editing and world-space character placement."""
from copy import deepcopy
from dataclasses import replace
import numpy as np

from live_motion import MotionSession
from take_sequencing import align_take, motion_statistics
from takes import Take, validate_take


def copy_take(take):
    # Arrays are immutable through the editing API. Copy metadata, share arrays.
    return replace(take, segments=deepcopy(take.segments), events=deepcopy(take.events))


def slice_take(take, start, end):
    segments = [dict(s, start=max(start, s['start']) - start, end=min(end, s['end']) - start)
                for s in take.segments if s['start'] < end and s['end'] > start]
    return replace(take, **{k: getattr(take, k)[start:end].copy()
                           for k in ('positions', 'rotations', 'motion')}, segments=segments, events=[])


def pose_of_take(take, frame=0):
    mean, scale = motion_statistics()
    facing = take.motion[frame, 3:5] * scale[3:5] + mean[3:5]
    if np.linalg.norm(facing) < 1e-6:
        raise ValueError('Take has no usable facing direction')
    root = take.positions[frame, 0]
    return float(root[0]), float(root[2]), float(np.degrees(np.arctan2(facing[1], facing[0])))


def place_take(take, x, z, heading):
    """Place start root/facing by reusing the verified normalized ARDY transform."""
    mean, scale = motion_statistics()
    anchor = slice_take(take, 0, 1)
    anchor.positions[0, 0, (0, 2)] = (x, z)
    features = anchor.motion * scale + mean
    angle = np.radians(heading)
    features[0, 3:5] = (np.cos(angle), np.sin(angle))
    anchor.motion = (features - mean) / scale
    p, r, m = align_take(take, anchor)
    return replace(take, positions=p, rotations=r, motion=m, events=[])


def splice_action(source, index, replacement=None):
    """Remove one segment or replace it, rigidly joining the remaining suffix."""
    segment = source.segments[index]
    start, end = segment['start'], segment['end']
    pieces = []
    if start:
        pieces.append(slice_take(source, 0, start))
    if replacement is not None:
        anchor = pieces[-1] if pieces else slice_take(source, 0, 1)
        p, r, m = align_take(replacement, anchor)
        pieces.append(replace(replacement, positions=p, rotations=r, motion=m))
    if end < len(source.positions):
        suffix = slice_take(source, end, len(source.positions))
        anchor = pieces[-1] if pieces else slice_take(source, 0, 1)
        p, r, m = align_take(suffix, anchor)
        pieces.append(replace(suffix, positions=p, rotations=r, motion=m))
    if not pieces:
        return None
    if sum(len(piece.positions) for piece in pieces) < 4:
        raise ValueError('Keep at least 0.16 s of motion, or delete the entire take')
    segments, offset = [], 0
    for piece in pieces:
        segments.extend(dict(s, start=s['start'] + offset, end=s['end'] + offset) for s in piece.segments)
        offset += len(piece.positions)
    arrays = {k: np.concatenate([getattr(piece, k) for piece in pieces], axis=0)
              for k in ('positions', 'rotations', 'motion')}
    edited = replace(source, **arrays, segments=segments, events=[])
    validate_take(edited)
    return edited


class TakeEditingMixin:
    @property
    def can_undo_edit(self):
        state = getattr(self, '_edit_undo', None)
        return bool(state and not self.busy and state['revision'] + 1 == self.project_revision)

    @property
    def can_undo(self):
        return self.can_undo_edit

    def _remember_edit(self, label):
        prior = getattr(self, '_edit_undo', None)
        if (label == 'move character start' and prior and prior['label'] == label
                and prior['active'] == self.active_take and prior['revision'] + 1 == self.project_revision):
            # Pointer-drag events are one user edit; preserve its original pose.
            prior['revision'] = self.project_revision
            return
        self._edit_undo = dict(takes={key: copy_take(t) for key, t in self.takes.items()},
                               active=self.active_take, frame=self.frame,
                               actor_start=deepcopy(self.scene.get('actor_start')),
                               revision=self.project_revision, label=label)

    def _editing_target(self):
        if self.busy:
            raise ValueError('Wait for generation to finish before editing')
        take = self.takes.get(self.active_take) if self.mode == 'Live ARDY' else None
        if take is None:
            raise ValueError('Select a take to edit')
        return take

    def _edited(self, message):
        self.edit_context = None
        self.project_revision += 1
        self.project_status = 'Unsaved changes · Save project stores every take'
        self.status = message

    def _detach_children(self, take_id):
        # Branch-frame provenance belongs to the old timeline, even if in range.
        for key, child in list(self.takes.items()):
            if child.parent == take_id:
                self.takes[key] = replace(child, parent=None, branch_frame=None)

    def delete_take(self, take_id=None):
        with self.lock:
            current = self._editing_target()
            identifier = current.id if take_id is None else take_id
            if identifier not in self.takes:
                raise ValueError('Choose an existing take to delete')
            removed = self.takes[identifier]
            self._remember_edit('delete take')
            del self.takes[identifier]
            self._detach_children(identifier)
            if self.active_take == identifier:
                if self.takes:
                    self._select(next(iter(self.takes)))
                else:
                    self.active_take = None
                    MotionSession.reset(self)
                    self._show_draft_start()
            self._edited(f'Deleted {removed.name} · Undo restores it')
            return removed

    def delete_action(self, index):
        with self.lock:
            take = self._editing_target()
            if type(index) is not int or not 0 <= index < len(take.segments):
                raise ValueError('Choose an action to delete')
            edited = splice_action(take, index)
            if edited is None:
                return self.delete_take()
            self._record_gate_events(edited)
            self._remember_edit('delete action')
            self.takes[take.id] = edited
            self._detach_children(take.id)
            self._select(take.id, min(take.segments[index]['start'], len(edited.positions) - 1))
            self._edited(f'Deleted action {index + 1} · remaining actions joined · Undo restores it')
            return edited

    def undo_edit(self):
        with self.lock:
            if not self.can_undo_edit:
                return False
            state = self._edit_undo
            self.mode = 'Live ARDY'
            self.takes = state['takes']
            if state['actor_start'] is None:
                self.scene.pop('actor_start', None)
            else:
                self.scene['actor_start'] = state['actor_start']
            if state['active'] in self.takes:
                self._select(state['active'], state['frame'])
            else:
                self.active_take = None
                MotionSession.reset(self)
                self._show_draft_start()
            self._edit_undo = None
            self._edited(f'Undid {state["label"]}')
            return True

    def get_start_pose(self):
        with self.lock:
            take = self.takes.get(self.active_take) if self.mode == 'Live ARDY' else None
            if take is not None:
                return pose_of_take(take)
            pending = self.scene.get('actor_start')
            if pending is not None:
                values = np.asarray(pending, dtype=float)
                if values.shape != (3,) or not np.isfinite(values).all():
                    raise ValueError('Invalid saved character start position')
                return tuple(float(value) for value in values)
            root = self.recorded[0][0, 0]
            return float(root[0]), float(root[2]), self._reference_heading()

    def _reference_heading(self):
        # G1 left/right hip vector matches ARDY's heading convention.
        pose = self.recorded[0][0]
        across = pose[8] - pose[1]
        return float(np.degrees(np.arctan2(across[2], -across[0]))) if np.linalg.norm(across[[0, 2]]) > 1e-6 else 0.

    def _draft_arrays(self):
        positions, rotations = self.recorded
        x, z, heading = self.get_start_pose()
        angle = np.radians(heading - self._reference_heading())
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=np.float32)
        offset = np.array((x, 0., z)) - positions[0, 0] @ rotation.T
        offset[1] = 0.
        return positions @ rotation.T + offset, rotation @ rotations

    def _show_draft_start(self):
        if self.mode == 'Live ARDY' and self.active_take is None and 'actor_start' in self.scene:
            self.positions, self.rotations = self._draft_arrays()
            self.clip_revision += 1

    def generation_start_positions(self, take, stop):
        if take is not None:
            return take.positions[max(0, stop - 1)].copy()
        if 'actor_start' in self.scene:
            # A new-generation request may occur while another take is selected.
            pending = self.scene['actor_start']
            positions = self.recorded[0][0].copy()
            angle = np.radians(float(pending[2]) - self._reference_heading())
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=np.float32)
            offset = np.array((pending[0], 0., pending[1])) - positions[0] @ rotation.T
            offset[1] = 0.
            return positions @ rotation.T + offset
        return self.recorded[0][0].copy()

    def set_start_pose(self, x, z, heading_degrees):
        values = np.asarray((x, z, heading_degrees), dtype=float)
        if not np.isfinite(values).all() or np.abs(values[:2]).max() > 100 or abs(values[2]) > 36000:
            raise ValueError('Choose a finite starting position within 100 m and a valid facing angle')
        x, z, heading = (float(values[0]), float(values[1]), float((values[2] + 180) % 360 - 180))
        with self.lock:
            if self.busy:
                raise ValueError('Wait for generation to finish before moving the character')
            if self.mode != 'Live ARDY':
                raise ValueError('Switch to the scene before moving the character')
            current = self.takes.get(self.active_take)
            edited = place_take(current, x, z, heading) if current is not None else None
            if edited is not None:
                validate_take(edited)
                self._record_gate_events(edited)
            self._remember_edit('move character start')
            if edited is not None:
                self.takes[current.id] = edited
                self._detach_children(current.id)
                self._select(current.id, 0)
            else:
                self.scene['actor_start'] = [x, z, heading]
                self.playing = False
                self.frame = 0
                self._show_draft_start()
            self._edited('Character start updated · preview from the beginning · Undo restores it')
            return edited

    def _place_first_generation_chunk(self, result, index, context):
        if index or context is not None:
            return result
        target = getattr(self, '_generation_start_override', None)
        if target is None:
            return result
        n = len(result['motion'])
        take = Take('generated', 'generated', result['positions'], result['rotations'], result['motion'],
                    segments=[dict(start=0, end=n, prompt=self.prompt)])
        placed = place_take(take, *target)
        return {**result, **{k: getattr(placed, k) for k in ('positions', 'rotations', 'motion')}}

    def _install_action_result(self, result, take, index):
        n = len(result['motion'])
        meta = result['metadata']
        generated = Take(take.id, take.name, result['positions'], result['rotations'], result['motion'],
                         segments=[dict(start=0, end=n, prompt=self.prompt,
                                        request_id=meta['request_id'], generation_seconds=meta['generation_seconds'])])
        edited = splice_action(take, index, generated)
        self._record_gate_events(edited)
        self._remember_edit('edit action')
        self.takes[take.id] = edited
        self._detach_children(take.id)
        self.active_take = take.id
        self.positions, self.rotations, self.motion = edited.positions, edited.rotations, edited.motion
        self._edited(f'Replaced action {index + 1} · following actions kept')
        return take.segments[index]['start']
