"""World-space start placement for a saved take or an empty live draft.

The motion features and visible joint transforms receive the same rigid
transform. Timeline, camera, voice, and scene metadata stay on the take.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import math

import numpy as np

from take_sequencing import align_take, motion_statistics
from takes import validate_take


def _reference_heading(session):
    pose = session.recorded[0][0]
    across = pose[8] - pose[1]
    return float(np.degrees(np.arctan2(across[2], -across[0]))) if np.linalg.norm(across[[0, 2]]) > 1e-6 else 0.


def _take_heading(take):
    mean, scale = motion_statistics()
    facing = take.motion[0, 3:5] * scale[3:5] + mean[3:5]
    if np.linalg.norm(facing) < 1e-6:
        raise ValueError('Take has no usable facing direction')
    return float(np.degrees(np.arctan2(facing[1], facing[0])))


def get_start_pose(session):
    with session.lock:
        take = session.takes.get(session.active_take) if session.mode == 'Live ARDY' else None
        if take is not None:
            root = take.positions[0, 0]
            return float(root[0]), float(root[2]), _take_heading(take)
        pending = session.scene.get('actor_start')
        if pending is not None:
            values = np.asarray(pending, dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError('Invalid saved character start position')
            return tuple(float(value) for value in values)
        root = session.recorded[0][0, 0]
        return float(root[0]), float(root[2]), _reference_heading(session)


def place_arrays(positions, rotations, motion, x, z, heading):
    """Apply one start transform while retaining every frame and feature."""
    source = SimpleNamespace(positions=positions, rotations=rotations, motion=motion)
    mean, scale = motion_statistics()
    anchor_motion = motion[:1].copy()
    features = anchor_motion * scale + mean
    angle = math.radians(heading)
    features[0, 3:5] = math.cos(angle), math.sin(angle)
    anchor_motion = ((features - mean) / scale).astype(motion.dtype)
    anchor_positions = positions[:1].copy()
    anchor_positions[0, 0, (0, 2)] = x, z
    anchor = SimpleNamespace(positions=anchor_positions, motion=anchor_motion)
    return align_take(source, anchor)


def place_take(take, x, z, heading):
    arrays = place_arrays(take.positions, take.rotations, take.motion, x, z, heading)
    placed = replace(take, positions=arrays[0], rotations=arrays[1], motion=arrays[2])
    validate_take(placed)
    return placed


def show_draft_start(session):
    """Keep an empty live draft visible at its chosen start."""
    if session.mode != 'Live ARDY' or session.active_take is not None:
        return
    x, z, heading = get_start_pose(session)
    positions, rotations = session.recorded[0][:1], session.recorded[1][:1]
    angle = math.radians(heading - _reference_heading(session))
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=np.float32)
    offset = np.array((x, 0., z)) - positions[0, 0] @ rotation.T
    offset[1] = 0.
    session.positions = positions @ rotation.T + offset
    session.rotations = rotation @ rotations
    session.motion = None
    session.frame = 0
    session.playing = False
    session.clip_revision += 1


def generation_start_positions(session, take, stop):
    if take is not None:
        return take.positions[max(0, stop - 1)].copy()
    with session.lock:
        if 'actor_start' in session.scene:
            x, z, heading = session.scene['actor_start']
        else:
            root = session.recorded[0][0, 0]
            x, z, heading = float(root[0]), float(root[2]), _reference_heading(session)
        source = session.recorded[0][0]
        angle = math.radians(heading - _reference_heading(session))
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=np.float32)
        offset = np.array((x, 0., z)) - source[0] @ rotation.T
        offset[1] = 0.
        return source @ rotation.T + offset


def place_generated_result(session, result):
    """Place a first generated result at an empty draft's stored start."""
    if 'actor_start' not in session.scene:
        return result
    x, z, heading = session.scene['actor_start']
    p, r, m = place_arrays(result['positions'], result['rotations'], result['motion'], x, z, heading)
    return dict(result, positions=p, rotations=r, motion=m)


def can_undo_start_pose(session):
    with session.lock:
        undo = getattr(session, '_placement_undo', None)
        return bool(undo and not session.busy and undo['revision'] + 1 == session.project_revision)


def set_start_pose(session, x, z, heading):
    values = np.asarray((x, z, heading), dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or np.abs(values[:2]).max() > 100 or abs(values[2]) > 180:
        raise ValueError('Choose a finite starting position within 100 m and a valid facing angle')
    x, z, heading = map(float, values)
    with session.lock:
        if session.busy or session.mode != 'Live ARDY':
            raise ValueError('Pause generation and switch to the live scene before moving the character')
        current = session.takes.get(session.active_take)
        prior = getattr(session, '_placement_undo', None)
        coalesce = (prior and prior['revision'] + 1 == session.project_revision
                    and prior['active'] == session.active_take)
        if not coalesce:
            affected = {current.id: current} if current is not None else {}
            if current is not None:
                affected.update((child.id, child) for child in session.takes.values() if child.parent == current.id)
            prior = {'takes': affected, 'active': session.active_take,
                     'actor_start': list(session.scene['actor_start']) if 'actor_start' in session.scene else None,
                     'revision': session.project_revision}
        if current is not None:
            # Gate hits depend on world coordinates; branch provenance depends
            # on the unmodified prefix. Recompute both for the moved take.
            edited = replace(place_take(current, x, z, heading), events=[],
                             parent=None, branch_frame=None)
            session._record_gate_events(edited)
            validate_take(edited)
            session.takes[current.id] = edited
            # A branch refers to the old world path. Keep its motion and all
            # media, while clearing only the now misleading parent link.
            for child in list(session.takes.values()):
                if child.parent == current.id:
                    session.takes[child.id] = replace(child, parent=None, branch_frame=None)
            session._select(current.id, 0)
        else:
            session.scene['actor_start'] = [x, z, heading]
            show_draft_start(session)
            edited = None
        session._undo_action_edit = None
        session.project_revision += 1
        session.project_status = 'Unsaved changes · Save project stores cameras and every take'
        session.status = 'Character start updated · Undo move restores it'
        prior['revision'] = session.project_revision - 1
        session._placement_undo = prior
        return edited


def undo_start_pose(session):
    with session.lock:
        if not can_undo_start_pose(session):
            return False
        undo = session._placement_undo
        for take_id, original in undo['takes'].items():
            session.takes[take_id] = original
        if undo['actor_start'] is None:
            session.scene.pop('actor_start', None)
        else:
            session.scene['actor_start'] = undo['actor_start']
        if undo['active'] in session.takes:
            session._select(undo['active'], 0)
        else:
            session.active_take = None
            show_draft_start(session)
        session._placement_undo = None
        session.project_revision += 1
        session.project_status = 'Unsaved changes · Save project stores cameras and every take'
        session.status = 'Character start move undone'
        return True
