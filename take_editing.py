"""Small, transactional edits to stored take timelines."""
from copy import deepcopy
from dataclasses import replace
import time
import uuid

import numpy as np

from camera_model import camera_at_frame, validate_camera_cuts
from take_sequencing import align_take
from takes import (MAX_FRAMES, MAX_TOTAL_FRAMES, MAX_PROJECT_AUDIO_BYTES, Take,
                   merge_dialogue, slice_dialogue, validate_take)


def _slice_cuts(cuts, start, end):
    """Keep the camera playing at the slice start and subsequent hard cuts."""
    if not cuts:
        return []
    active = camera_at_frame(cuts, start)
    result = ([dict(id=str(uuid.uuid4()), frame=0, camera_id=active)]
              if active is not None else [])
    result.extend(dict(cut, frame=cut['frame'] - start)
                  for cut in cuts if start < cut['frame'] < end)
    return result


def slice_take(take, start, end):
    """Copy a frame range with its timeline metadata rebased to frame zero."""
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(take.positions):
        raise ValueError('Invalid take slice')
    segments = [dict(segment, start=max(start, segment['start']) - start,
                     end=min(end, segment['end']) - start)
                for segment in take.segments
                if segment['start'] < end and segment['end'] > start]
    dialogue, audio_assets = slice_dialogue(take, start, end)
    return Take(take.id, take.name, *(getattr(take, name)[start:end].copy()
                                      for name in ('positions', 'rotations', 'motion')),
                segments=segments, camera_cuts=_slice_cuts(take.camera_cuts, start, end),
                dialogue=dialogue, audio_assets=audio_assets)


def _join_pieces(source, pieces):
    arrays = {name: np.concatenate([getattr(piece, name) for piece in pieces], axis=0)
              for name in ('positions', 'rotations', 'motion')}
    segments, cuts, offset = [], [], 0
    for piece in pieces:
        segments.extend(dict(segment, start=segment['start'] + offset,
                             end=segment['end'] + offset) for segment in piece.segments)
        cuts.extend(dict(cut, id=str(uuid.uuid4()), frame=cut['frame'] + offset)
                    for cut in piece.camera_cuts)
        offset += len(piece.positions)
    if cuts and cuts[0]['frame'] != 0:
        cuts.insert(0, dict(id=str(uuid.uuid4()), frame=0,
                            camera_id=cuts[0]['camera_id']))
    dialogue, assets = merge_dialogue((piece, sum(len(previous.positions)
                                                 for previous in pieces[:index]))
                                      for index, piece in enumerate(pieces))
    return Take(source.id, source.name, **arrays, segments=segments,
                parent=source.parent, branch_frame=source.branch_frame,
                camera_cuts=cuts, dialogue=dialogue, audio_assets=assets)


def splice_action(source, index, replacement=None):
    """Delete or replace one stored action, retaining and aligning later motion."""
    if type(index) is not int or not 0 <= index < len(source.segments):
        raise ValueError('Choose an action to edit')
    if replacement is not None:
        validate_take(replacement)
    segment = source.segments[index]
    start, end = segment['start'], segment['end']
    pieces = [slice_take(source, 0, start)] if start else []
    if replacement is not None:
        anchor = pieces[-1] if pieces else slice_take(source, 0, 1)
        aligned = align_take(replacement, anchor)
        pieces.append(Take(replacement.id, replacement.name, *aligned,
                           segments=[dict(s) for s in replacement.segments],
                           camera_cuts=[dict(cut) for cut in replacement.camera_cuts],
                           dialogue=[dict(cue) for cue in replacement.dialogue],
                           audio_assets=dict(replacement.audio_assets)))
    if end < len(source.positions):
        suffix = slice_take(source, end, len(source.positions))
        anchor = pieces[-1] if pieces else slice_take(source, 0, 1)
        aligned = align_take(suffix, anchor)
        pieces.append(Take(suffix.id, suffix.name, *aligned,
                           segments=suffix.segments, camera_cuts=suffix.camera_cuts,
                           dialogue=suffix.dialogue, audio_assets=suffix.audio_assets))
    if not pieces:
        return None
    if sum(len(piece.positions) for piece in pieces) < 4:
        raise ValueError('Keep at least 0.16 s of motion, or delete the entire take')
    edited = _join_pieces(source, pieces)
    if source.branch_frame is not None and start <= source.branch_frame:
        edited.parent = edited.branch_frame = None
    validate_take(edited)
    return edited


def append_saved_take(session, source_id):
    """Append a saved take to the selected take as one validated, undoable edit."""
    with session.lock:
        if session.busy:
            raise ValueError('Wait for generation to finish before adding a saved take')
        target = session.takes.get(session.active_take) if session.mode == 'Live ARDY' else None
        source = session.takes.get(source_id) if isinstance(source_id, str) else None
        if target is None:
            raise ValueError('Select the take you want to add to first')
        if source is None or source is target:
            raise ValueError('Choose a different saved take to add')
        if not session.character_motion_enabled:
            raise ValueError('Select a motion-ready character before editing takes')
        if any('beat_id' in segment for take in (target, source)
               for segment in take.segments):
            raise ValueError('Scene actions must be edited through the story editor')
        stop = len(target.positions)
        growth = len(source.positions)
        if stop + growth > MAX_FRAMES or sum(len(t.positions) for t in session.takes.values()) + growth > MAX_TOTAL_FRAMES:
            raise ValueError('Motion budget reached; shorten a take before adding it')
        aligned = align_take(source, target)
        placed = Take(source.id, source.name, *aligned,
                      segments=[dict(s) for s in source.segments],
                      camera_cuts=[dict(c) for c in source.camera_cuts],
                      dialogue=[dict(c) for c in source.dialogue],
                      audio_assets=dict(source.audio_assets))
        combined = _join_pieces(target, (target, placed))
        # The target path is untouched. Its earlier gate trigger remains valid;
        # the source trigger referred to a different world position.
        combined.events = [dict(event) for event in target.events]
        session._record_gate_events(combined)
        validate_take(combined)
        validate_camera_cuts(combined.camera_cuts, len(combined.positions),
                             {camera['id'] for camera in session.cameras})
        audio_bytes = sum(len(content) for key, take in session.takes.items() if key != target.id
                          for content in take.audio_assets.values())
        audio_bytes += sum(len(content) for content in combined.audio_assets.values())
        if audio_bytes > MAX_PROJECT_AUDIO_BYTES:
            raise ValueError('Project audio exceeds the supported budget')
        prior_frame = session.frame
        session.takes[target.id] = combined
        session._undo_action_edit = (target, combined, prior_frame, stop, [])
        session._select(target.id, stop)
        session.edit_context = None
        session.project_revision += 1
        session.project_status = 'Unsaved changes · Save project stores cameras and every take'
        session.status = f'Added {source.name} to the end · Undo is available'
        return combined


def delete_stored_action(session, segment_index):
    """Remove one ordinary action while keeping the stored suffix and media."""
    with session.lock:
        if session.busy:
            raise ValueError('Wait for generation to finish before deleting an action')
        source = session.takes.get(session.active_take) if session.mode == 'Live ARDY' else None
        if source is None:
            raise ValueError('Select a saved take to edit')
        if type(segment_index) is not int or not 0 <= segment_index < len(source.segments):
            raise ValueError('Choose an action to delete')
        if not session.character_motion_enabled:
            raise ValueError('Select a motion-ready character before editing takes')
        selected = source.segments[segment_index]
        if any('beat_id' in segment for segment in source.segments):
            raise ValueError('Scene actions must be edited through the story editor')
        if len(source.segments) == 1:
            session.remove_active_take()
            return None
        edited = splice_action(source, segment_index)
        edited.events = [dict(event) for event in source.events
                         if event['frame'] < selected['start']]
        session._record_gate_events(edited)
        validate_take(edited)
        validate_camera_cuts(edited.camera_cuts, len(edited.positions),
                             {camera['id'] for camera in session.cameras})
        start = selected['start']
        detached = []
        for child in session.takes.values():
            if child.parent == source.id and child.branch_frame >= start:
                detached.append((child, child.branch_frame))
                child.parent = None
                child.branch_frame = None
        prior_frame = session.frame
        session.takes[source.id] = edited
        session._undo_action_edit = (source, edited, prior_frame, start, detached)
        session._select(source.id, min(start, len(edited.positions) - 1))
        session.edit_context = None
        session.project_revision += 1
        session.action_edit_revision += 1
        session.project_status = 'Unsaved changes · Save project stores cameras and every take'
        session.status = f'Deleted action {segment_index + 1} · Undo is available'
        return edited


def extend_take_hold(session, take_id, required_frames, *, dialogue=None, audio_assets=None):
    """Commit a voice line and optional stationary tail as one validated edit.

    ``required_frames`` is the total desired take length at 25 fps. Motion is
    repeated exactly from the final stored pose; no generation service is used.
    Pass the complete updated cue list and asset map. Session state changes
    only after motion, dialogue, and the project audio budget pass validation.
    """
    if type(required_frames) is not int or required_frames < 4:
        raise ValueError('Invalid dialogue hold duration')
    with session.lock:
        take = session.takes.get(take_id)
        if take is None:
            raise ValueError('The selected take is no longer available')
        length = len(take.positions)
        new_length = max(length, required_frames)
        if (new_length == length and dialogue is None and audio_assets is None):
            return take
        total = sum(len(item.positions) for item in session.takes.values()) - length + new_length
        if new_length > MAX_FRAMES or total > MAX_TOTAL_FRAMES:
            raise ValueError('Dialogue exceeds the available take duration; shorten the line or remove another take')
        extra = new_length - length
        arrays = ({name: np.concatenate((getattr(take, name),
                                         np.repeat(getattr(take, name)[-1:], extra, axis=0)), axis=0)
                   for name in ('positions', 'rotations', 'motion')} if extra else {})
        segments = [dict(item) for item in take.segments]
        if extra:
            if segments[-1].get('prompt') == 'Dialogue hold':
                segments[-1]['end'] = new_length
            else:
                segments.append(dict(start=length, end=new_length, prompt='Dialogue hold'))
        candidate = replace(take, **arrays, segments=segments,
                            dialogue=deepcopy(take.dialogue if dialogue is None else dialogue),
                            audio_assets=dict(take.audio_assets if audio_assets is None else audio_assets))
        validate_take(candidate)
        audio_bytes = sum(len(content) for key, item in session.takes.items() if key != take_id
                          for content in item.audio_assets.values())
        audio_bytes += sum(len(content) for content in candidate.audio_assets.values())
        if audio_bytes > MAX_PROJECT_AUDIO_BYTES:
            raise ValueError('Project audio exceeds the supported budget')
        session.takes[take_id] = candidate
        if session.active_take == take_id:
            session.positions, session.rotations, session.motion = (candidate.positions,
                                                                      candidate.rotations,
                                                                      candidate.motion)
            session.clip_revision += 1
            if session.playing:
                session.started = time.perf_counter() - session.frame / (session.fps * session.playback_speed)
                session._clock_revision = session.clip_revision
        session._undo_action_edit = None
        session.project_revision += 1
        session.project_status = 'Unsaved changes · Save project stores cameras and every take'
        return candidate
