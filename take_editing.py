"""Small, transactional edits to stored take timelines."""
from copy import deepcopy
from dataclasses import replace
import time

import numpy as np

from takes import MAX_FRAMES, MAX_TOTAL_FRAMES, MAX_PROJECT_AUDIO_BYTES, validate_take


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
