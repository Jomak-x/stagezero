"""Opt-in native Core proximity reactions; no character poses are changed.

Only authored, ground-level automatic doors and proximity lamps participate.
The complete committed root trajectory is the event record, so seeking and
exact archive replay reproduce the same eased lift. This is an automatic door,
not learned hand contact, grasping, or arbitrary mesh articulation.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

DOOR_OPEN_SECONDS = .8
VERSION = 1


def _eligible(obj):
    interaction = obj.get('interaction', {})
    if (interaction.get('trigger') != 'proximity'
            or abs(obj['position'][1] - obj['size'][1] / 2) > .1):
        return False
    if obj['kind'] == 'lamp':
        return interaction.get('action') == 'switch_on'
    return (obj['kind'] == 'door' and interaction.get('action') == 'open'
            and abs(obj['position'][1] - obj['size'][1] / 2) <= .1)


def _first_trigger(obj, roots, start_frame):
    if start_frame >= roots.shape[1]:
        return None
    centers = np.asarray(obj['position'])
    angle = math.radians(obj.get('yaw', 0))
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array(((c, s), (-s, c)))
    offsets = (roots[:, start_frame:, [0, 2]] - centers[[0, 2]]) @ rotation
    outside = np.maximum(np.abs(offsets) - np.asarray(obj['size'])[[0, 2]] / 2, 0)
    nearby = np.linalg.norm(outside, axis=-1) <= obj['interaction']['radius']
    # A body on another floor must not trigger a ground-floor door/lamp.
    nearby &= np.abs(roots[:, start_frame:, 1] - 1.) < 1.0
    hits = np.flatnonzero(np.any(nearby, axis=0))
    return int(hits[0]) + start_frame if len(hits) else None


def _options(enabled, start_frame):
    if type(enabled) is not bool:
        raise ValueError("Reaction enabled must be a boolean")
    if type(start_frame) is not int or start_frame < 0:
        raise ValueError('Reaction start frame must be a non-negative integer')


def _states(scene, roots, frame, fps, *, enabled, start_frame, triggers=None):
    _options(enabled, start_frame)
    result = []
    for obj in scene['objects']:
        position, color = list(obj['position']), list(obj['color'])
        first = None
        if enabled and _eligible(obj):
            first = (triggers.get(obj['id']) if triggers is not None
                     else _first_trigger(obj, roots[:, :frame + 1], start_frame))
            if first is not None and first > frame:
                first = None
        active = first is not None
        fraction = 0.
        if active:
            if obj['kind'] == 'door':
                t = min(1., max(0., (frame - first) / (fps * DOOR_OPEN_SECONDS)))
                fraction = t * t * (3 - 2 * t)
                position[1] += obj['size'][1] * fraction
            else:
                color = [min(255, round(.35 * value + .65 * 255)) for value in color]
        result.append({'id': obj['id'], 'position': position, 'color': color,
                       'active': active, 'opening_fraction': fraction,
                       'trigger_frame': first})
    return result


def object_states(scene, clip, frame=None, *, enabled=False, start_frame=0):
    """Display states for this exact native frame; never inspect future roots."""
    _options(enabled, start_frame)
    if clip is None:
        roots = np.empty((1, 0, 3))
        return _states(scene, roots, -1, 20, enabled=False, start_frame=start_frame)
    if frame is None:
        frame = clip.frames - 1
    if type(frame) is not int or not 0 <= frame < clip.frames:
        raise ValueError('Reaction frame is outside the native clip')
    return _states(scene, clip.positions[:, :, 0, :], frame, clip.fps,
                   enabled=enabled, start_frame=start_frame)


def evaluated_scene(scene, clip, frame=None, *, enabled=False, start_frame=0):
    """Planning geometry matches the same evaluated transforms as rendering."""
    result = deepcopy(scene)
    for obj, state in zip(result['objects'], object_states(scene, clip, frame,
                                      enabled=enabled, start_frame=start_frame)):
        obj['position'], obj['color'] = state['position'], state['color']
    return result


def check_reactive_geometry(clip, scene, history=None, *, enabled=False, start_frame=0):
    """Reject solid overlaps against each candidate's actual door state.

    ``history`` must be the full committed prefix, not only the model context.
    A rejected proposal contributes no events. Floor/pair validation remains in
    CoreStudioSession. Static scenes retain the original batched check.
    """
    from interaction_scene_collision import scene_collision
    from studio_interaction_scene import adapt_studio_scene
    _options(enabled, start_frame)
    moving = enabled and any(_eligible(obj) and obj['kind'] == 'door' for obj in scene['objects'])
    if not moving:
        adapted = adapt_studio_scene(scene)
        for actor_id, positions in zip(clip.actor_ids, clip.positions):
            report = scene_collision(positions, 'core27', adapted['scene'], adapted['affordances'])
            if report['total_collision_frames']:
                raise ValueError(f'{actor_id} motion overlaps scene solid proxies; last good motion retained')
        return
    if history is not None and (history.actor_ids != clip.actor_ids or history.fps != clip.fps):
        raise ValueError('Reaction history must have matching actors and frame rate')
    offset = 0 if history is None else history.frames
    roots = clip.positions[:, :, 0, :]
    if history is not None:
        roots = np.concatenate((history.positions[:, :, 0, :], roots), axis=1)
    triggers = {obj['id']: _first_trigger(obj, roots, start_frame)
                for obj in scene['objects'] if _eligible(obj)}
    for local in range(clip.frames):
        document = deepcopy(scene)
        states = _states(scene, roots, offset + local, clip.fps,
                         enabled=True, start_frame=start_frame, triggers=triggers)
        for obj, state in zip(document['objects'], states):
            obj['position'] = state['position']
        adapted = adapt_studio_scene(document)
        for index, actor_id in enumerate(clip.actor_ids):
            report = scene_collision(clip.positions[index, local:local + 1], 'core27',
                                     adapted['scene'], adapted['affordances'])
            if report['total_collision_frames']:
                raise ValueError(f'{actor_id} motion overlaps scene solid proxies at reaction frame {offset + local}; last good motion retained')
