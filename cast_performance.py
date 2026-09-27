"""Separate immutable world-space native22 display tracks for a 1–3 actor cast.

These are composed display coordinates, not native model features or a Core/G1
continuation format. Source and authored segments belong in clip metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
from zipfile import BadZipFile, ZipFile

import numpy as np

from paired_scene import EMPTY_SCENE, json_copy, scene_copy

FPS = 30
MAX_ACTORS = 3
MAX_FRAMES = 1000
MAX_BYTES = 64 * 1024 * 1024
MAX_METADATA_BYTES = 1_500_000
FORMAT = 'stagezero_cast_performance'
SCHEMA = 'cast performance / world native22 / 30 fps'
EXTENSION = '.cast.stagezero.npz'
COLORS = ((52, 209, 220), (250, 178, 78), (182, 129, 242))


def validate_actor_ids(value):
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= MAX_ACTORS:
        raise ValueError('Cast performances require 1–3 actor tracks')
    ids = tuple(value)
    if any(not isinstance(item, str) or not 1 <= len(item) <= 80 or
           not item.replace('_', '').replace('-', '').isalnum() for item in ids):
        raise ValueError('Cast track IDs must be bounded alphanumeric identifiers')
    if len(set(ids)) != len(ids):
        raise ValueError('Cast track IDs must be unique')
    return ids


def validate_cast(cast, actor_ids):
    ids = validate_actor_ids(actor_ids)
    cast = json_copy(cast)
    if not isinstance(cast, list) or len(cast) != len(ids):
        raise ValueError('Cast records must match all performance tracks')
    for actor, identifier in zip(cast, ids):
        if not isinstance(actor, dict) or set(actor) != {'id', 'name', 'color'} or actor['id'] != identifier:
            raise ValueError('Cast records must preserve the exact track ID order')
        name, color = actor['name'], actor['color']
        if (not isinstance(name, str) or not 1 <= len(name.strip()) <= 80 or
                any(ord(char) < 32 for char in name) or not isinstance(color, list) or len(color) != 3 or
                any(type(channel) is not int or not 0 <= channel <= 255 for channel in color)):
            raise ValueError('Invalid cast performer name or color')
    return cast


def validated_segments(metadata, frames):
    segments = metadata.get('segments')
    if segments is None:
        return None
    if not isinstance(segments, list) or not 1 <= len(segments) <= 64:
        raise ValueError('Cast segments require 1–64 source records')
    output, next_start = [], 0
    required = {'label', 'start_frame', 'end_frame_exclusive', 'frames'}
    for segment in segments:
        if (not isinstance(segment, dict) or not required <= set(segment) or
                set(segment) - required - {'source', 'kind'}):
            raise ValueError('Invalid cast segment fields')
        start, end, count = segment['start_frame'], segment['end_frame_exclusive'], segment['frames']
        if (any(type(value) is not int for value in (start, end, count)) or
                start != next_start or not start < end <= frames or count != end - start):
            raise ValueError('Cast segments must contiguously cover every frame')
        for key, limit in (('label', 500), ('source', 80), ('kind', 80)):
            value = segment.get(key, 'mixed')
            if (not isinstance(value, str) or not 1 <= len(value.strip()) <= limit or
                    any(ord(char) < 32 for char in value)):
                raise ValueError('Cast segment labels must be bounded readable strings')
        output.append({'start': start, 'end': end, 'prompt': segment['label'],
                       'source': segment.get('source', 'mixed'), 'kind': segment.get('kind', 'mixed')})
        next_start = end
    if next_start != frames:
        raise ValueError('Cast segments must contiguously cover every frame')
    return output


@dataclass(frozen=True, init=False)
class CastPerformance:
    actor_ids: tuple[str, ...]
    joints: np.ndarray
    fps: int
    _metadata_json: str

    def __init__(self, actor_ids, joints, fps=FPS, metadata=None):
        ids = validate_actor_ids(actor_ids)
        array = np.asarray(joints)
        if (array.ndim != 4 or array.shape[1:] != (len(ids), 22, 3) or
                not 4 <= len(array) <= MAX_FRAMES or array.dtype.kind != 'f' or
                array.dtype.itemsize not in (4, 8) or array.nbytes > MAX_BYTES or not np.isfinite(array).all()):
            raise ValueError('Cast joints require 4–1000 finite floating-point frames [T,N,22,3]')
        if type(fps) is not int or fps != FPS:
            raise ValueError('Cast performances must remain at 30 fps')
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError('Cast metadata must be an object')
        metadata = json_copy(metadata or {}, limit=MAX_METADATA_BYTES)
        if metadata.get('fps', FPS) != FPS or ('frames' in metadata and
                (type(metadata['frames']) is not int or metadata['frames'] != len(array))):
            raise ValueError('Cast metadata timing disagrees with the display tracks')
        if metadata.get('render_hand_pose', 'relaxed') not in ('relaxed', 'fists'):
            raise ValueError('Cast hands must be relaxed or explicitly authored fists')
        validated_segments(metadata, len(array))
        object.__setattr__(self, 'actor_ids', ids)
        # Immutable bytes prevent callers from re-enabling array writes.
        object.__setattr__(self, 'joints', np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape))
        object.__setattr__(self, 'fps', FPS)
        object.__setattr__(self, '_metadata_json', json.dumps(metadata, allow_nan=False))

    @property
    def frames(self):
        return len(self.joints)

    @property
    def metadata(self):
        return json.loads(self._metadata_json)

    @property
    def segments(self):
        return validated_segments(self.metadata, self.frames)

    @property
    def source(self):
        return self.metadata.get('model', 'Composed cast performance')


def cast_from_performance(clip):
    """Apply reviewed plan names without changing any track identity or order."""
    metadata = clip.metadata
    plan = metadata.get('plan', {})
    if not isinstance(plan, dict):
        raise ValueError('Cast plan metadata must be an object')
    records = plan.get('actors', [])
    if not isinstance(records, list) or any(not isinstance(actor, dict) for actor in records):
        raise ValueError('Cast plan actors must be records')
    lookup = {actor.get('id'): actor for actor in records}
    cast = []
    for index, identifier in enumerate(clip.actor_ids):
        record = lookup.get(identifier, {})
        cast.append({'id': identifier, 'name': record.get('name', f'Actor {index + 1}'),
                     'color': record.get('color', list(COLORS[index]))})
    return validate_cast(cast, clip.actor_ids)


def encode_project(clip, cast, scene_document=None, frame=0):
    if not isinstance(clip, CastPerformance):
        raise ValueError('Expected a separate cast performance')
    cast = validate_cast(cast, clip.actor_ids)
    if type(frame) is not int or not 0 <= frame < clip.frames:
        raise ValueError('Cast project playhead is outside the performance')
    doc = {'format': FORMAT, 'version': 1, 'schema': SCHEMA, 'fps': FPS,
           'actor_ids': list(clip.actor_ids), 'cast': cast,
           'scene_document': scene_copy(EMPTY_SCENE if scene_document is None else scene_document),
           'frame': frame, 'clip_metadata': clip.metadata}
    encoded = json.dumps(json_copy(doc, limit=MAX_METADATA_BYTES), allow_nan=False)
    stream = io.BytesIO()
    np.savez_compressed(stream, joints=clip.joints, metadata=np.array(encoded))
    content = stream.getvalue()
    if len(content) > MAX_BYTES:
        raise ValueError('Cast project exceeds its 64 MiB archive limit')
    return content


def decode_project(content):
    if not isinstance(content, bytes) or not 1 <= len(content) <= MAX_BYTES:
        raise ValueError('Cast project must be nonempty bytes up to 64 MiB')
    try:
        with ZipFile(io.BytesIO(content)) as zipped:
            infos = zipped.infolist()
            if (len(infos) != 2 or {entry.filename for entry in infos} != {'joints.npy', 'metadata.npy'} or
                    sum(entry.file_size for entry in infos) > MAX_BYTES):
                raise ValueError('Invalid or oversized cast archive members')
            for entry in infos:
                with zipped.open(entry) as member:
                    version = np.lib.format.read_magic(member)
                    reader = {(1, 0): np.lib.format.read_array_header_1_0,
                              (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None:
                        raise ValueError('Unsupported cast array encoding')
                    shape, _, dtype = reader(member)
                if entry.filename == 'metadata.npy':
                    valid = shape == () and dtype.kind in 'US' and dtype.itemsize <= MAX_METADATA_BYTES * 4
                else:
                    valid = (len(shape) == 4 and 4 <= shape[0] <= MAX_FRAMES and
                             1 <= shape[1] <= MAX_ACTORS and shape[2:] == (22, 3) and
                             dtype.kind == 'f' and dtype.itemsize in (4, 8))
                if not valid:
                    raise ValueError('Invalid cast archive array shape or dtype')
        with np.load(io.BytesIO(content), allow_pickle=False) as data:
            doc = json_copy(json.loads(data['metadata'].item()), limit=MAX_METADATA_BYTES)
            joints = data['joints'].copy()
        fields = {'format', 'version', 'schema', 'fps', 'actor_ids', 'cast',
                  'scene_document', 'frame', 'clip_metadata'}
        if (not isinstance(doc, dict) or set(doc) != fields or doc['format'] != FORMAT or
                type(doc['version']) is not int or doc['version'] != 1 or doc['schema'] != SCHEMA or
                type(doc['fps']) is not int or doc['fps'] != FPS):
            raise ValueError('Expected a separate cast performance project')
        clip = CastPerformance(doc['actor_ids'], joints, metadata=doc['clip_metadata'])
        cast = validate_cast(doc['cast'], clip.actor_ids)
        if type(doc['frame']) is not int or not 0 <= doc['frame'] < clip.frames:
            raise ValueError('Cast project playhead is outside the performance')
        return clip, cast, scene_copy(doc['scene_document']), doc['frame']
    except (BadZipFile, KeyError, TypeError, EOFError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid cast performance archive') from exc
