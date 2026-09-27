"""Exact native paired motion and portable projects, independent of Core/G1."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile, BadZipFile

import numpy as np
from paired_scene import json_copy, scene_copy, EMPTY_SCENE

FPS = 30
FORMAT = 'stagezero_native_pair'
SCHEMA = 'native22 / 30 fps'
MAX_FRAMES = 1000
MAX_BYTES = 32_000_000
MAX_CAST = 8


def _array(value, tail, name):
    array = np.asarray(value)
    if (array.ndim != len(tail) + 1 or array.shape[1:] != tail
            or not 4 <= len(array) <= MAX_FRAMES or array.dtype.kind != 'f'
            or array.dtype.itemsize not in (4, 8) or not np.isfinite(array).all()):
        raise ValueError(f'{name} must be finite native floating point [T,{tail}], up to {MAX_FRAMES} frames')
    # An immutable bytes owner prevents callers re-enabling writes on the array.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def validated_segments(metadata, frames):
    """Normalize disclosed source segments only after complete coverage checks.

    Labels are provenance, not verification that transitions are physically valid.
    Unknown model labels never become implicit native InterGen provenance.
    """
    if 'segments' not in metadata:
        return None
    segments = metadata['segments']
    if not isinstance(segments, list) or not 1 <= len(segments) <= 64:
        raise ValueError('Native display segments must contain 1–64 records')
    normalized = []
    expected_start = 0
    required = {'label', 'start_frame', 'end_frame_exclusive', 'frames'}
    optional = {'source', 'kind'}
    for segment in segments:
        if not isinstance(segment, dict) or not required <= set(segment) or set(segment) - required - optional:
            raise ValueError('Invalid native display segment fields')
        start, end, count = segment['start_frame'], segment['end_frame_exclusive'], segment['frames']
        if (any(type(value) is not int for value in (start, end, count))
                or start != expected_start or not start < end <= frames or count != end-start):
            raise ValueError('Native display segments must contiguously cover the complete timeline')
        for key, limit in (('label', 500), ('source', 80), ('kind', 80)):
            value = segment.get(key, 'mixed' if key == 'source' else 'mixed_segment')
            if (not isinstance(value, str) or not 1 <= len(value.strip()) <= limit
                    or any(ord(char) < 32 for char in value)):
                raise ValueError('Native display segment labels must be bounded readable strings')
        normalized.append({'start': start, 'end': end, 'prompt': segment['label'],
                           'source': segment.get('source', 'mixed'),
                           'kind': segment.get('kind', 'mixed_segment')})
        expected_start = end
    if expected_start != frames:
        raise ValueError('Native display segments must contiguously cover the complete timeline')
    return normalized


@dataclass(frozen=True)
class NativePairClip:
    joints: np.ndarray
    features: np.ndarray | None = None
    metadata: dict | None = None

    def __post_init__(self):
        joints = _array(self.joints, (2, 22, 3), 'joints')
        features = None if self.features is None else _array(self.features, (2, 262), 'features')
        if features is not None and len(features) != len(joints):
            raise ValueError('Native features and joints must have identical frame counts')
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError('Native clip metadata must be an object')
        metadata = json_copy(self.metadata or {})
        if metadata.get('render_hand_pose', 'relaxed') not in ('relaxed', 'fists'):
            raise ValueError('Render hand pose must be relaxed or fists')
        validated_segments(metadata, len(joints))
        if 'frames' in metadata and (type(metadata['frames']) is not int or metadata['frames'] != len(joints)):
            raise ValueError('Native display metadata frame count disagrees with joints')
        if metadata.get('fps', FPS) != FPS:
            raise ValueError('Native paired motion must remain at 30 fps')
        object.__setattr__(self, 'joints', joints)
        object.__setattr__(self, 'features', features)
        object.__setattr__(self, 'metadata', metadata)

    @property
    def frames(self):
        return len(self.joints)

    @property
    def fps(self):
        return FPS

    @property
    def segments(self):
        return validated_segments(self.metadata, self.frames)

    @property
    def source(self):
        return self.metadata.get('model', 'Native paired model')


def _read_npz(content, *, project=False):
    if not isinstance(content, bytes) or not 1 <= len(content) <= MAX_BYTES:
        raise ValueError('Native archive must be nonempty and no larger than 32 MB')
    try:
        with ZipFile(io.BytesIO(content)) as zipped:
            infos = zipped.infolist()
            names = [entry.filename for entry in infos]
            allowed = {'joints.npy', 'features.npy', 'metadata.npy'}
            if not project:
                allowed |= {'smoothed_joints.npy'}  # Deliberately ignored; original joints are authoritative.
            if (len(names) != len(set(names)) or set(names) - allowed
                    or not {'joints.npy', 'metadata.npy'} <= set(names)
                    or sum(i.file_size for i in infos) > MAX_BYTES):
                raise ValueError('Invalid or oversized native archive members')
            for entry in infos:
                with zipped.open(entry) as member:
                    version = np.lib.format.read_magic(member)
                    reader = {(1, 0): np.lib.format.read_array_header_1_0,
                              (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None:
                        raise ValueError('Unsupported native array encoding')
                    shape, _, dtype = reader(member)
                if entry.filename == 'metadata.npy':
                    valid = shape == () and dtype.kind in 'US' and dtype.itemsize <= 6_000_000
                else:
                    tail = (2, 262) if entry.filename == 'features.npy' else (2, 22, 3)
                    valid = (len(shape) == len(tail) + 1 and shape[1:] == tail
                             and 4 <= shape[0] <= MAX_FRAMES and dtype.kind == 'f'
                             and dtype.itemsize in (4, 8))
                if not valid:
                    raise ValueError('Invalid native array shape or dtype')
        with np.load(io.BytesIO(content), allow_pickle=False) as data:
            metadata = json.loads(data['metadata'].item())
            if not isinstance(metadata, dict):
                raise ValueError('Native metadata must be an object')
            return data['joints'].copy(), data['features'].copy() if 'features' in data else None, metadata
    except (BadZipFile, KeyError, TypeError, EOFError, OSError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid native paired archive') from exc


def load_source(source):
    content = source if isinstance(source, bytes) else Path(source).read_bytes()
    joints, features, metadata = _read_npz(content)
    if 'format' in metadata:
        raise ValueError('Use project loading for saved studio projects')
    metadata = dict(metadata, source_sha256=hashlib.sha256(content).hexdigest())
    return NativePairClip(joints, features, metadata)


def validate_cast(cast, pair):
    cast = json_copy(cast)
    if not isinstance(cast, list) or not 2 <= len(cast) <= MAX_CAST:
        raise ValueError('Native interaction cast needs 2–8 actors')
    ids = []
    for actor in cast:
        if not isinstance(actor, dict) or set(actor) != {'id', 'name', 'color'}:
            raise ValueError('Invalid native cast record')
        identifier, name, color = actor['id'], actor['name'], actor['color']
        if (not isinstance(identifier, str) or not 1 <= len(identifier) <= 80
                or not identifier.replace('_', '').replace('-', '').isalnum()
                or not isinstance(name, str) or not 1 <= len(name.strip()) <= 80
                or any(ord(c) < 32 for c in name)
                or not isinstance(color, list) or len(color) != 3
                or any(type(c) is not int or not 0 <= c <= 255 for c in color)):
            raise ValueError('Invalid native cast name, ID or color')
        ids.append(identifier)
    if len(set(ids)) != len(ids):
        raise ValueError('Native actor IDs must be unique')
    if not isinstance(pair, (list, tuple)) or len(pair) != 2 or pair[0] == pair[1] or any(i not in ids for i in pair):
        raise ValueError('Choose two distinct actors in the cast')
    return cast, tuple(pair)


def validate_placement(value):
    if not isinstance(value, dict) or set(value) != {'x', 'z', 'yaw_degrees'}:
        raise ValueError('Placement requires x, z and yaw_degrees')
    if any(type(v) not in (int, float) or not np.isfinite(v) for v in value.values()):
        raise ValueError('Pair placement must be finite numbers')
    if any(abs(value[k]) > 1000 for k in ('x', 'z')) or abs(value['yaw_degrees']) > 360:
        raise ValueError('Pair placement is limited to 1000 m and ±360 degrees')
    return {key: float(value[key]) for key in ('x', 'z', 'yaw_degrees')}


def encode_project(clip, cast, pair, scene_document=None, frame=0, placement=None):
    if not isinstance(clip, NativePairClip):
        raise ValueError('Expected an exact native pair clip')
    cast, pair = validate_cast(cast, pair)
    if type(frame) is not int or not 0 <= frame < clip.frames:
        raise ValueError('Native frame is outside the clip')
    metadata = {'format': FORMAT, 'version': 1, 'fps': FPS, 'schema': SCHEMA,
                'cast': cast, 'pair': list(pair), 'scene_document': scene_copy(scene_document or EMPTY_SCENE),
                'frame': frame, 'clip_metadata': clip.metadata,
                'placement': validate_placement(placement or {'x': 0., 'z': 0., 'yaw_degrees': 0.})}
    stream = io.BytesIO()
    fields = {'joints': clip.joints, 'metadata': np.array(json.dumps(json_copy(metadata), allow_nan=False))}
    if clip.features is not None:
        fields['features'] = clip.features
    np.savez_compressed(stream, **fields)
    if len(stream.getvalue()) > MAX_BYTES:
        raise ValueError('Native project exceeds size limit')
    return stream.getvalue()


def decode_project(content):
    joints, features, doc = _read_npz(content, project=True)
    expected = {'format', 'version', 'fps', 'schema', 'cast', 'pair', 'scene_document', 'frame', 'clip_metadata'}
    if (set(doc) not in (expected, expected | {'placement'}) or doc['format'] != FORMAT or type(doc['version']) is not int
            or doc['version'] != 1 or doc['fps'] != FPS or doc['schema'] != SCHEMA):
        raise ValueError('Expected a separate native22 30 fps paired project')
    cast, pair = validate_cast(doc['cast'], doc['pair'])
    clip = NativePairClip(joints, features, doc['clip_metadata'])
    if type(doc['frame']) is not int or not 0 <= doc['frame'] < clip.frames:
        raise ValueError('Native project frame is outside the clip')
    return clip, cast, pair, scene_copy(doc['scene_document']), doc['frame'], validate_placement(doc.get('placement', {'x': 0., 'z': 0., 'yaw_degrees': 0.}))
