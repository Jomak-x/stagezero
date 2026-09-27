"""Strict paired InterGen research requests and a separate portable archive.

No native Core history, production director, model loading, or network calls.
"""
from __future__ import annotations

import io
import json
import uuid
from zipfile import BadZipFile, ZipFile

import numpy as np

from realtime_clip import CanonicalClip
from scene_composition import validate_scene

ACTOR_IDS = ('actor_1', 'actor_2')
FORMAT = 'stagezero_paired_research'
LICENSE = 'CC BY-NC-SA 4.0'
SCHEMA = 'InterGen research / retargeted Core27 / 20 fps'
MAX_ARCHIVE_BYTES = 8_000_000
MAX_METADATA_BYTES = 1_500_000
EMPTY_SCENE = {'version': 2, 'name': 'Paired research stage', 'objects': [],
               'effects': [], 'lighting': 'neutral'}


def json_copy(value, *, limit=MAX_METADATA_BYTES):
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Paired research metadata must be finite JSON') from exc
    if len(encoded.encode('utf-8')) > limit:
        raise ValueError('Paired research metadata exceeds its size limit')
    return json.loads(encoded)


def scene_copy(value):
    value = json_copy(value, limit=1_000_000)
    return validate_scene(value)


def validate_request(body):
    fields = {'request_id', 'stage_kind', 'frames', 'prompt', 'actor_ids', 'seed',
              'pair_sequence_id', 'source_start_frame', 'source_total_frames'}
    if not isinstance(body, dict) or set(body) != fields:
        raise ValueError('Invalid paired research request fields')
    for key in ('request_id', 'pair_sequence_id'):
        value = body[key]
        if not isinstance(value, str) or not 1 <= len(value) <= 100 or any(ord(c) < 32 for c in value):
            raise ValueError('Paired request identifiers must be short printable strings')
    if body['stage_kind'] != 'paired' or body['actor_ids'] != list(ACTOR_IDS):
        raise ValueError('Paired research requires exactly actor_1 and actor_2')
    if type(body['frames']) is not int or body['frames'] not in (40, 80, 120):
        raise ValueError('Paired frames must be 40, 80, or 120')
    if type(body['seed']) is not int or not 0 <= body['seed'] < 2**32:
        raise ValueError('Paired seed must be a uint32')
    if (not isinstance(body['prompt'], str) or not 1 <= len(body['prompt'].strip()) <= 500
            or any(ord(c) < 32 and c not in '\n\t' for c in body['prompt'])):
        raise ValueError('Paired prompt must contain 1–500 readable characters')
    if (type(body['source_start_frame']) is not int or body['source_start_frame'] != 0
            or type(body['source_total_frames']) is not int or body['source_total_frames'] != body['frames']):
        raise ValueError('Paired research must request one complete bounded source sample')
    return json_copy(body)


def request_body(prompt, seed, frames=120):
    identifier = uuid.uuid4().hex
    return validate_request({'request_id': 'paired-' + identifier, 'stage_kind': 'paired',
            'frames': frames, 'prompt': prompt.strip() if isinstance(prompt, str) else prompt,
            'actor_ids': list(ACTOR_IDS), 'seed': seed, 'pair_sequence_id': 'sequence-' + identifier,
            'source_start_frame': 0, 'source_total_frames': frames})


def validate_chunk(clip, body, index):
    if (not isinstance(clip, CanonicalClip) or clip.source != 'intergen'
            or clip.native_features is not None or clip.frames != 40 or clip.fps != 20
            or clip.actor_ids != ACTOR_IDS or index >= body['frames'] // 40):
        raise ValueError('Unexpected paired research source, actors, or chunk size')
    expected = {'request_id': body['request_id'], 'stage_kind': 'paired',
                'chunk_index': index, 'start_frame': index * 40, 'frames': 40,
                'pair_sequence_id': body['pair_sequence_id'], 'source_start_frame': index * 40,
                'source_total_frames': body['frames'], 'seed': body['seed']}
    if any(type(clip.metadata.get(key)) is not type(value) or clip.metadata.get(key) != value
           for key, value in expected.items()):
        raise ValueError('Paired research chunk does not match the requested source sequence')


def join_chunks(chunks, body, *, service_metadata=None):
    body = validate_request(body)
    if not isinstance(chunks, (tuple, list)) or len(chunks) != body['frames'] // 40:
        raise ValueError('Only a complete paired research sequence can be committed')
    for index, clip in enumerate(chunks):
        validate_chunk(clip, body, index)
    if service_metadata is not None and not isinstance(service_metadata, dict):
        raise ValueError('Paired service provenance must be a JSON object')
    metadata = {'version': 1, 'research_only': True, 'license': LICENSE,
                'source_format': 'InterGen 30 fps paired joints retargeted to Core27 at 20 fps',
                'native_features_available': False, 'physical_contact_verified': False,
                'scene_conditioned': False, 'request': body,
                'chunks': [dict(clip.metadata) for clip in chunks],
                'service_metadata': json_copy(service_metadata or {}, limit=100_000)}
    return CanonicalClip(np.concatenate([c.positions for c in chunks], axis=1),
                         np.concatenate([c.rotations for c in chunks], axis=1),
                         20, ACTOR_IDS, 'intergen', metadata)


def validate_clip(clip):
    if (not isinstance(clip, CanonicalClip) or clip.source != 'intergen'
            or clip.native_features is not None or clip.actor_ids != ACTOR_IDS
            or clip.frames not in (40, 80, 120) or clip.fps != 20):
        raise ValueError('Expected a complete non-native paired InterGen research clip')
    metadata = json_copy(dict(clip.metadata))
    required = {'version', 'research_only', 'license', 'source_format', 'native_features_available',
                'physical_contact_verified', 'scene_conditioned', 'request', 'chunks', 'service_metadata'}
    if (set(metadata) != required or type(metadata['version']) is not int or metadata['version'] != 1
            or metadata['research_only'] is not True or metadata['license'] != LICENSE
            or metadata['native_features_available'] is not False
            or metadata['physical_contact_verified'] is not False or metadata['scene_conditioned'] is not False
            or metadata['source_format'] != 'InterGen 30 fps paired joints retargeted to Core27 at 20 fps'
            or not isinstance(metadata['service_metadata'], dict)):
        raise ValueError('Paired research provenance is missing or inconsistent')
    body = validate_request(metadata['request'])
    if body['frames'] != clip.frames or not isinstance(metadata['chunks'], list) or len(metadata['chunks']) * 40 != clip.frames:
        raise ValueError('Paired research archive is incomplete')
    json_copy(metadata['service_metadata'], limit=100_000)
    for index, chunk_metadata in enumerate(metadata['chunks']):
        chunk = CanonicalClip(clip.positions[:, index * 40:(index + 1) * 40],
                              clip.rotations[:, index * 40:(index + 1) * 40],
                              20, ACTOR_IDS, 'intergen', chunk_metadata)
        validate_chunk(chunk, body, index)
    return metadata


def check_scene_geometry(clip, scene_document):
    """Check authored floor support and scene solids, without rejecting partner proximity."""
    from studio_interaction_scene import adapt_studio_scene
    from interaction_scene_collision import scene_collision
    from realtime_navigation import validate_ground_path
    # Torso centres catch gross body interpenetration while preserving intended
    # hand contact. This is a conservative proxy, not a mesh/contact solver.
    torso_distance = np.linalg.norm(clip.positions[0, :, :5, None, :]
                                    - clip.positions[1, :, None, :5, :], axis=-1)
    if float(torso_distance.min()) < .35:
        raise ValueError('Paired torso overlap below 0.35 m proxy clearance; last good motion retained')
    adapted = adapt_studio_scene(scene_document)
    for index, actor_id in enumerate(clip.actor_ids):
        root_path = clip.positions[index, :, 0, :][:, [0, 2]]
        validate_ground_path(adapted['scene'], root_path, actor_radius_m=.28)
        report = scene_collision(clip.positions[index], 'core27', adapted['scene'], adapted['affordances'])
        if report['total_collision_frames']:
            raise ValueError(f'{actor_id} research motion overlaps scene solid proxies; last good motion retained')
    # Close pair motion and hand contact are intentional. Do not apply Core's
    # 0.65 m root-disc gate or an all-joint distance threshold to this sample.


def encode_project(clip, scene_document, frame=0):
    metadata = validate_clip(clip)
    scene = scene_copy(scene_document)
    if type(frame) is not int or not 0 <= frame < clip.frames:
        raise ValueError('Saved paired playhead is outside the clip')
    doc = {'version': 1, 'format': FORMAT, 'source': 'intergen', 'schema': SCHEMA,
           'research_only': True, 'license': LICENSE, 'fps': 20,
           'actor_ids': list(ACTOR_IDS), 'scene_document': scene, 'frame': frame,
           'clip_metadata': metadata}
    encoded = json.dumps(json_copy(doc), allow_nan=False)
    stream = io.BytesIO()
    np.savez_compressed(stream, positions=clip.positions, rotations=clip.rotations,
                        metadata=np.array(encoded))
    content = stream.getvalue()
    if len(content) > MAX_ARCHIVE_BYTES:
        raise ValueError('Paired archive exceeds its size limit')
    return content


def decode_project(content):
    if not isinstance(content, bytes) or not 1 <= len(content) <= MAX_ARCHIVE_BYTES:
        raise ValueError('Paired research archive must be nonempty and at most 8 MB')
    try:
        with ZipFile(io.BytesIO(content)) as archive:
            infos = archive.infolist()
            names = {'positions.npy', 'rotations.npy', 'metadata.npy'}
            if (len(infos) != 3 or {entry.filename for entry in infos} != names
                    or sum(entry.file_size for entry in infos) > MAX_ARCHIVE_BYTES):
                raise ValueError('Invalid or oversized paired research archive members')
            for entry in infos:
                with archive.open(entry) as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                    else:
                        raise ValueError('Unsupported paired array encoding')
                if entry.filename == 'metadata.npy':
                    valid = shape == () and dtype.kind in 'US' and dtype.itemsize <= MAX_METADATA_BYTES * 4
                else:
                    tail = (27, 3) if entry.filename == 'positions.npy' else (27, 3, 3)
                    valid = (len(shape) == len(tail) + 2 and shape[0] == 2
                             and shape[1] in (40, 80, 120) and shape[2:] == tail
                             and dtype.kind == 'f' and dtype.itemsize <= 8)
                if not valid:
                    raise ValueError('Invalid paired research array shape or dtype')
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            doc = json.loads(archive['metadata'].item())
            fields = {'version', 'format', 'source', 'schema', 'research_only', 'license', 'fps',
                      'actor_ids', 'scene_document', 'frame', 'clip_metadata'}
            if (not isinstance(doc, dict) or set(doc) != fields or type(doc['version']) is not int
                    or doc['version'] != 1 or doc['format'] != FORMAT or doc['source'] != 'intergen'
                    or doc['schema'] != SCHEMA or doc['research_only'] is not True or doc['license'] != LICENSE
                    or type(doc['fps']) is not int or doc['fps'] != 20 or doc['actor_ids'] != list(ACTOR_IDS)):
                raise ValueError('Expected a separate paired InterGen research archive')
            json_copy(doc)
            clip = CanonicalClip(archive['positions'], archive['rotations'], 20, ACTOR_IDS,
                                 'intergen', doc['clip_metadata'])
        validate_clip(clip)
        scene = scene_copy(doc['scene_document'])
        check_scene_geometry(clip, scene)
        frame = doc['frame']
        if type(frame) is not int or not 0 <= frame < clip.frames:
            raise ValueError('Saved paired playhead is outside the clip')
        return clip, scene, frame
    except (BadZipFile, KeyError, TypeError, IndexError, EOFError, OSError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid paired research archive') from exc
