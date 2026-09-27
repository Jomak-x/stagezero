"""Versioned takes with actual motion, not prompt-based regeneration."""
from dataclasses import dataclass, field
from copy import deepcopy
import io
import json
import math
import zipfile
import numpy as np
from live_motion import MODEL
from camera_model import validate_cameras, validate_camera_cuts

MAX_FRAMES = 15000  # Ten minutes per take at native 25 fps.
MAX_TOTAL_FRAMES = 30000
MAX_TAKES = 12
MAX_ARCHIVE_BYTES = 220_000_000
MAX_METADATA_CHARS = 1_000_000
MAX_DIALOGUE_CUES = 128
MAX_AUDIO_ASSETS = 128
MAX_AUDIO_BYTES = 8_000_000
MAX_PROJECT_AUDIO_BYTES = 32_000_000


@dataclass
class Take:
    id: str
    name: str
    positions: np.ndarray
    rotations: np.ndarray
    motion: np.ndarray
    segments: list = field(default_factory=list)
    parent: str | None = None
    branch_frame: int | None = None
    events: list = field(default_factory=list)
    camera_cuts: list = field(default_factory=list)
    dialogue: list = field(default_factory=list)
    audio_assets: dict = field(default_factory=dict)

    def prefix(self, stop):
        """A copy of all metadata before an exclusive frame boundary."""
        segments = [dict(s, end=min(s['end'], stop)) for s in self.segments if s['start'] < stop]
        events = [dict(e) for e in self.events if e['frame'] < stop]
        return segments, events

    def dialogue_prefix(self, stop):
        return slice_dialogue(self, 0, stop)


def slice_dialogue(take, start, end):
    """Clip cues to [start, end), rebasing frames and the audio offset."""
    cues = []
    for cue in take.dialogue:
        first = max(start, cue['start_frame'])
        last = min(end, cue['end_frame'])
        if first >= last:
            continue
        clipped = deepcopy(cue)
        clipped['start_frame'] = first - start
        clipped['end_frame'] = last - start
        if first > cue['start_frame']:
            clipped['audio_offset_seconds'] = cue.get('audio_offset_seconds', 0.) + (first - cue['start_frame']) / 25
        cues.append(clipped)
    assets = {cue['audio_id']: take.audio_assets[cue['audio_id']] for cue in cues}
    return cues, assets


def merge_dialogue(parts):
    """Combine (take, frame offset) pairs, renaming conflicting audio IDs."""
    cues, assets = [], {}
    for take, offset in parts:
        local_ids = {}
        for cue in take.dialogue:
            audio_id = cue['audio_id']
            if audio_id not in local_ids:
                candidate = audio_id
                suffix = 2
                while candidate in assets and assets[candidate] != take.audio_assets[audio_id]:
                    candidate = f'{audio_id[:70]}-{suffix}'
                    suffix += 1
                local_ids[audio_id] = candidate
                assets[candidate] = take.audio_assets[audio_id]
            shifted = deepcopy(cue)
            shifted['audio_id'] = local_ids[audio_id]
            shifted['start_frame'] += offset
            shifted['end_frame'] += offset
            cues.append(shifted)
    return cues, assets


def validate_take(take):
    p, r, m = take.positions, take.rotations, take.motion
    n = len(p)
    if not 4 <= n <= MAX_FRAMES or p.shape != (n, 34, 3) or r.shape != (n, 34, 3, 3) or m.shape != (n, 414):
        raise ValueError('Incompatible G1 take dimensions or duration')
    if any(a.dtype.kind != 'f' or not np.isfinite(a).all() for a in (p, r, m)):
        raise ValueError('Take contains invalid motion values')
    if not np.allclose(r @ np.swapaxes(r, -1, -2), np.eye(3), atol=.02) or not np.allclose(np.linalg.det(r), 1, atol=.02):
        raise ValueError('Take contains invalid rotations')
    if not isinstance(take.id, str) or not 1 <= len(take.id) <= 80 or not isinstance(take.name, str) or not 1 <= len(take.name) <= 80:
        raise ValueError('Invalid take name or identifier')
    end = 0
    for s in take.segments:
        if not isinstance(s, dict) or type(s.get('start')) is not int or type(s.get('end')) is not int or s['start'] != end or not end < s['end'] <= n:
            raise ValueError('Invalid segment timeline')
        if not isinstance(s.get('prompt'), str) or len(s['prompt']) > 500:
            raise ValueError('Invalid segment instruction')
        if 'timing_mode' in s and s['timing_mode'] not in ('auto', 'fixed'):
            raise ValueError('Invalid segment timing mode')
        end = s['end']
    if end != n:
        raise ValueError('Segments do not cover the take')
    for e in take.events:
        if not isinstance(e, dict) or e.get('type') != 'gate_open' or type(e.get('frame')) is not int or not 0 <= e['frame'] < n:
            raise ValueError('Invalid scene event')
    validate_camera_cuts(take.camera_cuts, n)
    if not isinstance(take.dialogue, list) or len(take.dialogue) > MAX_DIALOGUE_CUES:
        raise ValueError('Invalid dialogue cue count')
    if not isinstance(take.audio_assets, dict) or len(take.audio_assets) > MAX_AUDIO_ASSETS:
        raise ValueError('Invalid audio asset count')
    audio_bytes = 0
    for audio_id, content in take.audio_assets.items():
        if (not isinstance(audio_id, str) or not 1 <= len(audio_id) <= 80 or
                not isinstance(content, bytes) or not 0 < len(content) <= MAX_AUDIO_BYTES):
            raise ValueError('Invalid dialogue audio asset')
        audio_bytes += len(content)
    if audio_bytes > MAX_PROJECT_AUDIO_BYTES:
        raise ValueError('Take audio exceeds the supported budget')
    for cue in take.dialogue:
        if not isinstance(cue, dict) or not isinstance(cue.get('text'), str) or not 1 <= len(cue['text']) <= 1000:
            raise ValueError('Invalid dialogue text')
        if not isinstance(cue.get('voice_id'), str) or not 1 <= len(cue['voice_id']) <= 128:
            raise ValueError('Invalid dialogue voice')
        if not isinstance(cue.get('audio_id'), str) or cue['audio_id'] not in take.audio_assets:
            raise ValueError('Dialogue cue has no bundled audio')
        first, last = cue.get('start_frame'), cue.get('end_frame')
        if type(first) is not int or type(last) is not int or not 0 <= first < last <= n:
            raise ValueError('Invalid dialogue timeline')
        offset = cue.get('audio_offset_seconds', 0.)
        if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset) or not 0 <= offset <= MAX_FRAMES / 25:
            raise ValueError('Invalid dialogue audio offset')
        for optional_id in ('character_id', 'line_id'):
            if optional_id in cue and (not isinstance(cue[optional_id], str) or not 1 <= len(cue[optional_id]) <= 128):
                raise ValueError(f'Invalid dialogue {optional_id}')


def validate_provenance(takes):
    for take in takes.values():
        if take.parent is None:
            if take.branch_frame is not None:
                raise ValueError('Invalid branch provenance')
        elif (not isinstance(take.parent, str) or take.parent not in takes or take.parent == take.id or
              type(take.branch_frame) is not int or
              not 3 <= take.branch_frame < len(takes[take.parent].positions)):
            raise ValueError('Invalid branch provenance')
    for take in takes.values():
        seen = set()
        current = take
        while current.parent is not None:
            if current.id in seen:
                raise ValueError('Cyclic branch provenance')
            seen.add(current.id)
            current = takes[current.parent]


def encode_project(takes, active, frame, scene, cameras=None):
    cameras = validate_cameras([] if cameras is None else cameras)
    camera_ids = {camera['id'] for camera in cameras}
    if not 0 <= len(takes) <= MAX_TAKES:
        raise ValueError('Invalid take count')
    if takes:
        if active not in takes or type(frame) is not int or not 0 <= frame < len(takes[active].positions):
            raise ValueError('Invalid saved playhead')
    elif active is not None or type(frame) is not int or frame != 0:
        raise ValueError('Invalid saved playhead')
    if sum(len(t.positions) for t in takes.values()) > MAX_TOTAL_FRAMES:
        raise ValueError('Project exceeds the supported motion budget')
    data, items = {}, []
    audio_bytes = 0
    for i, t in enumerate(takes.values()):
        validate_take(t)
        cuts = validate_camera_cuts(t.camera_cuts, len(t.positions), camera_ids)
        key = f't{i}'
        for field_name in ('positions', 'rotations', 'motion'):
            data[key + '_' + field_name] = getattr(t, field_name)
        audio_ids = list(t.audio_assets)
        for j, audio_id in enumerate(audio_ids):
            content = t.audio_assets[audio_id]
            audio_bytes += len(content)
            data[f'{key}_audio_{j}'] = np.frombuffer(content, dtype=np.uint8)
        items.append(dict(key=key, id=t.id, name=t.name, segments=t.segments,
                          parent=t.parent, branch_frame=t.branch_frame, events=t.events,
                          camera_cuts=cuts, dialogue=t.dialogue, audio_ids=audio_ids))
    if audio_bytes > MAX_PROJECT_AUDIO_BYTES:
        raise ValueError('Project audio exceeds the supported budget')
    validate_provenance(takes)
    manifest = dict(version=2, model=MODEL, fps=25, active=active, frame=int(frame),
                    scene=scene, cameras=cameras, takes=items)
    raw = json.dumps(manifest, allow_nan=False)
    if len(raw) > MAX_METADATA_CHARS:
        raise ValueError('Project metadata is too large')
    data['manifest'] = np.array(raw)
    out = io.BytesIO()
    np.savez_compressed(out, **data)
    return out.getvalue()


def decode_project(content, *, include_cameras=False):
    """Validate before replacing any live state. Never unpickle or extract paths."""
    if len(content) > MAX_ARCHIVE_BYTES:
        raise ValueError('Project file is too large')
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        entries = archive.infolist()
        if len(entries) > MAX_TAKES * (3 + MAX_AUDIO_ASSETS) + 1 or sum(e.file_size for e in entries) > MAX_ARCHIVE_BYTES:
            raise ValueError('Expanded project is too large')
        if len({e.filename for e in entries}) != len(entries):
            raise ValueError('Duplicate project entries')
    with np.load(io.BytesIO(content), allow_pickle=False) as data:
        raw = str(data['manifest'])
        if len(raw) > MAX_METADATA_CHARS:
            raise ValueError('Project metadata is too large')
        doc = json.loads(raw)
        if not isinstance(doc, dict) or type(doc.get('version')) is not int or doc['version'] not in (1, 2) or doc.get('model') != MODEL or doc.get('fps') != 25:
            raise ValueError('Unsupported project version or skeleton')
        cameras = validate_cameras(doc.get('cameras') if doc['version'] == 2 else [])
        camera_ids = {camera['id'] for camera in cameras}
        items = doc.get('takes')
        if not isinstance(items, list) or not 0 <= len(items) <= MAX_TAKES:
            raise ValueError('Invalid take count')
        takes, total, audio_bytes = {}, 0, 0
        for i, item in enumerate(items):
            if item['key'] != f't{i}' or item['id'] in takes:
                raise ValueError('Invalid take identifiers')
            arrays = [data[f't{i}_{name}'].copy() for name in ('positions', 'rotations', 'motion')]
            cuts = item.get('camera_cuts') if doc['version'] == 2 else []
            audio_ids = item.get('audio_ids', [])
            if (not isinstance(audio_ids, list) or len(audio_ids) > MAX_AUDIO_ASSETS or
                    any(not isinstance(audio_id, str) for audio_id in audio_ids) or
                    len(set(audio_ids)) != len(audio_ids)):
                raise ValueError('Invalid audio asset index')
            assets = {}
            for j, audio_id in enumerate(audio_ids):
                if not 1 <= len(audio_id) <= 80:
                    raise ValueError('Invalid audio asset identifier')
                blob = data[f't{i}_audio_{j}']
                if blob.ndim != 1 or blob.dtype != np.uint8 or not 0 < blob.size <= MAX_AUDIO_BYTES:
                    raise ValueError('Invalid audio asset data')
                audio_bytes += blob.size
                if audio_bytes > MAX_PROJECT_AUDIO_BYTES:
                    raise ValueError('Project audio exceeds the supported budget')
                assets[audio_id] = blob.tobytes()
            t = Take(item['id'], item['name'], *arrays, item['segments'], item.get('parent'),
                     item.get('branch_frame'), item.get('events', []), cuts,
                     item.get('dialogue', []), assets)
            validate_take(t)
            t.camera_cuts = validate_camera_cuts(t.camera_cuts, len(t.positions), camera_ids)
            total += len(t.positions)
            if total > MAX_TOTAL_FRAMES:
                raise ValueError('Project exceeds the supported motion budget')
            takes[t.id] = t
        active, frame = doc['active'], doc['frame']
        if takes:
            if active not in takes or type(frame) is not int or not 0 <= frame < len(takes[active].positions):
                raise ValueError('Invalid saved playhead')
        elif active is not None or type(frame) is not int or frame != 0:
            raise ValueError('Invalid saved playhead')
        validate_provenance(takes)
        scene = doc.get('scene', {})
        if not isinstance(scene, dict):
            raise ValueError('Invalid scene')
        result = (takes, active, frame, scene)
        return (*result, cameras) if include_cameras else result
