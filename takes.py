"""Versioned takes with actual motion, not prompt-based regeneration."""
from dataclasses import dataclass, field
import io
import json
import zipfile
import numpy as np
from live_motion import MODEL
from camera_model import validate_cameras, validate_camera_cuts

MAX_FRAMES = 15000  # Ten minutes per take at native 25 fps.
MAX_TOTAL_FRAMES = 30000
MAX_TAKES = 12
MAX_ARCHIVE_BYTES = 220_000_000
MAX_METADATA_CHARS = 1_000_000


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

    def prefix(self, stop):
        """A copy of all metadata before an exclusive frame boundary."""
        segments = [dict(s, end=min(s['end'], stop)) for s in self.segments if s['start'] < stop]
        events = [dict(e) for e in self.events if e['frame'] < stop]
        return segments, events


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
        end = s['end']
    if end != n:
        raise ValueError('Segments do not cover the take')
    for e in take.events:
        if not isinstance(e, dict) or e.get('type') != 'gate_open' or type(e.get('frame')) is not int or not 0 <= e['frame'] < n:
            raise ValueError('Invalid scene event')
    validate_camera_cuts(take.camera_cuts, n)


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
    for i, t in enumerate(takes.values()):
        validate_take(t)
        cuts = validate_camera_cuts(t.camera_cuts, len(t.positions), camera_ids)
        key = f't{i}'
        for field_name in ('positions', 'rotations', 'motion'):
            data[key + '_' + field_name] = getattr(t, field_name)
        items.append(dict(key=key, id=t.id, name=t.name, segments=t.segments,
                          parent=t.parent, branch_frame=t.branch_frame, events=t.events,
                          camera_cuts=cuts))
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
        if len(entries) > MAX_TAKES * 3 + 1 or sum(e.file_size for e in entries) > MAX_ARCHIVE_BYTES:
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
        takes, total = {}, 0
        for i, item in enumerate(items):
            if item['key'] != f't{i}' or item['id'] in takes:
                raise ValueError('Invalid take identifiers')
            arrays = [data[f't{i}_{name}'].copy() for name in ('positions', 'rotations', 'motion')]
            cuts = item.get('camera_cuts') if doc['version'] == 2 else []
            t = Take(item['id'], item['name'], *arrays, item['segments'], item.get('parent'),
                     item.get('branch_frame'), item.get('events', []), cuts)
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
