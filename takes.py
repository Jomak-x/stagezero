"""Versioned takes with actual motion, not prompt-based regeneration."""
from dataclasses import dataclass, field
import io
import json
import zipfile
import numpy as np
from live_motion import MODEL

MAX_FRAMES = 15000  # Ten minutes per take at native 25 fps.
MAX_TOTAL_FRAMES = 30000
MAX_TAKES = 12
MAX_ARCHIVE_BYTES = 220_000_000


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


def encode_project(takes, active, frame, scene):
    if not 1 <= len(takes) <= MAX_TAKES or active not in takes:
        raise ValueError('Select a generated take before saving')
    if sum(len(t.positions) for t in takes.values()) > MAX_TOTAL_FRAMES:
        raise ValueError('Project exceeds the supported motion budget')
    data, items = {}, []
    for i, t in enumerate(takes.values()):
        validate_take(t)
        key = f't{i}'
        for field_name in ('positions', 'rotations', 'motion'):
            data[key + '_' + field_name] = getattr(t, field_name)
        items.append(dict(key=key, id=t.id, name=t.name, segments=t.segments,
                          parent=t.parent, branch_frame=t.branch_frame, events=t.events))
    manifest = dict(version=1, model=MODEL, fps=25, active=active, frame=int(frame), scene=scene, takes=items)
    data['manifest'] = np.array(json.dumps(manifest, allow_nan=False))
    out = io.BytesIO()
    np.savez_compressed(out, **data)
    return out.getvalue()


def decode_project(content):
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
        if len(raw) > 1_000_000:
            raise ValueError('Project metadata is too large')
        doc = json.loads(raw)
        if doc.get('version') != 1 or doc.get('model') != MODEL or doc.get('fps') != 25:
            raise ValueError('Unsupported project version or skeleton')
        items = doc.get('takes')
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_TAKES:
            raise ValueError('Invalid take count')
        takes, total = {}, 0
        for i, item in enumerate(items):
            if item['key'] != f't{i}' or item['id'] in takes:
                raise ValueError('Invalid take identifiers')
            arrays = [data[f't{i}_{name}'].copy() for name in ('positions', 'rotations', 'motion')]
            t = Take(item['id'], item['name'], *arrays, item['segments'], item.get('parent'), item.get('branch_frame'), item.get('events', []))
            validate_take(t)
            total += len(t.positions)
            if total > MAX_TOTAL_FRAMES:
                raise ValueError('Project exceeds the supported motion budget')
            takes[t.id] = t
        active, frame = doc['active'], doc['frame']
        if active not in takes or type(frame) is not int or not 0 <= frame < len(takes[active].positions):
            raise ValueError('Invalid saved playhead')
        for t in takes.values():
            if t.parent is not None and (t.parent not in takes or t.parent == t.id or type(t.branch_frame) is not int or not 3 <= t.branch_frame < len(takes[t.parent].positions)):
                raise ValueError('Invalid branch provenance')
        scene = doc.get('scene', {})
        if not isinstance(scene, dict):
            raise ValueError('Invalid scene')
        return takes, active, frame, scene
