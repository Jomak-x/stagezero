"""Archive routing and exclusive browser playback for the main studio cast modes."""
from __future__ import annotations
import io
import json
from zipfile import BadZipFile, ZipFile
import numpy as np


def decode_native_project(content):
    """Inspect bounded metadata, then run the matching archive's full validator."""
    from cast_performance import SCHEMA as CAST_SCHEMA, MAX_BYTES, MAX_METADATA_BYTES
    from cast_performance import decode_project as decode_cast
    from native_pair_clip import SCHEMA as PAIR_SCHEMA, decode_project as decode_pair
    if not isinstance(content, bytes) or not 1 <= len(content) <= MAX_BYTES:
        raise ValueError('Native project must be nonempty bytes up to 64 MiB')
    try:
        with ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()
            if (len(entries) > 3 or len({item.filename for item in entries}) != len(entries)
                    or sum(item.file_size for item in entries) > MAX_BYTES):
                raise ValueError('Invalid or oversized native project archive')
            entry = archive.getinfo('metadata.npy')
            if entry.file_size > MAX_METADATA_BYTES * 4 + 1024:
                raise ValueError('Native project metadata is too large')
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                reader = {(1, 0): np.lib.format.read_array_header_1_0,
                          (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:
                    raise ValueError('Unsupported native metadata encoding')
                shape, _, dtype = reader(stream)
                if shape != () or dtype.kind not in 'US' or dtype.itemsize > MAX_METADATA_BYTES * 4:
                    raise ValueError('Invalid native project metadata')
            metadata = np.load(io.BytesIO(archive.read(entry)), allow_pickle=False).item()
            document = json.loads(metadata)
            schema = document.get('schema') if isinstance(document, dict) else None
    except (BadZipFile, KeyError, TypeError, EOFError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid native project archive') from exc
    if schema == CAST_SCHEMA:
        return 'cast', decode_cast(content)
    if schema == PAIR_SCHEMA:
        return 'paired', decode_pair(content)
    raise ValueError('Unsupported native project schema')


def set_cast_camera_view(client, position, target, fov):
    """Publish one upright view despite Viser's coupled camera setters."""
    with client.atomic():
        client.camera.position = position
        client.camera.look_at = target
        # Setting the endpoints can retain/recompute roll from the previous
        # main-studio camera. Restore world-up after both endpoint writes.
        client.camera.up_direction = (0., 1., 0.)
        client.camera.fov = fov


class NativePlaybackRouter:
    """Only the selected renderer may publish through the one protocol controller."""
    def __init__(self, server, modes, *, lock, controller_factory=None):
        if controller_factory is None:
            from native_pair_playback import NativePairPlaybackController
            controller_factory = NativePairPlaybackController
        self.modes, self.lock, self.selected = modes, lock, None
        self.controller = controller_factory(server, get_state=self.get_state)

    def get_state(self):
        with self.lock:
            if self.selected is None:
                return {'frame': 0, 'playing': False, 'capturing': False, 'enabled': False}
            session, _ = self.modes[self.selected]
            state = session.snapshot()
            return dict(state, enabled=bool(state['active'] and state['total_frames']))

    def select(self, mode):
        with self.lock:
            if mode == self.selected:
                return
            if self.selected is not None:
                self.modes[self.selected][1].local_playback = None
            self.controller.clear()
            self.selected = mode
            if mode is not None:
                self.modes[mode][1].local_playback = self.controller
                self.refresh(republish=True)

    def refresh(self, *, republish=False):
        with self.lock:
            if self.selected is None:
                return
            session, renderer = self.modes[self.selected]
            # A completed builder must not pair a new cast with an old frame.
            with session._lock:
                state, clip = session.snapshot(), session.timeline_clip()
                revision = self.controller.revision
                renderer.sync_cast(state)
                if clip is not None:
                    renderer.set_clip(clip)
                    if republish and revision == self.controller.revision:
                        identifiers = state['actor_ids'] if self.selected == 'cast' else state['selected_pair']
                        self.controller.load([(identifier, renderer.actors[identifier]) for identifier in identifiers],
                                             fps=clip.fps, frames=clip.frames)
                    renderer.tick(state['frame'])
                self.controller.update(state, enabled=bool(state['active'] and clip is not None))
