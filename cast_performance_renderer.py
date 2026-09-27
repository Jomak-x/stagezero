"""Render every composed cast track directly in its stored world coordinates."""
from copy import deepcopy
from pathlib import Path

import numpy as np

from cast_performance import CastPerformance, validate_cast
from experiments.native_pair_rig import NativeRigActor, NativeRigAsset

DEFAULT_ASSET = Path(__file__).parent / 'assets/paired/Xbot.glb'


class CastPerformanceRenderer:
    def __init__(self, server, asset_path=None, prefix='/cast-performance'):
        self.server, self.prefix = server, prefix
        self.asset = NativeRigAsset(asset_path or DEFAULT_ASSET)
        self.actors = {}
        self._cast = []
        self._clip = None
        self._frame = None
        self._visible = self._removed = False
        self.local_playback = None

    def _open(self):
        if self._removed:
            raise RuntimeError('Cast performance renderer was removed')

    def sync_cast(self, snapshot):
        self._open()
        cast = snapshot.get('cast', [])
        ids = tuple(actor['id'] for actor in cast)
        if cast:
            cast = validate_cast(cast, ids)
        elif snapshot.get('actor_ids'):
            raise ValueError('Cast snapshot is missing its performers')
        if 'actor_ids' in snapshot and tuple(snapshot['actor_ids']) != ids:
            raise ValueError('Cast snapshot must preserve the performance track order')
        if cast == self._cast:
            return
        previous = {actor['id']: actor for actor in self._cast}
        next_cast = {actor['id']: actor for actor in cast}
        if self.local_playback is not None:
            self.local_playback.clear()
        for identifier in tuple(self.actors):
            if identifier not in next_cast or previous[identifier]['color'] != next_cast[identifier]['color']:
                self.actors.pop(identifier).remove()
        for actor in cast:
            if actor['id'] not in self.actors:
                display = NativeRigActor(self.server, f"{self.prefix}/{actor['id']}", self.asset, tuple(actor['color']))
                display.visible = self._visible
                self.actors[actor['id']] = display
        self._cast = cast
        if self._clip is not None and self._clip.actor_ids == ids:
            frame = self._frame or 0
            self._prepare(self._clip)
            self.tick(frame)
        else:
            # A completed new take can change actor count. Do not apply the old
            # take to the new cast while the caller installs the replacement.
            self._clip = self._frame = None

    def _prepare(self, clip):
        previous = {}
        hand_pose = clip.metadata.get('render_hand_pose', 'relaxed')
        try:
            for index, identifier in enumerate(clip.actor_ids):
                actor = self.actors[identifier]
                previous[identifier] = (actor._prepared, actor._scale, actor._previous_normals,
                                        deepcopy(actor.provenance))
                options = {} if hand_pose == 'relaxed' else {
                    'hand_pose': 'fist', 'contact_weights': np.ones((clip.frames, 2))}
                actor.prepare_clip(clip.joints[:, index], **options)
            if self.local_playback is not None:
                self.local_playback.load([(identifier, self.actors[identifier]) for identifier in clip.actor_ids],
                                         fps=clip.fps, frames=clip.frames)
        except Exception:
            for identifier, state in previous.items():
                actor = self.actors[identifier]
                actor._prepared, actor._scale, actor._previous_normals, actor.provenance = state
            raise
        self._clip, self._frame = clip, None
        self.tick(0)

    def set_clip(self, clip):
        self._open()
        if not isinstance(clip, CastPerformance):
            raise ValueError('Cast renderer accepts only separate CastPerformance tracks')
        if clip.actor_ids != tuple(actor['id'] for actor in self._cast):
            raise ValueError('Sync the exact performance cast before loading its tracks')
        if clip is not self._clip:
            self._prepare(clip)

    def tick(self, frame):
        self._open()
        if self._clip is None:
            return
        if type(frame) is not int or not 0 <= frame < self._clip.frames:
            raise ValueError('Cast renderer frame is outside the performance')
        if frame == self._frame:
            return
        if self.local_playback is None:
            for identifier in self._clip.actor_ids:
                self.actors[identifier].set_frame(frame)
        self._frame = frame

    set_frame = tick

    def set_visible(self, visible):
        self._open()
        self._visible = bool(visible)
        for actor in self.actors.values():
            actor.visible = self._visible

    def actor_root(self, actor_id=None):
        if self._clip is None:
            if actor_id is not None:
                raise ValueError('No cast performance track is loaded')
            return np.zeros(3)
        roots = self._clip.joints[self._frame or 0, :, 0]
        if actor_id is None:
            return roots.mean(axis=0).copy()
        if actor_id not in self._clip.actor_ids:
            raise ValueError('Unknown cast performance actor')
        return roots[self._clip.actor_ids.index(actor_id)].copy()

    @property
    def provenance(self):
        return dict(self.asset.provenance, fps=30, joint_count=22,
            actor_ids=[] if self._clip is None else list(self._clip.actor_ids),
            coordinates='stored world-space native22 display tracks; no renderer retargeting or placement',
            physical_contact_verified=False)

    def remove(self):
        if not self._removed:
            if self.local_playback is not None:
                self.local_playback.clear()
            for actor in self.actors.values():
                actor.remove()
            self._removed = True
