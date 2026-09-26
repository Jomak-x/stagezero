"""Studio adapter for the visually reviewed exact-native authored rig.

The two selected performers share the source coordinate system. Other cast
members hold an explicitly static first-frame pose at separated sideline marks.
"""
from pathlib import Path
from functools import lru_cache
import numpy as np
from experiments.native_pair_rig import NativeRigActor, NativeRigAsset
from native_pair_clip import NativePairClip, validate_cast, validate_placement, load_source

DEFAULT_ASSET = Path(__file__).parent / 'assets/paired/Xbot.glb'
REVIEWED_IDLE_SOURCE = Path(__file__).parent / 'review/two-character/native-recovery/originals/handshake_seed42.npz'


@lru_cache(maxsize=1)
def _reviewed_idle_poses():
    # Cache the approved source once. These are static stances, never a new take.
    return load_source(REVIEWED_IDLE_SOURCE).joints[0]



class NativePairRenderer:
    def __init__(self, server, asset_path=None, prefix='/native-pair'):
        self.server, self.prefix = server, prefix
        self.asset = NativeRigAsset(asset_path or DEFAULT_ASSET)
        self.actors = {}
        self._cast = []
        self._pair = ()
        self._placement = {'x': 0., 'z': 0., 'yaw_degrees': 0.}
        self._world = None
        self._clip = None
        self._frame = None
        self._visible = False
        self._removed = False
        self._idle = {}

    def sync_cast(self, snapshot):
        cast, pair = validate_cast(snapshot['cast'], snapshot['selected_pair'])
        placement = validate_placement(snapshot.get('placement', {'x': 0., 'z': 0., 'yaw_degrees': 0.}))
        if cast == self._cast and pair == self._pair and placement == self._placement:
            return
        ids = {actor['id'] for actor in cast}
        for identifier in set(self.actors) - ids:
            self.actors.pop(identifier).remove()
        for actor in cast:
            if actor['id'] not in self.actors:
                display = NativeRigActor(self.server, f"{self.prefix}/{actor['id']}", self.asset, tuple(actor['color']))
                display.visible = self._visible
                self.actors[actor['id']] = display
        self._cast, self._pair, self._placement = cast, pair, placement
        if self._clip is not None:
            self._prepare()
        else:
            self._prepare_idle_cast()

    def _prepare_idle_cast(self):
        poses = _reviewed_idle_poses()
        angle = np.deg2rad(self._placement['yaw_degrees'])
        rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                             [-np.sin(angle), 0., np.cos(angle)]])
        translation = np.array([self._placement['x'], 0., self._placement['z']])
        self._idle = {}
        for index, actor in enumerate(self._cast):
            pose = poses[index % 2].copy()
            mark = np.array([-.9 if index % 2 == 0 else .9, pose[0, 1], -(index // 2) * 1.8])
            pose += mark - pose[0]
            pose = pose @ rotation.T + translation
            display = self.actors[actor['id']]
            display.reset_pose_history()
            display.set_pose(pose)
            self._idle[actor['id']] = pose

    def _prepare(self):
        if not self._pair:
            raise ValueError('Sync native cast before loading the clip')
        angle = np.deg2rad(self._placement['yaw_degrees'])
        rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                             [-np.sin(angle), 0., np.cos(angle)]])
        if all(value == 0 for value in self._placement.values()):
            self._world = self._clip.joints
        else:
            self._world = self._clip.joints @ rotation.T + np.array([self._placement['x'], 0., self._placement['z']])
        for index, identifier in enumerate(self._pair):
            if self._clip.metadata.get('render_hand_pose') == 'fists':
                self.actors[identifier].prepare_clip(self._world[:, index], hand_pose='fist',
                    contact_weights=np.ones((self._clip.frames, 2)))
            else:
                self.actors[identifier].prepare_clip(self._world[:, index])
        self._idle = {}
        # Use native stance; translate as a whole to marks beyond the entire
        # active clip footprint, never alter the paired root separation.
        poses = self._world
        min_x, max_x = float(poses[..., 0].min()), float(poses[..., 0].max())
        min_z = float(poses[..., 2].min())
        idle_ids = [a['id'] for a in self._cast if a['id'] not in self._pair]
        for index, identifier in enumerate(idle_ids):
            pose = poses[0, index % 2].copy()
            mark = np.array([min_x - 1.6 if index % 2 == 0 else max_x + 1.6,
                             pose[0, 1], min_z - (index // 2) * 1.5])
            pose += mark - pose[0]
            self.actors[identifier].reset_pose_history()
            self.actors[identifier].set_pose(pose)
            self._idle[identifier] = pose
        self._frame = None
        self.tick(0)

    def set_clip(self, clip):
        if not isinstance(clip, NativePairClip):
            raise ValueError('Native renderer accepts only exact native22 paired clips')
        if clip is self._clip:
            return
        self._clip = clip
        self._prepare()

    def tick(self, frame):
        if self._clip is None:
            return
        if type(frame) is not int or not 0 <= frame < self._clip.frames:
            raise ValueError('Native renderer frame is outside the clip')
        if frame == self._frame:
            return
        for identifier in self._pair:
            self.actors[identifier].set_frame(frame)
        self._frame = frame

    set_frame = tick

    def set_visible(self, visible):
        self._visible = bool(visible)
        for actor in self.actors.values():
            actor.visible = self._visible

    def actor_root(self, actor_id=None):
        if self._clip is None:
            if actor_id is None:
                return np.mean([pose[0] for pose in self._idle.values()], axis=0) if self._idle else np.zeros(3)
            if actor_id in self._idle:
                return self._idle[actor_id][0].copy()
            raise ValueError('Unknown native cast actor')
        if actor_id is None:
            return self._world[self._frame or 0, :, 0].mean(axis=0).copy()
        if actor_id in self._pair:
            return self._world[self._frame or 0, self._pair.index(actor_id), 0].copy()
        if actor_id in self._idle:
            return self._idle[actor_id][0].copy()
        raise ValueError('Unknown native cast actor')

    @property
    def provenance(self):
        return dict(self.asset.provenance, fps=30, joint_count=22,
                    authored_hand_pose=None if self._clip is None else self._clip.metadata.get('render_hand_pose', 'relaxed'),
                    hand_pose_source='Explicit manual finger pose; not model-generated or contact solved',
                    independent_actor_offsets=False, shared_rigid_placement=dict(self._placement), idle_cast='static source stance at sideline marks',
                    physical_contact_verified=False)

    def remove(self):
        if not self._removed:
            for actor in self.actors.values():
                actor.remove()
            self._removed = True
