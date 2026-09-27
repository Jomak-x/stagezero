"""Reject unsafe native terrain horizons without modifying poses or history.

These sampled Core foot/body proxies are a rejection gate, not a mesh-physics
certificate. Rendered character fitting must be inspected separately.
"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import numpy as np

from scene_interaction_geometry import SceneInteractionGeometry


def validate_terrain_clip(clip, scene, *, history=None, enabled=False, start_frame=0, terrain_start_frame=0):
    from core_scene_reactions import object_states
    from motion_bridge import _layout
    names, parents, neutral = _layout()
    index = {name: i for i, name in enumerate(names)}
    feet = [(index[side + 'Foot'], index[side + 'ToeBase']) for side in ('Left', 'Right')]
    clearance = float(-np.mean(neutral[[toe for _, toe in feet], 1]))
    if history is not None and (history.actor_ids != clip.actor_ids or history.fps != clip.fps):
        raise ValueError('Terrain history must match actors and frame rate')
    prefix = 0 if history is None else history.frames
    positions = clip.positions if history is None else np.concatenate((history.positions, clip.positions), axis=1)
    combined = SimpleNamespace(positions=positions, frames=positions.shape[1], fps=clip.fps)
    geometry, previous_key = None, None
    minimum = float('inf')
    max_gap = 0.
    # Check candidate frames plus interpolated samples from the preceding
    # committed root/feet: a boundary cannot jump across a narrow void.
    for frame in range(clip.frames):
        states = object_states(scene, combined, prefix + frame, enabled=enabled,
                               start_frame=start_frame, terrain=True, terrain_start_frame=terrain_start_frame)
        key = tuple(tuple(state['position']) for state in states)
        if key != previous_key:
            document = deepcopy(scene)
            for obj, state in zip(document['objects'], states):
                obj['position'] = state['position']
            geometry = SceneInteractionGeometry.from_scene(document)
            previous_key = key
        for actor, actor_id in enumerate(clip.actor_ids):
            pose = clip.positions[actor, frame]
            root = pose[0]
            support = geometry.support_height(root[0], root[2], root[1] - clearance,
                                              max_step_up=.35, max_drop=.35)
            if support is None:
                raise ValueError(f'{actor_id}: terrain has no support under root at frame {frame}; last good motion retained')
            for dx, dz in ((.12, 0), (-.12, 0), (0, .12), (0, -.12)):
                if geometry.support_height(root[0]+dx, root[2]+dz, support,
                                           max_step_up=.30, max_drop=.35) is None:
                    raise ValueError(f'{actor_id}: terrain footprint crosses a void at frame {frame}')
            for name in ('Hips', 'Spine3', 'Head', 'LeftLeg', 'RightLeg', 'LeftArm', 'RightArm'):
                point = pose[index[name]]
                if geometry.obstacle_at(*point, radius=.09):
                    raise ValueError(f'{actor_id}: native {name} overlaps rendered terrain at frame {frame}')
            foot_gaps = []
            for ankle, toe in feet:
                offset = neutral[toe] - neutral[ankle]
                samples = np.asarray([[x, offset[1], z] for x in (-.045, .045)
                                      for z in (-.035, offset[2])])
                sole = pose[ankle] + samples @ clip.rotations[actor, frame, ankle].T
                heights = []
                for point in sole:
                    h = geometry.support_height(point[0], point[2], support, max_step_up=.30, max_drop=.35)
                    if h is None:
                        continue  # A swinging foot may extend past a ledge.
                    heights.append(h)
                    gap = float(point[1] - h)
                    minimum = min(minimum, gap)
                    if gap < -.01:
                        raise ValueError(f'{actor_id}: raw native sole penetrates terrain by {-gap:.3f} m at frame {frame}; last good motion retained')
                h = geometry.support_height(pose[toe, 0], pose[toe, 2], support, max_step_up=.30, max_drop=.35)
                foot_gaps.append(float('inf') if h is None else abs(float(pose[toe, 1]-h)))
                if heights and min(foot_gaps[-1:]) <= .035 and max(heights)-min(heights) > .035:
                    raise ValueError(f'{actor_id}: planted sole straddles a riser at frame {frame}')
            gap = min(foot_gaps)
            max_gap = max(max_gap, gap)
            if gap > .15:
                raise ValueError(f'{actor_id}: neither foot has nearby terrain support at frame {frame}')
            if prefix + frame:
                before = positions[actor, prefix + frame - 1]
                swept_names = ('Hips', 'Spine3', 'Head', 'LeftLeg', 'RightLeg', 'LeftArm', 'RightArm')
                displacement = max(np.linalg.norm(pose[index[name]]-before[index[name]]) for name in swept_names)
                count = max(2, int(np.ceil(displacement/.04))+1)
                if count > 128:
                    raise ValueError(f'{actor_id}: native terrain body discontinuity')
                for t in np.linspace(0, 1, count)[1:-1]:
                    point = before[0]*(1-t) + root*t
                    if geometry.support_height(point[0], point[2], point[1]-clearance,
                                               max_step_up=.35, max_drop=.35) is None:
                        raise ValueError(f'{actor_id}: native root crossed unsupported terrain between frames')
                    for name in swept_names:
                        j = index[name]
                        point = before[j]*(1-t) + pose[j]*t
                        if geometry.obstacle_at(*point, radius=.09):
                            raise ValueError(f'{actor_id}: native body crossed rendered solid between frames')
    return {'raw_native_support_checked': getattr(clip, 'native_features', None) is not None,
            'presentation_support_checked': getattr(clip, 'native_features', None) is None,
            'proxy_only': True,
            'minimum_sampled_sole_clearance_m': minimum,
            'maximum_nearest_toe_support_gap_m': max_gap,
            'poses_modified': False}
