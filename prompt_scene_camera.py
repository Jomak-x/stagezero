"""Bounded camera framing that prefers clear views of props and the whole cast.

This tests sight lines against oriented prop bounds and actor body proxies,
not triangle visibility.
It never changes actors, props, user camera controls, or the playback camera on
ordinary ticks. Call only for explicit framing or a newly displayed take.
"""
from __future__ import annotations

import numpy as np

from interaction_scene import is_walkable_ground, local_axes, scene_objects

MAX_SAMPLED_FRAMES = 24
# Thirty-degree steps can leave only one prop-clear angle in a narrow street.
# Thirty-six fixed angles remain inexpensive for an explicit framing action.
ORBIT_DEGREES = tuple([0] + [angle for degree in range(10, 180, 10)
                            for angle in (degree, -degree)] + [180])
VISIBILITY_JOINTS = (0, 9, 15, 7, 8, 10, 11)  # Root, chest, head, ankles, toes.
VISIBILITY_WEIGHTS = (1., 2., 1., 1., 1., 1., 1.)
CAST_VISIBILITY_JOINTS = VISIBILITY_JOINTS + (20, 21)  # Include both wrists/contact.
CAST_VISIBILITY_WEIGHTS = VISIBILITY_WEIGHTS + (2., 2.)


def _framing_corners(clip, roots, center):
    points = np.concatenate((clip.joints.reshape(-1, 3),
                             roots + [-.55, -1.1, -.55], roots + [.55, 1.3, .55]))
    low, high = points.min(axis=0) - .25, points.max(axis=0) + .25
    return np.array([[x, y, z] for x in (low[0], high[0])
                     for y in (low[1], high[1]) for z in (low[2], high[2])]) - center


def _sample_frames(frames, frame):
    indices = np.linspace(0, frames - 1, min(MAX_SAMPLED_FRAMES - 1, frames)).round().astype(int)
    return np.unique(np.append(indices, frame))


def _blocked_rays(camera, targets, objects):
    """Segment-versus-OBB slabs; objects behind either endpoint cannot block."""
    targets = np.asarray(targets, dtype=float)
    blocked = np.zeros(len(targets), dtype=bool)
    for obj in objects:
        width, depth = local_axes(obj.yaw_degrees)
        basis = np.array([[width[0], 0., depth[0]], [0., 1., 0.], [width[1], 0., depth[1]]])
        center = np.array([obj.x, obj.y, obj.z])
        half_size = np.array([obj.width, obj.height, obj.depth]) / 2
        origin = (camera - center) @ basis
        direction = (targets - center) @ basis - origin
        parallel = np.abs(direction) < 1e-10
        outside_parallel = np.any(parallel & ((origin < -half_size) | (origin > half_size)), axis=1)
        first = np.divide(-half_size - origin, direction,
                          out=np.full_like(direction, -np.inf), where=~parallel)
        last = np.divide(half_size - origin, direction,
                         out=np.full_like(direction, np.inf), where=~parallel)
        near = np.maximum(np.minimum(first, last).max(axis=1), 1e-6)
        far = np.minimum(np.maximum(first, last).min(axis=1), 1 - 1e-6)
        blocked |= ~outside_parallel & (near <= far)
    return blocked


def _cast_blocked_rays(camera, joints):
    """Other actors' torso/head spheres block sight lines; self never does.

    Overlapping spheres along the spine approximate a torso capsule. Radii
    follow each actor's shoulder width. These conservative body proxies only
    rank camera views; they do not measure contacts or change any source pose.
    """
    actors = joints.shape[1]
    targets = joints[:, :, CAST_VISIBILITY_JOINTS, :]
    blocked = np.zeros(targets.shape[:-1], dtype=bool)
    if actors < 2:
        return blocked
    centers = joints[:, :, [0, 3, 6, 9, 12, 15], :]
    width = np.linalg.norm(joints[:, :, 16] - joints[:, :, 17], axis=-1)
    radii = np.clip(width * .55, .15, .32)[..., None] * [.85, .9, 1., 1., .65, .65]
    for actor in range(actors):
        rays = targets[:, actor] - camera
        lengths_squared = np.maximum(np.sum(rays * rays, axis=-1), 1e-12)
        for other in range(actors):
            if other == actor:
                continue
            to_centers = centers[:, other] - camera
            projection = np.einsum('fkj,fij->fik', to_centers, rays) / lengths_squared[..., None]
            perpendicular_squared = np.maximum(0., np.sum(to_centers * to_centers, axis=-1)[:, None, :]
                - projection * projection * lengths_squared[..., None])
            radius_squared = radii[:, other, None, :] ** 2
            intersects = perpendicular_squared < radius_squared
            half_interval = np.sqrt(np.maximum(0., radius_squared - perpendicular_squared) / lengths_squared[..., None])
            intersects &= (projection + half_interval > 1e-6) & (projection - half_interval < 1 - 1e-6)
            blocked[:, actor] |= np.any(intersects, axis=-1)
    return blocked


def prompt_scene_camera_view(clip, scene_document, *, frame=0, aspect=16/9):
    """Fit the full take and pick one of 36 orbit angles with fewer blockers.

    The existing 42-degree FOV and elevation are retained. Distance only grows
    when the full-take bounds need more space at a different angle. Start and
    current-frame visibility receive extra weight, with every actor's torso,
    head, ankles and toes represented so foreground props cannot hide the feet
    without affecting the score. Among equally clear prop views, other actors'
    body proxies are scored against torso, foot, and hand sight lines.
    """
    if (clip is None or type(frame) is not int or not 0 <= frame < clip.frames or
            clip.joints.ndim != 4 or clip.joints.shape[2:] != (22, 3)):
        raise ValueError('Camera framing requires a loaded native22 cast and valid frame')
    from director_viewer import native_cast_camera_view
    aspect = float(aspect) if aspect and np.isfinite(aspect) and aspect > 0 else 16/9
    roots = clip.joints[frame, :, 0]
    original, center, fov = native_cast_camera_view(clip,
        {'x': 0., 'z': 0., 'yaw_degrees': 0.}, roots, aspect=aspect)
    objects = [obj for obj in scene_objects(scene_document) if not is_walkable_ground(obj)]
    if not objects and clip.joints.shape[1] == 1:
        return original, center, fov
    samples = _sample_frames(clip.frames, frame)
    sampled_joints = clip.joints[samples]
    targets = sampled_joints[:, :, VISIBILITY_JOINTS, :]
    weights = np.ones((len(samples), 1))
    weights[(samples == 0) | (samples == frame)] = 3.
    corners = _framing_corners(clip, roots, center)
    original_distance = np.linalg.norm(original - center)
    original_direction = (original - center) / original_distance
    best_position, best_score = original, None
    for degrees in ORBIT_DEGREES:
        angle = np.deg2rad(degrees)
        rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                             [-np.sin(angle), 0., np.cos(angle)]])
        direction = rotation @ original_direction
        right = np.cross([0., 1., 0.], direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        depth = corners @ direction
        required = max(3., float(np.max(depth + np.abs(corners @ right) / (np.tan(fov / 2) * aspect))),
                       float(np.max(depth + np.abs(corners @ up) / np.tan(fov / 2)))) * 1.1
        distance = max(original_distance, required)
        camera = center + distance * direction
        blocked = _blocked_rays(camera, targets.reshape(-1, 3), objects).reshape(targets.shape[:-1])
        # Prefer avoiding a completely obscured torso for any actor, then
        # minimize overall blocked samples. Chest rays count twice.
        hidden_torsos = float(np.sum(np.all(blocked[..., :2], axis=-1) * weights))
        hidden_points = float(np.sum((blocked * VISIBILITY_WEIGHTS).sum(axis=-1) * weights))
        cast_blocked = _cast_blocked_rays(camera, sampled_joints)
        hidden_cast_torsos = float(np.sum(np.all(cast_blocked[..., :2], axis=-1) * weights))
        hidden_cast_points = float(np.sum((cast_blocked * CAST_VISIBILITY_WEIGHTS).sum(axis=-1) * weights))
        score = (hidden_torsos, hidden_points, hidden_cast_torsos, hidden_cast_points, distance, abs(degrees))
        if best_score is None or score < best_score:
            best_position, best_score = camera, score
        if degrees == 0 and hidden_points == 0 and hidden_cast_points == 0:
            return original, center, fov
    return best_position, center, fov
