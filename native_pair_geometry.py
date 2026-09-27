"""Scene-only placement guard for exact native22 paired InterGen motion.

Input joints are already in world coordinates. This samples body spheres
against authored scene solids and sweeps each root over authored floor. It does
not alter motion or reject close partner contact.
"""
from __future__ import annotations

import numpy as np

from interaction_scene_collision import scene_collision
from realtime_navigation import validate_ground_path
from studio_interaction_scene import adapt_studio_scene


def check_native_pair_geometry(world_pair, scene_document, *, actor_ids=('actor_1', 'actor_2')):
    """Validate a complete [frames, 2, 22, 3] world-space native pair.

    Raise before committing a replacement clip when an actor leaves the
    continuous authored floor or overlaps a sampled scene solid proxy.
    Pair-to-pair distance is intentionally unrestricted for contact actions.
    """
    positions = np.asarray(world_pair)
    if (positions.ndim != 4 or positions.shape[1:] != (2, 22, 3)
            or not 4 <= len(positions) <= 1000 or positions.dtype.kind not in 'fi'
            or not np.isfinite(positions).all()):
        raise ValueError('Native scene validation requires finite world joints [frames, 2, 22, 3]')
    if (not isinstance(actor_ids, (tuple, list)) or len(actor_ids) != 2
            or any(not isinstance(name, str) or not name or len(name) > 80 for name in actor_ids)
            or actor_ids[0] == actor_ids[1]):
        raise ValueError('Native scene validation requires two distinct actor IDs')
    adapted = adapt_studio_scene(scene_document)
    reports = []
    for index, actor_id in enumerate(actor_ids):
        root_path = positions[:, index, 0, :][:, [0, 2]]
        try:
            ground = validate_ground_path(adapted['scene'], root_path, actor_radius_m=.28)
        except ValueError as exc:
            raise ValueError(f'{actor_id} leaves continuous authored floor; last good motion retained') from exc
        collision = scene_collision(positions[:, index], 'native22', adapted['scene'], adapted['affordances'])
        if collision['total_collision_frames']:
            first = next(item for item in collision['per_object'] if item['collision_frames'])
            raise ValueError(f'{actor_id} native motion overlaps scene solid {first["object_id"]}; last good motion retained')
        reports.append({'actor_id': actor_id, 'ground': ground, 'collision': collision})
    return {'version': 1, 'skeleton': 'native22', 'frames': len(positions),
            'scene_conditioned': False, 'physical_contact_verified': False,
            'actors': reports}


check_scene_geometry = check_native_pair_geometry
check_native_scene = check_native_pair_geometry
