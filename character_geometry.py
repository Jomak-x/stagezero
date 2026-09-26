"""Rendered pose measurements for the local character diagnostic scene."""
from __future__ import annotations

import numpy as np

from character_assets import CharacterAsset


def posed_minimum_y(asset: CharacterAsset, world_matrices: np.ndarray) -> float:
    """Lowest displayed vertex after skinning, without moving the character.

    Used once for a standing/rest pose to place its carrier group. This is not a
    contact/IK solver and must not be used to pin a moving character each frame.
    The asset must already have passed the GLB importer.
    """
    world = np.asarray(world_matrices, dtype=np.float64)
    if world.shape != (len(asset.nodes), 4, 4) or not np.isfinite(world).all():
        raise ValueError('Pose must contain one finite 4x4 world matrix per node')
    doc = asset.document
    pending = list(doc['scenes'][doc.get('scene', 0)].get('nodes', []))
    reachable = set()
    while pending:
        index = pending.pop()
        if index not in reachable:
            reachable.add(index)
            pending.extend(asset.nodes[index].children)
    bottom = float('inf')
    for index in reachable:
        raw = doc['nodes'][index]
        if 'mesh' not in raw:
            continue
        skin_rows = None
        if 'skin' in raw:
            skin = doc['skins'][raw['skin']]
            if 'inverseBindMatrices' in skin:
                inverse_binds = (asset.read_accessor(skin['inverseBindMatrices'])
                                 .reshape((-1, 4, 4)).transpose((0, 2, 1)))
            else:
                inverse_binds = np.repeat(np.eye(4)[None], len(skin['joints']), axis=0)
            # In glTF world skinning the mesh-node transform cancels: do not
            # multiply it a second time onto joint-world × inverse-bind.
            skin_rows = np.stack([(world[joint] @ inverse_binds[i])[1]
                                  for i, joint in enumerate(skin['joints'])])
        for primitive in doc['meshes'][raw['mesh']]['primitives']:
            attrs = primitive['attributes']
            points = asset.read_accessor(attrs['POSITION'])
            references = (asset.read_accessor(primitive['indices'], normalize=False).reshape(-1)
                          if 'indices' in primitive else None)
            joints = asset.read_accessor(attrs['JOINTS_0'], normalize=False) if skin_rows is not None else None
            weights = asset.read_accessor(attrs['WEIGHTS_0']) if skin_rows is not None else None
            count = len(references) if references is not None else len(points)
            for start in range(0, count, 16384):
                selection = references[start:start + 16384] if references is not None else slice(start, start + 16384)
                selected = points[selection]
                if skin_rows is None:
                    heights = selected @ world[index, 1, :3] + world[index, 1, 3]
                else:
                    heights = np.zeros(len(selected), dtype=np.float64)
                    # GLTFLoader normalizes even near-unit float weights. Match
                    # its nonnegative L1 normalization rather than the raw sum.
                    selected_weights = weights[selection]
                    selected_weights = selected_weights / selected_weights.sum(axis=1)[:, None]
                    for influence in range(4):
                        rows = skin_rows[joints[selection, influence]]
                        moved = (rows[:, :3] * selected).sum(axis=1) + rows[:, 3]
                        heights += selected_weights[:, influence] * moved
                if not np.isfinite(heights).all():
                    raise ValueError('Posed mesh heights are not finite')
                bottom = min(bottom, float(heights.min()))
    if not np.isfinite(bottom):
        raise ValueError('Selected scene has no displayed vertices')
    return bottom


def ground_offset(asset: CharacterAsset, world_matrices=None, *, scale=1.0, floor_y=0.0) -> float:
    """World-space carrier offset; scale never changes the imported source.

    The caller supplies the deterministic standing pose for a moving rig, or
    omits it for the displayed rest pose. Keep this value fixed during motion.
    """
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(floor_y):
        raise ValueError('Ground placement needs a finite positive scale and finite floor height')
    if world_matrices is None:
        world_matrices = np.stack([node.world_matrix for node in asset.nodes])
    return float(floor_y - scale * posed_minimum_y(asset, world_matrices))
