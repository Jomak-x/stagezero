"""World-space contact geometry derived from the meshes used to draw a scene.

Custom assets use the renderer's mesh bounds normalization, including yaw and
repeated parts. A motion planner can sample a floor at the character's feet
without mistaking an elevated roof or bridge for the nearby ground.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from asset_geometry import _rotation as _part_rotation
from asset_geometry import compile_asset


@dataclass(frozen=True)
class WalkableSurface:
    object_id: str
    triangle: tuple[tuple[float, float, float], ...]
    normal_y: float


def _yaw(degrees):
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=float)


def _asset_transform(asset, obj):
    vertices, faces, _ = compile_asset(asset)
    lower, upper = vertices.min(axis=0).astype(float), vertices.max(axis=0).astype(float)
    centre = (lower + upper) / 2
    span = np.maximum(upper - lower, 1e-6)
    scale = np.asarray(obj['size'], dtype=float) / span
    rotation = _yaw(obj.get('yaw', 0))
    position = np.asarray(obj['position'], dtype=float)
    world = ((vertices.astype(float) - centre) * scale) @ rotation.T + position
    return world, faces, centre, scale, rotation, position


def _tread_steps(asset, obj, centre, scale, rotation, position):
    """Find broad box treads from the same part recipe that draws the stairs."""
    steps = []
    for part in asset['parts']:
        if part['shape'] != 'box':
            continue
        size = np.asarray(part['size'], dtype=float)
        if size[0] < .48 or size[2] < .15 or size[1] > .30:
            continue
        # Steps are horizontal. Decorative sloped blocks are not treads.
        if any(abs(degrees) > 1e-6 for degrees in part.get('rotation', [0, 0, 0])):
            continue
        repeat = part.get('repeat', {'count': [1, 1, 1], 'step': [0, 0, 0]})
        for ix in range(repeat['count'][0]):
            for iy in range(repeat['count'][1]):
                for iz in range(repeat['count'][2]):
                    raw = np.asarray(part['position'], dtype=float) + np.asarray(repeat['step'], dtype=float) * [ix, iy, iz]
                    raw[1] += size[1] / 2
                    point = ((raw - centre) * scale) @ rotation.T + position
                    steps.append(tuple(float(v) for v in point))
    # A railing or decorative edge cannot form a route on its own. Select the
    # main ascent by taking one broad tread per distinct height.
    steps.sort(key=lambda p: p[1])
    ordered = []
    for point in steps:
        if not ordered or abs(point[1] - ordered[-1][1]) > .025:
            ordered.append(point)
    return ordered


def _ordinary_top(obj):
    """Actual broad upper part of a built-in model, if it has one."""
    # Fractions below mirror ObjectSceneLayer._build. Narrow trim, handles,
    # foliage and a chair's high back must not become phantom floors.
    parts = {
        'chair': (.88, .025, .82, 0, -.02),
        'table': (1.0, .50, 1.0, 0, 0),
        'sofa': (.84, .065, .60, 0, -.09),
        'crate': (.96, .48, .96, 0, 0),
        'platform': (1.0, .50, 1.0, 0, 0),
        'pillar': (1.0, .50, 1.0, 0, 0),
        'wall': (1.0, .50, 1.0, 0, 0),
        'console': (1.0, .49, .92, 0, 0),
    }
    spec = parts.get(obj['kind'])
    if spec is None:
        return ()
    fx, fy, fz, ox, oz = spec
    sx, sy, sz = map(float, obj['size'])
    corners = np.asarray([[ox*sx-fx*sx/2, fy*sy, oz*sz-fz*sz/2],
                          [ox*sx+fx*sx/2, fy*sy, oz*sz-fz*sz/2],
                          [ox*sx+fx*sx/2, fy*sy, oz*sz+fz*sz/2],
                          [ox*sx-fx*sx/2, fy*sy, oz*sz+fz*sz/2]])
    return corners @ _yaw(obj.get('yaw', 0)).T + np.asarray(obj['position'], dtype=float)


class SceneInteractionGeometry:
    """Queryable walkable triangles and physical scene object bounds.

    `current_y` is the character's current sole height. A support query only
    considers surfaces reachable by the specified step and drop limits.
    """

    def __init__(self, surfaces, stair_routes, objects, solids):
        self.walkable_surfaces = tuple(surfaces)
        self.stair_routes = tuple(stair_routes)
        self._objects = tuple(objects)
        self._solids = tuple(solids)
        triangles = np.asarray([s.triangle for s in surfaces], dtype=float).reshape((-1, 3, 3))
        self._triangles = triangles
        if len(triangles):
            x = triangles[:, :, 0]
            z = triangles[:, :, 2]
            self._xz_bounds = np.column_stack((x.min(axis=1), x.max(axis=1), z.min(axis=1), z.max(axis=1)))
        else:
            self._xz_bounds = np.zeros((0, 4))

    @classmethod
    def from_scene(cls, scene):
        assets = {asset['id']: asset for asset in scene.get('assets', [])}
        surfaces = []
        stair_routes = []
        obstacles = []
        solids = []
        authored_ground = False
        for obj in scene.get('objects', []):
            if obj.get('kind') == 'custom' and obj.get('asset') in assets:
                asset = assets[obj['asset']]
                world, faces, centre, scale, rotation, position = _asset_transform(asset, obj)
                boxes = []
                for part in asset['parts']:
                    if part['shape'] != 'box':
                        continue
                    repeat = part.get('repeat', {'count': [1, 1, 1], 'step': [0, 0, 0]})
                    for ix in range(repeat['count'][0]):
                        for iy in range(repeat['count'][1]):
                            for iz in range(repeat['count'][2]):
                                origin = (np.asarray(part['position'], dtype=float) +
                                          np.asarray(repeat['step'], dtype=float) * [ix, iy, iz])
                                boxes.append((origin, np.asarray(part['size'], dtype=float) / 2,
                                              _part_rotation(part.get('rotation', [0, 0, 0]))))
                if boxes:
                    solids.append((obj, centre, scale, rotation, position, boxes))
                triangles = world[faces]
                edges1 = triangles[:, 1] - triangles[:, 0]
                edges2 = triangles[:, 2] - triangles[:, 0]
                normals = np.cross(edges1, edges2)
                magnitudes = np.linalg.norm(normals, axis=1)
                unsafe = obj['asset'] in {'temple-chasm', 'dock-water'}
                for tri, normal, magnitude in zip(triangles, normals, magnitudes):
                    if not unsafe and magnitude > .02 and normal[1] / magnitude >= .85:
                        surfaces.append(WalkableSurface(obj['id'], tuple(map(tuple, tri)), float(normal[1] / magnitude)))
                        if (-.25 <= float(np.mean(tri[:, 1])) <= .25 and
                                np.ptp(tri[:, 0]) >= 4 and np.ptp(tri[:, 2]) >= 4):
                            authored_ground = True
                if 'stair' in (obj.get('name', '') + ' ' + obj['asset']).lower():
                    steps = _tread_steps(asset, obj, centre, scale, rotation, position)
                    if len(steps) >= 2:
                        stair_routes.append({'object_id': obj['id'], 'name': obj.get('name', ''), 'steps': tuple(steps)})
                # The rendered mesh may be sparse or hollow. A whole-object
                # bound is only a coarse avoidance hint, so navigation must
                # prioritize the exact contact triangles above it.
                obstacles.append(obj)
            elif obj.get('kind') != 'custom':
                obstacles.append(obj)
                corners = _ordinary_top(obj)
                if len(corners):
                    for a, b, c in ((0, 1, 2), (0, 2, 3)):
                        surfaces.append(WalkableSurface(obj['id'], tuple(map(tuple, corners[[a, b, c]])), 1.0))
        if not authored_ground:
            # The studio has a visible y=0 stage. Prop-only scenes rely on it;
            # authored large terrain (including the temple's ravine) must keep
            # its real gaps and may never acquire this fallback plane.
            stage = ((-100., 0., -100.), (100., 0., -100.),
                     (100., 0., 100.), (-100., 0., 100.))
            for a, b, c in ((0, 1, 2), (0, 2, 3)):
                surfaces.append(WalkableSurface('__studio_floor__',
                                                (stage[a], stage[b], stage[c]), 1.0))
        return cls(surfaces, stair_routes, obstacles, solids)

    def _inside_solid(self, x, y, z):
        """Check box parts above a candidate foot surface, not whole asset bounds."""
        point = np.asarray((x, y, z), dtype=float)
        for obj, centre, scale, rotation, position, boxes in self._solids:
            local = (point - position) @ rotation
            half = np.asarray(obj['size'], dtype=float) / 2
            if np.any(np.abs(local) > half + .05):
                continue
            raw = local / scale + centre
            for origin, extent, part_rotation in boxes:
                part_point = (raw - origin) @ part_rotation
                if np.all(np.abs(part_point) < extent - 1e-4):
                    return True
        return False

    def support_height(self, x, z, current_y, max_step_up=.45, max_drop=2.0):
        """Highest reachable rendered surface below or just above the feet."""
        bounds = self._xz_bounds
        if not len(bounds):
            return None
        x, z, current_y = float(x), float(z), float(current_y)
        plausible = np.flatnonzero((bounds[:, 0] - 1e-6 <= x) & (x <= bounds[:, 1] + 1e-6) &
                                   (bounds[:, 2] - 1e-6 <= z) & (z <= bounds[:, 3] + 1e-6))
        highest = None
        for i in plausible:
            tri = self._triangles[i]
            a, b, c = tri
            denom = (b[2] - c[2]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[2] - c[2])
            if abs(denom) < 1e-10:
                continue
            u = ((b[2] - c[2]) * (x - c[0]) + (c[0] - b[0]) * (z - c[2])) / denom
            v = ((c[2] - a[2]) * (x - c[0]) + (a[0] - c[0]) * (z - c[2])) / denom
            w = 1 - u - v
            if min(u, v, w) < -1e-5:
                continue
            height = u*a[1] + v*b[1] + w*c[1]
            if current_y - max_drop - 1e-5 <= height <= current_y + max_step_up + 1e-5:
                if self._inside_solid(x, height + .03, z):
                    continue
                if highest is None or height > highest:
                    highest = float(height)
        return highest

    def obstacle_at(self, x, y, z, radius=0.0, ignore_object_ids=()):
        """Conservative occupied object bound at a body point.

        Callers should ignore their supporting object or pass a point above its
        top. This broad phase is intended for path avoidance, not hand contact.
        """
        ignore = set(ignore_object_ids)
        for obj in self._objects:
            if obj['id'] in ignore:
                continue
            # These authored slabs are navigable surfaces. Use exact asset
            # identifiers rather than user-editable names when exempting them.
            walk_through_assets = {'temple-earth', 'temple-terrace', 'temple-stairs',
                                   'temple-bridge', 'temple-chasm', 'temple-arch',
                                   'temple-vines', 'dock-ground', 'dock-quay',
                                   'dock-water', 'dock-platform', 'dock-stairs',
                                   'dock-catwalk', 'dock-gantry'}
            if obj.get('asset') in walk_through_assets:
                continue
            centre = np.asarray(obj['position'], dtype=float)
            local = (np.asarray((x, y, z), dtype=float) - centre) @ _yaw(obj.get('yaw', 0))
            half = np.asarray(obj['size'], dtype=float) / 2
            if np.all(np.abs(local) <= half + [radius, 0, radius]):
                return True
        return False


def build_scene_geometry(scene):
    """Convenience entry point for motion code."""
    return SceneInteractionGeometry.from_scene(scene)
