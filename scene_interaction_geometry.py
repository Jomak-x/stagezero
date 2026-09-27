"""World-space contact geometry derived from the meshes used to draw a scene.

Custom assets use the renderer's mesh bounds normalization, including yaw and
repeated parts. A motion planner can sample a floor at the character's feet
without mistaking an elevated roof or bridge for the nearby ground.
"""

from __future__ import annotations

import math
import re
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
        world_size = size * scale
        if min(world_size[0], world_size[2]) < .15 or max(world_size[0], world_size[2]) < .45 or world_size[1] > .50:
            continue
        # Steps are horizontal. Decorative sloped blocks are not treads.
        if any(abs(part.get('rotation', [0, 0, 0])[axis]) > 1e-6 for axis in (0, 2)):
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
        self._primitive_ids, transforms, offsets, extents, shapes = [], [], [], [], []
        for obj, centre, scale, rotation, position, parts in solids:
            for origin, extent, part_rotation, shape in parts:
                transform = (rotation / scale) @ part_rotation
                self._primitive_ids.append(obj['id'])
                transforms.append(transform)
                offsets.append((centre - origin) @ part_rotation - position @ transform)
                extents.append(extent)
                shapes.append(shape)
        self._primitive_transforms = np.asarray(transforms).reshape((-1, 3, 3))
        self._primitive_offsets = np.asarray(offsets).reshape((-1, 3))
        self._primitive_extents = np.asarray(extents).reshape((-1, 3))
        self._primitive_shapes = np.asarray(shapes)
        # Each rendered part has a fixed affine world-to-part transform.  A
        # world AABB of its (radius-expanded) local box rejects distant parts
        # before the exact box/sphere/cylinder/cone calculation below.
        if len(self._primitive_extents):
            inverse = np.linalg.inv(self._primitive_transforms)
            self._primitive_world_centres = -np.einsum(
                'ni,nij->nj', self._primitive_offsets, inverse)
            self._primitive_padding = np.linalg.norm(
                self._primitive_transforms[:, [0, 2]], axis=1)
            absolute_inverse = np.abs(inverse)
            self._primitive_world_half = np.einsum(
                'ni,nij->nj', self._primitive_extents, absolute_inverse)
            self._primitive_world_radius_half = np.einsum(
                'ni,nij->nj', self._primitive_padding, absolute_inverse)
        else:
            self._primitive_world_centres = np.zeros((0, 3))
            self._primitive_padding = np.zeros((0, 3))
            self._primitive_world_half = np.zeros((0, 3))
            self._primitive_world_radius_half = np.zeros((0, 3))
        self._custom_ids = {obj['id'] for obj, *_ in solids}
        triangles = np.asarray([s.triangle for s in surfaces], dtype=float).reshape((-1, 3, 3))
        self._triangles = triangles
        self._triangle_a, self._triangle_b, self._triangle_c = (
            triangles[:, 0], triangles[:, 1], triangles[:, 2])
        if len(triangles):
            x = triangles[:, :, 0]
            z = triangles[:, :, 2]
            self._xz_bounds = np.column_stack((x.min(axis=1), x.max(axis=1), z.min(axis=1), z.max(axis=1)))
            a, b, c = self._triangle_a, self._triangle_b, self._triangle_c
            self._triangle_denominator = ((b[:, 2]-c[:, 2])*(a[:, 0]-c[:, 0]) +
                                          (c[:, 0]-b[:, 0])*(a[:, 2]-c[:, 2]))
            self._triangle_y_bounds = np.column_stack((triangles[:, :, 1].min(axis=1),
                                                        triangles[:, :, 1].max(axis=1)))
        else:
            self._xz_bounds = np.zeros((0, 4))
            self._triangle_denominator = np.zeros(0)
            self._triangle_y_bounds = np.zeros((0, 2))

    @classmethod
    def from_scene(cls, scene, *, include_studio_floor=None):
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
                surface_name = (obj.get('name', '') + ' ' + asset.get('name', '')).lower()
                unsafe = bool(re.search(r'\b(water|ocean|lake|lava|chasm|ravine|void|basin)\b', surface_name)
                              and not re.search(r'\b(bridge|deck|walkway|platform|quay|bank|tower|tank|pump|pipe|wheel|door|wall)\b', surface_name))
                # Terrain recognition must survive translating the entire scene
                # vertically. Looking only near world y=0 invents a walkable
                # plane across transformed ravines. Thin broad authored floors
                # and explicitly unsafe terrain replace the implicit stage.
                sx, sy, sz = obj['size']
                if unsafe or (min(sx, sz) >= 4 and sy <= .5):
                    authored_ground = True
                for part in asset['parts']:
                    repeat = part.get('repeat', {'count': [1, 1, 1], 'step': [0, 0, 0]})
                    for ix in range(repeat['count'][0]):
                        for iy in range(repeat['count'][1]):
                            for iz in range(repeat['count'][2]):
                                origin = (np.asarray(part['position'], dtype=float) +
                                          np.asarray(repeat['step'], dtype=float) * [ix, iy, iz])
                                boxes.append((origin, np.asarray(part['size'], dtype=float) / 2,
                                              _part_rotation(part.get('rotation', [0, 0, 0])), part['shape']))
                if boxes and not unsafe:
                    solids.append((obj, centre, scale, rotation, position, boxes))
                triangles = world[faces]
                edges1 = triangles[:, 1] - triangles[:, 0]
                edges2 = triangles[:, 2] - triangles[:, 0]
                normals = np.cross(edges1, edges2)
                magnitudes = np.linalg.norm(normals, axis=1)
                for tri, normal, magnitude in zip(triangles, normals, magnitudes):
                    if not unsafe and magnitude > .02 and normal[1] / magnitude >= .85:
                        surfaces.append(WalkableSurface(obj['id'], tuple(map(tuple, tri)), float(normal[1] / magnitude)))
                        if (-.25 <= float(np.mean(tri[:, 1])) <= .25 and
                                np.ptp(tri[:, 0]) >= 4 and np.ptp(tri[:, 2]) >= 4):
                            authored_ground = True
                steps = _tread_steps(asset, obj, centre, scale, rotation, position)
                if len(steps) >= 3:
                    delta = np.diff(np.asarray(steps), axis=0)
                    distances = np.linalg.norm(delta[:, [0, 2]], axis=1)
                    # A flight is a sequence of rising, spaced, collinear
                    # surfaces. Detect its geometry independently of names.
                    direction = delta[0, [0, 2]] / max(distances[0], 1e-9)
                    aligned = np.all(delta[:, [0, 2]] @ direction > .05)
                    straight = np.max(np.abs(delta[:, 0]*direction[1] - delta[:, 2]*direction[0])) < .15
                    if aligned and straight and np.all(distances < 1.) and np.all(delta[:, 1] <= .55):
                        stair_routes.append({'object_id': obj['id'], 'name': obj.get('name', 'Stairs'), 'steps': tuple(steps)})
                # The rendered mesh may be sparse or hollow. A whole-object
                # bound is only a coarse avoidance hint, so navigation must
                # prioritize the exact contact triangles above it.
                if not unsafe:
                    obstacles.append(obj)
            elif obj.get('kind') != 'custom':
                obstacles.append(obj)
                corners = _ordinary_top(obj)
                if len(corners):
                    for a, b, c in ((0, 1, 2), (0, 2, 3)):
                        surfaces.append(WalkableSurface(obj['id'], tuple(map(tuple, corners[[a, b, c]])), 1.0))
        if include_studio_floor is True or (include_studio_floor is None and not authored_ground):
            # The studio has a visible y=0 stage. Prop-only scenes rely on it;
            # authored large terrain (including the temple's ravine) must keep
            # its real gaps and may never acquire this fallback plane.
            stage = ((-100., 0., -100.), (100., 0., -100.),
                     (100., 0., 100.), (-100., 0., 100.))
            for a, b, c in ((0, 1, 2), (0, 2, 3)):
                surfaces.append(WalkableSurface('__studio_floor__',
                                                (stage[a], stage[b], stage[c]), 1.0))
        return cls(surfaces, stair_routes, obstacles, solids)

    def _primitive_hits(self, x, y, z, radius=0., ignore_object_ids=()):
        if not len(self._primitive_extents):
            return False
        point = np.asarray((x, y, z), dtype=float)
        if radius >= 0.:
            half = self._primitive_world_half + radius*self._primitive_world_radius_half
            # Inclusive with a small numerical margin: the exact local test
            # below remains authoritative, including at rotated boundaries.
            candidates = np.flatnonzero(np.all(
                np.abs(point-self._primitive_world_centres) <= half+1e-7,
                axis=1))
        else:
            # Preserve the previous behaviour for the unusual negative-radius
            # caller instead of relying on an invalid expanded AABB.
            candidates = np.arange(len(self._primitive_extents))
        if ignore_object_ids and len(candidates):
            candidates = candidates[~np.isin(np.asarray(self._primitive_ids)[candidates],
                                              tuple(ignore_object_ids))]
        if not len(candidates):
            return False
        local = np.einsum('i,nij->nj', point, self._primitive_transforms[candidates]) + self._primitive_offsets[candidates]
        extent = self._primitive_extents[candidates] + radius*self._primitive_padding[candidates]
        normalized = local / np.maximum(extent, 1e-9)
        hit = np.all(np.abs(normalized) < 1. - 1e-6, axis=1)
        sphere = self._primitive_shapes[candidates] == 'sphere'
        hit[sphere] &= np.sum(normalized[sphere]**2, axis=1) < 1.
        round_parts = np.isin(self._primitive_shapes[candidates], ('cylinder', 'cone'))
        radial = np.sum(normalized[:, [0, 2]]**2, axis=1)
        cone = self._primitive_shapes[candidates] == 'cone'
        allowed = np.ones(len(radial))
        allowed[cone] = ((1.-normalized[cone, 1]) / 2.)**2
        hit[round_parts] &= radial[round_parts] < allowed[round_parts]
        return bool(np.any(hit))

    def _inside_solid(self, x, y, z):
        """Exclude surfaces buried inside rendered primitive volumes."""
        return self._primitive_hits(x, y, z)

    def support_height(self, x, z, current_y, max_step_up=.45, max_drop=2.0):
        """Highest reachable rendered surface below or just above the feet."""
        bounds = self._xz_bounds
        if not len(bounds):
            return None
        x, z, current_y = float(x), float(z), float(current_y)
        plausible = np.flatnonzero((bounds[:, 0] - 1e-6 <= x) & (x <= bounds[:, 1] + 1e-6) &
                                   (bounds[:, 2] - 1e-6 <= z) & (z <= bounds[:, 3] + 1e-6) &
                                   (self._triangle_y_bounds[:, 0] <= current_y+max_step_up+1e-5) &
                                   (self._triangle_y_bounds[:, 1] >= current_y-max_drop-1e-5) &
                                   (np.abs(self._triangle_denominator) >= 1e-10))
        if not len(plausible):
            return None
        # Batch barycentric tests, then inspect highest candidates first. Sole
        # fitting makes many queries; checking every lower/duplicate triangle
        # against every solid needlessly dominated replay time.
        a, b, c = (self._triangle_a[plausible], self._triangle_b[plausible],
                   self._triangle_c[plausible])
        denominator = self._triangle_denominator[plausible]
        u = ((b[:, 2]-c[:, 2])*(x-c[:, 0]) + (c[:, 0]-b[:, 0])*(z-c[:, 2]))/denominator
        v = ((c[:, 2]-a[:, 2])*(x-c[:, 0]) + (a[:, 0]-c[:, 0])*(z-c[:, 2]))/denominator
        w = 1-u-v
        heights = u*a[:, 1] + v*b[:, 1] + w*c[:, 1]
        valid = ((np.minimum(np.minimum(u, v), w) >= -1e-5)
                 & (heights >= current_y-max_drop-1e-5)
                 & (heights <= current_y+max_step_up+1e-5))
        for height in np.unique(heights[valid])[::-1]:
            if not self._inside_solid(x, float(height)+.03, z):
                return float(height)
        return None

    def obstacle_at(self, x, y, z, radius=0.0, ignore_object_ids=()):
        """Occupied rendered parts, preserving gaps in custom arches/bridges.

        No asset identifier receives a collision exemption. Procedural parts
        are checked in their own transformed coordinate system; ordinary props
        use conservative rotated bounds except the actual opening of arches.
        """
        if self._primitive_hits(x, y, z, radius, ignore_object_ids):
            return True
        ignore = set(ignore_object_ids) | self._custom_ids
        for obj in self._objects:
            if obj['id'] in ignore:
                continue
            centre = np.asarray(obj['position'], dtype=float)
            local = (np.asarray((x, y, z), dtype=float) - centre) @ _yaw(obj.get('yaw', 0))
            half = np.asarray(obj['size'], dtype=float) / 2
            if obj['kind'] == 'door':
                # Match ObjectSceneLayer._build: panel, header and jambs are
                # one static assembly. Automatic opening moves its evaluated
                # position upward. An unrendered angle flag cannot open it.
                sx, sy, sz = np.asarray(obj['size'], dtype=float)
                for origin, extent in (([0, 0, 0], [.42*sx, .46*sy, .275*sz]),
                                       ([0, .48*sy, 0], [sx/2, .02*sy, sz/2]),
                                       ([-.46*sx, 0, 0], [.04*sx, sy/2, sz/2]),
                                       ([.46*sx, 0, 0], [.04*sx, sy/2, sz/2])):
                    if np.all(np.abs(local-origin) <= np.asarray(extent)+[radius, 0, radius]):
                        return True
                handle = local - [.28*sx, -.04*sy, .32*sz]
                handle_radius = .16*min(sx, sy, sz)
                if (abs(handle[1]) <= handle_radius and
                        np.linalg.norm(handle[[0, 2]]) <= radius + math.sqrt(max(0., handle_radius**2-handle[1]**2))):
                    return True
                continue
            if np.all(np.abs(local) <= half + [radius, 0, radius]):
                if obj['kind'] == 'arch' and abs(local[0]) < obj['size'][0] * .34 - radius and local[1] < obj['size'][1] * .33:
                    continue
                return True
        return False


def build_scene_geometry(scene):
    """Convenience entry point for motion code."""
    return SceneInteractionGeometry.from_scene(scene)
