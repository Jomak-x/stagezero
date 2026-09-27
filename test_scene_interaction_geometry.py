"""Equivalence checks for immutable rendered-geometry broadphases."""
from __future__ import annotations

import unittest

import numpy as np

from scene_composition import make_preset
from scene_interaction_geometry import SceneInteractionGeometry


def _full_primitive_test(geometry, point, radius=0., ignored=()):
    """Original all-parts local calculation, kept independent of the AABB."""
    local = np.einsum('i,nij->nj', point, geometry._primitive_transforms) + geometry._primitive_offsets
    padding = radius*np.linalg.norm(geometry._primitive_transforms[:, [0, 2]], axis=1)
    extent = geometry._primitive_extents + padding
    normalized = local/np.maximum(extent, 1e-9)
    hit = np.all(np.abs(normalized) < 1.-1e-6, axis=1)
    sphere = geometry._primitive_shapes == 'sphere'
    hit[sphere] &= np.sum(normalized[sphere]**2, axis=1) < 1.
    round_parts = np.isin(geometry._primitive_shapes, ('cylinder', 'cone'))
    radial = np.sum(normalized[:, [0, 2]]**2, axis=1)
    cone = geometry._primitive_shapes == 'cone'
    allowed = np.ones(len(radial))
    allowed[cone] = ((1.-normalized[cone, 1])/2.)**2
    hit[round_parts] &= radial[round_parts] < allowed[round_parts]
    if ignored:
        hit &= ~np.isin(geometry._primitive_ids, tuple(ignored))
    return bool(np.any(hit))


def _scalar_support(geometry, x, z, current_y, max_step_up=.45, max_drop=2.):
    highest = None
    for tri in geometry._triangles:
        a, b, c = tri
        denominator = ((b[2]-c[2])*(a[0]-c[0]) +
                       (c[0]-b[0])*(a[2]-c[2]))
        if abs(denominator) < 1e-10:
            continue
        u = ((b[2]-c[2])*(x-c[0]) + (c[0]-b[0])*(z-c[2]))/denominator
        v = ((c[2]-a[2])*(x-c[0]) + (a[0]-c[0])*(z-c[2]))/denominator
        w = 1.-u-v
        if min(u, v, w) < -1e-5:
            continue
        height = float(u*a[1] + v*b[1] + w*c[1])
        if current_y-max_drop-1e-5 <= height <= current_y+max_step_up+1e-5:
            if not _full_primitive_test(geometry, (x, height+.03, z)):
                highest = height if highest is None else max(highest, height)
    return highest


class GeometryBroadphaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = SceneInteractionGeometry.from_scene(make_preset('Jungle temple'))

    def test_rotated_nonuniform_primitives_match_full_local_calculation(self):
        geometry = self.geometry
        self.assertIn('cone', set(geometry._primitive_shapes))
        self.assertIn('sphere', set(geometry._primitive_shapes))
        rng = np.random.default_rng(1907)
        points = rng.uniform((-15., -1., -30.), (15., 12., 5.), size=(350, 3))
        # Also probe directly around actual transformed part centres and AABBs.
        samples = rng.choice(len(geometry._primitive_world_centres), 150, replace=True)
        points = np.vstack((points, geometry._primitive_world_centres[samples] +
                            rng.normal(size=(len(samples), 3))*geometry._primitive_world_half[samples]))
        ignored = (geometry._primitive_ids[0], geometry._primitive_ids[-1])
        for radius, omit in ((0., ()), (.18, ()), (.18, ignored), (-.01, ())):
            for point in points:
                self.assertEqual(geometry._primitive_hits(*point, radius, omit),
                                 _full_primitive_test(geometry, point, radius, omit))

    def test_support_height_matches_scalar_triangles(self):
        geometry = self.geometry
        rng = np.random.default_rng(911)
        points = rng.uniform((-4., -9.), (4., 1.), size=(90, 2))
        for x, z in points:
            for reference in (0., 2., 3.):
                expected = _scalar_support(geometry, x, z, reference,
                                           max_step_up=.3, max_drop=.5)
                actual = geometry.support_height(x, z, reference,
                                                 max_step_up=.3, max_drop=.5)
                if expected is None:
                    self.assertIsNone(actual)
                else:
                    self.assertAlmostEqual(actual, expected, places=7)


if __name__ == '__main__':
    unittest.main()
