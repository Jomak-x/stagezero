"""Opt-in terrain fitting retains native motion and cached fitted frames."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from grounded_character import GroundedCharacter
from motion_bridge import _layout
from realtime_clip import CanonicalClip
from studio_core_renderer import StudioCoreRenderer
from test_studio_core_renderer import FakeScene


ROOT = Path(__file__).resolve().parent


class Plane:
    def __init__(self, height):
        self.height = height

    def support_height(self, x, z, current_y, max_step_up=.45, max_drop=2.):
        height = self.height(x, z) if callable(self.height) else self.height
        return height if current_y - max_drop <= height <= current_y + max_step_up else None


def neutral_frames(height, count=5):
    _, _, neutral = _layout()
    positions = np.repeat(neutral[None], count, axis=0)
    positions[:, :, 1] += height - neutral[:, 1].min()
    rotations = np.broadcast_to(np.eye(3), (count, 27, 3, 3)).copy()
    return positions, rotations


class TerrainCharacterFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.character = GroundedCharacter(ROOT / 'assets/core-characters/civilian.glb')

    def fit(self, native_height, render_height):
        positions, rotations = neutral_frames(native_height)
        terrain = Plane(render_height)
        source = Plane(native_height)
        return self.character.clip_payload(positions, rotations, preserve_feet=True,
            floor_y=0., source_terrain_geometry=source, render_terrain_geometry=terrain)

    def test_vertical_translation_keeps_calibration_and_ik_equivalent(self):
        ground = self.fit(0., 0.)
        raised = self.fit(1., 1.)
        low_p = np.asarray(ground['fitted_positions'])
        high_p = np.asarray(raised['fitted_positions'])
        np.testing.assert_allclose(high_p - low_p, np.broadcast_to([0., 1., 0.], high_p.shape), atol=1e-7)
        np.testing.assert_allclose(raised['fitted_rotations'], ground['fitted_rotations'], atol=1e-7)
        np.testing.assert_allclose(raised['character_provenance']['floor_offsets'],
                                   ground['character_provenance']['floor_offsets'], atol=1e-7)
        self.assertGreaterEqual(raised['character_provenance']['floor_calibration'][0]['support_sample_count'], 3)
        self.assertIn('max_knee_angle_change_deg', raised['character_provenance']['foot_retarget'])

    def test_nonpenetrating_terrain_never_pulls_ankles_down(self):
        positions, rotations = neutral_frames(0.)
        original = self.character.clip_payload(positions, rotations, floor_offsets=[0.])
        original_p = np.asarray(original['fitted_positions'])
        original_r = np.asarray(original['fitted_rotations'])
        sole = self.character.deform_vertices(original_p[0, 0], original_r[0, 0])
        bottom = min(float(sole[ids, 1].min()) for ids in self.character.foot_surface_indices)
        below = Plane(bottom - .2)
        fitted = self.character.clip_payload(positions, rotations, preserve_feet=True,
            floor_offsets=[0.], source_terrain_geometry=Plane(0.), render_terrain_geometry=below)
        np.testing.assert_array_equal(fitted['fitted_positions'], original['fitted_positions'])
        np.testing.assert_array_equal(fitted['fitted_rotations'], original['fitted_rotations'])
        self.assertEqual(fitted['character_provenance']['foot_retarget']['max_ankle_displacement_m'], 0.)

    def test_each_foot_uses_its_own_render_support(self):
        positions, rotations = neutral_frames(0.)
        baseline = self.character.clip_payload(positions, rotations, floor_offsets=[0.])
        base_p = np.asarray(baseline['fitted_positions'])
        base_r = np.asarray(baseline['fitted_rotations'])
        left_surface, right_surface = self.character.foot_surface_indices
        left_x = np.mean(self.character._sole_vertices(left_surface, base_p[0, 0], base_r[0, 0])[:, 0])
        right_x = np.mean(self.character._sole_vertices(right_surface, base_p[0, 0], base_r[0, 0])[:, 0])
        midpoint = (left_x + right_x) / 2
        raised_side = left_x > right_x
        terrain = Plane(lambda x, z: .2 if (x > midpoint) == raised_side else -1.)
        fitted = self.character.clip_payload(positions, rotations, preserve_feet=True,
            floor_offsets=[0.], source_terrain_geometry=Plane(0.), render_terrain_geometry=terrain)
        p = np.asarray(fitted['fitted_positions'])
        lifted, untouched = (11, 14) if raised_side else (14, 11)
        self.assertGreater(float(p[0, 0, lifted, 1] - base_p[0, 0, lifted, 1]), .01)
        self.assertAlmostEqual(float(p[0, 0, untouched, 1] - base_p[0, 0, untouched, 1]), 0., places=7)

    def test_renderer_append_and_seek_preserve_fitted_prefix(self):
        positions, rotations = neutral_frames(1., count=8)
        scene = FakeScene()
        renderer = StudioCoreRenderer(SimpleNamespace(scene=scene), terrain_geometry=Plane(1.))
        self.addCleanup(renderer.remove)
        first = CanonicalClip.from_arrays(positions[None, :5], rotations[None, :5], actor_ids=('a',), source='terrain_test')
        full = CanonicalClip.from_arrays(positions[None], rotations[None], actor_ids=('a',), source='terrain_test')
        renderer.set_clip(first)
        prefix_p = renderer.fitted_positions.copy()
        prefix_r = renderer.fitted_rotations.copy()
        offsets = renderer.floor_offsets
        renderer.set_clip(full)
        np.testing.assert_array_equal(renderer.fitted_positions[:, :5], prefix_p)
        np.testing.assert_array_equal(renderer.fitted_rotations[:, :5], prefix_r)
        self.assertEqual(renderer.floor_offsets, offsets)
        renderer.tick(7)
        renderer.tick(1)
        np.testing.assert_array_equal(renderer.fitted_positions[:, :5], prefix_p)
        self.assertIn('foot_clearance', renderer.fitting_provenance)


if __name__ == '__main__':
    unittest.main()
