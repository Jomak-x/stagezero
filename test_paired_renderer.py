"""CPU geometry checks for the opt-in paired research renderer.

These check faithful endpoint fitting, not animation quality or mesh contact.
"""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from realtime_clip import CanonicalClip
from studio_core_renderer import StudioCoreRenderer
from test_studio_core_renderer import FakeScene


class PairedRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(Path(__file__).parent / 'assets/core-motion/martial-combo.npz', allow_pickle=False) as a:
            cls.positions = a['positions'][:, :80].copy()
            cls.rotations = a['rotations'][:, :80].copy()

    def renderer(self, paired=True):
        result = StudioCoreRenderer(SimpleNamespace(scene=FakeScene()), paired_retarget=paired)
        self.addCleanup(result.remove)
        return result

    def clip(self, frames=80):
        # A geometry fixture: both source trajectories share one translation.
        # This is not presented as a generated paired performance.
        p = np.concatenate((self.positions[:, :frames], self.positions[:, :frames] + [.8, .025, 0]))
        r = np.concatenate((self.rotations[:, :frames], self.rotations[:, :frames]))
        return CanonicalClip.from_arrays(p, r, actor_ids=('a', 'b'), source='intergen')

    def test_wrists_follow_body_height_correction_and_preserve_pair_gap(self):
        renderer = self.renderer()
        clip = self.clip(40)
        renderer.set_clip(clip)
        self.assertEqual(renderer.floor_offsets[0], renderer.floor_offsets[1])
        ids = [renderer.characters[0].core_idx[name] for name in ('LeftHand', 'RightHand')]
        targets = clip.positions[:, :, ids].copy()
        targets[:, :, :, 1] += renderer.floor_offsets[0] + renderer.characters[0].root_height_offset
        np.testing.assert_allclose(renderer.fitted_positions[:, :, [5, 8]], targets, atol=1e-6)
        # Hand-root vectors retain their native height relationship rather
        # than lifting wrists relative to the lowered canonical mesh body.
        np.testing.assert_allclose(
            renderer.fitted_positions[:, :, [5, 8]] - renderer.fitted_positions[:, :, 0:1],
            clip.positions[:, :, ids] - clip.positions[:, :, 0:1], atol=1e-6)
        provenance = renderer.fitting_provenance
        self.assertAlmostEqual(provenance['shared_root_height_correction_m'],
                               renderer.characters[0].root_height_offset)
        self.assertLess(provenance['wrist_target_error_m']['max'], 1e-6)
        self.assertTrue(all(row['gap_drift_max_m'] < 1e-6 for row in provenance['wrist_gaps'].values()))
        # Provenance access cannot corrupt subsequent capture measurements.
        provenance['floor_offsets_m'][0] = 100
        self.assertNotEqual(renderer.fitting_provenance['floor_offsets_m'][0], 100)

    def test_mismatched_character_root_mappings_fail_before_rendering(self):
        with patch('studio_core_renderer.GroundedCharacter', side_effect=[
                SimpleNamespace(root_height_offset=-.15),
                SimpleNamespace(root_height_offset=-.05)]):
            with self.assertRaisesRegex(ValueError, 'matching character root-height'):
                StudioCoreRenderer(SimpleNamespace(scene=FakeScene()), paired_retarget=True)

    def test_append_keeps_common_offset_and_every_fitted_prefix(self):
        renderer = self.renderer()
        clip = self.clip()
        renderer.set_clip(clip.slice_frames(0, 40))
        positions = renderer.fitted_positions.copy()
        rotations = renderer.fitted_rotations.copy()
        offsets = renderer.floor_offsets
        renderer.set_clip(clip)
        np.testing.assert_array_equal(renderer.fitted_positions[:, :40], positions)
        np.testing.assert_array_equal(renderer.fitted_rotations[:, :40], rotations)
        self.assertEqual(renderer.floor_offsets, offsets)
        renderer.tick(70)
        renderer.tick(0)
        np.testing.assert_array_equal(renderer.fitted_positions[:, :40], positions)

    def test_unreachable_wrists_are_reported_instead_of_claimed_preserved(self):
        renderer = self.renderer()
        clip = self.clip(1)
        p = clip.positions.copy()
        wrist = renderer.characters[0].core_idx['LeftHand']
        p[0, 0, wrist, 0] += 5
        renderer.set_clip(CanonicalClip.from_arrays(p, clip.rotations, actor_ids=clip.actor_ids, source='intergen'))
        errors = renderer.fitting_provenance['wrist_target_error_m']
        self.assertGreater(errors['max'], 3)
        self.assertGreater(errors['fraction_over_5mm'], 0)

    def test_opt_in_requires_pair_and_default_does_not_enable_wrist_ik(self):
        renderer = self.renderer()
        clip = self.clip(1)
        single = CanonicalClip.from_arrays(clip.positions[:1], clip.rotations[:1], actor_ids=('a',), source='ardy_core')
        with self.assertRaisesRegex(ValueError, 'exactly two'):
            renderer.set_clip(single)
        ordinary = self.renderer(False)
        ordinary.set_clip(single)
        self.assertFalse(ordinary.fitting_provenance['preserve_native_wrists'])


if __name__ == '__main__':
    unittest.main()
