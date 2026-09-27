"""Bounded noncontact arrival targets and displayed approach acceptance."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.native_pair_rig import NativeRigAsset
from native_pair_clip import NativePairClip
from native_pair_transition import core_to_pair_anatomy, shared_place_pair
from paired_meetup import build_meetup, plan_meetup, _approach_body_clearance
from test_native_pair_rig import fixture_glb
from test_paired_meetup import Client, SCENE, IDS, STARTS, MEETING


class ContactMarginTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name)/'fixture.glb'
        path.write_bytes(fixture_glb()[0])
        pose = NativeRigAsset(path).rest
        joints = np.repeat(np.stack([pose+[-1, 0, 0], pose+[1, 0, 0]])[None], 8, axis=0)
        self.pair = NativePairClip(joints, metadata={'model': 'InterGen'})

    def plan(self, **kwargs):
        return plan_meetup(self.pair, SCENE, actor_ids=IDS, starts=STARTS, meeting=MEETING,
                           entry_policy='continuous', **kwargs)

    def placed(self):
        placement = self.plan()['placement']
        return shared_place_pair(self.pair.joints, translation=[placement['x'], 0, placement['z']])

    def build(self, client=None, **kwargs):
        return build_meetup(self.pair, client or Client(self.placed()), SCENE, actor_ids=IDS,
                            starts=STARTS, meeting=MEETING, entry_policy='continuous', **kwargs)

    def test_margin_is_additive_even_above_minimum_standoff(self):
        original = self.pair.joints.copy()
        plan = self.plan(arrival_standoff_m=.60, arrival_margin_m=.08)
        report = plan['arrival_standoff']
        self.assertAlmostEqual(report['native_entry_root_separation_m'], 2.)
        self.assertAlmostEqual(report['core_arrival_root_separation_m'], 2.16)
        self.assertAlmostEqual(report['maximum_target_offset_m'], .08)
        self.assertEqual(report['additive_margin_per_actor_m'], .08)
        for route, target in zip(plan['routes'], report['core_arrival_targets_xz']):
            np.testing.assert_allclose(route['points'][-1], target)
        np.testing.assert_array_equal(original, self.pair.joints)
        self.assertEqual(plan['limits']['max_arrival_error_m'], .4)
        json.dumps(plan, allow_nan=False)

    def test_margin_is_added_after_minimum_standoff(self):
        close = self.pair.joints.copy()
        close[:, 0, :, 0] += .73
        close[:, 1, :, 0] -= .73
        self.pair = NativePairClip(close, metadata={'model': 'InterGen'})
        plan = self.plan(arrival_standoff_m=.60, arrival_margin_m=.08)
        self.assertAlmostEqual(plan['arrival_standoff']['native_entry_root_separation_m'], .54)
        self.assertAlmostEqual(plan['arrival_standoff']['core_arrival_root_separation_m'], .76)
        self.assertAlmostEqual(plan['arrival_standoff']['maximum_target_offset_m'], .11)

    def test_zero_margin_keeps_existing_targets(self):
        self.assertEqual(self.plan(), self.plan(arrival_margin_m=0.))

    def test_margin_is_bounded_and_finite(self):
        for margin in (-.01, .201, float('nan'), float('inf'), True):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                self.plan(arrival_margin_m=margin)

    def test_margin_build_preserves_native_frames_and_reports_clearance(self):
        original = self.pair.joints.copy()
        result = self.build(arrival_margin_m=.08)
        report = result['metadata']['approach_body_clearance']
        self.assertTrue(report['checked'])
        self.assertTrue(report['passed'])
        self.assertFalse(report['contact_allowed'])
        self.assertEqual(report['frames'], result['metadata']['segments'][0]['frames'])
        self.assertEqual(result['plan']['blend_frames'], 21)
        np.testing.assert_allclose(result['metadata']['arrival_errors_m'], [.08, .08], atol=1e-12)
        np.testing.assert_allclose(result['metadata']['arrival_target_errors_m'], [0., 0.], atol=1e-12)
        np.testing.assert_array_equal(result['joints'][-self.pair.frames:], self.placed())
        np.testing.assert_array_equal(result['clip'].joints[-self.pair.frames:], original)
        np.testing.assert_array_equal(self.pair.joints, original)
        json.dumps(report, allow_nan=False)

    def test_display_overlap_rejected_after_raw_sources_are_archived(self):
        archived = []

        def overlapping_display(*args, **kwargs):
            approach, report = core_to_pair_anatomy(*args, **kwargs)
            approach[17, 0, 20] = approach[17, 1, 0]
            return approach, report

        with patch('paired_meetup.core_to_pair_anatomy', side_effect=overlapping_display):
            with self.assertRaisesRegex(ValueError, 'unintended body-proxy overlap') as caught:
                self.build(arrival_margin_m=.08, on_core_chunk=lambda *args: archived.append(args))
        self.assertTrue(archived)
        report = caught.exception.approach_body_clearance_report
        self.assertFalse(report['passed'])
        self.assertIn(17, report['overlap_frame_indices'])
        self.assertLess(report['minimum_clearance_m'], 0.)
        json.dumps(report, allow_nan=False)

    def test_clearance_checks_every_display_frame(self):
        approach = np.repeat(self.placed()[:1], 60, axis=0)
        approach[37, 0, 20] = approach[37, 1, 0]
        report = _approach_body_clearance(approach, IDS)
        self.assertEqual(report['overlap_frame_indices'], [37])
        self.assertEqual(report['minimum_clearance_frame'], 37)

    def test_margin_does_not_increase_native_arrival_error_budget(self):
        with self.assertRaisesRegex(ValueError, 'did not reach both meeting entry positions'):
            self.build(Client(self.placed(), drift=.41), arrival_margin_m=.08)

if __name__ == '__main__':
    unittest.main()
