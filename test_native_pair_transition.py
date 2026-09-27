"""Geometry/provenance checks, explicitly not animation acceptance."""
import ast
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np

from native_pair_transition import (
    CORE27_NAMES, CORE_TO_NATIVE, NATIVE22_NAMES, PARENTS, authored_boundary_bridge,
    authored_direction_bridge, boundary_diagnostics, core27_to_native22,
    core_to_pair_anatomy, compose_ardy_pair_context, build_ardy_pair_context,
    shared_place_pair, stabilize_authored_feet,
)
from test_native_pair_rig import fixture_glb
from experiments.native_pair_rig import NativeRigAsset
from tempfile import TemporaryDirectory


class NativePairTransitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / 'rig.glb'; path.write_bytes(fixture_glb()[0])
        self.pose = NativeRigAsset(path).rest
        self.pair = np.repeat(np.stack([self.pose, self.pose + [2, 0, 0]])[None], 8, axis=0)

    def test_semantic_mapping_matches_vendor_and_exact_endpoints(self):
        tree = ast.parse((Path(__file__).parent / 'vendor/ardy/ardy/skeleton/definitions.py').read_text())
        core = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CoreSkeleton27')
        names = next(ast.literal_eval(n.value) for n in core.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'bone_order_names_with_parents' for t in n.targets))
        self.assertEqual(tuple(n for n, _ in names), CORE27_NAMES)
        p = np.arange(2*5*27*3, dtype=np.float32).reshape(2, 5, 27, 3)
        out = core27_to_native22(p)
        np.testing.assert_array_equal(out, p[..., CORE_TO_NATIVE, :])
        order = np.arange(27)[::-1]
        np.testing.assert_array_equal(core27_to_native22(p[..., order, :], joint_names=np.array(CORE27_NAMES)[order]), out)
        out[...] = 0
        self.assertNotEqual(float(p.sum()), 0.)
        for name in ('LeftArm', 'LeftForeArm', 'LeftHand', 'Hips'):
            self.assertEqual(CORE27_NAMES[CORE_TO_NATIVE[NATIVE22_NAMES.index(name)]], name)

    def test_mapping_rejects_anonymous_wrong_schema_and_nonfinite(self):
        with self.assertRaises(ValueError): core27_to_native22(np.zeros((27, 3)), joint_names=['Hips']*27)
        with self.assertRaises(ValueError): core27_to_native22(np.full((27, 3), np.nan))
        with self.assertRaises(ValueError): core27_to_native22(np.zeros((22, 3)))

    def test_shared_rigid_placement_preserves_all_pair_contact_distances(self):
        before = self.pair.copy()
        out = shared_place_pair(self.pair, yaw=.8, translation=[4., .3, -2.])
        original_dist = np.linalg.norm(before[:, 0, :, None] - before[:, 1, None, :], axis=-1)
        placed_dist = np.linalg.norm(out[:, 0, :, None] - out[:, 1, None, :], axis=-1)
        np.testing.assert_allclose(original_dist, placed_dist, atol=2e-15)
        np.testing.assert_array_equal(before, self.pair)
        with self.assertRaises(ValueError): shared_place_pair(self.pair, translation=np.zeros((2, 3)))

    def test_diagnostics_expose_root_and_pose_and_mixed_fps_velocity(self):
        left = self.pair.copy(); right = self.pair.copy()
        left[-2] -= [0., 0., .01]; right += [1., 0., 0.]; right[1] += [0., 0., .02]
        right[:, :, 20, 1] += .1
        report = boundary_diagnostics(left, right, left_fps=20, right_fps=30)
        np.testing.assert_allclose(report['root_gap_m'], [1, 1])
        self.assertAlmostEqual(report['root_relative_pose_gap_m']['max'], .1)
        self.assertAlmostEqual(report['velocity_gap_m_s']['max'], .4)
        self.assertFalse(report['seamless_verified'])

    def test_authored_bridge_preserves_inputs_and_matches_discrete_velocity(self):
        left = self.pair.copy(); right = self.pair + [0, 0, .2]
        left[-2] -= [0, 0, .01]; right[1] += [0, 0, .02]
        saved = [left.copy(), right.copy()]
        bridge, report = authored_boundary_bridge(left, right, left_fps=20, right_fps=30)
        np.testing.assert_array_equal(left, saved[0]); np.testing.assert_array_equal(right, saved[1])
        np.testing.assert_allclose((bridge[0] - left[-1])*30, (left[-1]-left[-2])*20, atol=1e-14)
        np.testing.assert_allclose((right[0]-bridge[-1])*30, (right[1]-right[0])*30, atol=1e-14)
        self.assertTrue(report['mechanical_gate_passed'])
        self.assertFalse(report['model_generated'])
        self.assertFalse(report['source_pair_frames_modified'])
        self.assertEqual(report['visual_acceptance'], 'unverified')

    def test_impossible_or_distorted_bridge_is_explicitly_rejected(self):
        _, report = authored_boundary_bridge(self.pair, self.pair + [20, 0, 0], left_fps=20, right_fps=30)
        self.assertFalse(report['mechanical_gate_passed'])
        self.assertIn('speed', report['rejection_reasons'][0])
        other = self.pair.copy(); other[:, :, [18, 20]] = other[:, :, [20, 18]]
        _, report = authored_boundary_bridge(self.pair, other, left_fps=20, right_fps=30)
        self.assertFalse(report['mechanical_gate_passed'])
        self.assertTrue(any('segment' in s or 'proportions' in s for s in report['rejection_reasons']))

    def test_core_only_morphology_uses_actor_specific_source_lengths(self):
        core = np.zeros((8, 2, 27, 3))
        core[:, :, CORE_TO_NATIVE] = self.pair * 1.2
        target = self.pair.copy(); target[:, 1] *= .8
        original = core.copy(); source = target.copy()
        out, report = core_to_pair_anatomy(core, target)
        for joint, parent in enumerate(PARENTS[1:], 1):
            np.testing.assert_allclose(np.linalg.norm(out[:, :, joint]-out[:, :, parent], axis=-1),
                                       np.linalg.norm(target[:, :, joint]-target[:, :, parent], axis=-1))
        np.testing.assert_array_equal(out[:, :, 0][:, :, [0, 2]], original[:, :, 0][:, :, [0, 2]])
        feet = [7, 8, 10, 11]
        np.testing.assert_allclose(out[:, :, feet, 1].min(-1), core27_to_native22(core)[:, :, feet, 1].min(-1))
        np.testing.assert_array_equal(core, original); np.testing.assert_array_equal(target, source)
        self.assertFalse(report['foot_lock'])
        self.assertGreater(report['core_endpoint_displacement_m']['max'], .1)

    def test_direction_bridge_does_not_shorten_bones_while_turning(self):
        left = self.pair.copy()
        right = shared_place_pair(left, yaw=.6)
        bridge, report = authored_direction_bridge(left, right, left_fps=20, right_fps=30, frames=30)
        for joint, parent in enumerate(PARENTS[1:], 1):
            expected = np.linalg.norm(left[0, :, joint]-left[0, :, parent], axis=-1)
            np.testing.assert_allclose(np.linalg.norm(bridge[:, :, joint]-bridge[:, :, parent], axis=-1),
                                       np.broadcast_to(expected, (30, 2)), atol=2e-15)
        self.assertLess(report['max_bone_length_interpolation_error_m'], 1e-14)
        self.assertFalse(report['source_pair_frames_modified'])
        self.assertFalse(report['foot_lock'])
        self.assertEqual(report['visual_acceptance'], 'unverified')

    def test_direction_bridge_exposes_and_gates_sampled_velocity(self):
        right = self.pair + [1, 0, 0]
        _, report = authored_direction_bridge(self.pair, right, left_fps=30, right_fps=30,
                                             frames=3, max_endpoint_velocity_error_m_s=.01)
        self.assertFalse(report['mechanical_gate_passed'])
        self.assertIn('sampled endpoint velocity exceeds tolerance', report['rejection_reasons'])

    def test_composition_preserves_pair_and_discloses_every_segment(self):
        from native_pair_clip import NativePairClip
        core = np.zeros((8, 2, 27, 3))
        core[:, :, CORE_TO_NATIVE] = self.pair
        combined, metadata = compose_ardy_pair_context(core, self.pair, core)
        self.assertEqual([s['source'] for s in metadata['segments']],
                         ['ardy_core', 'authored_transition', 'intergen', 'authored_transition', 'ardy_core'])
        middle = metadata['segments'][2]
        np.testing.assert_array_equal(combined[middle['start_frame']:middle['end_frame_exclusive']], self.pair)
        clip = NativePairClip(combined, metadata=metadata)
        self.assertIsNone(clip.features)
        self.assertFalse(metadata['animation_accepted'])
        self.assertEqual(clip.segments[1]['source'], 'authored_transition')

    def test_builder_is_bounded_uses_only_real_core_history_and_retains_raw(self):
        from native_pair_clip import NativePairClip
        from paired_scene import EMPTY_SCENE
        core = np.zeros((2, 40, 27, 3))
        core[:, :, CORE_TO_NATIVE] = self.pair[0, :, None]
        class Client:
            def __init__(self): self.calls = []
            def wait(self, request, **kwargs):
                self.calls.append(request)
                return [SimpleNamespace(positions=core.copy(), rotations=np.broadcast_to(np.eye(3), (2, 40, 27, 3, 3)).copy(),
                        native_features=np.full((2, 40, 330), len(self.calls), dtype=float))]
        client = Client(); saved = []
        source = NativePairClip(self.pair, metadata={'model': 'InterGen', 'fps': 30})
        result = build_ardy_pair_context(source, client, EMPTY_SCENE,
                                         on_core_chunk=lambda *args: saved.append(args))
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(len(saved), 4)
        self.assertNotIn('history', client.calls[0]); self.assertNotIn('history', client.calls[2])
        np.testing.assert_array_equal(client.calls[1]['history']['native_features'], np.ones((2, 40, 330)))
        np.testing.assert_array_equal(client.calls[3]['history']['native_features'], np.full((2, 40, 330), 3))
        self.assertEqual(result['raw_core']['approach']['positions'].shape, (2, 80, 27, 3))
        self.assertIs(result['source_pair'], source)
        self.assertFalse(result['metadata']['animation_accepted'])
        stop = Client()
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            build_ardy_pair_context(source, stop, EMPTY_SCENE, cancelled=lambda: True)
        self.assertEqual(stop.calls, [])

    def test_selective_footplant_preserves_native_source_root_and_segment_edges(self):
        p = np.repeat(self.pair[:1], 40, axis=0)
        p[:, :, [4, 5], 2] += .08  # Reachable bent knees.
        p[..., 0] += np.arange(40)[:, None, None]*.001
        segments = [{'label': 'Core', 'source': 'ardy_core', 'kind': 'approach',
                     'start_frame': 0, 'end_frame_exclusive': 20, 'frames': 20},
                    {'label': 'native', 'source': 'intergen', 'kind': 'paired_action',
                     'start_frame': 20, 'end_frame_exclusive': 40, 'frames': 20}]
        out, report = stabilize_authored_feet(p, segments)
        np.testing.assert_array_equal(out[20:], p[20:])
        np.testing.assert_array_equal(out[:, :, 0], p[:, :, 0])
        np.testing.assert_array_equal(out[[0, 1, 18, 19]], p[[0, 1, 18, 19]])
        self.assertGreater(report['applied_foot_frames'], 0)
        self.assertLess(report['max_bone_length_error_m'], 1e-14)
        self.assertLess(np.linalg.norm(out[10, 0, 7]-out[9, 0, 7]), 1e-14)
        self.assertGreater(np.linalg.norm(p[10, 0, 7]-p[9, 0, 7]), .0009)

    def test_unreachable_plant_rejects_whole_interval_without_flickering(self):
        p = np.repeat(self.pair[:1], 20, axis=0)
        p[..., 0] += np.arange(20)[:, None, None]*.001
        segments = [{'label': 'Core', 'source': 'ardy_core', 'kind': 'approach',
                     'start_frame': 0, 'end_frame_exclusive': 20, 'frames': 20}]
        out, report = stabilize_authored_feet(p, segments)
        self.assertEqual(report['applied_foot_frames'], 0)
        self.assertGreater(report['unreachable_foot_frames_skipped'], 0)
        np.testing.assert_array_equal(out, p)


if __name__ == '__main__': unittest.main()
