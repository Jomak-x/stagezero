"""Offline presentation invariants; no assertion of visual naturalness."""
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from terrain_assisted_motion import assist_clip, _fk, OFFSETS, PARENTS, LEG_JOINTS


class FlatGeometry:
    def support_height(self, x, z, y, **kwargs):
        return 0.


class AssistedProofTests(unittest.TestCase):
    def test_separate_fk_consistent_candidate_preserves_native_and_upper_body(self):
        rotations = np.broadcast_to(Rotation.from_euler('y', np.pi).as_matrix(), (100, 27, 3, 3)).copy()
        progress = np.clip((np.arange(100)-30)/44, 0., 1.)
        positions = np.array([_fk([0., .94, 1.-2.5*progress[t]], rotations[t]) for t in range(100)])
        before_p, before_r = positions.copy(), rotations.copy()
        output, output_r, stance, report = assist_clip(positions, rotations, FlatGeometry())
        np.testing.assert_array_equal(positions, before_p)
        np.testing.assert_array_equal(rotations, before_r)
        np.testing.assert_array_equal(output[:, 0, [0, 2]], positions[:, 0, [0, 2]])
        untouched = [j for j in range(27) if j not in LEG_JOINTS]
        np.testing.assert_array_equal(output_r[:, untouched], rotations[:, untouched])
        for j in range(1, 27):
            np.testing.assert_allclose(np.linalg.norm(output[:, j]-output[:, PARENTS[j]], axis=1),
                                       np.linalg.norm(OFFSETS[j]), atol=1e-7)
        self.assertLess(report['max_stance_anchor_error_m'], 1e-6)
        self.assertLess(report['max_sole_penetration_m'], 1e-6)
        self.assertLess(report['max_local_leg_rotation_frame_step_deg'], 35.)
        self.assertFalse(report['accepted'])
        self.assertFalse(report['native_cadence_preserved'])

    def test_reject_inconsistent_native_fk(self):
        p = np.zeros((10, 27, 3))
        r = np.broadcast_to(np.eye(3), (10, 27, 3, 3))
        with self.assertRaisesRegex(ValueError, 'neutral Core FK'):
            assist_clip(p, r, FlatGeometry())

    def test_curved_route_keeps_root_xz_and_native_upper_body(self):
        phase = np.clip((np.arange(110)-20)/55, 0., 1.)
        angle = phase*np.pi/2
        rotations = np.stack([np.broadcast_to(Rotation.from_euler('y', float(np.pi-a)).as_matrix(), (27, 3, 3)) for a in angle])
        positions = np.array([_fk([2*(1-np.cos(a)), .94, 1-2*np.sin(a)], rotations[t]) for t, a in enumerate(angle)])
        output, output_r, stance, report = assist_clip(positions, rotations, FlatGeometry())
        np.testing.assert_array_equal(output[:, 0, [0, 2]], positions[:, 0, [0, 2]])
        untouched = [j for j in range(27) if j not in LEG_JOINTS]
        np.testing.assert_array_equal(output_r[:, untouched], rotations[:, untouched])
        self.assertTrue(stance[-10:].all())
        self.assertLess(report['max_stance_anchor_error_m'], 1e-6)
        self.assertLess(report['max_local_leg_rotation_frame_step_deg'], 35.)

    def test_stationary_hold_and_previous_assisted_boundary(self):
        rotations = np.broadcast_to(Rotation.from_euler('y', np.pi).as_matrix(), (30, 27, 3, 3)).copy()
        positions = np.array([_fk([0., .94, 0.], r) for r in rotations])
        first, first_r, _, _ = assist_clip(positions, rotations, FlatGeometry())
        output, output_r, stance, report = assist_clip(positions, rotations, FlatGeometry(),
            initial_assisted_positions=first[-1], initial_assisted_rotations=first_r[-1])
        np.testing.assert_allclose(output[0], first[-1], atol=1e-7)
        np.testing.assert_allclose(output_r[0], first_r[-1], atol=1e-7)
        self.assertTrue(stance.all())
        self.assertLess(report['max_stance_anchor_error_m'], 1e-6)
        self.assertEqual(report['moving_intervals'], [])

    def test_narrow_or_absent_support_rejected(self):
        class NarrowGeometry:
            def support_height(self, x, z, y, **kwargs):
                return 0. if abs(x) < .08 else None
        rotations = np.broadcast_to(np.eye(3), (20, 27, 3, 3)).copy()
        positions = np.array([_fk([0., .94, 0.], r) for r in rotations])
        with self.assertRaisesRegex(ValueError, 'supported flat footprint'):
            assist_clip(positions, rotations, NarrowGeometry())


class ActualRigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from grounded_character import GroundedCharacter
        cls.character = GroundedCharacter(Path(__file__).parent/'assets/core-characters/civilian.glb')

    def test_actual_rig_hold_preserves_native_and_has_stable_mesh_contact(self):
        from terrain_assisted_rig import assist_rig_clip, _fk as rig_fk, LEG_JOINTS as rig_legs
        rotations = np.broadcast_to(Rotation.from_euler('y', np.pi).as_matrix(), (16, 27, 3, 3)).copy()
        positions = np.array([_fk([0., .94, 0.], r) for r in rotations])
        before_p, before_r = positions.copy(), rotations.copy()
        output, output_r, stance, report = assist_rig_clip(positions, rotations, FlatGeometry(), self.character)
        np.testing.assert_array_equal(positions, before_p)
        np.testing.assert_array_equal(rotations, before_r)
        self.assertEqual(output.shape, (16, 17, 3))
        self.assertTrue(stance.all())
        self.assertTrue(report['native_root_xz_preserved'])
        self.assertTrue(report['native_retargeted_upperbody_rotations_preserved'])
        self.assertLess(report['max_actual_mesh_sole_penetration_m'], .001)
        self.assertLess(report['max_flat_stance_mesh_vertex_slip_m_s'], .001)
        for t in range(len(output)):
            np.testing.assert_allclose(output[t], rig_fk(output[t, 0], output_r[t], self.character.rest), atol=1e-8)
        continued, continued_r, _, continued_report = assist_rig_clip(positions, rotations, FlatGeometry(), self.character,
            initial_assisted_positions=output[-1], initial_assisted_rotations=output_r[-1])
        np.testing.assert_allclose(continued[0], output[-1], atol=1e-7)
        np.testing.assert_allclose(continued_r[0], output_r[-1], atol=1e-7)
        self.assertFalse(continued_report['accepted'])

    def test_actual_rig_follows_curved_native_route(self):
        from terrain_assisted_rig import assist_rig_clip
        phase = np.clip((np.arange(95)-15)/50, 0., 1.)
        angles = phase*np.pi/3
        rotations = np.stack([np.broadcast_to(Rotation.from_euler('y', float(np.pi-a)).as_matrix(), (27, 3, 3)) for a in angles])
        positions = np.array([_fk([2*(1-np.cos(a)), .94, 1-2*np.sin(a)], rotations[t]) for t, a in enumerate(angles)])
        output, output_r, stance, report = assist_rig_clip(positions, rotations, FlatGeometry(), self.character)
        self.assertTrue(report['native_root_xz_preserved'])
        self.assertTrue(report['native_retargeted_upperbody_rotations_preserved'])
        self.assertLess(report['max_actual_mesh_sole_penetration_m'], .001)
        self.assertLess(report['max_flat_stance_mesh_vertex_slip_m_s'], .001)
        self.assertTrue(stance[-10:].all())
        self.assertEqual(report['numerical_rejections'], [])

    def test_actual_rig_settles_during_intermediate_hold(self):
        from terrain_assisted_rig import assist_rig_clip
        times = np.arange(135)
        progress = 1.5*np.clip((times-15)/30, 0., 1.)+1.5*np.clip((times-75)/30, 0., 1.)
        rotations = np.broadcast_to(Rotation.from_euler('y', np.pi).as_matrix(), (len(times), 27, 3, 3)).copy()
        positions = np.array([_fk([0., .94, 1-distance], rotations[t]) for t, distance in enumerate(progress)])
        output, output_r, stance, report = assist_rig_clip(positions, rotations, FlatGeometry(), self.character)
        self.assertEqual(len(report['moving_intervals']), 2)
        self.assertTrue(stance[58:68].all())
        self.assertLess(report['max_flat_stance_mesh_vertex_slip_m_s'], .001)
        self.assertEqual(report['numerical_rejections'], [])

    def test_actual_rig_short_approach_keeps_intermediate_contact(self):
        from terrain_assisted_rig import assist_rig_clip
        times = np.arange(80)
        rotations = np.broadcast_to(Rotation.from_euler('y', np.pi).as_matrix(), (80, 27, 3, 3)).copy()
        previous_progress = 1.5*np.clip((times-15)/30, 0., 1.)
        previous_native = np.array([_fk([0., .94, 1-distance], rotations[t]) for t, distance in enumerate(previous_progress)])
        previous, previous_r, _, _ = assist_rig_clip(previous_native, rotations, FlatGeometry(), self.character)
        progress = 1.02*np.clip(times/30, 0., 1.)
        native = np.array([_fk([0., .94, -.5-distance], rotations[t]) for t, distance in enumerate(progress)])
        output, output_r, stance, report = assist_rig_clip(native, rotations, FlatGeometry(), self.character,
            initial_assisted_positions=previous[-1], initial_assisted_rotations=previous_r[-1])
        landings = sorted((row['first'], -row['contact_xyz'][2]) for row in report['stance_anchors'] if row['first'])
        advances = np.diff([point for _, point in landings])
        self.assertLessEqual(float(advances.max()), .52)
        self.assertEqual(report['numerical_rejections'], [])
        self.assertTrue(stance[-10:].all())


if __name__ == '__main__':
    unittest.main()
