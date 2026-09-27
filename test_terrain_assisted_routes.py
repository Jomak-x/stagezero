"""Independent authored-geometry route checks for the explicit rig17 solver."""

import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from grounded_character import GroundedCharacter
from scene_interaction_geometry import SceneInteractionGeometry
from terrain_assisted_motion import _fk as core_fk
from terrain_assisted_rig import LEG_JOINTS, _fk as rig_fk, assist_rig_clip


def authored_floor(width=12., depth=12.):
    box = {'id': 'floor_mesh', 'name': 'Wide stone floor', 'parts': [
        {'shape': 'box', 'position': [0, 0, 0], 'size': [1, 1, 1], 'color': [120, 120, 120]}]}
    floor = {'id': 'stone', 'name': 'Stone floor', 'kind': 'custom', 'asset': 'floor_mesh',
             'position': [1., -.1, -1.], 'size': [width, .2, depth]}
    return SceneInteractionGeometry.from_scene({'assets': [box], 'objects': [floor]},
                                               include_studio_floor=False)


def quarter_circle_native(frames=110):
    """An arc of radius 2 m with a smooth 90-degree heading change."""
    phase = np.clip((np.arange(frames)-15)/70, 0., 1.)
    angles = phase*np.pi/2
    rotations = np.stack([np.broadcast_to(
        Rotation.from_euler('y', float(np.pi-a)).as_matrix(), (27, 3, 3)) for a in angles])
    positions = np.array([core_fk([2*(1-np.cos(a)), .94, 1-2*np.sin(a)], rotations[t])
                          for t, a in enumerate(angles)])
    return positions, rotations


def stationary_pivot_native(turn_frames=40, frames=76):
    phase = np.clip((np.arange(frames)-12)/turn_frames, 0., 1.)
    angles = np.pi-phase*np.pi/2
    rotations = np.stack([np.broadcast_to(
        Rotation.from_euler('y', float(a)).as_matrix(), (27, 3, 3)) for a in angles])
    positions = np.array([core_fk([0., .94, 0.], r) for r in rotations])
    return positions, rotations


class RouteDirectedPivotTests(unittest.TestCase):
    def plan(self, *, initial_heading=90., destination=-165., hold=40,
             heading_assistance=True, walk=True):
        from terrain_assisted_motion import _plan_contacts
        times = np.arange(hold+95)
        # A large reversible pelvis excursion is not evidence of turn intent.
        angles = np.interp(times, [0, max(1, hold//4), hold-1], [-140., -249., -190.])
        native_r = np.array([np.broadcast_to(Rotation.from_euler('y', a, degrees=True).as_matrix(),
                                             (27, 3, 3)) for a in angles])
        heading = np.radians(destination)
        advance = 1.8*np.clip((times-hold)/60., 0., 1.) if walk else np.zeros(len(times))
        native_p = np.array([core_fk([v*np.sin(heading), .94, v*np.cos(heading)], r)
                             for v, r in zip(advance, native_r)])
        saved_p, saved_r = native_p.copy(), native_r.copy()
        initial_r = np.broadcast_to(Rotation.from_euler('y', initial_heading, degrees=True).as_matrix(),
                                    (27, 3, 3)).copy()
        initial_p = core_fk(native_p[0, 0], initial_r)
        result = _plan_contacts(native_p, native_r, 20., initial_p, initial_r,
                                initial_is_continuation=True, heading_assistance=heading_assistance)
        np.testing.assert_array_equal(native_p, saved_p)
        np.testing.assert_array_equal(native_r, saved_r)
        for entries in result[0]:
            np.testing.assert_array_equal(entries[0][3], initial_r[0])
        return result

    def test_explicit_route_intent_replaces_reversible_native_pelvis_excursion(self):
        contacts, yaw, bouts = self.plan()
        start = bouts[0][0]
        hold_yaw = np.unwrap(yaw[:start+1])
        self.assertAlmostEqual(hold_yaw[0], np.pi/2)
        self.assertAlmostEqual(hold_yaw[-1], np.radians(195.))
        self.assertTrue((np.diff(hold_yaw) >= -1e-10).all())
        self.assertLess(np.max(np.diff(hold_yaw)), np.radians(5))
        self.assertGreaterEqual(sum(0 < row[0] < start for side in contacts for row in side), 3)
        with self.assertRaisesRegex(ValueError, 'reverses direction'):
            self.plan(heading_assistance=False)

    def test_route_heading_crosses_wrap_in_shortest_direction(self):
        for start, end, change in [(170., -170., 20.), (-170., 170., -20.)]:
            with self.subTest(start=start):
                _, yaw, bouts = self.plan(initial_heading=start, destination=end)
                turn = np.unwrap(yaw[:bouts[0][0]+1])
                self.assertAlmostEqual(turn[-1]-turn[0], np.radians(change))
                self.assertLess(np.max(abs(np.diff(turn))), np.radians(1.))

    def test_heading_assistance_keeps_timing_and_intent_rejections(self):
        with self.assertRaisesRegex(ValueError, 'more time for alternating lifted contacts'):
            self.plan(hold=8)
        with self.assertRaisesRegex(ValueError, 'ambiguous half-turn'):
            self.plan(initial_heading=0., destination=180.)
        with self.assertRaisesRegex(ValueError, 'reverses direction'):
            self.plan(walk=False)


class AuthoredRouteRigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.character = GroundedCharacter(Path(__file__).parent/'assets/core-characters/civilian.glb')

    def test_missing_actual_mesh_support_returns_rejection_without_retry_error(self):
        from unittest.mock import patch
        native_p, native_r = stationary_pivot_native(frames=6)
        original = self.character._sole_vertices

        def unsupported_vertices(*args):
            # Simulate an actual mesh wholly outside the authored support while
            # leaving the planner's conservative contact footprint measurable.
            return original(*args)+[100., 0., 0.]

        with patch.object(self.character, '_sole_vertices', side_effect=unsupported_vertices):
            _, _, _, report = assist_rig_clip(native_p, native_r, authored_floor(), self.character)
        self.assertIsNone(report['max_actual_mesh_sole_penetration_m'])
        self.assertGreater(report['unsupported_actual_mesh_sole_vertices'], 0)
        self.assertIn('actual sole vertices leave authored support', report['numerical_rejections'])
        self.assertIn('actual mesh sole penetration exceeds8mm', report['numerical_rejections'])
        self.assertFalse(report['swept_clearance_contact_mode'])
        self.assertFalse(report['route_tangent_swing_paths'])

    def test_feasible_descent_also_reserves_vertical_landing(self):
        from terrain_assisted_rig import _clearance_profile

        class EarlyDescendingTread:
            def support_height(self, x, z, y, **kwargs):
                return .20 if z > -.4 else .12

        samples = np.array([[x, -.09, z] for x in (-.05, 0., .05)
                            for z in np.linspace(-.11, .20, 5)])
        pivot = np.array([0., -.09, .197])
        first, last = np.array([0., .204, 0.]), np.array([0., .124, -1.15])
        rotation = Rotation.from_euler('y', np.pi).as_matrix()
        # This lip can be cleared without the former infeasibility fallback;
        # descending still needs an explicit lowering phase before touchdown.
        path, rotations, lift = _clearance_profile(EarlyDescendingTread(), first, last,
                                                   rotation, rotation, samples, pivot, 10)
        np.testing.assert_array_equal(path[[0, -1]], [first, last])
        np.testing.assert_allclose(path[-2, [0, 2]], path[-1, [0, 2]], atol=1e-10)
        np.testing.assert_allclose(rotations[-2], rotations[-1], atol=1e-10)
        self.assertGreater(path[-2, 1], path[-1, 1])
        self.assertLessEqual(lift, .24)

    def test_descent_clears_heel_edge_before_vertical_landing(self):
        from scipy.spatial.transform import Slerp
        from terrain_assisted_rig import _clearance_profile

        class DescendingTread:
            def support_height(self, x, z, y, **kwargs):
                return .20 if z > -.84 else .12

        geometry = DescendingTread()
        samples = np.array([[x, y, z] for x in (-.05, 0., .05)
                            for y in (-.09, -.075) for z in np.linspace(-.11, .20, 5)])
        pivot = np.array([0., -.09, .197])
        first, last = np.array([0., .204, 0.]), np.array([0., .124, -1.15])
        last_r = Rotation.from_euler('y', np.pi).as_matrix()
        first_r = last_r@Rotation.from_euler('x', .225).as_matrix()
        path, rotations, lift = _clearance_profile(geometry, first, last,
                                                  first_r, last_r, samples, pivot, 10)
        np.testing.assert_array_equal(path[[0, -1]], [first, last])
        self.assertLessEqual(lift, .24)
        # The whole rigid envelope must clear the lip between stored poses,
        # including the heel immediately before the lower-tread touchdown.
        ankles = path-np.einsum('tij,j->ti', rotations, pivot)
        # The descent reaches its destination before touchdown, leaving the
        # final sample for vertical lowering rather than a late heel crossing.
        np.testing.assert_allclose(ankles[-2, [0, 2]], ankles[-1, [0, 2]], atol=1e-10)
        for t in range(len(path)-1):
            for fraction in np.linspace(0., 1., 9):
                rotation = Slerp([0., 1.], Rotation.from_matrix(rotations[t:t+2]))([fraction]).as_matrix()[0]
                points = (1-fraction)*ankles[t]+fraction*ankles[t+1]+samples@rotation.T
                self.assertGreaterEqual(min(q[1]-geometry.support_height(*q[[0, 2, 1]])
                                            for q in points), -1e-6)

    def test_smooth_90_degree_turn_on_authored_wide_floor(self):
        geometry = authored_floor()
        native_p, native_r = quarter_circle_native()
        saved_p, saved_r = native_p.copy(), native_r.copy()
        poses, rotations, stance, report = assist_rig_clip(native_p, native_r, geometry, self.character)

        np.testing.assert_array_equal(native_p, saved_p)
        np.testing.assert_array_equal(native_r, saved_r)
        np.testing.assert_array_equal(poses[:, 0, [0, 2]], native_p[:, 0, [0, 2]])
        nonlegs = [j for j in range(17) if j not in LEG_JOINTS]
        expected = [self.character.retarget(p, r, preserve_root_height=False,
                         preserve_wrists=False)['rotations'][nonlegs] for p, r in zip(native_p, native_r)]
        np.testing.assert_array_equal(rotations[:, nonlegs], expected)
        for pose, rotation in zip(poses, rotations):
            np.testing.assert_allclose(pose, rig_fk(pose[0], rotation, self.character.rest), atol=1e-8)
        self.assertTrue(stance[-10:].all())
        self.assertEqual(report['unsupported_actual_mesh_sole_vertices'], 0)
        self.assertEqual(report['unsupported_swept_sole_envelope_samples'], 0)
        self.assertLessEqual(report['max_actual_mesh_sole_penetration_m'], .008)
        self.assertLessEqual(report['max_flat_stance_mesh_vertex_slip_m_s'], .05)
        self.assertEqual(report['numerical_rejections'], [])

    def test_explicit_heading_assistance_preserves_relative_upperbody_and_native(self):
        from grounded_character import _PARENTS
        native_p, native_r = quarter_circle_native()
        # Add native body yaw opposite to travel while retaining local motion.
        yaw = Rotation.from_euler('y', np.radians(110.)).as_matrix()
        for t in range(len(native_p)):
            root = native_p[t, 0].copy()
            native_p[t] = root+(native_p[t]-root)@yaw.T
            native_r[t] = yaw@native_r[t]
        saved_p, saved_r = native_p.copy(), native_r.copy()
        poses, rotations, stance, report = assist_rig_clip(native_p, native_r,
            authored_floor(), self.character, heading_assistance=True)
        np.testing.assert_array_equal(native_p, saved_p)
        np.testing.assert_array_equal(native_r, saved_r)
        np.testing.assert_array_equal(poses[:, 0, [0, 2]], native_p[:, 0, [0, 2]])
        self.assertFalse(report['native_retargeted_upperbody_rotations_preserved'])
        self.assertTrue(report['native_relative_upperbody_rotations_preserved'])
        self.assertGreater(report['max_display_heading_correction_deg'], 90.)
        self.assertEqual(report['numerical_rejections'], [])
        # A committed heading-assisted pose remains exact through a hold;
        # native absolute pelvis yaw must not reset the displayed character.
        hold_p = np.repeat(native_p[-1:], 20, axis=0)
        hold_r = np.repeat(native_r[-1:], 20, axis=0)
        held_p, held_r, _, held_report = assist_rig_clip(hold_p, hold_r,
            authored_floor(), self.character, heading_assistance=True,
            initial_assisted_positions=poses[-1], initial_assisted_rotations=rotations[-1])
        np.testing.assert_allclose(held_p[0], poses[-1], atol=1e-8)
        np.testing.assert_allclose(held_r[:, 0], np.broadcast_to(rotations[-1, 0], held_r[:, 0].shape), atol=1e-8)
        self.assertEqual(held_report['numerical_rejections'], [])
        baseline = np.array([self.character.retarget(p, r, preserve_root_height=False,
            preserve_wrists=False)['rotations'] for p, r in zip(native_p, native_r)])
        for j in range(1, 17):
            if j in LEG_JOINTS:
                continue
            parent = _PARENTS[j]
            np.testing.assert_allclose(rotations[:, parent].transpose(0, 2, 1)@rotations[:, j],
                baseline[:, parent].transpose(0, 2, 1)@baseline[:, j], atol=1e-8)

    def test_actual_mesh_descends_industrial_four_cm_treads(self):
        from switchback_traversal import industrial_switchback_scene
        geometry = SceneInteractionGeometry.from_scene(industrial_switchback_scene(),
                                                       include_studio_floor=False)
        times = np.arange(160)
        z = -7.-5.5*np.clip((times-15)/110, 0., 1.)
        native_r = np.broadcast_to(Rotation.from_euler('y', np.pi).as_matrix(),
                                  (len(times), 27, 3, 3)).copy()
        native_p = np.array([core_fk([10.5, geometry.support_height(10.5, float(value), .2,
                max_step_up=.4, max_drop=.4)+.94, float(value)], native_r[t])
                            for t, value in enumerate(z)])
        poses, rotations, stance, report = assist_rig_clip(native_p, native_r,
                                                           geometry, self.character)
        np.testing.assert_array_equal(poses[:, 0, [0, 2]], native_p[:, 0, [0, 2]])
        self.assertEqual(report['numerical_rejections'], [])
        self.assertLess(report['max_knee_bend_deg'], 110.)
        self.assertLess(report['max_actual_mesh_sole_penetration_m'], .001)
        self.assertLess(report['max_flat_stance_mesh_vertex_slip_m_s'], .001)
        self.assertTrue(stance[-10:].all())

    def test_rotated_descent_repairs_interframe_riser_contact(self):
        from switchback_traversal import industrial_switchback_scene
        scene = industrial_switchback_scene()
        turn = Rotation.from_euler('y', 37., degrees=True).as_matrix()
        offset = np.array([3., 0., -2.])
        for obj in scene['objects']:
            obj['position'] = (turn@obj['position']+offset).tolist()
            if obj['kind'] in ('custom', 'door'):
                obj['yaw'] = obj.get('yaw', 0.)+37.
        geometry = SceneInteractionGeometry.from_scene(scene, include_studio_floor=False)
        times = np.arange(110)
        z = -7.5-3.875*np.clip((times-15)/70., 0., 1.)
        native_r = np.broadcast_to(turn@Rotation.from_euler('y', np.pi).as_matrix(),
                                   (len(times), 27, 3, 3)).copy()
        roots = np.array([turn@[10.5, .94, value]+offset for value in z])
        for root in roots:
            root[1] = geometry.support_height(root[0], root[2], .2,
                                              max_step_up=.4, max_drop=.4)+.94
        native_p = np.array([core_fk(root, r) for root, r in zip(roots, native_r)])
        saved_p, saved_r = native_p.copy(), native_r.copy()
        poses, rotations, stance, report = assist_rig_clip(native_p, native_r, geometry,
                                                          self.character, heading_assistance=True)
        # The inexpensive arc clears stored poses but crosses a riser between
        # them. Actual swept validation must trigger the bounded contact solve.
        self.assertTrue(report['swept_clearance_contact_mode'])
        self.assertEqual(report['numerical_rejections'], [])
        self.assertLessEqual(report['max_swept_sole_envelope_penetration_m'], .008)
        self.assertLessEqual(report['max_local_leg_rotation_frame_step_deg'], 35.)
        self.assertEqual(report['unsupported_swept_sole_envelope_samples'], 0)
        np.testing.assert_array_equal(native_p, saved_p)
        np.testing.assert_array_equal(native_r, saved_r)
        np.testing.assert_array_equal(poses[:, 0, [0, 2]], native_p[:, 0, [0, 2]])

    def test_authored_narrow_support_cannot_sustain_turning_footprint(self):
        native_p, native_r = quarter_circle_native()
        with self.assertRaisesRegex(ValueError, 'supported tread'):
            assist_rig_clip(native_p, native_r, authored_floor(width=.12, depth=.12), self.character)

    def test_stationary_90_degree_pivot_uses_alternating_lifted_contacts(self):
        native_p, native_r = stationary_pivot_native()
        poses, rotations, stance, report = assist_rig_clip(native_p, native_r,
                                                             authored_floor(), self.character)
        self.assertEqual(report['numerical_rejections'], [])
        np.testing.assert_array_equal(poses[:, 0, [0, 2]], native_p[:, 0, [0, 2]])
        self.assertGreaterEqual(sum(a['first'] > 0 for a in report['stance_anchors']), 4)
        self.assertTrue(stance[-10:].all())
        for side, foot in enumerate((11, 14)):
            self.assertTrue((~stance[:, side]).any())
            heading = np.unwrap(np.arctan2(rotations[:, foot, 0, 2], rotations[:, foot, 2, 2]))
            planted_pairs = stance[1:, side] & stance[:-1, side]
            self.assertLess(float(np.max(abs(np.diff(heading)[planted_pairs]))), 1e-7)

    def test_pelvis_turn_transports_inherited_contacts_into_walking_heading(self):
        from terrain_assisted_motion import _plan_contacts
        times = np.arange(121)
        yaw = np.radians(-138.-42.*np.clip(times/15., 0., 1.))
        x = 1.4*np.clip((times-42)/30., 0., 1.)
        native_r = np.stack([np.broadcast_to(
            Rotation.from_euler('y', float(angle)).as_matrix(), (27, 3, 3)) for angle in yaw])
        native_p = np.array([core_fk([value, .94, 0.], native_r[t])
                             for t, value in enumerate(x)])
        initial_r = np.broadcast_to(Rotation.from_euler('y', np.pi/2).as_matrix(),
                                    (27, 3, 3)).copy()
        initial_p = core_fk([0., .94, 0.], initial_r)
        contacts, planning_yaw, _ = _plan_contacts(native_p, native_r, 20.,
            initial_p, initial_r, initial_is_continuation=True)
        self.assertAlmostEqual(float(planning_yaw[0] % (2*np.pi)), np.pi/2)
        self.assertLess(float(np.max(abs(np.diff(np.unwrap(planning_yaw))))), np.radians(8))
        for side in contacts:
            np.testing.assert_array_equal(side[0][3], initial_r[0])
        np.testing.assert_array_equal(native_r[:, 0], np.stack([
            Rotation.from_euler('y', float(angle)).as_matrix() for angle in yaw]))

    def test_stationary_pivot_rejects_insufficient_contact_time(self):
        native_p, native_r = stationary_pivot_native(turn_frames=5)
        with self.assertRaisesRegex(ValueError, 'more time for alternating lifted contacts'):
            assist_rig_clip(native_p, native_r, authored_floor(), self.character)

    def test_pivot_then_walk_preserves_contacts_and_native_route(self):
        times = np.arange(150)
        pivot = np.clip((times-12)/40, 0., 1.)
        yaw = np.pi-pivot*np.pi/2
        advance = 1.4*np.clip((times-85)/35, 0., 1.)
        native_r = np.stack([np.broadcast_to(
            Rotation.from_euler('y', float(angle)).as_matrix(), (27, 3, 3)) for angle in yaw])
        native_p = np.array([core_fk([x, .94, 0.], native_r[t])
                             for t, x in enumerate(advance)])
        poses, rotations, stance, report = assist_rig_clip(native_p, native_r,
                                                             authored_floor(), self.character)
        np.testing.assert_array_equal(poses[:, 0, [0, 2]], native_p[:, 0, [0, 2]])
        self.assertEqual(len(report['moving_intervals']), 1)
        self.assertTrue(stance[65:78].all())
        self.assertTrue(stance[-10:].all())
        self.assertEqual(report['numerical_rejections'], [])

    def test_40_frame_pivot_immediately_enters_walk(self):
        times = np.arange(118)
        yaw = np.pi-np.clip((times-12)/40, 0., 1.)*np.pi/2
        advance = 1.4*np.clip((times-52)/35, 0., 1.)
        native_r = np.stack([np.broadcast_to(
            Rotation.from_euler('y', float(angle)).as_matrix(), (27, 3, 3)) for angle in yaw])
        native_p = np.array([core_fk([x, .94, 0.], native_r[t])
                             for t, x in enumerate(advance)])
        poses, rotations, stance, report = assist_rig_clip(native_p, native_r,
                                                             authored_floor(), self.character)
        np.testing.assert_array_equal(poses[:, 0, [0, 2]], native_p[:, 0, [0, 2]])
        self.assertTrue(stance[-10:].all())
        self.assertEqual(report['numerical_rejections'], [])


if __name__ == '__main__':
    unittest.main()
