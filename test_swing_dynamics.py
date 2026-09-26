"""Dynamics validation against the actual checked-in generated city geometry."""
import json
import math
import unittest
from swing_dynamics import SwingController, DT, norm, sub
from swing_scene import load_swing_scene


class SwingDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.scene=load_swing_scene()

    def run_until(self,c,predicate,seconds=25):
        for _ in range(round(seconds/DT)):
            snapshot=c.step()
            if predicate(snapshot): return snapshot
        self.fail(f'State did not resolve: {c.phase} root={c.root} route={c.route}')

    def test_live_sequence_against_actual_city(self):
        now=[100.]
        c=SwingController(self.scene,clock=lambda:now[0])
        positions=[]; mj_positions=[]
        def tick(n):
            for _ in range(n):
                now[0]+=DT; s=c.step(); positions.append(c.root); mj_positions.append(c.mj_root)
                self.assertGreaterEqual(c.clearance(),-1e-7)
            return s
        c.command('swing');tick(70)
        first_direction=c.direction
        c.command('left');tick(45); left_direction=c.direction
        c.command('right');tick(35)
        self.assertGreater(norm(sub(left_direction,first_direction)),.5)
        self.assertLess(norm(sub(c.direction,first_direction)),1e-6)
        c.command('carry')
        for _ in range(1500):
            s=tick(1)
            if c.carrying: break
        self.assertTrue(c.carrying)
        tick(40);c.command('right');tick(55)
        self.assertEqual(c.phase,'swing')
        self.assertTrue(c.carrying)
        c.command('land')
        for _ in range(1800):
            s=tick(1)
            if c.phase=='landed': break
        self.assertEqual(c.phase,'landed')
        c.command('kiss');s=tick(100)
        self.assertEqual(c.phase,'kiss');self.assertAlmostEqual(s['kiss_amount'],1.)
        self.assertLess(max(norm(sub(b,a)) for a,b in zip(positions,positions[1:])),.16)
        self.assertLess(max(norm(sub(b,a)) for a,b in zip(mj_positions,mj_positions[1:])),.16)
        self.assertLess(s['metrics']['max_speed_m_s'],9.01)
        self.assertTrue(all(x['response_latency_ms']<=17 for x in s['commands']))
        json.dumps(s,allow_nan=False)

    def test_half_minute_roaming_then_carry_land_has_no_stall(self):
        c=SwingController(self.scene);c.command('swing')
        roots=[]
        for i in range(1800):
            if i in (140,800): c.command('left')
            if i in (260,1000): c.command('right')
            c.step();roots.append(c.root)
            self.assertGreaterEqual(c.clearance(),-1e-7)
        self.assertLess(max(abs(p[0]) for p in roots),3.)
        self.assertTrue(any(e['event']=='city_boundary_turn' for e in c.events))
        self.assertGreater(max(p[2] for p in roots[-300:])-min(p[2] for p in roots[-300:]),4.)
        c.command('carry'); self.run_until(c,lambda s:s['carrying'])
        c.advance(1.);c.command('right');c.advance(1.)
        c.command('land');self.run_until(c,lambda s:s['phase']=='landed')
        c.command('kiss');c.advance(.8);before=c.snapshot()['kiss_amount']
        c.command('kiss');c.step()
        self.assertGreaterEqual(c.snapshot()['kiss_amount'],before)

    def test_fixed_step_determinism_and_fractional_elapsed(self):
        a=SwingController(self.scene,clock=lambda:0.)
        b=SwingController(self.scene,clock=lambda:0.)
        for c in (a,b): c.command('swing')
        for _ in range(120): a.step()
        for _ in range(40): b.advance(.05)
        self.assertEqual(a.root,b.root);self.assertEqual(a.velocity,b.velocity)
        self.assertEqual(a.frame,b.frame)
        with self.assertRaises(ValueError): a.step(.1)
        with self.assertRaises(ValueError): a.advance(float('nan'))

    def test_command_is_applied_next_tick_with_real_receipt_latency(self):
        now=[1.]
        c=SwingController(self.scene,clock=lambda:now[0]);c.command('left',timestamp=.9)
        self.assertEqual(c.snapshot()['commands'][0]['status'],'queued')
        now[0]=1.008;c.step()
        row=c.snapshot()['commands'][0]
        self.assertEqual(row['applied_frame'],1)
        self.assertAlmostEqual(row['response_latency_ms'],8.)
        self.assertEqual(row['submitted_at'],.9)

    def test_exact_rendered_hand_releases_root_visible_but_occluded_web(self):
        c=SwingController(self.scene);c.root=(0.,2.,0.);c.phase='swing';c.direction=(1.,0.,0.)
        c.buildings=[{'id':'upper-facade','min':[2.,4.2,-.2],'max':[2.5,5.,.2]}]
        blocked={'id':'behind-facade','position':[5.,6.,0.]}
        c.anchors=[blocked];c.anchor=blocked
        self.assertTrue(c.web_clear(c.root,blocked['position']))
        self.assertFalse(c.validate_web((0.,3.,0.)))
        self.assertIsNone(c.snapshot()['anchor'])
        self.assertEqual(c.events[-1]['reason'],'rendered_hand_segment_occluded')
        self.assertEqual(c.snapshot()['metrics']['web_occlusion_releases'],1)
        c.validate_web((0.,3.,0.))
        self.assertEqual(c.snapshot()['metrics']['web_validated_swing_frames'],1)

    def test_exact_hand_occlusion_can_retether_to_visible_source_anchor(self):
        c=SwingController(self.scene);c.root=(0.,2.,0.);c.phase='swing';c.direction=(1.,0.,0.)
        c.buildings=[{'id':'upper-facade','min':[2.,4.2,-.2],'max':[2.5,5.,.2]}]
        blocked={'id':'behind-facade','position':[5.,6.,0.]}
        visible={'id':'visible-facade','position':[5.,6.,5.]}
        c.anchors=[blocked,visible];c.anchor=blocked
        self.assertTrue(c.validate_web((0.,3.,0.)))
        self.assertEqual(c.anchor['id'],'visible-facade')
        self.assertEqual(c.snapshot()['metrics']['web_tether_coverage'],1.)
        self.assertAlmostEqual(c.rope_length,norm(sub(c.root,visible['position'])))

    def test_pickup_after_turn_preserves_mj_heading_continuity(self):
        c=SwingController(self.scene);c.command('swing');c.advance(1.)
        c.command('left');c.advance(1.)
        previous=c.snapshot()['mj_yaw'];c.command('carry');c.step()
        self.assertLessEqual(abs(c.snapshot()['mj_yaw']-previous),2.5*DT+1e-9)

    def test_airborne_kiss_rejected_and_carry_does_not_teleport(self):
        c=SwingController(self.scene);c.command('swing');c.advance(1.)
        c.command('kiss');c.step()
        self.assertEqual(c.commands[-1].status,'rejected')
        before=c.root;c.command('carry');c.step()
        self.assertEqual(c.phase,'pickup');self.assertFalse(c.carrying)
        self.assertLess(norm(sub(c.root,before)),.16)

    def test_sweep_cannot_tunnel_through_thin_facade(self):
        c=SwingController(self.scene)
        c.buildings=[{'id':'thin-facade','min':[1.,0.,-10.],'max':[1.01,12.,10.]}]
        c.root=(0.,3.,0.);c.velocity=(9.,0.,0.)
        c._move_safely((4.,0.,0.))
        self.assertLess(c.root[0],1.)
        self.assertGreaterEqual(c.clearance(),-1e-8)
        self.assertGreater(c.collision_corrections,0)

    def test_carried_actor_footprint_participates_in_collision(self):
        c=SwingController(self.scene);c.root=(0.,3.,0.);c.yaw=0.
        c.buildings=[{'id':'behind','min':[-.2,2.8,-.74],'max':[.2,3.3,-.64]}]
        self.assertGreater(c.clearance(),0.)
        c.carrying=True
        self.assertLess(c.clearance(),0.)

if __name__=='__main__': unittest.main()
