"""Geometry, contact and streaming continuity checks for the swing pose layer."""
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from swing_pose import PoseController


class SwingPoseTests(unittest.TestCase):
    def setUp(self):
        self.pose=PoseController()

    def check_geometry(self,d):
        p,r=d['positions'],d['rotations']
        expected=np.linalg.norm(self.pose.offsets[1:],axis=-1)*self.pose.scales[:,None]
        np.testing.assert_allclose(np.linalg.norm(p[:,1:]-p[:,self.pose.parents[1:]],axis=-1),expected,atol=1e-7)
        np.testing.assert_allclose(r.swapaxes(-1,-2)@r,np.broadcast_to(np.eye(3),r.shape),atol=1e-7)
        np.testing.assert_allclose(np.linalg.det(r),1.,atol=1e-7)
        self.assertTrue(np.isfinite(p).all())

    def test_fixed_bones_and_stable_pair_grips_under_turning(self):
        for yaw in np.linspace(-np.pi,np.pi,9):
            rot=Rotation.from_euler('y',yaw).as_matrix()
            root=np.array([2,5,-3.])
            d=self.pose.get_pose(dict(root=root,yaw=yaw,velocity=rot@[3,0,8],phase='swing',
                carry_amount=1,flight_amount=1,mj_root=root+rot@[0,.22,-.29]),2.)
            self.check_geometry(d)
            for key,value in d['grip_metrics'].items():
                if key.endswith('_m'):
                    self.assertLess(value,.005,key)

    def test_transitions_and_face_contact(self):
        previous=None
        for amount in np.linspace(0,1,101):
            d=self.pose.get_pose(dict(root=[0,.96,0],phase='kiss',carry_amount=0,
                kiss_amount=amount,flight_amount=0),amount)
            self.check_geometry(d)
            if previous is not None:
                self.assertLess(np.linalg.norm(d['positions']-previous,axis=-1).max(),.025)
            previous=d['positions']
        face=d['face_factors']
        self.assertLess(np.linalg.norm(np.diff(face['face_centres'],axis=0)),.04)
        self.assertGreater(face['head_distance_m'],.30)
        self.assertLess(face['head_distance_m'],.34)

    def test_actual_skinned_heads_have_separating_plane_through_ending(self):
        skin_path=Path(__file__).parent/'vendor/ardy/ardy/assets/skeletons/cskel27/skin_standard.npz'
        with np.load(skin_path,allow_pickle=False) as skin:
            vertices=skin['bind_vertices'];indices=skin['lbs_indices'];weights=skin['lbs_weights']
            inverse=np.linalg.inv(skin['bind_rig_transform'])
        # Select the actual head surface by its skinning influence, not an
        # assumed sphere. Apply the same scaled five-weight LBS as the viewer.
        head_weight=np.sum(weights*(indices==6),axis=1)
        selected=head_weight>.5
        vertices=vertices[selected];indices=indices[selected];weights=weights[selected]
        homogeneous=np.c_[vertices,np.ones(len(vertices))]
        local=np.einsum('vwij,vj->vwi',inverse[indices],homogeneous)[...,:3]
        for amount in np.linspace(0,1,21):
            d=self.pose.get_pose(dict(root=[0,.96,0],phase='kiss',kiss_amount=amount,flight_amount=0),amount)
            mesh=[]
            for actor,scale in enumerate(self.pose.scales):
                transformed=np.einsum('vwij,vwj->vwi',d['rotations'][actor,indices],local*scale)+d['positions'][actor,indices]
                mesh.append(np.sum(transformed*weights[...,None],axis=1))
            axis=d['positions'][1,6]-d['positions'][0,6]
            axis/=np.linalg.norm(axis)
            separation=np.min(mesh[1]@axis)-np.max(mesh[0]@axis)
            self.assertGreater(separation,.01,(amount,separation))

    def test_authoritative_mj_root_and_ground_clearance(self):
        root=np.array([4.,7.,-2.])
        d=self.pose.get_pose(dict(root=[0,.96,0],mj_root=root,phase='ready'),0)
        np.testing.assert_allclose(d['positions'][1,0],root)
        d=self.pose.get_pose(dict(root=[0,.96,0],phase='ready'),0)
        feet=[self.pose.idx[x] for x in ['LeftFoot','RightFoot','LeftToeBase','RightToeBase']]
        self.assertGreaterEqual(d['positions'][:,feet,1].min(),0.)

    def test_native_core_adoption_is_continuous_and_has_measurable_effect(self):
        path=Path(__file__).parent/'assets/swing-motion/swing.npz'
        if not path.exists():
            self.skipTest('Native Core clip unavailable')
        with np.load(path,allow_pickle=False) as data:
            p,r=data['positions'],data['rotations']
        snapshot=dict(root=[0,5,0],phase='swing',carry_amount=1,flight_amount=1)
        before=self.pose.get_pose(snapshot,3.)
        self.pose.set_clip(p,r,{'fps':20,'model':'Core'})
        initial=self.pose.get_pose(snapshot,3.)
        np.testing.assert_allclose(initial['positions'],before['positions'],atol=1e-8)
        later=self.pose.get_pose(snapshot,3.6)
        self.check_geometry(later)
        self.assertGreater(np.linalg.norm(later['positions']-before['positions']),.001)
        self.assertEqual(later['provenance']['model'],'Core')
        self.pose.set_clip(p[::-1],r[::-1],{'fps':20,'model':'Core refresh'})
        refreshed=self.pose.get_pose(snapshot,3.6)
        np.testing.assert_allclose(refreshed['positions'],later['positions'],atol=1e-8)
        # Another result arriving during an unfinished adoption must start at
        # the current blend, not at the preceding clip's full target.
        halfway=self.pose.get_pose(snapshot,3.8)
        self.pose.set_clip(p,r,{'fps':20,'model':'rapid refresh'})
        rapid=self.pose.get_pose(snapshot,3.8)
        np.testing.assert_allclose(rapid['positions'],halfway['positions'],atol=1e-8)

    def test_invalid_model_rotations_rejected(self):
        p=np.zeros((2,27,3)); r=np.zeros((2,27,3,3))
        with self.assertRaises(ValueError):
            self.pose.set_clip(p,r)

    def test_actual_city_pickup_dismount_and_kiss_have_no_pose_pops(self):
        from swing_dynamics import SwingController
        from swing_scene import load_swing_scene
        controller=SwingController(load_swing_scene())
        clip=Path(__file__).parent/'assets/swing-motion/swing.npz'
        pose=PoseController(clip if clip.exists() else None)
        previous=pose.get_pose(controller.snapshot(),0)['positions']
        phase=controller.phase
        boundaries=[]

        def tick():
            nonlocal previous,phase
            snapshot=controller.step()
            result=pose.get_pose(snapshot,snapshot['time'])
            step=np.linalg.norm(result['positions']-previous,axis=-1)
            self.assertLess(step.max(),.20,(snapshot['frame'],phase,snapshot['phase']))
            if phase!=snapshot['phase']:
                boundaries.append((phase,snapshot['phase']))
                self.assertLess(step.max(),.15,(snapshot['frame'],phase,snapshot['phase']))
            if result['grip_metrics']['active']:
                for key,value in result['grip_metrics'].items():
                    if key.endswith('_m'):
                        self.assertLess(value,.04,key)
            previous=result['positions'];phase=snapshot['phase']
            return result

        controller.command('swing')
        for _ in range(180):tick()
        controller.command('carry')
        for _ in range(1800):
            tick()
            if controller.carrying:break
        self.assertTrue(controller.carrying)
        for _ in range(120):tick()
        controller.command('land')
        for _ in range(1800):
            tick()
            if controller.phase=='landed':break
        self.assertEqual(controller.phase,'landed')
        controller.command('kiss')
        for _ in range(100):result=tick()
        self.assertEqual(controller.phase,'kiss')
        self.assertIn(('pickup','swing'),boundaries)
        self.assertIn(('landing','settling'),boundaries)
        faces=result['face_factors']['face_centres']
        self.assertLess(np.linalg.norm(faces[0]-faces[1]),.04)

    def test_requesting_pickup_after_direction_change_preserves_mj_heading(self):
        from swing_dynamics import SwingController
        from swing_scene import load_swing_scene
        controller=SwingController(load_swing_scene())
        controller.command('right')
        controller.advance(3)
        before=controller.snapshot()
        previous=self.pose.get_pose(before,before['time'])['positions']
        controller.command('carry')
        after=controller.step()
        current=self.pose.get_pose(after,after['time'])['positions']
        self.assertLess(np.linalg.norm(current-previous,axis=-1).max(),.15)
        self.assertLess(abs(after['mj_yaw']-before['mj_yaw']),.05)


if __name__=='__main__':
    unittest.main()
