import unittest
from collections import defaultdict
import torch
from core_terrain_constraints import RootHeightConstraint, root_height_condition_hook

class NativeHeightTests(unittest.TestCase):
    def test_height_only_dictionary(self):
        data,index=defaultdict(list),defaultdict(list)
        RootHeightConstraint([7,39],[1.0,1.4]).update_constraints(data,index)
        self.assertEqual(set(data),{'root_y_pos'})
        self.assertEqual(set(index),{'root_y_pos'})
        self.assertEqual(index['root_y_pos'][0].tolist(),[7,39])

    def test_window_mapping_and_no_ordinary_conditions(self):
        self.assertEqual(root_height_condition_hook(actor={},generated_offset=40,history_length=4,device='cpu'),[])
        actor={'root_targets':[{'frame':39,'root_height':1.},{'frame':47,'root_height':1.2}, {'frame':79,'root_height':1.5}, {'frame':80,'root_height':2.}]}
        result=root_height_condition_hook(actor=actor,generated_offset=40,history_length=4,device='cpu')
        self.assertEqual(result[0].frame_indices.tolist(),[11,43])

    def test_validation_and_crop(self):
        for frames,heights in [([-1],[1]),([1,1],[1,2]),([1],[float('nan')]),([1],[1,2])]:
            with self.assertRaises(ValueError): RootHeightConstraint(frames,heights)
        result=RootHeightConstraint([7,39,47],[1,2,3]).crop_move(40,80)
        self.assertEqual(result.frame_indices.tolist(),[7])
        self.assertEqual(result.root_height.tolist(),[3])

    def test_official_mask_adds_only_root_y(self):
        from ardy.skeleton import CoreSkeleton27
        from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep
        from ardy.constraints import Root2DConstraintSet
        skeleton=CoreSkeleton27().to('cpu')
        rep=ArdyMotionRep(skeleton,20)
        root=Root2DConstraintSet(skeleton,torch.tensor([11]),torch.tensor([[2.,3.]]))
        obs,mask=rep.create_conditions_from_constraints([root,RootHeightConstraint([11],[1.2])],44,False,'cpu')
        self.assertEqual(torch.nonzero(mask).tolist(),[[11,0],[11,1],[11,2]])
        self.assertTrue(torch.allclose(obs[11,:3],torch.tensor([2.,1.2,3.])))
        self.assertEqual(int(mask[:4].sum()),0)

class NativeFrameTests(unittest.TestCase):
    def test_rigid_fk_translation_and_inverse_pair(self):
        from ardy.skeleton import CoreSkeleton27
        from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep
        from core_terrain_constraints import translate_native_y
        rep=ArdyMotionRep(CoreSkeleton27().to('cpu'),20)
        class AffineStats:
            def normalize(self,x): return (x-.37)/1.23
            def unnormalize(self,x): return x*1.23+.37
        rep.stats=AffineStats()
        raw=torch.zeros(2,4,rep.motion_rep_dim)
        raw[...,rep.slice_dict['root_pos']]=torch.tensor([2.,1.,3.])
        raw[...,rep.slice_dict['global_rot_data']]=torch.tensor([1.,0,0,0,1,0]).repeat(rep.nbjoints)
        raw[...,rep.slice_dict['velocities']]=.13
        raw[...,rep.slice_dict['foot_contacts']]=1.
        native=rep.normalize(raw); original=native.clone()
        moved=translate_native_y(rep,native,torch.tensor([2.3,-.8]))
        before=rep.inverse(native,is_normalized=True)
        after=rep.inverse(moved,is_normalized=True)
        expected=before['posed_joints'].clone()
        expected[...,1]+=torch.tensor([2.3,-.8])[:,None,None]
        self.assertTrue(torch.allclose(after['posed_joints'],expected,atol=1e-6))
        self.assertTrue(torch.equal(before['global_rot_mats'],after['global_rot_mats']))
        self.assertTrue(torch.equal(before['foot_contacts'],after['foot_contacts']))
        self.assertTrue(torch.equal(native,original))
        for key in ['velocities','global_rot_data','foot_contacts','global_root_heading']:
            self.assertTrue(torch.equal(moved[...,rep.slice_dict[key]],native[...,rep.slice_dict[key]]))
        restored=translate_native_y(rep,moved,torch.tensor([-2.3,.8]))
        self.assertTrue(torch.allclose(restored,native,atol=5e-7))
        with self.assertRaises(ValueError): translate_native_y(rep,native,[1,2,3])
        with self.assertRaises(ValueError): translate_native_y(rep,native,float('nan'))

if __name__=='__main__': unittest.main()
