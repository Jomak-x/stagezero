"""CPU geometry tests. Fixture tests use recorded, native model outputs."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from motion_bridge import (SMPL22_NAMES, DIRECT_MAP, _fk_numpy, _layout,
                           _swing, bridge_to_history, core_skeleton,
                           retarget_intergen_pair)

FIXTURES = Path('/Users/jakob/Desktop/Shellhacks/.runtime')


def test_representation():
    from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep
    from ardy.motion_rep.stats import Stats
    rep = ArdyMotionRep(core_skeleton(), fps=20)
    # Non-identity test normalization exercises actual official Stats; this is
    # not a checkpoint/model and is never used by the runtime bridge.
    rep.stats = Stats(load=False)
    rep.stats.register_from_tensors(torch.linspace(-.2,.3,330), torch.linspace(.3,1.5,330))
    return rep


def anatomical_fixture(frames=7):
    """Synthetic neutral geometry used ONLY for deterministic unit tests."""
    names, _, neutral = _layout()
    source = np.zeros((frames,2,22,3))
    for target, original in DIRECT_MAP.items():
        source[:,:,SMPL22_NAMES.index(original)] = neutral[names.index(target)]
    for smpl, target in [('spine1','Spine1'), ('spine2','Spine2')]:
        source[:,:,SMPL22_NAMES.index(smpl)] = neutral[names.index(target)]
    source[...,1] += 1.
    source[:,1,:,0] += 1.5
    source[...,2] += np.linspace(0,.05,frames)[:,None,None]
    return source


class BridgeTests(unittest.TestCase):
    def test_fixed_lengths_proper_rotations_and_exact_shared_root_transform(self):
        source = anatomical_fixture()
        positions, rotations, meta = retarget_intergen_pair(source)
        self.assertEqual(positions.shape, (2,5,27,3))
        names, parents, neutral = _layout()
        lengths = np.linalg.norm(neutral[1:]-neutral[parents[1:]], axis=-1)
        np.testing.assert_allclose(np.linalg.norm(positions[:,:,1:]-positions[:,:,parents[1:]],axis=-1),
                                   np.broadcast_to(lengths,(2,5,26)),atol=2e-7)
        np.testing.assert_allclose(rotations.swapaxes(-1,-2) @ rotations,
                                   np.broadcast_to(np.eye(3),rotations.shape),atol=2e-6)
        np.testing.assert_allclose(np.linalg.det(rotations),1.,atol=2e-6)
        np.testing.assert_allclose(positions[1,:,0]-positions[0,:,0],
                                   np.tile([1.5*meta['common_scale'],0,0],(5,1)),atol=2e-7)

    def test_rigid_pair_equivariance_and_actor_identity(self):
        source = anatomical_fixture()
        p,r,m = retarget_intergen_pair(source)
        yaw = Rotation.from_euler('y', 1.3).as_matrix()
        translation = np.array([3.,0.,-2.])
        pp,rr,mm = retarget_intergen_pair(source @ yaw.T + translation)
        np.testing.assert_allclose(pp, p @ yaw.T + translation*m['common_scale'],atol=2e-6)
        np.testing.assert_allclose(rr, yaw @ r,atol=2e-6)
        swapped,sr,_ = retarget_intergen_pair(source[:,::-1])
        np.testing.assert_allclose(swapped,p[::-1],atol=2e-6)
        np.testing.assert_allclose(sr,r[::-1],atol=2e-6)
        self.assertAlmostEqual(m['common_scale'],mm['common_scale'])

    def test_antiparallel_swing_and_timing(self):
        rot = _swing(np.array([1.,0,0]),np.array([-1.,0,0]))
        np.testing.assert_allclose(rot @ [1,0,0],[-1,0,0],atol=1e-8)
        self.assertAlmostEqual(np.linalg.det(rot),1.)
        p,r,m = retarget_intergen_pair(anatomical_fixture(8))
        self.assertEqual(m['frames'],5)  # final timestamp .2 <= 7/30
        self.assertEqual(p.shape[1],5)

    def test_native_330_roundtrip_and_history_slicing(self):
        p,r,_ = retarget_intergen_pair(anatomical_fixture())
        rep = test_representation()
        model = SimpleNamespace(motion_rep=rep)
        full = bridge_to_history(model,p,r)
        self.assertEqual(full.shape,(2,5,330))
        decoded = rep.inverse(torch.from_numpy(full),is_normalized=True)
        np.testing.assert_allclose(decoded['posed_joints'].numpy(),p,atol=2e-6)
        np.testing.assert_allclose(decoded['global_rot_mats'].numpy(),r,atol=2e-6)
        np.testing.assert_array_equal(bridge_to_history(model,p,r,history_frames=3),full[:,-3:])
        # Features carry both independently moving roots in the SAME frame.
        unnormalized = rep.unnormalize(torch.from_numpy(full)).numpy()
        np.testing.assert_allclose(unnormalized[:,:,:3],p[:,:,0],atol=2e-6)

    def test_rejects_corrupt_input_and_fk_inconsistent_history(self):
        for source in [np.zeros((3,22,3)), np.zeros((4,2,22,3)), np.full((3,2,22,3),np.nan)]:
            with self.assertRaises(ValueError):
                retarget_intergen_pair(source)
        p,r,_ = retarget_intergen_pair(anatomical_fixture())
        model = SimpleNamespace(motion_rep=test_representation())
        p[0,1,10,0] += .1
        with self.assertRaisesRegex(ValueError,'inconsistent'):
            bridge_to_history(model,p,r)
        r[0,1,10,0,0] = 2
        with self.assertRaisesRegex(ValueError,'SO'):
            bridge_to_history(model,p,r)

    @unittest.skipUnless((FIXTURES/'intergen-lab/handshake_seed37.npz').exists(),'recorded InterGen fixture unavailable')
    def test_recorded_intergen_and_explicit_contact_drift(self):
        source = np.load(FIXTURES/'intergen-lab/handshake_seed37.npz')['joints']
        p,r,m = retarget_intergen_pair(source)
        self.assertEqual(p.shape,(2,140,27,3))
        metrics=m['metrics']
        self.assertLess(metrics['bone_length_error']['max_m'],1e-6)
        self.assertLess(metrics['pair_root_vector_max_error_m'],1e-6)
        self.assertGreater(metrics['pair_wrist_distance_drift']['mean_m'],.005)
        self.assertLess(metrics['pair_wrist_distance_drift']['p95_m'],.10)
        f=bridge_to_history(SimpleNamespace(motion_rep=test_representation()),p,r)
        self.assertTrue(np.isfinite(f).all())

    @unittest.skipUnless((FIXTURES/'interaction-review/contact__right_touch__seed105__conditioned.npz').exists(),'recorded Core fixture unavailable')
    def test_official_native_core_pose_identity_alignment(self):
        archive=np.load(FIXTURES/'interaction-review/contact__right_touch__seed105__conditioned.npz')
        p=archive['actor_0_positions'][None]
        r=archive['actor_0_rotations'][None]
        rep=test_representation()
        f=bridge_to_history(SimpleNamespace(motion_rep=rep),p,r)
        decoded=rep.inverse(torch.from_numpy(f),is_normalized=True)
        np.testing.assert_allclose(decoded['posed_joints'].numpy(),p,atol=3e-6)
        np.testing.assert_allclose(decoded['global_rot_mats'].numpy(),r,atol=3e-6)


if __name__ == '__main__':
    unittest.main()
