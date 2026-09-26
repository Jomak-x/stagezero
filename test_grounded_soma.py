"""Named SOMA motion mapping and uniform SO(3) resampling checks."""
import unittest
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from grounded_soma import DIRECT_MAP, soma77_to_core27, convert_file
from motion_bridge import _layout, _fk_numpy


class SomaAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core_names,_,_=_layout()
        from ardy.skeleton import SOMASkeleton77
        sk=SOMASkeleton77()
        cls.skeleton={'names':list(sk.bone_order_names),'parents':sk.joint_parents.numpy().tolist(),
            'rest':sk.neutral_joints.numpy().tolist()}
        cls.names=cls.skeleton['names'];cls.parents=np.array(cls.skeleton['parents'])
        rest=np.array(cls.skeleton['rest']);cls.offsets=rest-rest[np.maximum(cls.parents,0)]

    def motion(self):
        # Synthetic fixture only: deterministic articulation verifies geometry.
        frames=120
        local=np.broadcast_to(np.eye(3),(2,frames,77,3,3)).copy()
        rotvec=np.zeros((frames,3));rotvec[:,2]=np.linspace(0,1.4,frames)
        local[0,:,self.names.index('LeftArm')]=Rotation.from_rotvec(rotvec).as_matrix()
        local[1,:,self.names.index('RightForeArm')]=Rotation.from_rotvec(-rotvec).as_matrix()
        roots=np.zeros((2,frames,3));roots[:,:,1]=1.;roots[:,:,0]=np.linspace(0,1,frames)
        roots[1,:,2]=2.
        return _fk_numpy(local,roots,self.parents,self.offsets)

    def test_named_endpoints_and_world_motion_are_unchanged_at_coincident_times(self):
        p,r=self.motion()
        out,rot,meta=soma77_to_core27(p,r,self.skeleton)
        self.assertEqual(out.shape,(2,80,27,3))
        for target,source in DIRECT_MAP.items():
            np.testing.assert_allclose(out[:,::2,self.core_names.index(target)],
                p[:,::3,self.names.index(source)],atol=1e-7)
        np.testing.assert_allclose(rot.swapaxes(-1,-2)@rot,np.broadcast_to(np.eye(3),rot.shape),atol=1e-7)
        np.testing.assert_allclose(np.linalg.det(rot),1.,atol=1e-7)
        self.assertEqual(meta['soma_conversion']['target_duration_seconds'],3.95)

    def test_intermediate_pose_preserves_native_limb_lengths(self):
        p,r=self.motion();out,_,_=soma77_to_core27(p,r,self.skeleton)
        for side in ('Left','Right'):
            for start,end,sstart,send in [('Arm','ForeArm','Arm','ForeArm'),('ForeArm','Hand','ForeArm','Hand'),
                    ('UpLeg','Leg','Leg','Shin'),('Leg','Foot','Shin','Foot')]:
                expected=np.linalg.norm(p[0,0,self.names.index(side+send)]-p[0,0,self.names.index(side+sstart)])
                actual=np.linalg.norm(out[:,:,self.core_names.index(side+end)]-out[:,:,self.core_names.index(side+start)],axis=-1)
                np.testing.assert_allclose(actual,expected,atol=1e-7)

    def test_unverified_source_rotation_convention_rejected(self):
        p,r=self.motion();p[:,:,self.names.index('LeftHand'),0]+=.1
        with self.assertRaisesRegex(ValueError,'Unverified SOMA'):
            soma77_to_core27(p,r,self.skeleton)

    def test_raw_archive_cannot_be_overwritten(self):
        with self.assertRaisesRegex(ValueError,'untouched'):
            convert_file('/tmp/native-soma.npz','/tmp/native-soma.npz','/tmp/skeleton.json')

    def test_actual_kimodo_native_arrays_map_without_modifying_raw_archives(self):
        directory=Path(__file__).parent/'grounded_assets/source-motion'
        checks={'kimodo_guard_dodge':'4baea08c9e77bcba1d95efd0cd078f606a63061cd136794702489845912f631b',
            'kimodo_grounded_dance':'e92039b2b8cbbf0d026e65ac0071d191f93d69feb429e2ed4a4380725a5cabe1',
            'kimodo_hiphop_8s':'7f4436e78988f7b5004cc447509a8e7d744e16e7e2471acbe74497ff149e6da4',
            'kimodo_martial_combo_8s':'7040da737abb041ebf04bcd1d8173284949b36efe0ac66b253976860f1b0482c'}
        self.assertTrue(all((directory/(name+'.npz')).exists() for name in checks),'Bundled native Kimodo fixtures are required')
        for name,digest in checks.items():
            path=directory/(name+'.npz')
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)
            with np.load(path,allow_pickle=False) as data:
                p,r=data['positions'],data['rotations'];metadata=json.loads(str(data['metadata']))
            out,rot,meta=soma77_to_core27(p,r,directory/'soma-skeleton.json',metadata=metadata)
            self.assertEqual(out.shape,(1,int(np.floor((len(p)-1)*20/30))+1,27,3))
            self.assertLess(meta['soma_conversion']['source_fk_error_max_m'],1e-6)
            self.assertEqual(meta['original_source'],'unmodified native Kimodo output')
            self.assertIn('retarget',meta['source'])
            for target in ('Hips','LeftHand','RightHand','LeftFoot','RightFoot','Head'):
                source=DIRECT_MAP[target]
                np.testing.assert_allclose(out[0,::2,self.core_names.index(target)],p[::3,self.names.index(source)],atol=1e-6)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)


if __name__=='__main__':unittest.main()
