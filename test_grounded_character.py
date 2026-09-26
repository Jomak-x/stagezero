"""Native motion preservation and actual generated-mesh skinning checks."""
import base64
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from grounded_character import GroundedCharacter, _DIRECTION_MAP, _PARENTS
from motion_bridge import _layout, _fk_numpy


ASSET=Path(__file__).resolve().parent/'grounded_assets/characters/civilian.glb'


class GroundedCharacterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.character=GroundedCharacter(ASSET)
        cls.names,cls.parents,cls.neutral=_layout()
        cls.offsets=cls.neutral-cls.neutral[np.maximum(cls.parents,0)]

    def source(self,raised=False):
        local=np.broadcast_to(np.eye(3),(27,3,3)).copy()
        if raised:
            local[self.names.index('LeftArm')]=Rotation.from_euler('z',.9).as_matrix()
            local[self.names.index('RightArm')]=Rotation.from_euler('z',-.7).as_matrix()
            local[self.names.index('LeftLeg')]=Rotation.from_euler('x',1.0).as_matrix()
            local[self.names.index('Spine3')]=Rotation.from_euler('x',.20).as_matrix()
        return _fk_numpy(local,np.array([0.,.9544128252334833,0.]),self.parents,self.offsets)

    def test_bind_mesh_and_material_are_preserved_by_identity_skin(self):
        g=self.character
        rendered=g.deform_vertices(g.rest,np.broadcast_to(np.eye(3),(17,3,3)))
        np.testing.assert_allclose(rendered,g.vertices,atol=2e-7)
        np.testing.assert_allclose(g.skin_weights.sum(axis=1),1.,atol=2e-7)
        self.assertEqual(g.skin_indices.shape,(len(g.vertices),4))
        payload=g.rig_payload()
        self.assertEqual(len(payload['uv']),len(payload['vertices']))
        self.assertEqual(base64.b64decode(payload['texture_url'].split(',',1)[1]),g.texture_png)
        self.assertEqual(len(g.faces),56210)
        self.assertEqual(g.mesh_sha256,'37ac5945f6522c5ede44d590049521ff350b414c2087c996ca24a6979453aad7')

    def test_first_native_pose_articulation_is_not_erased_by_calibration(self):
        p,r=self.source(raised=True)
        fitted=self.character.retarget(p,r)
        f,fr=fitted['positions'],fitted['rotations']
        for bone,(child,start,end) in _DIRECTION_MAP.items():
            before=p[self.names.index(end)]-p[self.names.index(start)]
            after=f[child]-f[bone]
            np.testing.assert_allclose(after/np.linalg.norm(after),before/np.linalg.norm(before),atol=2e-7)
        self.assertGreater(f[4,1]-f[3,1],.15) # left elbow immediately raised
        np.testing.assert_allclose(fr.swapaxes(-1,-2)@fr,np.broadcast_to(np.eye(3),fr.shape),atol=1e-7)
        np.testing.assert_allclose(np.linalg.det(fr),1.,atol=1e-7)

    def test_fixed_human_bones_and_world_path_preserved(self):
        g=self.character;p,r=self.source(raised=True)
        a=g.retarget(p,r);shift=np.array([4.,2.,-3.])
        b=g.retarget(p+shift,r)
        np.testing.assert_allclose(b['positions']-a['positions'],np.broadcast_to(shift,(17,3)),atol=1e-7)
        np.testing.assert_allclose(a['positions'][0],p[0]+[0,g.root_height_offset,0],atol=1e-7)
        for j in range(1,17):
            self.assertAlmostEqual(np.linalg.norm(a['positions'][j]-a['positions'][_PARENTS[j]]),
                np.linalg.norm(g.rest[j]-g.rest[_PARENTS[j]]),places=7)
        untouched=g.retarget(p,r,preserve_root_height=True)
        np.testing.assert_allclose(untouched['positions'][0],p[0])

    def test_visible_arm_and_leg_mesh_motion_follows_source(self):
        g=self.character
        a=g.retarget(*self.source());b=g.retarget(*self.source(raised=True))
        av=g.deform_vertices(**a);bv=g.deform_vertices(**b)
        for bones in [(3,4,5),(9,10,11)]:
            mask=g.weights[:,bones].sum(axis=1)>.9
            self.assertGreater(np.linalg.norm(bv[mask]-av[mask],axis=1).max(),.15)
        self.assertTrue(np.isfinite(bv).all())

    def test_clip_shape_actor_identity_and_native_motion(self):
        path=Path(__file__).parent/'grounded_assets/clips/core_martial_combo_8s.npz'
        self.assertTrue(path.exists(),'Bundled native Core fixture is required')
        with np.load(path) as data:p,r=data['positions'][0,:12],data['rotations'][0,:12]
        pair=np.stack([p,p+[1.5,0,0]])
        output=self.character.clip_payload(pair,np.stack([r,r]))
        fitted=np.array(output['fitted_positions'])
        self.assertEqual(fitted.shape,(2,12,17,3))
        np.testing.assert_allclose(fitted[1]-fitted[0],np.broadcast_to([1.5,0,0],(12,17,3)),atol=1e-7)
        # Native timing is untouched: every input frame produces one output.
        self.assertGreater(np.linalg.norm(fitted[0,-1]-fitted[0,0]),.05)

    def test_invalid_native_pose_is_rejected(self):
        p,r=self.source()
        with self.assertRaises(ValueError):self.character.retarget(p,np.zeros_like(r))
        p[2,0]=np.nan
        with self.assertRaises(ValueError):self.character.retarget(p,r)

    def test_native_wrist_target_variant_preserves_contacts_and_bone_lengths(self):
        path=Path(__file__).parent/'grounded_assets/clips/pair_dance.npz'
        self.assertTrue(path.exists(),'Bundled paired dance fixture is required')
        with np.load(path) as data:p,r=data['positions'],data['rotations']
        g=self.character
        out=g.clip_payload(p,r,preserve_wrists=True)
        fitted=np.array(out['fitted_positions']);matrices=np.array(out['fitted_rotations'])
        expected=p[:,:,[16,10]]+[0,g.root_height_offset,0]
        np.testing.assert_allclose(fitted[:,:,[5,8]],expected,atol=1e-7)
        native_gap=np.linalg.norm(p[0,:,16]-p[1,:,10],axis=-1)
        fitted_gap=np.linalg.norm(fitted[0,:,5]-fitted[1,:,8],axis=-1)
        np.testing.assert_allclose(fitted_gap,native_gap,atol=1e-7)
        lengths=np.linalg.norm(g.rest[1:]-g.rest[_PARENTS[1:]],axis=-1)
        np.testing.assert_allclose(np.linalg.norm(fitted[:,:,1:]-fitted[:,:,_PARENTS[1:]],axis=-1),
            np.broadcast_to(lengths,fitted[:,:,1:,0].shape),atol=1e-7)
        np.testing.assert_allclose(matrices.swapaxes(-1,-2)@matrices,np.broadcast_to(np.eye(3),matrices.shape),atol=1e-7)
        self.assertEqual(out['character_provenance']['wrist_target_error_m']['unreachable_fraction'],0.)

    def test_explicit_floor_calibration_is_constant_and_never_removes_airborne_motion(self):
        g=self.character
        p,r=self.source()
        source=np.stack([p,p+[0,.05,0],p,p+[0,.35,0],p+[0,.65,0],p])
        rot=np.broadcast_to(r,(len(source),27,3,3))
        baseline=g.clip_payload(source,rot)
        corrected=g.clip_payload(source,rot,floor_y=-.01)
        before=np.array(baseline['fitted_positions']);after=np.array(corrected['fitted_positions'])
        delta=after-before
        np.testing.assert_allclose(delta,np.broadcast_to(delta[0,0,0],delta.shape),atol=1e-7)
        np.testing.assert_allclose(np.ptp(after,axis=1),np.ptp(before,axis=1),atol=1e-7)
        airborne=g.clip_payload(source+[0,2,0],rot,floor_y=-.01)
        self.assertFalse(airborne['character_provenance']['floor_calibration'][0]['applied'])

    def test_native_continuation_preserves_exact_fitted_prefix_with_frozen_floor(self):
        path=Path(__file__).parent/'grounded_assets/clips/pair_dance.npz'
        self.assertTrue(path.exists(),'Bundled paired dance fixture is required')
        with np.load(path) as data:p,r=data['positions'],data['rotations']
        g=self.character
        initial=g.clip_payload(p[:,:40],r[:,:40],preserve_wrists=True,floor_y=-.01)
        offsets=initial['character_provenance']['floor_offsets']
        continued=g.clip_payload(p,r,preserve_wrists=True,floor_y=-.01,floor_offsets=offsets)
        np.testing.assert_array_equal(np.array(initial['fitted_positions']),np.array(continued['fitted_positions'])[:,:40])
        np.testing.assert_array_equal(np.array(initial['fitted_rotations']),np.array(continued['fitted_rotations'])[:,:40])
        self.assertEqual(continued['character_provenance']['floor_offsets'],offsets)

    def test_vertical_boot_clearance_is_causal_fixed_length_and_leaves_airborne_motion(self):
        g=self.character;p,r=self.source(raised=True)
        source=np.stack([p+[0,-.10,0],p+[0,.6,0],p+[0,-.05,0]])
        rotations=np.broadcast_to(r,(3,27,3,3))
        options=dict(floor_y=-.01,floor_offsets=[0.],preserve_feet=True)
        before=g.clip_payload(source,rotations,floor_y=-.01,floor_offsets=[0.])
        after=g.clip_payload(source,rotations,**options)
        bp=np.array(before['fitted_positions']);ap=np.array(after['fitted_positions']);ar=np.array(after['fitted_rotations'])
        np.testing.assert_array_equal(ap[:,:,0],bp[:,:,0])
        np.testing.assert_array_equal(ap[:,1],bp[:,1])
        np.testing.assert_array_equal(ar[:,1],np.array(before['fitted_rotations'])[:,1])
        for frame in range(3):
            isolated=g.clip_payload(source[frame:frame+1],rotations[frame:frame+1],**options)
            np.testing.assert_array_equal(ap[:,frame:frame+1],isolated['fitted_positions'])
            np.testing.assert_array_equal(ar[:,frame:frame+1],isolated['fitted_rotations'])
        for foot in (11,14):
            np.testing.assert_allclose(ap[:,:,foot][:,:,[0,2]],bp[:,:,foot][:,:,[0,2]],atol=1e-6)
        lengths=np.linalg.norm(g.rest[1:]-g.rest[_PARENTS[1:]],axis=-1)
        np.testing.assert_allclose(np.linalg.norm(ap[:,:,1:]-ap[:,:,_PARENTS[1:]],axis=-1),np.broadcast_to(lengths,ap[:,:,1:,0].shape),atol=1e-7)
        self.assertLess(after['character_provenance']['foot_retarget']['max_unreachable_target_error_m'],1e-6)

    def test_native_world_wrists_exclude_canonical_pelvis_offset(self):
        p,r=self.source(raised=True)
        out=self.character.retarget(p,r,preserve_wrists=True,wrist_target_space='native_world')
        np.testing.assert_allclose(out['positions'][[5,8]],p[[16,10]],atol=1e-7)


if __name__=='__main__':unittest.main()
