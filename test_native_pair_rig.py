"""Authored skin/endpoint geometry checks; no assertion of animation quality."""
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.native_pair_rig import NativeRigActor, NativeRigAsset, NAMES, PARENTS


def fixture_glb():
    # Small authored hierarchy + nontrivial 2-bone weights. No private assets.
    rest = np.array([[0,1,0], [.08,.96,0],[-.08,.96,0],[0,1.1,0],
       [.08,.53,0],[-.08,.53,0],[0,1.2,0],[.08,.1,0],[-.08,.1,0],
       [0,1.3,0],[.08,.03,.13],[-.08,.03,.13],[0,1.48,0],
       [.045,1.4,0],[-.045,1.4,0],[0,1.6,0],[.15,1.4,0],[-.15,1.4,0],
       [.43,1.4,0],[-.43,1.4,0],[.71,1.4,0],[-.71,1.4,0]],dtype=float)
    rest=np.concatenate((rest,[[.74,1.4,.02],[.78,1.4,.02],[.81,1.4,.02],[.84,1.4,.02]]))
    names=NAMES+tuple(f'LeftHandIndex{i}' for i in range(1,5))
    parents=PARENTS+(20,22,23,24)
    vertices = np.array([rest[16],rest[18],rest[20],rest[18]+[0,.04,.02],rest[22],rest[25]],dtype='<f4')
    joints = np.array([[16,0,0,0],[16,18,0,0],[20,0,0,0],[18,0,0,0],[22,0,0,0],[25,0,0,0]],dtype='<u2')
    weights = np.array([[1,0,0,0],[.25,.75,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]],dtype='<f4')
    ibm = np.repeat(np.eye(4)[None],len(rest),axis=0);ibm[:,:3,3]=-rest
    arrays = [(vertices,5126,'VEC3'),(joints,5123,'VEC4'),(weights,5126,'VEC4'),
              (np.array([0,1,3,1,2,3,3,4,5],dtype='<u2'),5123,'SCALAR'),
              (ibm.transpose(0,2,1).reshape(len(rest),16).astype('<f4'),5126,'MAT4'),
              (np.tile(np.array([0,0,1],dtype='<f4'),(len(vertices),1)),5126,'VEC3')]
    binary = bytearray();views=[];accessors=[]
    for array,component,kind in arrays:
        binary.extend(b'\0'*(-len(binary)%4));offset=len(binary);data=array.tobytes();binary.extend(data)
        views.append({'buffer':0,'byteOffset':offset,'byteLength':len(data)})
        accessor={'bufferView':len(views)-1,'componentType':component,'count':len(array),'type':kind}
        if kind=='VEC3':accessor.update(min=array.min(0).tolist(),max=array.max(0).tolist())
        accessors.append(accessor)
    nodes=[]
    for i,(name,parent) in enumerate(zip(names,parents)):
        n={'name':'mixamorig:'+name,'translation':(rest[i]-(rest[parent] if parent>=0 else 0)).tolist()}
        kids=[j for j,p in enumerate(parents) if p==i]
        if kids:n['children']=kids
        nodes.append(n)
    nodes.append({'mesh':0,'skin':0})
    doc={'asset':{'version':'2.0'},'scene':0,'scenes':[{'nodes':[0,len(rest)]}],'nodes':nodes,
         'meshes':[{'primitives':[{'attributes':{'POSITION':0,'NORMAL':5,'JOINTS_0':1,'WEIGHTS_0':2},'indices':3,'material':0}]}],
         'skins':[{'joints':list(range(len(rest))),'inverseBindMatrices':4}],
         'materials':[{'pbrMetallicRoughness':{'baseColorFactor':[1,1,1,1]}}],
         'buffers':[{'byteLength':len(binary)}],'bufferViews':views,'accessors':accessors}
    encoded=json.dumps(doc).encode();encoded+=b' '*(-len(encoded)%4);binary.extend(b'\0'*(-len(binary)%4))
    chunks=struct.pack('<I4s',len(encoded),b'JSON')+encoded+struct.pack('<I4s',len(binary),b'BIN\0')+binary
    return struct.pack('<4sII',b'glTF',2,len(chunks)+12)+chunks,vertices


class Scene:
    def __init__(self):self.handles=[]
    def add_mesh_simple(self,name,vertices,faces,**kwargs):
        handle=SimpleNamespace(vertices=vertices.copy(),visible=True,removed=False)
        handle.remove=lambda:setattr(handle,'removed',True)
        self.handles.append(handle)
        return handle


class NativePairRigTests(unittest.TestCase):
    def setUp(self):
        self.folder=TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        content,self.vertices=fixture_glb();path=Path(self.folder.name)/'authored.glb';path.write_bytes(content)
        self.asset=NativeRigAsset(path);self.scene=Scene()
        self.actor=NativeRigActor(SimpleNamespace(scene=self.scene),'/rig',self.asset,(30,150,220))
        self.addCleanup(self.actor.remove)

    def test_authored_inverse_binds_and_weights_reconstruct_rest_geometry(self):
        self.actor.set_pose(self.asset.rest)
        np.testing.assert_allclose(self.actor.vertices[0],self.vertices,atol=1e-6)
        np.testing.assert_allclose(self.asset.parts[0]['weights'][1],[.25,.75,0,0])
        self.assertFalse(self.actor.provenance['synthetic_skin_weights'])
        self.assertIsNone(self.actor.provenance['asset_url'])

    def test_rigid_world_transform_and_uniform_scale_preserve_source_world(self):
        rotation=Rotation.from_euler('y',.7).as_matrix();translation=np.array([2.,.3,-1.])
        target=self.asset.rest@rotation.T*1.15+translation
        self.actor.set_pose(target)
        np.testing.assert_allclose(self.actor.joint_positions,target,atol=1e-12)
        np.testing.assert_allclose(self.actor.vertices[0],self.vertices@rotation.T*1.15+translation,atol=1e-6)
        self.assertLess(self.actor.metrics['segment_endpoint_error_max_m'],1e-10)

    def test_foreign_proportions_do_not_move_source_elbow_or_wrist(self):
        target=self.asset.rest.copy();target[18]+=[-.08,-.12,.10];target[20]+=[-.17,-.1,.06]
        self.actor.set_pose(target)
        np.testing.assert_array_equal(self.actor.joint_positions,target)
        # Authored wrist vertex has a single hand-bone influence.
        np.testing.assert_allclose(self.actor.vertices[0][2],target[20],atol=1e-6)
        self.assertLess(self.actor.metrics['segment_endpoint_error_max_m'],1e-10)
        self.assertGreater(self.actor.metrics['authored_proportion_change_max_fraction'],.1)

    def test_invalid_pose_does_not_replace_visible_geometry(self):
        self.actor.set_pose(self.asset.rest)
        before=self.actor.vertices[0].copy();metrics=self.actor.metrics
        for bad in (np.zeros((22,3)),np.full((22,3),np.nan),np.zeros((27,3))):
            with self.assertRaises(ValueError):self.actor.set_pose(bad)
            np.testing.assert_array_equal(self.scene.handles[0].vertices,before)
            self.assertEqual(self.actor.metrics,metrics)

    def test_reset_reproduces_bend_history_and_visibility_is_local(self):
        poses=[self.asset.rest.copy() for _ in range(3)]
        poses[1][18]+=[-.06,-.1,.08];poses[2][18]+=[-.03,-.05,.03]
        first=[]
        for pose in poses:self.actor.set_pose(pose);first.append(self.actor.vertices[0].copy())
        self.actor.reset_pose_history()
        for pose,expected in zip(poses,first):
            self.actor.set_pose(pose);np.testing.assert_array_equal(self.actor.vertices[0],expected)
        self.actor.set_visible(False);self.assertFalse(self.scene.handles[0].visible)
        self.actor.visible=True;self.assertTrue(self.scene.handles[0].visible)
        self.actor.remove();self.assertTrue(self.scene.handles[0].removed)
        with self.assertRaises(RuntimeError):self.actor.set_pose(self.asset.rest)


    def test_prepared_random_access_is_independent_of_seek_history(self):
        poses=np.repeat(self.asset.rest[None],8,axis=0)
        poses[:,18,1]-=np.linspace(0,.12,8)
        poses[:,18,2]+=np.linspace(0,.07,8)
        self.actor.prepare_clip(poses)
        self.actor.set_frame(6);expected=self.actor.vertices[0].copy()
        for frame in (1,7,0,3,6):self.actor.set_frame(frame)
        np.testing.assert_array_equal(self.actor.vertices[0],expected)
        self.assertTrue(self.actor.provenance['deterministic_scrubbing'])
        self.assertIsNone(self.actor.provenance['native_rotation_prior_audit'])
        with self.assertRaises(ValueError):self.actor.prepare_clip(np.zeros((8,22,3)))
        self.actor.set_frame(6)
        np.testing.assert_array_equal(self.actor.vertices[0],expected)

    def test_known_native_axial_rotation_moves_surface_without_moving_endpoints(self):
        frames=8
        positions=np.repeat(self.asset.rest[None],frames,axis=0)
        local=np.broadcast_to(np.eye(3),(frames,21,3,3)).copy()
        angles=np.linspace(0,.6,frames)
        local[:,15]=Rotation.from_euler('x',angles[:,None]).as_matrix()  # LeftArm joint16.
        features=np.zeros((frames,262))
        features[:,:66]=positions.reshape(frames,66)
        features[:,132:258]=local[...,:,:2].reshape(frames,126)
        self.actor.prepare_clip(positions,native_features=features)
        self.actor.set_frame(frames-1)
        np.testing.assert_array_equal(self.actor.joint_positions,positions[-1])
        self.assertTrue(self.actor.metrics['native_arm_rotation_prior'])
        expected=Rotation.from_euler('x',.6).apply(self.vertices[3]-self.asset.rest[18])+self.asset.rest[18]
        np.testing.assert_allclose(self.actor.vertices[0][3],expected,atol=1e-6)
        np.testing.assert_allclose(self.actor.vertices[0][2],self.asset.rest[20],atol=1e-6)
        self.assertFalse(self.actor.provenance['native_rotation_prior_audit']['palm_contact_verified'])
        # Passing no features explicitly restores the position-only baseline.
        self.actor.prepare_clip(positions);self.actor.set_frame(frames-1)
        np.testing.assert_allclose(self.actor.vertices[0],self.vertices,atol=1e-6)
        self.assertFalse(self.actor.metrics['native_arm_rotation_prior'])

    def test_native_prior_rejects_features_from_a_different_source(self):
        positions=np.repeat(self.asset.rest[None],8,axis=0)
        features=np.zeros((8,262))
        with self.assertRaisesRegex(ValueError,'disagree'):
            self.actor.prepare_clip(positions,native_features=features)


    def test_authored_finger_curl_uses_actual_pivots_and_keeps_bone_lengths(self):
        warps=self.asset.finger_warps([1.,0.])
        pivots=np.array([node.world_matrix[:3,3] for node in self.asset.nodes])
        transformed=np.einsum('nij,nj->ni',warps[:,:3,:3],pivots)+warps[:,:3,3]
        np.testing.assert_allclose(transformed[22],pivots[22],atol=1e-12)
        np.testing.assert_allclose(transformed[20],pivots[20],atol=1e-12)
        self.assertLess(transformed[25,1],pivots[25,1]-.04)
        for a,b in ((22,23),(23,24),(24,25)):
            self.assertAlmostEqual(np.linalg.norm(transformed[b]-transformed[a]),
                                   np.linalg.norm(pivots[b]-pivots[a]),places=12)
        # Curling about the wrist instead of each authored knuckle fails these
        # pivot and segment-length assertions.

    def test_finger_layer_only_runs_in_explicit_weighted_frames_and_hand(self):
        poses=np.repeat(self.asset.rest[None],8,axis=0)
        self.actor.prepare_clip(poses);self.actor.set_frame(3)
        baseline=self.actor.vertices[0].copy()
        weights=np.zeros((8,2));weights[2:6,0]=[.4,1.,1.,.4]
        self.actor.prepare_clip(poses,hand_pose='handshake',contact_weights=weights)
        self.actor.set_frame(0);np.testing.assert_array_equal(self.actor.vertices[0],baseline)
        self.actor.set_frame(3)
        np.testing.assert_array_equal(self.actor.joint_positions,poses[3])
        np.testing.assert_array_equal(self.actor.vertices[0][:4],baseline[:4])
        self.assertGreater(np.linalg.norm(self.actor.vertices[0][5]-baseline[5]),.04)
        self.assertEqual(self.actor.metrics['authored_finger_weights_left_right'],[1.,0.])
        self.assertFalse(self.actor.provenance['authored_hand_pose']['contact_verified'])
        self.actor.set_frame(7);np.testing.assert_array_equal(self.actor.vertices[0],baseline)
        self.actor.set_frame(3);expected=self.actor.vertices[0].copy()
        self.actor.set_frame(0);self.actor.set_frame(3)
        np.testing.assert_array_equal(self.actor.vertices[0],expected)

    def test_finger_weights_are_validated_before_replacing_prepared_motion(self):
        poses=np.repeat(self.asset.rest[None],8,axis=0)
        self.actor.prepare_clip(poses);self.actor.set_frame(0);before=self.actor.vertices[0].copy()
        for weights in (np.ones(8),np.full((8,2),np.nan),np.full((8,2),1.1),np.full((8,2),-.1)):
            with self.assertRaises(ValueError):
                self.actor.prepare_clip(poses,hand_pose='handshake',contact_weights=weights)
        with self.assertRaisesRegex(ValueError,'explicit'):
            self.actor.prepare_clip(poses,contact_weights=np.zeros((8,2)))
        self.actor.set_frame(0);np.testing.assert_array_equal(self.actor.vertices[0],before)


if __name__=='__main__':unittest.main()
