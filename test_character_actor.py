"""Geometry and transport checks for generated character skinning."""

from types import SimpleNamespace
from io import BytesIO
import unittest

import msgspec
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh

from character_actor import GeneratedCharacterActor, _BONES, _mesh_from_glb, _weights, _PARENTS, _texture_bytes, _material_properties, _smooth_normals, _surface_arm_masks, _raise_ankle, _shoulder_blend_weights


def _test_glb() -> bytes:
    # One textured upright mesh with UVs. The mesh itself need not be human for
    # transport validation; the fitting weights are tested separately.
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    mesh.vertices[:, 1] *= 2
    uv = np.zeros((len(mesh.vertices), 2), dtype=np.float32)
    uv[:, 0] = np.clip(mesh.vertices[:, 0] + 0.5, 0, 1)
    uv[:, 1] = np.clip(mesh.vertices[:, 1] / 2 + 0.5, 0, 1)
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGB", (8, 8), (120, 80, 40)))
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return mesh.export(file_type="glb")


class _FakeScene:
    def __init__(self):
        self.messages = []
        self._websock_interface = SimpleNamespace(queue_message=self.messages.append)
        self.kwargs = None

    def add_mesh_skinned(self, name, vertices, faces, **kwargs):
        self.kwargs = kwargs
        values = dict(vertices=vertices, faces=faces,
                      skin_indices=np.zeros((len(vertices), 4), np.uint16),
                      skin_weights=kwargs["skin_weights"][:, :4],
                      bone_wxyzs=kwargs["bone_wxyzs"], bone_positions=kwargs["bone_positions"],
                      color=kwargs["color"], wireframe=False, opacity=None, material="standard",
                      flat_shading=False, side=kwargs["side"], cast_shadow=True, receive_shadow=True)
        from viser import _messages
        props = _messages.SkinnedMeshProps(**values)
        bones = [SimpleNamespace(position=p.copy(), wxyz=q.copy()) for p, q in
                 zip(kwargs["bone_positions"], kwargs["bone_wxyzs"])]
        handle = SimpleNamespace(_impl=SimpleNamespace(props=props), bones=bones, visible=True, removed=False)
        handle.remove = lambda: setattr(handle, "removed", True)
        return handle


def test_weights_are_smooth_and_normalized():
    points = np.array([[0, .40, 0], [.13, .22, 0], [.17, .03, 0],
                       [.19, -.10, 0], [.08, -.16, 0], [.08, -.36, 0],
                       [.08, -.49, 0]], dtype=np.float32)
    weights = _weights(points)
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-6)
    assert list(weights.argmax(axis=1)) == [2, 3, 4, 5, 9, 10, 11]
    assert (np.count_nonzero(weights, axis=1) <= 4).all()
    outer_boot = _weights(np.array([[0.19, -0.44, 0]], dtype=np.float32))[0]
    assert outer_boot[3:9].sum() == 0


def test_actor_sends_textured_skin_and_updates_bones():
    names = dict.fromkeys(joint for _, joint, _ in _BONES)
    names.update(pelvis_skel=None)
    names = {name: i for i, name in enumerate(names)}
    neutral = np.zeros((len(names), 3), np.float32)
    skeleton = SimpleNamespace(bone_index=names, nbjoints=len(names), neutral_joints=neutral)
    scene = _FakeScene()
    actor = GeneratedCharacterActor(SimpleNamespace(scene=scene), _test_glb(), "/generated/test", skeleton)
    assert len(actor.handle.bones) == len(_BONES)
    message = scene.messages[-1]
    payload = message.as_serializable_dict()
    assert payload["type"] == "SkinnedMeshMessage"
    assert "uv" in payload["props"] and "texture_png" in payload["props"]
    assert payload["props"]["roughness_factor"] == 1
    assert payload["props"]["metallic_factor"] == 1
    assert payload["props"]["normal_texture_png"] is None
    assert payload["props"]["metallic_roughness_texture_png"] is None
    assert msgspec.msgpack.encode(payload)
    assert scene.kwargs["skin_weights"].shape[1] == len(_BONES)

    positions = np.zeros((len(names), 3), np.float32)
    positions[:, 1] = .75
    rotations = np.broadcast_to(np.eye(3), (len(names), 3, 3)).copy()
    actor.update(positions, rotations)
    np.testing.assert_allclose(actor.handle.bones[0].position[1], actor.rest[0,1])
    elbow = names["left_elbow_skel"]
    before = actor.bone_positions.copy()
    positions[elbow, 0] += 12  # Independent source-joint translations cannot stretch the rig.
    actor.update(positions, rotations)
    np.testing.assert_allclose(actor.bone_positions, before)
    shoulder = names["left_shoulder_yaw_skel"]
    rotations[shoulder] = Rotation.from_euler("z", 60, degrees=True).as_matrix()
    actor.update(positions, rotations)
    assert np.linalg.norm(actor.bone_positions[4]-before[4]) > .10
    actor.visible = False
    assert not actor.handle.visible
    actor.remove()
    assert actor.handle.removed


def test_reject_untextured_mesh():
    data = trimesh.creation.icosphere().export(file_type="glb")
    try:
        _mesh_from_glb(data)
    except ValueError as exc:
        assert "textured GLB" in str(exc)
    else:
        raise AssertionError("untextured mesh accepted")


def test_first_motion_pose_preserves_original_silhouette():
    names = dict.fromkeys(joint for _, joint, _ in _BONES)
    names.update(pelvis_skel=None)
    names = {name: i for i, name in enumerate(names)}
    skeleton = SimpleNamespace(bone_index=names, nbjoints=len(names),
                               neutral_joints=np.zeros((len(names), 3), np.float32))
    actor = GeneratedCharacterActor(SimpleNamespace(scene=_FakeScene()), _test_glb(), "/generated/bind", skeleton)
    positions = np.zeros((len(names), 3), np.float32)
    positions[:, 1] = .75
    rotations = np.broadcast_to(np.eye(3), (len(names), 3, 3)).copy()
    root_rotation = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    rotations[:] = root_rotation
    rotations[names["left_wrist_yaw_skel"]] = Rotation.from_euler("xyz", [40, 65, 10], degrees=True).as_matrix()
    rotations[names["right_ankle_roll_skel"]] = Rotation.from_euler("xyz", [-20, 25, 90], degrees=True).as_matrix()
    actor.update(positions, rotations)
    root_quat = Rotation.from_matrix(root_rotation).as_quat(scalar_first=True)
    for bone in actor.handle.bones:
        np.testing.assert_allclose(bone.wxyz, root_quat, atol=1e-6)
    expected = (actor.vertices-actor.rest[0]) @ root_rotation.T + actor.bone_positions[0]
    np.testing.assert_allclose(actor.deform_vertices(), expected, atol=2e-7)


def test_wave_and_squat_preserve_lengths_and_bounds():
    names = {name:i for i,name in enumerate(dict.fromkeys(j for _,j,_ in _BONES))}
    skeleton = SimpleNamespace(bone_index=names, nbjoints=len(names))
    actor = GeneratedCharacterActor(SimpleNamespace(scene=_FakeScene()), _test_glb(), "/motion", skeleton)
    pos = np.zeros((len(names),3))
    rot = np.broadcast_to(np.eye(3),(len(names),3,3)).copy()
    actor.update(pos,rot)
    np.testing.assert_allclose(actor.deform_vertices(), actor.vertices, atol=1e-7)
    lengths = np.linalg.norm(actor.rest[1:]-actor.rest[_PARENTS[1:]],axis=1)
    for t in np.linspace(0,2*np.pi,41):
        angle = .7*(1-np.cos(t))
        wave = Rotation.from_rotvec([0,0,angle]).as_matrix()
        elbow = Rotation.from_rotvec([0, .5*np.sin(t),0]).as_matrix()
        for name in ("left_shoulder_yaw_skel","left_elbow_skel","left_wrist_yaw_skel"):
            rot[names[name]] = wave if "shoulder" in name else wave @ elbow
        for side in ("left","right"):
            hip = Rotation.from_rotvec([angle*.5,0,0]).as_matrix()
            knee = Rotation.from_rotvec([-angle*.6,0,0]).as_matrix()
            rot[names[f"{side}_hip_yaw_skel"]] = hip
            rot[names[f"{side}_knee_skel"]] = knee
        pos[0,1] = -.15*(1-np.cos(t))
        actor.update(pos,rot)
        posed_lengths = np.linalg.norm(actor.bone_positions[1:]-actor.bone_positions[_PARENTS[1:]],axis=1)
        np.testing.assert_allclose(posed_lengths,lengths,atol=1e-12)
        mesh = actor.deform_vertices()
        assert np.isfinite(mesh).all()
        assert mesh[actor._support_indices,1].min() >= -1e-7
        assert np.max(np.linalg.norm(mesh-actor.bone_positions[0],axis=1)) < 2
        np.testing.assert_allclose(np.linalg.det(actor.bone_matrices),1,atol=1e-12)


def test_texture_and_uv_seam_normals():
    mesh = _mesh_from_glb(_test_glb())
    mesh.visual.material.baseColorTexture = Image.new("RGB",(2048,2048),(30,70,90))
    assert Image.open(BytesIO(_texture_bytes(mesh))).size == (2048,2048)
    vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,0],[0,1,0],[0,0,1]],float)
    normals = _smooth_normals(vertices,np.array([[0,1,2],[3,4,5]]))
    np.testing.assert_allclose(normals[0],normals[3])
    np.testing.assert_allclose(normals[2],normals[4])
    np.testing.assert_allclose(np.linalg.norm(normals,axis=1),1,atol=1e-7)


def test_pbr_material_maps_and_factors_survive_glb_transport():
    mesh = _mesh_from_glb(_test_glb())
    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.new("RGB", (8, 8), (120, 80, 40)),
        normalTexture=Image.new("RGB", (8, 8), (128, 128, 255)),
        metallicRoughnessTexture=Image.new("RGB", (8, 8), (0, 153, 64)),
        baseColorFactor=[200, 100, 50, 255], metallicFactor=.25,
        roughnessFactor=.6,
    )
    data = mesh.export(file_type="glb")
    properties = _material_properties(_mesh_from_glb(data))
    assert properties["metallic_factor"] == .25
    assert properties["roughness_factor"] == .6
    np.testing.assert_allclose(properties["base_color_factor"], [200/255, 100/255, 50/255, 1])
    assert Image.open(BytesIO(properties["normal_texture_png"])).getpixel((0, 0)) == (128, 128, 255)
    assert Image.open(BytesIO(properties["metallic_roughness_texture_png"])).getpixel((0, 0)) == (0, 153, 64)
    names = {name: i for i, name in enumerate(dict.fromkeys(joint for _, joint, _ in _BONES))}
    skeleton = SimpleNamespace(bone_index=names, nbjoints=len(names))
    scene = _FakeScene()
    GeneratedCharacterActor(SimpleNamespace(scene=scene), data, "/pbr", skeleton)
    payload = scene.messages[-1].as_serializable_dict()["props"]
    assert payload["roughness_factor"] == .6
    assert payload["metallic_factor"] == .25
    assert payload["normal_texture_png"] == properties["normal_texture_png"]
    assert payload["metallic_roughness_texture_png"] == properties["metallic_roughness_texture_png"]


def test_root_stays_on_scene_trajectory_and_seeking_is_deterministic():
    names = {name:i for i,name in enumerate(dict.fromkeys(j for _,j,_ in _BONES))}
    skeleton = SimpleNamespace(bone_index=names, nbjoints=len(names))
    pos = np.zeros((len(names),3))
    pos[:, :] = [2.0, .75, -3.0]
    rot = np.broadcast_to(np.eye(3),(len(names),3,3)).copy()
    actor = GeneratedCharacterActor(SimpleNamespace(scene=_FakeScene()), _test_glb(),
                                    "/trajectory", skeleton, bind_positions=pos, bind_rotations=rot)
    target = pos.copy()
    target[0] += [4.0, .2, -2.0]
    actor.update(target,rot)
    np.testing.assert_allclose(actor.bone_positions[0,[0,2]], target[0,[0,2]], atol=1e-12)
    np.testing.assert_allclose(actor.bone_positions[0,1], actor.rest[0,1]+.2*actor.rest[0,1]/.75, atol=1e-12)
    expected = actor.deform_vertices().copy()
    # Timeline scrubbing, repeated frames, and reverse playback must not carry
    # a floor correction or filtering state from the previously viewed pose.
    crouch = pos.copy()
    crouch[0,1] -= .6
    actor.update(crouch,rot)
    actor.update(pos,rot)
    actor.update(target,rot)
    np.testing.assert_array_equal(actor.deform_vertices(), expected)


def test_boot_clearance_bends_one_leg_without_bouncing_pelvis():
    names = {name:i for i,name in enumerate(dict.fromkeys(j for _,j,_ in _BONES))}
    skeleton = SimpleNamespace(bone_index=names, nbjoints=len(names))
    actor = GeneratedCharacterActor(SimpleNamespace(scene=_FakeScene()), _test_glb(), "/boots", skeleton)
    # Rigid sole samples isolate the contact behavior from generated topology.
    actor.vertices = np.array([actor.rest[foot]+[x,-actor.rest[foot,1],z]
                               for foot in (11,14) for x in (-.03,.03) for z in (-.05,.18)])
    actor.weights = np.zeros((8,len(_BONES)))
    actor.weights[:4,11] = 1
    actor.weights[4:,14] = 1
    actor._support_indices = np.arange(8)
    actor._leg_support_indices = (np.arange(4),np.arange(4,8))
    pos = np.zeros((len(names),3))
    rot = np.broadcast_to(np.eye(3),(len(names),3,3)).copy()
    actor.update(pos,rot)
    rot[names['right_hip_yaw_skel']] = Rotation.from_euler('x',-20,degrees=True).as_matrix()
    rot[names['right_knee_skel']] = Rotation.from_euler('x',20,degrees=True).as_matrix()
    actor.update(pos,rot)
    before = actor.bone_positions.copy()
    lengths = np.linalg.norm(actor.rest[1:]-actor.rest[_PARENTS[1:]],axis=1)
    # The right boot points down. The left planted foot and pelvis should stay
    # fixed instead of being raised together to clear the right toe.
    rot[names['right_ankle_roll_skel']] = Rotation.from_euler('x',60,degrees=True).as_matrix()
    actor.update(pos,rot)
    np.testing.assert_allclose(actor.bone_positions[[0,9,10,11]],before[[0,9,10,11]],atol=1e-7)
    assert actor.bone_positions[14,1] > before[14,1]+.04
    np.testing.assert_allclose(np.linalg.norm(actor.bone_positions[1:]-actor.bone_positions[_PARENTS[1:]],axis=1),lengths,atol=1e-12)
    assert actor.deform_vertices()[:,1].min() >= -1e-7
    np.testing.assert_allclose(actor.bone_matrices[14],rot[names['right_ankle_roll_skel']],atol=1e-12)
    pose = actor.bone_positions.copy()
    actor.update(pos,np.broadcast_to(np.eye(3),rot.shape))
    actor.update(pos,rot)
    np.testing.assert_array_equal(actor.bone_positions,pose)
    # With clear soles, the same source rotations must pass through unchanged.
    airborne = pos.copy()
    airborne[0,1] = 1.
    actor.update(airborne,rot)
    np.testing.assert_allclose(actor.bone_matrices,rot[actor.bone_indices],atol=1e-12)


def test_boot_ik_has_stable_straight_knee_pole_and_bounded_flexion():
    knees = []
    for tiny_bend in np.linspace(-.001,.001,21):
        height = np.sqrt(.4**2-tiny_bend**2)
        positions = np.array([[0,2*height,0],[0,height,tiny_bend],[0,0,0.]])
        matrices = np.broadcast_to(np.eye(3),(3,3,3)).copy()
        _raise_ankle(positions,matrices,0,.06)
        knees.append(positions[1].copy())
        assert positions[1,2] > 0
        np.testing.assert_allclose(np.linalg.norm(np.diff(positions,axis=0),axis=1),.4,atol=1e-12)
    assert np.linalg.norm(np.diff(knees,axis=0),axis=1).max() < .001
    # An unreachable high target must not fold through the hip or stretch.
    positions = np.array([[0,.8,0],[0,.4,0],[0,0,0.]])
    matrices = np.broadcast_to(np.eye(3),(3,3,3)).copy()
    _raise_ankle(positions,matrices,0,.79)
    upper, lower = np.diff(positions,axis=0)
    flexion = np.degrees(np.arccos(np.clip((upper@lower)/(.4*.4),-1,1)))
    assert flexion <= 150+1e-7
    np.testing.assert_allclose(np.linalg.norm([upper,lower],axis=1),.4,atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(matrices),1,atol=1e-12)


def test_surface_branches_keep_hands_off_hips():
    # A connected torso/arms surface with a gap below the armpits, including
    # closely spaced hand/hip vertices. Seed labels must travel over the surface.
    vertices, faces = [], []
    xs, ys = np.arange(-.18,.181,.01), np.arange(-.50,.301,.01)
    for y in ys:
        for x in xs:
            vertices.append([x,y,0])
    width = len(xs)
    for row,y in enumerate(ys[:-1]):
        for col,x in enumerate(xs[:-1]):
            middle = abs(x+.005)
            if y < .17 and .10 < middle < .13:
                continue
            if y < -.12 and middle > .10:
                continue
            a = row*width+col
            faces.extend([[a,a+1,a+width],[a+1,a+width+1,a+width]])
    vertices,faces = np.array(vertices),np.array(faces)
    masks = _surface_arm_masks(vertices,faces,[np.zeros(len(vertices))]*2)
    for sign,index in ((1,0),(-1,1)):
        hand = np.argmin(np.linalg.norm(vertices-[sign*.15,-.08,0],axis=1))
        hip = np.argmin(np.linalg.norm(vertices-[sign*.09,-.08,0],axis=1))
        assert masks[index][hand] > .999
        assert masks[index][hip] < .001


def test_raised_shoulder_retains_volume_without_moving_limb_joints():
    names = {name:i for i,name in enumerate(dict.fromkeys(j for _,j,_ in _BONES))}
    skeleton = SimpleNamespace(bone_index=names, nbjoints=len(names))
    actor = GeneratedCharacterActor(SimpleNamespace(scene=_FakeScene()), _test_glb(), "/shoulders", skeleton)
    pos = np.zeros((len(names),3))
    rot = np.broadcast_to(np.eye(3),(len(names),3,3)).copy()
    actor.update(pos,rot)
    # A cross-section perpendicular to a 120-degree shoulder rotation exposes
    # the actual radial loss, which finite coordinates/triangle checks miss.
    angles = np.linspace(0,2*np.pi,32,endpoint=False)
    ring = .06*np.column_stack([np.cos(angles),np.sin(angles),np.zeros(32)])
    actor.vertices = np.tile(ring,(5,1)) + actor.rest[3]
    original = np.zeros((len(actor.vertices),len(_BONES)))
    fraction = np.repeat([0.,.25,.5,.75,1.],32)
    original[:,1], original[:,3] = 1-fraction, fraction
    actor.weights = _shoulder_blend_weights(original)
    actor._support_indices = np.array([],dtype=int)
    actor._leg_support_indices = (actor._support_indices,actor._support_indices)
    rotation = Rotation.from_euler("z",120,degrees=True).as_matrix()
    for joint in ("left_shoulder_yaw_skel","left_elbow_skel","left_wrist_yaw_skel"):
        rot[names[joint]] = rotation
    actor.update(pos,rot)
    radii = np.linalg.norm(actor.deform_vertices()-actor.bone_positions[3],axis=1).reshape(5,32)
    np.testing.assert_allclose(radii[[0,2,4]],.06,atol=1e-9)
    np.testing.assert_allclose(radii[[1,3]],.06*np.cos(np.pi/6),atol=1e-9)
    # Legacy equal blending collapsed the same cross-section to half its radius.
    legacy = ring @ ((np.eye(3)+rotation)/2).T
    np.testing.assert_allclose(np.linalg.norm(legacy,axis=1),.03,atol=1e-9)
    np.testing.assert_allclose(actor.weights.sum(axis=1),1)
    assert (np.count_nonzero(actor.weights,axis=1)<=2).all()
    # Existing limb hierarchy and the GPU transforms retain the same pivots.
    for i in range(1,15):
        parent = _PARENTS[i]
        expected = actor.bone_positions[parent] + actor.bone_matrices[parent]@(actor.rest[i]-actor.rest[parent])
        np.testing.assert_allclose(actor.bone_positions[i],expected,atol=1e-12)
    np.testing.assert_allclose(actor.bone_positions[15],actor.bone_positions[3],atol=1e-12)
    for i, bone in enumerate(actor.handle.bones):
        np.testing.assert_allclose(bone.position,actor.bone_positions[i],atol=1e-7)
        gpu_rotation = Rotation.from_quat(bone.wxyz,scalar_first=True).as_matrix()
        np.testing.assert_allclose(gpu_rotation,actor.bone_matrices[i],atol=1e-7)


def test_shoulder_blends_preserve_other_weights_and_reflection():
    rng = np.random.default_rng(17)
    weights = np.zeros((100,len(_BONES)))
    weights[:,:15] = rng.random((100,15))
    weights /= weights.sum(axis=1,keepdims=True)
    result = _shoulder_blend_weights(weights)
    other = [0,2,4,5,7,8,9,10,11,12,13,14]
    np.testing.assert_array_equal(result[:,other],weights[:,other])
    np.testing.assert_allclose(result.sum(axis=1),1)
    assert result.min() >= 0
    mirror = np.arange(len(_BONES))
    mirror[[3,6,15,16]] = [6,3,16,15]
    np.testing.assert_allclose(_shoulder_blend_weights(weights[:,mirror]),result[:,mirror])



def test_world_elevation_is_not_scaled_by_character_stature():
    names = {name:i for i,name in enumerate(dict.fromkeys(j for _,j,_ in _BONES))}
    skeleton = SimpleNamespace(bone_index=names, nbjoints=len(names))
    pos = np.zeros((len(names),3))
    pos[:,1] = .75
    rot = np.broadcast_to(np.eye(3),(len(names),3,3)).copy()
    actor = GeneratedCharacterActor(SimpleNamespace(scene=_FakeScene()), _test_glb(),
                                    "/elevation", skeleton, bind_positions=pos, bind_rotations=rot)
    original = actor.deform_vertices().copy()
    for elevation in (1., 2., .35):
        actor.update(pos+[0,elevation,0],rot)
        np.testing.assert_allclose(actor.deform_vertices(), original+[0,elevation,0], atol=2e-7)


def test_generated_soles_follow_source_treads_and_airborne_heights():
    names = {name:i for i,name in enumerate(dict.fromkeys(j for _,j,_ in _BONES))}
    names.update(left_toe_base=len(names), right_toe_base=len(names)+1)
    skeleton = SimpleNamespace(bone_index=names, nbjoints=len(names))
    actor = GeneratedCharacterActor(SimpleNamespace(scene=_FakeScene()), _test_glb(), "/stairs", skeleton)
    # Independent boot samples model real soles, including a projecting toe.
    actor.vertices = np.array([actor.rest[foot]+[x,-actor.rest[foot,1],z]
                               for foot in (11,14) for x in (-.03,.03) for z in (-.05,.18)])
    actor.weights = np.zeros((8,len(_BONES)))
    actor.weights[:4,11] = 1
    actor.weights[4:,14] = 1
    actor._support_indices = np.arange(8)
    actor._leg_support_indices = (np.arange(4),np.arange(4,8))
    pos = np.zeros((len(names),3))
    pos[0,1] = .65
    rot = np.broadcast_to(np.eye(3),(len(names),3,3)).copy()
    actor.update(pos,rot)
    lengths = np.linalg.norm(actor.rest[1:]-actor.rest[_PARENTS[1:]],axis=1)
    poses = []
    # Exercise an elevated planted foot, a raised swing foot, tilted boots,
    # descent, and a wholly airborne pose without a floor snap.
    for elevation, swing, tilt in ((1., .18, 25), (1.4,.0,-15), (.6,.12,35), (1.8,.08,10)):
        target = pos+[0,elevation,0]
        target[actor._source_feet[1],1] += swing
        rotations = rot.copy()
        rotations[names['left_hip_yaw_skel']] = Rotation.from_euler('x',-12,degrees=True).as_matrix()
        rotations[names['left_knee_skel']] = Rotation.from_euler('x',18,degrees=True).as_matrix()
        rotations[names['right_hip_yaw_skel']] = Rotation.from_euler('x',-20,degrees=True).as_matrix()
        rotations[names['right_knee_skel']] = Rotation.from_euler('x',25,degrees=True).as_matrix()
        rotations[names['right_ankle_roll_skel']] = Rotation.from_euler('x',tilt,degrees=True).as_matrix()
        actor.update(target,rotations)
        mesh = actor.deform_vertices()
        soles = [mesh[ids,1].min() for ids in actor._leg_support_indices]
        np.testing.assert_allclose(soles,[elevation,elevation+swing],atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(actor.bone_positions[1:]-actor.bone_positions[_PARENTS[1:]],axis=1),lengths,atol=1e-12)
        poses.append((target,rotations,mesh.copy()))
    # Revisit in reverse order: contact placement must be pose-local.
    for target,rotations,expected in reversed(poses):
        actor.update(target,rotations)
        np.testing.assert_array_equal(actor.deform_vertices(),expected)


class CharacterActorTests(unittest.TestCase):
    def test_world_elevation(self):
        test_world_elevation_is_not_scaled_by_character_stature()

    def test_source_sole_stair_contacts(self):
        test_generated_soles_follow_source_treads_and_airborne_heights()

    def test_shoulder_volume_and_pivots(self):
        test_raised_shoulder_retains_volume_without_moving_limb_joints()

    def test_shoulder_weight_locality(self):
        test_shoulder_blends_preserve_other_weights_and_reflection()

    def test_anatomical_weights(self):
        test_weights_are_smooth_and_normalized()

    def test_textured_skin_transport_and_motion(self):
        test_actor_sends_textured_skin_and_updates_bones()

    def test_missing_texture(self):
        test_reject_untextured_mesh()

    def test_initial_joint_offsets_do_not_twist_body(self):
        test_first_motion_pose_preserves_original_silhouette()

    def test_hierarchical_wave_and_squat(self):
        test_wave_and_squat_preserve_lengths_and_bounds()

    def test_native_texture_and_shared_normals(self):
        test_texture_and_uv_seam_normals()

    def test_geodesic_arm_membership(self):
        test_surface_branches_keep_hands_off_hips()

    def test_scene_trajectory_and_repeatable_seeking(self):
        test_root_stays_on_scene_trajectory_and_seeking_is_deterministic()

    def test_independent_boot_floor_clearance(self):
        test_boot_clearance_bends_one_leg_without_bouncing_pelvis()

    def test_knee_pole_and_flexion_bounds(self):
        test_boot_ik_has_stable_straight_knee_pole_and_bounded_flexion()
