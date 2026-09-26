"""Experimental native InterGen22 → authored Xbot GLB skin, without Core27.

Joint coordinates remain the model's coordinates. Authored skin weights and
inverse binds are retained; segment-affine deformation fits limb lengths to the
source anatomy instead of moving source elbows to fit the asset. This is CPU
linear-blend skinning, not learned animation, contact IK, or recovered fingers.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import hashlib

import numpy as np

from character_assets import DEFAULT_LIMITS, _nodes, _parse_glb, _read_accessor, inspect_glb

ASSET_URL = 'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/Xbot.glb'
XBOT_SHA256 = '002f8d269de68e5dce3d25195caf390d1aa359bbfaae3fcf4c8dc78ec36c3ba5'
NAMES = ('Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine', 'LeftLeg', 'RightLeg',
         'Spine1', 'LeftFoot', 'RightFoot', 'Spine2', 'LeftToeBase', 'RightToeBase',
         'Neck', 'LeftShoulder', 'RightShoulder', 'Head', 'LeftArm', 'RightArm',
         'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand')
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
PRIMARY_CHILD = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11,
                 9: 12, 12: 15, 13: 16, 14: 17, 16: 18, 17: 19, 18: 20, 19: 21}
LIMBS = ((16, 18, 20), (17, 19, 21), (1, 4, 7), (2, 5, 8))


def _unit(vector):
    length = np.linalg.norm(vector)
    if length < 1e-8:
        raise ValueError('Native rig requires nonzero anatomical segments')
    return vector / length


def _basis(axis, normal):
    x = _unit(axis)
    z = normal - np.dot(normal, x) * x
    if np.linalg.norm(z) < 1e-7:
        fallback = np.eye(3)[np.argmin(abs(x))]
        z = fallback - np.dot(fallback, x) * x
    z = _unit(z)
    return np.stack((x, np.cross(z, x), z), axis=1)


def _body_basis(joints):
    x = _unit(joints[1] - joints[2])
    y = joints[9] - joints[0]
    y = _unit(y - np.dot(y, x) * x)
    return np.stack((x, y, np.cross(x, y)), axis=1)


class NativeRigAsset:
    """Read the actual authored skin and calculate bounded direct deformations."""
    def __init__(self, path):
        path = Path(path)
        if path.stat().st_size > DEFAULT_LIMITS.max_file_bytes:
            raise ValueError('Rig GLB exceeds the asset size limit')
        content = path.read_bytes()
        inspect_glb(content)
        self.sha256 = hashlib.sha256(content).hexdigest()
        doc, binary = _parse_glb(content, DEFAULT_LIMITS)
        nodes = _nodes(doc, DEFAULT_LIMITS)
        lookup = {node.name.split(':')[-1]: node.index for node in nodes}
        if any(name not in lookup for name in NAMES):
            raise ValueError('Experiment requires all 22 authored Mixamo body bones')
        self.nodes = nodes
        self.mapped_nodes = tuple(lookup[name] for name in NAMES)
        self.rest = np.stack([nodes[index].world_matrix[:3, 3] for index in self.mapped_nodes])
        mapped = {node: index for index, node in enumerate(self.mapped_nodes)}
        # Finger/eye/toe-end bones inherit their nearest mapped authored ancestor.
        self.ancestor = {}
        for node in nodes:
            ancestor = node.index
            while ancestor not in mapped and ancestor is not None:
                ancestor = nodes[ancestor].parent
            if ancestor is not None:
                self.ancestor[node.index] = mapped[ancestor]
        self.finger_controls = {}
        for side_index, side in enumerate(('Left', 'Right')):
            for finger in ('Index', 'Middle', 'Ring', 'Pinky', 'Thumb'):
                angles = (10., 15., 10.) if finger == 'Thumb' else (25., 35., 20.)
                for segment, degrees in enumerate(angles, 1):
                    node = lookup.get(f'{side}Hand{finger}{segment}')
                    if node is not None:
                        # Authored T-pose fingers extend ±X, with palms down.
                        # This is a disclosed manual curl, not source inference.
                        axis = np.array([0., 0., -1. if side == 'Left' else 1.])
                        self.finger_controls[node] = (side_index, axis, np.deg2rad(degrees))
        self.parts = []
        for node in nodes:
            raw = doc['nodes'][node.index]
            if 'mesh' not in raw:
                continue
            if 'skin' not in raw:
                raise ValueError('Experiment requires authored skin on every mesh')
            skin = doc['skins'][raw['skin']]
            joint_nodes = skin['joints']
            if any(index not in self.ancestor for index in joint_nodes):
                raise ValueError('Authored skin has joints outside the mapped body hierarchy')
            binds = _read_accessor(doc, binary, skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
            rest_skin = np.stack([nodes[index].world_matrix @ binds[i] for i, index in enumerate(joint_nodes)])
            ancestors = np.array([self.ancestor[index] for index in joint_nodes])
            for primitive in doc['meshes'][raw['mesh']]['primitives']:
                attrs = primitive['attributes']
                vertices = _read_accessor(doc, binary, attrs['POSITION']).astype(float)
                joints = _read_accessor(doc, binary, attrs['JOINTS_0'], normalize=False).astype(int)
                weights = _read_accessor(doc, binary, attrs['WEIGHTS_0']).astype(float)
                weights /= weights.sum(axis=1, keepdims=True)
                faces = _read_accessor(doc, binary, primitive['indices']).reshape(-1, 3).astype(np.uint32)
                transforms = rest_skin[joints]
                bind_world = np.einsum('vwij,vj->vwi', transforms[:, :, :3, :3], vertices) + transforms[:, :, :3, 3]
                influence_bones = ancestors[joints]
                material = doc.get('materials', [])[primitive.get('material', 0)].get('pbrMetallicRoughness', {})
                self.parts.append({'bind_world': bind_world, 'bones': influence_bones, 'weights': weights,
                                   'skin_nodes': np.asarray(joint_nodes)[joints],
                                   'faces': faces, 'rest_vertices': np.sum(bind_world * weights[..., None], axis=1),
                                   'metallic': float(material.get('metallicFactor', 0)),
                                   'roughness': float(material.get('roughnessFactor', .8))})
        if not self.parts:
            raise ValueError('Authored GLB has no skinned geometry')
        self.provenance = {
            'asset_path': str(path.resolve()), 'asset_sha256': self.sha256,
            'asset_url': ASSET_URL if self.sha256 == XBOT_SHA256 else None,
            'authored_bone_count': sum(len(s['joints']) for s in doc['skins']),
            'vertex_count': sum(len(p['weights']) for p in self.parts),
            'method': 'authored inverse binds and skin weights; exact native22 joint positions; segment-affine source-proportion fitting',
            'core27_used': False, 'synthetic_skin_weights': False,
            'limitations': ['No contact solver or foot locking; native coordinates are retained.',
                            'Limb lengths follow source anatomy including its framewise length variation; authored mesh proportions are intentionally adapted.',
                            'Position-only twist uses source bend planes with a temporal fallback near straight limbs; optional native arm rotation priors are separately audited.',
                            'Fingers default to authored rest pose; an optional manual curl layer is separately disclosed in authored_hand_pose.',
                            'Mesh skin blending can change visible surface contact despite exact joint endpoints.'],
        }

    def finger_warps(self, weights):
        """Authored finger-local rotations about actual GLB pivots, in bind world."""
        from scipy.spatial.transform import Rotation
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (2,) or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
            raise ValueError('Finger contact weights must be two finite values in [0,1]')
        transforms = np.empty((len(self.nodes), 4, 4))
        visited = set()
        def pose(index):
            if index in visited:
                return transforms[index]
            node = self.nodes[index]
            parent = np.eye(4) if node.parent is None else pose(node.parent)
            local = np.eye(4)
            control = self.finger_controls.get(index)
            if control is not None:
                side, axis, angle = control
                rotation = Rotation.from_rotvec(axis * angle * weights[side]).as_matrix()
                pivot = node.world_matrix[:3, 3]
                local[:3, :3] = rotation
                local[:3, 3] = pivot - rotation @ pivot
            transforms[index] = parent @ local
            visited.add(index)
            return transforms[index]
        for index in range(len(self.nodes)):
            pose(index)
        return transforms

    def skin(self, linear, translations, finger_weights=None):
        outputs = []
        finger = None if finger_weights is None or not np.any(finger_weights) else self.finger_warps(finger_weights)
        for part in self.parts:
            ids = part['bones']
            bind_world = part['bind_world']
            if finger is not None:
                warps = finger[part['skin_nodes']]
                bind_world = np.einsum('vwij,vwj->vwi', warps[:, :, :3, :3], bind_world) + warps[:, :, :3, 3]
            local = bind_world - self.rest[ids]
            moved = np.einsum('vwij,vwj->vwi', linear[ids], local) + translations[ids]
            outputs.append(np.sum(moved * part['weights'][..., None], axis=1).astype(np.float32))
        return outputs


class NativeRigActor:
    """CPU-skinned authored mesh driven directly by one native22 pose."""
    def __init__(self, server, prefix, asset, color):
        self.asset = asset if isinstance(asset, NativeRigAsset) else NativeRigAsset(asset)
        self.handles = []
        self._removed = False
        self._visible = True
        self._scale = None
        self._previous_normals = {}
        self._metrics = {}
        self._prepared = None
        self.joint_positions = None
        self.vertices = None
        self.provenance = deepcopy(self.asset.provenance)
        for index, part in enumerate(self.asset.parts):
            handle = server.scene.add_mesh_simple(f'{prefix}/part-{index}', part['rest_vertices'].astype(np.float32),
                part['faces'], color=color, side='double', material='standard',
                flat_shading=False, cast_shadow=True, receive_shadow=True)
            self.handles.append(handle)

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        self.set_visible(value)

    @property
    def metrics(self):
        return deepcopy(self._metrics)

    def set_visible(self, value):
        self._visible = bool(value)
        for handle in self.handles:
            handle.visible = self._visible

    def reset_pose_history(self):
        """Reset first-pose calibration and bend-plane fallback before replay."""
        self._scale = None
        self._previous_normals = {}

    def _solve_pose(self, joints22, rotation_prior=None):
        if self._removed:
            raise RuntimeError('Native rig actor was removed')
        target = np.asarray(joints22, dtype=float)
        if target.shape != (22, 3) or not np.isfinite(target).all():
            raise ValueError('Expected finite native joints22[22,3]')
        rest = self.asset.rest
        ratios = {joint: np.linalg.norm(target[child] - target[joint]) /
                  np.linalg.norm(rest[child] - rest[joint]) for joint, child in PRIMARY_CHILD.items()}
        if min(ratios.values()) < .05 or max(ratios.values()) > 5:
            raise ValueError('Native anatomical proportions are degenerate or outside this experiment')
        scale = self._scale
        if scale is None:
            scale = float(np.median([ratios[j] for limb in LIMBS for j in limb[:2]]))
        body_rotation = _body_basis(target) @ _body_basis(rest).T
        limb_normals = {}
        pending_normals = {}
        for root, elbow, wrist in LIMBS:
            upper, lower = target[elbow] - target[root], target[wrist] - target[elbow]
            normal = np.cross(upper, lower)
            # Rest arms are nearly straight. The character's authored forward
            # direction establishes a deterministic rest bend plane.
            rest_axis = rest[elbow] - rest[root]
            rest_normal = np.cross(rest_axis, [0., 0., 1.])
            if np.linalg.norm(rest_normal) < 1e-7:
                rest_normal = np.array([1., 0., 0.])
            rest_normal = _unit(rest_normal)
            fallback = self._previous_normals.get(root, body_rotation @ rest_normal)
            if np.linalg.norm(normal) / (np.linalg.norm(upper) * np.linalg.norm(lower)) < .05:
                normal = fallback
            else:
                normal = _unit(normal)
                if np.dot(normal, fallback) < 0:
                    normal = -normal
            pending_normals[root] = normal.copy()
            limb_normals[root] = (rest_normal, normal)
            limb_normals[elbow] = (rest_normal, normal)
        linear = np.empty((22, 3, 3))
        rotations = np.empty_like(linear)
        endpoint_errors = []
        for joint in range(22):
            parent = PARENTS[joint]
            reference = body_rotation if parent < 0 else rotations[parent]
            child = PRIMARY_CHILD.get(joint)
            if child is None:
                rotation = rotation_prior[joint] if rotation_prior is not None and joint in (20, 21) else reference
                linear[joint] = rotation * scale
            else:
                before, after = rest[child] - rest[joint], target[child] - target[joint]
                if joint in limb_normals:
                    before_normal, after_normal = limb_normals[joint]
                else:
                    before_normal = np.array([0., 0., 1.])
                    after_normal = reference @ before_normal
                if rotation_prior is not None and joint in (16, 17, 18, 19):
                    # Native feature rotations contribute axial orientation;
                    # a swing correction still aims at exact source endpoints.
                    before_normal = limb_normals[joint][0]
                    after_normal = rotation_prior[joint] @ before_normal
                rotation = _basis(after, after_normal) @ _basis(before, before_normal).T
                axis = _unit(before)
                linear[joint] = rotation @ (scale * np.eye(3) + (ratios[joint] - scale) * np.outer(axis, axis))
                endpoint_errors.append(np.linalg.norm(linear[joint] @ before + target[joint] - target[child]))
            rotations[joint] = rotation
        self._scale = scale
        self._previous_normals = pending_normals
        metrics = {
            'joint_position_error_max_m': 0.,
            'segment_endpoint_error_max_m': float(max(endpoint_errors)),
            'transverse_scale_calibrated_first_pose': scale,
            'segment_length_scale': {NAMES[j]: float(v) for j, v in ratios.items()},
            'authored_proportion_change_max_fraction': float(max(abs(v - 1) for v in ratios.values())),
            'source_elbow_positions_preserved': True,
            'native_arm_rotation_prior': rotation_prior is not None,
            'physical_contact_verified': False,
        }
        return linear, target.copy(), metrics

    def _publish_pose(self, linear, target, metrics, finger_weights=None):
        vertices = self.asset.skin(linear, target, finger_weights)
        if any(not np.isfinite(v).all() for v in vertices):
            raise ValueError('Native rig skinning produced nonfinite vertices')
        for handle, points in zip(self.handles, vertices):
            handle.vertices = points
        self.joint_positions = target.copy()
        self.vertices = vertices
        self._metrics = deepcopy(metrics)
        self._metrics['authored_finger_weights_left_right'] = [0., 0.] if finger_weights is None else np.asarray(finger_weights).tolist()
        self._metrics['authored_finger_pose_active'] = bool(finger_weights is not None and np.any(finger_weights))
        self._metrics['mesh_min_y_m'] = float(min(v[:, 1].min() for v in vertices))
        return self.metrics

    def set_pose(self, joints22):
        """Sequential position-driven pose; use prepare_clip/set_frame for scrubbing."""
        return self._publish_pose(*self._solve_pose(joints22))

    def _native_priors(self, positions, features):
        from scipy.spatial.transform import Rotation
        from experiments.native_rotation_audit import (
            audit_actor, decode_native_6d, globals_from_local, pelvis_frames)
        features = np.asarray(features, dtype=float)
        if features.shape != (len(positions), 262) or not np.isfinite(features).all():
            raise ValueError('Native prior requires finite features[T,262]')
        if not np.allclose(features[:, :66].reshape(-1, 22, 3), positions, atol=1e-6, rtol=0):
            raise ValueError('Native prior feature positions disagree with displayed source')
        local = decode_native_6d(features[:, 132:258].reshape(-1, 21, 6))
        report, _ = audit_actor(positions, local)
        if not report['fit_success'] or not report['eligible_for_native_rotation_skin_experiment']:
            raise ValueError('Native rotation prior failed held-out FK consistency checks; use position-only rig')
        correction = Rotation.from_rotvec(report['basis_correction_rotvec']).as_matrix()
        global_rotations = globals_from_local(local, pelvis_frames(positions) @ correction)
        offsets = np.asarray(report['fixed_rest_offsets_m'])
        # SMPL's inferred rest basis → authored GLB world rest basis. This one
        # constant alignment does not fit palms to one another or adjust roots.
        source_x = _unit(offsets[0] - offsets[1])
        source_y = _unit(offsets[2] - np.dot(offsets[2], source_x) * source_x)
        source_basis = np.stack((source_x, source_y, np.cross(source_x, source_y)), axis=1)
        rest = self.asset.rest
        asset_x = _unit(rest[1] - rest[2])
        asset_y = rest[3] - rest[0]
        asset_y = _unit(asset_y - np.dot(asset_y, asset_x) * asset_x)
        asset_basis = np.stack((asset_x, asset_y, np.cross(asset_x, asset_y)), axis=1)
        prior = global_rotations @ (source_basis @ asset_basis.T)
        audit = {key: report[key] for key in ('root_method', 'basis_correction_rotvec',
                 'fit_success', 'held_out_fk_joint_error_m', 'held_out_fk_wrist_error_m',
                 'eligible_for_native_rotation_skin_experiment')}
        audit['usage'] = 'arm axial twist and hand orientation only; all positions remain exact source joints'
        audit['palm_contact_verified'] = False
        return prior, audit

    def prepare_clip(self, joints, native_features=None, *, hand_pose=None, contact_weights=None):
        """Cache pose transforms in source order for deterministic random access.

        Optional native features are independently audited before their arm and
        hand rotations are used as a twist prior. Source joint positions always
        win over FK. The default position-only baseline remains unchanged.
        """
        if self._removed:
            raise RuntimeError('Native rig actor was removed')
        positions = np.array(joints, dtype=float, copy=True)
        if positions.ndim != 3 or positions.shape[1:] != (22, 3) or not 4 <= len(positions) <= 1000 or not np.isfinite(positions).all():
            raise ValueError('Expected 4–1000 finite native poses[T,22,3]')
        if hand_pose not in (None, 'handshake'):
            raise ValueError('Only the explicit authored handshake finger pose is supported')
        weights = None
        if hand_pose is None:
            if contact_weights is not None:
                raise ValueError('Contact weights require explicit hand_pose=handshake')
        else:
            weights = np.array(contact_weights, dtype=float, copy=True)
            if weights.shape != (len(positions), 2) or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
                raise ValueError('Authored handshake requires finite contact_weights[T,2] in [0,1]')
            available_sides = {control[0] for control in self.asset.finger_controls.values()}
            if any(np.any(weights[:, side]) and side not in available_sides for side in range(2)):
                raise ValueError('Asset lacks authored finger joints for a selected hand')
        prior, audit = (None, None) if native_features is None else self._native_priors(positions, native_features)
        previous_scale, previous_normals = self._scale, self._previous_normals
        self.reset_pose_history()
        try:
            prepared = [(*self._solve_pose(pose, None if prior is None else prior[index]),
                         None if weights is None else weights[index].copy())
                        for index, pose in enumerate(positions)]
        except Exception:
            self._scale, self._previous_normals = previous_scale, previous_normals
            raise
        self._prepared = prepared
        self.provenance['prepared_frames'] = len(prepared)
        self.provenance['deterministic_scrubbing'] = True
        self.provenance['native_rotation_prior_audit'] = audit
        self.provenance['authored_hand_pose'] = None if hand_pose is None else {
            'name': hand_pose, 'source': 'manually authored finger curls; not generated by InterGen',
            'weight_columns': ['SMPL left wrist20', 'SMPL right wrist21'],
            'nonzero_frames_per_hand': [int(np.count_nonzero(weights[:, side])) for side in range(2)],
            'max_weights_per_hand': weights.max(axis=0).tolist(),
            'finger_curl_degrees_mcp_pip_dip': [25., 35., 20.],
            'thumb_curl_degrees': [10., 15., 10.],
            'body_wrist_root_positions': 'unchanged native source coordinates',
            'contact_verified': False,
            'limitations': 'Partial rest-to-grasp curl only; no palm alignment, finger collision, force, or partner-contact solve.',
        }
        return deepcopy(self.provenance)

    def set_frame(self, frame):
        if self._removed:
            raise RuntimeError('Native rig actor was removed')
        if self._prepared is None or type(frame) is not int or not 0 <= frame < len(self._prepared):
            raise ValueError('Prepare a native clip and choose a valid frame')
        return self._publish_pose(*self._prepared[frame])

    def remove(self):
        if not self._removed:
            for handle in self.handles:
                handle.remove()
            self._removed = True
