"""One compatibility result for GLB validation, rig mapping and user actions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from character_assets import AssetValidationError, inspect_glb
from retargeting import REQUIRED_ROLES, RigMappingError, build_retargeter, detect_rig_profile


VISUAL_CAVEAT = 'Technical compatibility does not guarantee natural joint deformation. Inspect a standing pose, raised arms, walking and a squat.'
RIGGING_GUIDANCE = ('In Blender, work on a copy: place a humanoid armature at the hips, spine, head, arms, hands, legs and feet; '
                    'use the supported Mixamo names and hierarchy; bind skin weights, correct shoulders, hips, elbows and knees, '
                    'then export a GLB with its skin, bones and textures. A mapping file cannot create bones or weights.')


@dataclass(frozen=True)
class CharacterCompatibility:
    status: str
    title: str
    reasons: tuple[str, ...]
    actions: tuple[str, ...]
    retargeter: object | None = None
    checks: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def motion_ready(self):
        return self.status == 'motion_ready'

    @property
    def can_preview(self):
        return self.status != 'unsupported'

    @property
    def can_map(self):
        return self.status in ('motion_ready', 'mapping_required')


def unsupported_character(reason):
    return CharacterCompatibility('unsupported', 'Unsupported', (str(reason),), ('Choose another file',))


def _selected_scene_content(asset):
    """Nodes displayed by GLTFLoader's default scene and their actual skins."""
    document = asset.document
    pending = list(document['scenes'][document.get('scene', 0)].get('nodes', []))
    reachable = set()
    while pending:
        index = pending.pop()
        if index not in reachable:
            reachable.add(index)
            pending.extend(asset.nodes[index].children)
    skin_indices = {document['nodes'][index]['skin'] for index in reachable
                    if 'mesh' in document['nodes'][index] and 'skin' in document['nodes'][index]}
    joints = {joint for index in skin_indices for joint in document['skins'][index]['joints']}
    return reachable, joints


def _rig_limitations(asset, joints):
    """Failures that renaming/reassigning existing bones cannot repair."""
    limitations = []
    if len(joints) < len(REQUIRED_ROLES):
        limitations.append(f'The skin has {len(joints)} joints; motion needs at least {len(REQUIRED_ROLES)} distinct humanoid joints.')
    roots = set()
    joint_parents = set()
    for joint in joints:
        ancestors = []
        parent = asset.nodes[joint].parent
        while parent is not None:
            if parent in joints:
                ancestors.append(parent)
            parent = asset.nodes[parent].parent
        roots.add(ancestors[-1] if ancestors else joint)
        if ancestors:
            joint_parents.add(ancestors[0])
    if len(roots) != 1:
        limitations.append('The skin has independent joint roots; all humanoid skin joints must descend from one pelvis.')
    if len(joints - joint_parents) < 4:
        limitations.append('The joint hierarchy needs separate left/right arm and leg chains.')
    for node in asset.nodes:
        for matrix in (node.local_matrix, node.world_matrix):
            scale = np.linalg.norm(matrix[:3, :3], axis=0)
            rotation = matrix[:3, :3] / scale[0]
            if (not np.allclose(scale, scale[0], atol=1e-7, rtol=1e-5)
                    or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
                    or not np.allclose(np.linalg.det(rotation), 1, atol=1e-5, rtol=0)):
                limitations.append('The rig contains nonuniform, reflected or sheared transforms. Apply/correct transforms in Blender before retargeting.')
                return tuple(limitations)
    return tuple(limitations)


def _minimal_rig_calibration_failure(asset, scene_joints, skeleton):
    """A named minimal rig has no spare joints to repair bad rest geometry.

    Validate the native profile separately so malformed user JSON remains a
    mapping error. Larger or unnamed rigs may have an alternative valid role
    assignment, so retain their mapping action instead of blocking it.
    """
    if len(scene_joints) != len(REQUIRED_ROLES) or detect_rig_profile(asset) is None:
        return None
    try:
        build_retargeter(asset, skeleton=skeleton)
    except RigMappingError as exc:
        return str(exc)
    return None


def assess_character(asset, mapping=None, *, skeleton=None):
    """Assess an importer-validated asset without modifying it or its mapping."""
    checks = ('GLB integrity', 'Supported triangle geometry', 'Finite node transforms')
    scene_nodes, scene_joints = _selected_scene_content(asset)
    if not scene_joints:
        return CharacterCompatibility('static_preview', 'Static preview only',
            ('No usable skin with bone weights is present in the selected scene. Motion generation is unavailable.',),
            ('Open static preview', 'Choose another file'), checks=checks, warnings=(RIGGING_GUIDANCE,))
    checks += ('Selected scene skin', 'Skin joint indices and weights', 'Acyclic bone hierarchy')
    limitations = _rig_limitations(asset, scene_joints)
    if limitations:
        return CharacterCompatibility('static_preview', 'Static preview only', limitations,
            ('Open static preview', 'Choose another file'), checks=checks, warnings=(RIGGING_GUIDANCE,))
    try:
        rig = build_retargeter(asset, mapping=mapping, skeleton=skeleton)
        # The browser maps both traversed scene nodes and bones referenced by
        # its displayed SkinnedMesh skeletons. Do not use unrelated scenes to
        # satisfy required mappings. The importer currently also requires the
        # selected skin's joints to be reachable from that scene's roots.
        available_nodes = scene_nodes | scene_joints
        mapped_nodes = set(rig.profile.bones.values())
        if not mapped_nodes.issubset(available_nodes) or not mapped_nodes.issubset(scene_joints):
            raise RigMappingError('Mapped humanoid bones must belong to a skin displayed in the selected scene')
    except RigMappingError as exc:
        calibration_failure = _minimal_rig_calibration_failure(asset, scene_joints, skeleton)
        if calibration_failure is not None:
            return CharacterCompatibility('static_preview', 'Static preview only',
                (calibration_failure, 'This recognized rig has only the required humanoid joints. Correct its rest joint placement in Blender; no alternative skin joints are available for remapping.'),
                ('Open static preview', 'Choose another file'), checks=checks, warnings=(RIGGING_GUIDANCE,))
        return CharacterCompatibility('mapping_required', 'Bone mapping required',
            (str(exc), 'Map the existing pelvis, spine, upper arms, forearms, thighs, shins and feet to the required humanoid roles. If no alternative joints fit, correct their rest placement in Blender.'),
            ('Load rig mapping', 'Open static preview', 'Choose another file'), checks=checks, warnings=(VISUAL_CAVEAT,))
    return CharacterCompatibility('motion_ready', 'Ready for motion transfer',
        ('Required humanoid bones, hierarchy and calibration are compatible.',),
        ('Load rig mapping',), retargeter=rig,
        checks=checks + ('Uniform rig transforms', 'Required humanoid bone mapping', 'Bind calibration'),
        warnings=(VISUAL_CAVEAT, *rig.warnings))


def inspect_character(data, *, display_name='Imported GLB', mapping=None, skeleton=None):
    """Report parser failures with the same result type used for valid rigs."""
    try:
        asset = inspect_glb(data, display_name=display_name)
    except AssetValidationError as exc:
        return None, unsupported_character(exc)
    return asset, assess_character(asset, mapping, skeleton=skeleton)
