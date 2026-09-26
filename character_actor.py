"""Animate an upright, textured generated human with the studio's G1 motion.

The generated mesh has no authored rig.  We estimate anatomical skin weights from
its silhouette and surface connectivity, then use the existing G1 joint poses to drive a native Viser
skinned mesh.  This is an approximate human fit, not a facial or finger rig.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from io import BytesIO

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from viser import _messages


@dataclass
class TexturedSkinnedMeshProps(_messages.SkinnedMeshProps):
    """App client extension; only our studio client reads these texture attributes."""

    uv: np.ndarray
    normals: np.ndarray
    texture_png: bytes
    normal_texture_png: bytes | None
    metallic_roughness_texture_png: bytes | None
    metallic_factor: float
    roughness_factor: float
    base_color_factor: tuple[float, float, float, float]


_BONES = (
    ("pelvis", "pelvis_skel", (0.0, -0.08, 0.0)),
    ("torso", "waist_pitch_skel", (0.0, 0.15, 0.0)),
    ("head", "waist_pitch_skel", (0.0, 0.37, 0.0)),
    ("left_upper_arm", "left_shoulder_yaw_skel", (0.13, 0.24, 0.0)),
    ("left_forearm", "left_elbow_skel", (0.16, 0.06, 0.0)),
    ("left_hand", "left_wrist_yaw_skel", (0.17, -0.12, 0.0)),
    ("right_upper_arm", "right_shoulder_yaw_skel", (-0.13, 0.24, 0.0)),
    ("right_forearm", "right_elbow_skel", (-0.16, 0.06, 0.0)),
    ("right_hand", "right_wrist_yaw_skel", (-0.17, -0.12, 0.0)),
    ("left_thigh", "left_hip_yaw_skel", (0.075, -0.14, 0.0)),
    ("left_shin", "left_knee_skel", (0.075, -0.30, 0.0)),
    ("left_foot", "left_ankle_roll_skel", (0.075, -0.44, 0.025)),
    ("right_thigh", "right_hip_yaw_skel", (-0.075, -0.14, 0.0)),
    ("right_shin", "right_knee_skel", (-0.075, -0.30, 0.0)),
    ("right_foot", "right_ankle_roll_skel", (-0.075, -0.44, 0.025)),
    # Intermediate shoulder rotations retain volume where the thorax and arm
    # meet. They share the upper-arm pivot; they are not additional limb joints.
    ("left_shoulder_blend", "left_shoulder_yaw_skel", (0.13, 0.24, 0.0)),
    ("right_shoulder_blend", "right_shoulder_yaw_skel", (-0.13, 0.24, 0.0)),
)


def _smoothstep(value: np.ndarray, low: float, high: float) -> np.ndarray:
    x = np.clip((value - low) / (high - low), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# The renderer accepts world-space bone transforms. These parent links are used
# to solve a real human hierarchy before publishing those transforms.
_PARENTS = np.array([-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 0, 12, 13, 1, 1])
_SHOULDER_BLENDS = ((3, 15), (6, 16))


def _fit_human(vertices: np.ndarray):
    """Fit an upright separated-limb silhouette in height-normalized coordinates.

    Horizontal empty-space gaps locate the actual arms, rather than assigning
    every wide vertex to a wrist. Local medians fit pivots inside each limb.
    This deliberately supports the generation contract (relaxed A-pose), not
    arbitrary posed characters, skirts or attached props.
    """
    v = np.asarray(vertices, dtype=np.float64)
    rest = np.array([pivot for _, _, pivot in _BONES], dtype=np.float64)
    # Human pelvis, thorax and neck; the torso rotates at the waist.
    rest[:3] = [[0, -.025, 0], [0, .055, 0], [0, .335, 0]]
    arm_masks = []
    for sign, offset in ((1, 3), (-1, 6)):
        x = sign * v[:, 0]
        samples = []
        for y in np.arange(-.16, .231, .008):
            ids = np.flatnonzero((abs(v[:, 1] - y) < .006) & (x > .065))
            if len(ids) < 16:
                continue
            order = ids[np.argsort(x[ids])]
            gaps = np.diff(x[order])
            if not len(gaps):
                continue
            j = int(np.argmax(gaps))
            outer = order[j + 1:]
            if gaps[j] > .009 and len(outer) >= 8 and j >= 5:
                boundary = (x[order[j]] + x[order[j + 1]]) / 2
                if boundary > .105 and np.median(x[outer]) > .125:
                    depth = np.quantile(v[outer,2], [.1,.9]).mean()
                    samples.append((y, boundary, np.median(x[outer]), depth))
        if samples:
            rows = np.asarray(samples)
            runs = np.split(rows, np.flatnonzero(np.diff(rows[:,0]) > .025)+1)
            # Discard isolated trouser/cuff gaps; the arm separation passes the waist.
            runs = [run for run in runs if run[0,0] <= .04 and run[-1,0] >= -.02]
            samples = max(runs, key=len).tolist() if runs else []
        if len(samples) >= 4:
            rows = np.asarray(samples)
            low = rows[0, 0]
            # Finger tips extend below the wrist by roughly half a palm.
            wrist_y = np.clip(low + .046, -.12, .035)
            shoulder_y = .275
            shoulder_slice = x[(abs(v[:, 1] - shoulder_y) < .012) & (x > .035)]
            shoulder_x = np.clip(np.quantile(shoulder_slice, .90) - .022, .085, .155) if len(shoulder_slice) else .115
            wrist_x = np.interp(wrist_y, rows[:, 0], rows[:, 2])
            elbow_y = (shoulder_y + wrist_y) / 2
            elbow_x = np.interp(elbow_y, rows[:, 0], rows[:, 2])
            # Silhouette centers above the armpit are not separately observable.
            elbow_x = max(elbow_x, (shoulder_x + wrist_x) / 2)
            rest[offset:offset+3] = [[sign * shoulder_x, shoulder_y, np.median(rows[:, 3])],
                                    [sign * elbow_x, elbow_y, np.interp(elbow_y, rows[:, 0], rows[:, 3])],
                                    [sign * wrist_x, wrist_y, np.interp(wrist_y, rows[:, 0], rows[:, 3])]]
            boundary = np.interp(v[:, 1], rows[:, 0], rows[:, 1])
            # Extend the observed armpit boundary into the shoulder cap.
            high_y = rows[-1, 0]
            blend = _smoothstep(v[:, 1], high_y, shoulder_y + .015)
            boundary = boundary * (1-blend) + (shoulder_x-.018)*blend
            amount = _smoothstep(x-boundary, -.009, .009)
            amount *= _smoothstep(v[:, 1], wrist_y-.110, wrist_y-.085)
            amount *= 1-_smoothstep(v[:, 1], shoulder_y-.008, shoulder_y+.043)
        else:
            # Sparse test meshes and imperfect silhouettes retain a bounded,
            # conservative A-pose fit instead of fitting noise as anatomy.
            rest[offset:offset+3] = [[sign*.115, .275, 0], [sign*.15, .10, 0], [sign*.18, -.065, 0]]
            boundary = np.interp(v[:, 1], [-.15, .08, .28], [.145, .115, .098])
            amount = _smoothstep(x-boundary, -.012, .012)
            amount *= _smoothstep(v[:, 1], -.16, -.12) * (1-_smoothstep(v[:, 1], .275, .32))
        arm_masks.append(amount)
    for sign, offset in ((1, 9), (-1, 12)):
        for j, y in enumerate((-.035, -.265, -.447)):
            ids = (sign*v[:, 0] > .015) & (abs(v[:, 1]-y) < .015)
            if j == 0:
                ids &= sign*v[:, 0] < .12
            point = np.quantile(v[ids], [.1,.9], axis=0).mean(axis=0) if ids.sum() >= 8 else np.array([sign*.075,y,0])
            rest[offset+j] = [point[0], y, point[2]]
    # The contracted reference is an approximately symmetric A-pose. Missing
    # gaps where one cuff touches the hip must not produce unequal arm lengths.
    wrist_y = (rest[5,1]+rest[8,1])/2
    for offset in (3,6):
        rest[offset+2,1] = wrist_y
        rest[offset+1,1] = (rest[offset,1]+wrist_y)/2
        ids = (abs(v[:,0]-rest[offset,0])<.025) & (abs(v[:,1]-rest[offset,1])<.016)
        if ids.sum() >= 8:
            rest[offset,2] = np.quantile(v[ids,2],[.1,.9]).mean()
    # Waist/head depth are also fitted rather than forcing a planar skeleton.
    for i in range(3):
        ids = (abs(v[:, 0]) < .055) & (abs(v[:, 1]-rest[i,1]) < .02)
        if ids.sum() >= 8:
            rest[i,2] = np.quantile(v[ids,2], [.1,.9]).mean()
    rest[15:17] = rest[[3, 6]]
    return rest, arm_masks


def _chain_weights(vertices, pivots, transition):
    """Blend only around anatomical joints, with rigid segment interiors."""
    first, joint, end = pivots
    upper_axis = joint-first
    lower_axis = end-joint
    upper_axis /= max(np.linalg.norm(upper_axis), 1e-8)
    lower_axis /= max(np.linalg.norm(lower_axis), 1e-8)
    elbow = _smoothstep((vertices-joint) @ upper_axis, -transition, transition)
    wrist = _smoothstep((vertices-end) @ lower_axis, -transition*.65, transition*.65)
    return np.stack([1-elbow, elbow*(1-wrist), elbow*wrist],axis=1)


def _surface_arm_masks(vertices, faces, fallback):
    """Separate hands from hips using surface connectivity, not proximity.

    Below the armpits an A-pose has three disjoint surface branches. These are
    unambiguous seeds. Geodesic distances extend their labels over the shoulder
    junction, where smooth arm/torso blending is actually anatomically valid.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    unique, groups = np.unique(np.round(vertices,6),axis=0,return_inverse=True)
    edges = np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    edges = groups[edges]
    edges = np.unique(np.sort(edges,axis=1),axis=0)
    edges = edges[edges[:,0] != edges[:,1]]
    lengths = np.linalg.norm(unique[edges[:,0]]-unique[edges[:,1]],axis=1)
    edges = np.concatenate([edges,edges[:,::-1]])
    lengths = np.tile(lengths,2)
    graph = coo_matrix((lengths,(edges[:,0],edges[:,1])),shape=(len(unique),len(unique))).tocsr()
    below = (unique[edges[:,0],1] < .15) & (unique[edges[:,1],1] < .15)
    cut = coo_matrix((lengths[below],(edges[below,0],edges[below,1])),shape=graph.shape).tocsr()
    _, labels = connected_components(cut,directed=False)
    sizes = np.bincount(labels)
    body = int(sizes.argmax())
    arm_labels = []
    for sign in (1,-1):
        candidates = []
        for label in np.flatnonzero(sizes > 30):
            part = unique[labels==label]
            if sign*np.median(part[:,0]) > .11 and part[:,1].min() > -.22 and part[:,1].max() > .08:
                candidates.append(label)
        if not candidates:
            return fallback
        arm_labels.append(max(candidates,key=lambda label:sizes[label]))
    if body in arm_labels or arm_labels[0] == arm_labels[1]:
        return fallback
    distances = np.stack([dijkstra(graph,directed=False,indices=np.flatnonzero(labels==label),min_only=True)
                          for label in [body,*arm_labels]],axis=1)
    finite = np.isfinite(distances).any(axis=1)
    scores = np.zeros_like(distances)
    scores[finite] = np.exp(-(distances[finite]-distances[finite].min(axis=1,keepdims=True))/.035)
    scores[~finite,0] = 1
    scores /= scores.sum(axis=1,keepdims=True)
    return [scores[groups,i] * (1-_smoothstep(vertices[:,1],.29,.335)) for i in (1,2)]


def _weights(vertices_normalized: np.ndarray, fit=None, faces=None) -> np.ndarray:
    v = np.asarray(vertices_normalized, dtype=np.float64)
    rest, arms = _fit_human(v) if fit is None else fit
    if faces is not None:
        arms = _surface_arm_masks(v,faces,arms)
    weights = np.zeros((len(v), len(_BONES)), dtype=np.float64)
    arm_total = np.clip(arms[0]+arms[1], 0, 1)
    leg = (1-arm_total) * (1-_smoothstep(v[:,1], -.080, -.005))
    body = np.clip(1-arm_total-leg, 0, 1)
    head = _smoothstep(v[:,1], .315, .35)
    torso = _smoothstep(v[:,1], -.025, .085)
    weights[:,0] = body*(1-torso)
    weights[:,1] = body*torso*(1-head)
    weights[:,2] = body*torso*head
    for amount, offset in zip(arms, (3,6)):
        weights[:,offset:offset+3] = amount[:,None]*_chain_weights(v, rest[offset:offset+3], .024)
    # Blend the connected crotch smoothly; separated lower legs are fully rigid.
    for sign, offset in ((1,9),(-1,12)):
        side = _smoothstep(sign*v[:,0],-.025,.025)
        weights[:,offset:offset+3] = (leg*side)[:,None]*_chain_weights(v, rest[offset:offset+3], .025)
    # UV islands duplicate positions; give all copies exactly the same weights
    # so rounding cannot open cracks along texture seams during articulation.
    _, groups = np.unique(np.round(v,decimals=6),axis=0,return_inverse=True)
    sums = np.zeros((groups.max()+1,len(_BONES)))
    np.add.at(sums,groups,weights)
    weights = (sums / np.bincount(groups)[:,None])[groups]
    weights = _shoulder_blend_weights(weights)
    keep = np.argpartition(weights, -4, axis=1)[:, -4:]
    trimmed = np.zeros_like(weights)
    rows = np.arange(len(v))[:,None]
    trimmed[rows,keep] = weights[rows,keep]
    empty = trimmed.sum(axis=1) < 1e-8
    trimmed[empty,1] = 1
    return (trimmed/trimmed.sum(axis=1,keepdims=True)).astype(np.float32)


def _shoulder_blend_weights(weights):
    """Route thorax/arm interpolation through a halfway shoulder rotation.

    Linear skinning between rotations 120 degrees apart can halve the shoulder
    radius. Splitting that interval in two bounds this contraction to cos(30°).
    Only the dominant upper arm participates, so tiny geodesic influences from
    the opposite arm cannot make the result depend on left/right processing
    order. Other joints and the total weight are unchanged.
    """
    result = weights.copy()
    dominant = np.argmax(weights[:, [3, 6]], axis=1)
    for side, (arm, blend) in enumerate(_SHOULDER_BLENDS):
        shared = np.minimum(weights[:, 1], weights[:, arm])
        shared = np.where(dominant == side, shared, 0.)
        result[:, 1] -= shared
        result[:, arm] -= shared
        result[:, blend] += 2 * shared
    return result


def _mesh_from_glb(data: bytes) -> trimesh.Trimesh:
    if not isinstance(data, bytes) or len(data) > 40 * 1024 * 1024:
        raise ValueError("Character GLB must be at most 40 MiB")
    scene = trimesh.load(BytesIO(data), file_type="glb", force="scene", process=False)
    meshes = [mesh for mesh in scene.dump() if isinstance(mesh, trimesh.Trimesh)]
    if len(meshes) != 1:
        raise ValueError("Character animation needs one textured human mesh")
    mesh = meshes[0]
    if len(mesh.vertices) > 250_000 or len(mesh.faces) > 500_000:
        raise ValueError("Character mesh is too dense for live animation")
    if not np.isfinite(mesh.vertices).all() or len(mesh.vertices) < 300:
        raise ValueError("Character mesh has invalid geometry")
    uv = getattr(mesh.visual, "uv", None)
    material = getattr(mesh.visual, "material", None)
    texture = getattr(material, "baseColorTexture", None)
    if uv is None or np.asarray(uv).shape != (len(mesh.vertices), 2) or texture is None:
        raise ValueError("Character animation needs a textured GLB with UV coordinates")
    return mesh


def _smooth_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted normals shared across duplicated UV seam positions."""
    _, groups = np.unique(np.round(vertices, decimals=6), axis=0, return_inverse=True)
    triangles = vertices[faces]
    area_normals = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
    normals = np.zeros((groups.max()+1,3),dtype=np.float64)
    for corner in range(3):
        np.add.at(normals,groups[faces[:,corner]],area_normals)
    normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    return normals[groups].astype(np.float32)


def _image_png_bytes(texture: object, mode: str = "RGB") -> bytes:
    texture = texture.convert(mode)
    output = BytesIO()
    texture.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _texture_bytes(mesh: trimesh.Trimesh) -> bytes:
    return _image_png_bytes(mesh.visual.material.baseColorTexture)


def _material_properties(mesh: trimesh.Trimesh) -> dict:
    """Carry glTF material maps through the native skinned-mesh transport."""
    material = mesh.visual.material
    normal = getattr(material, "normalTexture", None)
    metallic_roughness = getattr(material, "metallicRoughnessTexture", None)
    base_color = getattr(material, "baseColorFactor", None)
    return dict(
        normal_texture_png=_image_png_bytes(normal) if normal is not None else None,
        metallic_roughness_texture_png=(
            _image_png_bytes(metallic_roughness) if metallic_roughness is not None else None
        ),
        # glTF defaults to 1 when the factor is absent; authored nonmetallic
        # character GLBs carry an explicit zero, which remains zero here.
        metallic_factor=float(
            1.0 if getattr(material, "metallicFactor", None) is None
            else material.metallicFactor
        ),
        roughness_factor=float(
            1.0 if getattr(material, "roughnessFactor", None) is None
            else material.roughnessFactor
        ),
        base_color_factor=tuple(float(channel) / 255.0 for channel in base_color)
        if base_color is not None else (1.0, 1.0, 1.0, 1.0),
    )


def _as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _align_direction(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Shortest proper rotation, including stable parallel/antiparallel cases."""
    before = before / np.linalg.norm(before)
    after = after / np.linalg.norm(after)
    cross = np.cross(before, after)
    sine = np.linalg.norm(cross)
    cosine = np.clip(before @ after, -1., 1.)
    if sine > 1e-10:
        return Rotation.from_rotvec(cross * (np.arctan2(sine, cosine) / sine)).as_matrix()
    if cosine > 0:
        return np.eye(3)
    axis = np.cross(before, np.eye(3)[np.argmin(abs(before))])
    return Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.pi).as_matrix()


def _raise_ankle(positions, matrices, thigh, amount):
    """Pose-local two-bone IK: clear a boot without lifting the other leg.

    The existing knee bend guides the solution near contact onset. Additional
    flexion uses a stable anatomical forward pole near a straight knee. Segment
    lengths and ankle orientation stay fixed; flexion is bounded to 150 degrees.
    """
    hip, knee, ankle = positions[thigh:thigh+3].copy()
    upper, lower = knee-hip, ankle-knee
    lengths = np.array([np.linalg.norm(upper), np.linalg.norm(lower)])
    target = ankle + np.array([0., amount, 0.])
    axis = target-hip
    distance = np.linalg.norm(axis)
    if distance < 1e-8 or lengths.min() < 1e-8:
        return
    axis /= distance
    min_distance = np.sqrt(np.sum(lengths**2) + 2*np.prod(lengths)*np.cos(np.deg2rad(150)))
    distance = np.clip(distance, min_distance, lengths.sum()*(1-1e-7))
    target = hip + axis*distance
    along = (lengths[0]**2 + distance**2 - lengths[1]**2)/(2*distance)
    radius = np.sqrt(max(0., lengths[0]**2-along**2))
    old_axis = ankle-hip
    old_axis /= max(np.linalg.norm(old_axis), 1e-8)
    prior_bend = upper-old_axis*(upper @ old_axis)
    prior_radius = np.linalg.norm(prior_bend)
    prior_bend = _align_direction(old_axis, axis) @ prior_bend
    forward = matrices[thigh] @ np.array([0.,0.,1.])
    forward -= axis*(forward @ axis)
    if np.linalg.norm(forward) < 1e-5:
        forward = np.eye(3)[np.argmin(abs(axis))]
        forward -= axis*(forward @ axis)
    forward /= np.linalg.norm(forward)
    # Preserve the existing plane at contact onset. As IK introduces more
    # flexion than the source supplied, trust the anatomical forward pole;
    # millimeter source offsets near a locked knee must not choose its side.
    prior_weight = min(1., (prior_radius/max(radius,1e-8))**2)
    bend = prior_weight*prior_bend/max(prior_radius,1e-8) + (1-prior_weight)*forward
    if np.linalg.norm(bend) < 1e-8:
        bend = forward
    bend /= np.linalg.norm(bend)
    new_knee = hip + axis*along + bend*radius
    matrices[thigh] = _align_direction(upper, new_knee-hip) @ matrices[thigh]
    matrices[thigh+1] = _align_direction(lower, target-new_knee) @ matrices[thigh+1]
    positions[thigh+1:thigh+3] = [new_knee,target]


class GeneratedCharacterActor:
    """A skinned generated actor driven by global G1 joint positions/rotations.

    The generated model should depict one upright adult in a relaxed A-pose,
    viewed from the front with arms and legs clearly separated. Fitted
    clothing works best; dramatic poses and merged limbs reduce fitting quality.
    """

    def __init__(self, server, data: bytes, name: str, skeleton,
                 bind_positions=None, bind_rotations=None):
        mesh = _mesh_from_glb(data)
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        extent = bounds[1] - bounds[0]
        if not 0.4 <= extent[1] <= 4.0 or not 1.5 <= extent[1] / max(extent[0], 1e-6) <= 6.0:
            raise ValueError("Character animation needs an upright, full-body human")
        scale = 1.70 / extent[1]
        center = (bounds[0] + bounds[1]) / 2.0
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        normalized = (vertices - center) / extent[1]
        vertices = (vertices - np.array([center[0], bounds[0, 1], center[2]])) * scale
        fit = _fit_human(normalized)
        rest = fit[0].copy()
        rest[:,1] += .5
        rest *= 1.70
        weights = _weights(normalized, fit=fit, faces=np.asarray(mesh.faces))
        handle = None
        if server is not None:
            identity = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(rest), 1))
            handle = server.scene.add_mesh_skinned(
                name, np.asarray(vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.uint32),
                bone_wxyzs=identity, bone_positions=np.asarray(rest, dtype=np.float32),
                skin_weights=weights, color=(255, 255, 255), side="double",
                cast_shadow=True, receive_shadow=True,
            )
            try:
                original = handle._impl.props
                extra = TexturedSkinnedMeshProps(
                    **{field.name: getattr(original, field.name) for field in fields(_messages.SkinnedMeshProps)},
                    uv=np.asarray(mesh.visual.uv, dtype=np.float32),
                    normals=_smooth_normals(vertices, np.asarray(mesh.faces)),
                    texture_png=_texture_bytes(mesh),
                    **_material_properties(mesh),
                )
                handle._impl.props = extra
                server.scene._websock_interface.queue_message(_messages.SkinnedMeshMessage(name, extra))
            except Exception:
                handle.remove()
                raise
        self.handle = handle
        self._initialize_pose(vertices, np.asarray(mesh.faces), weights, rest, skeleton,
                              bind_positions, bind_rotations)

    @classmethod
    def from_fitted_mesh(cls, vertices, faces, weights, rest, skeleton,
                         bind_positions=None, bind_rotations=None):
        """Reuse a validated stored fit without creating a renderer or refitting."""
        actor = cls.__new__(cls)
        actor.handle = None
        actor._initialize_pose(vertices, faces, weights, rest, skeleton,
                               bind_positions, bind_rotations)
        return actor

    def _initialize_pose(self, vertices, faces, weights, rest, skeleton,
                         bind_positions=None, bind_rotations=None):
        self.skeleton = skeleton
        self.bone_indices = np.array([skeleton.bone_index[joint] for _, joint, _ in _BONES], dtype=np.int32)
        self.rest = rest
        self.vertices = vertices
        self.faces = np.asarray(faces)
        self._support_indices = np.flatnonzero(vertices[:,1] < .11)
        self.weights = weights
        support = self._support_indices
        left = weights[support,9:12].sum(axis=1) >= weights[support,12:15].sum(axis=1)
        self._leg_support_indices = (support[left], support[~left])
        self.bone_positions = rest.copy()
        self.bone_matrices = np.broadcast_to(np.eye(3), (len(rest),3,3)).copy()
        self.bind_positions = None
        self.bind_rotations = None
        self._vertical_scale = 1.70 / 1.27
        self._visible = True
        if bind_positions is not None and bind_rotations is not None:
            self.update(bind_positions, bind_rotations)

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._visible = bool(value)
        if self.handle is not None:
            self.handle.visible = self._visible

    def update(self, positions, rotations) -> None:
        pos = _as_numpy(positions).astype(np.float64)
        rot = _as_numpy(rotations).astype(np.float64)
        n = self.skeleton.nbjoints
        if pos.shape != (n, 3) or rot.shape != (n, 3, 3):
            raise ValueError("Character pose does not match G1 skeleton")
        if not np.isfinite(pos).all() or not np.isfinite(rot).all():
            raise ValueError("Character pose contains invalid values")
        if self.bind_positions is None:
            # G1 joint frames differ greatly from a human A-pose, particularly
            # at the wrists and ankles. Use the first actual motion pose as the
            # calibration frame so the original silhouette stays intact.
            self.bind_positions = pos.copy()
            self.bind_rotations = Rotation.from_matrix(rot).as_matrix()
            # Calibrate pelvis travel from standing hip height, not overall
            # head-to-toe stature: the robot and human have different torso
            # proportions. Retain the nominal fallback for zero-height test
            # or incomplete calibration poses.
            source_height = pos[self.bone_indices[0],1]
            if source_height > 1e-6:
                self._vertical_scale = float(np.clip(self.rest[0,1]/source_height, .5, 2.))
        # Calibrate joint frames once. Use the complete shoulder yaw frame
        # (pitch+roll+yaw), not the pitch-only frame from the robot chain.
        source_bind = self.bind_rotations[self.bone_indices]
        source_now = Rotation.from_matrix(rot[self.bone_indices]).as_matrix()
        bind_root = self.bind_rotations[self.bone_indices[0]]
        # Keep the generated A-pose upright; the robot's initial lean is a
        # calibration offset. Retain its heading and all subsequent root tilt.
        lateral = bind_root[:,0].copy()
        lateral[1] = 0
        lateral /= max(np.linalg.norm(lateral),1e-12)
        heading = np.column_stack([lateral,[0,1,0],np.cross(lateral,[0,1,0])])
        calibrated = source_now @ np.transpose(source_bind, (0,2,1)) @ heading
        bone_matrices = np.empty_like(calibrated)
        bone_positions = np.empty_like(self.rest)
        bone_matrices[0] = calibrated[0]
        # Horizontal positions are stage coordinates shared with interactions
        # and other actors. Scaling them by human height moves the actor away
        # from scene targets. Only vertical motion needs stature compensation:
        # robot pelvis height is not a human hip height.
        root_index = self.bone_indices[0]
        bone_positions[0] = pos[root_index]
        bone_positions[0,1] = (self.rest[0,1] +
                              (pos[root_index,1] - self.bind_positions[root_index,1]) * self._vertical_scale)
        for i in range(1,len(self.rest)):
            parent = _PARENTS[i]
            local_delta = calibrated[parent].T @ calibrated[i]
            bone_matrices[i] = bone_matrices[parent] @ local_delta
            bone_positions[i] = bone_positions[parent] + bone_matrices[parent] @ (self.rest[i]-self.rest[parent])
        for arm, blend in _SHOULDER_BLENDS:
            relative = bone_matrices[1].T @ bone_matrices[arm]
            halfway = Rotation.from_rotvec(Rotation.from_matrix(relative).as_rotvec() * .5).as_matrix()
            bone_matrices[blend] = bone_matrices[1] @ halfway
            bone_positions[blend] = bone_positions[arm]
        # Robot ankle poses can point a longer human boot through the stage.
        # First bend that leg to clear its sole, so an airborne foot does not
        # bounce the pelvis and lift the planted foot. This depends only on the
        # current pose, so timeline seeking and reverse playback are identical.
        for thigh, support in zip((9,12), self._leg_support_indices):
            if not len(support):
                continue
            for _ in range(3):
                support_y = np.zeros(len(support))
                for i in range(len(self.rest)):
                    support_y += self.weights[support,i] * ((self.vertices[support]-self.rest[i]) @ bone_matrices[i,1] + bone_positions[i,1])
                penetration = -float(support_y.min())
                if penetration <= 1e-7:
                    break
                _raise_ankle(bone_positions, bone_matrices, thigh, penetration)
        # A global lift is a last resort for unreachable or unusually weighted
        # support geometry; it still preserves every segment length.
        support = self._support_indices
        support_y = np.zeros(len(support))
        for i in range(len(self.rest)):
            support_y += self.weights[support,i] * ((self.vertices[support]-self.rest[i]) @ bone_matrices[i,1] + bone_positions[i,1])
        if len(support_y):
            bone_positions[:,1] += max(0., -float(support_y.min()))
        self.bone_positions = bone_positions
        self.bone_matrices = bone_matrices
        bone_rotations = Rotation.from_matrix(bone_matrices).as_quat(scalar_first=True)
        for bone, position, quat in zip(self.handle.bones if self.handle is not None else (), bone_positions, bone_rotations):
            bone.position = np.asarray(position, dtype=np.float32)
            bone.wxyz = np.asarray(quat, dtype=np.float32)

    def deform_vertices(self) -> np.ndarray:
        """CPU equivalent of the renderer's linear skinning, for QA/export."""
        result = np.zeros_like(self.vertices)
        for i in range(len(self.rest)):
            transformed = (self.vertices-self.rest[i]) @ self.bone_matrices[i].T + self.bone_positions[i]
            result += self.weights[:,i,None]*transformed
        return result

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
