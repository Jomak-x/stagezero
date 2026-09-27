"""Generated civilian mesh with explicit Core27-to-human skeletal retargeting.

The mesh is model-generated, but its 17-bone anatomical rig is an approximate
fit. Fitting and skin-weight helpers are a read-only snapshot of the existing
character_actor.py implementation. Native body motion is transferred directly,
without authored pose replacement or first-frame A-pose calibration. Optional
wrist endpoint retargeting follows the actual native targets in every frame.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import base64
import hashlib

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from motion_bridge import _layout, _two_bone

FIT_SOURCE_SHA256 = "daaafa95ce27cdc77383749966aedcbe1fe620161023df8c036e0abe27b4e0d5"

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


CORE_MAP = ('Hips', 'Spine3', 'Head', 'LeftArm', 'LeftForeArm', 'LeftHand',
            'RightArm', 'RightForeArm', 'RightHand', 'LeftUpLeg', 'LeftLeg',
            'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftArm', 'RightArm')
# A generated A-pose has different limb axes from Core's canonical T-pose.
# These explicit anatomical segments establish the bind correction once.
_DIRECTION_MAP = {1: (2, 'Spine3', 'Head'), 3: (4, 'LeftArm', 'LeftForeArm'),
    4: (5, 'LeftForeArm', 'LeftHand'), 6: (7, 'RightArm', 'RightForeArm'),
    7: (8, 'RightForeArm', 'RightHand'), 9: (10, 'LeftUpLeg', 'LeftLeg'),
    10: (11, 'LeftLeg', 'LeftFoot'), 12: (13, 'RightUpLeg', 'RightLeg'),
    13: (14, 'RightLeg', 'RightFoot')}


def _data_url(data):
    return None if data is None else 'data:image/png;base64,'+base64.b64encode(data).decode('ascii')


class GroundedCharacter:
    """Actual textured GLB plus an approximate anatomical Core-driven skin rig.

    The mesh's relaxed A-pose is the bind pose, with identity bone rotations.
    Native Core source poses retain their articulation from their first frame.
    Retargeting changes limb proportions but does not synthesize gestures,
    remove motion, or calibrate against a selected frame. Optional two-bone arm
    IK preserves native wrist trajectories using the source elbow plane.
    """
    def __init__(self, glb_path, height=1.70):
        self.path=Path(glb_path)
        source=self.path.read_bytes()
        self.mesh_sha256=hashlib.sha256(source).hexdigest()
        mesh=_mesh_from_glb(source)
        bounds=np.asarray(mesh.bounds,dtype=float)
        extent=bounds[1]-bounds[0]
        if not .4<=extent[1]<=4. or not 1.5<=extent[1]/max(extent[0],1e-6)<=6.:
            raise ValueError('Character must be an upright separated-limb human')
        if not np.isfinite(height) or not 1.0<=height<=2.5:
            raise ValueError('Character height must be 1.0 to 2.5 metres')
        center=bounds.mean(axis=0)
        normalized=(np.asarray(mesh.vertices)-center)/extent[1]
        self.vertices=(np.asarray(mesh.vertices)-[center[0],bounds[0,1],center[2]])*height/extent[1]
        self.faces=np.asarray(mesh.faces,dtype=np.int32)
        self.uv=np.asarray(mesh.visual.uv,dtype=np.float32)
        self.normals=_smooth_normals(self.vertices,self.faces)
        fit=_fit_human(normalized)
        self.rest=fit[0].copy()
        self.rest[:,1]+=.5
        self.rest*=height
        self.weights=_weights(normalized,fit=fit,faces=self.faces)
        self.skin_indices=np.argsort(self.weights,axis=1)[:,-4:][:,::-1]
        self.skin_weights=np.take_along_axis(self.weights,self.skin_indices,axis=1)
        self.skin_weights/=self.skin_weights.sum(axis=1,keepdims=True)
        self.foot_surface_indices=tuple(np.flatnonzero(self.weights[:,bone]>=.5) for bone in (11,14))
        self.texture_png=_texture_bytes(mesh)
        props=_material_properties(mesh)
        self.normal_texture_png=props.pop('normal_texture_png')
        self.metallic_roughness_texture_png=props.pop('metallic_roughness_texture_png')
        self.material=props
        names,_,neutral=_layout()
        self.core_names=list(names)
        self.core_idx={name:i for i,name in enumerate(names)}
        self.core_map=np.asarray([self.core_idx[name] for name in CORE_MAP])
        self.correction=np.broadcast_to(np.eye(3),(17,3,3)).copy()
        for bone,(child,start,end) in _DIRECTION_MAP.items():
            self.correction[bone]=_align_direction(self.rest[child]-self.rest[bone],
                neutral[self.core_idx[end]]-neutral[self.core_idx[start]])
        # Hands inherit the forearm's bind-axis correction; fingers have no
        # generated joint rig and remain part of the hand segment.
        self.correction[5]=self.correction[4]
        self.correction[8]=self.correction[7]
        # A fixed canonical height correction, never a first-frame calibration:
        # both the source neutral toe and generated mesh's sole rest on y=0.
        self.root_height_offset=float(self.rest[0,1]+neutral[:,1].min())
        self.provenance={
            'character_source':'existing generated textured GLB',
            'mesh_sha256':self.mesh_sha256,'source_name':self.path.name,
            'rig_method':'anatomical silhouette/connectivity fit, 17 bones, four-weight LBS',
            'fit_source_sha256':FIT_SOURCE_SHA256,
            'retarget_method':'canonical Core27 T-pose to fitted human A-pose axis alignment; fixed-length FK',
            'root_height_offset_m':self.root_height_offset,
            'limitations':['Approximate fitted rig, not an artist-authored production rig.',
                'Core spine is compressed to one torso bone; no independent clavicle, finger or facial motion.',
                'Different limb proportions change contact locations; no contact or floor IK is applied.',
                'Texture and anatomical defects in the generated GLB remain visible.']}

    def rig_payload(self):
        """JSON-ready bind mesh. Skin local = vertex minus rest_positions[j]."""
        return {'vertices':self.vertices.tolist(),'faces':self.faces.tolist(),
            'normals':self.normals.tolist(),'uv':self.uv.tolist(),
            'rest_positions':self.rest.tolist(),'bone_names':[x[0] for x in _BONES],
            'parents':_PARENTS.tolist(),'skin_indices':self.skin_indices.tolist(),
            'skin_weights':self.skin_weights.tolist(),'material':self.material,
            'texture_url':_data_url(self.texture_png),
            'normal_texture_url':_data_url(self.normal_texture_png),
            'metallic_roughness_texture_url':_data_url(self.metallic_roughness_texture_png),
            'bind_rotation_convention':'identity; posed_vertex=sum(w*(R*(v-rest)+position))',
            'provenance':self.provenance}

    def retarget(self, positions, rotations, *, preserve_root_height=False, preserve_wrists=False,
                 wrist_target_space='retargeted_root'):
        """Return one native Core27 frame as fitted positions/rotations[17,...]."""
        source=np.asarray(positions,dtype=float)
        rotation=np.asarray(rotations,dtype=float)
        if source.shape!=(27,3) or rotation.shape!=(27,3,3):
            raise ValueError('Expected Core27 positions[27,3] and global rotations[27,3,3]')
        if not np.isfinite(source).all() or not np.isfinite(rotation).all():
            raise ValueError('Native motion must be finite')
        if wrist_target_space not in ('retargeted_root','native_world'):
            raise ValueError('wrist_target_space must be retargeted_root or native_world')
        if np.max(np.abs(rotation.swapaxes(-1,-2)@rotation-np.eye(3)))>.01 or np.min(np.linalg.det(rotation))<.99:
            raise ValueError('Native motion requires proper global rotation matrices')
        matrices=rotation[self.core_map]@self.correction
        # Global rotations carry twist, while observed model endpoints fix the
        # exact limb direction. This also handles retargeted InterGen Core data.
        for bone,(child,start,end) in _DIRECTION_MAP.items():
            target=source[self.core_idx[end]]-source[self.core_idx[start]]
            if np.linalg.norm(target)<1e-7:
                raise ValueError('Native motion contains a zero-length anatomical segment')
            matrices[bone]=_align_direction(matrices[bone]@(self.rest[child]-self.rest[bone]),target)@matrices[bone]
        fitted=np.empty((17,3))
        fitted[0]=source[0]
        if not preserve_root_height:
            fitted[0,1]+=self.root_height_offset
        for bone in range(1,17):
            parent=_PARENTS[bone]
            fitted[bone]=fitted[parent]+matrices[parent]@(self.rest[bone]-self.rest[parent])
        if preserve_wrists:
            # Standard endpoint-preserving retargeting: targets and elbow poles
            # come from EVERY native frame. This does not create a shared hand
            # lock, invent a gesture, or hold the hand at a fixed world point.
            offset=np.array([0.,0. if preserve_root_height or wrist_target_space=='native_world' else self.root_height_offset,0.])
            for arm,elbow,hand in ((3,4,5),(6,7,8)):
                target=source[self.core_map[hand]]+offset
                pole=source[self.core_map[elbow]]+offset
                upper=fitted[elbow]-fitted[arm]
                lower=fitted[hand]-fitted[elbow]
                middle,end=_two_bone(fitted[arm],target,pole,
                    np.linalg.norm(upper),np.linalg.norm(lower),upper)
                matrices[arm]=_align_direction(upper,middle-fitted[arm])@matrices[arm]
                matrices[elbow]=_align_direction(lower,end-middle)@matrices[elbow]
                fitted[elbow],fitted[hand]=middle,end
        for arm,blend in _SHOULDER_BLENDS:
            relative=matrices[1].T@matrices[arm]
            halfway=Rotation.from_rotvec(Rotation.from_matrix(relative).as_rotvec()*.5).as_matrix()
            matrices[blend]=matrices[1]@halfway
            fitted[blend]=fitted[arm]
        return {'positions':fitted,'rotations':matrices}

    def clip_payload(self, positions, rotations, *, preserve_root_height=False,
                     preserve_wrists=False, wrist_target_space='retargeted_root',
                     preserve_feet=False, floor_y=None, floor_offsets=None, fps=20.,
                     source_terrain_geometry=None, render_terrain_geometry=None):
        """JSON-ready fitted arrays [actors,frames,17,...] for native clips."""
        p,r=np.asarray(positions),np.asarray(rotations)
        if p.ndim==3:p,r=p[None],r[None]
        if p.ndim!=4 or p.shape[2:]!=(27,3) or r.shape!=p.shape[:-1]+(3,3):
            raise ValueError('Expected Core clip [actors,frames,27,3] and matching rotations')
        terrain_mode=source_terrain_geometry is not None or render_terrain_geometry is not None
        if terrain_mode and (source_terrain_geometry is None or render_terrain_geometry is None):
            raise ValueError('Terrain fitting requires both source and render geometry')
        if terrain_mode and any(not callable(getattr(geometry,'support_height',None))
                                for geometry in (source_terrain_geometry,render_terrain_geometry)):
            raise TypeError('Terrain geometry must provide support_height')
        if preserve_feet and floor_y is None and not terrain_mode:
            raise ValueError('Foot mesh clearance requires an explicit floor_y')
        fitted_p=np.empty(p.shape[:2]+(17,3));fitted_r=np.empty(p.shape[:2]+(17,3,3))
        for actor in range(len(p)):
            for frame in range(p.shape[1]):
                pose=self.retarget(p[actor,frame],r[actor,frame],preserve_root_height=preserve_root_height,
                    preserve_wrists=preserve_wrists,wrist_target_space=wrist_target_space)
                fitted_p[actor,frame]=pose['positions'];fitted_r[actor,frame]=pose['rotations']
        provenance={**self.provenance,'preserve_native_wrists':bool(preserve_wrists)}
        if preserve_wrists:
            provenance['retarget_method']+='; two-bone arms follow per-frame native wrist targets and native elbow poles'
            provenance['wrist_target_space']=wrist_target_space
            provenance['wrist_target_reference']=('native global wrist plus disclosed clip-wide floor translation; canonical pelvis offset is not applied to world-space hand targets'
                if wrist_target_space=='native_world' else 'native global wrist plus fixed canonical pelvis offset and any disclosed clip-wide floor translation')
            provenance['limitations']=[x for x in self.provenance['limitations'] if not x.startswith('Different limb')]+[
                'Native wrist targets are retained where reachable; hand geometry and rigid fingers can still change visible contact. No foot IK is applied.']
            offset=np.array([0.,0. if preserve_root_height or wrist_target_space=='native_world' else self.root_height_offset,0.])
            targets=p[:,:,[self.core_idx['LeftHand'],self.core_idx['RightHand']]]+offset
            errors=np.linalg.norm(fitted_p[:,:,[5,8]]-targets,axis=-1)
            provenance['wrist_target_error_m']={'max':float(errors.max()),'mean':float(errors.mean()),
                'unreachable_fraction':float(np.mean(errors>.005))}
        if floor_y is not None or floor_offsets is not None or terrain_mode:
            if (floor_y is not None and not np.isfinite(floor_y)) or not np.isfinite(fps) or fps<=0:
                raise ValueError('Floor and positive frame rate must be finite')
            if floor_offsets is not None:
                floor_offsets=np.asarray(floor_offsets,dtype=float)
                if floor_offsets.shape!=(len(p),) or not np.isfinite(floor_offsets).all():
                    raise ValueError('floor_offsets must contain one finite fixed offset per actor')
            calibrations=[]
            for actor in range(len(p)):
                if floor_offsets is None:
                    if terrain_mode:
                        shift,details=self._terrain_calibration(p[actor],fitted_p[actor],fitted_r[actor],
                            source_terrain_geometry,render_terrain_geometry,float(fps))
                    else:
                        shift,details=self._floor_calibration(p[actor],fitted_p[actor],fitted_r[actor],float(floor_y),float(fps))
                else:
                    shift=float(floor_offsets[actor])
                    details={'applied':True,'offset_m':shift,'floor_y':floor_y,
                        'method':'reused caller-supplied fixed floor offset; fitted prefix preserved'}
                fitted_p[actor,:,:,1]+=shift
                calibrations.append(details)
            provenance['floor_calibration']=calibrations
            provenance['floor_offsets']=[row['offset_m'] for row in calibrations]
        if preserve_feet:
            provenance['limitations']=[x.replace('No foot IK is applied.','Vertical boot clearance is enabled; it does not prevent foot sliding or guarantee contact.') for x in provenance['limitations']]
            corrections=[]
            for actor in range(len(p)):
                floor_offset=provenance['floor_offsets'][actor]
                for frame in range(p.shape[1]):
                    corrections.append(self._preserve_feet(p[actor,frame],fitted_p[actor,frame],
                        fitted_r[actor,frame],float(floor_y) if floor_y is not None else None,
                        floor_offset,source_terrain_geometry,render_terrain_geometry))
            provenance['preserve_native_feet']=False
            provenance['foot_retarget']={
                'method':'causal vertical mesh clearance from existing fitted ankles; fixed-length leg IK; smooth native near-ground eligibility; no downward snap',
                'root_motion':'unchanged from fixed clip-wide floor calibration; no per-frame root correction',
                'target_space':'existing fitted ankle x/z retained; only penetrating near-ground boot targets lift vertically',
                'max_sole_target_lift_m':max(row['max_sole_target_lift_m'] for row in corrections),
                'max_unreachable_target_error_m':max(row['max_unreachable_target_error_m'] for row in corrections),
                'unreachable_fraction':float(np.mean([row['unreachable_fraction'] for row in corrections]))}
            if terrain_mode:
                provenance['foot_retarget'].update({
                    'support_mode':'explicit terrain-relative source toes and rendered sole vertices',
                    'max_ankle_displacement_m':max(row['max_ankle_displacement_m'] for row in corrections),
                    'max_knee_angle_change_deg':max(row['max_knee_angle_change_deg'] for row in corrections),
                    'max_remaining_sole_penetration_m':max(row['max_remaining_sole_penetration_m'] for row in corrections),
                    'reach_limited_fraction':float(np.mean([row['reach_limited_fraction'] for row in corrections]))})
        return {'fitted_positions':fitted_p.tolist(),'fitted_rotations':fitted_r.tolist(),
                'character_provenance':provenance}

    @staticmethod
    def _support_height(geometry,x,z,y):
        height=geometry.support_height(float(x),float(z),float(y),max_step_up=.45,max_drop=2.)
        if height is None:return None
        if not np.isfinite(height):raise ValueError('Terrain support height must be finite')
        return float(height)

    def _sole_vertices(self, surface, positions, rotations):
        si=self.skin_indices[surface];sw=self.skin_weights[surface]
        local=self.vertices[surface,None]-self.rest[si]
        return np.sum((np.einsum('vwij,vwj->vwi',rotations[si],local)+positions[si])*sw[...,None],axis=1)

    @staticmethod
    def _knee_angle(hip,knee,ankle):
        upper=hip-knee;lower=ankle-knee
        cosine=np.dot(upper,lower)/max(np.linalg.norm(upper)*np.linalg.norm(lower),1e-12)
        return float(np.degrees(np.arccos(np.clip(cosine,-1.,1.))))

    def _preserve_feet(self, source, positions, rotations, floor_y, floor_offset,
                       source_terrain_geometry=None, render_terrain_geometry=None):
        """Frame-local mesh clearance, leaving nonpenetrating motion intact."""
        errors=[];lifts=[];ankle_displacements=[];knee_changes=[];remaining=[];limited=[]
        for side,(thigh,knee,foot),surface in zip(('Left','Right'),((9,10,11),(12,13,14)),self.foot_surface_indices):
            target=positions[foot].copy()
            original_target=target.copy()
            pole=positions[knee].copy()
            original_angle=self._knee_angle(positions[thigh],positions[knee],positions[foot]) if render_terrain_geometry is not None else 0.
            lengths=(np.linalg.norm(self.rest[knee]-self.rest[thigh]),
                     np.linalg.norm(self.rest[foot]-self.rest[knee]))
            def solve():
                old_upper=positions[knee]-positions[thigh]
                old_lower=positions[foot]-positions[knee]
                middle,end=_two_bone(positions[thigh],target,pole,*lengths,old_upper)
                rotations[thigh]=_align_direction(old_upper,middle-positions[thigh])@rotations[thigh]
                rotations[knee]=_align_direction(old_lower,end-middle)@rotations[knee]
                positions[knee],positions[foot]=middle,end
            # Continuous support eligibility avoids a threshold-crossing pop.
            toe=source[self.core_idx[side+'ToeBase']]
            if source_terrain_geometry is None:
                clearance=toe[1]
            else:
                support=self._support_height(source_terrain_geometry,toe[0],toe[2],toe[1])
                clearance=toe[1]-support if support is not None else np.inf
            u=np.clip((.30-clearance)/.15,0.,1.)
            eligibility=u*u*(3.-2.*u)
            if eligibility>0 and len(surface):
                floor_target=None
                for _ in range(3):
                    v=self._sole_vertices(surface,positions,rotations)
                    if render_terrain_geometry is None:
                        if floor_target is None:
                            initial_y=float(v[:,1].min())
                            floor_target=initial_y+max(0.,floor_y-initial_y)*eligibility
                        penetration=float(floor_target-v[:,1].min())
                    else:
                        supports=[self._support_height(render_terrain_geometry,row[0],row[2],row[1]) for row in v]
                        if floor_target is None:
                            floor_target=np.asarray([row[1]+max(0.,height-row[1])*eligibility
                                if height is not None else -np.inf for row,height in zip(v,supports)])
                        penetration=float(np.max(floor_target-v[:,1]))
                    if penetration<=1e-6:break
                    target[1]+=penetration+1e-5
                    solve()
            lifts.append(float(target[1]-original_target[1]))
            errors.append(float(np.linalg.norm(positions[foot]-target)))
            if render_terrain_geometry is not None:
                ankle_displacements.append(float(np.linalg.norm(positions[foot]-original_target)))
                knee_changes.append(abs(self._knee_angle(positions[thigh],positions[knee],positions[foot])-original_angle))
                v=self._sole_vertices(surface,positions,rotations) if len(surface) else np.empty((0,3))
                supports=[self._support_height(render_terrain_geometry,row[0],row[2],row[1]) for row in v]
                remaining.append(max([max(0.,height-row[1]) for row,height in zip(v,supports)
                                      if height is not None] or [0.]))
                limited.append(errors[-1]>.005)
        result={'max_sole_target_lift_m':max(lifts),'max_unreachable_target_error_m':max(errors),
                'unreachable_fraction':float(np.mean(np.asarray(errors)>.005))}
        if render_terrain_geometry is not None:
            result.update(max_ankle_displacement_m=max(ankle_displacements),
                max_knee_angle_change_deg=max(knee_changes),
                max_remaining_sole_penetration_m=max(remaining),
                reach_limited_fraction=float(np.mean(limited)))
        return result

    def _terrain_calibration(self, source, positions, rotations, source_geometry, render_geometry, fps):
        """One fixed shift from terrain-relative native contacts and fitted soles."""
        clearances=[]
        for bone,core in ((11,self.core_idx['LeftToeBase']),(14,self.core_idx['RightToeBase'])):
            selected=self.weights[:,bone]>=.5
            if not np.any(selected):continue
            sole=selected&(self.vertices[:,1]<=self.vertices[selected,1].min()+.015)
            ids=np.flatnonzero(sole)
            if len(ids)>128:ids=ids[np.linspace(0,len(ids)-1,128).astype(int)]
            toe=source[:,core]
            velocity=np.linalg.norm(np.gradient(toe,axis=0)*fps,axis=-1) if len(toe)>1 else np.zeros(len(toe))
            supports=np.asarray([self._support_height(source_geometry,row[0],row[2],row[1])
                                 for row in toe],dtype=object)
            native_clearance=np.asarray([row[1]-height if height is not None else np.inf
                                         for row,height in zip(toe,supports)])
            finite=native_clearance[np.isfinite(native_clearance)]
            if not len(finite):continue
            baseline=np.quantile(finite,.1)
            grounded=(native_clearance<=baseline+.025)&(velocity<.35)&(native_clearance<.15)
            for frame in np.flatnonzero(grounded):
                world=self._sole_vertices(ids,positions[frame],rotations[frame])
                local=[row[1]-height for row in world
                       if (height:=self._support_height(render_geometry,row[0],row[2],row[1])) is not None]
                if local:clearances.append(float(min(local)))
        if len(clearances)<3:
            return 0.,{'applied':False,'offset_m':0.,'reason':'insufficient terrain-relative support samples'}
        reference=float(np.quantile(clearances,.1))
        shift=float(np.clip(-reference,-.15,.15))
        return shift,{'applied':True,'offset_m':shift,'support_sample_count':len(clearances),
            'reference_support_clearance_m':reference,
            'method':'single clip-wide terrain-relative sole translation; no frame snapping'}

    def _floor_calibration(self, source, positions, rotations, floor_y, fps):
        """One disclosed clip-wide translation estimated only from support feet.

        No frame-by-frame snapping: airborne heights, vertical excursions,
        joint articulation and timing remain exactly unchanged by this step.
        A clip without identifiable low, slowly moving feet is left untouched.
        """
        support_heights=[]
        for bone,core in ((11,self.core_idx['LeftToeBase']),(14,self.core_idx['RightToeBase'])):
            selected=self.weights[:,bone]>=.5
            if not np.any(selected):continue
            sole=selected&(self.vertices[:,1]<=self.vertices[selected,1].min()+.015)
            ids=np.flatnonzero(sole)
            if len(ids)>128:ids=ids[np.linspace(0,len(ids)-1,128).astype(int)]
            si=self.skin_indices[ids];sw=self.skin_weights[ids]
            local=self.vertices[ids,None]-self.rest[si]
            toe=source[:,core]
            velocity=np.linalg.norm(np.gradient(toe,axis=0)*fps,axis=-1) if len(toe)>1 else np.zeros(len(toe))
            baseline=np.quantile(toe[:,1],.1)
            grounded=(toe[:,1]<=baseline+.025)&(velocity<.35)&(toe[:,1]<.15)
            for frame in np.flatnonzero(grounded):
                world=np.sum((np.einsum('vwij,vwj->vwi',rotations[frame,si],local)+positions[frame,si])*sw[...,None],axis=1)
                support_heights.append(float(np.min(world[:,1])))
        if len(support_heights)<3:
            return 0.,{'applied':False,'offset_m':0.,'reason':'insufficient grounded native support samples'}
        reference=float(np.quantile(support_heights,.10))
        shift=float(np.clip(floor_y-reference,-.15,.15))
        return shift,{'applied':True,'offset_m':shift,'floor_y':floor_y,
            'support_sample_count':len(support_heights),'support_height_quantile':.10,
            'reference_support_height_m':reference,
            'method':'single clip-wide translation from robust low slowly-moving sole support; no frame snapping'}

    def deform_vertices(self, positions, rotations):
        """Sparse CPU equivalent of the browser skin, for exact visual QA."""
        p,r=np.asarray(positions),np.asarray(rotations)
        if p.shape!=(17,3) or r.shape!=(17,3,3):
            raise ValueError('Expected fitted Human17 transforms')
        i=self.skin_indices
        local=self.vertices[:,None]-self.rest[i]
        transformed=np.einsum('vwij,vwj->vwi',r[i],local)+p[i]
        return np.sum(transformed*self.skin_weights[...,None],axis=1)
