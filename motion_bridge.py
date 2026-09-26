"""Geometric InterGen SMPL22 -> official ARDY Core27 bridge (Y-up metres).

This is retargeting, not motion generation. Missing hand/finger orientations are
inherited from the forearm. Both actors share one anatomical scale and world
placement; limb IK fits each actor's OWN source endpoints, never pair contacts.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# InterGen's first 66 features are the standard SMPL body joint order (22).
SMPL22_NAMES = (
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
)
SMPL22_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
DIRECT_MAP = {
    'Hips': 'pelvis', 'Spine3': 'spine3', 'Neck': 'neck', 'Head': 'head',
    **{f'{side}{core}': f'{side.lower()}_{smpl}'
       for side in ('Left', 'Right')
       for core, smpl in [('Shoulder', 'collar'), ('Arm', 'shoulder'),
                          ('ForeArm', 'elbow'), ('Hand', 'wrist'),
                          ('UpLeg', 'hip'), ('Leg', 'knee'),
                          ('Foot', 'ankle'), ('ToeBase', 'foot')]},
}


@lru_cache(maxsize=1)
def core_skeleton():
    """Load hierarchy and neutral offsets from the vendored authoritative asset."""
    vendor = str(Path(__file__).resolve().parent / 'vendor' / 'ardy')
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from ardy.skeleton import CoreSkeleton27
    return CoreSkeleton27()


def _layout():
    skel = core_skeleton()
    return (list(skel.bone_order_names), skel.joint_parents.cpu().numpy(),
            skel.neutral_joints.cpu().numpy().astype(np.float64))


def _unit(v, fallback=None):
    n = np.linalg.norm(v)
    if n > 1e-9:
        return v / n
    if fallback is None:
        raise ValueError('degenerate source skeleton: zero-length anatomical direction')
    return np.asarray(fallback, dtype=np.float64)


def _swing(a, b):
    """Shortest proper rotation including deterministic antiparallel handling."""
    a, b = _unit(a), _unit(b)
    cross = np.cross(a, b)
    c = np.clip(np.dot(a, b), -1., 1.)
    sn = np.linalg.norm(cross)
    if sn < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = _unit(np.cross(a, np.eye(3)[np.argmin(np.abs(a))]))
        return Rotation.from_rotvec(axis * np.pi).as_matrix()
    return Rotation.from_rotvec(cross / sn * np.arctan2(sn, c)).as_matrix()


def _fit_rotation(offsets, directions, reference):
    valid = np.linalg.norm(directions, axis=-1) > 1e-8
    a, b = offsets[valid], directions[valid]
    if not len(a):
        return reference.copy()
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    if len(a) == 1 or np.linalg.matrix_rank(a, tol=1e-6) == 1:
        return _swing(reference @ a[0], b[0]) @ reference
    u, _, vt = np.linalg.svd(b.T @ a)
    fix = np.eye(3)
    fix[-1, -1] = np.linalg.det(u @ vt)
    return u @ fix @ vt


def _two_bone(start, target, pole, length1, length2, reference):
    delta = target - start
    distance = np.linalg.norm(delta)
    direction = _unit(delta, _unit(reference))
    reachable = np.clip(distance, abs(length1 - length2) + 1e-7, length1 + length2 - 1e-7)
    x = (length1**2 - length2**2 + reachable**2) / (2 * reachable)
    height = np.sqrt(max(0., length1**2 - x*x))
    bend = pole - start
    bend -= np.dot(bend, direction) * direction
    if np.linalg.norm(bend) < 1e-7:
        bend = reference - np.dot(reference, direction) * direction
    if np.linalg.norm(bend) < 1e-7:
        axis = np.eye(3)[np.argmin(np.abs(direction))]
        bend = axis - np.dot(axis, direction) * direction
    middle = start + x * direction + height * _unit(bend)
    return middle, start + reachable * direction


def _fk_numpy(local, roots, parents, offsets):
    positions = np.empty(local.shape[:-2] + (3,), dtype=np.float64)
    glob = np.empty_like(local)
    positions[..., 0, :] = roots
    glob[..., 0, :, :] = local[..., 0, :, :]
    for j in range(1, len(parents)):
        p = parents[j]
        positions[..., j, :] = positions[..., p, :] + np.einsum('...ij,j->...i', glob[..., p, :, :], offsets[j])
        glob[..., j, :, :] = glob[..., p, :, :] @ local[..., j, :, :]
    return positions, glob


def _resample(values, old_times, new_times):
    flat = values.reshape(len(values), -1)
    return np.stack([np.interp(new_times, old_times, x) for x in flat.T], axis=-1).reshape((len(new_times),) + values.shape[1:])


def _summary(values):
    x = np.asarray(values)
    return {'mean_m': float(np.mean(x)), 'p95_m': float(np.percentile(x, 95)), 'max_m': float(np.max(x))}


def retarget_intergen_pair(joints, source_fps=30, target_fps=20):
    """Return positions[2,T,27,3], global rotations[2,T,27,3,3], metadata.

    Samples cover timestamps 0 .. (source_frames-1)/source_fps at a uniform
    target rate; no duplicated endpoint or time stretching. SO(3) interpolation
    operates on local rotations and positions are rebuilt by fixed-length FK.
    Source positions alone cannot recover axial twist or finger articulation.
    """
    source = np.asarray(joints, dtype=np.float64)
    if source.ndim != 4 or source.shape[1:] != (2, 22, 3) or len(source) < 2:
        raise ValueError('joints must have shape [T>=2,2,22,3]')
    if not np.isfinite(source).all():
        raise ValueError('joints must be finite')
    if not np.isfinite([source_fps, target_fps]).all() or min(source_fps, target_fps) <= 0:
        raise ValueError('frame rates must be positive and finite')
    names, parents, neutral = _layout()
    idx = {n: i for i, n in enumerate(names)}
    src = {n: i for i, n in enumerate(SMPL22_NAMES)}
    offsets = neutral - neutral[np.maximum(parents, 0)]
    # Pool both actors and all frames equally. Long limb lengths are more stable
    # than small collars/fingers and match the body support/reach scale.
    pairs = [(f'{side}{a}', f'{side}{b}') for side in ('Left', 'Right')
             for a, b in [('UpLeg', 'Leg'), ('Leg', 'Foot'), ('Arm', 'ForeArm'), ('ForeArm', 'Hand')]]
    lengths, target_lengths = [], []
    for a, b in pairs:
        lengths.append(np.median(np.linalg.norm(source[:, :, src[DIRECT_MAP[b]]] - source[:, :, src[DIRECT_MAP[a]]], axis=-1)))
        target_lengths.append(np.linalg.norm(neutral[idx[b]] - neutral[idx[a]]))
    lengths = np.asarray(lengths)
    if np.min(lengths) < .025:
        raise ValueError('degenerate source limbs; expected Y-up metre SMPL22 joints')
    target_lengths = np.asarray(target_lengths)
    scale = float(lengths @ target_lengths / (lengths @ lengths))
    scaled = source * scale
    guides = np.zeros((len(source), 2, 27, 3))
    for name, source_name in DIRECT_MAP.items():
        guides[:, :, idx[name]] = scaled[:, :, src[source_name]]
    # Core adds a fourth spine link. Interpolate the source spine polyline by
    # neutral Core arc fractions, without moving the source pelvis/chest.
    core_spine = [idx[n] for n in ('Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3')]
    fractions = np.cumsum([0.] + [np.linalg.norm(offsets[j]) for j in core_spine[1:]])
    fractions /= fractions[-1]
    path = scaled[:, :, [src[n] for n in ('pelvis', 'spine1', 'spine2', 'spine3')]]
    for t in range(len(source)):
        for actor in range(2):
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path[t, actor], axis=0), axis=-1))]
            if arc[-1] < 1e-5:
                raise ValueError('degenerate source spine')
            guides[t, actor, core_spine] = _resample(path[t, actor], arc / arc[-1], fractions)
    children = [np.flatnonzero(parents == j) for j in range(27)]
    limb_starts = {idx[f'{side}{part}']: (idx[f'{side}{mid}'], idx[f'{side}{end}'])
                   for side in ('Left', 'Right')
                   for part, mid, end in [('Arm', 'ForeArm', 'Hand'), ('UpLeg', 'Leg', 'Foot')]}
    missing = {idx[n] for n in names if n.endswith(('HandEnd', 'HandThumb1'))}
    local = np.empty((len(source), 2, 27, 3, 3))
    for t in range(len(source)):
        for actor in range(2):
            g = np.empty((27, 3, 3)); p = np.empty((27, 3)); p[0] = guides[t, actor, 0]
            overrides = {}
            for j in range(27):
                par = parents[j]
                reference = g[par] if par >= 0 else (local[t-1, actor, 0] if t else np.eye(3))
                if par >= 0:
                    p[j] = p[par] + g[par] @ offsets[j]
                if j in limb_starts:
                    middle, end = limb_starts[j]
                    m, e = _two_bone(p[j], guides[t, actor, end], guides[t, actor, middle],
                                     np.linalg.norm(offsets[middle]), np.linalg.norm(offsets[end]),
                                     reference @ offsets[middle])
                    overrides[j] = (middle, m)
                    overrides[middle] = (end, e)
                if j in overrides:
                    child, position = overrides[j]
                    g[j] = _fit_rotation(offsets[[child]], (position-p[j])[None], reference)
                else:
                    ch = [c for c in children[j] if c not in missing]
                    # Terminal hands inherit forearm orientation; they are not
                    # inferred from contacts or assigned artificial palm normals.
                    if ch:
                        directions = guides[t, actor, ch] - guides[t, actor, j]
                        g[j] = _fit_rotation(offsets[ch], directions, reference)
                    else:
                        g[j] = reference.copy()
                local[t, actor, j] = g[j] if par < 0 else g[par].T @ g[j]
    old_times = np.arange(len(source), dtype=np.float64) / source_fps
    new_times = np.arange(int(np.floor(old_times[-1]*target_fps + 1e-8)) + 1) / target_fps
    sampled_local = np.empty((len(new_times), 2, 27, 3, 3))
    for actor in range(2):
        for joint in range(27):
            sampled_local[:, actor, joint] = Slerp(old_times, Rotation.from_matrix(local[:, actor, joint]))(new_times).as_matrix()
    roots = _resample(scaled[:, :, 0], old_times, new_times)
    positions, rotations = _fk_numpy(sampled_local, roots, parents, offsets)
    # Audit the actual float32 arrays that callers receive.
    positions = positions.astype(np.float32).astype(np.float64)
    rotations = rotations.astype(np.float32).astype(np.float64)
    target_source = _resample(scaled, old_times, new_times)
    hands = [idx['LeftHand'], idx['RightHand']]
    source_hands = [src['left_wrist'], src['right_wrist']]
    feet = [idx['LeftFoot'], idx['RightFoot'], idx['LeftToeBase'], idx['RightToeBase']]
    source_feet = [src['left_ankle'], src['right_ankle'], src['left_foot'], src['right_foot']]
    hand_error = np.linalg.norm(positions[:, :, hands] - target_source[:, :, source_hands], axis=-1)
    foot_error = np.linalg.norm(positions[:, :, feet] - target_source[:, :, source_feet], axis=-1)
    # Pairwise wrist distance change records relationship/contact drift explicitly.
    source_wrists = target_source[:, :, source_hands]
    target_wrists = positions[:, :, hands]
    before = np.linalg.norm(source_wrists[:, 0, :, None] - source_wrists[:, 1, None, :], axis=-1)
    after = np.linalg.norm(target_wrists[:, 0, :, None] - target_wrists[:, 1, None, :], axis=-1)
    foot_speed = np.linalg.norm(np.diff(positions[:, :, feet][..., [0, 2]], axis=0), axis=-1) * target_fps
    source_speed = np.linalg.norm(np.diff(target_source[:, :, source_feet][..., [0, 2]], axis=0), axis=-1) * target_fps
    planted = (source_speed < .1) & (target_source[:-1, :, source_feet][..., 1] < .15)
    slide = foot_speed[planted]
    close_wrists = before < .12
    bone_errors = np.abs(np.linalg.norm(positions[:, :, 1:] - positions[:, :, parents[1:]], axis=-1) - np.linalg.norm(offsets[1:], axis=-1))
    metadata = {
        'method': 'SMPL22_named_mapping_shared_scale_limb_IK_local_SO3_FK_v1',
        'source_fps': float(source_fps), 'fps': float(target_fps), 'source_frames': len(source),
        'frames': len(new_times), 'common_scale': scale,
        'scale_method': 'pooled median bilateral upper/lower arm/leg lengths, least-squares common scale',
        'placement': {'rotation': np.eye(3).tolist(), 'translation_m': [0., 0., 0.], 'shared_by_both_actors': True},
        'skeleton': 'core27', 'joint_names': names, 'source_joint_names': list(SMPL22_NAMES),
        'joint_mapping': DIRECT_MAP.copy(),
        'metrics': {'wrist_target_error': _summary(hand_error), 'foot_target_error': _summary(foot_error),
                    'pair_wrist_distance_drift': _summary(np.abs(after-before)),
                    'source_close_wrist_threshold_m': .12,
                    'source_close_wrist_pair_samples': int(close_wrists.sum()),
                    'close_wrist_distance_drift': _summary(np.abs(after-before)[close_wrists]) if close_wrists.any() else None,
                    'retarget_close_wrist_distance': _summary(after[close_wrists]) if close_wrists.any() else None,
                    'bone_length_error': _summary(bone_errors),
                    'root_shared_transform_max_error_m': float(np.abs(positions[:, :, 0]-target_source[:, :, 0]).max()),
                    'pair_root_vector_max_error_m': float(np.abs((positions[:, 1, 0]-positions[:, 0, 0])-(target_source[:, 1, 0]-target_source[:, 0, 0])).max()),
                    'source_planted_foot_samples': int(planted.sum()),
                    'source_planted_definition': 'source foot y < 0.15m and planar speed < 0.10m/s after shared scale',
                    'source_planted_foot_speed_mean_mps': float(source_speed[planted].mean()) if planted.any() else None,
                    'retarget_planted_foot_speed_mean_mps': float(slide.mean()) if len(slide) else None,
                    'retarget_planted_foot_speed_p95_mps': float(np.percentile(slide,95)) if len(slide) else None,
                    'minimum_foot_y_m': float(positions[:, :, feet][..., 1].min())},
        'limitations': ['Position-only input cannot identify axial twist, palm normals, or finger articulation.',
                       'Core hand ends/thumbs inherit forearm orientation; source supplies wrists only.',
                       'Fixed target proportions and unreachable endpoints may change contact distances.',
                       'No actor-specific root correction, floor snapping, contact locking, or mesh collision solve.',
                       'SO3 interpolation preserves bone lengths; it does not low-pass filter source jitter.'],
    }
    return positions.transpose(1,0,2,3).astype(np.float32), rotations.transpose(1,0,2,3,4).astype(np.float32), metadata


def bridge_to_history(model, positions, rotations, history_frames=None):
    """Encode canonical posed joints through the model's official representation.

    Returns NORMALIZED float32 NumPy features [actors,frames,330]. The model's
    loaded normalization stats are mandatory. No per-actor canonicalization is
    applied. History slicing is after encoding, so boundary velocities use the
    full clip. This does not assert that the model will continue it seamlessly.
    """
    import torch
    core_skeleton()
    from ardy.skeleton.transforms import global_rots_to_local_rots
    from ardy.skeleton.kinematics import fk
    rep = model.motion_rep
    if rep.motion_rep_dim != 330 or rep.fps != 20 or rep.skeleton.nbjoints != 27:
        raise ValueError('history bridge requires official Core27 330-feature representation at 20 fps')
    if list(rep.skeleton.bone_order_names) != _layout()[0]:
        raise ValueError('history bridge requires authoritative Core27 joint ordering')
    if not hasattr(rep, 'stats') or not rep.stats.is_loaded():
        raise ValueError('history bridge requires loaded model normalization statistics')
    p, r = np.asarray(positions), np.asarray(rotations)
    if p.ndim != 4 or p.shape[2:] != (27,3) or r.shape != p.shape[:3] + (3,3) or p.shape[1] < 2:
        raise ValueError('expected positions[A,T>=2,27,3] and global rotations[A,T,27,3,3]')
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError('history poses must be finite')
    if not np.allclose(r.swapaxes(-1,-2) @ r, np.eye(3), atol=2e-4) or not np.allclose(np.linalg.det(r), 1., atol=2e-4):
        raise ValueError('history rotations must be proper SO(3) matrices')
    if history_frames is not None and (isinstance(history_frames, bool) or not isinstance(history_frames, int) or not 1 <= history_frames <= p.shape[1]):
        raise ValueError('history_frames must be an integer within the clip')
    device = rep.skeleton.neutral_joints.device
    with torch.inference_mode():
        roots = torch.tensor(p[:,:,0], dtype=torch.float32, device=device)
        global_rots = torch.tensor(r, dtype=torch.float32, device=device)
        local = global_rots_to_local_rots(global_rots, rep.skeleton)
        _, fk_positions, _ = fk(local, roots, rep.skeleton)
        if not np.allclose(fk_positions.cpu().numpy(), p, atol=2e-4):
            raise ValueError('positions/rotations are inconsistent with the model Core27 neutral skeleton')
        lengths = torch.full((p.shape[0],), p.shape[1], dtype=torch.long, device=device)
        features = rep(local, roots, to_normalize=True, to_canonicalize=False, lengths=lengths)
        result = features.detach().cpu().numpy().astype(np.float32)
    if result.shape != p.shape[:2] + (330,) or not np.isfinite(result).all():
        raise ValueError('official representation returned invalid features')
    return result[:, -history_frames:] if history_frames is not None else result
