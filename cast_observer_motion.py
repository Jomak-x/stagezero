"""Bounded authored observer motion on otherwise stationary native22 tracks.

This is a display refinement, not model output, a contact solve, or an idle
motion model. Active motion and the entire lower body remain exact. Callers
must rerun scene/body clearance checks and watch complete performances.
"""
from __future__ import annotations

import math

import numpy as np


PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
LOWER_BODY = (0, 1, 2, 4, 5, 7, 8, 10, 11)
UPPER_BODY = (3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
CHEST_TREE = (9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
_UP = np.array([0., 1., 0.])


def _unit(vector):
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-10 else None


def _rotate(pose, indices, pivot, axis, angle):
    """Rigidly rotate a complete subtree, preserving all incident bone lengths."""
    axis = _unit(np.asarray(axis, dtype=float))
    if axis is None or abs(angle) < 1e-12:
        return
    relative = pose[list(indices)] - pose[pivot]
    c, s = math.cos(angle), math.sin(angle)
    pose[list(indices)] = (pose[pivot] + relative*c + np.cross(axis, relative)*s
                           + np.outer(relative @ axis, axis)*(1-c))


def _toward(pose, indices, pivot, joint, direction, fraction, limit):
    current, desired = _unit(pose[joint]-pose[pivot]), _unit(np.asarray(direction))
    if current is None or desired is None:
        return
    angle = math.acos(float(np.clip(current @ desired, -1., 1.)))
    axis = np.cross(current, desired)
    if np.linalg.norm(axis) < 1e-9:
        if angle < 1e-9:
            return
        # An exact opposite direction has no unique axis. Choose a stable one.
        axis = np.cross(current, np.eye(3)[int(np.argmin(abs(current)))])
    _rotate(pose, indices, pivot, axis, min(angle, limit)*fraction)


def _smooth(value):
    t = np.clip(value, 0., 1.)
    return t*t*t*(t*(t*6-15)+10)


def _validate(joints, actor_ids, activities, fps, seed):
    p = np.asarray(joints)
    ids = tuple(actor_ids)
    if (p.ndim != 4 or p.shape[1:] != (len(ids), 22, 3) or len(p) < 1
            or not 1 <= len(ids) <= 3 or p.dtype.kind != 'f' or not np.isfinite(p).all()):
        raise ValueError('Observer motion requires finite floating native22 cast joints [T,N,22,3]')
    if (any(not isinstance(a, str) or not a for a in ids) or len(set(ids)) != len(ids)):
        raise ValueError('Observer actor IDs must be unique nonempty strings')
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or not 1 <= fps <= 240:
        raise ValueError('Observer fps must be finite and between 1 and 240')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Observer seed must be uint32')
    if not isinstance(activities, (list, tuple)) or not activities:
        raise ValueError('Observer activity must cover the entire timeline')
    mask = np.zeros((len(p), len(ids)), dtype=bool)
    cursor = 0
    for activity in activities:
        if not isinstance(activity, dict):
            raise ValueError('Observer activity must contain records')
        start, end = activity.get('start_frame'), activity.get('end_frame_exclusive')
        active = activity.get('active_actor_ids')
        if (type(start) is not int or type(end) is not int or start != cursor
                or not start < end <= len(p) or not isinstance(active, (list, tuple))
                or not active or any(not isinstance(a, str) or a not in ids for a in active)
                or len(set(active)) != len(active)):
            raise ValueError('Observer activity must contiguously cover frames with known active IDs')
        mask[start:end, [ids.index(a) for a in active]] = True
        controls = activity.get('observer_control_spans', [])
        if not isinstance(controls, list):
            raise ValueError('Observer controller spans must be a list')
        for span in controls:
            if (not isinstance(span, dict) or span.get('actor_id') not in ids
                    or type(span.get('start_frame')) is not int
                    or type(span.get('end_frame_exclusive')) is not int
                    or not start <= span['start_frame'] < span['end_frame_exclusive'] <= end):
                raise ValueError('Observer controller span is outside its activity interval')
            mask[span['start_frame']:span['end_frame_exclusive'], ids.index(span['actor_id'])] = True
        cursor = end
    if cursor != len(p):
        raise ValueError('Observer activity must cover the entire timeline')
    return p, ids, mask


def refine_observers(joints, actor_ids, activities, *, fps=30, seed=42):
    """Return a copy with deterministic upper-body motion in inactive spans.

    Each complete inactive span receives one smooth envelope, independent of
    source segmentation. The first two samples stay exact. The last two samples
    stay exact before reentry; a terminal observer keeps its relaxed arm posture.
    Only stationary source spans are refined: moving inactive tracks belong to
    their own controller and are passed through unchanged.
    """
    source, ids, active = _validate(joints, actor_ids, activities, fps, seed)
    out = source.copy()
    rng = np.random.default_rng(seed)
    records = []
    for actor_index, aid in enumerate(ids):
        phase = rng.uniform(0., 2*math.pi, 3)
        frequency = rng.uniform([.16, .29, .095], [.21, .37, .13])
        edges = np.flatnonzero(np.diff(np.r_[False, ~active[:, actor_index], False]))
        for start, end in zip(edges[::2], edges[1::2]):
            start, end = int(start), int(end)
            count, reentry = end-start, end < len(source)
            original = source[start:end, actor_index]
            record = {'actor_id': aid, 'start_frame': start, 'end_frame_exclusive': end,
                      'reenters_active_motion': reentry, 'applied_frames': 0}
            records.append(record)
            if count < max(9, round(.3*fps)):
                record['skipped'] = 'inactive span too short for a smooth observer gesture'
                continue
            if float(np.max(abs(original-original[:1]))) > 1e-6:
                record['skipped'] = 'inactive track already contains motion; preserve its controller'
                continue

            # Focus tracks the current active group. Smooth wrapped yaw by
            # filtering unit vectors, avoiding jumps when the active pair changes.
            root = original[0, 0]
            across = source[start, actor_index, 1]-source[start, actor_index, 2]
            across[1] = 0.
            right = _unit(across)
            if right is None:
                record['skipped'] = 'ambiguous horizontal hip axis'
                continue
            forward = np.cross(right, _UP)
            focus = []
            for frame in range(start, end):
                target = source[frame, active[frame], 0].mean(axis=0)-root
                target[1] = 0.
                direction = _unit(target)
                focus.append(forward if direction is None else direction)
            focus = np.asarray(focus)
            radius = max(1, round(.35*fps))
            padded = np.pad(focus, ((radius, radius), (0, 0)), mode='edge')
            focus = np.stack([padded[i:i+2*radius+1].mean(axis=0) for i in range(count)])
            # Use a bounded frontal cone: raw wrapped yaw flips sign when a
            # partner crosses directly behind, creating a needless head snap.
            yaw = np.arctan2(focus @ right, np.maximum(focus @ forward, .35))
            # A person behind the observer should not cause a 180-degree twist.
            yaw = np.clip(yaw, -math.radians(12), math.radians(12))
            envelope = _smooth((np.arange(count)-1)/(1.15*fps))
            if reentry:
                envelope *= _smooth((count-2-np.arange(count))/(.85*fps))
            envelope[:2] = 0.
            if reentry:
                envelope[-2:] = 0.
            t = np.arange(count)/fps

            for local, weight in enumerate(envelope):
                if weight == 0:
                    continue
                pose = original[local].astype(float, copy=True)
                torso = _unit(pose[9]-pose[0])
                if torso is not None:
                    angle = math.acos(float(np.clip(torso @ _UP, -1., 1.)))
                    # Gradual partial recovery from the selected incoming lean.
                    _rotate(pose, UPPER_BODY, 0, np.cross(torso, _UP),
                            min(angle*.70, math.radians(12))*weight)
                sway = (math.radians(.75)*math.sin(2*math.pi*frequency[0]*t[local]+phase[0])
                        + math.radians(.25)*math.sin(2*math.pi*frequency[1]*t[local]+phase[1]))
                breath = math.radians(.40)*math.sin(2*math.pi*frequency[2]*t[local]+phase[2])
                _rotate(pose, UPPER_BODY, 0, forward, sway*weight)
                _rotate(pose, UPPER_BODY, 0, right, breath*weight)
                _rotate(pose, CHEST_TREE, 6, _UP, float(yaw[local])*weight)

                # Blend shoulder and elbow directions, never Cartesian joint
                # positions. A released greeting arm settles beside the body.
                shoulder_right = pose[16]-pose[17]
                shoulder_right[1] = 0.
                lateral = _unit(shoulder_right)
                lateral = right if lateral is None else lateral
                facing = np.cross(lateral, _UP)
                for side, shoulder, elbow, wrist in ((1, 16, 18, 20), (-1, 17, 19, 21)):
                    upper_target = -_UP + side*.12*lateral + .07*facing
                    lower_target = -_UP + side*.035*lateral + .17*facing
                    _toward(pose, (elbow, wrist), shoulder, elbow, upper_target,
                            weight*.92, math.radians(100))
                    _toward(pose, (wrist,), elbow, wrist, lower_target,
                            weight*.92, math.radians(100))
                nod = math.radians(.8)*math.sin(2*math.pi*.23*t[local]+phase[1])
                _rotate(pose, (15,), 12, lateral, nod*weight)
                out[start+local, actor_index] = pose
                record['applied_frames'] += 1
            record['maximum_joint_correction_m'] = float(np.linalg.norm(out[start:end, actor_index]-original, axis=-1).max())
            record['maximum_attention_yaw_degrees'] = float(np.degrees(abs(yaw*envelope)).max())

    lengths_before = np.linalg.norm(source[:, :, 1:]-source[:, :, np.asarray(PARENTS[1:])], axis=-1)
    lengths_after = np.linalg.norm(out[:, :, 1:]-out[:, :, np.asarray(PARENTS[1:])], axis=-1)
    report = {
        'source': 'authored bounded observer continuation', 'model_generated': False,
        'version': 1, 'seed': seed, 'fps': fps, 'spans': records,
        'method': 'whole inactive span easing; length-preserving torso/arm rotations; attention to active roots',
        'active_frames_modified': False, 'lower_body_modified': False,
        'source_pair_frames_modified': False, 'physical_contact_verified': False,
        'visual_acceptance': 'unverified',
        'applied_actor_frames': sum(record['applied_frames'] for record in records),
        'maximum_joint_correction_m': float(np.linalg.norm(out-source, axis=-1).max()),
        'maximum_bone_length_error_m': float(abs(lengths_after-lengths_before).max()),
        'limits': {'upright_correction_degrees': 12, 'attention_yaw_degrees': 12,
                   'arm_rotation_degrees_per_joint': 92, 'preserved_boundary_frames': 2},
        'limitations': ['Authored observation, not generated contextual acting or a learned idle model.',
                       'No stepping or lower-body weight transfer; existing foot positions remain exact.',
                       'No finger, eye or palm contact solve; positional head motion is not eye gaze.',
                       'Arm relaxation can change body clearance; caller must check final scene and cast geometry.',
                       'Exact reentry endpoints can require anticipatory return toward the incoming pose.'],
    }
    return out, report
