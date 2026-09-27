"""Bounded display-only cast refinement; all native interaction frames survive.

Observer motion is authored, not model output. Ankle planting is a low/slow
heuristic. Scene gates and sampled inter-body clearance must be checked before
publication; none of these measurements is mesh contact or physics proof.
"""
from copy import deepcopy
import numpy as np
from interaction_scene_collision import _body_proxies
from native_pair_transition import stabilize_authored_feet


class RefinementRejected(ValueError):
    """Rejected display candidate retained for exact failure reproduction."""
    def __init__(self, joints, report):
        super().__init__('Authored cast refinement worsens unintended sampled body overlap')
        self.candidate_joints = joints
        self.refinement_report = report


def cast_body_clearance(joints, actor_ids, activities):
    """Per-frame minimum sampled sphere clearance for unintended actor pairs."""
    points = np.asarray(joints)
    proxies = [_body_proxies(points[:, index], 'native22') for index in range(len(actor_ids))]
    values = {}
    for a in range(len(actor_ids)):
        for b in range(a + 1, len(actor_ids)):
            pa, ra = proxies[a]; pb, rb = proxies[b]
            clearance = (np.linalg.norm(pa[:, :, None] - pb[:, None, :], axis=-1)
                         - ra[None, :, None] - rb[None, None, :]).min(axis=(1, 2))
            intended = np.zeros(len(points), dtype=bool)
            for activity in activities:
                if set(activity.get('contact_actor_ids', ())) == {actor_ids[a], actor_ids[b]}:
                    intended[activity['start_frame']:activity['end_frame_exclusive']] = True
            values[(a, b)] = (clearance, ~intended)
    return values


def refine_cast_motion(joints, actor_ids, segments, activities, *, fps=30, seed=42,
                       observers=True, feet=False):
    from cast_observer_motion import refine_observers
    from cast_performance import validated_segments
    original = np.asarray(joints)
    if not isinstance(segments, list) or not isinstance(activities, (list, tuple)) or len(segments) != len(activities):
        raise ValueError('Refinement needs matching source segments and actor activity')
    validated_segments({'segments': segments}, len(original))
    for segment, activity in zip(segments, activities):
        if any(segment[key] != activity.get(key) for key in ('start_frame', 'end_frame_exclusive')):
            raise ValueError('Refinement source and actor activity boundaries disagree')
    out = original.copy()
    observer_report = {'enabled': False}
    if observers:
        out, observer_report = refine_observers(out, actor_ids, activities, fps=fps, seed=seed)
    plant_reports = []
    if feet:
        for index, aid in enumerate(actor_ids):
            tagged = deepcopy(segments)
            for segment, activity in zip(tagged, activities):
                if aid not in activity['active_actor_ids']:
                    segment['source'] = 'observer_hold'
            duplicate = np.repeat(out[:, index:index+1], 2, axis=1)
            planted, report = stabilize_authored_feet(duplicate, tagged, fps=fps)
            out[:, index] = planted[:, 0]
            report['actor_id'] = aid
            report['intervals'] = [dict(item, actor=aid) for item in report['intervals'] if item['actor'] == 0]
            report['applied_foot_frames'] //= 2
            report['unreachable_foot_frames_skipped'] //= 2
            plant_reports.append(report)
    for segment, activity in zip(segments, activities):
        if segment['source'] == 'intergen':
            lo, hi = segment['start_frame'], segment['end_frame_exclusive']
            indices = [actor_ids.index(aid) for aid in activity['active_actor_ids']]
            if not np.array_equal(original[lo:hi, indices], out[lo:hi, indices]):
                raise ValueError('Cast refinement changed a native interaction frame')
    before = cast_body_clearance(original, actor_ids, activities)
    after = cast_body_clearance(out, actor_ids, activities)
    clearance_reports = []
    for pair, (new, mask) in after.items():
        old = before[pair][0]
        worsened = mask & (new < 0) & (new < old - 1e-4)
        record = {'actor_ids': [actor_ids[i] for i in pair],
                  'unintended_frames': int(mask.sum()),
                  'baseline_overlap_frames': int(np.count_nonzero(mask & (old < 0))),
                  'refined_overlap_frames': int(np.count_nonzero(mask & (new < 0))),
                  'worsened_overlap_frames': int(worsened.sum()),
                  'minimum_baseline_clearance_m': float(old[mask].min()) if mask.any() else None,
                  'minimum_refined_clearance_m': float(new[mask].min()) if mask.any() else None}
        clearance_reports.append(record)
    report = {'version': 1, 'observer_motion': observer_report, 'foot_planting': plant_reports,
                 'body_clearance': clearance_reports, 'body_clearance_proxy': 'named joint and limb midpoint spheres, all frames; intended contact excluded',
                 'source_pair_frames_modified': False, 'root_positions_modified': False,
                 'physical_contact_verified': False, 'visual_acceptance': 'unverified'}
    if any(record['worsened_overlap_frames'] for record in clearance_reports):
        raise RefinementRejected(out, report)
    return out, report
