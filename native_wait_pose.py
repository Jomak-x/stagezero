"""Select an exact source stance for initial waiting; never solve ground contact.

Foot markers describe support only approximately. An ankle is above the sole,
so ankle and toe heights are not interchangeable floor-contact measurements.
"""
from __future__ import annotations

import numpy as np


def select_native_wait_pose(joints, actor_index):
    """Prefer stable bilateral support; upper-body relaxation owns raised arms.

    The entire selected pose is one unmodified source frame. Selection cannot
    manufacture a grounded stance when the source contains none. In particular,
    native toe-height quantiles are not the scene's floor or rendered soles.
    """
    native = np.asarray(joints)
    if (native.ndim != 4 or native.shape[1:] != (2, 22, 3) or len(native) < 4
            or actor_index not in (0, 1) or not np.isfinite(native).all()):
        raise ValueError('Waiting pose selection requires finite native pair joints')
    actor = native[:, actor_index]
    torso = actor[:, 9]-actor[:, 0]
    angles = np.degrees(np.arctan2(np.linalg.norm(torso[:, [0, 2]], axis=1), torso[:, 1]))
    hands_above_hips = np.maximum(actor[:, [20, 21], 1]-actor[:, 0, None, 1], 0.).mean(axis=1)
    hand_reach = np.linalg.norm(actor[:, [20, 21]][:, :, [0, 2]]-actor[:, 0, None][:, :, [0, 2]], axis=2).mean(axis=1)
    feet = actor[:, [7, 8, 10, 11]]
    steps = np.linalg.norm(np.diff(feet[:, :, [0, 2]], axis=0)*30, axis=-1).max(axis=1)
    speeds = np.array([np.mean(steps[max(0, f-2):min(len(steps), f+3)]) for f in range(len(actor))])
    # Both actors share one native coordinate system. An actor-specific floor
    # would conceal an actor whose complete track is elevated above its partner.
    floor = float(np.quantile(native[:, :, [10, 11], 1], .05))
    toes, ankles = actor[:, [10, 11], 1], actor[:, [7, 8], 1]
    support = np.stack([actor[:, [7, 10], 1].min(axis=1), actor[:, [8, 11], 1].min(axis=1)], axis=1)
    foot_height = support.max(axis=1)-floor
    toe_height = np.maximum(toes.max(axis=1)-floor, 0.)
    toe_asymmetry = abs(toes[:, 0]-toes[:, 1])
    ankle_asymmetry = abs(ankles[:, 0]-ankles[:, 1])
    # Per-foot excess ankle height penalizes a heel rise without interpreting
    # normal ankle-to-sole anatomy as an error or imposing a flat foot vector.
    ankle_excess = np.maximum(ankles-np.quantile(ankles, .1, axis=0), 0.).max(axis=1)
    leg_length = np.stack([np.linalg.norm(actor[:, hip]-actor[:, knee], axis=1)
                           +np.linalg.norm(actor[:, knee]-actor[:, ankle], axis=1)
                           for hip, knee, ankle in ((1, 4, 7), (2, 5, 8))], axis=1).mean(axis=1)
    standing = (actor[:, 0, 1]-support.mean(axis=1))/np.maximum(leg_length, 1e-6)
    scores = (angles/20+toe_height*6+toe_asymmetry*8+ankle_asymmetry*8
              +ankle_excess*3+speeds*.5+hands_above_hips*.1+hand_reach*.1)
    gates = {'strong_torso_lean': angles <= 20,
             'feet_moving_in_source': speeds <= .35,
             'support_high_above_native_floor_proxy': toe_height <= .10,
             'crouched_or_degenerate_legs': (standing >= .70) & (leg_length > 1e-6)}
    eligible = np.logical_and.reduce(list(gates.values()))
    candidates = np.flatnonzero(eligible)
    best = int(candidates[np.argmin(scores[candidates])]) if len(candidates) else 0
    improvement = max(.10, float(scores[0])*.10)
    chosen = best
    reason = 'selected better supported exact source pose'
    if not len(candidates):
        reason = 'no eligible standing stable source pose; first-frame fallback is not verified suitable'
    elif best == 0:
        reason = 'first frame has the best eligible score'
    elif eligible[0] and scores[0]-scores[best] < improvement:
        chosen = 0
        reason = 'eligible first frame retained; improvement below threshold'
    # An ineligible baseline never blocks a valid candidate merely because a
    # weighted score happens to differ by less than the hysteresis threshold.

    def metrics(index):
        return {'score': float(scores[index]), 'torso_tilt_degrees': float(angles[index]),
                'mean_hand_height_above_hips_m': float(hands_above_hips[index]),
                'mean_hand_horizontal_reach_m': float(hand_reach[index]),
                'local_max_foot_xz_speed_mean_m_s': float(speeds[index]),
                'highest_support_above_native_floor_m': float(foot_height[index]),
                'standing_leg_extension_ratio': float(standing[index]),
                'highest_toe_above_native_floor_proxy_m': float(toe_height[index]),
                'toe_height_asymmetry_m': float(toe_asymmetry[index]),
                'ankle_height_asymmetry_m': float(ankle_asymmetry[index]),
                'maximum_ankle_height_excess_m': float(ankle_excess[index]),
                'toe_heights_native_m': toes[index].tolist(),
                'ankle_heights_native_m': ankles[index].tolist(),
                'eligible': bool(eligible[index]),
                'failed_gates': [name for name, passed in gates.items() if not passed[index]]}

    report = {'criterion': 'bilateral toe support, ankle/toe symmetry, upright standing and locally stable feet; hands have secondary weight without an eligibility veto',
        'source_frame': chosen, 'baseline_frame': 0, 'changed_from_first_frame': chosen != 0,
        'eligible_frame_count': int(len(candidates)), 'minimum_score_improvement': improvement,
        'baseline_metrics': metrics(0), 'selected_metrics': metrics(chosen),
        'selection_reason': reason,
        'ineligible_frame_counts': {name: int(np.count_nonzero(~passed)) for name, passed in gates.items()},
        'thresholds': {'maximum_torso_tilt_degrees': 20, 'maximum_local_foot_speed_m_s': .35,
                       'maximum_support_above_native_floor_m': .10, 'minimum_standing_leg_extension_ratio': .70},
        'score_weights': {'torso_tilt_degrees': 1/20, 'toe_height_m': 6, 'toe_asymmetry_m': 8,
                          'ankle_asymmetry_m': 8, 'ankle_height_excess_m': 3,
                          'local_foot_speed_m_s': .5, 'hand_height_m': .1, 'hand_reach_m': .1},
        'support_reference': {'method': 'shared native pair toe-height fifth percentile',
                              'native_floor_proxy_y_m': floor, 'scene_floor_known': False,
                              'rendered_sole_contact_known': False},
        'joint_pose_modified': False, 'physical_contact_verified': False,
        'display_policy': 'one exact generated pose held stationary; upper-body continuation is separately authored',
        'limitations': ['Support markers and native height reference do not establish rendered sole contact with the scene.',
                       'Whole-source suspension cannot be detected without an external scene floor.',
                       'Selection does not repair source motion, post-action waiting, or foot glide.']}
    return actor[chosen].copy(), report
