"""Deterministic standing reference and synthetic character diagnostic motion."""
from __future__ import annotations

import numpy as np
import torch

from retargeting import neutral_source_pose


def diagnostic_clip(skeleton, frames=150):
    """FK narrow, softly bent stand with an outward-and-return arm sweep.

    This is a lab motion, distinct from the identity-local G1 calibration
    pose used by production retargeting.
    """
    if frames < 2:
        raise ValueError('Diagnostic clip needs at least two frames')
    neutral_positions, _ = neutral_source_pose(skeleton)
    local = np.tile(np.eye(3), (frames, 34, 1, 1))
    roots = np.tile(neutral_positions[0], (frames, 1))

    def pitch_rotation(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def roll_rotation(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    for side in ('left', 'right'):
        hip = neutral_positions[skeleton.bone_index[f'{side}_hip_pitch_skel']]
        hip_roll = neutral_positions[skeleton.bone_index[f'{side}_hip_roll_skel']]
        knee = neutral_positions[skeleton.bone_index[f'{side}_knee_skel']]
        ankle = neutral_positions[skeleton.bone_index[f'{side}_ankle_roll_skel']]

        # The G1 hip motors are separated laterally. Rotate the lower chain
        # inward around its roll pivot until the knee is under the anatomical
        # hip, then restore an upright shin through the knee's local orientation.
        hip_to_roll = hip_roll - hip
        roll_to_knee = knee - hip_roll
        radius = np.linalg.norm(roll_to_knee[:2])
        roll_angle = (np.arctan2(-hip_to_roll[0], (radius**2 - hip_to_roll[0]**2)**.5)
                      - np.arctan2(roll_to_knee[0], -roll_to_knee[1]))
        roll = roll_rotation(roll_angle)
        thigh = hip_to_roll + roll @ roll_to_knee
        shin = ankle - knee

        # The TASM knee pivot sits at the front edge of its skin; a 35 mm G1
        # forward offset gives the visible mesh a gentle ~12-degree bend.
        # Counter-pitch at the ankle preserves level soles.
        knee_forward = .035
        thigh_pitch = np.arctan2(thigh[2], -thigh[1]) - np.arctan2(
            knee_forward, (np.linalg.norm(thigh[1:])**2 - knee_forward**2)**.5)
        shin_pitch = np.arctan2(shin[2], -shin[1]) + np.arctan2(
            knee_forward, (np.linalg.norm(shin[1:])**2 - knee_forward**2)**.5)
        hip_pitch = pitch_rotation(thigh_pitch)
        shin_global = pitch_rotation(shin_pitch)
        local[:, skeleton.bone_index[f'{side}_hip_pitch_skel']] = hip_pitch
        local[:, skeleton.bone_index[f'{side}_hip_roll_skel']] = roll
        local[:, skeleton.bone_index[f'{side}_knee_skel']] = (hip_pitch @ roll).T @ shin_global
        local[:, skeleton.bone_index[f'{side}_ankle_pitch_skel']] = pitch_rotation(-shin_pitch)
        elbow = neutral_positions[skeleton.bone_index[f'{side}_elbow_skel']]
        wrist = neutral_positions[skeleton.bone_index[f'{side}_wrist_yaw_skel']]
        forearm = wrist - elbow
        elbow_pitch = np.arctan2(forearm[2], -forearm[1])
        local[:, skeleton.bone_index[f'{side}_elbow_skel']] = pitch_rotation(elbow_pitch)

    # Ground the actual toe endpoints after the leg correction. Keep the
    # resulting world positions and rotations coupled through the same FK.
    with torch.inference_mode():
        _, stand_positions, _ = skeleton.fk(torch.tensor(local[0], dtype=torch.float64), torch.tensor(roots[0], dtype=torch.float64))
    toe_indices = [skeleton.bone_index[f'{side}_toe_base'] for side in ('left', 'right')]
    roots[:, 1] -= float(stand_positions[toe_indices, 1].min())

    for frame in range(frames):
        phase = 2 * np.pi * frame / (frames - 1)
        angle = .1 + .325 * (1 - np.cos(phase))
        c, s = np.cos(angle), np.sin(angle)
        local[frame, skeleton.bone_index['left_shoulder_roll_skel']] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        local[frame, skeleton.bone_index['right_shoulder_roll_skel']] = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
    with torch.inference_mode():
        global_rotations, joint_positions, _ = skeleton.fk(torch.tensor(local, dtype=torch.float64), torch.tensor(roots, dtype=torch.float64))
    return joint_positions.numpy(), global_rotations.numpy()



def standing_reference(skeleton):
    """The first diagnostic frame, independent of loaded or generated motion."""
    positions, rotations = diagnostic_clip(skeleton, frames=2)
    return positions[0], rotations[0]
