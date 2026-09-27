"""One bounded, raw-native Core trial on real shallow Scene3 box stairs.

Run on the existing GPU host.  This generates seed 33 first; seed 11 is a
held-out repeat and is refused unless a passing seed-33 report is supplied.
No contact adapter, IK, root correction, or pose mutation is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from traversal_kit import (FPS, FRAMES, PELVIS_CLEARANCE_M, START_Z,
                           planned_root, route_metadata, selftest,
                           target_surface_y)


STAND_PROMPT = "A person stands still in a relaxed neutral pose."
WALK_PROMPT = ("A person walks up a short flight of shallow stairs, lifting "
               "each foot onto the next step, with a relaxed upright posture.")
SAMPLE_FRAMES = np.asarray([7, 15, 23, 31, 39], dtype=int)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _support(geometry, point, reference):
    return geometry.support_height(float(point[0]), float(point[2]),
                                   float(reference), max_step_up=.32,
                                   max_drop=.50)


def _sole_samples(positions, rotations, prefix_positions, prefix_rotations, names):
    """Four rigid foot points calibrated from the native prefix toe offsets."""
    feet = [('LeftFoot', 'LeftToeBase'), ('RightFoot', 'RightToeBase')]
    samples = np.empty((len(positions), 2, 4, 3), dtype=float)
    bind_offsets = []
    for side, (foot_name, toe_name) in enumerate(feet):
        foot, toe = names.index(foot_name), names.index(toe_name)
        displacement = prefix_positions[:, toe] - prefix_positions[:, foot]
        local_toe = np.einsum('tji,tj->ti', prefix_rotations[:, foot], displacement)
        toe_offset = np.median(local_toe, axis=0)
        bind_offsets.append(toe_offset.tolist())
        local = np.asarray([[x, toe_offset[1], z]
                            for z in (-.035, toe_offset[2])
                            for x in (-.045, .045)], dtype=float)
        samples[:, side] = positions[:, foot, None, :] + np.einsum(
            'tij,pj->tpi', rotations[:, foot], local)
    return samples, bind_offsets


def prefix_support(geometry, positions, rotations, names):
    """Measure the unmodified standing prefix on the rendered entry landing."""
    toes = positions[:, [names.index('LeftToeBase'), names.index('RightToeBase')]]
    soles, _ = _sole_samples(positions, rotations, positions, rotations, names)
    toe_support = np.full((len(positions), 2), np.nan)
    sole_support = np.full((len(positions), 2, 4), np.nan)
    for frame in range(len(positions)):
        for side in range(2):
            value = _support(geometry, toes[frame, side], 0.)
            if value is not None:
                toe_support[frame, side] = value
            for point in range(4):
                value = _support(geometry, soles[frame, side, point], 0.)
                if value is not None:
                    sole_support[frame, side, point] = value
    toe_clearance = toes[:, :, 1] - toe_support
    sole_clearance = soles[:, :, :, 1] - sole_support
    finite_toes = toe_clearance[np.isfinite(toe_clearance)]
    finite_soles = sole_clearance[np.isfinite(sole_clearance)]
    metrics = {
        'min_raw_toe_clearance_m': float(finite_toes.min()) if len(finite_toes) else None,
        'min_four_point_sole_clearance_m': float(finite_soles.min()) if len(finite_soles) else None,
        'unsupported_raw_toe_fraction': float(np.isnan(toe_support).mean()),
        'unsupported_sole_point_fraction': float(np.isnan(sole_support).mean()),
    }
    arrays = {'raw_toe_support': toe_support, 'raw_toe_clearance': toe_clearance,
              'four_point_support': sole_support, 'four_point_clearance': sole_clearance}
    return metrics, arrays


def evaluate(geometry, positions, rotations, prefix_positions,
             prefix_rotations, names):
    root_idx = names.index('Hips') if 'Hips' in names else 0
    root = positions[:, root_idx]
    toe_idx = [names.index('LeftToeBase'), names.index('RightToeBase')]
    toes = positions[:, toe_idx]
    soles, bind_offsets = _sole_samples(positions, rotations, prefix_positions,
                                        prefix_rotations, names)
    planned = np.asarray([planned_root(i) for i in range(len(positions))])
    toe_support = np.full((len(positions), 2), np.nan)
    sole_support = np.full((len(positions), 2, 4), np.nan)
    for frame in range(len(positions)):
        reference = target_surface_y(planned[frame, 2])
        for side in range(2):
            value = _support(geometry, toes[frame, side], reference)
            if value is not None:
                toe_support[frame, side] = value
            for point in range(4):
                value = _support(geometry, soles[frame, side, point], reference)
                if value is not None:
                    sole_support[frame, side, point] = value
    toe_clearance = toes[:, :, 1] - toe_support
    sole_clearance = soles[:, :, :, 1] - sole_support
    toe_speed = np.linalg.norm(np.gradient(toes[:, :, [0, 2]], axis=0)*FPS, axis=-1)
    stance = (np.abs(toe_clearance) <= .08) & (toe_speed < .42)
    # Require observed three-frame planted intervals; no imposed gait phases.
    for side in range(2):
        flags = stance[:, side].copy()
        edges = np.flatnonzero(np.diff(np.r_[False, flags, False]))
        for first, last in zip(edges[::2], edges[1::2]):
            if last-first < 3:
                stance[first:last, side] = False
    slip = toe_speed[stance]
    anchors = []
    for side in range(2):
        edges = np.flatnonzero(np.diff(np.r_[False, stance[:, side], False]))
        for first, last in zip(edges[::2], edges[1::2]):
            anchors.append(float(np.max(np.linalg.norm(
                toes[first:last, side][:, [0, 2]] - toes[first, side, [0, 2]], axis=1))))
    head = positions[:, names.index('Head')]
    torso = head-root
    lean = np.degrees(np.arctan2(np.linalg.norm(torso[:, [0, 2]], axis=1),
                                 np.maximum(torso[:, 1], 1e-6)))
    body_names = [name for name in ('Hips', 'Spine', 'Spine1', 'Spine2', 'Neck',
                                    'Head', 'LeftUpLeg', 'RightUpLeg',
                                    'LeftLeg', 'RightLeg', 'LeftArm', 'RightArm')
                  if name in names]
    body_hits = np.zeros((len(positions), len(body_names)), dtype=bool)
    for frame in range(len(positions)):
        for j, name in enumerate(body_names):
            body_hits[frame, j] = geometry.obstacle_at(*positions[frame, names.index(name)],
                                                       radius=.025)
    finite_toes = toe_clearance[np.isfinite(toe_clearance)]
    finite_soles = sole_clearance[np.isfinite(sole_clearance)]
    endpoint_error = float(np.linalg.norm(root[-1]-planned[-1]))
    metrics = {
        'min_raw_toe_clearance_m': float(finite_toes.min()) if len(finite_toes) else None,
        'raw_toe_below_minus_2cm_fraction': float(np.mean(toe_clearance < -.02)),
        'unsupported_raw_toe_fraction': float(np.isnan(toe_support).mean()),
        'min_four_point_sole_clearance_m': float(finite_soles.min()) if len(finite_soles) else None,
        'four_point_sole_below_minus_1cm_fraction': float(np.mean(sole_clearance < -.01)),
        'unsupported_sole_point_fraction': float(np.isnan(sole_support).mean()),
        'body_joint_collision_samples': int(body_hits.sum()),
        'stance_frames': int(stance.sum()),
        'stance_slip_p95_m_s': float(np.percentile(slip, 95)) if len(slip) else None,
        'stance_anchor_max_drift_m': max(anchors) if anchors else None,
        'upper_body_lean_median_deg': float(np.median(lean)),
        'upper_body_lean_p95_deg': float(np.percentile(lean, 95)),
        'root_target_mean_error_m': float(np.linalg.norm(root-planned, axis=1).mean()),
        'root_end_error_m': endpoint_error,
        'root_end_xyz': root[-1].tolist(),
    }
    gates = {
        'raw_toes': metrics['min_raw_toe_clearance_m'] is not None and metrics['min_raw_toe_clearance_m'] >= -.02,
        'sole': metrics['min_four_point_sole_clearance_m'] is not None and metrics['min_four_point_sole_clearance_m'] >= -.01,
        'supported': metrics['unsupported_raw_toe_fraction'] <= .05 and metrics['unsupported_sole_point_fraction'] <= .05,
        'body': metrics['body_joint_collision_samples'] == 0,
        'stance': (metrics['stance_frames'] >= 12 and
                   metrics['stance_slip_p95_m_s'] is not None and
                   metrics['stance_slip_p95_m_s'] <= .20 and
                   metrics['stance_anchor_max_drift_m'] is not None and
                   metrics['stance_anchor_max_drift_m'] <= .025),
        'lean': metrics['upper_body_lean_median_deg'] <= 25.,
        'route': metrics['root_end_error_m'] <= .20 and metrics['root_target_mean_error_m'] <= .20,
    }
    arrays = {'planned_root': planned, 'raw_toes': toes, 'raw_toe_support': toe_support,
              'raw_toe_clearance': toe_clearance, 'four_point_soles': soles,
              'four_point_support': sole_support, 'four_point_clearance': sole_clearance,
              'observed_stance': stance, 'body_joint_collision': body_hits}
    return metrics, gates, arrays, {'body_joint_names': body_names,
                                    'native_local_toe_offsets': bind_offsets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(33, 11), default=33)
    parser.add_argument('--seed33-report', type=Path,
                        help='Required passing seed-33 report to run held-out seed 11')
    args = parser.parse_args()
    if args.seed == 11:
        if not args.seed33_report or not json.loads(args.seed33_report.read_text()).get('passed'):
            parser.error('seed 11 requires a passing --seed33-report')
    import torch
    from ardy.model import load_model
    from ardy.tools import seed_everything
    from ardy.constraints import Root2DConstraintSet
    from core_terrain_constraints import RootHeightConstraint, shift_vertical_coordinate_frame
    from scene_interaction_geometry import SceneInteractionGeometry

    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    scene = selftest()
    (args.output/'scene.json').write_text(json.dumps(scene, indent=2))
    geometry = SceneInteractionGeometry.from_scene(scene)
    paths = {key: args.embeddings/(key+'_embedding.pt') for key in ('stand', 'walk')}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    report = {
        'status': 'loading', 'seed': args.seed, 'passed': False,
        'route': route_metadata(), 'prefix_frames': 40,
        'native_pose_policy': 'unchanged model output; rigid local-floor coordinate transforms only',
        'prompts': {'stand': STAND_PROMPT, 'walk': WALK_PROMPT},
        'embedding_sources': {key: {'path': str(path), 'sha256': _sha256(path)}
                              for key, path in paths.items()},
        'checkpoints': str(args.checkpoints),
        'warning': 'Joint and four-point sole proxies are not skinned mesh or dynamics proof.',
    }
    def save():
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    save()
    print('Loading native Core model', flush=True)
    model = load_model('ARDY-Core-RP-20FPS-Horizon40', device='cuda',
                       checkpoints_dir=str(args.checkpoints), text_encoder_mode='local',
                       text_encoder=False)
    names = list(model.skeleton.bone_order_names)
    report['joint_names'] = names
    report['parents'] = [p for _, p in model.skeleton.bone_order_names_with_parents]
    report['feature_slices'] = {k: [v.start, v.stop]
                                for k, v in model.motion_rep.slice_dict.items()}
    report['model_denoising_steps'] = int(model.diffusion.num_base_steps)
    embeddings = {key: tuple(value.to('cuda') for value in torch.load(path, weights_only=True))
                  for key, path in paths.items()}
    model.text_encoder = None

    def constraints(frame_indices, points, floor_y):
        frames = torch.as_tensor(frame_indices, dtype=torch.long, device='cuda')
        xyz = torch.as_tensor(points, dtype=torch.float32, device='cuda').clone()
        xyz[:, 1] -= floor_y
        return [Root2DConstraintSet(model.skeleton, frames, xyz[:, [0, 2]]),
                RootHeightConstraint(frames, xyz[:, 1])]

    def generate(history, conditions, text_key, floor_y):
        h = 0 if history is None else history.shape[1]
        observed, mask = model.motion_rep.create_conditions_from_constraints_batched(
            conditions, torch.tensor([h+40], device='cuda'), True, 'cuda')
        if mask[:, :h].any():
            raise RuntimeError('conditions unexpectedly target history')
        history_local = (shift_vertical_coordinate_frame(history, model.motion_rep, -floor_y)
                         if history is not None and floor_y else history)
        emb = embeddings[text_key]
        with torch.inference_mode():
            native = model.autoregressive_step(
                num_frames=h+40, num_denoising_steps=model.diffusion.num_base_steps,
                motion_mask=mask, observed_motion=observed, cfg_weight=(2., 2.),
                text_feat=emb[0], text_pad_mask=emb[1],
                init_history_sequence=history_local,
                init_global_translation=torch.tensor([[0., 0., START_Z]], device='cuda'),
                init_first_heading_angle=torch.tensor([np.pi], device='cuda', dtype=torch.float32))
        generated = native[:, h:]
        restored = (shift_vertical_coordinate_frame(generated, model.motion_rep, floor_y)
                    if floor_y else generated)
        return restored, sorted(torch.nonzero(mask[0].any(0)).flatten().cpu().tolist())

    seed_everything(8171)
    standing = np.tile(np.asarray([0., PELVIS_CLEARANCE_M, START_Z]), (len(SAMPLE_FRAMES), 1))
    prefix, prefix_mask = generate(None, constraints(SAMPLE_FRAMES, standing, 0.), 'stand', 0.)
    with torch.inference_mode():
        prefix_pose = model.motion_rep.inverse(prefix, is_normalized=True)
    prefix_positions = prefix_pose['posed_joints'][0].cpu().numpy()
    prefix_rotations = prefix_pose['global_rot_mats'][0].cpu().numpy()
    prefix_metrics, prefix_arrays = prefix_support(geometry, prefix_positions,
                                                    prefix_rotations, names)
    np.savez_compressed(args.output/'prefix.npz', native_features=prefix[0].cpu().numpy(),
                        positions=prefix_positions, rotations=prefix_rotations,
                        contacts=prefix_pose['foot_contacts'][0].cpu().numpy(),
                        **prefix_arrays)
    report['prefix_mask_feature_indices'] = prefix_mask
    report['prefix_support_metrics'] = prefix_metrics
    report['status'] = 'generating'
    save()
    seed_everything(args.seed)
    history = prefix[:, -4:].clone()
    chunks, masks, origins = [], [], []
    began = time.monotonic()
    for horizon in range(FRAMES//40):
        offset = horizon*40
        floor_y = target_surface_y(float(planned_root(offset-1 if offset else 0)[2]))
        origins.append(floor_y)
        points = np.asarray([planned_root(offset+int(i)) for i in SAMPLE_FRAMES])
        piece, mask = generate(history, constraints(SAMPLE_FRAMES+len(history[0]), points, floor_y),
                               'walk', floor_y)
        chunks.append(piece)
        masks.append(mask)
        history = piece[:, -4:].clone()
        print(f'Completed horizon {horizon+1}/3', flush=True)
    native = torch.cat(chunks, dim=1)
    with torch.inference_mode():
        posed = model.motion_rep.inverse(native, is_normalized=True)
    positions = posed['posed_joints'][0].cpu().numpy()
    rotations = posed['global_rot_mats'][0].cpu().numpy()
    contacts = posed['foot_contacts'][0].cpu().numpy()
    metrics, gates, arrays, proxy = evaluate(geometry, positions, rotations,
                                               prefix_positions, prefix_rotations, names)
    full_features = torch.cat([prefix, native], dim=1)[0].cpu().numpy()
    np.savez_compressed(args.output/f'seed{args.seed}.npz',
                        native_features=native[0].cpu().numpy(),
                        with_prefix_native_features=full_features,
                        positions=positions, rotations=rotations, contacts=contacts,
                        with_prefix_positions=np.concatenate([prefix_positions, positions]),
                        with_prefix_rotations=np.concatenate([prefix_rotations, rotations]),
                        **arrays)
    report.update(status='completed', duration_seconds=time.monotonic()-began,
                  local_floor_origins_m=origins, walk_mask_feature_indices=masks,
                  metrics=metrics, gates=gates, passed=all(gates.values()),
                  proxy=proxy, archive=f'seed{args.seed}.npz')
    save()
    print(json.dumps({'passed': report['passed'], 'metrics': metrics, 'gates': gates}), flush=True)


if __name__ == '__main__':
    main()
