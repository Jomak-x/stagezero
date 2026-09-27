"""Measured native22 endpoint gait proxies using the crowd renderer's phase math.

This measures ankle/toe endpoints, not skin contact or ground-reaction forces.
Contact candidates depend only on height and vertical speed: selecting slow XZ
feet would hide the very sliding this module is intended to measure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FEET = (7, 8, 10, 11)
FOOT_NAMES = ('left_ankle', 'right_ankle', 'left_toe', 'right_toe')
PHASE_OFFSET = .61803398875


def distribution(values):
    v = np.asarray(values, dtype=float).ravel()
    if not len(v):
        return {'samples': 0, 'mean': None, 'median': None, 'p95': None, 'max': None}
    return {'samples': int(len(v)), 'mean': float(v.mean()),
            'median': float(np.median(v)), 'p95': float(np.percentile(v, 95)),
            'max': float(v.max())}


def rendered_feet(manifest, poses, trajectory, *, sample_fps=60.):
    """Reconstruct GPU-affine endpoint targets at actual timeline sample times.

    Contract: CrowdRenderer.update(), walking clips selected by actor gait,
    accumulated root XZ distance, golden-ratio phase, wrapped linear walk blend,
    nearest-frame idle, rate-limited smoothstep(.025,.42), shortest yaw arc and
    actor scales. Weights are computed at trajectory ticks then interpolated.
    """
    poses = np.asarray(poses, dtype=float)
    if poses.shape != (manifest['totalFrames'], 22, 3) or not np.isfinite(poses).all():
        raise ValueError('Expected finite poses[totalFrames,22,3]')
    if sample_fps <= 0 or not np.isfinite(sample_fps):
        raise ValueError('sample_fps must be positive')
    frames = trajectory['frames']
    people = np.array([f['people'] for f in frames], dtype=float)
    if len(frames) < 2 or people.ndim != 3 or people.shape[2] < 4:
        raise ValueError('At least two trajectory frames are required')
    dt = float(trajectory['dt'])
    frame_times = np.array([f['t'] for f in frames])
    if dt <= 0 or not np.allclose(frame_times, np.arange(len(frames))*dt, atol=.000051, rtol=0):
        raise ValueError('Renderer requires evenly spaced trajectory frames starting at zero')
    if not np.isfinite(people).all():
        raise ValueError('Nonfinite trajectory')
    distances = np.zeros(people.shape[:2])
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(people[:, :, :2], axis=0), axis=-1), axis=0)
    # Include the endpoint exactly even when the duration is not a sample multiple.
    times = np.unique(np.r_[np.arange(0., frame_times[-1], 1./sample_fps), frame_times[-1]])
    index = np.minimum(len(frames)-2, np.floor(times/dt).astype(int))
    alpha = np.clip((times-frame_times[index])/dt, 0., 1.)
    a, b = people[index], people[index+1]
    root = a[:, :, :2]*(1-alpha[:, None, None]) + b[:, :, :2]*alpha[:, None, None]
    distance = distances[index]*(1-alpha[:, None]) + distances[index+1]*alpha[:, None]
    delta = b[:, :, 2]-a[:, :, 2]
    heading = a[:, :, 2]+np.arctan2(np.sin(delta), np.cos(delta))*alpha[:, None]
    walks = [c for c in manifest['clips'] if c.get('type') == 'walk' or 'walk' in c['id']]
    idle = next((c for c in manifest['clips'] if 'idle' in c['id']), manifest['clips'][0])
    if not walks:
        raise ValueError('Renderer requires a walk clip')
    target = np.clip((people[:, :, 3]-.025)/(.42-.025), 0., 1.)
    target = target*target*(3-2*target)
    weights = np.empty_like(target)
    weights[0] = target[0]
    for frame in range(1, len(frames)):
        limit = 1.8*(frame_times[frame]-frame_times[frame-1])
        weights[frame] = weights[frame-1] + np.clip(target[frame]-weights[frame-1], -limit, limit)
    blend = weights[index]*(1-alpha[:, None]) + weights[index+1]*alpha[:, None]
    feet = np.empty((len(times), people.shape[1], len(FEET), 3))
    for actor in range(people.shape[1]):
        gait = trajectory.get('agents', [{}]*people.shape[1])[actor].get('gait')
        style = 'brisk' if gait == 'brisk' else 'relaxed'
        fallback = min(1, len(walks)-1) if gait == 'brisk' else 0
        clip = next((c for c in walks if style in c['id']), walks[fallback])
        xz_scale = .96+(actor*7 % 9)/100
        phase = (distance[:, actor]/(max(.3, clip['strideMeters'])*xz_scale)*clip['frames']
                 + actor*PHASE_OFFSET*clip['frames']) % clip['frames']
        left = np.floor(phase).astype(int)
        weight = (phase-left)[:, None, None]
        walk = (poses[clip['offset']+left][:, FEET]*(1-weight)
                + poses[clip['offset']+(left+1) % clip['frames']][:, FEET]*weight)
        idle_index = (np.floor(times*idle['fps']+actor*7).astype(int) % idle['frames'])+idle['offset']
        local = poses[idle_index][:, FEET]*(1-blend[:, actor, None, None]) + walk*blend[:, actor, None, None]
        local *= [xz_scale, .94+(actor*17 % 13)/100, xz_scale]
        c, s = np.cos(heading[:, actor, None]), np.sin(heading[:, actor, None])
        feet[:, actor, :, 0] = local[:, :, 0]*c + local[:, :, 2]*s + root[:, actor, 0, None]
        feet[:, actor, :, 1] = local[:, :, 1]
        feet[:, actor, :, 2] = -local[:, :, 0]*s + local[:, :, 2]*c + root[:, actor, 1, None]
    return times, feet, blend


def foot_metrics(times, feet, blend, *, height_band_m=.06, vertical_speed_mps=.25):
    """Low/vertically-stable endpoints are a contact proxy, never ground truth."""
    step_dt = np.diff(times)
    delta = np.diff(feet, axis=0)
    xz_step = np.linalg.norm(delta[..., [0, 2]], axis=-1)
    xz_speed = xz_step/step_dt[:, None, None]
    vertical = np.abs(delta[..., 1])/step_dt[:, None, None]
    low = (feet[:-1, ..., 1] <= height_band_m) & (feet[1:, ..., 1] <= height_band_m)
    proxy = low & (vertical <= vertical_speed_mps)
    transition = ((blend[:-1] > 0) & (blend[:-1] < 1)) | ((blend[1:] > 0) & (blend[1:] < 1))
    stopped = (blend[:-1] == 0) & (blend[1:] == 0)
    walking = (blend[:-1] == 1) & (blend[1:] == 1)
    worst = None
    if np.any(proxy):
        sample, actor, foot = np.unravel_index(np.where(proxy, xz_speed, -1).argmax(), proxy.shape)
        worst = {'time_start_s': float(times[sample]), 'time_end_s': float(times[sample+1]),
                 'actor_index': int(actor), 'foot': FOOT_NAMES[foot],
                 'xz_speed_mps': float(xz_speed[sample, actor, foot]),
                 'blend_start': float(blend[sample, actor]), 'blend_end': float(blend[sample+1, actor]),
                 'start_xyz_m': feet[sample, actor, foot].tolist(), 'end_xyz_m': feet[sample+1, actor, foot].tolist()}
    bouts, bout_path, bout_seconds = [], [], []
    for actor in range(feet.shape[1]):
        for foot in range(feet.shape[2]):
            valid = proxy[:, actor, foot]
            starts = np.flatnonzero(valid & ~np.r_[False, valid[:-1]])
            ends = np.flatnonzero(valid & ~np.r_[valid[1:], False])+1
            for start, end in zip(starts, ends):
                bouts.append(np.linalg.norm(feet[end, actor, foot, [0, 2]]-feet[start, actor, foot, [0, 2]]))
                bout_path.append(xz_step[start:end, actor, foot].sum())
                bout_seconds.append(times[end]-times[start])
    return {
        'low_foot_proxy': {'height_ceiling_m': height_band_m, 'vertical_speed_ceiling_mps': vertical_speed_mps,
                           'candidate_fraction': float(proxy.mean()), 'uses_horizontal_speed_to_select': False},
        'proxy_world_xz_speed_mps': distribution(xz_speed[proxy]),
        'proxy_bout_endpoint_drift_m': distribution(bouts),
        'proxy_bout_path_length_m': distribution(bout_path),
        'proxy_bout_duration_s': distribution(bout_seconds),
        'transition_proxy_world_xz_speed_mps': distribution(xz_speed[proxy & transition[:, :, None]]),
        'transition_all_feet_world_xz_speed_mps': distribution(xz_speed[np.broadcast_to(transition[:, :, None], xz_speed.shape)]),
        'stopped_proxy_world_xz_speed_mps': distribution(xz_speed[proxy & stopped[:, :, None]]),
        'full_walk_proxy_world_xz_speed_mps': distribution(xz_speed[proxy & walking[:, :, None]]),
        'worst_proxy_interval': worst,
        'all_endpoint_frame_step_m': distribution(np.linalg.norm(delta, axis=-1)),
        'minimum_endpoint_y_m': float(feet[..., 1].min()),
        'endpoint_samples_below_y_minus_1cm': int((feet[..., 1] < -.01).sum()),
        'feet': {name: distribution(xz_speed[:, :, k][proxy[:, :, k]]) for k, name in enumerate(FOOT_NAMES)},
    }


def source_loop_metrics(manifest, poses):
    result = []
    for clip in manifest['clips']:
        p = poses[clip['offset']:clip['offset']+clip['frames']]
        seam = p[0]-p[-1]
        result.append({'id': clip['id'], 'frames': clip['frames'],
                       'loop_endpoint_discontinuity_rms_m': float(np.sqrt(np.mean(np.sum(seam*seam, axis=-1)))),
                       'foot_loop_endpoint_discontinuity_m': distribution(np.linalg.norm(seam[list(FEET)], axis=-1)),
                       'loop_edge_speed_at_nominal_fps_mps': distribution(np.linalg.norm(seam, axis=-1)*clip['fps']),
                       'ordinary_endpoint_step_m': distribution(np.linalg.norm(np.diff(p, axis=0), axis=-1))})
    return result


def evaluate(manifest, poses, trajectory, *, sample_fps=60.):
    times, feet, blend = rendered_feet(manifest, poses, trajectory, sample_fps=sample_fps)
    return {'schema': 'stagezero.crowd-gait-metrics.v1', 'actors': feet.shape[1],
            'duration_s': float(times[-1]), 'sample_fps': sample_fps, 'sampled_frames': len(times),
            'renderer_contract': 'CrowdRenderer.update + locomotionWeights: gait selection (casual maps relaxed), XZ-scale-adjusted distance phase, golden offset, linear wrapped walk, nearest idle, tick smoothstep rate-limited 1.8/s and interpolated, actor XYZ scale',
            'source_loops': source_loop_metrics(manifest, poses),
            'metrics': foot_metrics(times, feet, blend),
            'limitations': ['Low and vertically stable joint endpoints are contact candidates, not measured physical contacts.',
                            'Native22 endpoint targets are measured, not deformed sole vertices; skin-floor contact can differ.',
                            'Procedural idle/walk blends and source loop seams remain included in the measured motion.',
                            'Frame sampling may miss peaks between samples; no metric threshold implies visual acceptance.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--trajectory', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--sample-fps', type=float, default=60.)
    args = parser.parse_args()
    manifest_path = args.assets/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    poses_path = args.assets/manifest.get('posesUrl', 'poses.bin')
    poses = np.fromfile(poses_path, dtype='<f4').reshape(-1, 22, 3)
    result = evaluate(manifest, poses, json.loads(args.trajectory.read_text()), sample_fps=args.sample_fps)
    result['inputs'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [manifest_path, poses_path, args.trajectory, Path(__file__).resolve().parent/'studio_client/src/crowd/CrowdRenderer.ts']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'actors': result['actors'],
                      'proxy_world_xz_speed_mps': result['metrics']['proxy_world_xz_speed_mps']}))


if __name__ == '__main__':
    main()
