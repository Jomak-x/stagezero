"""Bounded warm-Core approach/exit around an untouched native InterGen pair.

Four serial 40-frame Core jobs. Core native features condition only its own
subsequent window; InterGen never becomes synthetic Core history. Boundaries
and optional authored bridge candidates are measured, not hidden or accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.verify_studio_core import BoundedClient
from native_pair_transition import (authored_boundary_bridge, authored_direction_bridge,
                                    boundary_diagnostics, compose_ardy_pair_context,
                                    core27_to_native22, core_adapter_provenance, shared_place_pair)
from realtime_navigation import validate_ground_path
from studio_interaction_scene import adapt_studio_scene


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def heading(pose):
    across = pose[:, 1] - pose[:, 2]
    return np.arctan2(-across[:, 2], across[:, 0])


def refine_saved_trial(folder):
    """CPU-only explicit Core morphology refinement; preserve every raw archive."""
    report = json.loads((folder/'report.json').read_text())
    with np.load(folder/'pair-world.npz', allow_pickle=False) as a:
        pair = a['joints'].copy()
    result = {'status': 'review candidate; animation acceptance unverified',
              'gpu_jobs': 0, 'source_trial': str(folder),
              'source_pair_sha256': report['pair_sha256'], 'shared_placement': report['shared_pair_placement'],
              'source_pair_frames_modified': False, 'refinement': {}, 'boundaries': {}}
    cores = {}
    for phase in ('approach', 'exit'):
        clips = []
        for window in range(2):
            with np.load(folder/f'{phase}-{window}-core.npz', allow_pickle=False) as a:
                clips.append(a['positions'].transpose(1, 0, 2, 3))
        cores[phase] = np.concatenate(clips)
    combined, metadata = compose_ardy_pair_context(cores['approach'], pair, cores['exit'])
    result.update(metadata['transition_provenance'])
    metadata['transition_provenance'] = dict(result)
    np.savez_compressed(folder/'refined-review-native22.npz', joints=combined, metadata=np.array(json.dumps(metadata)))
    result['segments'] = metadata['segments']
    result['archive'] = str(folder/'refined-review-native22.npz')
    result['all_mechanical_gates_passed'] = all(b['mechanical_gate_passed'] for b in result['boundaries'].values())
    write_json(folder/'refined-report.json', result)
    print(json.dumps({'archive': result['archive'], 'mechanical_gates': result['all_mechanical_gates_passed']}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--token-file', type=Path)
    ap.add_argument('--refine-only', action='store_true', help='CPU refinement of existing output; no GPU or service requests')
    ap.add_argument('--url', default='http://127.0.0.1:8769')
    ap.add_argument('--pair', type=Path, default=ROOT/'review/two-character/native-recovery/originals/handshake_seed42.npz')
    ap.add_argument('--scene', type=Path, default=ROOT/'review/scene-integration/live-city.json')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=92642)
    args = ap.parse_args()
    if args.refine_only:
        refine_saved_trial(args.output)
        return
    if args.token_file is None:
        ap.error('--token-file is required for a live trial')
    args.output.mkdir(parents=True, exist_ok=False)
    client = BoundedClient(args.url, args.token_file.read_text(), timeout=10, job_timeout=90, max_jobs=4)
    report = {'status': 'running', 'jobs': [], 'seed': args.seed, 'scene': str(args.scene),
              'pair_source': str(args.pair), 'pair_sha256': hashlib.sha256(args.pair.read_bytes()).hexdigest(),
              'joint_generation': 'Only the InterGen interaction is jointly generated. Core approach/exit actors are independent.',
              'visual_acceptance': 'unverified', 'synthetic_bridge_inserted': False,
              'exit_limitation': 'Exit starts a fresh Core sample at the source pair final roots. No valid Core history exists for InterGen.',
              'adapter': core_adapter_provenance()}
    write_json(args.output/'report.json', report)
    try:
        report['health_before'] = client.health()
        with np.load(args.pair, allow_pickle=False) as source:
            raw = source['joints'].copy()
            source_meta = json.loads(source['metadata'].item())
        if raw.shape != (210, 2, 22, 3) or source_meta['fps'] != 30:
            raise ValueError('Trial requires a complete 210-frame native30 pair')
        offset = -raw[0, :, 0].mean(0); offset[1] = .025
        pair = shared_place_pair(raw, translation=offset)
        report['shared_pair_placement'] = {'yaw_radians': 0., 'translation_xyz_m': offset.tolist(),
                                          'per_actor_offsets': False, 'scale': 1.}
        shutil.copyfile(args.pair, args.output/'source-pair-original.npz')
        np.savez_compressed(args.output/'pair-world.npz', joints=pair, metadata=np.array(json.dumps({
            'source': source_meta, 'placement': report['shared_pair_placement'], 'fps': 30})))
        scene = adapt_studio_scene(json.loads(args.scene.read_text()))['scene']
        ids = ['actor_1', 'actor_2']
        paths = {}
        for phase, edge in [('approach', 0), ('exit', -1)]:
            roots = pair[edge, :, 0][:, [0, 2]]
            yaws = heading(pair[edge])
            outward = roots - roots.mean(0)
            outward /= np.linalg.norm(outward, axis=-1, keepdims=True)
            start = roots + outward*.65 if phase == 'approach' else roots
            end = roots if phase == 'approach' else roots + outward*.65
            report[phase + '_planned_floor'] = [validate_ground_path(scene, [a, b]) for a, b in zip(start, end)]
            clips = []
            for window in range(2):
                prompt = ('A person walks slowly forward and stops in a relaxed standing pose.'
                          if phase == 'approach' and window == 0 else
                          'A person stands still in a relaxed neutral pose.' if phase == 'approach' or window == 0 else
                          'A person turns away and walks slowly forward.')
                goals = {}
                for actor, aid in enumerate(ids):
                    goals[aid] = []
                    for frame in (0, 7, 15, 23, 31, 39):
                        progress = min(1., frame / 31.) if window == (0 if phase == 'approach' else 1) else (1. if phase == 'approach' else 0.)
                        point = start[actor]*(1-progress) + end[actor]*progress
                        angle = yaws[actor]
                        if phase == 'exit' and window == 1:
                            angle = np.arctan2(outward[actor, 0], outward[actor, 1])
                        goals[aid].append({'frame': frame, 'position_xz': point.tolist(), 'heading': float(angle)})
                body = {'request_id': 'native-boundary-' + uuid.uuid4().hex,
                        'stage_kind': 'approach' if window == 0 else 'continuation',
                        'frames': 40, 'prompt': prompt, 'actor_ids': ids, 'seed': args.seed,
                        'root_targets': goals}
                if window == 0:
                    body['initial_placements'] = {aid: {'position_xz': start[i].tolist(), 'yaw': float(yaws[i])} for i, aid in enumerate(ids)}
                else:
                    body['history'] = {'native_features': clips[-1].native_features[:, -40:].tolist()}
                write_json(args.output/f'{phase}-{window}-request.json', body)
                entry = {'phase': phase, 'window': window, 'request_id': body['request_id'], 'status': 'submitted'}
                report['jobs'].append(entry); write_json(args.output/'report.json', report)
                clip = client.wait(body)[0]
                clips.append(clip)
                np.savez_compressed(args.output/f'{phase}-{window}-core.npz', positions=clip.positions,
                                    rotations=clip.rotations, native_features=clip.native_features,
                                    metadata=np.array(json.dumps(clip.metadata)))
                entry['status'] = 'complete'; write_json(args.output/'report.json', report)
            positions = np.concatenate([c.positions for c in clips], axis=1)
            paths[phase] = core27_to_native22(positions).transpose(1, 0, 2, 3)
            np.savez_compressed(args.output/f'{phase}-display.npz', joints=paths[phase],
                                metadata=np.array(json.dumps({'fps': 20, 'source': 'ARDY Core', 'adapter': report['adapter']})))
            report[phase + '_actual_floor'] = [validate_ground_path(scene, p[:, 0, :][:, [0, 2]]) for p in positions]
            report[phase + '_final_root_target_error_m'] = np.linalg.norm(positions[:, -1, 0][:, [0, 2]]-end, axis=-1).tolist()
        report['boundaries'] = {}
        for label, a, b, fa, fb in [('entry', paths['approach'], pair, 20, 30), ('exit', pair, paths['exit'], 30, 20)]:
            bridge, diagnostics = authored_boundary_bridge(a, b, left_fps=fa, right_fps=fb)
            report['boundaries'][label] = diagnostics
            np.savez_compressed(args.output/f'{label}-bridge-candidate.npz', joints=bridge,
                                metadata=np.array(json.dumps(diagnostics)))
        report['pair_world_floor'] = [validate_ground_path(scene, pair[:, i, 0][:, [0, 2]]) for i in range(2)]
        report['status'] = 'complete; transition visual acceptance pending'
        report['health_after'] = client.health()
        write_json(args.output/'timeline.json', {'fps_policy': 'Each segment retains source FPS; no interaction frame is resampled.',
                   'hard_cuts_explicit': True, 'authored_bridges_inserted': False,
                   'segments': [{'file': 'approach-display.npz', 'fps': 20, 'frames': 80, 'start_seconds': 0},
                                {'file': 'pair-world.npz', 'fps': 30, 'frames': 210, 'start_seconds': 4},
                                {'file': 'exit-display.npz', 'fps': 20, 'frames': 80, 'start_seconds': 11}]})
    except Exception as exc:
        report['status'] = 'failed'; report['error'] = str(exc)
        raise
    finally:
        report['gpu_jobs_submitted'] = client.submitted_jobs
        write_json(args.output/'report.json', report)
        print(json.dumps({'status': report['status'], 'jobs': client.submitted_jobs, 'output': str(args.output)}), flush=True)


if __name__ == '__main__': main()
