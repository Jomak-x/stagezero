"""Export reviewed Core27 sources as native22 affine crowd clips.

The mesh is the repository's authored Xbot skin.  Each affine is solved by the
same NativeRigActor code used in live native playback; only root travel and a
single global heading are removed for reusable locomotion.  No clip is claimed
to be a native crowd model or a learned blend.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.native_pair_rig import NativeRigActor, NativeRigAsset
from native_pair_transition import core27_to_native22


N_BONES = 22
SOURCE_FPS = 20.0
OUTPUT_FPS = 30.0
REPO_ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def request_evidence(source_path: Path, metadata: dict) -> dict:
    request_path = source_path.with_name(source_path.name.replace('-core.npz', '-request.json'))
    request = json.loads(request_path.read_text()) if request_path.is_file() else metadata.get('request', {})
    return {'request_id': request.get('request_id', metadata.get('request_id')),
            'prompt': request.get('prompt'), 'seed': request.get('seed', metadata.get('seed')),
            'stage_kind': request.get('stage_kind', metadata.get('stage_kind')),
            'request_file': repo_path(request_path) if request_path.is_file() else None,
            'request_sha256': sha256(request_path) if request_path.is_file() else None}


def read_core_actor(path: Path, actor_index: int) -> tuple[np.ndarray, dict]:
    """Return model positions, leaving the original npz untouched."""
    with np.load(path, allow_pickle=False) as archive:
        source = np.asarray(archive['positions'], dtype=np.float64)
        metadata = json.loads(str(archive['metadata'].item())) if 'metadata' in archive else {}
    if source.ndim != 4 or source.shape[0] <= actor_index or source.shape[2:] != (27, 3):
        raise ValueError(f'{path}: expected Core positions[actors,frames,27,3]')
    return core27_to_native22(source[actor_index]), metadata


def resample(poses: np.ndarray, source_fps: float = SOURCE_FPS,
             output_fps: float = OUTPUT_FPS) -> np.ndarray:
    if len(poses) < 4:
        raise ValueError('Source clip is too short')
    duration = (len(poses) - 1) / source_fps
    count = round(duration * output_fps) + 1
    sampled = np.arange(count, dtype=np.float64) / output_fps * source_fps
    left = np.minimum(sampled.astype(int), len(poses) - 2)
    alpha = (sampled - left)[:, None, None]
    return poses[left] * (1 - alpha) + poses[left + 1] * alpha


def canonicalize(poses: np.ndarray, *, walking: bool) -> tuple[np.ndarray, dict]:
    """Align locomotion with +Z, remove horizontal root travel, retain bob."""
    poses = np.array(poses, dtype=np.float64, copy=True)
    root = poses[:, 0].copy()
    planar = root[-1, [0, 2]] - root[0, [0, 2]]
    distance = float(np.linalg.norm(planar))
    if walking and distance < 0.15:
        raise ValueError('Walk source lacks meaningful root travel')
    heading = float(np.arctan2(planar[0], planar[1])) if walking else 0.0
    c, s = np.cos(-heading), np.sin(-heading)
    rotation = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    poses -= root[:, None, :] * np.array([1., 0., 1.])[None, None, :]
    poses = poses @ rotation.T
    foot_indices = [7, 8, 10, 11]
    floor = float(np.percentile(poses[:, foot_indices, 1], 3))
    poses[:, :, 1] -= floor
    return poses, {'source_root_distance_m': distance,
                   'source_heading_rad': heading,
                   'source_speed_mps': distance / ((len(poses) - 1) / OUTPUT_FPS),
                   'floor_reference_y_m': floor}


def solve_affines(asset: NativeRigAsset, poses: np.ndarray) -> np.ndarray:
    """Use the live native solver without allocating a Viser scene or handles."""
    solver = object.__new__(NativeRigActor)
    solver.asset = asset
    solver._removed = False
    solver._scale = None
    solver._previous_normals = {}
    output = np.empty((len(poses), N_BONES, 3, 4), dtype='<f4')
    for index, pose in enumerate(poses):
        linear, targets, _ = solver._solve_pose(pose)
        output[index, :, :, :3] = linear
        output[index, :, :, 3] = targets
    return output


def export_parts(asset: NativeRigAsset) -> list[dict]:
    parts = []
    for index, part in enumerate(asset.parts):
        bones = np.asarray(part['bones'], dtype=np.int32)
        local = np.asarray(part['bind_world'], dtype=np.float32) - asset.rest[bones].astype(np.float32)
        positions = np.asarray(part['rest_vertices'], dtype=np.float32)
        parts.append({
            'name': f'xbot-part-{index}',
            'positions': positions.reshape(-1).tolist(),
            'indices': np.asarray(part['faces'], dtype=np.uint32).reshape(-1).tolist(),
            'localBind': local.reshape(-1).tolist(),
            'bones': bones.reshape(-1).tolist(),
            'weights': np.asarray(part['weights'], dtype=np.float32).reshape(-1).tolist(),
            'color': '#8faeb8' if index % 3 == 0 else '#88949f',
        })
    return parts


def build_library(sources: list[dict], output: Path, *, rig_path: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    asset = NativeRigAsset(rig_path)
    clips = []
    affines = []
    joint_poses = []
    offset = 0
    for spec in sources:
        paths = [Path(p) if Path(p).is_absolute() else REPO_ROOT / p
                 for p in spec.get('paths', [spec.get('path')])]
        segments = [read_core_actor(path, int(spec['actor'])) for path in paths]
        source_poses = np.concatenate([segment[0] for segment in segments], axis=0)
        start = int(spec.get('start_frame', 0))
        end = int(spec.get('end_frame_inclusive', len(source_poses) - 1))
        if not 0 <= start < end < len(source_poses):
            raise ValueError(f'{spec["id"]}: source frame window is invalid')
        poses = source_poses[start:end + 1]
        source_meta = [segment[1] for segment in segments]
        poses = resample(poses)
        clip_type = spec.get('type', 'walk')
        walking = clip_type != 'idle'
        poses, metrics = canonicalize(poses, walking=walking)
        seam = np.linalg.norm(poses[-1] - poses[0], axis=-1)
        metrics['loop_seam_rms_m'] = float(np.sqrt(np.mean((poses[-1] - poses[0]) ** 2)))
        metrics['loop_seam_max_joint_m'] = float(seam.max())
        if clip_type == 'walk' and metrics['loop_seam_rms_m'] > .06:
            raise ValueError(f'{spec["id"]}: unreviewable loop seam {metrics["loop_seam_rms_m"]:.3f} m')
        transforms = solve_affines(asset, poses)
        affines.append(transforms)
        joint_poses.append(poses.astype('<f4'))
        source = {
            'files': [{'path': repo_path(path), 'sha256': sha256(path)} for path in paths],
            'actor_index': int(spec['actor']),
            'model': 'ARDY Core', 'source_frames': len(source_poses), 'source_fps': SOURCE_FPS,
            'source_window_inclusive': [start, end],
            'requests': [request_evidence(path, meta) for path, meta in zip(paths, source_meta)],
            'generation_seconds': spec.get('generation_seconds'),
            'fresh_for_crowd': bool(spec.get('fresh_for_crowd', False)),
            'review': spec.get('review', 'awaiting visual review'),
        }
        clips.append({'id': spec['id'], 'fps': OUTPUT_FPS, 'frames': len(poses),
                      'offset': offset, 'strideMeters': metrics['source_root_distance_m'],
                      'speed': metrics['source_speed_mps'] if walking else 0.,
                      'type': clip_type, 'source': source,
                      'calibration': metrics})
        offset += len(poses)
    if not clips:
        raise ValueError('At least one generated source is required')
    np.concatenate(affines, axis=0).astype('<f4').tofile(output / 'affine.bin')
    np.concatenate(joint_poses, axis=0).astype('<f4').tofile(output / 'poses.bin')
    rig_provenance = dict(asset.provenance)
    rig_provenance['asset_path'] = repo_path(rig_path)
    manifest = {
        'version': 1, 'upAxis': 'y', 'forward': '+z', 'affineUrl': 'affine.bin', 'posesUrl': 'poses.bin',
        'totalFrames': offset, 'transformShape': [offset, N_BONES, 3, 4],
        'rig': rig_provenance, 'parts': export_parts(asset), 'clips': clips,
        'usage': 'Core generated source poses; authored resampling, root removal and canonical heading; live native segment affine skin',
        'limitations': ['Short generated Core sources require repeated playback in the crowd.',
                        'Core body positions are mapped from 27 to 22 joints; fingers use authored rest pose.',
                        'Floor reference uses source foot joints, not a contact solver.'],
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, separators=(',', ':')))
    return {'clips': clips, 'totalFrames': offset,
            'affine_bytes': (output / 'affine.bin').stat().st_size,
            'poses_bytes': (output / 'poses.bin').stat().st_size,
            'manifest_bytes': (output / 'manifest.json').stat().st_size,
            'rig_sha256': asset.sha256}
