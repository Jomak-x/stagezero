"""Build an explicitly composed ARDY/pair scene and preserve all source data."""
import json
import math
from pathlib import Path
import time
import numpy as np
from native_pair_clip import NativePairClip
from native_pair_transition import build_ardy_pair_context, shared_place_pair


def build_studio_context(pair, client, scene, placement, output_root, *, cancelled=lambda: False):
    if client is None:
        raise RuntimeError('Configure the existing ARDY Core service before adding approach and exit.')
    folder = Path(output_root) / str(time.time_ns())
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    yaw = math.radians(placement['yaw_degrees'])
    translation = np.array([placement['x'], 0., placement['z']])
    source = {'joints': pair.joints, 'metadata': np.array(json.dumps(pair.metadata))}
    if pair.features is not None:
        source['features'] = pair.features
    np.savez_compressed(folder/'source-pair.npz', **source)
    def save_chunk(phase, window, clip, request):
        np.savez_compressed(folder/f'{phase}-{window}.core.npz', positions=clip.positions,
            rotations=clip.rotations, native_features=clip.native_features,
            metadata=np.array(json.dumps({'request': request, 'fps': clip.fps, 'clip_metadata': dict(getattr(clip, 'metadata', {}))})))
    try:
        result = build_ardy_pair_context(pair, client, scene, yaw=yaw, translation=translation,
            cancelled=cancelled, on_core_chunk=save_chunk)
        from native_pair_geometry import check_native_pair_geometry
        geometry = check_native_pair_geometry(result['joints'], scene)
        world = result['joints']
        local = shared_place_pair(world-translation, yaw=-yaw)
        # The native interaction is still the exact original sequence. Undoing
        # its rigid placement numerically must not introduce archive roundoff.
        for segment in result['metadata']['segments']:
            if segment['source'] == 'intergen':
                local[segment['start_frame']:segment['end_frame_exclusive']] = pair.joints
        metadata = result['metadata']
        metadata['context_wall_seconds'] = time.perf_counter() - started
        metadata['source_archive_directory'] = str(folder)
        metadata['source_pair_metadata'] = pair.metadata
        metadata['scene_geometry'] = geometry
        metadata['placement_policy'] = 'World-space planning, shared local playback placement; native pair restored exactly.'
        (folder/'report.json').write_text(json.dumps(metadata, indent=2)+'\n')
        np.savez_compressed(folder/'composed-review.npz', joints=local, metadata=np.array(json.dumps(metadata)))
        return NativePairClip(local, metadata=metadata)
    except Exception as exc:
        (folder/'failure.json').write_text(json.dumps({'error': str(exc), 'source_retained': True}))
        raise
