"""Opt-in Core motion probes. These do not change the production director.

Each actor owns a private native330 continuation history. Raw Core arrays are
archived before validation, and display tracks are composed only after all
source horizons have returned.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import uuid

import numpy as np

from cast_motion_refinement import cast_body_clearance
from cast_performance import CastPerformance, COLORS, decode_project, encode_project
from native_pair_transition import core27_to_native22
from paired_meetup import _heading
from paired_scene import scene_copy
from prompt_scene_builder import _bridge, check_cast_geometry


def _seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be uint32')


def _prompt(value):
    if (not isinstance(value, str) or not 1 <= len(value.strip()) <= 500 or
            any(ord(char) < 32 and char not in '\n\t' for char in value)):
        raise ValueError('Prompt must contain 1–500 readable characters')
    return value.strip()


def _actors(actors):
    if not isinstance(actors, (list, tuple)) or not 1 <= len(actors) <= 10:
        raise ValueError('Independent research supports 1–10 actors')
    out = []
    for actor in actors:
        if not isinstance(actor, dict) or not {'id', 'prompt', 'x', 'z'} <= actor.keys() or not ({'yaw', 'yaw_degrees'} & actor.keys()):
            raise ValueError('Actor requires id, prompt, x, z, and yaw (degrees)')
        aid = actor['id']
        if (not isinstance(aid, str) or not 1 <= len(aid) <= 80 or
                not aid.replace('_', '').replace('-', '').isalnum()):
            raise ValueError('Invalid actor ID')
        position = {}
        for key in ('x', 'z', 'yaw_degrees'):
            value = actor.get(key, actor.get('yaw') if key == 'yaw_degrees' else None)
            if type(value) not in (int, float) or not math.isfinite(value) or (key in ('x', 'z') and abs(value) > 100):
                raise ValueError(f'{aid} {key} must be finite')
            position[key] = float(value)
        out.append({'id': aid, 'prompt': _prompt(actor['prompt']), **position,
                    'name': actor.get('name', aid)})
    if len({a['id'] for a in out}) != len(out):
        raise ValueError('Actor IDs must be unique')
    for i, a in enumerate(out):
        for b in out[i+1:]:
            distance = math.hypot(a['x']-b['x'], a['z']-b['z'])
            if distance < 2:
                raise ValueError('Starting roots must be at least 2 m apart')
    return out


def _folder(output_root, prefix):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    folder = root / f'{prefix}-{uuid.uuid4().hex}'
    folder.mkdir(exist_ok=False)
    return folder


def _save(folder, manifest):
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')


def _check_cancel(cancelled):
    if cancelled():
        raise RuntimeError('Group motion generation cancelled')


def _archive_return(folder, manifest, request, returned):
    # The transport exposes decoded clips; retain every returned array, even if
    # ID, shape, timing or finiteness validation subsequently fails.
    for clip in returned:
        number = len(manifest['sources'])
        path = folder / f'core-{number:03d}.npz'
        arrays = {key: np.asarray(getattr(clip, key)) for key in
                  ('positions', 'rotations', 'native_features')
                  if getattr(clip, key, None) is not None}
        metadata = {'request': request, 'actor_ids': list(getattr(clip, 'actor_ids', ())),
                    'fps': getattr(clip, 'fps', None), 'frames': getattr(clip, 'frames', None)}
        arrays['metadata'] = np.array(json.dumps(metadata, allow_nan=False))
        np.savez_compressed(path, **arrays)
        manifest['sources'].append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                    'actor_ids': metadata['actor_ids'], 'request_id': request['request_id']})
        _save(folder, manifest)


def _core_track(client, folder, manifest, *, actor, frames, seed, cancelled, initial):
    aid, prompt = actor['id'], actor['prompt']
    history, chunks = None, []
    for window in range(math.ceil(frames / 60)):
        _check_cancel(cancelled)
        request = {'request_id': 'group-' + uuid.uuid4().hex,
                   'stage_kind': 'approach' if history is None else 'continuation',
                   'frames': 40, 'prompt': prompt, 'actor_ids': [aid], 'seed': seed,
                   'actor_prompts': {aid: prompt}}
        if history is None:
            request['initial_placements'] = {aid: initial}
        else:
            request['history'] = {'native_features': history[None].tolist()}
        manifest.setdefault('requests', []).append(request)
        _save(folder, manifest)
        returned = client.wait(request, cancelled=cancelled)
        _archive_return(folder, manifest, request, returned)
        if len(returned) != 1:
            raise ValueError('Expected exactly one Core horizon')
        clip = returned[0]
        positions = np.asarray(getattr(clip, 'positions', None))
        native = np.asarray(getattr(clip, 'native_features', None))
        if (tuple(getattr(clip, 'actor_ids', ())) != (aid,) or clip.fps != 20 or clip.frames != 40 or
                positions.shape != (1, 40, 27, 3) or native.shape != (1, 40, 330) or
                positions.dtype.kind != 'f' or native.dtype.kind != 'f' or
                not np.isfinite(positions).all() or not np.isfinite(native).all()):
            raise ValueError('Core returned incompatible actors, timing, positions or native330 history')
        _check_cancel(cancelled)
        chunks.append(positions[0].copy())
        history = native[0].copy()
    raw = np.concatenate(chunks, axis=0)
    old = np.arange(len(raw)) / 20.
    new = np.arange(frames) / 30.
    display = np.stack([np.interp(new, old, column) for column in raw.reshape(len(raw), -1).T], axis=-1)
    return core27_to_native22(display.reshape(frames, 27, 3))


def _body_report(joints, ids, activities, *, changed_actor=None, baseline=None):
    after = cast_body_clearance(joints, ids, activities)
    before = cast_body_clearance(baseline, ids, activities) if baseline is not None else None
    records = []
    for pair, (clearance, mask) in after.items():
        if changed_actor is not None and changed_actor not in pair:
            continue
        selected = clearance[mask]
        if not len(selected):
            continue
        overlap = mask & (clearance < 0)
        worsened = overlap if before is None else overlap & (clearance < before[pair][0] - 1e-4)
        records.append({'actor_ids': [ids[i] for i in pair], 'minimum_clearance_m': float(selected.min()),
                        'overlap_frames': int(overlap.sum()), 'worsened_overlap_frames': int(worsened.sum())})
    return records


def _geometry(joints, scene, ids, activities, *, changed_actor=None, baseline=None):
    body = _body_report(joints, ids, activities, changed_actor=changed_actor, baseline=baseline)
    reports = []
    try:
        if any(item['worsened_overlap_frames'] for item in body):
            raise ValueError('Generated track has unintended sampled body-proxy overlap')
        for activity in activities:
            start, end = activity['start_frame'], activity['end_frame_exclusive']
            reports.append(check_cast_geometry(joints[max(0, start-1):end], scene, ids,
                                               contact_pair=activity.get('contact_actor_ids', ())))
    except ValueError as exc:
        exc.geometry_diagnostics = {'body_clearance': body, 'passed_scene_segments': reports,
                                    'failed_segment': len(reports),
                                    'activity': activities[len(reports)] if len(reports) < len(activities) else None}
        raise
    return {'scene_and_root': reports, 'body_clearance': body,
            'physical_contact_verified': False}


def _overlay_activity(metadata, start_frame, end_frame, third_id):
    """Split source spans at overlay edges without changing pair/contact labels."""
    for key in ('segment_activity', 'segments'):
        original = metadata.get(key)
        if original is None:
            continue
        split = []
        for item in original:
            lo, hi = item['start_frame'], item['end_frame_exclusive']
            edges = sorted({lo, hi, *[edge for edge in (start_frame, end_frame) if lo < edge < hi]})
            for a, b in zip(edges, edges[1:]):
                piece = deepcopy(item)
                piece['start_frame'], piece['end_frame_exclusive'] = a, b
                if key == 'segments':
                    piece['frames'] = b-a
                else:
                    if 'observer_control_spans' in piece:
                        clipped = []
                        for span in piece['observer_control_spans']:
                            left = max(a, span['start_frame'])
                            right = min(b, span['end_frame_exclusive'])
                            if left < right:
                                clipped.append(dict(span, start_frame=left, end_frame_exclusive=right))
                        piece['observer_control_spans'] = clipped
                    if start_frame <= a and b <= end_frame:
                        piece['active_actor_ids'] = list(dict.fromkeys(piece['active_actor_ids'] + [third_id]))
                        if isinstance(piece.get('held_actor_ids'), list):
                            piece['held_actor_ids'] = [aid for aid in piece['held_actor_ids'] if aid != third_id]
                        piece['inactive_motion'] = 'third actor independently generated Core action'
                        piece.setdefault('observer_control_spans', []).append({
                            'actor_id': third_id, 'start_frame': a, 'end_frame_exclusive': b,
                            'source': 'ardy_core_independent_action'})
                split.append(piece)
        metadata[key] = split


def generate_independent_tracks(client, scene, *, actors, seconds, seed, output_root,
                                cancelled=lambda: False):
    """Return an accepted <=3 cast or a separate 4–10 actor research archive.

    Returns a dictionary with status, folder, manifest, performance (<=3),
    research_path (>3), and geometry. Rejected candidates remain in folder.
    """
    _seed(seed)
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 2 <= seconds <= 10:
        raise ValueError('Duration must be 2–10 seconds')
    scene = scene_copy(scene)
    actors = _actors(actors)
    frames = round(seconds * 30)
    folder = _folder(output_root, 'independent')
    ids = tuple(actor['id'] for actor in actors)
    manifest = {'version': 1, 'kind': 'independent_core_tracks', 'status': 'generating',
                'seed': seed, 'frames': frames, 'actor_ids': list(ids), 'actors': actors,
                'scene': scene, 'sources': [], 'physical_contact_verified': False,
                'visual_acceptance': 'unverified'}
    _save(folder, manifest)
    try:
        tracks = []
        manifest['actor_seeds'] = {}
        for index, actor in enumerate(actors):
            actor_seed = (seed + 1009 * index) % 2**32
            manifest['actor_seeds'][actor['id']] = actor_seed
            _save(folder, manifest)
            initial = {'position_xz': [actor['x'], actor['z']], 'yaw': math.radians(actor['yaw_degrees'])}
            tracks.append(_core_track(client, folder, manifest, actor=actor, frames=frames, seed=actor_seed,
                                      cancelled=cancelled, initial=initial))
        joints = np.stack(tracks, axis=1)
        candidate_path = folder / 'candidate.npz'
        np.savez_compressed(candidate_path, joints=joints)
        manifest['candidate_path'] = str(candidate_path)
        _save(folder, manifest)
        activities = [{'start_frame': 0, 'end_frame_exclusive': frames,
                       'active_actor_ids': list(ids), 'contact_actor_ids': []}]
        geometry = _geometry(joints, scene, ids, activities)
        manifest['geometry'] = geometry
        manifest['status'] = 'accepted_by_geometry_gates'
        if len(ids) <= 3:
            segments = [{'label': 'Independent Core motion', 'source': 'ardy_core', 'kind': 'independent_motion',
                         'start_frame': 0, 'end_frame_exclusive': frames, 'frames': frames}]
            metadata = {'version': 1, 'model': 'Independent ARDY Core', 'fps': 30, 'frames': frames,
                        'actor_ids': list(ids), 'segments': segments, 'segment_activity': activities,
                        'source_manifest': str(folder / 'manifest.json'), 'sources': manifest['sources'],
                        'scene_geometry': geometry, 'physical_contact_verified': False,
                        'visual_acceptance': 'unverified', 'animation_accepted': False,
                        'core_display_sampling': '20 fps horizons concatenated, timestamp-interpolated to 30 fps; final endpoint held'}
            performance = CastPerformance(ids, joints, metadata=metadata)
            result = {'status': manifest['status'], 'folder': folder, 'manifest': folder / 'manifest.json',
                      'performance': performance, 'research_path': None, 'geometry': geometry}
        else:
            path = folder / 'research-group.npz'
            doc = {'format': 'stagezero_independent_group_research', 'version': 1,
                   'fps': 30, 'actor_ids': list(ids), 'scene': scene, 'actors': actors,
                   'source_manifest': str(folder / 'manifest.json'), 'geometry': geometry,
                   'physical_contact_verified': False, 'visual_acceptance': 'unverified'}
            np.savez_compressed(path, joints=joints, metadata=np.array(json.dumps(doc, allow_nan=False)))
            result = {'status': manifest['status'], 'folder': folder, 'manifest': folder / 'manifest.json',
                      'performance': None, 'research_path': path, 'geometry': geometry}
        _save(folder, manifest)
        return result
    except Exception as exc:
        manifest.update(status='cancelled' if cancelled() else 'rejected', error=str(exc))
        if hasattr(exc, 'geometry_diagnostics'):
            manifest['failure_diagnostics'] = exc.geometry_diagnostics
        _save(folder, manifest)
        raise


def overlay_third_track(client, source_project_bytes, *, prompt, start_frame, end_frame,
                        seed, output_root, cancelled=lambda: False):
    """Try fresh solo Core motion over an inactive third track of a saved cast.

    Geometry/mechanical rejection returns the original saved project as an
    explicit fallback. Cancellation and worker exceptions propagate. Only a
    three-track cast project with activity metadata is accepted as input.
    """
    _seed(seed)
    prompt = _prompt(prompt)
    source, cast, scene, playhead = decode_project(source_project_bytes)
    if len(source.actor_ids) != 3:
        raise ValueError('Overlay requires a saved three-track cast project')
    if (type(start_frame) is not int or type(end_frame) is not int or
            not 0 <= start_frame < end_frame <= source.frames or end_frame-start_frame < 60 or
            end_frame-start_frame > 300):
        raise ValueError('Overlay interval must be 2–10 seconds inside the saved take')
    activities = source.metadata.get('segment_activity')
    if not isinstance(activities, list) or not activities:
        raise ValueError('Saved cast requires contiguous activity metadata')
    cursor = 0
    third_id = source.actor_ids[2]
    for item in activities:
        start, end = item.get('start_frame'), item.get('end_frame_exclusive')
        if (type(start) is not int or type(end) is not int or start != cursor or
                not start < end <= source.frames or not isinstance(item.get('active_actor_ids'), list) or
                not isinstance(item.get('contact_actor_ids', []), list)):
            raise ValueError('Saved cast activity metadata is not contiguous')
        if end > start_frame and third_id in item['active_actor_ids']:
            raise ValueError('Selected interval is not in a terminal inactive third-track span')
        cursor = end
    if cursor != source.frames:
        raise ValueError('Saved cast activity does not cover the take')
    folder = _folder(output_root, 'overlay')
    source_path = folder / 'original.cast.stagezero.npz'
    source_path.write_bytes(source_project_bytes)
    manifest = {'version': 1, 'kind': 'third_track_overlay', 'status': 'generating',
                'seed': seed, 'prompt': prompt, 'actor_ids': list(source.actor_ids),
                'start_frame': start_frame, 'end_frame_exclusive': end_frame,
                'source_project': str(source_path),
                'source_sha256': hashlib.sha256(source_project_bytes).hexdigest(),
                'sources': [], 'physical_contact_verified': False,
                'visual_acceptance': 'unverified'}
    _save(folder, manifest)
    generated = False
    candidate = None
    try:
        _check_cancel(cancelled)
        pose = source.joints[start_frame, 2]
        initial = {'position_xz': pose[0, [0, 2]].tolist(), 'yaw': _heading(pose)}
        actor = {'id': third_id, 'prompt': prompt}
        replacement = _core_track(client, folder, manifest, actor=actor,
                                  frames=end_frame-start_frame, seed=seed,
                                  cancelled=cancelled, initial=initial)
        generated = True
        candidate = source.joints.copy()
        candidate[start_frame:end_frame, 2] = replacement.astype(candidate.dtype, copy=False)
        raw_path = folder / 'candidate-unbridged.npz'
        np.savez_compressed(raw_path, joints=candidate)
        manifest['unbridged_candidate'] = str(raw_path)
        _save(folder, manifest)
        bridges = []
        if start_frame:
            if start_frame < 2 or end_frame-start_frame < 24:
                raise ValueError('Entry bridge lacks two source samples or interval budget')
            left = source.joints[start_frame-2:start_frame, 2:3]
            right = candidate[start_frame+21:start_frame+23, 2:3]
            entry, report = _bridge(left, right, maximum_frames=21)
            candidate[start_frame:start_frame+21, 2] = entry[:, 0].astype(candidate.dtype, copy=False)
            bridges.append({'kind': 'entry', 'start_frame': start_frame,
                            'end_frame_exclusive': start_frame+21, 'report': report})
        if end_frame < source.frames:
            if source.frames-end_frame < 2 or end_frame-start_frame < 48:
                raise ValueError('Exit bridge lacks two source samples or interval budget')
            left = candidate[end_frame-23:end_frame-21, 2:3]
            right = source.joints[end_frame:end_frame+2, 2:3]
            exit_frames, report = _bridge(left, right, maximum_frames=21)
            candidate[end_frame-21:end_frame, 2] = exit_frames[:, 0].astype(candidate.dtype, copy=False)
            bridges.append({'kind': 'exit', 'start_frame': end_frame-21,
                            'end_frame_exclusive': end_frame, 'report': report})
        candidate_path = folder / 'candidate.npz'
        np.savez_compressed(candidate_path, joints=candidate)
        manifest['candidate_path'] = str(candidate_path)
        manifest['bridges'] = bridges
        _save(folder, manifest)
        if candidate[:, :2].tobytes() != source.joints[:, :2].tobytes():
            raise ValueError('Overlay changed the saved pair payload')
        if (candidate[:start_frame, 2].tobytes() != source.joints[:start_frame, 2].tobytes() or
                candidate[end_frame:, 2].tobytes() != source.joints[end_frame:, 2].tobytes()):
            raise ValueError('Overlay changed the third track outside the selected interval')
        geometry = _geometry(candidate, scene, source.actor_ids, activities,
                             changed_actor=2, baseline=source.joints)
        manifest['geometry'] = geometry
        metadata = deepcopy(source.metadata)
        _overlay_activity(metadata, start_frame, end_frame, third_id)
        metadata['third_track_overlay'] = {'actor_id': third_id, 'prompt': prompt,
            'start_frame': start_frame, 'end_frame_exclusive': end_frame,
            'source_sha256': manifest['source_sha256'], 'source_manifest': str(folder / 'manifest.json'),
            'sources': manifest['sources'], 'bridges': bridges,
            'display_processing': 'Core27 semantic subset to native22; bounded authored bridge at available seams',
            'physical_contact_verified': False, 'visual_acceptance': 'unverified',
            'pair_payload_preserved': True}
        metadata['physical_contact_verified'] = False
        metadata['visual_acceptance'] = 'unverified'
        metadata['animation_accepted'] = False
        result_clip = CastPerformance(source.actor_ids, candidate, metadata=metadata)
        project_bytes = encode_project(result_clip, cast, scene, frame=playhead)
        decoded, _, _, _ = decode_project(project_bytes)
        if (decoded.joints[:, :2].tobytes() != source.joints[:, :2].tobytes() or
                decoded.joints[:start_frame, 2].tobytes() != source.joints[:start_frame, 2].tobytes() or
                decoded.joints[end_frame:, 2].tobytes() != source.joints[end_frame:, 2].tobytes()):
            raise ValueError('Overlay project roundtrip changed exact saved track payloads')
        output_path = folder / 'overlay.cast.stagezero.npz'
        output_path.write_bytes(project_bytes)
        manifest.update(status='accepted_by_geometry_gates', project_path=str(output_path))
        _save(folder, manifest)
        return {'status': manifest['status'], 'folder': folder, 'manifest': folder / 'manifest.json',
                'performance': result_clip, 'project_bytes': project_bytes,
                'project_path': output_path, 'fallback': False, 'geometry': geometry}
    except ValueError as exc:
        if not generated:
            manifest.update(status='rejected', error=str(exc))
            if hasattr(exc, 'geometry_diagnostics'):
                manifest['failure_diagnostics'] = exc.geometry_diagnostics
            _save(folder, manifest)
            raise
        manifest.update(status='rejected_preserved_original', error=str(exc))
        if hasattr(exc, 'geometry_diagnostics'):
            manifest['failure_diagnostics'] = exc.geometry_diagnostics
        _save(folder, manifest)
        return {'status': manifest['status'], 'folder': folder, 'manifest': folder / 'manifest.json',
                'performance': source, 'project_bytes': source_project_bytes,
                'project_path': source_path, 'fallback': True, 'reason': str(exc)}
    except Exception as exc:
        manifest.update(status='cancelled' if cancelled() else 'failed_generation', error=str(exc))
        _save(folder, manifest)
        raise
