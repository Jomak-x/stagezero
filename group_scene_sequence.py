"""Bounded concurrent Core stages with private native history and real routes.

The stage clock specifies action order, not semantic success or phase-locked
choreography. Returned generated horizons are retained in full, including the
final action, and geometry failures leave their sources and candidates on disk.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import uuid

import numpy as np

from cast_performance import CastPerformance, MAX_FRAMES, validate_actor_ids
from independent_group_motion import _actors, _check_cancel, _folder, _geometry, _prompt, _save, _seed
from interaction_planner import _obstacles, _path
from interaction_scene import scene_objects
from native_pair_transition import core27_to_native22, authored_direction_bridge
from paired_meetup import (MAX_ARRIVAL_ERROR_M, MAX_TARGET_ERROR_M,
                           MIN_ROUTE_SEPARATION_M, _route_separation, _sample, _heading)
from paired_scene import scene_copy
from realtime_navigation import validate_ground_path
from studio_interaction_scene import adapt_studio_scene

HORIZON = 40
SOURCE_FPS = 20
DISPLAY_FPS = 30
MAX_APPROACH_SECONDS = 20
ACTION_HISTORY_FRAMES = 4
FRESH_STAGE_BRIDGE_FRAMES = 21
HISTORY_POLICY = {
    'approach_frames': HORIZON, 'action_frames': ACTION_HISTORY_FRAMES,
    'source': 'unaltered suffix of the same actor’s previous native330 horizon',
    'reason': 'Action carry matches InteractionRuntime’s four-frame internal horizon carry; longer approach history remains for route stability',
    'complete_source_horizons_archived': True,
}
# Same mean joint-displacement limit used for actual Core history seams in
# motion_policy. Reject a discontinuity; never erase source action to hide it.
MAX_SEAM_MEAN_JOINT_M = .15


def _inputs(plan, starts, targets):
    ids = validate_actor_ids([actor['id'] for actor in plan['actors']])
    beats = plan['beats']
    if not isinstance(beats, list) or not 1 <= len(beats) <= 4:
        raise ValueError('Group sequence requires 1–4 all-solo stages')
    stages = []
    for index, beat in enumerate(beats):
        primary = beat.get('actor_ids')
        concurrent = beat.get('concurrent_solos', {})
        if (not isinstance(primary, list) or len(primary) != 1 or primary[0] not in ids
                or not isinstance(concurrent, dict) or primary[0] in concurrent
                or set(concurrent) != set(ids)-set(primary)):
            raise ValueError('Every group stage must give every actor exactly one independent solo')
        seconds = beat['seconds']
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 2 <= seconds <= 10:
            raise ValueError('Group stage duration must be 2–10 seconds')
        prompts = {primary[0]: _prompt(beat['prompt']),
                   **{aid: _prompt(value) for aid, value in concurrent.items()}}
        stages.append({'id': beat.get('id', f'beat-{index+1}'), 'kind': 'independent_motion',
                       'label': f'Group stage {index+1}: '+prompts[primary[0]].replace('\n', ' ').replace('\t', ' ')[:460],
                       'requested_seconds': float(seconds), 'windows': math.ceil(seconds/2),
                       'actor_prompts': prompts})
    if sum(stage['requested_seconds'] for stage in stages) > 30:
        raise ValueError('Group actions exceed the 30-second planned action limit')
    normalized = []
    for label, values in (('starts', starts), ('targets', targets)):
        if not isinstance(values, dict) or set(values) != set(ids):
            raise ValueError(f'Group {label} must cover the exact cast')
        actors = _actors([{'id': aid, 'prompt': stages[0]['actor_prompts'][aid],
                           'x': values[aid]['x'], 'z': values[aid]['z'],
                           'yaw_degrees': values[aid]['yaw_degrees']} for aid in ids])
        if any(abs(actor[key]) > 25 for actor in actors for key in ('x', 'z')):
            raise ValueError('Group anchors exceed the Core worker ±25 m bounds')
        normalized.append({actor['id']: {key: actor[key] for key in ('x', 'z', 'yaw_degrees')}
                           for actor in actors})
    return ids, stages, *normalized


def _routes(scene, ids, starts, targets):
    """Plan static-obstacle paths and a shared, collision-checked arrival clock."""
    adapted = adapt_studio_scene(scene)['scene']
    obstacles = _obstacles(scene_objects(adapted), None, 1.65, .34)
    routes = []
    for aid in ids:
        start, target = starts[aid], targets[aid]
        points = [list(point) for point in _path((start['x'], start['z']),
                                                (target['x'], target['z']), obstacles, .34)]
        if any(abs(value) > 25 for point in points for value in point):
            raise ValueError('Group route exceeds Core worker ±25 m bounds')
        ground = validate_ground_path(adapted, points)
        routes.append({'actor_id': aid, 'points': points,
                       'distance_m': sum(math.dist(a, b) for a, b in zip(points, points[1:])),
                       'initial_yaw': math.radians(start['yaw_degrees']),
                       'arrival_yaw': math.radians(target['yaw_degrees']), 'ground': ground})
    separations = []
    for index, route in enumerate(routes):
        for other in routes[index+1:]:
            distance = _route_separation([route, other])
            separations.append({'actor_ids': [route['actor_id'], other['actor_id']],
                                'minimum_root_separation_m': distance})
            if distance < MIN_ROUTE_SEPARATION_M:
                raise ValueError('Simultaneous group approach routes cross too closely; choose separated starts or meeting marks')
    moving = any(route['distance_m'] > 1e-6 for route in routes)
    turning = any(abs(math.atan2(math.sin(route['arrival_yaw']-route['initial_yaw']),
                                math.cos(route['arrival_yaw']-route['initial_yaw']))) > 1e-6 for route in routes)
    if not moving and not turning:
        return {'routes': routes, 'horizons': [], 'arrival_seconds': 0., 'approach_seconds': 0.,
                'planned_clearance': separations, 'already_at_targets': True}
    travel_windows = max(1, math.ceil(max(route['distance_m'] for route in routes)/.65/2)) if moving else 0
    arrival_seconds = travel_windows*2
    windows = travel_windows+1  # Preserve the full generated stop/settle horizon.
    if windows*2 > MAX_APPROACH_SECONDS:
        raise ValueError('Group approach exceeds the 20-second approach limit')
    horizons = []
    for window in range(windows):
        samples = {0, 7, 15, 23, 31, 39}
        for route in routes:
            distance = 0.
            for a, b in zip(route['points'], route['points'][1:-1]):
                distance += math.dist(a, b)
                frame = round(distance/max(route['distance_m'], 1e-9)*arrival_seconds*SOURCE_FPS)-1
                if frame//HORIZON == window:
                    samples.add(max(0, frame % HORIZON))
        if len(samples) > 24:
            raise ValueError('Group route needs too many turns in one Core horizon')
        root_targets, prompts = {}, {}
        for route in routes:
            aid = route['actor_id']
            root_targets[aid] = []
            for local in sorted(samples):
                second = (window*HORIZON+local+1)/SOURCE_FPS
                point, heading = _sample(route['points'], route['distance_m']*(min(1., second/arrival_seconds) if arrival_seconds else 1.))
                if second >= arrival_seconds:
                    _, prior = _sample(route['points'], max(0., route['distance_m']-1e-6))
                    prior = route['initial_yaw'] if prior is None else prior
                    delta = math.atan2(math.sin(route['arrival_yaw']-prior), math.cos(route['arrival_yaw']-prior))
                    heading = prior+min(1., second-arrival_seconds)*delta
                if heading is None:
                    heading = route['initial_yaw']
                if second <= 1.+1/SOURCE_FPS:
                    alpha = max(0., min(1., second-1/SOURCE_FPS))
                    delta = math.atan2(math.sin(heading-route['initial_yaw']), math.cos(heading-route['initial_yaw']))
                    heading = route['initial_yaw']+alpha*alpha*(3-2*alpha)*delta
                root_targets[aid].append({'frame': local, 'position_xz': point,
                                          'heading': math.atan2(math.sin(heading), math.cos(heading))})
            yaw_gap = math.atan2(math.sin(route['arrival_yaw']-route['initial_yaw']),
                                  math.cos(route['arrival_yaw']-route['initial_yaw']))
            prompts[aid] = ('A person turns in place to face the requested direction, then stands relaxed and ready.'
                            if route['distance_m'] < .05 and abs(yaw_gap) > 1e-6 else
                            'A person stands at their gathering position, relaxed and ready.'
                            if window == travel_windows or route['distance_m'] < .05 else
                            'A person walks forward along the route toward their gathering position and comes to a relaxed stop.')
        horizons.append({'root_targets': root_targets, 'actor_prompts': prompts})
    return {'routes': routes, 'horizons': horizons, 'arrival_seconds': arrival_seconds,
            'approach_seconds': windows*2, 'planned_clearance': separations, 'turn_only': not moving,
            'already_at_targets': False}


def _archive(folder, manifest, request, returned, stage, window):
    """Retain every decoded array before interpreting its shape or metadata."""
    indices = []
    for clip in returned:
        number = len(manifest['sources'])
        path = folder / f'core-{number:03d}.npz'
        arrays = {key: np.asarray(getattr(clip, key)) for key in
                  ('positions', 'rotations', 'native_features') if getattr(clip, key, None) is not None}
        # Raw metadata may itself be malformed; archival must precede rejection.
        arrays['metadata'] = np.array(json.dumps({'request': request,
            'actor_ids': getattr(clip, 'actor_ids', None), 'fps': getattr(clip, 'fps', None),
            'frames': getattr(clip, 'frames', None)}, default=str))
        np.savez_compressed(path, **arrays)
        manifest['sources'].append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'request_id': request['request_id'], 'actor_ids': request['actor_ids'],
            'stage_id': stage['id'], 'window': window, 'source_frames_retained': HORIZON,
            'conditioning_history_frames': len(request.get('history', {}).get('native_features', [[]])[0])})
        indices.append(number)
        _save(folder, manifest)
    return indices


def _horizon(client, folder, manifest, state, *, aid, prompt, initial, seed,
             stage, window, root_targets, cancelled):
    _check_cancel(cancelled)
    request = {'request_id': 'group-sequence-'+uuid.uuid4().hex,
               'stage_kind': 'approach' if state['history'] is None else 'continuation',
               'frames': HORIZON, 'prompt': prompt, 'actor_ids': [aid],
               'actor_prompts': {aid: prompt}, 'seed': seed}
    if state['history'] is None:
        request['initial_placements'] = {aid: {'position_xz': [initial['x'], initial['z']],
                                              'yaw': math.radians(initial['yaw_degrees'])}}
    else:
        # Short native carry is the runtime's existing within-request policy.
        # Keep it across separate action requests too, so the previous full
        # action does not dominate each new prompt. Never change source arrays.
        carry = HORIZON if stage['kind'] == 'approach' else ACTION_HISTORY_FRAMES
        request['history'] = {'native_features': state['history'][-carry:][None].tolist()}
    if root_targets is not None:
        request['root_targets'] = {aid: root_targets}
    manifest['requests'].append(request)
    _save(folder, manifest)
    returned = client.wait(request, cancelled=cancelled)
    indices = _archive(folder, manifest, request, returned, stage, window)
    if len(returned) != 1:
        raise ValueError('Expected exactly one Core horizon')
    clip = returned[0]
    positions = np.asarray(getattr(clip, 'positions', None))
    native = np.asarray(getattr(clip, 'native_features', None))
    if (tuple(getattr(clip, 'actor_ids', ())) != (aid,) or getattr(clip, 'fps', None) != SOURCE_FPS
            or getattr(clip, 'frames', None) != HORIZON or positions.shape != (1, HORIZON, 27, 3)
            or native.shape != (1, HORIZON, 330) or positions.dtype.kind != 'f' or native.dtype.kind != 'f'
            or not np.isfinite(positions).all() or not np.isfinite(native).all()):
        raise ValueError('Core returned incompatible actors, timing, positions or native330 history')
    _check_cancel(cancelled)
    if state['last_pose'] is not None:
        jump = float(np.linalg.norm(positions[0, 0]-state['last_pose'], axis=-1).mean())
        manifest['seams'].append({'actor_id': aid, 'stage_id': stage['id'], 'window': window,
                                 'mean_joint_jump_m': jump, 'maximum_mean_joint_jump_m': MAX_SEAM_MEAN_JOINT_M})
        _save(folder, manifest)
        if jump > MAX_SEAM_MEAN_JOINT_M:
            raise ValueError(f'Core continuation seam for {aid} exceeds the mean joint displacement limit')
    if root_targets is not None:
        error = max(float(np.linalg.norm(positions[0, target['frame'], 0, [0, 2]]-target['position_xz']))
                    for target in root_targets)
        manifest['route_measurements'].append({'actor_id': aid, 'window': window, 'maximum_target_error_m': error})
        _save(folder, manifest)
        if error > MAX_TARGET_ERROR_M:
            raise ValueError(f'Core missed the planned group route by {error:.2f} m')
    state['history'], state['last_pose'] = native[0].copy(), positions[0, -1].copy()
    return positions[0].copy(), indices


def _display(raw):
    frames = len(raw)*DISPLAY_FPS//SOURCE_FPS
    old, new = np.arange(len(raw))/SOURCE_FPS, np.arange(frames)/DISPLAY_FPS
    interpolated = np.stack([np.interp(new, old, column) for column in raw.reshape(len(raw), -1).T], axis=-1)
    return core27_to_native22(interpolated.reshape(frames, 27, 3))


def _shared_action_bridge(left, right, ids, *, maximum_frames, folder, manifest,
                          transition, stage_index, cancelled):
    """Choose one existing bounded bridge duration that passes for every actor."""
    # Keep prompt_scene_builder._bridge's unchanged entry-height guard. A fresh
    # source cannot be connected by inventing a fall or getting-up motion.
    if float(np.max(np.abs(left[-1, :, 0, 1]-right[0, :, 0, 1]))) > .30:
        raise ValueError('Scene beat root-height gap exceeds 0.30 m; cannot invent a fall or get-up transition')
    transition['attempts'] = []
    reasons = []
    for frames in (21, 24, 27, 30):
        if frames > maximum_frames:
            break
        _check_cancel(cancelled)
        attempt = {'frames': frames, 'actor_reports': {}, 'actor_candidate_paths': {},
                   'status': 'checking'}
        transition['attempts'].append(attempt)
        _save(folder, manifest)
        bridges, rejected = [], False
        for actor_index, aid in enumerate(ids):
            _check_cancel(cancelled)
            # The existing anatomical bridge takes two tracks. Duplicate one
            # actor for the solve, just as _bridge does for a one-person beat.
            a = np.repeat(left[:, actor_index:actor_index+1], 2, axis=1)
            b = np.repeat(right[:, actor_index:actor_index+1], 2, axis=1)
            try:
                candidate, report = authored_direction_bridge(a, b, left_fps=30, right_fps=30, frames=frames)
            except ValueError as exc:
                report = {'mechanical_gate_passed': False, 'rejection_reasons': [str(exc)]}
                rejected = True
            else:
                bridge = candidate[:, :1]
                path = folder / f'transition-{stage_index:02d}-{frames:02d}-{aid}.npz'
                np.savez_compressed(path, joints=bridge)
                attempt['actor_candidate_paths'][aid] = str(path)
                bridges.append(bridge)
                rejected |= not report['mechanical_gate_passed']
            attempt['actor_reports'][aid] = report
            reasons.extend(report.get('rejection_reasons', []))
            _save(folder, manifest)
        attempt['status'] = 'rejected_by_mechanical_gates' if rejected else 'mechanical_gates_passed'
        _save(folder, manifest)
        if not rejected:
            transition['actor_reports'] = attempt['actor_reports']
            transition['actor_candidate_paths'] = attempt['actor_candidate_paths']
            return np.concatenate(bridges, axis=1)
    suffix = '; '.join(dict.fromkeys(reasons))
    raise ValueError('Scene beat transition rejected within the 21/24/27/30-frame and total-frame budgets: '+suffix)


def build_group_sequence(client, scene, plan, starts, targets, *, seed, output_root,
                         cancelled=lambda: False, fresh_action_stages=False):
    """Compose shared stages, optionally comparing fresh actions with explicit bridges."""
    if type(fresh_action_stages) is not bool:
        raise ValueError('fresh_action_stages must be boolean')
    _seed(seed)
    ids, stages, starts, targets = _inputs(plan, starts, targets)
    scene = scene_copy(scene)
    folder = _folder(output_root, 'group-sequence')
    manifest = {'version': 1, 'kind': 'concurrent_core_sequence', 'status': 'generating',
                'seed': seed, 'actor_ids': list(ids), 'scene': scene, 'plan': deepcopy(plan),
                'starts': starts, 'targets': targets, 'sources': [], 'requests': [], 'stages': [],
                'seams': [], 'route_measurements': [], 'history_policy': deepcopy(HISTORY_POLICY),
                'fresh_action_stages': fresh_action_stages, 'transitions': [],
                'physical_contact_verified': False,
                'visual_acceptance': 'unverified', 'animation_accepted': False}
    _save(folder, manifest)
    try:
        _check_cancel(cancelled)
        route_plan = _routes(scene, ids, starts, targets)
        manifest['approach'] = route_plan
        if route_plan['horizons']:
            stages.insert(0, {'id': 'group-approach', 'kind': 'approach',
                'label': 'Core approach to separated gathering marks',
                'requested_seconds': route_plan['approach_seconds'],
                'windows': len(route_plan['horizons']), 'actor_prompts': {}})
        transition_count = sum(index > 0 and stage['kind'] != 'approach'
                               for index, stage in enumerate(stages)) if fresh_action_stages else 0
        frames = sum(stage['windows']*60 for stage in stages)+transition_count*FRESH_STAGE_BRIDGE_FRAMES
        manifest['frames'] = frames
        _save(folder, manifest)
        if frames > MAX_FRAMES:
            raise ValueError(f'Complete group sequence needs {frames} frames, exceeding {MAX_FRAMES}; final actions cannot be trimmed')
        states = {aid: {'history': None, 'last_pose': None} for aid in ids}
        manifest['actor_seeds'] = {aid: (seed+1009*index) % 2**32 for index, aid in enumerate(ids)}
        pieces, segments, activities = [], [], []
        cursor = 0
        for stage_index, stage in enumerate(stages):
            fresh_boundary = bool(fresh_action_stages and pieces and stage['kind'] != 'approach')
            source_start = cursor+(FRESH_STAGE_BRIDGE_FRAMES if fresh_boundary else 0)
            count = stage['windows']*60
            record = {**deepcopy(stage), 'start_frame': source_start, 'end_frame_exclusive': source_start+count,
                      'fresh_initialization': fresh_boundary, 'initial_placements': {},
                      'compiled_seconds': count/30, 'actor_sources': {}, 'status': 'generating',
                      'source': 'ardy_core', 'authored_frames': 0, 'visual_acceptance': 'unverified'}
            manifest['stages'].append(record)
            tracks = []
            for aid in ids:
                initial = starts[aid]
                if fresh_boundary:
                    prior = core27_to_native22(states[aid]['last_pose'])
                    initial = {'x': float(prior[0, 0]), 'z': float(prior[0, 2]),
                               'yaw_degrees': math.degrees(_heading(prior))}
                    states[aid] = {'history': None, 'last_pose': None}
                record['initial_placements'][aid] = initial if states[aid]['history'] is None else None
                chunks, source_indices = [], []
                for window in range(stage['windows']):
                    horizon = route_plan['horizons'][window] if stage['kind'] == 'approach' else None
                    prompt = horizon['actor_prompts'][aid] if horizon else stage['actor_prompts'][aid]
                    raw, indices = _horizon(client, folder, manifest, states[aid], aid=aid, prompt=prompt,
                        initial=initial, seed=manifest['actor_seeds'][aid], stage=stage, window=window,
                        root_targets=horizon['root_targets'][aid] if horizon else None, cancelled=cancelled)
                    chunks.append(raw)
                    source_indices.extend(indices)
                raw = np.concatenate(chunks)
                track = _display(raw)
                track_path = folder / f'stage-{stage_index:02d}-{aid}.npz'
                np.savez_compressed(track_path, joints=track)
                record['actor_sources'][aid] = {'source_indices': source_indices,
                    'candidate_path': str(track_path), 'source_frames': len(raw), 'display_frames': len(track),
                    'source_frames_trimmed': 0, 'generated_action_replaced': False}
                _save(folder, manifest)
                tracks.append(track)
            joints = np.stack(tracks, axis=1)
            candidate = folder / f'stage-{stage_index:02d}-candidate.npz'
            np.savez_compressed(candidate, joints=joints)
            record['candidate_path'] = str(candidate)
            _save(folder, manifest)
            if stage['kind'] == 'approach':
                errors = {aid: float(np.linalg.norm(joints[-1, index, 0, [0, 2]]-
                          [targets[aid]['x'], targets[aid]['z']])) for index, aid in enumerate(ids)}
                record['arrival_errors_m'] = errors
                if max(errors.values()) > MAX_ARRIVAL_ERROR_M:
                    raise ValueError('Core did not reach all separated gathering marks')
            if fresh_boundary:
                transition = {'source': 'authored_transition', 'kind': 'transition',
                    'label': 'Authored transition into '+stage['label'], 'model_generated': False,
                    'start_frame': cursor, 'end_frame_exclusive': source_start,
                    'frames': FRESH_STAGE_BRIDGE_FRAMES, 'source_frames_modified': False,
                    'actor_reports': {}, 'actor_candidate_paths': {}, 'status': 'checking'}
                manifest['transitions'].append(transition)
                _save(folder, manifest)
                # `frames` already reserves 21 for every future transition;
                # spend only remaining headroom on this shared bridge.
                bridge = _shared_action_bridge(pieces[-1], joints, ids,
                    maximum_frames=min(30, MAX_FRAMES-frames+FRESH_STAGE_BRIDGE_FRAMES),
                    folder=folder, manifest=manifest, transition=transition,
                    stage_index=stage_index, cancelled=cancelled)
                frames += len(bridge)-FRESH_STAGE_BRIDGE_FRAMES
                manifest['frames'] = frames
                source_start = cursor+len(bridge)
                record['start_frame'], record['end_frame_exclusive'] = source_start, source_start+count
                transition['frames'] = len(bridge)
                transition['end_frame_exclusive'] = source_start
                bridge_path = folder / f'transition-{stage_index:02d}-candidate.npz'
                np.savez_compressed(bridge_path, joints=bridge)
                transition['candidate_path'] = str(bridge_path)
                transition['status'] = 'mechanical_gates_passed'
                pieces.append(bridge)
                segments.append({key: transition[key] for key in
                    ('source', 'kind', 'label', 'start_frame', 'end_frame_exclusive', 'frames')})
                activities.append({'start_frame': cursor, 'end_frame_exclusive': source_start,
                    'active_actor_ids': list(ids), 'contact_actor_ids': [], 'held_actor_ids': []})
                cursor = source_start
                _save(folder, manifest)
            pieces.append(joints)
            segments.append({'label': stage['label'], 'source': 'ardy_core', 'kind': stage['kind'],
                             'start_frame': cursor, 'end_frame_exclusive': cursor+count, 'frames': count})
            activities.append({'start_frame': cursor, 'end_frame_exclusive': cursor+count,
                               'active_actor_ids': list(ids), 'contact_actor_ids': [], 'held_actor_ids': []})
            # Full prefix checks include continuous root clearance at stage seams.
            record['geometry'] = _geometry(np.concatenate(pieces), scene, ids, activities)
            record['status'] = 'compiled_and_geometry_checked'
            cursor += count
            _save(folder, manifest)
        combined = np.concatenate(pieces)
        path = folder / 'candidate.npz'
        np.savez_compressed(path, joints=combined)
        manifest['candidate_path'] = str(path)
        geometry = record['geometry']
        _check_cancel(cancelled)
        coverage = {'requested': plan.get('prompt', ''), 'planned': [stage['id'] for stage in stages],
                    'compiled': [{key: item[key] for key in ('id', 'kind', 'start_frame', 'end_frame_exclusive', 'actor_sources')}
                                 for item in manifest['stages']],
                    'observed': 'unverified', 'final_stage_retained_in_full': True,
                    'already_at_gathering_marks': route_plan['already_at_targets']}
        metadata = {'version': 1, 'model': 'Independent ARDY Core sequence', 'fps': 30, 'frames': frames,
                    'actor_ids': list(ids), 'segments': segments, 'segment_activity': activities,
                    'source_manifest': str(folder/'manifest.json'), 'sources': manifest['sources'],
                    'group_stages': manifest['stages'], 'action_coverage': coverage,
                    'authored_transitions': manifest['transitions'], 'fresh_action_stages': fresh_action_stages,
                    'scene_geometry': geometry, 'physical_contact_verified': False,
                    'visual_acceptance': 'unverified', 'animation_accepted': False,
                    'core_display_sampling': 'Complete 20 fps horizons interpolated to 30 fps per stage; final endpoint held',
                    'continuity': {'native_history': ('private real native330 within each fresh action stage' if fresh_action_stages
                                                       else 'private real native330 per actor across every stage'),
                                   'authored_transitions': bool(manifest['transitions']), 'seams': manifest['seams'],
                                   'history_policy': deepcopy(HISTORY_POLICY)}}
        performance = CastPerformance(ids, combined, metadata=metadata)
        manifest.update(status='accepted_by_geometry_gates', geometry=geometry, action_coverage=coverage)
        _save(folder, manifest)
        return {'status': manifest['status'], 'folder': folder, 'manifest': folder/'manifest.json',
                'performance': performance, 'geometry': geometry, 'routes': route_plan['routes'],
                'approach': route_plan}
    except Exception as exc:
        manifest.update(status='cancelled' if cancelled() else 'rejected', error=str(exc))
        if hasattr(exc, 'geometry_diagnostics'):
            manifest['failure_diagnostics'] = exc.geometry_diagnostics
        _save(folder, manifest)
        raise
