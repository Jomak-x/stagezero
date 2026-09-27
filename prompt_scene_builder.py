"""Compile a semantic plan into an archived, measured 1–3 actor performance.

Core's native330 history is used only between consecutive Core horizons.
InterGen's native262 features are archived, never converted to Core features.
Inactive tracks hold a previously generated pose and are explicitly disclosed.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
import uuid

import numpy as np

from interaction_scene_collision import scene_collision
from native_pair_clip import NativePairClip, MAX_FRAMES
from native_pair_geometry import check_native_pair_geometry
from native_pair_transition import (authored_direction_bridge, core27_to_native22,
                                    core_to_pair_anatomy, shared_place_pair)
from paired_meetup import build_meetup, plan_meetup, _minimum_separation, _heading, _checked_generated_geometry
from realtime_navigation import validate_ground_path
from scene_objects import make_object
from studio_interaction_scene import adapt_studio_scene

ARRIVAL_STANDOFF_M = .60
IDLE_ROUTE_PADDING_M = .25
INITIAL_HEADING_RAMP_SECONDS = 1.


def select_native_wait_pose(joints, actor_index):
    """Choose an unchanged native pose only when it is clearly better for waiting.

    These are observable posture/stability heuristics, not a contact solver or
    an animation generator. The first frame remains the fallback.
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
    speeds = np.array([np.mean(steps[max(0, frame-2):min(len(steps), frame+3)]) for frame in range(len(actor))])
    floor = float(np.quantile(actor[:, [10, 11], 1], .05))
    support = np.stack([actor[:, [7, 10], 1].min(axis=1), actor[:, [8, 11], 1].min(axis=1)], axis=1)
    foot_height = support.max(axis=1)-floor
    leg_length = np.stack([np.linalg.norm(actor[:, hip]-actor[:, knee], axis=1)
                           +np.linalg.norm(actor[:, knee]-actor[:, ankle], axis=1)
                           for hip, knee, ankle in ((1, 4, 7), (2, 5, 8))], axis=1).mean(axis=1)
    standing = (actor[:, 0, 1]-support.mean(axis=1))/np.maximum(leg_length, 1e-6)
    scores = angles/30+hands_above_hips*3+hand_reach*2+speeds*.6+foot_height*3
    eligible = ((angles <= 20) & (hands_above_hips <= .08) & (hand_reach <= .40)
                & (speeds <= .35) & (foot_height <= .10) & (standing >= .70))
    candidates = np.flatnonzero(eligible)
    chosen = int(candidates[np.argmin(scores[candidates])]) if len(candidates) else 0
    improvement = max(.15, float(scores[0])*.20)
    if scores[0]-scores[chosen] < improvement:
        chosen = 0

    def metrics(index):
        return {'score': float(scores[index]), 'torso_tilt_degrees': float(angles[index]),
                'mean_hand_height_above_hips_m': float(hands_above_hips[index]),
                'mean_hand_horizontal_reach_m': float(hand_reach[index]),
                'local_max_foot_xz_speed_mean_m_s': float(speeds[index]),
                'highest_support_above_native_floor_m': float(foot_height[index]),
                'standing_leg_extension_ratio': float(standing[index])}

    report = {'criterion': 'lower hands, upright torso, standing leg extension and locally stable feet',
        'source_frame': chosen, 'baseline_frame': 0, 'changed_from_first_frame': chosen != 0,
        'eligible_frame_count': len(candidates), 'minimum_score_improvement': improvement,
        'baseline_metrics': metrics(0), 'selected_metrics': metrics(chosen),
        'thresholds': {'maximum_torso_tilt_degrees': 20, 'maximum_hand_height_above_hips_m': .08,
                       'maximum_hand_horizontal_reach_m': .40, 'maximum_local_foot_speed_m_s': .35,
                       'maximum_support_above_native_floor_m': .10, 'minimum_standing_leg_extension_ratio': .70},
        'score_weights': {'torso_tilt_degrees': 1/30, 'hand_height_m': 3, 'hand_reach_m': 2,
                          'local_foot_speed_m_s': .6, 'support_height_m': 3},
        'joint_pose_modified': False, 'physical_contact_verified': False,
        'display_policy': 'one exact generated pose held stationary; no breathing or synthesized idle motion'}
    return actor[chosen].copy(), report


def pair_source_prompt(prompt, actor_ids, actors):
    """Translate explicit cast names/indices into this two-person source's roles."""
    aliases = {}
    names = {actor['id']: actor.get('name', '') for actor in actors}
    for index, aid in enumerate(actor_ids):
        role = 'the first person' if index == 0 else 'the second person'
        candidates = [aid, names.get(aid, '')]
        match = re.fullmatch(r'actor_(\d+)', aid)
        if match:
            number = match.group(1)
            candidates += [f'Person {number}', f'Person{number}', f'Actor {number}', f'Actor{number}']
        for name in candidates:
            if name.strip():
                aliases[name.casefold()] = role
    if not aliases:
        return prompt
    pattern = r'(?<!\w)('+ '|'.join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True)) +r')(?!\w)'
    result = re.sub(pattern, lambda match: aliases[match.group(0).casefold()], prompt, flags=re.I)
    if len(result) > 500:
        raise ValueError('Pair role translation exceeds the native prompt limit')
    return result


def align_pair(source, prior):
    """Shared orientation-preserving 2D root Procrustes; preserve native Y."""
    a = np.asarray(source)[0, :, 0][:, [0, 2]]
    b = np.asarray(prior)[-1, :, 0][:, [0, 2]]
    if min(np.linalg.norm(a[1]-a[0]), np.linalg.norm(b[1]-b[0])) < .05:
        raise ValueError('Pair roots are too close for an unambiguous shared placement')
    da, db = a[1]-a[0], b[1]-b[0]
    yaw = math.atan2(da[1], da[0])-math.atan2(db[1], db[0])
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    rotated = shared_place_pair(source, yaw=yaw)
    center = rotated[0, :, 0].mean(axis=0)
    target = b.mean(axis=0)
    translation = [float(target[0]-center[0]), 0., float(target[1]-center[2])]
    return shared_place_pair(source, yaw=yaw, translation=translation), {
        'method': 'shared XZ rigid root alignment; no scale or Y offset',
        'yaw_radians': yaw, 'translation': translation}


def _scene_with_idle_roots(scene, poses, active):
    """Transient navigation proxies; never mutate or render the authored scene."""
    copied = deepcopy(scene)
    copied.setdefault('objects', [])
    proxies = []
    for aid, pose in poses.items():
        if aid in active:
            continue
        root = pose[0]
        proxy = make_object('crate', len(copied['objects']), position=[float(root[0]), .9, float(root[2])])
        proxy['id'] = 'scene-idle-'+aid
        while any(obj['id'] == proxy['id'] for obj in copied['objects']):
            proxy['id'] += '-proxy'
        proxy['name'] = 'Temporary idle performer clearance'
        proxy['size'] = [.50, 1.8, .50]
        copied['objects'].append(proxy)
        proxies.append({'actor_id': aid, 'root_xz': [float(root[0]), float(root[2])],
                        'proxy_width_m': .50, 'planning_only': True})
    return copied, proxies


def select_initial_staging(pair, scene, plan, placement, poses, active, *, cancelled=lambda: False,
                           candidate_reports=None):
    """Bounded source-aware automatic starts; explicit user starts never move.

    This is CPU planning only. The same native source, meeting anchor, scene
    checks and route clearance gates apply to every candidate.
    """
    reports = candidate_reports if candidate_reports is not None else []
    explicit = {actor['id'] for actor in plan['actors'] if actor.get('start') is not None}
    meeting = placement['meeting']
    rotated = shared_place_pair(pair.joints, yaw=math.radians(meeting.get('yaw_degrees', 0.)))
    center = rotated[0, :, 0].mean(axis=0)
    placed = rotated+np.array([meeting['x']-center[0], 0., meeting['z']-center[2]])
    roots = placed[0, :, 0][:, [0, 2]]
    delta = roots[1]-roots[0]
    length = float(np.linalg.norm(delta))
    if length < .05:
        raise ValueError('Native pair entry roots are too close for automatic staging')
    axis = delta/length
    lateral = np.array([-axis[1], axis[0]])
    midpoint = roots.mean(axis=0)
    idle_ids = [aid for aid in poses if aid not in active]
    candidates = []
    for mode in ('outward', 'backward'):
        for distance in (.8, 1.2, 1.8):
            base = deepcopy(placement['starts'])
            for index, aid in enumerate(active):
                if aid in explicit:
                    continue
                heading = _heading(placed[0, index])
                direction = axis*(-1 if index == 0 else 1) if mode == 'outward' else -np.array([math.sin(heading), math.cos(heading)])
                start = roots[index]+direction*distance
                base[aid] = {'x': float(start[0]), 'z': float(start[1]), 'yaw_degrees': math.degrees(heading)}
            for idle_side in (0, -1, 1):
                candidate = deepcopy(base)
                if idle_side:
                    for aid in idle_ids:
                        if aid not in explicit:
                            idle = midpoint+lateral*2.4*idle_side
                            candidate[aid] = {'x': float(idle[0]), 'z': float(idle[1])}
                candidates.append((f'{mode}-{distance:g}-idle-{idle_side}', candidate))
    candidates.append(('original', deepcopy(placement['starts'])))
    seen = set()
    best = None
    for label, starts in candidates:
        if cancelled():
            raise RuntimeError('Scene generation cancelled')
        encoded = json.dumps(starts, sort_keys=True)
        if encoded in seen:
            continue
        seen.add(encoded)
        record = {'candidate': label, 'starts': starts, 'accepted': False}
        reports.append(record)
        candidate_poses = {aid: pose.copy() for aid, pose in poses.items()}
        for aid in idle_ids:
            root = candidate_poses[aid][0]
            candidate_poses[aid] += np.array([starts[aid]['x']-root[0], 0., starts[aid]['z']-root[2]])
        try:
            planning_scene, proxies = _scene_with_idle_roots(scene, candidate_poses, active)
            route = plan_meetup(pair, planning_scene, actor_ids=active, starts=[starts[aid] for aid in active],
                                meeting=meeting, entry_policy='continuous', speed_mps=.85,
                                arrival_standoff_m=ARRIVAL_STANDOFF_M, idle_route_padding_m=IDLE_ROUTE_PADDING_M)
            all_ids = tuple(actor['id'] for actor in plan['actors'])
            world = np.stack([placed[:, active.index(aid)] if aid in active else
                              np.repeat(candidate_poses[aid][None], len(placed), axis=0) for aid in all_ids], axis=1)
            check_cast_geometry(world, scene, all_ids, contact_pair=active)
        except ValueError as exc:
            record['rejection'] = str(exc)
            continue
        record['accepted'] = True
        record['approach_seconds'] = route['approach_seconds']
        record['minimum_planned_root_separation_m'] = route['minimum_planned_root_separation_m']
        record['total_route_distance_m'] = sum(item['distance_m'] for item in route['routes'])
        # Prefer the shortest measured route, with source-oriented candidates
        # preceding generic marks on ties. Explicit starts stay fixed throughout.
        score = (route['approach_seconds'], record['total_route_distance_m'])
        if best is None or score < best[0]:
            best = (score, label, starts, candidate_poses, planning_scene, proxies, record)
    if best is not None:
        _, label, starts, candidate_poses, planning_scene, proxies, record = best
        record['selected'] = True
        selected = deepcopy(placement)
        selected['starts'] = starts
        selected['initial_source_staging'] = {'candidate': label, 'explicit_starts_preserved': sorted(explicit),
            'candidate_count': len(reports), 'planning_only': True,
            'selection': 'shortest approach duration, then total route distance; source-oriented starts win ties'}
        return selected, candidate_poses, planning_scene, proxies
    reason = reports[-1].get('rejection', 'no feasible source-aware start') if reports else 'no staging candidates'
    raise ValueError('No bounded automatic staging fits the unchanged native pair: '+reason)


def select_later_meeting(pair, scene, plan, placement, poses, active, prior, *,
                         cancelled=lambda: False, candidate_reports=None, idle_route_padding_m=IDLE_ROUTE_PADDING_M,
                         excluded_meetings=()):
    """Select clear shared native placement and measured travel; idle poses stay fixed."""
    reports = candidate_reports if candidate_reports is not None else []
    base, base_alignment = align_pair(pair.joints, prior)
    if float(np.max(np.abs(prior[-1, :, 0, 1]-base[0, :, 0, 1]))) > .30:
        raise ValueError('Scene beat root-height gap exceeds 0.30 m; cannot invent a fall or get-up transition')
    center = base[0, :, 0].mean(axis=0)[[0, 2]]
    explicit = plan.get('meeting') is not None
    if explicit:
        requested = placement['meeting']
        offsets = [('explicit-meeting', np.array([requested['x'], requested['z']])-center)]
    else:
        offsets = [('aligned-meeting', np.zeros(2))]
        idle = [pose[0, [0, 2]] for aid, pose in poses.items() if aid not in active]
        away = center-np.mean(idle, axis=0) if idle else np.array([1., 0.])
        away = away/np.linalg.norm(away) if np.linalg.norm(away) > .05 else np.array([1., 0.])
        side = np.array([-away[1], away[0]])
        directions = [away, side, -side, -away, (away+side)/math.sqrt(2),
                      (away-side)/math.sqrt(2), (-away+side)/math.sqrt(2), (-away-side)/math.sqrt(2)]
        for radius in (.6, 1.2, 1.8):
            offsets.extend((f'offset-{radius:g}-direction-{index}', direction*radius)
                           for index, direction in enumerate(directions))
    planning_scene, proxies = _scene_with_idle_roots(scene, poses, active)
    starts = [{'x': float(prior[-1, index, 0, 0]), 'z': float(prior[-1, index, 0, 2]),
               'yaw_degrees': math.degrees(_heading(prior[-1, index]))} for index in range(2)]
    all_ids = tuple(actor['id'] for actor in plan['actors'])

    def world(joints):
        return np.stack([joints[:, active.index(aid)] if aid in active else
                         np.repeat(poses[aid][None], len(joints), axis=0) for aid in all_ids], axis=1)

    best = None
    for label, shift in offsets:
        if cancelled():
            raise RuntimeError('Scene generation cancelled')
        placed = base+np.array([shift[0], 0., shift[1]])
        meeting = {'x': float(center[0]+shift[0]), 'z': float(center[1]+shift[1]),
                   'yaw_degrees': math.degrees(base_alignment['yaw_radians'])}
        record = {'candidate': label, 'meeting': meeting, 'meeting_offset_xz': shift.tolist(),
                  'explicit_meeting_preserved': explicit, 'idle_route_padding_m': idle_route_padding_m, 'accepted': False}
        reports.append(record)
        if not explicit and any(math.hypot(meeting['x']-old['x'], meeting['z']-old['z']) < .05 for old in excluded_meetings):
            record['rejection'] = 'Previous generated attempt at this automatic meeting collided with an idle actor'
            continue
        bridge, bridge_report, direct_rejection, route = None, None, None, None
        gaps = np.linalg.norm(prior[-1, :, 0][:, [0, 2]]-placed[0, :, 0][:, [0, 2]], axis=-1)
        try:
            # The complete native action must clear both the authored scene and
            # static idle proxies, not merely its first-frame root targets.
            check_native_pair_geometry(placed, planning_scene, actor_ids=active)
            check_cast_geometry(world(placed), scene, all_ids, contact_pair=active)
            if float(gaps.max()) <= .30:
                try:
                    bridge, bridge_report = _bridge(prior, placed)
                    check_native_pair_geometry(bridge, planning_scene, actor_ids=active)
                    check_cast_geometry(world(bridge), scene, all_ids, contact_pair=active)
                except ValueError as exc:
                    bridge = None
                    direct_rejection = str(exc)
            if bridge is None:
                route = plan_meetup(pair, planning_scene, actor_ids=active, starts=starts, meeting=meeting,
                    entry_policy='continuous', speed_mps=.85, arrival_standoff_m=ARRIVAL_STANDOFF_M,
                    idle_route_padding_m=idle_route_padding_m, initial_heading_ramp_seconds=INITIAL_HEADING_RAMP_SECONDS)
        except ValueError as exc:
            record['rejection'] = str(exc)
            continue
        record['accepted'] = True
        record['approach_seconds'] = 0. if route is None else route['approach_seconds']
        record['total_route_distance_m'] = 0. if route is None else sum(r['distance_m'] for r in route['routes'])
        score = (record['approach_seconds'], record['total_route_distance_m'], float(np.linalg.norm(shift)))
        if best is None or score < best[0]:
            alignment = dict(base_alignment)
            alignment['translation'] = (np.asarray(base_alignment['translation'])+np.array([shift[0], 0., shift[1]])).tolist()
            alignment['meeting_offset_xz'] = shift.tolist()
            best = (score, record, {'placed': placed, 'alignment': alignment, 'meeting': meeting,
                'starts': starts, 'planning_scene': planning_scene, 'idle_proxies': proxies,
                'bridge': bridge, 'bridge_report': bridge_report, 'direct_rejection': direct_rejection,
                'root_gaps_m': gaps.tolist(), 'route_plan': route})
        if bridge is not None:
            # No travel can beat a mechanically checked, collision-clear direct
            # bridge at the first aligned candidate (zero offset).
            if not np.any(shift):
                break
    if best is None:
        reason = reports[-1].get('rejection', 'no feasible meeting') if reports else 'no meeting candidates'
        raise ValueError('No bounded later meeting preserves the native pair and idle cast: '+reason)
    best[1]['selected'] = True
    return best[2]


def generate_later_approach(pair, client, scene, plan, placement, poses, active, prior, selected, *,
                            seed, cancelled=lambda: False, on_progress=None, candidate_reports=None,
                            attempt_reports=None, source_paths=lambda: []):
    """At most three real Core attempts; retry only measured idle collisions."""
    attempts = attempt_reports if attempt_reports is not None else []
    excluded = []
    for attempt, padding in enumerate((IDLE_ROUTE_PADDING_M, .40, .55), 1):
        if cancelled():
            raise RuntimeError('Scene generation cancelled')
        if attempt > 1:
            selected = select_later_meeting(pair, scene, plan, placement, poses, active, prior,
                cancelled=cancelled, candidate_reports=candidate_reports, idle_route_padding_m=padding,
                excluded_meetings=excluded)
        record = {'attempt': attempt, 'maximum_attempts': 3, 'idle_route_padding_m': padding,
                  'meeting': selected['meeting'], 'starts': selected['starts'], 'status': 'generating',
                  'history_policy': 'fresh Core initial placements; real native330 continuation only within this attempt',
                  'native_pair_frames_modified': False}
        attempts.append(record)
        previous_archives = set(source_paths())
        tick = time.monotonic()
        if on_progress:
            on_progress({'phase': 'pair_approach_attempt', 'attempt': attempt, 'maximum_attempts': 3,
                         'actor_ids': list(active), 'idle_route_padding_m': padding})
        try:
            result = build_meetup(pair, client, selected['planning_scene'], actor_ids=active,
                starts=selected['starts'], meeting=selected['meeting'], entry_policy='continuous', speed_mps=.85,
                arrival_standoff_m=ARRIVAL_STANDOFF_M, idle_route_padding_m=padding,
                initial_heading_ramp_seconds=INITIAL_HEADING_RAMP_SECONDS,
                seed=seed, cancelled=cancelled, on_progress=on_progress)
            entry, report = _bridge(prior, result['joints'][:2], maximum_frames=30)
            _checked_generated_geometry(entry, selected['planning_scene'], active,
                                        context={'stage': 'prior_to_core_entry', 'fps': 30})
            record['status'] = 'accepted'
            return selected, result, entry, report
        except ValueError as exc:
            collisions = getattr(exc, 'idle_collision_report', None)
            if not collisions:
                record.update(status='rejected', error=str(exc))
                raise
            record.update(status='rejected_idle_collision', error=str(exc), idle_collisions=collisions)
            excluded.append(selected['meeting'])
            if attempt == 3:
                raise
        except Exception as exc:
            record.update(status='rejected', error=str(exc))
            raise
        finally:
            record['seconds'] = time.monotonic()-tick
            record['source_core_archives'] = [path for path in source_paths() if path not in previous_archives]
    raise RuntimeError('Unreachable Core retry state')


def _bridge(left, right, *, maximum_frames=21):
    """Reuse the measured pair bridge for one or two active actors."""
    count = left.shape[1]
    if float(np.max(np.abs(left[-1, :, 0, 1]-right[0, :, 0, 1]))) > .30:
        raise ValueError('Scene beat root-height gap exceeds 0.30 m; cannot invent a fall or get-up transition')
    a = np.repeat(left, 2, axis=1) if count == 1 else left
    b = np.repeat(right, 2, axis=1) if count == 1 else right
    candidates = tuple(frames for frames in (12, 15, 18, 21, 24, 27, 30) if frames <= maximum_frames)
    if not candidates or maximum_frames not in (21, 30):
        raise ValueError('Bridge search is bounded to 21 or 30 frames')
    for frames in candidates:
        try:
            out, report = authored_direction_bridge(a, b, left_fps=30, right_fps=30, frames=frames)
        except ValueError:
            if frames == candidates[-1]:
                raise
            continue
        if report['mechanical_gate_passed']:
            return out[:, :count], report
    raise ValueError('Scene beat transition rejected: '+'; '.join(report['rejection_reasons']))


def check_cast_geometry(joints, scene, actor_ids, *, contact_pair=()):
    """Check all scene solids/floor and continuous unpaired root clearance."""
    adapted = adapt_studio_scene(scene)
    reports, clearance = [], []
    for index, aid in enumerate(actor_ids):
        points = joints[:, index]
        ground = validate_ground_path(adapted['scene'], points[:, 0][:, [0, 2]], actor_radius_m=.28)
        collision = scene_collision(points, 'native22', adapted['scene'], adapted['affordances'])
        if collision['total_collision_frames']:
            raise ValueError(f'{aid} motion overlaps scene geometry; scene rejected')
        reports.append({'actor_id': aid, 'ground': ground, 'collision': collision})
    for a in range(len(actor_ids)):
        for b in range(a+1, len(actor_ids)):
            if set((actor_ids[a], actor_ids[b])) == set(contact_pair):
                continue
            distance = _minimum_separation(joints[:, [a, b], 0][:, :, [0, 2]])
            clearance.append({'actor_ids': [actor_ids[a], actor_ids[b]], 'minimum_root_separation_m': distance})
            if distance < .55:
                raise ValueError(f'{actor_ids[a]} and {actor_ids[b]} unpaired paths overlap ({distance:.2f} m); scene rejected')
    return {'actors': reports, 'unpaired_clearance': clearance,
            'intended_contact_pair': list(contact_pair), 'physical_contact_verified': False}


class PromptSceneBuilder:
    def __init__(self, prompt, planner, provider, core_client, output_root, seed=42):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('A scene prompt is required')
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Seed must be uint32')
        self.prompt, self.planner, self.provider = prompt, planner, provider
        self.core_client, self.output_root, self.seed = core_client, Path(output_root), seed

    def __call__(self, scene, cancelled=lambda: False, on_progress=None):
        from cast_performance import CastPerformance
        from prompt_scene_plan import auto_place

        started = time.monotonic()
        folder = self.output_root / ('scene-'+uuid.uuid4().hex)
        folder.mkdir(parents=True, exist_ok=False)
        manifest = {'version': 1, 'prompt': self.prompt, 'seed': self.seed,
                    'status': 'generating', 'sources': [], 'timings': [], 'beats': []}
        archive_index = 0
        external_cancelled = cancelled
        stop = threading.Event()
        source_lock = threading.Lock()
        pair_records, pair_errors, pair_futures = {}, [], {}
        executor = None

        def cancelled():
            return stop.is_set() or external_cancelled()

        def save():
            (folder/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')

        def check_cancel():
            if cancelled():
                with source_lock:
                    error = pair_errors[0] if pair_errors else None
                if error is not None and not external_cancelled():
                    raise error
                raise RuntimeError('Scene generation cancelled')

        def collect_pair_record(index, *, used=False):
            # Only the main thread mutates the manifest. The worker writes its
            # independent NPZ + JSON sidecar before starting another provider call.
            with source_lock:
                record = dict(pair_records[index])
            existing = next((item for item in manifest['sources'] if item['path'] == record['path']), None)
            if existing is not None:
                existing['used_in_performance'] |= used
                return
            record['used_in_performance'] = used
            manifest['sources'].append(record)
            manifest['timings'].append({'stage': 'intergen_generation', 'beat_index': index, 'seconds': record['seconds']})

        def finish_prefetch():
            nonlocal executor
            if executor is not None:
                # Active provider calls receive stop through their normal bounded
                # HTTP/SSH cancellation path. Join before returning so no hidden
                # generation survives a failed/cancelled performance.
                for future in pair_futures.values():
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                executor = None
            with source_lock:
                completed = sorted(pair_records)
            for index in completed:
                collect_pair_record(index)

        def generate_pair(index, beat, actors):
            try:
                check_cancel()
                active = list(beat['actor_ids'])
                source_prompt = pair_source_prompt(beat['prompt'], active, actors)
                tick = time.monotonic()
                pair = self.provider.generate(source_prompt, (self.seed+index) % 2**32,
                    round(beat['seconds']*30), cancelled=cancelled)
                elapsed = time.monotonic()-tick
                # Capture last_raw_archive while this sole worker still owns the
                # provider sequence. No subsequent generate can replace its bytes.
                raw = getattr(self.provider, 'last_raw_archive', None)
                path = folder/f'pair-{index:02d}.npz'
                if raw is not None:
                    path.write_bytes(raw)
                else:
                    arrays = {'joints': pair.joints, 'metadata': np.array(json.dumps(pair.metadata))}
                    if pair.features is not None:
                        arrays['features'] = pair.features
                    np.savez_compressed(path, **arrays)
                record = {'source': 'intergen', 'path': str(path), 'seconds': elapsed,
                    'beat_index': index, 'replayed_native_source': bool(getattr(self.provider, 'replay', False)),
                    'actor_ids': active, 'source_prompt': source_prompt, 'planned_prompt': beat['prompt'],
                    'source_role_map': [{'source_actor_index': role, 'actor_id': aid} for role, aid in enumerate(active)],
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                path.with_suffix('.json').write_text(json.dumps(record, indent=2)+'\n')
                with source_lock:
                    pair_records[index] = record
                return pair
            except Exception as exc:
                with source_lock:
                    if not stop.is_set() and not external_cancelled():
                        pair_errors.append(exc)
                stop.set()
                raise

        def await_pair(index):
            while True:
                check_cancel()
                try:
                    pair = pair_futures[index].result(timeout=.1)
                    collect_pair_record(index, used=True)
                    save()
                    if (not isinstance(pair, NativePairClip) or pair.metadata.get('model') != 'InterGen'
                            or pair.segments is not None or pair.frames != round(beats[index]['seconds']*30)):
                        raise ValueError('Pair provider must return original native InterGen frames')
                    return pair
                except FutureTimeout:
                    continue

        def progress(phase, **values):
            check_cancel()
            if on_progress:
                on_progress({'phase': phase, **values})

        def archive_core(clip, request, elapsed):
            nonlocal archive_index
            path = folder/f'core-{archive_index:03d}.npz'; archive_index += 1
            arrays = {key: getattr(clip, key) for key in ('positions', 'rotations', 'native_features')
                      if getattr(clip, key, None) is not None}
            arrays['metadata'] = np.array(json.dumps({'request': request, 'actor_ids': list(clip.actor_ids), 'fps': clip.fps}))
            np.savez_compressed(path, **arrays)
            manifest['sources'].append({'source': 'ardy_core', 'path': str(path), 'seconds': elapsed,
                                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            save()

        class ArchivingClient:
            def wait(_, request, cancelled):
                check_cancel()
                if self.core_client is None:
                    raise RuntimeError('Configure ARDY Core before generating a scene')
                tick = time.monotonic()
                returned = self.core_client.wait(request, cancelled=cancelled)
                elapsed = time.monotonic()-tick
                # Archive before any schema, geometry or transition rejection.
                for clip in returned:
                    archive_core(clip, request, elapsed)
                manifest['timings'].append({'stage': 'core_generation', 'request_id': request['request_id'], 'seconds': elapsed})
                return returned

        client = ArchivingClient()
        histories = {}
        try:
            progress('planning')
            tick = time.monotonic()
            plan = self.planner.plan(self.prompt, scene, cancelled=cancelled)
            manifest['plan'] = plan
            manifest['timings'].append({'stage': 'planning', 'seconds': time.monotonic()-tick})
            ids = tuple(actor['id'] for actor in plan['actors'])
            if not 1 <= len(ids) <= 3 or len(set(ids)) != len(ids):
                raise ValueError('Scene requires one to three distinct actors')
            beats = plan['beats']
            if not 1 <= len(beats) <= 4:
                raise ValueError('Scene plan requires one to four beats')
            minimum_frames = 0
            for beat in beats:
                active = beat['actor_ids']
                seconds = beat['seconds']
                if (len(active) not in (1, 2) or len(set(active)) != len(active)
                        or not set(active) <= set(ids) or type(seconds) not in (int, float)
                        or not math.isfinite(seconds) or not 0 < seconds <= 30):
                    raise ValueError('Invalid scene beat actors or duration')
                if len(active) == 2 and not 1 <= seconds <= 7:
                    raise ValueError('Native pair beats must last 1–7 seconds; split the plan into ordered beats')
                minimum_frames += round(seconds*30) if len(active) == 2 else math.ceil(seconds/2)*60
            if minimum_frames > MAX_FRAMES:
                raise ValueError('Scene exceeds the 1000-frame (33.3-second) limit; request a shorter scene')
            tick = time.monotonic()
            placement = auto_place(plan, scene, cancelled=cancelled)
            manifest['placement'] = placement
            manifest['timings'].append({'stage': 'placement', 'seconds': time.monotonic()-tick})
            save(); check_cancel()
            paired_beats = [(index, beat) for index, beat in enumerate(beats) if len(beat['actor_ids']) == 2]
            if paired_beats:
                if self.provider is None:
                    raise RuntimeError('Configure native InterGen before generating a paired beat')
                executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='scene-intergen')
                for index, beat in paired_beats:
                    pair_futures[index] = executor.submit(generate_pair, index, beat, plan['actors'])
            manifest['prefetch'] = {'maximum_concurrent_pair_calls': 1, 'scheduled_pair_sources': len(paired_beats),
                                    'core_horizons': 'sequential native330 history'}
            poses, parts, segments, activities, geometry, boundaries = {}, [], [], [], [], []
            total = 0

            def core_windows(active, prompt, windows, *, initialize=False):
                chunks = []
                continuous = all(aid in histories for aid in active) and not initialize
                for window in range(windows):
                    progress('core', actor_ids=list(active), completed_windows=window, total_windows=windows)
                    request = {'request_id': 'scene-'+uuid.uuid4().hex, 'stage_kind': 'continuation' if continuous else 'approach',
                               'frames': 40, 'prompt': prompt, 'actor_ids': list(active), 'seed': self.seed,
                               'actor_prompts': {aid: prompt for aid in active}}
                    if continuous:
                        request['history'] = {'native_features': np.stack([histories[aid] for aid in active]).tolist()}
                    else:
                        request['initial_placements'] = {}
                        for aid in active:
                            if aid in poses:
                                pose = poses[aid]
                                xz, yaw = pose[0, [0, 2]].tolist(), _heading(pose)
                            else:
                                start = placement['starts'][aid]
                                xz, yaw = [start['x'], start['z']], math.radians(start.get('yaw_degrees', 0.))
                            request['initial_placements'][aid] = {'position_xz': xz, 'yaw': yaw}
                    returned = client.wait(request, cancelled=cancelled)
                    if len(returned) != 1:
                        raise ValueError('Expected exactly one real Core horizon')
                    clip = returned[0]
                    if (tuple(clip.actor_ids) != tuple(active) or clip.fps != 20 or clip.frames != 40
                            or clip.positions.shape != (len(active), 40, 27, 3)
                            or clip.native_features is None or clip.native_features.shape != (len(active), 40, 330)
                            or not np.isfinite(clip.positions).all() or not np.isfinite(clip.native_features).all()):
                        raise ValueError('Core returned incompatible actors, timing or native history')
                    check_cancel()
                    chunks.append(clip.positions.transpose(1, 0, 2, 3))
                    histories.update({aid: clip.native_features[index].copy() for index, aid in enumerate(active)})
                    continuous = True
                raw = np.concatenate(chunks)
                old, new = np.arange(len(raw))/20., np.arange(len(raw)*3//2)/30.
                resampled = np.stack([np.interp(new, old, column) for column in raw.reshape(len(raw), -1).T], axis=-1).reshape(-1, len(active), 27, 3)
                return resampled

            def append(active_joints, active, source, kind, label, *, contact=False):
                nonlocal total
                count = len(active_joints)
                if total+count > MAX_FRAMES:
                    raise ValueError(f'Scene needs {(total+count)/30:.1f} seconds, exceeding the 1000-frame (33.3-second) limit; request a shorter scene')
                world = np.empty((count, len(ids), 22, 3), dtype=float)
                for index, aid in enumerate(ids):
                    world[:, index] = active_joints[:, active.index(aid)] if aid in active else poses[aid]
                tick = time.monotonic()
                checked = np.concatenate([parts[-1][-1:], world]) if parts else world
                report = check_cast_geometry(checked, scene, ids, contact_pair=active if contact else ())
                manifest['timings'].append({'stage': 'geometry', 'seconds': time.monotonic()-tick})
                segments.append({'source': source, 'kind': kind, 'label': label,
                                 'start_frame': total, 'end_frame_exclusive': total+count, 'frames': count})
                activities.append({'start_frame': total, 'end_frame_exclusive': total+count,
                                   'active_actor_ids': list(active), 'held_actor_ids': [a for a in ids if a not in active],
                                   'inactive_motion': 'explicit stationary hold of last real generated pose'})
                geometry.append(report); parts.append(world); total += count
                poses.update({aid: world[-1, i].copy() for i, aid in enumerate(ids)})

            # A future native participant waits in an unchanged generated pose
            # from their own source, preserving anatomy without synthetic idle motion.
            inactive = [aid for aid in ids if aid not in beats[0]['actor_ids']]
            core_inactive = []
            for aid in inactive:
                incoming = next(((index, beat) for index, beat in paired_beats if aid in beat['actor_ids']), None)
                if incoming is None:
                    core_inactive.append(aid)
                    continue
                source_index, incoming_beat = incoming
                native = await_pair(source_index)
                role = incoming_beat['actor_ids'].index(aid)
                start = placement['starts'][aid]
                yaw = math.radians(start.get('yaw_degrees', 0.))
                waiting_pose, waiting_report = select_native_wait_pose(native.joints, role)
                rotated = shared_place_pair(np.repeat(waiting_pose[None, None], 2, axis=1), yaw=yaw)[0, 0]
                translation = np.array([start['x']-rotated[0, 0], 0., start['z']-rotated[0, 2]])
                poses[aid] = rotated+translation
                manifest.setdefault('idle_initializations', []).append({'actor_ids': [aid], 'source': 'intergen',
                    'source_beat_index': source_index, 'source_actor_index': role,
                    'source_frame': waiting_report['source_frame'], 'waiting_pose_selection': waiting_report,
                    'yaw_radians': yaw, 'translation': translation.tolist(),
                    'display_policy': 'explicit stationary hold of a real incoming native pose at the assigned start'})
            for start in range(0, len(core_inactive), 2):
                active = core_inactive[start:start+2]
                seed_poses = core27_to_native22(core_windows(active, 'A person stands still in a relaxed neutral stance.', 1, initialize=True))
                for index, aid in enumerate(active):
                    poses[aid] = seed_poses[-1, index].copy()
                    histories.pop(aid, None)
                manifest.setdefault('idle_initializations', []).append({'actor_ids': active, 'generated_core_frames': 40,
                    'display_policy': 'hold final generated pose; initialization motion is archived, not played'})

            previous_active = ()
            for beat_index, beat in enumerate(beats):
                tick = time.monotonic()
                active = list(beat['actor_ids'])
                progress('beat', beat_index=beat_index, total_beats=len(beats), actor_ids=active, prompt=beat['prompt'])
                for aid in list(histories):
                    if aid not in active or aid not in previous_active:
                        histories.pop(aid, None)
                if len(active) == 1:
                    was_continuous = active[0] in histories
                    core = core_windows(active, beat['prompt'], math.ceil(beat['seconds']/2))
                    display = core27_to_native22(core)
                    refinement = None
                    if active[0] in poses:
                        # A Core display may be fitted to the last real actor anatomy;
                        # duplicated arguments only satisfy a two-track math helper.
                        fitted, refinement = core_to_pair_anatomy(np.repeat(core, 2, axis=1),
                            np.repeat(poses[active[0]][None, None], 2, axis=1))
                        display = fitted[:, :1]
                    if parts and not was_continuous:
                        left = parts[-1][-2:, [ids.index(a) for a in active]]
                        bridge, report = _bridge(left, display)
                        append(bridge, active, 'authored_transition', 'transition', 'Authored transition into Core solo beat')
                        boundaries.append(report)
                    append(display, active, 'ardy_core', 'solo_action', beat['prompt'])
                    manifest['beats'].append({'beat_id': beat['id'], 'requested_seconds': beat['seconds'],
                        'source_seconds': len(core)/30, 'duration_policy': 'complete sequential 40-frame Core horizons at 20 fps',
                        'refinement': refinement})
                else:
                    progress('pair_generation', beat_index=beat_index, actor_ids=active)
                    pair = await_pair(beat_index)
                    check_cancel()
                    if (not isinstance(pair, NativePairClip) or pair.metadata.get('model') != 'InterGen'
                            or pair.segments is not None or pair.frames != round(beat['seconds']*30)):
                        raise ValueError('Pair provider must return original native InterGen frames')
                    if beat_index == 0:
                        candidates = manifest.setdefault('initial_staging_candidates', [])
                        original_idle_roots = {aid: pose[0, [0, 2]].copy() for aid, pose in poses.items()}
                        staging_tick = time.monotonic()
                        placement, poses, planning_scene, idle_proxies = select_initial_staging(
                            pair, scene, plan, placement, poses, active, cancelled=cancelled, candidate_reports=candidates)
                        manifest['timings'].append({'stage': 'native_initial_staging', 'seconds': time.monotonic()-staging_tick})
                        for entry in manifest.get('idle_initializations', []):
                            entry['automatic_hold_placement_adjustments_xz'] = {
                                aid: (poses[aid][0, [0, 2]]-original_idle_roots[aid]).tolist() for aid in entry['actor_ids']}
                        manifest['placement'] = placement
                        save()
                        result = build_meetup(pair, client, planning_scene, actor_ids=active,
                            starts=[placement['starts'][aid] for aid in active], meeting=placement['meeting'],
                            entry_policy='continuous', speed_mps=.85, seed=self.seed, cancelled=cancelled,
                            arrival_standoff_m=ARRIVAL_STANDOFF_M,
                            idle_route_padding_m=IDLE_ROUTE_PADDING_M,
                            on_progress=on_progress)
                        for segment in result['metadata']['segments']:
                            chunk = result['joints'][segment['start_frame']:segment['end_frame_exclusive']]
                            append(chunk, active, segment['source'], segment['kind'], segment['label'],
                                   contact=segment['kind'] in ('transition', 'paired_action'))
                        manifest['beats'].append({'beat_id': beat['id'], 'meeting': result['metadata'],
                                                  'idle_navigation_proxies': idle_proxies})
                    else:
                        prior = parts[-1][-2:, [ids.index(a) for a in active]]
                        candidates = manifest.setdefault('later_meeting_candidates', {}).setdefault(beat['id'], [])
                        staging_tick = time.monotonic()
                        selected = select_later_meeting(pair, scene, plan, placement, poses, active, prior,
                            cancelled=cancelled, candidate_reports=candidates)
                        manifest['timings'].append({'stage': 'later_meeting_planning', 'beat_id': beat['id'],
                                                   'seconds': time.monotonic()-staging_tick})
                        save()
                        placed, alignment = selected['placed'], selected['alignment']
                        bridge, report = selected['bridge'], selected['bridge_report']
                        if bridge is not None:
                            append(bridge, active, 'authored_transition', 'transition', 'Authored transition between scene beats', contact=True)
                            boundaries.append(report)
                            append(placed, active, 'intergen', 'paired_action', beat['prompt'], contact=True)
                            manifest['beats'].append({'beat_id': beat['id'], 'shared_placement': alignment,
                                                      'navigation': 'nearby mechanically checked direct transition'})
                        else:
                            # A short authored bridge is not a travel controller.
                            # Reset native330 history and request real Core travel
                            # from current world roots into the complete next pair.
                            progress('pair_approach', beat_index=beat_index, actor_ids=active)
                            attempts = manifest.setdefault('core_approach_attempts', {}).setdefault(beat['id'], [])
                            selected, result, entry, report = generate_later_approach(
                                pair, client, scene, plan, placement, poses, active, prior, selected,
                                seed=(self.seed+beat_index) % 2**32, cancelled=cancelled, on_progress=on_progress,
                                candidate_reports=candidates, attempt_reports=attempts,
                                source_paths=lambda: [item['path'] for item in manifest['sources'] if item['source'] == 'ardy_core'])
                            alignment = selected['alignment']
                            append(entry, active, 'authored_transition', 'transition', 'Authored transition into Core travel for next pair')
                            boundaries.append(report)
                            for segment in result['metadata']['segments']:
                                chunk = result['joints'][segment['start_frame']:segment['end_frame_exclusive']]
                                append(chunk, active, segment['source'], segment['kind'], segment['label'],
                                       contact=segment['kind'] in ('transition', 'paired_action'))
                            manifest['beats'].append({'beat_id': beat['id'], 'shared_placement': alignment,
                                'navigation': 'real Core approach from current poses', 'root_gaps_m': selected['root_gaps_m'],
                                'direct_bridge_rejection': selected['direct_rejection'], 'meeting': result['metadata'],
                                'idle_navigation_proxies': selected['idle_proxies']})
                    for aid in active:
                        histories.pop(aid, None)
                previous_active = tuple(active)
                manifest['timings'].append({'stage': 'beat_total', 'beat_id': beat['id'], 'seconds': time.monotonic()-tick})
                save()
            check_cancel()
            finish_prefetch()
            joints = np.concatenate(parts)
            manifest.update(status='complete', frames=len(joints), duration_seconds=len(joints)/30,
                            wall_seconds=time.monotonic()-started)
            save()
            metadata = {'version': 1, 'model': 'ARDY Core + InterGen' if any(s['source'] == 'intergen' for s in segments) else 'ARDY Core',
                        'prompt': self.prompt, 'title': plan.get('title', self.prompt), 'fps': 30, 'frames': len(joints),
                        'actor_ids': list(ids), 'plan': plan, 'placement': placement, 'segments': segments,
                        'segment_activity': activities, 'scene_geometry': geometry, 'transition_boundaries': boundaries,
                        'source_manifest': str(folder/'manifest.json'), 'sources': manifest['sources'],
                        'stage_timings': manifest['timings'], 'wall_seconds': manifest['wall_seconds'],
                        'prefetch': manifest['prefetch'],
                        'source_pair_frames_modified': False, 'visual_acceptance': 'unverified',
                        'core_display_sampling': '20 fps source resampled to 30 fps by timestamp interpolation; complete 40-frame solo horizons preserved in raw archives',
                        'physical_contact_verified': False, 'animation_accepted': False,
                        'features': 'Separate real source archives only; no cross-model features or joint three-person model.'}
            result = CastPerformance(actor_ids=ids, joints=joints, fps=30, metadata=metadata)
            progress('complete', frames=len(joints), duration_seconds=len(joints)/30)
            return result
        except Exception as exc:
            stop.set()
            finish_prefetch()
            with source_lock:
                if pair_errors and not external_cancelled():
                    exc = pair_errors[0]
            manifest.update(status='cancelled' if external_cancelled() else 'rejected', error=str(exc), wall_seconds=time.monotonic()-started)
            save()
            raise exc
