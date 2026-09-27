"""Narrow production routing for one whole-beat independent cast action."""
from __future__ import annotations

from copy import deepcopy
import math

from realtime_navigation import validate_ground_path
from scene_composition import validate_scene


def _merged_warnings(metadata, plan):
    existing = metadata.get('warnings', [])
    if not isinstance(existing, list):
        raise ValueError('Generated cast warnings must be a list')
    return list(dict.fromkeys([*existing, *plan['warnings']]))


def independent_starts(plan, scene, *, cancelled=lambda: False):
    """Bind desired starts and find clear automatic starts at least 2 m apart."""
    from interaction_scene import scene_objects
    from interaction_planner import _inside, _obstacles

    scene = validate_scene(scene)
    obstacles = _obstacles(scene_objects(scene), None, 1.65, .4)

    def clear(point):
        if max(map(abs, point)) > 24 or any(_inside(point, box, .4) for box in obstacles):
            return False
        try:
            validate_ground_path(scene, [point], actor_radius_m=.4)
            return True
        except ValueError:
            return False

    selected = {}
    for actor in plan['actors']:
        if actor['start'] is not None:
            point = (actor['start']['x'], actor['start']['z'])
            if not clear(point):
                raise ValueError(f'Desired start for {actor["id"]} overlaps scene geometry or unsupported ground')
            if any(math.dist(point, prior) < 2 for prior in selected.values()):
                raise ValueError('Concurrent starts must be at least 2 metres apart')
            selected[actor['id']] = point

    candidates = [(x * 2.5, z * 2.5) for x in range(-9, 10) for z in range(-9, 10)]
    candidates.sort(key=lambda p: (math.hypot(*p), p))
    for actor in plan['actors']:
        if actor['id'] in selected:
            continue
        for point in candidates:
            if cancelled():
                raise RuntimeError('Concurrent scene staging cancelled')
            if clear(point) and all(math.dist(point, prior) >= 2 for prior in selected.values()):
                selected[actor['id']] = point
                break
        else:
            raise ValueError('No clear 2-metre-separated concurrent starts fit the actual scene')
    return {aid: {'x': float(point[0]), 'z': float(point[1]), 'yaw_degrees': 0.}
            for aid, point in selected.items()}


def build_independent_solos(client, scene, plan, *, seed, output_root, cancelled=lambda: False):
    from independent_group_motion import generate_independent_tracks

    beat = plan['beats'][0]
    if len(plan['beats']) != 1 or len(beat['actor_ids']) != 1 or not beat.get('concurrent_solos'):
        raise ValueError('Independent concurrency requires one whole-performance solo beat')
    starts = independent_starts(plan, scene, cancelled=cancelled)
    prompts = dict(beat['concurrent_solos'])
    prompts[beat['actor_ids'][0]] = beat['prompt']
    actors = [{'id': actor['id'], 'name': actor['name'], 'prompt': prompts[actor['id']],
               **starts[actor['id']]} for actor in plan['actors']]
    generated = generate_independent_tracks(client, scene, actors=actors, seconds=beat['seconds'],
                                            seed=seed, output_root=output_root, cancelled=cancelled)
    clip = generated['performance']
    metadata = deepcopy(clip.metadata)
    metadata.update(prompt=plan['prompt'], title=plan['title'], plan=plan,
                    placement={'starts': starts, 'meeting': None, 'routes': {},
                               'planned_only': True, 'physical_contact_verified': False},
                    concurrency_status='accepted_by_geometry_gates',
                    concurrency_kind='independent_solos',
                    warnings=_merged_warnings(metadata, plan))
    from cast_performance import CastPerformance
    return CastPerformance(clip.actor_ids, clip.joints, metadata=metadata), generated


def overlay_concurrent_third(client, baseline, scene, plan, *, seed, output_root,
                             cancelled=lambda: False):
    """Replace only held actor_3 over the resolved native pair action frames."""
    from cast_performance import CastPerformance, cast_from_performance, encode_project
    from independent_group_motion import overlay_third_track

    beat = plan['beats'][0]
    if (len(plan['beats']) != 1 or beat['actor_ids'] != ['actor_1', 'actor_2']
            or set(beat.get('concurrent_solos', {})) != {'actor_3'}
            or baseline.actor_ids != ('actor_1', 'actor_2', 'actor_3')):
        raise ValueError('Pair concurrency requires actor_1 and actor_2 with actor_3 solo')
    pair_actions = [segment for segment in baseline.metadata.get('segments', [])
                    if segment.get('kind') == 'paired_action']
    if len(pair_actions) != 1:
        raise ValueError('Pair concurrency requires one resolved native paired action interval')
    action = pair_actions[0]
    start, end = action['start_frame'], action['end_frame_exclusive']
    if end - start != round(beat['seconds'] * 30) or not 60 <= end - start <= 210:
        raise ValueError('Resolved paired action interval does not match the concurrent 2–7 second beat')
    original = encode_project(baseline, cast_from_performance(baseline), scene)
    generated = overlay_third_track(client, original, prompt=beat['concurrent_solos']['actor_3'],
                                    start_frame=start, end_frame=end, seed=seed,
                                    output_root=output_root, cancelled=cancelled)
    clip = generated['performance']
    metadata = deepcopy(clip.metadata)
    metadata['warnings'] = _merged_warnings(metadata, plan)
    metadata['concurrency_kind'] = 'pair_plus_independent_third'
    if generated['fallback']:
        reason = generated['reason']
        metadata['concurrency_fallback'] = f'Requested actor_3 simultaneous solo was not achieved: {reason}'
        metadata['warnings'].append(metadata['concurrency_fallback'])
        metadata['plan']['warnings'].append(metadata['concurrency_fallback'])
        metadata['concurrency_status'] = 'fallback_original_pair_only'
    else:
        metadata['concurrency_status'] = 'accepted_by_geometry_gates'
    return CastPerformance(clip.actor_ids, clip.joints, metadata=metadata), generated
