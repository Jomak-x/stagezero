"""Bounded, measured Core approaches into an unchanged native paired take."""
from __future__ import annotations
import hashlib
import math
import re
import uuid
from dataclasses import replace
from collections.abc import Mapping
import numpy as np
from interaction_planner import _obstacles, _path, _intersects, _inside
from interaction_scene import scene_objects
from interaction_scene_collision import scene_collision
from native_pair_clip import NativePairClip, MAX_FRAMES
from native_pair_geometry import check_native_pair_geometry
from native_pair_transition import shared_place_pair, core_to_pair_anatomy, authored_direction_bridge, core27_to_native22
from realtime_navigation import validate_ground_path
from studio_interaction_scene import adapt_studio_scene

FPS, HORIZON, BLEND_FRAMES = 20, 40, 21
# Acceptance bounds, not expected model precision or collision radii.
MAX_TARGET_ERROR_M, MAX_ARRIVAL_ERROR_M, MIN_ROUTE_SEPARATION_M = .8, .4, .55
MAX_ENTRY_ROOT_HEIGHT_GAP_M = .30

# Classify only a short source prompt and send Core a fixed motion cue. The
# source text is never copied into a worker request's actor prompts.
_GUARD_INTENT = re.compile(r'\b(?:spar(?:ring)?|boxers?|kickbox(?:ing)?|punch(?:es|ing)?|jab(?:s|bing)?|fight(?:s|ing)?)\b|\bboxing\s+(?:match|bout|practice|spar)\b|\b(?:are|start|begin|people|persons|both|two)\s+boxing\b')
_GREETING_INTENT = re.compile(r'\b(?:greet(?:s|ed|ing)?|hello|handshak(?:e|es|ing)|shake hands?|wave(?:s|d|ing)?|hug(?:s|ged|ging)?|embrace(?:s|d)?|high[- ]five)\b')
_ENTRY_CUES = {
    'ready': (
        'A person walks smoothly along a curved route toward their partner, turns while walking, and arrives facing their partner with hands raised in a ready stance.',
        'A person faces their partner, feet planted and hands raised in a ready stance.',
        'A person stands in place and turns to face their partner, relaxed and ready.'),
    'relaxed': (
        'A person walks smoothly along a curved route toward their partner, turns while walking, and arrives facing their partner with arms relaxed at their sides.',
        'A person faces their partner, feet planted with arms relaxed at their sides, ready to greet.',
        'A person stands in place and turns to face their partner, feet planted with arms relaxed at their sides.'),
    'guard': (
        'A person walks smoothly along a curved route toward their partner, turns while walking, and arrives facing their partner with hands up in a boxing guard.',
        'A person faces their partner, feet planted with hands up in a boxing guard.',
        'A person stands in place and turns to face their partner, feet planted with hands up in a boxing guard.'),
}


def _entry_posture(prompt):
    """Choose one bounded Core cue; keep legacy readiness for unknown actions."""
    if not isinstance(prompt, str):
        return 'ready'
    short = prompt[:500].casefold()
    if _GUARD_INTENT.search(short):
        return 'guard'
    if _GREETING_INTENT.search(short):
        return 'relaxed'
    return 'ready'


def _checked_generated_geometry(world, scene, actor_ids, *, context):
    """Keep the existing gate, attaching precise idle collisions for bounded retries."""
    try:
        return check_native_pair_geometry(world, scene, actor_ids=actor_ids)
    except ValueError as exc:
        if 'scene-idle-' in str(exc):
            adapted = adapt_studio_scene(scene)
            collisions = []
            for index, aid in enumerate(actor_ids):
                report = scene_collision(world[:, index], 'native22', adapted['scene'], adapted['affordances'])
                collisions.extend(dict(item, actor_id=aid, context=context) for item in report['per_object']
                                  if item['object_id'].startswith('scene-idle-') and item['collision_frames'])
            exc.idle_collision_report = collisions
        raise


def _number(v, name, lo, hi):
    if type(v) not in (int, float) or not math.isfinite(v) or not lo <= v <= hi:
        raise ValueError(f'{name} must be finite and between {lo:g} and {hi:g}')
    return float(v)


def _heading(pose):
    across = pose[1]-pose[2]
    if np.linalg.norm(across[[0, 2]]) < 1e-6:
        raise ValueError('Native entry pose has an ambiguous heading')
    return float(math.atan2(-across[2], across[0]))


def _sample(points, distance):
    for a, b in zip(points, points[1:]):
        length = math.dist(a, b)
        if length > 1e-9 and distance < length:
            return [a[i]+(b[i]-a[i])*max(0., distance)/length for i in range(2)], math.atan2(b[0]-a[0], b[1]-a[1])
        distance -= length
    return list(points[-1]), None


def _minimum_separation(paths):
    """Exact closest approach between synchronous piecewise linear roots."""
    d = paths[:, 0]-paths[:, 1]
    minimum = float(np.linalg.norm(d, axis=-1).min())
    for a, b in zip(d, d[1:]):
        delta = b-a
        squared = float(delta @ delta)
        u = 0. if squared < 1e-12 else float(np.clip(-(a @ delta)/squared, 0., 1.))
        minimum = min(minimum, float(np.linalg.norm(a+u*delta)))
    return minimum


def _schedule(routes, starts, actor_ids, speed, maximum, pair_frames, entry_policy='settled', initial_heading_ramp_seconds=0., entry_posture='ready'):
    """Keep the same common duration and model target sampling for every route."""
    travel_windows = max(1, math.ceil(max(r['distance_m'] for r in routes)/speed/2.))
    arrival_seconds, windows = travel_windows*2., travel_windows+1
    approach_seconds = windows*2.
    if entry_policy == 'continuous':
        arrival_seconds = max(2., math.ceil(max(r['distance_m'] for r in routes)/speed*FPS)/FPS)
        approach_seconds = arrival_seconds+1.
        windows = math.ceil(approach_seconds/2.)
    frames = round(approach_seconds*FPS)*3//2+BLEND_FRAMES+pair_frames
    if approach_seconds > maximum or frames > MAX_FRAMES:
        raise ValueError('Meeting route exceeds the 20-second approach or 1000-frame take limit; move starts closer')
    horizons = []
    for window in range(windows):
        samples = {0, 7, 15, 23, 31, 39}
        for route in routes:
            distance = 0.
            for a, b in zip(route['points'], route['points'][1:-1]):
                distance += math.dist(a, b)
                frame = round(distance/max(route['distance_m'], 1e-9)*arrival_seconds*FPS)-1
                if frame//HORIZON == window:
                    samples.add(max(0, frame % HORIZON))
        if len(samples) > 24:
            raise ValueError('Meeting route needs too many turns in one Core horizon')
        targets, prompts = {}, {}
        for route in routes:
            aid = route['actor_id']; targets[aid] = []
            for local in sorted(samples):
                second = (window*HORIZON+local+1)/FPS
                point, heading = _sample(route['points'], route['distance_m']*min(1., second/arrival_seconds))
                if entry_policy == 'continuous' and second > arrival_seconds-2.:
                    # Spread the final turn across the moving approach, instead
                    # of asking Core to reverse heading at a short route corner.
                    turn_start = max(0., arrival_seconds-2.)
                    _, prior = _sample(route['points'], route['distance_m']*turn_start/arrival_seconds)
                    prior = route['initial_yaw'] if prior is None else prior
                    delta = math.atan2(math.sin(route['arrival_yaw']-prior), math.cos(route['arrival_yaw']-prior))
                    alpha = min(1., (second-turn_start)/(arrival_seconds-.15-turn_start))
                    heading = prior+(alpha*alpha*(3-2*alpha))*delta
                if second >= arrival_seconds and entry_policy == 'continuous':
                    heading = route['arrival_yaw']
                elif second >= arrival_seconds:
                    _, prior = _sample(route['points'], max(0., route['distance_m']-1e-6))
                    prior = route['initial_yaw'] if prior is None else prior
                    delta = math.atan2(math.sin(route['arrival_yaw']-prior), math.cos(route['arrival_yaw']-prior))
                    heading = prior+min(1., (second-arrival_seconds)/1.)*delta
                heading = route['arrival_yaw'] if heading is None else math.atan2(math.sin(heading), math.cos(heading))
                if initial_heading_ramp_seconds and second <= initial_heading_ramp_seconds+1/FPS:
                    # The first Core target is at t=1/FPS. Match the actual
                    # supplied prior heading there, then turn smoothly onto the
                    # route instead of requesting a frame-zero yaw discontinuity.
                    alpha = max(0., min(1., (second-1/FPS)/initial_heading_ramp_seconds))
                    smooth = alpha*alpha*(3-2*alpha)
                    prior = route['initial_yaw']
                    delta = math.atan2(math.sin(heading-prior), math.cos(heading-prior))
                    heading = math.atan2(math.sin(prior+smooth*delta), math.cos(prior+smooth*delta))
                targets[aid].append({'frame': local, 'position_xz': point, 'heading': heading})
            if entry_policy == 'continuous':
                prompts[aid] = _ENTRY_CUES[entry_posture][0 if window*2 < arrival_seconds else 1]
            else:
                prompts[aid] = (_ENTRY_CUES[entry_posture][2] if window == travel_windows or route['distance_m'] < .05
                                else 'A person walks forward along the route toward their partner and comes to a relaxed stop.')
        horizons.append({'root_targets': targets, 'actor_prompts': prompts})
    samples = [[[s['x'], s['z']] for s in starts]]
    for horizon in horizons:
        values = [horizon['root_targets'][aid] for aid in actor_ids]
        samples.extend([[a['position_xz'], b['position_xz']] for a, b in zip(*values)])
    minimum = _minimum_separation(np.asarray(samples))
    return horizons, arrival_seconds, windows, frames, minimum


def _route_separation(routes):
    """Check every true route corner at shared progress, not just Core samples."""
    times = {0., 1.}
    for route in routes:
        distance = 0.
        for a, b in zip(route['points'], route['points'][1:]):
            distance += math.dist(a, b)
            if route['distance_m'] > 1e-9:
                times.add(min(1., distance/route['distance_m']))
    paths = [[_sample(r['points'], r['distance_m']*t)[0] for r in routes] for t in sorted(times)]
    return _minimum_separation(np.asarray(paths))


def _idle_clearance_path(start, end, physical_obstacles, routing_obstacles):
    """Depart a safe close start outward before using extra idle routing margin.

    This only changes route targets. The short departure clears the original
    physical obstacles and monotonically increases distance from its idle actor.
    Every generated pose still passes the original scene/body collision checks.
    """
    enclosing = [box for box in routing_obstacles if _inside(start, box, .34)]
    if not enclosing:
        return _path(start, end, routing_obstacles, .34), []
    if (len(enclosing) != 1 or not enclosing[0].object_id.startswith('scene-idle-')
            or any(_inside(start, box, .34) for box in physical_obstacles)):
        raise ValueError('Actor or route target overlaps scene geometry')
    box = enclosing[0]
    direction = np.asarray(start)-box.center
    length = float(np.linalg.norm(direction))
    if length < .05:
        raise ValueError('Cannot leave an ambiguous idle-actor overlap')
    direction /= length
    for distance in (.15, .30, .45, .60, .80, 1.):
        exit_point = tuple(np.asarray(start)+direction*distance)
        if any(_inside(exit_point, obstacle, .34) for obstacle in routing_obstacles):
            continue
        if any(_intersects(start, exit_point, obstacle, .34) for obstacle in physical_obstacles):
            continue
        try:
            tail = _path(exit_point, end, routing_obstacles, .34)
        except ValueError:
            continue
        return [start]+tail, [{'idle_object_id': box.object_id, 'start': list(start),
                               'exit': list(exit_point), 'distance_m': distance,
                               'physical_clearance_preserved': True}]
    raise ValueError('No bounded outward departure clears the padded idle route')


def _entry_route(points, arrival_yaw, scene, obstacles):
    """Round clear corners and reach the native pose along its forward tangent.

    All sampled curve chords get the same obstacle/floor checks as straight
    routes. Obstacle corners that cannot be rounded retain their safe routing.
    This affects Core targets only; no generated or native pose is edited.
    """
    end = np.asarray(points[-1], dtype=float)
    if math.dist(points[0], end) < .05:
        return points
    lead = end-.5*np.array([math.sin(arrival_yaw), math.cos(arrival_yaw)])
    tail = _path(tuple(points[-2]), tuple(lead), obstacles, .34)
    # A blocked final tangent is not silently replaced by a different heading.
    if any(_intersects(tuple(lead), tuple(end), box, .34) for box in obstacles):
        raise ValueError('Native entry tangent overlaps scene geometry')
    routed = [np.asarray(p, dtype=float) for p in points[:-2]+list(tail)+[end]]
    rounded = [routed[0].tolist()]
    for before, corner, after in zip(routed, routed[1:], routed[2:]):
        incoming, outgoing = corner-before, after-corner
        a, b = float(np.linalg.norm(incoming)), float(np.linalg.norm(outgoing))
        if min(a, b) < 1e-6:
            continue
        cut = min(.65, a*.4, b*.4)
        left, right = corner-incoming/a*cut, corner+outgoing/b*cut
        curve = [(1-t)**2*left+2*(1-t)*t*corner+t*t*right for t in np.linspace(0, 1, 9)]
        clear = not any(_intersects(tuple(x), tuple(y), box, .34)
                        for x, y in zip(curve, curve[1:]) for box in obstacles)
        if clear:
            try:
                validate_ground_path(scene, [p.tolist() for p in curve])
            except ValueError:
                clear = False
        rounded.extend([p.tolist() for p in curve] if clear else [corner.tolist()])
    rounded.append(end.tolist())
    validate_ground_path(scene, rounded)
    return rounded


def _alternate_routes(routes, starts, scene, obstacles):
    """At most six deterministic lateral detours per actor, preserving roles.

    Each leg uses the existing obstacle router and continuous floor check. Try
    combinations by total travel distance; never exchange actor destinations.
    """
    choices = [[route] for route in routes]
    for index, route in enumerate(routes):
        if route.get('idle_departure'):
            # Do not replace a verified outward departure with a shortcut through
            # the extra routing envelope. The other actor can still be rerouted.
            continue
        start, end = np.asarray(route['points'][0]), np.asarray(route['points'][-1])
        delta = end-start
        length = float(np.linalg.norm(delta))
        if length < 1e-6:
            continue
        lateral = np.array([-delta[1], delta[0]])/length
        midpoint = (start+end)/2
        for radius in (.8, 1.4, 2.):
            for sign in (1., -1.):
                waypoint = midpoint+lateral*radius*sign
                try:
                    first = _path(tuple(start), tuple(waypoint), obstacles, .34)
                    last = _path(tuple(waypoint), tuple(end), obstacles, .34)
                    points = [list(p) for p in first+last[1:]]
                    if route.get('entry_policy') == 'continuous':
                        points = _entry_route(points, route['arrival_yaw'], scene, obstacles)
                    if any(abs(v) > 25 for point in points for v in point):
                        continue
                    ground = validate_ground_path(scene, points)
                except ValueError:
                    continue
                distance = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
                _, heading = _sample(points, 0.)
                initial = route['initial_yaw'] if 'yaw_degrees' in starts[index] or heading is None else heading
                choices[index].append(dict(route, points=points, distance_m=distance,
                                           initial_yaw=initial, ground=ground))
    combinations = [(a, b) for a in range(len(choices[0])) for b in range(len(choices[1])) if a or b]
    combinations.sort(key=lambda indices: (sum(choices[i][j]['distance_m'] for i, j in enumerate(indices)), indices))
    for indices in combinations:
        yield [choices[i][j] for i, j in enumerate(indices)]


def plan_meetup(pair_clip, scene, *, actor_ids=('actor_1', 'actor_2'), starts,
                meeting, speed_mps=.65, max_seconds=20, entry_policy='settled', arrival_standoff_m=None,
                idle_route_padding_m=0., initial_heading_ramp_seconds=0.):
    """CPU-only JSON plan. Starts are two world {x,z,yaw_degrees?} anchors.

    Meeting XZ anchors the native first-frame root midpoint, with optional yaw.
    Routes share a travel duration and real Core settle horizon. max_seconds
    bounds the approach; the complete take, including pair, has a 1000-frame cap.
    Optional arrival_standoff_m moves only Core arrival targets outward when
    native entry roots are closer. The native frames and every gate are unchanged.
    """
    if not isinstance(pair_clip, NativePairClip) or pair_clip.metadata.get('model') != 'InterGen' or pair_clip.segments is not None:
        raise ValueError('Meeting requires an original native InterGen pair')
    if (not isinstance(actor_ids, (list, tuple)) or len(actor_ids) != 2 or
            any(not isinstance(a, str) or not 1 <= len(a) <= 64 for a in actor_ids) or actor_ids[0] == actor_ids[1]):
        raise ValueError('Meeting requires two unique stable actor IDs')
    if not isinstance(starts, (list, tuple)) or len(starts) != 2:
        raise ValueError('Choose two independent actor start positions')
    def anchor(value, label):
        if not isinstance(value, Mapping) or not {'x', 'z'} <= set(value) or set(value)-{'x', 'z', 'yaw_degrees'}:
            raise ValueError(f'{label} requires x, z and optional yaw_degrees')
        return {'x': _number(value['x'], label+'.x', -25, 25), 'z': _number(value['z'], label+'.z', -25, 25),
                **({'yaw_degrees': _number(value['yaw_degrees'], label+'.yaw_degrees', -180, 180)} if 'yaw_degrees' in value else {})}
    starts = [anchor(v, f'Start {i+1}') for i, v in enumerate(starts)]
    meeting = anchor(meeting, 'Meeting')
    speed = _number(speed_mps, 'Walking speed', .2, 1.5)
    maximum = _number(max_seconds, 'Maximum approach seconds', 4, 20)
    standoff = None if arrival_standoff_m is None else _number(
        arrival_standoff_m, 'Core arrival standoff', MIN_ROUTE_SEPARATION_M, .80)
    idle_padding = _number(idle_route_padding_m, 'Extra idle route clearance', 0., .60)
    heading_ramp = _number(initial_heading_ramp_seconds, 'Initial heading ramp seconds', 0., 1.)
    if entry_policy not in ('settled', 'continuous'):
        raise ValueError('Unknown meeting entry policy')
    entry_posture = _entry_posture(pair_clip.metadata.get('prompt', ''))
    yaw = math.radians(meeting.get('yaw_degrees', 0.))
    rotated = shared_place_pair(pair_clip.joints, yaw=yaw)
    center = rotated[0, :, 0].mean(axis=0)
    offset = [meeting['x']-float(center[0]), 0., meeting['z']-float(center[2])]
    pair = rotated+offset
    native_entry = pair[0, :, 0][:, [0, 2]].copy()
    arrivals = native_entry.copy()
    axis = native_entry[1]-native_entry[0]
    native_separation = float(np.linalg.norm(axis))
    if standoff is not None and native_separation < standoff:
        if native_separation < .05:
            raise ValueError('Native entry roots are too close to define safe Core standoff directions')
        shift = axis/native_separation*(standoff-native_separation)/2
        arrivals[0] -= shift
        arrivals[1] += shift
    arrival_standoff = {'requested_minimum_separation_m': standoff,
        'native_entry_root_separation_m': native_separation,
        'core_arrival_root_separation_m': float(np.linalg.norm(arrivals[1]-arrivals[0])),
        'native_entry_roots_xz': native_entry.tolist(), 'core_arrival_targets_xz': arrivals.tolist(),
        'target_offsets_xz': (arrivals-native_entry).tolist(),
        'maximum_target_offset_m': float(np.linalg.norm(arrivals-native_entry, axis=-1).max()),
        'source_pair_frames_modified': False}
    placement = {'x': offset[0], 'z': offset[2], 'yaw_degrees': meeting.get('yaw_degrees', 0.)}
    adapted = adapt_studio_scene(scene)['scene']
    geometry = check_native_pair_geometry(pair, scene, actor_ids=actor_ids)
    physical_obstacles = _obstacles(scene_objects(adapted), None, 1.65, .34)
    obstacles = [replace(box, half_width=box.half_width+idle_padding, half_depth=box.half_depth+idle_padding)
                 if box.object_id.startswith('scene-idle-') else box for box in physical_obstacles]
    routes = []
    for i, aid in enumerate(actor_ids):
        start = [starts[i]['x'], starts[i]['z']]
        end = arrivals[i].tolist()
        routed, departure = _idle_clearance_path(tuple(start), tuple(end), physical_obstacles, obstacles)
        points = [list(p) for p in routed]
        arrival = _heading(pair[0, i])
        if entry_policy == 'continuous':
            if departure:
                # Preserve the physically checked outward prefix exactly; round
                # only the route after it has left the larger planning envelope.
                points = points[:1]+_entry_route(points[1:], arrival, adapted, obstacles)
            else:
                points = _entry_route(points, arrival, adapted, obstacles)
        if any(abs(v) > 25 for p in points for v in p):
            raise ValueError('Meeting route exceeds the Core worker ±25 m bounds')
        ground = validate_ground_path(adapted, points)
        length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        _, first = _sample(points, 0.)
        initial = math.radians(starts[i]['yaw_degrees']) if 'yaw_degrees' in starts[i] else (arrival if first is None else first)
        routes.append({'actor_id': aid, 'points': points, 'distance_m': length,
                       'initial_yaw': initial, 'arrival_yaw': arrival, 'ground': ground,
                       'idle_departure': departure,
                       **({'entry_policy': entry_policy} if entry_policy == 'continuous' else {})})
    horizons, arrival_seconds, windows, frames, minimum = _schedule(routes, starts, actor_ids, speed, maximum, pair_clip.frames, entry_policy, heading_ramp, entry_posture)
    minimum = min(minimum, _route_separation(routes))
    rerouted = False
    if minimum < MIN_ROUTE_SEPARATION_M:
        for candidate in _alternate_routes(routes, starts, adapted, obstacles):
            if _route_separation(candidate) < MIN_ROUTE_SEPARATION_M:
                continue
            try:
                scheduled = _schedule(candidate, starts, actor_ids, speed, maximum, pair_clip.frames, entry_policy, heading_ramp, entry_posture)
            except ValueError:
                continue
            if scheduled[-1] < MIN_ROUTE_SEPARATION_M:
                continue
            routes = candidate
            horizons, arrival_seconds, windows, frames, minimum = scheduled
            minimum = min(minimum, _route_separation(routes))
            rerouted = True
            break
    if minimum < MIN_ROUTE_SEPARATION_M:
        raise ValueError('Actor routes cross too closely and no bounded clear detour fits; move the starts or meeting spot')
    return {'version': 1, 'actor_ids': list(actor_ids), 'starts': starts, 'meeting': meeting,
            'placement': placement, 'routes': routes, 'horizons': horizons, 'arrival_seconds': arrival_seconds,
            'approach_seconds': arrival_seconds+1. if entry_policy == 'continuous' else windows*2,
            'entry_policy': entry_policy, 'entry_posture': entry_posture,
            'total_frames': frames, 'blend_frames': BLEND_FRAMES,
            'minimum_planned_root_separation_m': minimum, 'native_geometry': geometry,
            'approach_rerouted': rerouted,
            'arrival_standoff': arrival_standoff,
            'idle_route_padding_m': idle_padding,
            'initial_heading_ramp_seconds': heading_ramp,
            'source_sha256': hashlib.sha256(pair_clip.joints.tobytes()).hexdigest(), 'planned_only': True,
            'limits': {'max_target_error_m': MAX_TARGET_ERROR_M, 'max_arrival_error_m': MAX_ARRIVAL_ERROR_M,
                       'minimum_route_root_separation_m': MIN_ROUTE_SEPARATION_M,
                       'max_entry_root_height_gap_m': MAX_ENTRY_ROOT_HEIGHT_GAP_M}}


def build_meetup(pair_clip, client, scene, *, actor_ids=('actor_1', 'actor_2'), starts,
                 meeting, speed_mps=.65, max_seconds=20, seed=92642,
                 cancelled=lambda: False, on_progress=None, on_core_chunk=None, entry_policy='settled',
                 arrival_standoff_m=None, idle_route_padding_m=0., initial_heading_ramp_seconds=0.):
    """Return local clip, playback placement, world joints, plan and metadata.

    on_progress(dict) reports windows. on_core_chunk('approach', index, clip,
    request) archives each completed raw window, including rejected candidates.
    Exceptions never publish partial output. Callers retain their last good take.
    """
    if client is None:
        raise RuntimeError('Configure ARDY Core before generating a meeting')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be a uint32 integer')
    plan = plan_meetup(pair_clip, scene, actor_ids=actor_ids, starts=starts, meeting=meeting, speed_mps=speed_mps, max_seconds=max_seconds, entry_policy=entry_policy, arrival_standoff_m=arrival_standoff_m, idle_route_padding_m=idle_route_padding_m, initial_heading_ramp_seconds=initial_heading_ramp_seconds)
    ids, placement = plan['actor_ids'], plan['placement']
    yaw = math.radians(placement['yaw_degrees']); offset = [placement['x'], 0., placement['z']]
    pair = shared_place_pair(pair_clip.joints, yaw=yaw, translation=offset)
    clips, measured = [], []
    def check_cancel():
        if cancelled():
            raise RuntimeError('Meeting generation cancelled')
    for index, horizon in enumerate(plan['horizons']):
        check_cancel()
        if on_progress:
            on_progress({'phase': 'approach', 'completed_windows': index, 'total_windows': len(plan['horizons'])})
        request = {'request_id': 'meetup-'+uuid.uuid4().hex, 'stage_kind': 'approach' if index == 0 else 'continuation',
                   'frames': HORIZON, 'prompt': 'Two people approach a meeting from separate locations.',
                   'actor_ids': ids, 'seed': seed, **horizon}
        if clips:
            request['history'] = {'native_features': clips[-1].native_features[:, -HORIZON:].tolist()}
        else:
            request['initial_placements'] = {r['actor_id']: {'position_xz': r['points'][0], 'yaw': r['initial_yaw']} for r in plan['routes']}
        result = client.wait(request, cancelled=cancelled)
        if len(result) != 1:
            raise ValueError('Expected exactly one synchronized Core horizon')
        clip = result[0]
        if (tuple(clip.actor_ids) != tuple(ids) or clip.fps != FPS or clip.frames != HORIZON or
                clip.positions.shape != (2, HORIZON, 27, 3) or clip.native_features is None or
                clip.native_features.shape != (2, HORIZON, 330) or not np.isfinite(clip.positions).all() or
                not np.isfinite(clip.native_features).all()):
            raise ValueError('Core returned incompatible actors, timing, positions or native history')
        if on_core_chunk:
            on_core_chunk('approach', index, clip, request)
        check_cancel()
        errors = [float(np.linalg.norm(clip.positions[i, target['frame'], 0, [0, 2]]-target['position_xz']))
                  for i, aid in enumerate(ids) for target in horizon['root_targets'][aid]]
        worst = max(errors); measured.append({'window': index, 'maximum_target_error_m': worst})
        if worst > MAX_TARGET_ERROR_M:
            raise ValueError(f'Core missed the planned route by {worst:.2f} m; meeting rejected')
        _checked_generated_geometry(core27_to_native22(clip.positions).transpose(1, 0, 2, 3), scene, ids,
                                    context={'stage': 'core_window', 'window': index, 'source_fps': 20})
        clips.append(clip)
    core = np.concatenate([c.positions for c in clips], axis=1).transpose(1, 0, 2, 3)
    generated_core_frames = len(core)
    if entry_policy == 'continuous':
        core = core[:round(plan['approach_seconds']*FPS)]
    errors = np.linalg.norm(core[-1, :, 0][:, [0, 2]]-pair[0, :, 0][:, [0, 2]], axis=-1)
    target_errors = np.linalg.norm(core[-1, :, 0][:, [0, 2]]-np.asarray(plan['arrival_standoff']['core_arrival_targets_xz']), axis=-1)
    if float(errors.max()) > MAX_ARRIVAL_ERROR_M:
        raise ValueError('Core did not reach both meeting entry positions; meeting rejected')
    if _minimum_separation(core[:, :, 0][:, :, [0, 2]]) < MIN_ROUTE_SEPARATION_M:
        raise ValueError('Generated actor approaches cross too closely; meeting rejected')
    count = len(core)*3//2
    times, new_times = np.arange(len(core))/FPS, np.arange(count)/30.
    resampled = np.stack([np.interp(new_times, times, col) for col in core.reshape(len(core), -1).T], axis=-1).reshape(count, 2, 27, 3)
    approach, refinement = core_to_pair_anatomy(resampled, pair)
    # Measure the display boundary after anatomy adjustment: accurate XZ roots
    # and a velocity-safe bridge do not justify a large vertical pose drop.
    entry_height_gaps = pair[0, :, 0, 1]-approach[-1, :, 0, 1]
    incompatible = np.flatnonzero(np.abs(entry_height_gaps) > MAX_ENTRY_ROOT_HEIGHT_GAP_M)
    if len(incompatible):
        actor_index = int(incompatible[0])
        raise ValueError(f'Meeting entry root-height gap for {ids[actor_index]} is '
                         f'{abs(entry_height_gaps[actor_index]):.2f} m, exceeding '
                         f'{MAX_ENTRY_ROOT_HEIGHT_GAP_M:.2f} m; meeting rejected')
    bridge_candidates = (12, 15, 18, BLEND_FRAMES) if entry_policy == 'continuous' else (BLEND_FRAMES,)
    for bridge_frames in bridge_candidates:
        try:
            entry, entry_report = authored_direction_bridge(approach, pair, left_fps=30, right_fps=30, frames=bridge_frames)
        except ValueError:
            if bridge_frames == bridge_candidates[-1]:
                raise
            continue
        if entry_report['mechanical_gate_passed']:
            break
    if not entry_report['mechanical_gate_passed']:
        raise ValueError('Meeting entry transition rejected: '+ '; '.join(entry_report['rejection_reasons']))
    plan['blend_frames'] = len(entry)
    plan['total_frames'] = len(approach)+len(entry)+len(pair)
    world = np.concatenate([approach, entry, pair])
    geometry = _checked_generated_geometry(world, scene, ids, context={'stage': 'composed_meetup', 'fps': 30})
    check_cancel(); start = len(approach)+len(entry)
    segments = [{'source': 'ardy_core', 'kind': 'approach', 'label': 'Core approach from independent starts', 'start_frame': 0, 'end_frame_exclusive': len(approach)},
                {'source': 'authored_transition', 'kind': 'transition', 'label': 'Authored entry transition', 'start_frame': len(approach), 'end_frame_exclusive': start},
                {'source': 'intergen', 'kind': 'paired_action', 'label': 'Native paired interaction', 'start_frame': start, 'end_frame_exclusive': len(world)}]
    for segment in segments:
        segment['frames'] = segment['end_frame_exclusive']-segment['start_frame']
    metadata = {'model': 'ARDY Core + InterGen', 'fps': 30, 'frames': len(world), 'segments': segments,
                'prompt': pair_clip.metadata.get('prompt', ''), 'render_hand_pose': pair_clip.metadata.get('render_hand_pose', 'relaxed'),
                'meeting_plan': plan, 'scene_geometry': geometry, 'actor_ids': ids,
                'source_pair_preserved_exactly_after_shared_placement': True,
                'transition_provenance': {'source_pair_frames_modified': False, 'all_mechanical_gates_passed': True,
                    'boundaries': {'entry': entry_report}, 'refinement': {'approach': refinement}, 'gpu_jobs': len(clips), 'visual_acceptance': 'unverified'},
                'approach_measurements': measured, 'arrival_errors_m': errors.tolist(),
                'arrival_standoff': plan['arrival_standoff'], 'arrival_target_errors_m': target_errors.tolist(),
                'actual_arrival_native_offsets_xz': (core[-1, :, 0][:, [0, 2]]-pair[0, :, 0][:, [0, 2]]).tolist(),
                'core_playback_selection': {'generated_frames': generated_core_frames, 'retained_frames': len(core),
                    'source_fps': FPS, 'selection': 'contiguous prefix; raw horizons archived separately'},
                'entry_root_height_gaps_m': entry_height_gaps.tolist(),
                'max_entry_root_height_gap_m': MAX_ENTRY_ROOT_HEIGHT_GAP_M, 'animation_accepted': False,
                'physical_contact_verified': False, 'features': 'No unified model features; preserve source archives separately.'}
    if not np.array_equal(world[start:], pair):
        raise RuntimeError('Native interaction preservation failed')
    local = shared_place_pair(world-np.asarray(offset), yaw=-yaw)
    local[start:] = pair_clip.joints
    output = NativePairClip(local, metadata=metadata)
    if on_progress:
        on_progress({'phase': 'complete', 'completed_windows': len(clips), 'total_windows': len(clips)})
    return {'clip': output, 'placement': placement, 'joints': world, 'metadata': metadata, 'report': metadata, 'plan': plan}
