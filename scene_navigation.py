"""Bounded deterministic navigation over the scene's rendered support geometry."""
from __future__ import annotations

import heapq
import math
import numpy as np


def plan_navigation_route(geometry, start_xyz, goal_xyz, *, radius=.22,
                          max_step_up=.30, max_drop=.35, ignore_object_ids=(),
                          sample_spacing=.10, max_expansions=16000):
    """Return a supported obstacle-avoiding floor-space route, or fail clearly.

    Uses the same transformed primitives as contact solving. Elevated floors
    have separate search states, so a bridge never connects to the ground below
    it. The bounded search is deliberately conservative about ledges and gaps.
    """
    start, goal = np.asarray(start_xyz, dtype=float).copy(), np.asarray(goal_xyz, dtype=float).copy()
    if start.shape != (3,) or goal.shape != (3,) or not np.isfinite([start, goal]).all():
        raise ValueError('Navigation endpoints must be finite world positions.')
    if (not all(math.isfinite(v) and v >= 0 for v in (radius, max_step_up, max_drop))
            or not math.isfinite(sample_spacing) or not .01 <= sample_spacing <= .10
            or isinstance(max_expansions, bool) or not isinstance(max_expansions, int)
            or not 1 <= max_expansions <= 100000):
        raise ValueError('Navigation requires finite nonnegative limits and a bounded search.')
    ignored = tuple(ignore_object_ids)

    def point(x, z, y):
        h = geometry.support_height(x, z, y, max_step_up=max_step_up, max_drop=max_drop)
        if h is None:
            return None
        if any(geometry.obstacle_at(x, h + offset, z, radius=radius, ignore_object_ids=ignored)
               for offset in (.40, .90, 1.35)):
            return None
        # Require space for both soles; a root path alone can straddle a void.
        for dx, dz in ((.12, 0), (-.12, 0), (0, .12), (0, -.12)):
            if geometry.support_height(x+dx, z+dz, h, max_step_up=max_step_up, max_drop=max_drop) is None:
                return None
        return np.array((x, h, z))

    def segment(a, b):
        current = a
        samples = [a]
        distance = np.linalg.norm((b-a)[[0, 2]])
        for t in np.linspace(0, 1, max(2, math.ceil(distance / sample_spacing) + 1))[1:]:
            position = a + (b-a)*t
            current = point(position[0], position[2], current[1])
            if current is None:
                return None
            samples.append(current)
        if abs(current[1] - b[1]) > .03:
            return None
        return np.asarray(samples)

    start = point(start[0], start[2], start[1])
    goal = point(goal[0], goal[2], goal[1])
    if start is None or goal is None:
        raise ValueError('The destination or starting position has no clear walkable approach.')
    direct = segment(start, goal)
    if direct is not None:
        return _compact_support_samples(direct)
    spacing = .25
    margin = min(10., max(4., np.linalg.norm((goal-start)[[0, 2]])*.6))
    low = np.minimum(start[[0, 2]], goal[[0, 2]]) - margin
    high = np.maximum(start[[0, 2]], goal[[0, 2]]) + margin
    # Anchor the grid at the actual starting position, avoiding initial snaps.
    origin = start[[0, 2]]
    key = lambda i, j, y: (i, j, round(float(y)/.10))
    first = key(0, 0, start[1])
    points, costs, previous = {first: start}, {first: 0.}, {}
    queue, serial = [(0., 0, first)], 0
    closed, cache = set(), {}
    terminal = None
    while queue and len(closed) < max_expansions:
        _, _, current_key = heapq.heappop(queue)
        if current_key in closed:
            continue
        closed.add(current_key)
        current = points[current_key]
        if np.linalg.norm((current-goal)[[0, 2]]) <= .45 and segment(current, goal) is not None:
            terminal = current_key
            break
        for di, dj in ((1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)):
            i, j = current_key[0]+di, current_key[1]+dj
            x, z = origin + spacing*np.array((i, j))
            if x < low[0] or x > high[0] or z < low[1] or z > high[1]:
                continue
            cache_key = (i, j, current_key[2])
            if cache_key not in cache:
                cache[cache_key] = point(x, z, current[1])
            candidate = cache[cache_key]
            if candidate is None:
                continue
            candidate_key = key(i, j, candidate[1])
            if candidate_key in closed or segment(current, candidate) is None:
                continue
            cost = costs[current_key] + np.linalg.norm((candidate-current)[[0, 2]]) + .3*abs(candidate[1]-current[1])
            if cost >= costs.get(candidate_key, float('inf')):
                continue
            points[candidate_key], costs[candidate_key], previous[candidate_key] = candidate, cost, current_key
            serial += 1
            heuristic = np.linalg.norm((goal-candidate)[[0, 2]]) + .3*abs(goal[1]-candidate[1])
            heapq.heappush(queue, (cost+heuristic, serial, candidate_key))
    if terminal is None:
        raise ValueError('No safe walking route reaches that destination; the way is blocked or disconnected.')
    route = [goal, points[terminal]]
    while terminal in previous:
        terminal = previous[terminal]
        route.append(points[terminal])
    route.reverse()
    # Remove grid zigzags only when the complete shortcut remains supported.
    result, index = [route[0]], 0
    while index < len(route)-1:
        furthest = index+1
        for candidate in range(len(route)-1, index+1, -1):
            if segment(route[index], route[candidate]) is not None:
                furthest = candidate
                break
        result.append(route[furthest])
        index = furthest
    # Shortcutting changes only floor-space direction. Retain the actual
    # sampled tread/deck heights rather than replacing stairs with a ramp.
    supported = [result[0]]
    for a, b in zip(result[:-1], result[1:]):
        samples = segment(a, b)
        if samples is None:
            raise ValueError('Route support changed while planning.')
        supported.extend(samples[1:])
    return _compact_support_samples(np.asarray(supported))


def _compact_support_samples(points):
    """Remove only truly collinear XYZ samples; preserve each riser bracket.

    These are sole support samples, not a generated root or foot trajectory.
    A controller must query geometry between samples when placing a foot.
    """
    result = [points[0]]
    for index in range(1, len(points)-1):
        before = points[index] - result[-1]
        after = points[index+1] - points[index]
        if (np.linalg.norm(np.cross(before, after)) > 1e-8
                or before @ after < 0):
            result.append(points[index])
    if np.linalg.norm(points[-1]-result[-1]) > 1e-10:
        result.append(points[-1])
    return np.asarray(result)
