"""Scene-aware gathering marks for independent multi-stage group motion."""
from copy import deepcopy
import math

from interaction_scene import scene_objects
from interaction_planner import _inside, _obstacles, _path
from prompt_scene_plan import _landmarks
from realtime_navigation import validate_ground_path
from scene_composition import validate_scene


def group_sequence_placement(plan, scene, *, cancelled=lambda: False):
    """Preserve explicit starts/facing and find separated, reachable gathering marks.

    These marks describe non-contact group staging. They do not imply shared
    physical contact or validate the subsequently generated body motion.
    """
    from prompt_group_adapter import independent_starts
    from group_scene_sequence import _routes
    if cancelled():
        raise RuntimeError('Group staging cancelled')
    scene = validate_scene(scene)
    desired = plan.get('meeting')
    if desired is None:
        starts = independent_starts(plan, scene, cancelled=cancelled)
        return {'starts': starts, 'targets': deepcopy(starts), 'meeting': None,
                'routes': {}, 'physical_contact_verified': False}
    obstacles = _obstacles(scene_objects(scene), None, 1.65, .4)
    def clear(point):
        if max(map(abs, point)) > 24 or any(_inside(point, box, .4) for box in obstacles):
            return False
        try:
            validate_ground_path(scene, [point], actor_radius_m=.4)
            return True
        except ValueError:
            return False
    def route(start, target):
        points = _path(tuple(start), tuple(target), obstacles, .4)
        validate_ground_path(scene, points, actor_radius_m=.4)
        return [list(p) for p in points]
    fixed = {a['id']: (a['start']['x'], a['start']['z']) for a in plan['actors'] if a.get('start') is not None}
    for aid, point in fixed.items():
        if not clear(point):
            raise ValueError(f'Requested start for {aid} overlaps the selected background or unsupported ground')
    values = list(fixed.values())
    if any(math.dist(a,b) < 2 for i,a in enumerate(values) for b in values[i+1:]):
        raise ValueError('Group starting positions must be at least 2 metres apart')
    target_id = desired.get('target_id')
    if target_id:
        landmark = _landmarks(scene)[target_id]
        x, _, z = landmark['position']
        owner = landmark if 'size' in landmark else _landmarks(scene).get(landmark.get('object_id'))
        radius = max(owner['size'][0],owner['size'][2])/2+3 if owner else 3
        candidates = [(x+r*math.sin(a*math.pi/4),z+r*math.cos(a*math.pi/4)) for r in (radius,radius+3,radius+6) for a in range(8)]
    else:
        candidates = [(desired['x'],desired['z'])]
    count = plan['actor_count']
    best = None
    for center in candidates:
        for formation_yaw in (0.,math.pi/2):
            if cancelled():
                raise RuntimeError('Group staging cancelled')
            lateral = (math.cos(formation_yaw),-math.sin(formation_yaw))
            forward = (math.sin(formation_yaw),math.cos(formation_yaw))
            slots = [(center[0]+(i-(count-1)/2)*2.6*lateral[0],center[1]+(i-(count-1)/2)*2.6*lateral[1]) for i in range(count)]
            if not all(clear(p) for p in slots):
                continue
            starts, targets, routes = {}, {}, {}
            try:
                for actor, slot in zip(plan['actors'], slots):
                    aid = actor['id']
                    options = [fixed[aid]] if aid in fixed else [
                        (slot[0]-distance*forward[0],slot[1]-distance*forward[1]) for distance in (2.6,3.6,1.8)]
                    selected = None
                    for point in options:
                        reserved = [p for other, p in fixed.items() if other != aid]
                        if (not clear(point) or any(math.dist(point, p) < 2 for p in reserved)
                                or any(math.dist(point,(p['x'],p['z'])) < 2 for p in starts.values())):
                            continue
                        try:
                            points = route(point,slot)
                        except ValueError:
                            continue
                        selected = point,points
                        break
                    if selected is None:
                        raise ValueError('No clear approach fits the group')
                    point, points = selected
                    yaw = actor.get('start_yaw_degrees')
                    if yaw is None:
                        direction = next(((p[0]-point[0],p[1]-point[1]) for p in points[1:] if math.dist(p,point)>.01),forward)
                        yaw = math.degrees(math.atan2(*direction))
                    starts[aid] = {'x':float(point[0]),'z':float(point[1]),'yaw_degrees':float(yaw)}
                    targets[aid] = {'x':float(slot[0]),'z':float(slot[1]),'yaw_degrees':math.degrees(formation_yaw)}
                    routes[aid] = points
                checked = _routes(scene, list(starts), starts, targets)
                # Prefer room for generated limb motion and natural root excursion.
                # This is staging headroom, never permission to skip final body gates.
                margin = next((m for m in (2., 1.5, 1., .4)
                               if all(not any(_inside(p, box, m) for box in obstacles) for p in slots)), 0.)
                result = {'starts':starts,'targets':targets,'meeting':{'x':center[0],'z':center[1],'target_id':target_id},
                          'routes':routes,'formation':'separated gathering row',
                          'action_obstacle_margin_m':margin,'physical_contact_verified':False}
                if cancelled():
                    raise RuntimeError('Group staging cancelled')
                if margin >= 2.:
                    return result
                if best is None or margin > best[0]:
                    best = (margin, result)
            except ValueError:
                continue
    if best is not None:
        return best[1]
    raise ValueError('No clear gathering formation and approaches fit the selected meeting location; choose more open space')
