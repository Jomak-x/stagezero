"""Live, reversible editing of a validated AI cast staging plan.

This layer does not generate motion. The accepted starts can be exported as a
version-1 prompt scene plan for the existing performance builder.
"""
from __future__ import annotations

from copy import deepcopy
import math
from threading import RLock

from prompt_scene_plan import ScenePromptPlanner, auto_place, validate_plan
from scene_composition import validate_scene


EMPTY_SCENE = {
    'version': 2, 'name': 'Clear staging floor', 'objects': [],
    'effects': [], 'lighting': 'neutral',
}


class InitialPlacementSession:
    def __init__(self, scene=None, planner=None):
        self.scene = validate_scene(EMPTY_SCENE if scene is None else scene)
        self.planner = planner if planner is not None else ScenePromptPlanner()
        self._lock = RLock()
        self._ai_plan = None
        self._plan = None
        self._placement = None
        self._ai_placement = None
        self._changed_actor_ids = set()

    def plan(self, prompt):
        """Ask the configured planner for a fresh plan and publish it on success."""
        candidate = self.planner.plan(prompt, self.scene)
        return self.set_plan(candidate)

    def set_plan(self, plan):
        """Validate and resolve a supplied AI plan without partially publishing it."""
        clean = validate_plan(plan, self.scene)
        placement = auto_place(clean, self.scene)
        with self._lock:
            self._ai_plan = deepcopy(plan)
            self._plan = clean
            self._placement = placement
            self._ai_placement = deepcopy(placement)
            self._changed_actor_ids = set()
            return self._snapshot_unlocked()

    @staticmethod
    def _coordinate(value, name, limit):
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > limit:
            raise ValueError(f'{name} must be a finite number within ±{limit}')
        return float(value)

    def edit_actor(self, actor_id, *, x=None, z=None, yaw_degrees=None):
        """Apply an absolute floor pose if all cast positions and routes stay valid."""
        if x is None and z is None and yaw_degrees is None:
            raise ValueError('Supply a position or facing to edit')
        with self._lock:
            if self._plan is None:
                raise ValueError('Plan the cast before editing its placement')
            if actor_id not in self._placement['starts']:
                raise ValueError('Unknown cast actor')
            starts = deepcopy(self._placement['starts'])
            updated = starts[actor_id]
            if x is not None:
                updated['x'] = self._coordinate(x, 'X', 24)
            if z is not None:
                updated['z'] = self._coordinate(z, 'Z', 24)
            if yaw_degrees is not None:
                updated['yaw_degrees'] = self._coordinate(yaw_degrees, 'Facing', 180)

            # Pin every displayed start and the current meeting during validation.
            # This prevents an edit from silently moving a different performer.
            candidate = self._plan_with_starts(starts)
            candidate = validate_plan(candidate, self.scene)
            placement = auto_place(candidate, self.scene)
            for aid, start in starts.items():
                actual = placement['starts'][aid]
                if any(abs(actual[key] - start[key]) > 1e-7 for key in ('x', 'z', 'yaw_degrees')):
                    raise ValueError('Staging solver changed a pinned performer')
            self._plan = candidate
            self._placement = placement
            self._changed_actor_ids = {
                aid for aid, start in placement['starts'].items()
                if any(abs(start[key] - self._ai_placement['starts'][aid][key]) > 1e-7
                       for key in ('x', 'z', 'yaw_degrees'))
            }
            return self._snapshot_unlocked()

    def _plan_with_starts(self, starts):
        candidate = deepcopy(self._plan)
        for actor in candidate['actors']:
            start = starts[actor['id']]
            actor['start'] = {'x': start['x'], 'z': start['z']}
            actor['start_yaw_degrees'] = start['yaw_degrees']
        meeting = self._placement['meeting']
        candidate['meeting'] = {'x': meeting['x'], 'z': meeting['z']}
        return candidate

    def reset(self):
        """Restore the exact validated plan and resolved marks from the AI run."""
        with self._lock:
            if self._ai_plan is None:
                raise ValueError('No AI placement to reset')
            self._plan = validate_plan(self._ai_plan, self.scene)
            self._placement = deepcopy(self._ai_placement)
            self._changed_actor_ids = set()
            return self._snapshot_unlocked()

    def export_plan(self):
        """Return a complete v1 plan with the displayed starts made explicit."""
        with self._lock:
            if self._plan is None:
                raise ValueError('No cast plan to export')
            return validate_plan(self._plan_with_starts(self._placement['starts']), self.scene)

    def _snapshot_unlocked(self):
        if self._plan is None:
            return {'plan': None, 'placement': None, 'changed_actor_ids': []}
        return deepcopy({
            'plan': self._plan,
            'ai_plan': self._ai_plan,
            'placement': self._placement,
            'changed_actor_ids': sorted(self._changed_actor_ids),
        })

    def snapshot(self):
        with self._lock:
            return self._snapshot_unlocked()
