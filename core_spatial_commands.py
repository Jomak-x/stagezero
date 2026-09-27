"""Bounded spatial language and measured completion for native Core motion.

No pose edits, object recognition, vertical movement, or hand-operated doors.
Relative directions mean turn and walk from the committed root's local heading.
"""
from __future__ import annotations

import math
import re

import numpy as np

from realtime_navigation import _placements, plan_navigation
from studio_interaction_scene import resolve_target

MAX_ACTIONS = 4
MAX_DISTANCE_M = 8.
ARRIVAL_TOLERANCE_M = .30
_NUMBERS = {name: i for i, name in enumerate(("zero", "one", "two", "three", "four", "five", "six", "seven", "eight"))}
_DISTANCE = r"(\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight)\s*(?:m|metres?|meters?)"
_DIRECTION = r"(forward|forwards|backward|backwards|back|left|right)"


def parse_commands(text, adapted):
    """Full-match every clause and resolve every object before changing work."""
    if not isinstance(text, str) or not text.strip() or len(text) > 600:
        raise ValueError("Spatial command must contain 1–600 characters")
    clauses = re.split(r"\s*(?:,?\s+then\s+|;)\s*", text.strip().rstrip("."), flags=re.I)
    if not 1 <= len(clauses) <= MAX_ACTIONS:
        raise ValueError(f"Use at most {MAX_ACTIONS} ordered actions")
    actions = []
    for clause in clauses:
        if re.match(r"(?:close|unlock|push|pull)\b", clause, re.I):
            raise ValueError("Door actuation is unsupported; approach a proximity door, then go through only once it is open")
        match = re.fullmatch(rf"(?:walk|go|move)\s+{_DISTANCE}\s+{_DIRECTION}", clause, re.I)
        if match:
            number, direction = match.groups()
        else:
            match = re.fullmatch(rf"(?:walk|go|move)\s+{_DIRECTION}\s+{_DISTANCE}", clause, re.I)
            if match:
                direction, number = match.groups()
        if match:
            distance = float(_NUMBERS[number.lower()] if number.lower() in _NUMBERS else number)
            if not .25 <= distance <= MAX_DISTANCE_M:
                raise ValueError("Each relative walk must be between 0.25 and 8 metres")
            actions.append({"verb": "move", "direction": direction.lower(), "distance_m": distance})
            continue
        match = re.fullmatch(r"(open|approach|(?:walk|go|move)\s+to|(?:walk|go)\s+through)\s+(.+)", clause, re.I)
        if not match:
            if re.search(r"\b(stairs?|steps?|upstairs|downstairs|climb|ascend|descend|jump)\b", clause, re.I):
                raise ValueError("Stairs, climbing, and vertical navigation are unsupported")
            raise ValueError(f"Unsupported spatial clause: {clause!r}. Use 'walk 2 metres forward', 'approach object', or 'go through object', joined by 'then'.")
        verb = "open" if match[1].lower() == "open" else "go_through" if match[1].lower().endswith("through") else "approach"
        name = match[2].strip()
        # Exact IDs/names win, including names beginning with 'the'.
        try:
            target_id = resolve_target(adapted, name)
        except ValueError:
            if not name.lower().startswith("the "):
                raise
            target_id = resolve_target(adapted, name[4:])
        target = next(obj for obj in adapted["objects"] if obj["id"] == target_id)
        if verb == "open":
            obj = next(obj for obj in adapted["scene"]["objects"] if obj["id"] == target_id)
            interaction = obj.get("interaction", {})
            if (obj["kind"] != "door" or interaction.get("trigger") != "proximity"
                    or interaction.get("action") != "open"):
                raise ValueError("Open supports configured automatic proximity doors only; hand-operated doors are unsupported")
        if verb not in target["actions"] and target["kind"] != "door":
            raise ValueError(f"{target_id} has no verified open passage")
        actions.append({"verb": verb, "target_id": target_id})
    return actions


def plan_command(action, adapted, actor_ids, actor_id, last_clip, initial_placements):
    args = dict(actor_id=actor_id, verb="approach" if action["verb"] == "open" else action["verb"], last_clip=last_clip,
                initial_placements=initial_placements, affordances=adapted["affordances"], turn_before_travel=True,
                gait_profile="spatial", speed_mps=1.2)
    if action["verb"] == "move":
        positions, yaws = _placements(tuple(actor_ids), last_clip, initial_placements)
        offsets = {"forward": 0., "forwards": 0., "back": math.pi, "backward": math.pi,
                   "backwards": math.pi, "left": -math.pi/2, "right": math.pi/2}
        heading = yaws[actor_id] + offsets[action["direction"]]
        start = positions[actor_id]
        args["target_xz"] = [start[0] + math.sin(heading)*action["distance_m"],
                             start[1] + math.cos(heading)*action["distance_m"]]
    else:
        args["target_id"] = action["target_id"]
    return plan_navigation(adapted["scene"], actor_ids, **args)


def measure_completion(route, clip, start_frame):
    """Measure actual terminal arrival and an actual aperture crossing, not intent."""
    actor = clip.actor_ids.index(route["actor_id"])
    roots = clip.positions[actor, start_frame:, 0][:, [0, 2]]
    if not len(roots):
        raise ValueError("No committed motion exists for this action")
    end = np.asarray(route["waypoints"][-1]["position_xz"])
    error = float(np.linalg.norm(roots[-1] - end))
    tolerance = ARRIVAL_TOLERANCE_M
    if route["verb"] == "move":
        displacement = float(np.linalg.norm(end - route["waypoints"][0]["position_xz"]))
        tolerance = min(tolerance, displacement / 2)
    report = {"terminal_xz": roots[-1].tolist(), "target_xz": end.tolist(),
              "arrival_error_m": error, "arrival_tolerance_m": tolerance,
              "arrival_verified": error <= tolerance, "crossing_verified": None}
    if route["verb"] == "go_through":
        points = route["waypoints"]
        center = np.asarray(next(w["position_xz"] for w in points if w["role"] == "center"))
        entry = np.asarray(next(w["position_xz"] for w in points if w["role"] == "entry"))
        along = (entry - center) / np.linalg.norm(entry - center)
        lateral = np.asarray([-along[1], along[0]])
        # Only observed roots establish a crossing. A planned origin must not
        # make a first generated frame teleported to the exit look successful.
        observed = roots
        if start_frame:
            prior = clip.positions[actor, start_frame-1, 0, [0, 2]]
            observed = np.vstack((prior, roots))
        path = observed - center
        signed = path @ along
        half_depth = route["geometry"]["passage_depth_m"]/2 + .28
        clearance = route["geometry"]["minimum_lateral_clearance_m"]
        crossed = False
        for i in range(len(path)-1):
            if signed[i] >= 0 and signed[i+1] < 0:
                alpha = signed[i] / (signed[i] - signed[i+1])
                intersection = path[i] + alpha * (path[i+1] - path[i])
                crossed |= abs(float(intersection @ lateral)) <= clearance
        report["crossing_verified"] = bool(crossed and signed[0] >= half_depth and signed[-1] <= -half_depth)
    report["completed"] = report["arrival_verified"] and report["crossing_verified"] is not False
    return report


class SpatialSequence:
    """Session-lock-owned orchestration; the director still owns native motion."""

    def __init__(self, session):
        self.session = session
        self.report = None

    @property
    def pending(self):
        return bool(self.report and self.report["status"] in ("running", "generation_failed"))

    def record(self):
        import copy
        self.session._director.project_metadata.setdefault("studio_core", {})["last_spatial_commands"] = copy.deepcopy(self.report)

    def cancel(self):
        if self.pending:
            self.report.update(status="cancelled", detail="Pending commands cancelled; committed motion retained")
            self.record()

    def adapted(self, *, enabling=False):
        from core_scene_reactions import evaluated_scene
        from studio_interaction_scene import adapt_studio_scene
        session = self.session
        return adapt_studio_scene(evaluated_scene(session._scene, session._director.timeline_clip(),
                                                  **session._reaction_options(enabling=enabling)))

    def start(self, actor_id, text):
        from realtime_clip import MAX_CLIP_FRAMES
        from realtime_director import RealtimeDirector
        session = self.session
        adapted = self.adapted(enabling=True)
        actions = parse_commands(text, adapted)
        stages, route = plan_command(actions[0], adapted, session._director.actor_ids,
                                     actor_id, session._director.timeline_clip(), session._placements)
        if session._director.total_frames + route["schedule"]["frames"] + 600*(len(actions)-1) > MAX_CLIP_FRAMES:
            raise ValueError("Spatial sequence would exceed the 15000-frame timeline limit")
        RealtimeDirector(session._director.actor_ids).queue_sequence(stages)
        session._invalidate()
        metadata = session._director.project_metadata["studio_core"]
        if not session.scene_reactions_enabled:
            metadata.update(scene_reactions_version=1, scene_reactions_start_frame=session._director.total_frames)
        self.report = {"version": 1, "text": text.strip(), "actor_id": actor_id,
                       "actions": actions, "status": "running", "action_index": 0,
                       "completed_actions": 0, "legs": [],
                       "submitted_after_committed_frame": session._director.total_frames}
        self.queue(stages, route)
        session._run_generation()
        return self.report

    def queue(self, stages, route):
        import copy
        session = self.session
        index = self.report["action_index"]
        for stage in stages:
            stage.metadata["spatial_command"] = {"action_index": index, "text": self.report["text"]}
        session._director.queue_sequence(stages)
        self.report["legs"].append({"action_index": index, "start_frame": session._director.total_frames,
                                    "route": copy.deepcopy(route), "measurement": None})
        session._route = route
        self.record()

    def committed(self):
        session = self.session
        if not self.pending or self.report["status"] != "running" or session._director.snapshot()["queued_stages"]:
            return
        try:
            leg = self.report["legs"][-1]
            leg["end_frame"] = session._director.total_frames
            leg["measurement"] = measure_completion(leg["route"], session._director.timeline_clip(), leg["start_frame"])
            action = self.report["actions"][self.report["action_index"]]
            if action["verb"] == "open":
                from interaction_scene import scene_objects, passage_for
                target = next(obj for obj in scene_objects(self.adapted()["scene"])
                              if obj.id == action["target_id"])
                try:
                    passage_for(target, None, actor_height_m=1.65)
                    opened = True
                except ValueError:
                    opened = False
                leg["measurement"]["automatic_door_open_verified"] = opened
                leg["measurement"]["completed"] &= opened
            if not leg["measurement"]["completed"]:
                self.report.update(status="arrival_failed", detail="Arrival, crossing, or requested automatic opening was not verified; later commands stopped")
                session._generation_enabled = False
                self.record()
                return
            self.report["completed_actions"] += 1
            self.report["action_index"] += 1
            if self.report["action_index"] == len(self.report["actions"]):
                self.report.update(status="completed", detail="Every command reached its measured destination")
                self.record()
                return
            action = self.report["actions"][self.report["action_index"]]
            stages, route = plan_command(action, self.adapted(), session._director.actor_ids,
                                         self.report["actor_id"], session._director.timeline_clip(), session._placements)
            self.queue(stages, route)
        except Exception as exc:
            # A committed good prefix is never undone when a later plan fails.
            self.report.update(status="planning_failed", detail=str(exc))
            session._generation_enabled = False
            self.record()

    def failed(self, error):
        if self.pending:
            self.report.update(status="generation_failed", detail=str(error))
            self.record()

    def retry(self):
        if self.report and self.report["status"] == "generation_failed":
            self.report.update(status="running", detail="Retrying failed native horizon")
            self.record()

    def restore(self):
        """Restore provenance only; an archive never starts spatial planning."""
        import copy
        director = self.session._director
        report = director.project_metadata.get("studio_core", {}).get("last_spatial_commands")
        # Provenance is untrusted archive metadata, never executable commands.
        if not isinstance(report, dict) or report.get("version") != 1 or not isinstance(report.get("actions"), list) or not 1 <= len(report["actions"]) <= MAX_ACTIONS or type(report.get("completed_actions")) is not int or not 0 <= report["completed_actions"] <= len(report["actions"]) or not isinstance(report.get("status"), str) or not isinstance(report.get("detail", ""), str):
            self.report = None
            return
        self.report = copy.deepcopy(report)
        if self.pending:
            director.cancel_pending()
            self.report.update(status="cancelled", detail="Archive restored for playback; unfinished commands were not resumed")
            self.record()
