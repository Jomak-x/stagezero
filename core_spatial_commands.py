"""Bounded spatial language and measured completion for native Core motion.

No pose edits, object recognition, or hand-operated doors. Explicit terrain
commands use rendered support geometry and require measured native validation.
Relative directions mean turn and walk from the committed root's local heading.
"""
from __future__ import annotations

import math
import re

import numpy as np

from realtime_navigation import _placements, plan_navigation
from studio_interaction_scene import resolve_target
from core_terrain_navigation import plan_terrain_command

MAX_ACTIONS = 4
MAX_DISTANCE_M = 8.
ARRIVAL_TOLERANCE_M = .30
_NUMBERS = {name: i for i, name in enumerate(("zero", "one", "two", "three", "four", "five", "six", "seven", "eight"))}
_DISTANCE = r"(\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight)\s*(?:m|metres?|meters?)"
_DIRECTION = r"(forward|forwards|backward|backwards|back|left|right)"
_ACTION_START = r"(?:walk|go|move|climb|ascend|descend|cross|open|enter|approach)\b"


def _action_clauses(text, adapted):
    """Keep commas inside exact target names before recognizing action commas."""
    protected = set()
    for obj in adapted["objects"]:
        name = obj["name"]
        if "," not in name:
            continue
        for found in re.finditer(re.escape(name), text, re.I):
            before = text[:found.start()]
            if not re.search(r"(?:^|[,;]\s*|\bthen\s+|\band\s+)"
                             r"(?:open|approach|enter|(?:walk|go|move)\s+to|(?:walk|go)\s+through)"
                             r"\s+(?:(?:the|a|an)\s+)?$", before, re.I):
                continue
            protected.update(found.start()+index for index, char in enumerate(found.group())
                             if char == ",")
    separator = re.compile(rf"\s*(?:,?\s+then\s+|;|,\s*(?:and\s+)?(?={_ACTION_START})"
                           rf"|\s+and\s+(?={_ACTION_START}))\s*", re.I)
    clauses = []
    start = 0
    for found in separator.finditer(text):
        comma = text.find(",", found.start(), found.end())
        if comma in protected:
            continue
        clauses.append(text[start:found.start()].strip())
        start = found.end()
    clauses.append(text[start:].strip())
    return clauses


def parse_commands(text, adapted):
    """Full-match every clause and resolve every object before changing work."""
    if not isinstance(text, str) or not text.strip() or len(text) > 600:
        raise ValueError("Spatial command must contain 1–600 characters")
    clauses = _action_clauses(text.strip().rstrip("."), adapted)
    if not 1 <= len(clauses) <= MAX_ACTIONS:
        raise ValueError(f"Use at most {MAX_ACTIONS} ordered actions")
    actions = []
    prior_gate = None
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
            prior_gate = None
            continue
        terrain_match = re.fullmatch(r"(?:(?:walk|go|move|climb)\s+(up|down)|(?:(ascend|descend)))\s+(.+)", clause, re.I)
        if terrain_match:
            verb = "ascend" if (terrain_match[1] or terrain_match[2]).lower() in ("up", "ascend") else "descend"
            target = _resolve_terrain_name(adapted, terrain_match[3], "stairs")
            actions.append({"verb": verb, **target})
            prior_gate = None
            continue
        terrain_match = re.fullmatch(r"(?:(?:walk|go|move)\s+across|cross)\s+(.+)", clause, re.I)
        if terrain_match:
            actions.append({"verb": "cross", **_resolve_terrain_name(adapted, terrain_match[1], "bridge")})
            prior_gate = None
            continue
        if re.fullmatch(r"enter", clause, re.I):
            if prior_gate is None:
                raise ValueError("Enter needs a preceding open gate command or an explicit target")
            actions.append({"verb": "go_through", **prior_gate})
            prior_gate = None
            continue
        match = re.fullmatch(r"(open|approach|enter|(?:walk|go|move)\s+to|(?:walk|go)\s+through)\s+(.+)", clause, re.I)
        if not match:
            if re.search(r"\bjump\b", clause, re.I):
                raise ValueError("Jumping across terrain is unsupported")
            raise ValueError(f"Unsupported spatial clause: {clause!r}. Use 'walk 2 metres forward', 'approach object', or 'go through object', joined by 'then'.")
        verb = "open" if match[1].lower() == "open" else "go_through" if match[1].lower().endswith("through") or match[1].lower() == "enter" else "approach"
        name = match[2].strip()
        target_ref = _resolve_terrain_name(adapted, name, "gate" if verb in ("open", "go_through") else None)
        target_id = target_ref.get("target_id")
        target = next((obj for obj in adapted["objects"] if obj["id"] == target_id), None)
        if verb == "open":
            from core_terrain_navigation import _objects_for_alias
            candidates = ([next(obj for obj in adapted["scene"]["objects"] if obj["id"] == target_id)]
                          if target_id else _objects_for_alias(adapted["scene"], target_ref["target_alias"]))
            for obj in candidates:
                interaction = obj.get("interaction", {})
                if (obj["kind"] != "door" or interaction.get("trigger") != "proximity"
                        or interaction.get("action") != "open"):
                    raise ValueError("Open supports configured automatic proximity doors only; hand-operated doors are unsupported")
        if target is not None and verb not in target["actions"] and target["kind"] != "door":
            raise ValueError(f"{target_id} has no verified open passage")
        actions.append({"verb": verb, **target_ref})
        prior_gate = target_ref if verb == "open" else None
    terrain_context = adapted.get("terrain_active") is True
    for action in actions:
        if terrain_context:
            action["terrain"] = True
        if action["verb"] in ("ascend", "descend", "cross"):
            terrain_context = True
    return actions


def _resolve_terrain_name(adapted, name, category):
    """Resolve exact names, then only supported generic nouns and qualifiers."""
    from core_terrain_navigation import _objects_for_alias
    name = name.strip()
    try:
        return {"target_id": resolve_target(adapted, name)}
    except ValueError as exc:
        if "ambiguous" in str(exc).lower():
            raise
        if name.lower().startswith("the "):
            try:
                return {"target_id": resolve_target(adapted, name[4:])}
            except ValueError as inner:
                if "ambiguous" in str(inner).lower():
                    raise
    if category is None:
        raise ValueError("Unknown navigation target; choose an exact scene object")
    simple = name.casefold().strip()
    words = re.findall(r"[a-z0-9]+", simple)
    if " ".join(words) != simple:
        raise ValueError("Unknown terrain target; use an exact name or supported noun")
    while words and words[0] in ("the", "a", "an"):
        words.pop(0)
    nouns = {"stairs": ("stair", "stairs", "step", "steps"),
             "bridge": ("bridge", "walkway"), "gate": ("gate", "door")}
    noun = category if words and words[-1] in nouns[category] else None
    if noun is None:
        raise ValueError("Unknown terrain target; choose an authored object or supported affordance")
    options = _objects_for_alias(adapted["scene"], noun)
    scene_words = set(re.findall(r"[a-z0-9]+", adapted["scene"].get("name", "").casefold()))
    qualifiers = words[:-1]
    options = [obj for obj in options if all(
        word in scene_words or word in set(re.findall(r"[a-z0-9]+", obj.get("name", "").casefold()))
        for word in qualifiers)]
    if not options:
        raise ValueError("Unknown terrain target qualifier; choose an exact authored name")
    if len(options) == 1:
        return {"target_id": options[0]["id"]}
    if noun != "stairs" or any(word not in scene_words for word in qualifiers):
        raise ValueError(f"{noun} target is ambiguous; name the exact object")
    return {"target_alias": noun}


def plan_command(action, adapted, actor_ids, actor_id, last_clip, initial_placements):
    source = adapted.get("original_scene", adapted["scene"])
    targets = [obj for obj in source["objects"]
               if obj["id"] == action.get("target_id")]
    elevated_target = any(obj["kind"] == "door" and
                          abs(obj["position"][1]-obj["size"][1]/2) > .1
                          for obj in targets)
    if (action["verb"] in ("ascend", "descend", "cross") or
            action.get("target_alias") is not None or
            action.get("terrain") is True or adapted.get("terrain_active") is True
            or elevated_target):
        return plan_terrain_command(action, adapted, actor_ids, actor_id, last_clip, initial_placements)
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
    if route.get("terrain_navigation_version") == 1:
        target_root_y = route["waypoints"][-1]["support_y"] + .95
        root_y = float(clip.positions[actor, -1, 0, 1])
        height_error = abs(root_y-target_root_y)
        report.update(terminal_root_y=root_y, target_root_y=target_root_y,
                      root_height_error_m=height_error,
                      root_height_verified=height_error <= .25)
        report["arrival_verified"] &= report["root_height_verified"]
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
        options = session._reaction_options(enabling=enabling)
        scene = evaluated_scene(session._scene, session._director.timeline_clip(), **options)
        result = adapt_studio_scene(scene)
        result["original_scene"] = session._scene
        result["terrain_active"] = (
            session._director.project_metadata.get("studio_core", {}).get("terrain_navigation_version") == 1)
        return result

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
        action = self.report["actions"][index]
        if action.get("target_alias") and route.get("target_id"):
            alias = action.pop("target_alias")
            action["target_id"] = route["target_id"]
            # A bare "enter" is bound to the specific gate selected by open.
            if action["verb"] == "open" and index + 1 < len(self.report["actions"]):
                following = self.report["actions"][index + 1]
                if following["verb"] == "go_through" and following.get("target_alias") == alias:
                    following.pop("target_alias")
                    following["target_id"] = route["target_id"]
        for stage in stages:
            stage.metadata["spatial_command"] = {"action_index": index, "text": self.report["text"]}
        session._director.queue_sequence(stages)
        if route.get("terrain_navigation_version") == 1:
            metadata = session._director.project_metadata.setdefault("studio_core", {})
            metadata.setdefault("terrain_navigation_version", 1)
            metadata.setdefault("terrain_navigation_start_frame", session._director.total_frames)
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
                adapted = self.adapted()
                if leg["route"].get("terrain_navigation_version") == 1:
                    from core_terrain_navigation import resolve_terrain_object
                    terminal = np.asarray(session._director.timeline_clip().positions[
                        session._director.actor_ids.index(self.report["actor_id"]), -1, 0])
                    target = resolve_terrain_object(action, adapted, terminal)
                    original = next(obj for obj in adapted["original_scene"]["objects"]
                                    if obj["id"] == target["id"])
                    opened = (target["kind"] == "door" and
                              target["position"][1]-original["position"][1] >= original["size"][1]-.03)
                else:
                    from interaction_scene import scene_objects, passage_for
                    target = next(obj for obj in scene_objects(adapted["scene"])
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
