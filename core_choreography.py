"""Bounded shared beats for independently generated native Core actors.

Plans coordinate prompts and optional root conditions on one scene clock. They
are not a learned interaction/contact model. Offsets are world XZ metres from
the cast's committed roots at submission, never replacements for generated poses.
"""
from __future__ import annotations

import json
import math

from realtime_clip import FPS
from realtime_director import HORIZON, StageSpec
from realtime_navigation import _placements

MAX_SECONDS = 16
PRESETS = ("feint_dodge", "dance_response", "surprise_celebration", "pose_duet")
SAMPLE_FRAMES = (7, 15, 23, 31, 39)


def _number(value, label, bound):
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > bound:
        raise ValueError(f"{label} must be finite and within ±{bound}")
    return float(value)


def _text(value, label, maximum):
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= maximum or any(ord(c) < 32 for c in value):
        raise ValueError(f"{label} must contain 1–{maximum} printable characters")
    return value.strip()


def _ids(actor_ids):
    if not isinstance(actor_ids, (tuple, list)) or len(actor_ids) != 2:
        raise ValueError("Shared choreography requires exactly two actor IDs")
    ids = tuple(actor_ids)
    if any(not isinstance(a, str) or not 1 <= len(a) <= 64 for a in ids) or len(set(ids)) != 2:
        raise ValueError("Shared choreography requires two distinct stable actor IDs")
    return ids


def validate_plan(plan, actor_ids=("actor_1", "actor_2")):
    """Validate untrusted director/gateway JSON without mutation, models or I/O.

    Every beat provides both actors' prompts and lasts 2 or 4 seconds. Optional
    root_offsets provide both actors' absolute offsets from their sequence-start
    roots, bounded to two metres per axis. Optional headings require offsets.
    Unknown fields are errors, so contact/prop claims cannot become commands.
    """
    ids = _ids(actor_ids)
    required = {"version", "name", "seed", "beats"}
    if not isinstance(plan, dict) or not required <= set(plan) or set(plan) - required - {"recipe"}:
        raise ValueError("Plan requires version, name, seed and beats; only a trusted recipe is optional")
    if "recipe" in plan and plan["recipe"] != "pose_duet_v1":
        raise ValueError("Unknown choreography recipe")
    if type(plan["version"]) is not int or plan["version"] != 1:
        raise ValueError("Unsupported choreography plan version")
    name = _text(plan["name"], "Plan name", 120)
    seed = plan["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Choreography seed must be a uint32")
    beats = plan["beats"]
    if not isinstance(beats, list) or not 1 <= len(beats) <= 8:
        raise ValueError("Plan requires 1–8 beats")
    normalized, names, seconds = [], set(), 0
    for beat in beats:
        required = {"name", "seconds", "actor_prompts"}
        if not isinstance(beat, dict) or not required <= set(beat) or set(beat) - required - {"root_offsets", "headings"}:
            raise ValueError("Each beat requires name, seconds and actor_prompts; only root_offsets/headings are optional")
        beat_name = _text(beat["name"], "Beat name", 80)
        if beat_name in names:
            raise ValueError("Beat names must be unique")
        names.add(beat_name)
        if type(beat["seconds"]) is not int or beat["seconds"] not in (2, 4):
            raise ValueError("Each beat must last 2 or 4 seconds")
        seconds += beat["seconds"]
        if seconds > MAX_SECONDS:
            raise ValueError("Choreography is limited to 16 seconds")
        prompts = beat["actor_prompts"]
        if not isinstance(prompts, dict) or set(prompts) != set(ids):
            raise ValueError("Each beat must provide prompts for both stable actor IDs")
        item = {"name": beat_name, "seconds": beat["seconds"],
                "actor_prompts": {a: _text(prompts[a], f"{a} prompt", 500) for a in ids}}
        if "root_offsets" in beat:
            offsets = beat["root_offsets"]
            if not isinstance(offsets, dict) or set(offsets) != set(ids):
                raise ValueError("root_offsets must cover both stable actor IDs")
            item["root_offsets"] = {}
            for a in ids:
                point = offsets[a]
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError("Each root offset must be [x, z]")
                item["root_offsets"][a] = [_number(v, "Root offset", 2) for v in point]
        if "headings" in beat:
            headings = beat["headings"]
            if "root_offsets" not in beat or not isinstance(headings, dict) or set(headings) != set(ids):
                raise ValueError("headings require root_offsets and must cover both actor IDs")
            item["headings"] = {a: _number(headings[a], "Heading", math.pi) for a in ids}
        normalized.append(item)
    result = {"version": 1, "name": name, "seed": seed, "beats": normalized}
    if "recipe" in plan:
        from core_pose_cues import recipe_roles
        recipe_roles(normalized, ids)
        result["recipe"] = plan["recipe"]
    return result


def build_choreography(plan, actor_ids=("actor_1", "actor_2"), *, initial_placements=None, last_clip=None):
    """Compile validated beats into immutable 40-frame stages and an audit report.

    This does not change the cast placement. Native history remains authoritative
    when continuing. Every conditioned horizon includes its own final-frame goal.
    Plans with mixed free/spatial beats should be reviewed for drift: these goals
    remain sequence-relative, not predictions of where unconstrained motion ends.
    """
    ids = _ids(actor_ids)
    plan = validate_plan(plan, ids)
    origins, yaws = _placements(ids, last_clip, initial_placements)
    for position in origins.values():
        for value in position:
            _number(value, "Actor starting position", 25)
    pose_recipe = plan.get("recipe") == "pose_duet_v1"
    if pose_recipe:
        from core_pose_cues import MIN_INITIAL_SEPARATION_M, profile_metadata, recipe_roles
        if math.dist(origins[ids[0]], origins[ids[1]]) + 1e-6 < MIN_INITIAL_SEPARATION_M:
            raise ValueError("Pose duet requires at least 2.25 m initial root separation; reposition the cast first")
        swapped = recipe_roles(plan["beats"], ids)
    anchors = {a: list(origins[a]) for a in ids}
    stages, spans = [], []
    start = 0
    for beat_index, beat in enumerate(plan["beats"]):
        frames = beat["seconds"] * FPS
        goals = None
        if "root_offsets" in beat:
            ends = {a: [origins[a][i] + beat["root_offsets"][a][i] for i in range(2)] for a in ids}
            for values in ends.values():
                for value in values:
                    _number(value, "Planned root position", 25)
            goals = {a: [] for a in ids}
            for frame in range(frames):
                fraction = (frame + 1) / frames
                points = {a: [anchors[a][i] + fraction * (ends[a][i] - anchors[a][i]) for i in range(2)] for a in ids}
                if math.dist(points[ids[0]], points[ids[1]]) < .65:
                    raise ValueError("Choreography root paths violate the 0.65 m pair separation proxy")
                for a in ids:
                    goals[a].append(points[a])
        for offset in range(0, frames, HORIZON):
            metadata = {"seed": plan["seed"], "choreography": {
                "version": 1, "name": plan["name"], "beat": beat["name"],
                "beat_index": beat_index, "beat_start_frame": start,
                "beat_frames": frames, "window_start_frame": start + offset,
                "shared_clock": True, "physical_contact_verified": False}}
            if pose_recipe:
                index = (start + offset) // HORIZON
                metadata["seed"] = (plan["seed"] + index) % (2**32)
                metadata["pose_cue_profile"] = profile_metadata(index, swapped=swapped)
            if start == 0 and offset == 0:
                metadata["choreography_plan"] = plan
            if start == 0 and offset == 0 and last_clip is None:
                metadata["initial_placements"] = {a: {"position_xz": origins[a], "yaw": yaws[a]} for a in ids}
            if goals is not None:
                metadata["root_targets"] = {a: [{"frame": f, "position_xz": goals[a][offset + f],
                    **({"heading": beat["headings"][a]} if "headings" in beat else {})}
                    for f in SAMPLE_FRAMES] for a in ids}
            summary = " / ".join(f"{a}: {beat['actor_prompts'][a]}" for a in ids)[:500]
            stages.append(StageSpec(summary, frames=HORIZON, actor_prompts=beat["actor_prompts"], metadata=metadata))
        spans.append({"name": beat["name"], "start_frame": start, "end_frame": start + frames})
        start += frames
        if goals is not None:
            anchors = ends
    report = {"version": 1, "plan": plan, "actor_ids": list(ids), "frames": start, "fps": FPS,
              "origins_xz": origins, "beats": spans, "planned_only": True,
              "coordination": "shared timed prompts and optional native root conditions; independent actor generation",
              "physical_contact_verified": False}
    # Detach report values from stage metadata and caller data.
    return tuple(stages), json.loads(json.dumps(report, allow_nan=False))


def choreography_preset(name, actor_ids=("actor_1", "actor_2"), *, seed=6201):
    """Candidate directions, not a motion-quality success matrix or saved clips."""
    a, b = _ids(actor_ids)
    if name == "pose_duet":
        from core_pose_cues import POSE_DUET_BEATS, RECIPE
        return validate_plan({"version": 1, "name": "Pose-cued sparring duet", "seed": seed, "recipe": RECIPE,
            "beats": [{"name": title, "seconds": 2, "actor_prompts": {a: pa, b: pb}}
                      for title, pa, pb in POSE_DUET_BEATS]}, (a, b))
    definitions = {
        "feint_dodge": [
            ("Ready together", 2, "Face your partner and raise a relaxed boxing guard, knees bent.", "Face your partner and raise a relaxed boxing guard, knees bent."),
            ("Lead and evade", 4, "Perform a sharp left jab and right cross toward your partner, then recover your guard. Keep a safe distance.", "React to your partner's punches: duck deeply, lean sideways to dodge, then recover your guard. Keep a safe distance."),
            ("Exchange roles", 4, "React to your partner's punches: duck and lean sideways to dodge, then recover your guard. Keep a safe distance.", "Perform a sharp left jab and right cross toward your partner, then recover your guard. Keep a safe distance."),
            ("Shared finish", 2, "Relax your guard and give your partner an enthusiastic two-handed celebration.", "Relax your guard and celebrate with both arms raised toward your partner."),
        ],
        "dance_response": [
            ("Dance together", 4, "Dance energetically with rhythmic side steps and broad arm swings alongside your partner.", "Dance energetically with rhythmic side steps and broad arm swings alongside your partner."),
            ("Leader accent", 2, "Perform an expressive dance flourish, sweeping both arms up while your partner watches.", "Bounce rhythmically in place and clap to encourage your dancing partner."),
            ("Partner answer", 2, "Bounce rhythmically in place and clap to encourage your dancing partner.", "Answer your partner with an expressive dance flourish, sweeping both arms up."),
            ("Shared finish", 4, "Dance with broad rhythmic arm movements, then raise both arms in a triumphant finish beside your partner.", "Dance with broad rhythmic arm movements, then raise both arms in a triumphant finish beside your partner."),
        ],
        "surprise_celebration": [
            ("Notice together", 2, "Stand beside your partner, notice something surprising ahead, and turn your upper body toward it.", "Stand beside your partner, notice something surprising ahead, and turn your upper body toward it."),
            ("Shared recoil", 2, "React with a dramatic startled recoil, bending your knees and raising your hands defensively.", "React with a dramatic startled recoil, bending your knees and raising your hands defensively."),
            ("Reassure and answer", 4, "Turn toward your partner, relax and gesture reassuringly with open hands.", "Look toward your reassuring partner, relax your shoulders, then respond with an enthusiastic hand gesture."),
            ("Celebrate together", 4, "Celebrate excitedly beside your partner, raising both arms high and bouncing on your feet.", "Celebrate excitedly beside your partner, raising both arms high and bouncing on your feet."),
        ],
    }
    if name not in definitions:
        raise ValueError(f"Unknown choreography preset; choose one of {', '.join(PRESETS)}")
    plan = {"version": 1, "name": name, "seed": seed,
            "beats": [{"name": title, "seconds": seconds, "actor_prompts": {a: pa, b: pb}}
                      for title, seconds, pa, pb in definitions[name]]}
    return validate_plan(plan, (a, b))
