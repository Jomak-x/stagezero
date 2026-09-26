"""Thirty-second, geometry-grounded plans for native model-generated motion.

All frame indices are global, zero-based 20 fps output frames. Root goals are
ARDY conditioning requests, never replacement joint poses. Required outcome
metrics decide whether a generated scene may be called complete.
"""

from __future__ import annotations

import math

from interaction_planner import plan_action
from scene_composition import validate_scene
from scene_objects import make_object

FPS = 20
TOTAL_FRAMES = 600
SCENARIOS = ("gate_meet_handshake", "staged_fight", "object_reach_inspect")
RECOMMENDED_SEEDS = {"gate_meet_handshake": 42, "staged_fight": 38,
                     "object_reach_inspect": 42}


def _prop(kind, index, name, position, size, color):
    prop = make_object(kind, index)
    prop.update(name=name, position=list(position), size=list(size), color=list(color))
    return prop


def _scene(name, objects, lighting):
    return validate_scene({"version": 2, "name": name, "objects": objects,
                           "effects": [], "lighting": lighting})


def _goals(start, end, knots, heading=None):
    """Sample every eight frames and at each Core window endpoint."""
    if start % 40 or end % 40 or end <= start or knots[0][0] != 0 or knots[-1][0] != 1:
        raise ValueError("Invalid beat interval or route fractions")
    if any(b[0] <= a[0] for a, b in zip(knots, knots[1:])):
        raise ValueError("Route fractions must increase")

    def point(frame):
        fraction = (frame - start) / (end - start - 1)
        for (u0, p0), (u1, p1) in zip(knots, knots[1:]):
            if fraction <= u1:
                alpha = (fraction - u0) / (u1 - u0)
                return [p0[0] + alpha * (p1[0] - p0[0]),
                        p0[1] + alpha * (p1[1] - p0[1])]
        return list(knots[-1][1])

    targets = []
    frames = sorted({*range(start + 7, end, 8), *range(start + 39, end, 40)})
    for frame in frames:
        here = point(frame)
        before = point(max(start, frame - 4))
        after = point(min(end - 1, frame + 4))
        dx, dz = after[0] - before[0], after[1] - before[1]
        yaw = math.atan2(dx, dz) if math.hypot(dx, dz) > 1e-6 else heading
        target = {"frame": frame, "position_xz": [round(x, 5) for x in here]}
        if yaw is not None:
            target["heading"] = max(-math.pi, min(math.pi, round(yaw, 6)))
        targets.append(target)
    return targets


def _planner_goals(plan, start, end):
    waypoints = plan["waypoints"]
    t0, t1 = waypoints[0]["time_seconds"], waypoints[-1]["time_seconds"]
    if t1 <= t0:
        raise ValueError("Planner returned an empty route")
    knots = [((w["time_seconds"] - t0) / (t1 - t0), w["position_xz"])
             for w in waypoints]
    knots[0] = (0., knots[0][1])
    knots[-1] = (1., knots[-1][1])
    return _goals(start, end, knots)


def _hold(start, end, position, heading):
    return _goals(start, end, [(0., position), (1., position)], heading)


def _actor(prompt, goals=None):
    return {"prompt": prompt, "root_targets": goals or []}


def _beat(identifier, kind, start, end, prompt, actors, *, source="ardy_core",
          gates=(), metadata=None):
    if start % 40 or end % 40 or end <= start:
        raise ValueError("Beats must cover complete 40-frame windows")
    return {"id": identifier, "kind": kind, "source": source,
            "start_frame": start, "end_frame": end, "frames": end - start,
            "prompt": prompt, "actors": actors, "quality_gates": list(gates),
            "metadata": metadata or {}}


def _gate(seed):
    arch = _prop("arch", 0, "Stone gate", (0, 1.5, .3), (3.2, 3., .45), (144, 149, 154))
    left = _prop("pillar", 1, "West lamp post", (-3.4, 1.2, 2.6), (.35, 2.4, .35), (111, 124, 143))
    right = _prop("pillar", 2, "East lamp post", (3.4, 1.2, 2.6), (.35, 2.4, .35), (111, 124, 143))
    scene = _scene("Gate meeting", [arch, left, right], "moonlight")
    route = plan_action({"verb": "go_through", "actor_id": "visitor", "target_id": arch["id"]},
                        scene, [0, 0, -2.4], actor_radius_m=.32, speed_mps=.8)
    gate_exit = route["waypoints"][-1]["position_xz"]
    beats = [
        _beat("cross_gate", "approach", 0, 120,
              "A visitor walks through the wide stone arch toward a waiting friend.",
              {"visitor": _actor("Walk through the arch and slow down.", _planner_goals(route, 0, 120)),
               "friend": _actor("Wait calmly, facing the visitor.", _hold(0, 120, (1., 2.35), math.pi))},
              gates=({"metric": "gate_traversal", "actor_id": "visitor", "object_id": arch["id"], "required": True},
                     {"metric": "scene_collision", "actor_id": "visitor", "max_overlap_frames": 0})),
        _beat("meet", "approach", 120, 200,
              "They approach each other, stop face to face, and exchange a greeting.",
              {"visitor": _actor("Approach a friend, decelerate, and stand.",
                                  _goals(120, 200, [(0., gate_exit), (1., (0., 1.55))], 0.)),
               "friend": _actor("Step toward the visitor and face them.",
                                 _goals(120, 200, [(0., (1., 2.35)), (1., (.85, 2.25))], math.pi))},
              gates=({"metric": "pair_separation", "min_m": .55},)),
        _beat("align_handshake", "transition", 200, 240,
              "Both friends settle into the opening pose of a shared handshake.",
              {"visitor": _actor("Prepare the greeting."), "friend": _actor("Prepare the greeting.")},
              gates=({"metric": "continuity", "max_boundary_jump_m": .35},)),
        _beat("handshake", "paired", 240, 320,
              "Two people shake hands, acknowledge each other, release, and step apart.",
              {"visitor": _actor("Shake hands and release."), "friend": _actor("Shake hands and release.")},
              source="intergen",
              gates=({"metric": "hand_contact", "minimum_duration_s": .35, "tolerance_m": .15,
                      "research_only": True}, {"metric": "pair_separation", "min_m": .42}),
              metadata={"license": "CC BY-NC-SA 4.0", "research_preview": True,
                        "pair_sequence_id": f"gate_handshake_{seed}", "source_start_frame": 0,
                        "source_total_frames": 120, "seed": seed}),
        _beat("handshake_release", "exit", 320, 360,
              "Two people shake hands, acknowledge each other, release, and step apart.",
              {"visitor": _actor("Release and step apart."),
               "friend": _actor("Release and step apart.")}, source="intergen",
              gates=({"metric": "hand_release", "research_only": True},
                     {"metric": "pair_separation", "min_m": .42}),
              metadata={"license": "CC BY-NC-SA 4.0", "research_preview": True,
                        "pair_sequence_id": f"gate_handshake_{seed}", "source_start_frame": 80,
                        "source_total_frames": 120, "release_window": [80, 120], "seed": seed}),
        _beat("part", "continuation", 360, 480,
              "After releasing hands, both friends take several steps apart and turn away.",
              {"visitor": _actor("Walk west after the greeting.",
                                  _goals(360, 480, [(0., (0., 1.55)), (1., (-1.55, 2.1))])),
               "friend": _actor("Walk east after the greeting.",
                                 _goals(360, 480, [(0., (.85, 2.25)), (1., (2.1, 2.8))]))},
              gates=({"metric": "pair_separation", "min_m": .5},)),
        _beat("depart", "continuation", 480, 600,
              "They walk away on separate paths beyond the gate.",
              {"visitor": _actor("Walk away to the west and stop.",
                                  _goals(480, 600, [(0., (-1.55, 2.1)), (1., (-2.5, 3.2))])),
               "friend": _actor("Walk away to the east and stop.",
                                 _goals(480, 600, [(0., (2.1, 2.8)), (1., (3., 3.55))]))},
              gates=({"metric": "pair_separation", "min_m": .5},)),
    ]
    return scene, beats, {"visitor_gate": route}, {
        "visitor": {"position_xz": [0., -2.4], "yaw": 0.},
        "friend": {"position_xz": [1., 2.35], "yaw": math.pi}}


def _fight(seed):
    props = [_prop("crate", 0, "West stage crate", (-4.5, .4, 0), (.8, .8, .8), (96, 109, 119)),
             _prop("crate", 1, "East stage crate", (4.5, .4, 0), (.8, .8, .8), (96, 109, 119))]
    scene = _scene("Rehearsal yard", props, "sunset")
    beats = [
        _beat("square_up", "approach", 0, 120,
              "Two stunt performers enter a clear rehearsal space and take guarded stances.",
              {"fighter_a": _actor("Walk into a guarded stance.",
                                   _goals(0, 120, [(0., (-2.8, -.6)), (1., (-.9, -.35))], math.pi / 2)),
               "fighter_b": _actor("Walk into a guarded stance facing the partner.",
                                   _goals(0, 120, [(0., (2.8, .6)), (1., (.9, .35))], -math.pi / 2))},
              gates=({"metric": "pair_separation", "min_m": .65},
                     {"metric": "scene_collision", "actor_id": "fighter_a", "max_overlap_frames": 0},
                     {"metric": "scene_collision", "actor_id": "fighter_b", "max_overlap_frames": 0})),
        _beat("align_dodge", "transition", 120, 160,
              "Both performers prepare a synchronized evasive move.",
              {"fighter_a": _actor("Guard and prepare to feint."),
               "fighter_b": _actor("Guard and prepare to evade.")},
              gates=({"metric": "continuity", "max_boundary_jump_m": .35},)),
        _beat("dodge_block_push", "paired", 160, 240,
              "A staged sparring sequence: feint, sidestep dodge, guarded block, light push, then release and step apart.",
              {"fighter_a": _actor("Feint, block, stage push, and release."),
               "fighter_b": _actor("Dodge, guard, react to push, and recover.")},
              source="intergen", gates=({"metric": "pair_separation", "min_m": .42,
                                         "research_only": True},
                                        {"metric": "dodge_lateral", "min_m": .2,
                                         "research_only": True},
                                        {"metric": "block_guard", "tolerance_m": .22,
                                         "minimum_duration_s": .1, "research_only": True},
                                        {"metric": "push_reaction", "research_only": True}),
              metadata={"choreography": "dodge_block_push", "license": "CC BY-NC-SA 4.0",
                        "research_preview": True, "pair_sequence_id": f"fight_{seed}",
                        "source_start_frame": 0, "source_total_frames": 120, "seed": seed}),
        _beat("fight_release", "exit", 240, 280,
              "A staged sparring sequence: feint, sidestep dodge, guarded block, light push, then release and step apart.",
              {"fighter_a": _actor("Release and step apart."),
               "fighter_b": _actor("Release and recover balance.")}, source="intergen",
              gates=({"metric": "hand_release", "research_only": True},
                     {"metric": "pair_separation", "min_m": .42}),
              metadata={"choreography": "dodge_block_push", "license": "CC BY-NC-SA 4.0",
                        "research_preview": True, "pair_sequence_id": f"fight_{seed}",
                        "source_start_frame": 80, "source_total_frames": 120,
                        "release_window": [80, 120], "seed": seed}),
        _beat("recover", "continuation", 280, 440,
              "Both performers lower their guard, catch their breath, and take space.",
              {"fighter_a": _actor("Lower guard and step back.",
                                   _goals(280, 440, [(0., (-.9, -.35)), (1., (-1.8, -.5))])),
               "fighter_b": _actor("Lower guard and step back.",
                                   _goals(280, 440, [(0., (.9, .35)), (1., (1.8, .5))]))},
              gates=({"metric": "pair_separation", "min_m": .6},
                     {"metric": "scene_collision", "actor_id": "fighter_a", "max_overlap_frames": 0},
                     {"metric": "scene_collision", "actor_id": "fighter_b", "max_overlap_frames": 0})),
        _beat("disengage", "continuation", 440, 600,
              "They return to separate marks, ending the rehearsal.",
              {"fighter_a": _actor("Lower guard and walk west.",
                                   _goals(440, 600, [(0., (-1.8, -.5)), (1., (-2.8, -.6))])),
               "fighter_b": _actor("Lower guard and walk east.",
                                   _goals(440, 600, [(0., (1.8, .5)), (1., (2.8, .6))]))},
              gates=({"metric": "pair_separation", "min_m": .6},)),
    ]
    return scene, beats, {}, {"fighter_a": {"position_xz": [-2.8, -.6], "yaw": math.pi / 2},
                              "fighter_b": {"position_xz": [2.8, .6], "yaw": -math.pi / 2}}


def _object(seed):
    console = _prop("console", 0, "Survey console", (0, .575, 1.5),
                    (1.1, 1.15, .65), (59, 109, 128))
    scene = _scene("Quiet inspection", [console], "neon")
    route = plan_action({"verb": "approach", "actor_id": "inspector", "target_id": console["id"]},
                        scene, [0, 0, -2], actor_radius_m=.28, speed_mps=.65)
    stand = route["waypoints"][-1]["position_xz"]
    beats = [
        _beat("approach_console", "approach", 0, 120,
              "An inspector walks to the console and stops at reach distance.",
              {"inspector": _actor("Walk to the console and stop.", _planner_goals(route, 0, 120))},
              gates=({"metric": "scene_collision", "actor_id": "inspector", "max_overlap_frames": 0},)),
        _beat("look_over_console", "continuation", 120, 200,
              "The inspector studies the console before touching it.",
              {"inspector": _actor("Look carefully at the console.", _hold(120, 200, stand, 0.))}),
        _beat("reach_inspect", "action", 200, 320,
              "The inspector reaches a right hand to a control, rests briefly, and examines it.",
              {"inspector": _actor("Reach to inspect the console with the right hand.")},
              gates=({"metric": "object_contact", "actor_id": "inspector", "hand": "right",
                      "object_id": console["id"], "minimum_duration_s": .2,
                      "tolerance_m": .12, "native_reference_required": True},),
              metadata={"native_hand_target": {"source": "coherent_prior_core_pose",
                                               "source_beat": "look_over_console",
                                               "hand": "RightHand", "position_xyz": [.2, .95, 1.18],
                                               "global_frames": [239, 259, 279, 299, 319],
                                               "constraint": "EndEffectorConstraintSet"},
                        "object_cue": "touch_only_no_grasp"}),
        _beat("release", "release", 320, 400,
              "The inspector lowers the hand and steps back.",
              {"inspector": _actor("Lower the hand and step back.",
                                   _goals(320, 400, [(0., stand), (1., (0., -.3))]))}),
        _beat("turn_away", "continuation", 400, 480,
              "The inspector turns away after the inspection.",
              {"inspector": _actor("Turn away and pause.", _hold(400, 480, (0., -.3), math.pi))}),
        _beat("leave", "continuation", 480, 600,
              "The inspector walks away from the console.",
              {"inspector": _actor("Walk away from the console.",
                                   _goals(480, 600, [(0., (0., -.3)), (1., (-1.3, -2.2))]))}),
    ]
    return scene, beats, {"console_approach": route}, {
        "inspector": {"position_xz": [0., -2.], "yaw": 0.}}


_BUILDERS = {"gate_meet_handshake": _gate, "staged_fight": _fight,
             "object_reach_inspect": _object}


def build_scene(name: str, seed: int | None = None) -> dict:
    """Return a complete plan; generated motion still needs measured acceptance."""
    if name not in _BUILDERS:
        raise ValueError(f"Unknown scene scenario: {name!r}")
    if seed is None:
        seed = RECOMMENDED_SEEDS[name]
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a uint32")
    scene, beats, routes, placements = _BUILDERS[name](seed)
    if beats[0]["start_frame"] != 0 or beats[-1]["end_frame"] != TOTAL_FRAMES:
        raise AssertionError("Scene must span exactly 30 seconds")
    for left, right in zip(beats, beats[1:]):
        if left["end_frame"] != right["start_frame"]:
            raise AssertionError("Scene beats have a gap or overlap")
    for beat in beats:
        if set(beat["actors"]) != set(placements):
            raise AssertionError("Every beat must include every actor")
        for actor in beat["actors"].values():
            frames = [t["frame"] for t in actor["root_targets"]]
            if frames != sorted(set(frames)) or any(not beat["start_frame"] <= f < beat["end_frame"]
                                                     for f in frames):
                raise AssertionError("Root goals are not ordered within their beat")
    return {"version": 1, "name": name, "seed": seed, "fps": FPS,
            "total_frames": TOTAL_FRAMES, "scene": scene,
            "actor_ids": list(placements), "initial_placements": placements,
            "beats": beats, "route_plans": routes,
            "provenance": {"planner": "interaction_planner.plan_action",
                           "root_goals": "official ARDY Core constraints",
                           "joint_animation": "model_output_only",
                           "research_pair_model": "InterGen" if any(
                               beat["source"] == "intergen" for beat in beats) else None}}
