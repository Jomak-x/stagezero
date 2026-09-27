"""Offline, explicit terrain-action transaction with separate presentation poses.

This runner never changes the live Core session. Native chunks stay in a private
director until every requested action, observed reaction, and presentation
check succeeds. Its returned presentation is a visual candidate only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from core_scene_reactions import evaluated_scene, object_states
from core_spatial_commands import measure_completion, parse_commands, plan_command
from realtime_client import request_body
from realtime_clip import CanonicalClip, FPS, MAX_CLIP_FRAMES
from realtime_director import HORIZON, RealtimeDirector
from scene_composition import validate_scene
from scene_interaction_geometry import SceneInteractionGeometry
from studio_interaction_scene import adapt_studio_scene

MAX_ARCHIVE_BYTES = 200_000_000
MAX_UNCOMPRESSED_BYTES = 160_000_000


@dataclass(frozen=True)
class PresentationClip:
    """Validated display-only pose stream; it never carries native features."""

    positions: np.ndarray
    rotations: np.ndarray
    stance: np.ndarray
    fps: int
    actor_ids: tuple[str, ...]
    rig_asset_sha256: tuple[str, ...]
    report: dict | None = None

    def __post_init__(self):
        ids = tuple(self.actor_ids)
        p = np.asarray(self.positions, dtype=np.float32)
        r = np.asarray(self.rotations, dtype=np.float32)
        if (self.fps != FPS or not 1 <= len(ids) <= 2 or
                p.ndim != 4 or p.shape[0] != len(ids) or
                not 1 <= p.shape[1] <= MAX_CLIP_FRAMES or p.shape[2:] != (17, 3) or
                r.shape != p.shape[:2]+(17, 3, 3) or
                not np.isfinite(p).all() or not np.isfinite(r).all() or
                not np.allclose(r @ np.swapaxes(r, -1, -2), np.eye(3), atol=.03) or
                not np.allclose(np.linalg.det(r), 1, atol=.03)):
            raise ValueError("Presentation must contain finite Human17 world poses")
        hashes = tuple(self.rig_asset_sha256)
        if (len(hashes) != len(ids) or any(not isinstance(h, str) or len(h) != 64 or
                                            any(c not in "0123456789abcdef" for c in h) for h in hashes)):
            raise ValueError("Presentation needs a SHA256 for each exact render asset")
        stance = np.asarray(self.stance)
        if stance.shape != (len(ids), p.shape[1], 2) or stance.dtype != np.bool_:
            raise ValueError("Presentation stance must be boolean [actors, frames, 2]")
        frozen = np.frombuffer(stance.tobytes(), dtype=np.bool_).reshape(stance.shape)
        object.__setattr__(self, "positions", np.frombuffer(p.tobytes(), np.float32).reshape(p.shape))
        object.__setattr__(self, "rotations", np.frombuffer(r.tobytes(), np.float32).reshape(r.shape))
        object.__setattr__(self, "stance", frozen)
        object.__setattr__(self, "actor_ids", ids)
        object.__setattr__(self, "rig_asset_sha256", hashes)

    @property
    def frames(self):
        return self.positions.shape[1]


@dataclass(frozen=True)
class AssistedTerrainResult:
    scene: dict
    native_clip: CanonicalClip
    presentation: PresentationClip
    report: dict
    routes: tuple[dict, ...]
    reaction_states: tuple[dict, ...]


@dataclass(frozen=True)
class NativeTerrainResult:
    scene: dict
    native_clip: CanonicalClip
    routes: tuple[dict, ...]
    action_spans: tuple[tuple[int, int], ...]
    measurements: tuple[dict, ...]
    reaction_states: tuple[dict, ...]
    committed_prefix: CanonicalClip | None
    terrain_start_frame: int = 0


def _join(first, second):
    if first is None:
        return second
    if second is None:
        return first
    if first.actor_ids != second.actor_ids or first.fps != second.fps:
        raise ValueError("Native prefix actor order or frame rate changed")
    native = None if first.native_features is None or second.native_features is None else np.concatenate(
        (first.native_features, second.native_features), axis=1)
    return CanonicalClip(np.concatenate((first.positions, second.positions), axis=1),
                         np.concatenate((first.rotations, second.rotations), axis=1),
                         FPS, first.actor_ids, "offline_combined", {}, native)


def _native_digest(clip):
    digest = hashlib.sha256()
    for array in (clip.positions, clip.rotations, clip.native_features):
        if array is None:
            raise ValueError("Terrain action requires exact native feature history")
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _validate_committed_assisted(previous, prefix, scene, asset_sha=None):
    """Validate paired provenance before spending generation or solve work."""
    if previous is None:
        if prefix is not None:
            raise ValueError("Rig17 assistance requires the previous actual presentation pose; native-only prefix is insufficient")
        return
    if not isinstance(previous, AssistedTerrainResult) or prefix is None:
        raise ValueError("Expected matching committed assisted and native history")
    if (previous.native_clip.frames != previous.presentation.frames or
            previous.native_clip.actor_ids != previous.presentation.actor_ids or
            previous.native_clip.actor_ids != prefix.actor_ids or
            _native_digest(previous.native_clip) != _native_digest(prefix) or
            previous.report.get("native_sha256") != _native_digest(prefix) or
            previous.report.get("accepted") is not False or
            tuple(previous.report.get("rig_asset_sha256", ())) != previous.presentation.rig_asset_sha256):
        raise ValueError("Committed assisted/native prefix provenance mismatch")
    if validate_scene(scene) != previous.scene:
        raise ValueError("Cannot append terrain motion after changing its authored scene")
    if asset_sha is not None and previous.presentation.rig_asset_sha256 != (asset_sha,):
        raise ValueError("Committed presentation render asset changed")


def _cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise RuntimeError("Offline terrain action cancelled; provisional native motion discarded")


def _validate_native_body(action_native, scene, full_clip, start_frame, terrain_start_frame):
    """Check observed native root and major body proxies, deliberately not raw feet."""
    from motion_bridge import _layout
    names, _, _ = _layout()
    indices = {name: names.index(name) for name in
               ("Hips", "Spine3", "Head", "LeftLeg", "RightLeg", "LeftArm", "RightArm")}
    max_support_gap = 0.
    geometry = None
    previous_key = None
    previous_pose = None if start_frame == 0 else full_clip.positions[0, start_frame-1]
    previous_geometry = None
    for local in range(action_native.frames):
        frame = start_frame + local
        states = object_states(scene, full_clip, frame=frame, enabled=True,
                               terrain=True, terrain_start_frame=terrain_start_frame)
        key = tuple(tuple(row["position"]) for row in states)
        if key != previous_key:
            state = deepcopy(scene)
            for obj, row in zip(state["objects"], states):
                obj["position"] = row["position"]
            geometry = SceneInteractionGeometry.from_scene(state)
            previous_key = key
        pose = action_native.positions[0, local]
        root = pose[0]
        support = geometry.support_height(root[0], root[2], root[1]-.95,
                                          max_step_up=.35, max_drop=.35)
        if support is None:
            raise ValueError(f"Native root lacks authored terrain support at frame {frame}")
        max_support_gap = max(max_support_gap, abs(float(root[1]-.95-support)))
        if max_support_gap > .35:
            raise ValueError(f"Native root height misses authored terrain support at frame {frame}")
        for name, joint in indices.items():
            if geometry.obstacle_at(*pose[joint], radius=.09):
                raise ValueError(f"Native {name} overlaps authored terrain at frame {frame}")
        if previous_pose is not None:
            travel = max(np.linalg.norm(pose[j]-previous_pose[j]) for j in indices.values())
            count = max(2, int(np.ceil(travel/.04))+1)
            if count > 128:
                raise ValueError("Native terrain body jumps between frames")
            for amount in np.linspace(0., 1., count)[1:-1]:
                mixed = previous_pose*(1-amount)+pose*amount
                for candidate in (geometry, previous_geometry or geometry):
                    root = mixed[0]
                    if candidate.support_height(root[0], root[2], root[1]-.95,
                                                max_step_up=.35, max_drop=.35) is None:
                        raise ValueError(f"Native root swept across unsupported terrain near frame {frame}")
                    if any(candidate.obstacle_at(*mixed[j], radius=.09) for j in indices.values()):
                        raise ValueError(f"Native body swept through rendered terrain near frame {frame}")
        previous_pose, previous_geometry = pose, geometry
    return {"native_root_body_checked": True, "native_raw_feet_checked": False,
            "max_root_support_height_gap_m": max_support_gap}


def generate_native_terrain_commands(scene, text, *, actor_ids, actor_id,
                                     initial_placements, client, committed_prefix=None,
                                     cancelled=None, terrain_start_frame=None, previous_routes=(),
                                     planning_heading=None):
    """Buffer every native action privately; return only a fully measured route."""
    if not callable(getattr(client, "wait", None)):
        raise ValueError("A native client with wait(request_body, cancelled=...) is required")
    if cancelled is not None and not callable(cancelled):
        raise ValueError("cancelled must be callable")
    ids = tuple(actor_ids)
    if len(ids) != 1 or actor_id != ids[0]:
        raise ValueError("Offline assisted terrain currently supports one native actor")
    if committed_prefix is not None:
        if (committed_prefix.actor_ids != ids or committed_prefix.fps != FPS
                or committed_prefix.native_features is None
                or committed_prefix.frames < 4 or committed_prefix.frames % 4):
            raise ValueError("Committed prefix requires matching native Core features")
    authored = validate_scene(scene)
    private = RealtimeDirector(ids, target_buffer_frames=MAX_CLIP_FRAMES,
                               max_buffer_frames=MAX_CLIP_FRAMES)
    prefix_frames = 0 if committed_prefix is None else committed_prefix.frames
    if terrain_start_frame is None:
        terrain_start_frame = prefix_frames
    if type(terrain_start_frame) is not int or not 0 <= terrain_start_frame <= prefix_frames:
        raise ValueError("Terrain reaction origin must belong to committed history")
    native_full = committed_prefix
    adapted = adapt_studio_scene(evaluated_scene(authored, native_full, enabled=True,
                                                terrain=True, terrain_start_frame=terrain_start_frame))
    adapted.update(original_scene=authored, terrain_active=True)
    if isinstance(text, str) and text.strip().rstrip(".").casefold() == "enter" and previous_routes:
        previous_route = previous_routes[-1]
        gate = next((obj for obj in authored["objects"] if obj["id"] == previous_route.get("target_id")), None)
        if previous_route.get("verb") != "open" or gate is None or gate["kind"] != "door":
            raise ValueError("Enter needs the preceding committed open gate or an explicit target")
        actions = [{"verb": "go_through", "target_id": gate["id"]}]
    else:
        actions = parse_commands(text, adapted)
    routes, spans, measurements = [], [], []
    for index, action in enumerate(actions):
        _cancelled(cancelled)
        evaluated = evaluated_scene(authored, native_full, enabled=True,
                                    terrain=True, terrain_start_frame=terrain_start_frame)
        adapted = adapt_studio_scene(evaluated)
        adapted.update(original_scene=authored, terrain_active=True)
        if planning_heading is not None:
            adapted["terrain_planning_heading"] = planning_heading
        stages, route = plan_command(action, adapted, ids, actor_id,
                                     native_full, initial_placements)
        start = private.total_frames
        end = start + route["schedule"]["frames"]
        if prefix_frames + end > MAX_CLIP_FRAMES:
            raise ValueError("Offline terrain sequence exceeds the Core frame limit")
        private.queue_sequence(stages)
        while private.total_frames < end:
            _cancelled(cancelled)
            request = private.claim_request()
            if request is None:
                raise ValueError("Private native director cannot claim the next terrain horizon")
            if private.total_frames == 0 and committed_prefix is not None:
                history = committed_prefix.slice_frames(
                    committed_prefix.frames-min(HORIZON, committed_prefix.frames), committed_prefix.frames)
                request = replace(request, history=history)
            body = request_body(request)
            clips = client.wait(body, cancelled=cancelled)
            _cancelled(cancelled)
            if (len(clips) != 1 or not isinstance(clips[0], CanonicalClip)
                    or clips[0].native_features is None):
                raise ValueError("Core client must return one exact native feature horizon")
            native = clips[0]
            if request.history is not None:
                jump = np.linalg.norm(native.positions[0, 0, 0]-request.history.positions[0, -1, 0])
                if jump > .35:
                    raise ValueError("Native terrain continuation jumps across a frame boundary")
            if native.frames > 1 and np.max(np.linalg.norm(np.diff(native.positions[0, :, 0], axis=0), axis=1)) > .35:
                raise ValueError("Native terrain root jumps inside a horizon")
            if not private.complete(request.request_id, native):
                raise ValueError("Private native terrain request became stale")
        generated = private.timeline_clip()
        native_full = _join(committed_prefix, generated)
        action_native = generated.slice_frames(start, end)
        body_evidence = _validate_native_body(action_native, authored, native_full,
                                              prefix_frames+start, terrain_start_frame)
        measured = measure_completion(route, native_full, prefix_frames+start)
        measured.update(body_evidence)
        if action["verb"] == "open":
            states = object_states(authored, native_full, enabled=True, terrain=True,
                                   terrain_start_frame=terrain_start_frame)
            gate = next(state for state in states if state["id"] == route.get("target_id"))
            measured["automatic_door_open_verified"] = gate["opening_fraction"] >= .99
            measured["completed"] &= measured["automatic_door_open_verified"]
        if not measured["completed"]:
            raise ValueError(f"Native terrain action {index + 1} did not reach its measured goal: {measured}")
        routes.append(deepcopy(route))
        spans.append((prefix_frames+start, prefix_frames+end))
        measurements.append(measured)
        # The next relative command starts in the displayed direction of this
        # planned route. Native poses and feature history remain untouched.
        path = np.asarray(route["support_xyz"], dtype=float)
        for segment in np.diff(path[:, [0, 2]], axis=0)[::-1]:
            if np.linalg.norm(segment) > .05:
                planning_heading = math.atan2(float(segment[0]), float(segment[1]))
                break
    _cancelled(cancelled)
    states = object_states(authored, native_full, enabled=True, terrain=True,
                           terrain_start_frame=terrain_start_frame)
    return NativeTerrainResult(authored, private.timeline_clip(), tuple(routes),
                               tuple(spans), tuple(measurements), tuple(states),
                               committed_prefix, terrain_start_frame)


def _validate_rig_body(scene, full_native, poses, *, start_frame,
                       terrain_start_frame, previous_pose=None):
    """Sweep actual rig17 body bones against each observed gate/terrain state."""
    # Feet and sole vertices are checked by assist_rig_clip. The lower-leg
    # segments remain here because a clean sole can still cut through a riser.
    bones = ((0, 1), (1, 2), (1, 3), (3, 4), (4, 5),
             (1, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11),
             (0, 12), (12, 13), (13, 14))
    cache = {}
    previous = previous_pose
    previous_geometry = None
    checked = 0
    for local, pose in enumerate(poses):
        frame = start_frame + local
        states = object_states(scene, full_native, frame=frame, enabled=True,
                               terrain=True, terrain_start_frame=terrain_start_frame)
        key = tuple(tuple(row["position"]) for row in states)
        geometry = cache.get(key)
        if geometry is None:
            document = deepcopy(scene)
            for obj, row in zip(document["objects"], states):
                obj["position"] = row["position"]
            geometry = SceneInteractionGeometry.from_scene(document)
            cache[key] = geometry
        for a, b in bones:
            length = np.linalg.norm(pose[b]-pose[a])
            count = max(2, int(np.ceil(length/.04))+1)
            if count > 128:
                raise ValueError("Rig17 limb segment is implausibly long")
            for fraction in np.linspace(0., 1., count):
                point = pose[a]*(1-fraction)+pose[b]*fraction
                checked += 1
                if geometry.obstacle_at(*point, radius=.06):
                    raise ValueError(f"Rig17 body or limb intersects rendered scene at frame {frame}")
        if previous is not None:
            distance = np.linalg.norm(pose-previous, axis=1).max()
            count = max(2, int(np.ceil(distance/.04))+1)
            if count > 128:
                raise ValueError("Rig17 body motion jumps across frames")
            for fraction in np.linspace(0., 1., count)[1:-1]:
                swept = previous*(1-fraction)+pose*fraction
                for a, b in bones:
                    for along in (0., .25, .5, .75, 1.):
                        point = swept[a]*(1-along)+swept[b]*along
                        checked += 1
                        if any(candidate.obstacle_at(*point, radius=.06)
                               for candidate in (geometry, previous_geometry or geometry)):
                            raise ValueError(f"Rig17 swept body intersects rendered scene near frame {frame}")
        previous = pose
        previous_geometry = geometry
    return {"rig17_dynamic_body_sweep_checked": True,
            "body_proxy_radius_m": .06, "body_proxy_samples": checked,
            "proxy_only": True}


def assist_native_terrain_result(native_result, *, committed_assisted=None, cancelled=None, assistor=None):
    """Return a complete paired timeline, preserving committed arrays exactly.

    A native prefix requires its matching prior AssistedTerrainResult. Only new
    action frames are solved; the prior actual rig17 endpoint is constrained.
    """
    from types import SimpleNamespace
    from grounded_character import GroundedCharacter, _PARENTS, _SHOULDER_BLENDS
    from studio_core_renderer import DEFAULT_ASSETS
    if not isinstance(native_result, NativeTerrainResult):
        raise ValueError("Expected a complete native terrain result")
    if assistor is None:
        from terrain_assisted_rig import assist_rig_clip
        assistor = assist_rig_clip
    if not callable(assistor):
        raise ValueError("Rig17 assistor must be callable")
    character = GroundedCharacter(DEFAULT_ASSETS[0])
    native = native_result.native_clip
    prefix = native_result.committed_prefix
    _validate_committed_assisted(committed_assisted, prefix, native_result.scene,
                                 character.mesh_sha256)
    prefix_frames = 0 if prefix is None else prefix.frames
    origin = native_result.terrain_start_frame
    if type(origin) is not int or not 0 <= origin <= prefix_frames:
        raise ValueError("Invalid terrain reaction origin")
    if committed_assisted is not None and origin != committed_assisted.report.get("terrain_start_frame", 0):
        raise ValueError("Terrain reaction origin changed across append")
    spans = native_result.action_spans
    if (not spans or len(spans) != len(native_result.routes) or len(spans) != len(native_result.measurements)
            or spans[0][0] != prefix_frames or spans[-1][1] != prefix_frames+native.frames
            or any(a >= b for a, b in spans)
            or any(a[1] != b[0] for a, b in zip(spans, spans[1:]))):
        raise ValueError("Terrain action spans must cover the complete appended native timeline")
    full_native = _join(prefix, native)
    pose_parts, rotation_parts, stance_parts, evidence = [], [], [], []
    display_roots = []
    previous_assisted = None
    if committed_assisted is not None:
        previous = committed_assisted.presentation
        pose_parts.append(previous.positions)
        rotation_parts.append(previous.rotations)
        stance_parts.append(previous.stance)
        display_roots.append(previous.positions[:, :, 0])
        previous_assisted = (previous.positions[0, -1], previous.rotations[0, -1])
    shoulder_blends = {blend for _, blend in _SHOULDER_BLENDS}
    for index, (global_start, global_end) in enumerate(native_result.action_spans):
        _cancelled(cancelled)
        start, end = global_start-prefix_frames, global_end-prefix_frames
        action = native.slice_frames(start, end)
        prior = (prefix if start == 0 else native.slice_frames(start-1, start))
        input_p, input_r = action.positions[0], action.rotations[0]
        if prior is not None:
            input_p = np.concatenate((prior.positions[0, -1:], input_p), axis=0)
            input_r = np.concatenate((prior.rotations[0, -1:], input_r), axis=0)
        geometry = SceneInteractionGeometry.from_scene(evaluated_scene(
            native_result.scene, full_native, frame=global_end-1,
            enabled=True, terrain=True, terrain_start_frame=origin))
        kwargs = {"fps": FPS, "heading_assistance": True}
        if previous_assisted is not None:
            kwargs.update(initial_assisted_positions=previous_assisted[0],
                          initial_assisted_rotations=previous_assisted[1])
        p, r, stance, report = assistor(input_p, input_r, geometry, character, **kwargs)
        p, r, stance = np.asarray(p), np.asarray(r), np.asarray(stance)
        if prior is not None:
            if (previous_assisted is None or
                    not np.allclose(p[0], previous_assisted[0], atol=2e-5, rtol=0) or
                    not np.allclose(r[0], previous_assisted[1], atol=2e-5, rtol=0)):
                raise ValueError("Rig17 assistor changed the committed boundary pose")
            p, r, stance = p[1:], r[1:], stance[1:]
        if (p.shape != (action.frames, 17, 3) or
                r.shape != (action.frames, 17, 3, 3) or
                stance.shape != (action.frames, 2) or stance.dtype != np.bool_):
            raise ValueError("Rig17 assistance does not cover the complete native action")
        if not np.allclose(p[:, 0][:, [0, 2]], action.positions[0, :, 0][:, [0, 2]], atol=1e-5):
            raise ValueError("Rig17 assistance changed native root XZ route")
        if (not isinstance(report, dict) or report.get("accepted") is not False or
                report.get("rig_mesh_sha256") != character.mesh_sha256 or
                report.get("rig_bone_count") != 17 or
                report.get("numerical_rejections")):
            raise ValueError(f"Actual mesh17 assistance failed numerical or asset validation: {report}")
        for frame in range(action.frames):
            for joint in range(1, 17):
                if joint in shoulder_blends:
                    continue
                parent = _PARENTS[joint]
                predicted = p[frame, parent]+r[frame, parent]@(character.rest[joint]-character.rest[parent])
                if np.linalg.norm(predicted-p[frame, joint]) > .005:
                    raise ValueError("Rig17 presentation violates its exact bone lengths")
        if previous_assisted is not None:
            if (np.linalg.norm(p[0, 0]-previous_assisted[0][0]) > .35 or
                    np.linalg.norm(p[0]-previous_assisted[0], axis=1).max() > .5):
                raise ValueError("Rig17 action join jumps from the previous displayed pose")
        body_report = _validate_rig_body(native_result.scene, full_native, p,
                                         start_frame=global_start,
                                         terrain_start_frame=origin,
                                         previous_pose=None if previous_assisted is None else previous_assisted[0])
        report = {**report, "dynamic_body_validation": body_report}
        previous_assisted = (p[-1], r[-1])
        pose_parts.append(p[None])
        rotation_parts.append(r[None])
        stance_parts.append(stance[None])
        display_roots.append(p[None, :, 0])
        evidence.append(report)
    _cancelled(cancelled)
    presentation = PresentationClip(np.concatenate(pose_parts, axis=1),
                                    np.concatenate(rotation_parts, axis=1),
                                    np.concatenate(stance_parts, axis=1), FPS,
                                    native.actor_ids, (character.mesh_sha256,),
                                    {"actions": ([] if committed_assisted is None else
                                        deepcopy(committed_assisted.presentation.report or {}).get("actions", []))+evidence})
    if presentation.frames != full_native.frames:
        raise AssertionError("Native and rig17 presentation timelines diverged")
    root_history = np.concatenate(display_roots, axis=1)
    observed_display = SimpleNamespace(positions=root_history[:, :, None, :],
                                       frames=root_history.shape[1], fps=FPS)
    display_states = object_states(native_result.scene, observed_display,
                                   enabled=True, terrain=True,
                                   terrain_start_frame=origin)
    if any((a["trigger_frame"], a["opening_fraction"]) !=
           (b["trigger_frame"], b["opening_fraction"])
           for a, b in zip(native_result.reaction_states, display_states)):
        raise ValueError("Rig17 root Y changes a native-observed reaction")
    report = {"version": 1, "provenance": "offline actual-rig terrain-assisted candidate",
              "accepted": False, "visual_review": "pending", "runtime_publishable": False,
              "publish_scope": "explicit terrain-aware mode only", "ordinary_core_publishable": False,
              "native_features_unchanged": True, "native_sha256": _native_digest(full_native),
              "terrain_start_frame": origin, "prefix_preserved_exactly": True,
              "rig_asset_sha256": [character.mesh_sha256],
              "committed_prefix_frames": prefix_frames,
              "actions": ([] if committed_assisted is None else deepcopy(committed_assisted.report["actions"]))+[{"route": route, "native_span": span,
                           "measurement": measurement, "assistance": assist}
                          for route, span, measurement, assist in zip(
                              native_result.routes, native_result.action_spans,
                              native_result.measurements, evidence)]}
    return AssistedTerrainResult(native_result.scene, full_native, presentation, report,
                                 (() if committed_assisted is None else committed_assisted.routes)+native_result.routes,
                                 native_result.reaction_states)


def run_assisted_terrain_commands(scene, text, *, actor_ids, actor_id,
                                  initial_placements, client, committed_prefix=None,
                                  cancelled=None, assistor=None, on_native_ready=None,
                                  committed_assisted=None):
    """Return an all-or-nothing offline candidate for one bounded command sequence.

    `client.wait(request_body, cancelled=...)` must return one exact native
    CanonicalClip per 40-frame request. No provisional clip is published.
    """
    if committed_assisted is not None:
        if committed_prefix is None:
            committed_prefix = committed_assisted.native_clip
        _validate_committed_assisted(committed_assisted, committed_prefix, scene)
    elif committed_prefix is not None:
        raise ValueError("Rig17 assistance requires the previous actual presentation pose")
    origin = None if committed_assisted is None else committed_assisted.report.get("terrain_start_frame", 0)
    planning_heading = None
    if committed_assisted is not None:
        forward = committed_assisted.presentation.rotations[0, -1, 0, :, 2]
        planning_heading = math.atan2(float(forward[0]), float(forward[2]))
    native_result = generate_native_terrain_commands(
        scene, text, actor_ids=actor_ids, actor_id=actor_id,
        initial_placements=initial_placements, client=client,
        committed_prefix=committed_prefix, cancelled=cancelled, terrain_start_frame=origin,
        previous_routes=() if committed_assisted is None else committed_assisted.routes,
        planning_heading=planning_heading)
    if on_native_ready is not None:
        if not callable(on_native_ready):
            raise ValueError("on_native_ready must be callable")
        on_native_ready(native_result)
    return assist_native_terrain_result(native_result, cancelled=cancelled,
                                        assistor=assistor, committed_assisted=committed_assisted)

def save_assisted_result(result, path):
    """Save a self-contained offline candidate; never write a Core project."""
    if not isinstance(result, AssistedTerrainResult):
        raise ValueError("Expected an offline assisted terrain result")
    if (result.native_clip.frames != result.presentation.frames or
            result.native_clip.actor_ids != result.presentation.actor_ids or
            result.report.get("accepted") is not False or
            tuple(result.report.get("rig_asset_sha256", ())) != result.presentation.rig_asset_sha256 or
            result.report.get("native_sha256") != _native_digest(result.native_clip)):
        raise ValueError("Offline terrain streams or provenance do not match")
    manifest = {"version": 1, "scene": result.scene, "report": result.report,
                "routes": result.routes, "reaction_states": result.reaction_states,
                "native_sha256": _native_digest(result.native_clip),
                "rig_asset_sha256": result.presentation.rig_asset_sha256,
                "presentation_report": result.presentation.report,
                "actor_ids": result.native_clip.actor_ids, "fps": FPS}
    _write_archive(path, manifest,
                        native_positions=result.native_clip.positions,
                        native_rotations=result.native_clip.rotations,
                        native_features=result.native_clip.native_features,
                        presentation_positions=result.presentation.positions,
                        presentation_rotations=result.presentation.rotations,
                        presentation_stance=result.presentation.stance)


def save_native_terrain_result(result, path):
    """Persist native ARDY features and measured action boundaries for CPU retry."""
    if not isinstance(result, NativeTerrainResult):
        raise ValueError("Expected a complete native terrain result")
    manifest = {"version": 1, "kind": "native_terrain", "scene": result.scene,
                "routes": result.routes, "action_spans": result.action_spans,
                "measurements": result.measurements,
                "reaction_states": result.reaction_states,
                "terrain_start_frame": result.terrain_start_frame,
                "native_sha256": _native_digest(result.native_clip),
                "actor_ids": result.native_clip.actor_ids, "fps": FPS,
                "prefix_frames": 0 if result.committed_prefix is None else result.committed_prefix.frames,
                "prefix_sha256": None if result.committed_prefix is None else
                                 _native_digest(result.committed_prefix)}
    arrays = {"native_positions": result.native_clip.positions,
              "native_rotations": result.native_clip.rotations,
              "native_features": result.native_clip.native_features}
    if result.committed_prefix is not None:
        arrays.update(prefix_positions=result.committed_prefix.positions,
                      prefix_rotations=result.committed_prefix.rotations,
                      prefix_features=result.committed_prefix.native_features)
    _write_archive(path, manifest, **arrays)


def _write_archive(path, manifest, **arrays):
    encoded = json.dumps(manifest, allow_nan=False).encode("utf-8")
    if len(encoded) > 1_000_000:
        raise ValueError("Offline terrain manifest is too large")
    if sum(np.asarray(value).nbytes for value in arrays.values()) > MAX_UNCOMPRESSED_BYTES:
        raise ValueError("Offline terrain arrays are too large")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, manifest=np.frombuffer(encoded, np.uint8), **arrays)
    data = buffer.getvalue()
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError("Offline terrain archive is too large")
    target = Path(path)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name+".",
                                     suffix=".tmp", delete=False) as temporary:
        scratch = Path(temporary.name)
        try:
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        except Exception:
            scratch.unlink(missing_ok=True)
            raise
    try:
        os.replace(scratch, target)
    except Exception:
        scratch.unlink(missing_ok=True)
        raise


def _validate_array_headers(zipped, entries):
    """Reject impossible NPY allocations before NumPy materializes ZIP members."""
    for entry in entries:
        with zipped.open(entry) as member:
            version = np.lib.format.read_magic(member)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(member)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(member)
            else:
                raise ValueError("Unsupported terrain array encoding")
            size = math.prod(shape)*dtype.itemsize
            if (dtype.hasobject or len(shape) > 6 or any(n < 0 for n in shape)
                    or size > MAX_UNCOMPRESSED_BYTES
                    or size != entry.file_size-member.tell()):
                raise ValueError("Invalid or oversized terrain array header")


def _archive_arrays(path, expected):
    path = Path(path)
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Offline terrain archive is too large")
    with zipfile.ZipFile(path) as zipped:
        entries = zipped.infolist()
        if (len(entries) != len(expected) or
                {item.filename for item in entries} != {name+".npy" for name in expected} or
                len({item.filename for item in entries}) != len(entries) or
                sum(item.file_size for item in entries) > MAX_UNCOMPRESSED_BYTES or
                any(item.flag_bits & 1 for item in entries)):
            raise ValueError("Offline terrain archive has unexpected or oversized entries")
        manifest_info = zipped.getinfo("manifest.npy")
        if manifest_info.file_size > 1_000_128:
            raise ValueError("Offline terrain manifest is too large")
        _validate_array_headers(zipped, entries)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in expected}
    manifest_bytes = np.asarray(arrays.pop("manifest"))
    if manifest_bytes.dtype != np.uint8 or manifest_bytes.size > 1_000_000:
        raise ValueError("Invalid offline terrain manifest encoding")
    return json.loads(manifest_bytes.tobytes()), arrays


def load_assisted_result(path):
    """Read only a bounded, versioned offline candidate with distinct streams."""
    manifest, arrays = _archive_arrays(path, {"manifest", "native_positions", "native_rotations",
        "native_features", "presentation_positions", "presentation_rotations", "presentation_stance"})
    if manifest.get("version") != 1 or manifest.get("fps") != FPS:
        raise ValueError("Unsupported offline terrain archive version")
    ids = tuple(manifest["actor_ids"])
    native = CanonicalClip(arrays["native_positions"], arrays["native_rotations"],
                           FPS, ids, "ardy_core", {}, arrays["native_features"])
    presentation = PresentationClip(arrays["presentation_positions"],
                                    arrays["presentation_rotations"],
                                    arrays["presentation_stance"], FPS, ids,
                                    tuple(manifest["rig_asset_sha256"]),
                                    manifest.get("presentation_report"))
    if (native.frames != presentation.frames or _native_digest(native) != manifest["native_sha256"]
            or manifest["report"].get("accepted") is not False or
            tuple(manifest["report"].get("rig_asset_sha256", ())) != presentation.rig_asset_sha256):
        raise ValueError("Offline terrain archive provenance mismatch")
    return AssistedTerrainResult(validate_scene(manifest["scene"]), native, presentation,
                                 manifest["report"], tuple(manifest["routes"]),
                                 tuple(manifest["reaction_states"]))


def load_native_terrain_result(path):
    """Load a native-only route so rig17 assistance can be retried without GPU."""
    path = Path(path)
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Offline terrain archive is too large")
    expected = {"manifest", "native_positions", "native_rotations", "native_features"}
    with zipfile.ZipFile(path) as zipped:
        if "prefix_positions.npy" in zipped.namelist():
            expected.update(("prefix_positions", "prefix_rotations", "prefix_features"))
    manifest, arrays = _archive_arrays(path, expected)
    if manifest.get("version") != 1 or manifest.get("kind") != "native_terrain" or manifest.get("fps") != FPS:
        raise ValueError("Unsupported native terrain archive")
    ids = tuple(manifest["actor_ids"])
    native = CanonicalClip(arrays["native_positions"], arrays["native_rotations"],
                           FPS, ids, "ardy_core", {}, arrays["native_features"])
    prefix = None
    if "prefix_positions" in arrays:
        prefix = CanonicalClip(arrays["prefix_positions"], arrays["prefix_rotations"],
                               FPS, ids, "ardy_core", {}, arrays["prefix_features"])
    if (_native_digest(native) != manifest["native_sha256"] or
            (0 if prefix is None else prefix.frames) != manifest["prefix_frames"] or
            (None if prefix is None else _native_digest(prefix)) != manifest.get("prefix_sha256") or
            any(type(span) is not list or len(span) != 2 or
                any(type(value) is not int for value in span) or
                span[0] < manifest["prefix_frames"] or span[1] > manifest["prefix_frames"]+native.frames or
                span[0] >= span[1] for span in manifest["action_spans"])):
        raise ValueError("Native terrain archive provenance mismatch")
    return NativeTerrainResult(validate_scene(manifest["scene"]), native,
                               tuple(manifest["routes"]),
                               tuple(tuple(span) for span in manifest["action_spans"]),
                               tuple(manifest["measurements"]),
                               tuple(manifest["reaction_states"]), prefix,
                               manifest.get("terrain_start_frame", manifest["prefix_frames"]))
