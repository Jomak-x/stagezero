"""Explicit, independent native Core studio playback and inference ownership.

The G1 take/session is never read or written here. A single lazy worker performs
bounded Core horizons; cancellation invalidates its result before another take,
scene, or display mode can use it. No network health probe runs on construction.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import threading
import time

import numpy as np

from realtime_client import RealtimeClient, request_body
from realtime_clip import CanonicalClip, FPS
from realtime_director import RealtimeDirector, StageSpec
from realtime_navigation import plan_navigation
from core_spatial_commands import SpatialSequence
from scene_composition import validate_scene

SCHEMA = "Native Core27 / 20 fps (separate from G1)"
EMPTY_SCENE = {"version": 2, "name": "Native Core studio", "objects": [], "effects": [], "lighting": "neutral"}
EXAMPLE = Path(__file__).parent / "assets/core-motion/martial-combo.npz"


def _copy(value):
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("Scene and placement metadata must be finite JSON") from exc
    if len(encoded) > 900_000:
        raise ValueError("Scene metadata is too large")
    return json.loads(encoded)


def _placements(ids, value):
    if value is None:
        value = {aid: {"position_xz": [0. if len(ids) == 1 else (-1.2 if i == 0 else 1.2), 0.], "yaw": 0.}
                 for i, aid in enumerate(ids)}
    result = _copy(value)
    if not isinstance(result, dict) or set(result) != set(ids):
        raise ValueError("Placements must cover all native actor IDs")
    for item in result.values():
        if not isinstance(item, dict) or set(item) - {"position_xz", "yaw"}:
            raise ValueError("Placement requires position_xz and optional yaw")
        xz, yaw = item.get("position_xz"), item.get("yaw", 0.)
        if (not isinstance(xz, list) or len(xz) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 100 for v in xz)
                or type(yaw) not in (int, float) or not math.isfinite(yaw) or abs(yaw) > math.pi):
            raise ValueError("Invalid native actor placement")
    return result


class CoreStudioSession:
    """UI-independent native timeline. ``client=None`` supports offline playback."""

    def __init__(self, client: RealtimeClient | None = None, *, clock=time.monotonic):
        self._lock = threading.RLock()
        self._clock = clock
        self._client = client
        self._director = RealtimeDirector(("actor_1",), clock=clock)
        self._scene = _copy(EMPTY_SCENE)
        self._placements = _placements(self._director.actor_ids, None)
        self._active = False
        self._initialized = False
        self._scene_changed_since_motion = False
        self._closed = False
        self._generation_enabled = False
        self._epoch = 0
        self._wake = threading.Event()
        self._thread = None
        self._route = None
        self._spatial_sequence = SpatialSequence(self)

    @property
    def active(self):
        with self._lock:
            return self._active

    @property
    def available(self):
        with self._lock:
            return self._client is not None and not self._closed

    @property
    def example_available(self):
        return EXAMPLE.is_file()

    @property
    def scene_document(self):
        with self._lock:
            return _copy(self._scene)

    @property
    def scene_reactions_enabled(self):
        return self._director.project_metadata.get("studio_core", {}).get("scene_reactions_version") == 1

    def _reaction_options(self, *, enabling=False):
        metadata = self._director.project_metadata.get("studio_core", {})
        enabled = self.scene_reactions_enabled
        return {"enabled": enabled or enabling,
                "start_frame": metadata.get("scene_reactions_start_frame", 0) if enabled
                               else self._director.total_frames if enabling else 0}

    def _metadata(self, scene, placements):
        return {"studio_core": {"version": 1, "schema": SCHEMA,
                "original_scene_document": _copy(scene), "scene_document": _copy(scene),
                "initial_placements": _copy(placements), "scene_changed_since_motion": False}}

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("Native Core studio is closed")

    def _require_active(self, *, inference=False):
        self._ensure_open()
        if not self._active:
            raise RuntimeError("Activate Native Core studio first")
        if inference and not self._initialized:
            raise RuntimeError("Start a native Core actor cast first")
        if inference and self._client is None:
            raise RuntimeError("Native Core service is not configured; load a native archive or example for playback")

    def _invalidate(self):
        self._spatial_sequence.cancel()
        self._epoch += 1
        self._generation_enabled = False
        self._director.pause()
        self._director.cancel_pending()
        self._director.take_cancellations()
        self._wake.set()

    def configure_client(self, client):
        with self._lock:
            self._ensure_open()
            # A transport repair must retain the failed instruction for retry.
            # Invalidate the old worker result without dropping queued intent.
            self._epoch += 1
            self._generation_enabled = False
            self._director.pause()
            inflight = self._director.snapshot()["inflight_request_id"]
            if inflight is not None:
                self._director.fail(inflight, "Service configuration changed; retry pending native motion")
            self._client = client
            self._wake.set()

    def start(self, actor_count=1, scene_document=None, placements=None):
        if type(actor_count) is not int or actor_count not in (1, 2):
            raise ValueError("Native Core supports one or two actors")
        ids = tuple(f"actor_{i + 1}" for i in range(actor_count))
        scene = _copy(EMPTY_SCENE if scene_document is None else scene_document)
        if not isinstance(scene, dict):
            raise ValueError("Scene document must be an object")
        validate_scene(scene)
        if placements is None:
            from studio_interaction_scene import recommend_placements
            placements = dict(zip(ids, recommend_placements(scene, actor_count)))
        place = _placements(ids, placements)
        director = RealtimeDirector(ids, clock=self._clock, project_metadata=self._metadata(scene, place))
        with self._lock:
            self._ensure_open()
            self._invalidate()
            self._director, self._scene, self._placements = director, scene, place
            self._spatial_sequence.restore()
            self._scene_changed_since_motion = False
            self._route = None
            self._active = True
            self._initialized = True
        return self.snapshot()

    reset = start

    def activate(self):
        with self._lock:
            self._ensure_open()
            self._active = True
        return self.snapshot()

    def deactivate(self):
        with self._lock:
            self._invalidate()
            self._active = False

    stop = deactivate

    def cancel(self):
        with self._lock:
            self._ensure_open()
            self._invalidate()

    def update_scene(self, scene_document, placements=None):
        scene = _copy(scene_document)
        if not isinstance(scene, dict):
            raise ValueError("Scene document must be an object")
        validate_scene(scene)
        with self._lock:
            self._ensure_open()
            place = self._placements if placements is None else _placements(self._director.actor_ids, placements)
            if scene == self._scene and place == self._placements:
                return False
            geometry_changed = any(scene.get(key, []) != self._scene.get(key, []) for key in ("objects", "assets"))
            if geometry_changed and self._director.total_frames:
                self._scene_changed_since_motion = True
            self._invalidate()
            self._scene, self._placements = scene, place
            metadata = self._director.project_metadata.setdefault("studio_core", self._metadata(scene, place)["studio_core"])
            metadata["scene_document"] = _copy(scene)
            metadata["initial_placements"] = _copy(place)
            metadata["scene_changed_since_motion"] = self._scene_changed_since_motion
            self._route = None
            return True

    def _run_generation(self):
        self._generation_enabled = True
        self._director.play()
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, name="native-core-studio", daemon=True)
            self._thread.start()
        self._wake.set()

    def direct(self, prompts, seconds=2, *, queued=False):
        if type(seconds) not in (int, float) or seconds not in (2, 6, 12):
            raise ValueError("Choose a 2, 6, or 12 second native action")
        with self._lock:
            self._require_active(inference=True)
            if not isinstance(prompts, dict) or set(prompts) != set(self._director.actor_ids):
                raise ValueError("Prompts must cover every native actor ID")
            if queued and self._spatial_sequence.pending:
                raise ValueError("Wait for spatial commands before queueing ordinary motion")
            ids = self._director.actor_ids
            summary = (prompts[ids[0]] if len(ids) == 1 else
                       " / ".join(f"{aid}: {prompts[aid]}" for aid in ids)[:500])
            spec = StageSpec(summary, frames=int(seconds * FPS), actor_prompts=prompts,
                             metadata={"initial_placements": self._placements,
                                       "actor_prompts": dict(prompts)})
            if queued:
                result = self._director.queue_sequence((spec,))[0]
            else:
                result = self._director.interrupt(spec.prompt, frames=spec.frames,
                    actor_prompts=spec.actor_prompts, metadata=spec.metadata)
                self._epoch += 1
                self._spatial_sequence.cancel()
            self._director.take_cancellations()
            self._route = None
            self._run_generation()
            return result

    def choreograph(self, plan, *, queued=False):
        """Schedule a bounded shared beat plan without replacing committed poses.

        Validation and capacity checks finish before pending work is cancelled.
        All output still passes the existing native history, collision and floor
        checks. A spatial plan cannot queue behind unresolved motion because its
        sequence-start roots would then be unknown.
        """
        from core_choreography import build_choreography
        from realtime_clip import MAX_CLIP_FRAMES
        if type(queued) is not bool:
            raise ValueError("queued must be a boolean")
        with self._lock:
            self._require_active(inference=True)
            stages, report = build_choreography(plan, self._director.actor_ids,
                initial_placements=self._placements, last_clip=self._director.timeline_clip())
            pending = self._director.snapshot()
            if queued and self._spatial_sequence.pending:
                raise ValueError("Wait for spatial commands before queueing choreography")
            spatial = bool(report["plan"].get("recipe")) or any("root_offsets" in beat for beat in report["plan"]["beats"])
            if queued and spatial and (pending["queued_stages"] or pending["inflight_request_id"]):
                raise ValueError("Spatial choreography must start from known committed roots; wait for pending motion")
            if self._director.total_frames + report["frames"] > MAX_CLIP_FRAMES:
                raise ValueError("timeline exceeds the 15000-frame limit")
            if spatial:
                from studio_interaction_scene import adapt_studio_scene
                from realtime_navigation import validate_ground_path
                scene = adapt_studio_scene(self._scene)["scene"]
                for actor_id in self._director.actor_ids:
                    path = [report["origins_xz"][actor_id]]
                    for stage in stages:
                        path.extend(target["position_xz"] for target in stage.metadata.get("root_targets", {}).get(actor_id, []))
                    validate_ground_path(scene, path, actor_radius_m=.28)
            # A fresh director validates the complete native sequence before the
            # existing queue can be changed. queue_sequence itself is atomic.
            RealtimeDirector(self._director.actor_ids).queue_sequence(stages)
            if not queued:
                self._invalidate()
            stage_ids = self._director.queue_sequence(stages)
            report["stage_ids"] = list(stage_ids)
            report["submitted_after_committed_frame"] = self._director.total_frames
            report["queued"] = queued
            self._director.project_metadata["studio_core"]["last_choreography"] = _copy(report)
            self._route = None
            self._run_generation()
            return _copy(report)

    def navigate(self, actor_id, target_id, verb="approach"):
        from studio_interaction_scene import adapt_studio_scene
        with self._lock:
            self._require_active(inference=True)
            from core_scene_reactions import evaluated_scene
            adapted = adapt_studio_scene(evaluated_scene(self._scene, self._director.timeline_clip(),
                                                        **self._reaction_options()))
            stages, route = plan_navigation(adapted["scene"], self._director.actor_ids,
                actor_id=actor_id, target_id=target_id, verb=verb,
                last_clip=self._director.timeline_clip(), initial_placements=self._placements,
                affordances=adapted["affordances"])
            # Validate/planning finishes before cancellation, preserving a usable
            # existing queue if the target is invalid or route is obstructed.
            self._invalidate()
            self._director.queue_sequence(stages)
            self._route = route
            self._run_generation()
            return _copy(route)

    def spatial_commands(self, actor_id, text):
        """Submit bounded spatial commands; each leg starts from committed motion."""
        with self._lock:
            self._require_active(inference=True)
            return _copy(self._spatial_sequence.start(actor_id, text))

    def _worker(self):
        while True:
            with self._lock:
                if self._closed:
                    return
                director, client, epoch = self._director, self._client, self._epoch
                scene = self._scene
                reaction_options = self._reaction_options()
                committed_history = director.timeline_clip() if reaction_options["enabled"] else None
                request = (director.claim_request() if self._active and self._generation_enabled
                           and client is not None else None)
            if request is None:
                self._wake.wait(.05)
                self._wake.clear()
                continue

            def cancelled():
                with self._lock:
                    return (self._closed or not self._active or self._epoch != epoch
                            or self._director is not director or not self._generation_enabled)

            try:
                body = request_body(request)
                clips = client.wait(body, cancelled=cancelled)
                if len(clips) != 1 or not isinstance(clips[0], CanonicalClip) or clips[0].native_features is None:
                    raise ValueError("Core service must return one native 40-frame horizon")
                if not cancelled():
                    self._check_geometry(clips[0], scene, history=request.history,
                                         reaction_history=committed_history, **reaction_options)
                    self._check_continuity(request.history, clips[0])
                    if request.metadata.get("pose_cue_profile"):
                        separation = np.linalg.norm(clips[0].positions[0, :, :, None, :] - clips[0].positions[1, :, None, :, :], axis=-1)
                        if float(separation.min()) < .15:
                            raise ValueError("Pose duet joint clearance below 0.15 m; last good motion retained (proxy, not mesh contact)")
                        from core_pose_cues import measure_cue_result
                        clip = clips[0]
                        audit = {"minimum_cross_actor_joint_distance_m": float(separation.min()),
                                 "clearance_proxy_only": True, "cue_result": measure_cue_result(body, clip)}
                        clips = [CanonicalClip(clip.positions, clip.rotations, clip.fps, clip.actor_ids,
                                 clip.source, {**clip.metadata, "pose_cue_audit": audit}, clip.native_features)]
                with self._lock:
                    if not cancelled():
                        if director.complete(request.request_id, clips[0]):
                            self._spatial_sequence.committed()
            except Exception as exc:
                with self._lock:
                    if not cancelled():
                        director.fail(request.request_id, exc)
                        self._spatial_sequence.failed(exc)

    @staticmethod
    def _check_continuity(history, clip):
        if history is None:
            return
        root_steps = np.linalg.norm(clip.positions[:, 0, 0] - history.positions[:, -1, 0], axis=-1)
        if float(root_steps.max()) > .35:
            raise ValueError(f"Native continuation root discontinuity ({root_steps.max():.2f} m); last good motion retained")

    @staticmethod
    def _check_geometry(clip, scene, *, history=None, reaction_history=None, enabled=False, start_frame=0):
        from studio_interaction_scene import adapt_studio_scene
        from core_scene_reactions import check_reactive_geometry
        from interaction_metrics import pair_separation
        from realtime_navigation import validate_ground_path
        adapted = adapt_studio_scene(scene)
        check_reactive_geometry(clip, scene, reaction_history if enabled else history,
                                enabled=enabled, start_frame=start_frame)
        for index, actor_id in enumerate(clip.actor_ids):
            root_path = clip.positions[index, :, 0, :][:, [0, 2]]
            if history is not None:
                root_path = np.vstack((history.positions[index, -1, 0, [0, 2]], root_path))
            validate_ground_path(adapted["scene"], root_path, actor_radius_m=.28)
        if len(clip.actor_ids) == 2:
            report = pair_separation(clip.positions[0], clip.positions[1],
                                    skeleton_a="core27", skeleton_b="core27",
                                    radius_a_m=.325, radius_b_m=.325)
            if report["root_disc_overlap_proxy_frames"]:
                raise ValueError("Native actors overlap the 0.65 m root separation proxy; last good motion retained")

    def play(self):
        with self._lock:
            self._require_active()
            self._director.play()
            self._wake.set()

    def pause(self):
        with self._lock:
            self._director.pause()

    def seek(self, frame):
        with self._lock:
            self._require_active()
            self._director.seek(frame)
            self._wake.set()

    def retry(self):
        with self._lock:
            self._require_active(inference=True)
            self._director.retry()
            pending = self._director.snapshot()["queued_stages"] > 0
            if pending:
                self._spatial_sequence.retry()
                self._run_generation()
            return pending

    def snapshot(self):
        with self._lock:
            result = self._director.snapshot()
            result.update(active=self._active, epoch=self._epoch, initialized=self._initialized,
                          scene_changed_since_motion=self._scene_changed_since_motion, available=self.available, schema=SCHEMA,
                          example_available=self.example_available, route=_copy(self._route),
                          spatial_commands=_copy(self._spatial_sequence.report),
                          scene_reactions_enabled=self.scene_reactions_enabled,
                          scene_reactions_start_frame=self._reaction_options()["start_frame"],
                          fps=FPS, segments=self._director.segments,
                          geometry_check="sampled body spheres vs scene boxes; actor root discs; continuous root footprint on authored floors (not mesh physics)")
            spatial = self._spatial_sequence.report
            if spatial:
                result["status"] = (f"Spatial commands: {spatial['completed_actions']}/{len(spatial['actions'])} verified; "
                                    f"{spatial['status']}. " + spatial.get("detail", "") + " " + result["status"])
            if not self._active:
                result["phase"], result["status"] = "inactive", "Native Core inactive; the G1 session is independent."
            elif self._client is None and not result["generated"]:
                result["status"] = "Native Core service is not configured. Load a native archive or bundled example."
            if self._scene_changed_since_motion:
                result["status"] = ("Scene changed; pending motion cancelled. Stored motion was created against an earlier layout. "
                                    + result["status"])
            return result

    def tick(self, now=None):
        with self._lock:
            if self._active:
                self._director.tick(now=now)
                self._wake.set()
            return self.snapshot()

    def frame_pose(self, frame=None):
        with self._lock:
            if not self._director.total_frames:
                return None
            return self._director.frame_pose(frame)

    def timeline_clip(self):
        with self._lock:
            return self._director.timeline_clip()

    def save(self):
        with self._lock:
            content = self._director.save_project()
            if len(content) > 64_000_000:
                raise ValueError("Native studio archive exceeds the 64 MB limit")
            return content

    save_project = save

    def load(self, content):
        if not isinstance(content, bytes) or len(content) > 64_000_000:
            raise ValueError("Native studio archive must be at most 64 MB")
        director = RealtimeDirector.load_project(content, clock=self._clock)
        metadata = director.project_metadata.get("studio_core")
        if (director.mode != "production" or director.snapshot()["contact_open"]
                or not isinstance(metadata, dict) or metadata.get("version") != 1
                or metadata.get("schema") != SCHEMA
                or any(s["source"] != "ardy_core" or s["kind"] not in ("action", "approach")
                       for s in director.segments)
                or director.timeline_clip().native_features is None):
            raise ValueError("Expected a production Native Core studio archive, separate from G1 and research motion")
        # Loaded pending stages are validated as production by the director. Do
        # not infer simply because an archive contained a queued instruction.
        scene = _copy(metadata["scene_document"])
        original = _copy(metadata["original_scene_document"])
        if not isinstance(scene, dict) or not isinstance(original, dict):
            raise ValueError("Native archive is missing its scene snapshots")
        validate_scene(scene)
        validate_scene(original)
        changed = metadata.get("scene_changed_since_motion", any(
            scene.get(key, []) != original.get(key, []) for key in ("objects", "assets")))
        if type(changed) is not bool:
            raise ValueError("Invalid saved scene-change provenance")
        if "scene_reactions_version" in metadata:
            version = metadata["scene_reactions_version"]
            start = metadata.get("scene_reactions_start_frame")
            if type(version) is not int or version != 1 or type(start) is not int or not 0 <= start <= director.total_frames:
                raise ValueError("Invalid native scene reaction provenance")
        elif "scene_reactions_start_frame" in metadata:
            raise ValueError("Native scene reaction start requires a supported version")
        place = _placements(director.actor_ids, metadata["initial_placements"])
        with self._lock:
            self._ensure_open()
            self._invalidate()
            self._director, self._scene, self._placements = director, scene, place
            self._spatial_sequence.restore()
            self._scene_changed_since_motion = changed
            self._route = None
            self._active = True
            self._initialized = True
        return self.snapshot()

    load_project = load

    def load_example(self):
        with np.load(EXAMPLE, allow_pickle=False) as data:
            meta = json.loads(data["metadata"].item())
            clip = CanonicalClip(data["positions"], data["rotations"], FPS, ("actor_1",),
                                 "ardy_core", meta, data["native_features"])
        if clip.frames > 160:
            raise ValueError("Bundled example exceeds its 160-frame playback buffer")
        # The included sample has absolute native world poses. Show it in its
        # own empty reference stage, never inside arbitrary authored props.
        scene = {**_copy(EMPTY_SCENE), "name": "Native Core martial arts reference"}
        validate_scene(scene)
        placements = _placements(clip.actor_ids, None)
        metadata = self._metadata(scene, placements)
        metadata["studio_core"]["bundled_example"] = EXAMPLE.name
        director = RealtimeDirector(clip.actor_ids, clock=self._clock, target_buffer_frames=160,
                                    max_buffer_frames=160, project_metadata=metadata)
        director.queue_sequence((StageSpec(meta["prompt"], frames=clip.frames),))
        for start in range(0, clip.frames, 40):
            request = director.claim_request()
            director.complete(request.request_id, clip.slice_frames(start, start + 40))
        with self._lock:
            self._ensure_open()
            self._invalidate()
            self._director, self._scene, self._placements, self._active = director, scene, placements, True
            self._spatial_sequence.restore()
            self._scene_changed_since_motion = False
            self._initialized = True
            self._route = None
        return self.snapshot()

    def close(self, timeout=1.):
        with self._lock:
            if self._closed:
                return
            self._invalidate()
            self._active = False
            self._closed = True
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0., min(float(timeout), 2.)))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
