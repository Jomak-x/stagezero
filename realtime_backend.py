"""Bounded, authenticated, stage-based Core/InterGen research service.

Core windows are published as they finish. A chunk is a complete synchronous
multi-actor window; no endpoint claims to stream individual model frames.
"""
from __future__ import annotations

import io
import json
import queue
import secrets
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from numbers import Integral
from typing import Callable
from urllib.parse import quote, unquote, urlsplit

import numpy as np

from interaction_runtime import (
    FEATURES, FPS, HORIZON, JOINTS, MAX_FRAMES, GenerationCancelled,
    InteractionRuntime, validate_request,
)

MAX_BODY_BYTES = 8_000_000
MAX_QUEUED = 2
MAX_RETAINED = 16
MAX_JOB_SECONDS = 600
MAX_PAIR_FRAMES = 120
STAGES = {"approach", "transition", "paired", "continuation"}


def _array(value, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite numbers") from exc
    if array.ndim != len(shape) or any(s is not None and array.shape[i] != s for i, s in enumerate(shape)):
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numbers")
    return np.array(array, copy=True)


def _rotations(value, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    rotations = _array(value, shape, name)
    if not np.allclose(rotations @ np.swapaxes(rotations, -1, -2), np.eye(3), atol=.03):
        raise ValueError(f"{name} must be orthonormal")
    if not np.allclose(np.linalg.det(rotations), 1, atol=.03):
        raise ValueError(f"{name} must have determinant one")
    return rotations


def validate_job(body: dict, *, pair_enabled: bool = False) -> dict:
    """Validate the HTTP boundary before any CUDA work or queue insertion."""
    allowed = {"request_id", "stage_kind", "frames", "prompt", "actor_ids", "seed",
               "actor_prompts", "history", "root_targets", "target", "coordinate_frames_y",
               "pair_sequence_id", "source_start_frame", "source_total_frames",
               "initial_placements", "hand_target"}
    if not isinstance(body, dict) or set(body) - allowed or not {"request_id", "stage_kind", "frames", "prompt", "actor_ids", "seed"} <= set(body):
        raise ValueError("Invalid realtime job fields")
    request_id = body["request_id"]
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 100 or any(ord(c) < 32 for c in request_id):
        raise ValueError("request_id must be 1–100 printable characters")
    kind = body["stage_kind"]
    if kind not in STAGES:
        raise ValueError("Unsupported stage_kind")
    if kind == "paired" and not pair_enabled:
        raise ValueError("InterGen research mode is disabled")
    frames = body["frames"]
    upper = MAX_PAIR_FRAMES if kind == "paired" else MAX_FRAMES
    if isinstance(frames, bool) or not isinstance(frames, Integral) or not HORIZON <= frames <= upper or frames % HORIZON:
        raise ValueError(f"frames must be a multiple of {HORIZON} between {HORIZON} and {upper}")
    prompt = body["prompt"]
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 500:
        raise ValueError("prompt must contain 1–500 characters")
    ids = body["actor_ids"]
    if not isinstance(ids, list) or not 1 <= len(ids) <= 2 or len(set(ids)) != len(ids) or any(
            not isinstance(x, str) or not 1 <= len(x) <= 64 for x in ids):
        raise ValueError("actor_ids must contain one or two unique strings")
    if kind in ("paired", "transition") and len(ids) != 2:
        raise ValueError(f"{kind} requires two actors")
    seed = body["seed"]
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be a uint32")
    prompts = body.get("actor_prompts", {})
    if not isinstance(prompts, dict) or set(prompts) - set(ids) or any(
            not isinstance(p, str) or not 1 <= len(p.strip()) <= 500 for p in prompts.values()):
        raise ValueError("actor_prompts must map actor IDs to short text")
    root_targets = body.get("root_targets", {})
    if not isinstance(root_targets, dict) or set(root_targets) - set(ids):
        raise ValueError("root_targets must map actor IDs to targets")
    placements = body.get("initial_placements", {})
    if not isinstance(placements, dict) or set(placements) - set(ids):
        raise ValueError("initial_placements must map actor IDs to placements")
    frames_y = body.get("coordinate_frames_y", {})
    if not isinstance(frames_y, dict) or set(frames_y) - set(ids):
        raise ValueError("coordinate_frames_y must map actor IDs to terrain frame origins")
    actors = []
    for actor_id in ids:
        placement = placements.get(actor_id, {})
        if not isinstance(placement, dict) or set(placement) - {"position_xz", "yaw"}:
            raise ValueError("initial placement needs position_xz and optional yaw")
        actor = {"id": actor_id, "prompt": prompts.get(actor_id, prompt), "seed": int(seed),
                 "root_targets": root_targets.get(actor_id, [])}
        if "position_xz" in placement:
            actor["initial_position_xz"] = placement["position_xz"]
        if "yaw" in placement:
            actor["initial_yaw"] = placement["yaw"]
        if actor_id in frames_y:
            actor["coordinate_frame_y"] = frames_y[actor_id]
        actors.append(actor)
    core_request = {"request_id": request_id, "frames": frames, "actors": actors}
    validate_request(core_request)
    history = body.get("history")
    if history is not None:
        if not isinstance(history, dict) or set(history) not in ({"native_features"}, {"positions", "rotations"}):
            raise ValueError("history needs native_features or positions and rotations")
        if "native_features" in history:
            features = _array(history["native_features"], (len(ids), None, FEATURES), "history.native_features")
            if not 4 <= features.shape[1] <= 40 or features.shape[1] % 4:
                raise ValueError("native history must be 4–40 frames in four-frame tokens")
            history = {"native_features": features}
        else:
            positions = _array(history["positions"], (len(ids), None, JOINTS, 3), "history.positions")
            rotations = _rotations(history["rotations"], (len(ids), positions.shape[1], JOINTS, 3, 3), "history.rotations")
            if not 4 <= positions.shape[1] <= 40 or positions.shape[1] % 4:
                raise ValueError("canonical history must be 4–40 frames in four-frame tokens")
            history = {"positions": positions, "rotations": rotations}
    target = body.get("target")
    if kind == "transition":
        if history is None or not isinstance(target, dict) or set(target) != {"positions", "rotations"}:
            raise ValueError("transition requires history and a canonical full-body target")
        positions = _array(target["positions"], (2, frames, JOINTS, 3), "target.positions")
        rotations = _rotations(target["rotations"], (2, frames, JOINTS, 3, 3), "target.rotations")
        target = {"positions": positions, "rotations": rotations}
    elif target is not None:
        raise ValueError("target is supported only for transition")
    if kind == "continuation" and history is None:
        raise ValueError("continuation requires history")
    hand_target = body.get("hand_target")
    if hand_target is not None:
        if kind != "continuation" or len(ids) != 1 or history is None:
            raise ValueError("hand_target requires a one-actor Core continuation")
        if not isinstance(hand_target, dict) or set(hand_target) != {
                "source_job_id", "source_frame", "hand", "position_xyz", "frames"}:
            raise ValueError("Invalid hand_target fields")
        if not isinstance(hand_target["source_job_id"], str) or not 1 <= len(hand_target["source_job_id"]) <= 100:
            raise ValueError("Invalid hand target source_job_id")
        if hand_target["hand"] not in ("RightHand", "LeftHand"):
            raise ValueError("hand_target hand must be RightHand or LeftHand")
        source_frame = hand_target["source_frame"]
        if isinstance(source_frame, bool) or not isinstance(source_frame, Integral) or source_frame < 0:
            raise ValueError("Invalid hand target source_frame")
        goal_xyz = _array(hand_target["position_xyz"], (3,), "hand_target.position_xyz")
        if np.max(np.abs(goal_xyz)) > 25:
            raise ValueError("Hand target position is outside the scene bounds")
        target_frames = hand_target["frames"]
        if (not isinstance(target_frames, list) or not 1 <= len(target_frames) <= 24
                or any(isinstance(frame, bool) or not isinstance(frame, Integral)
                       or not 0 <= frame < frames for frame in target_frames)
                or len(set(target_frames)) != len(target_frames)):
            raise ValueError("hand_target frames must be distinct generated frame indices")
        hand_target = {**hand_target, "position_xyz": goal_xyz,
                       "frames": sorted(int(frame) for frame in target_frames)}
    if kind == "paired" and (history is not None or root_targets):
        raise ValueError("paired InterGen samples do not consume Core history or waypoints")
    pair_sequence_id = body.get("pair_sequence_id")
    source_start_frame = body.get("source_start_frame", 0)
    source_total_frames = body.get("source_total_frames", frames)
    if kind == "paired":
        if not isinstance(pair_sequence_id, str) or not 1 <= len(pair_sequence_id) <= 100:
            raise ValueError("paired stages require pair_sequence_id")
        if (isinstance(source_start_frame, bool) or not isinstance(source_start_frame, Integral)
                or source_start_frame < 0 or source_start_frame % HORIZON):
            raise ValueError("source_start_frame must be a nonnegative 40-frame boundary")
        if (isinstance(source_total_frames, bool) or not isinstance(source_total_frames, Integral)
                or not HORIZON <= source_total_frames <= MAX_PAIR_FRAMES
                or source_total_frames % HORIZON or source_start_frame + frames > source_total_frames):
            raise ValueError("Invalid paired source_total_frames or staged slice")
    elif any(key in body for key in ("pair_sequence_id", "source_start_frame", "source_total_frames")):
        raise ValueError("Pair sequence fields are supported only for paired stages")
    return {"request_id": request_id, "stage_kind": kind, "frames": frames,
            "prompt": prompt.strip(), "actor_ids": tuple(ids), "seed": int(seed),
            "core_request": core_request, "history": history, "target": target,
            "hand_target": hand_target,
            "pair_sequence_id": pair_sequence_id, "source_start_frame": int(source_start_frame),
            "source_total_frames": int(source_total_frames)}


def encode_chunk(arrays: dict[str, np.ndarray], *, request_id: str, index: int,
                 stage_kind: str, actor_ids: tuple[str, ...], extra: dict | None = None) -> bytes:
    buffer = io.BytesIO()
    metadata = {"request_id": request_id, "chunk_index": index,
                "start_frame": index * HORIZON, "frames": arrays["positions"].shape[1],
                "fps": FPS, "stage_kind": stage_kind, "actor_ids": list(actor_ids),
                "source": "InterGen research" if stage_kind == "paired" else "ARDY Core"}
    metadata.update(extra or {})
    np.savez_compressed(buffer, **arrays, metadata=np.array(json.dumps(metadata, allow_nan=False)))
    return buffer.getvalue()


class CoreStageAdapter:
    """One model call per stage; on_window publishes actual completed Core horizons."""

    def __init__(self, runtime: InteractionRuntime):
        self.runtime = runtime
        self.reference_provider: Callable[[str, int, str, np.ndarray], dict] | None = None

    def generate(self, job: dict, on_chunk: Callable[[int, bytes], None],
                 is_cancelled: Callable[[str], bool], deadline: float) -> dict:
        request = job["core_request"]
        ids = job["actor_ids"]
        history = job["history"]
        if history is not None:
            if "native_features" in history:
                native = history["native_features"]
            else:
                from motion_bridge import bridge_to_history
                native = bridge_to_history(self.runtime.model, history["positions"],
                                           history["rotations"], history_frames=history["positions"].shape[1])
            expected_frames = next(iter(history.values())).shape[1]
            if native.shape != (len(ids), expected_frames, FEATURES) or not np.isfinite(native).all():
                raise ValueError("Bridge returned incompatible Core history")
            for i, actor in enumerate(request["actors"]):
                actor["history"] = native[i]
        hook = None
        if job["stage_kind"] == "transition":
            target = job["target"]
            actor_index = {actor_id: i for i, actor_id in enumerate(ids)}

            def hook(*, model, actor, generated_offset, history_length, device, current_history):
                import torch
                from ardy.constraints import FullBodyConstraintSet

                if generated_offset + HORIZON < job["frames"]:
                    return []
                i = actor_index[actor["id"]]
                start = generated_offset + HORIZON - 4
                stop = start + 4
                positions = torch.from_numpy(target["positions"][i, start:stop]).to(device)
                rotations = torch.from_numpy(target["rotations"][i, start:stop]).to(device)
                indices = torch.arange(history_length + HORIZON - 4, history_length + HORIZON,
                                       dtype=torch.long)
                return [FullBodyConstraintSet(model.skeleton, indices, positions, rotations)]

        if job["hand_target"] is not None:
            if self.reference_provider is None:
                raise RuntimeError("Trusted Core reference store is unavailable")
            spec = job["hand_target"]
            reference = self.reference_provider(spec["source_job_id"], spec["source_frame"],
                                                spec["hand"], spec["position_xyz"])
            actor_id = ids[0]

            def hook(*, model, actor, generated_offset, history_length, device, current_history):
                import torch
                from ardy.constraints import EndEffectorConstraintSet

                if actor["id"] != actor_id:
                    return []
                frames = [frame for frame in spec["frames"]
                          if generated_offset <= frame < generated_offset + HORIZON]
                if not frames:
                    return []
                positions = torch.from_numpy(np.repeat(reference["positions"][None], len(frames), axis=0)).to(device)
                rotations = torch.from_numpy(np.repeat(reference["rotations"][None], len(frames), axis=0)).to(device)
                indices = torch.as_tensor([history_length + frame - generated_offset for frame in frames], dtype=torch.long)
                root = positions[:, model.skeleton.root_idx][:, (0, 2)]
                return [EndEffectorConstraintSet(model.skeleton, indices, positions, rotations,
                                                 root, joint_names=[spec["hand"], "Hips"])]

        runtime = self.runtime if hook is None else InteractionRuntime(
            self.runtime.model, device=self.runtime.device, condition_hook=hook)
        published = set()

        def window(index: int, selected_indices: list[int], data: dict[str, np.ndarray]):
            if selected_indices != list(range(len(ids))):
                raise RuntimeError("Core actor batch split; a synchronized pair chunk cannot be published")
            if index in published:
                raise RuntimeError("Duplicate Core window")
            published.add(index)
            payload = {"positions": data["positions"], "rotations": data["rotations"],
                       "native_features": data["motion"]}
            on_chunk(index, encode_chunk(payload, request_id=job["request_id"], index=index,
                                         stage_kind=job["stage_kind"], actor_ids=ids,
                                         extra={"seed": job["seed"]}))

        _, metadata = runtime.generate(request, is_cancelled=is_cancelled, deadline=deadline,
                                       on_window=window)
        if job["hand_target"] is not None:
            metadata["native_hand_target"] = {"joint": job["hand_target"]["hand"],
                                               "goal_xyz": job["hand_target"]["position_xyz"].tolist(),
                                               "frames": job["hand_target"]["frames"],
                                               "source_job_id": job["hand_target"]["source_job_id"],
                                               "reference_translation_m": reference["translation_m"],
                                               "constraint": "EndEffectorConstraintSet"}
        if published != set(range(job["frames"] // HORIZON)):
            raise RuntimeError("Core did not publish every requested window")
        return metadata


class InterGenStageAdapter:
    """Research-only published checkpoint adapter; one paired clip per job."""

    def __init__(self, model, normalizer, *, device="cuda", source_revision=None):
        self.model = model
        self.normalizer = normalizer
        self.device = device
        self.source_revision = source_revision
        self._sequences: dict[str, tuple[tuple, np.ndarray, np.ndarray, dict]] = {}

    def generate(self, job: dict, on_chunk: Callable[[int, bytes], None],
                 is_cancelled: Callable[[str], bool], deadline: float) -> dict:
        import torch
        from motion_bridge import retarget_intergen_pair

        if is_cancelled(job["request_id"]):
            raise GenerationCancelled("Realtime job cancelled")
        if time.monotonic() >= deadline:
            raise TimeoutError("Realtime job exceeded its time limit")
        key = job["pair_sequence_id"]
        expected = (job["seed"], job["prompt"], job["source_total_frames"], job["actor_ids"])
        started = time.perf_counter()
        cached = self._sequences.get(key)
        if cached is None:
            if job["source_start_frame"]:
                raise ValueError("Paired sequence must start at source frame zero")
            source_frames = job["source_total_frames"] * 3 // 2
            torch.manual_seed(job["seed"])
            torch.cuda.manual_seed_all(job["seed"])
            with torch.inference_mode():
                output = self.model.forward_test({
                    "motion_lens": torch.tensor([source_frames], dtype=torch.long, device=self.device),
                    "text": [job["prompt"]],
                })["output"]
            if tuple(output.shape) != (1, source_frames, 524):
                raise ValueError("InterGen returned an invalid paired motion shape")
            features = self.normalizer.backward(output[0].reshape(source_frames, 2, 262).cpu().numpy())
            joints = np.asarray(features[..., :66], dtype=np.float32).reshape(source_frames, 2, 22, 3)
            positions, rotations, bridge_metadata = retarget_intergen_pair(joints, source_fps=30, target_fps=20)
            if positions.shape != (2, job["source_total_frames"], JOINTS, 3) or rotations.shape != (2, job["source_total_frames"], JOINTS, 3, 3):
                raise ValueError("InterGen bridge returned incompatible canonical motion")
            if len(self._sequences) >= 4:
                self._sequences.pop(next(iter(self._sequences)))
            self._sequences[key] = (expected, positions, rotations, bridge_metadata)
        else:
            signature, positions, rotations, bridge_metadata = cached
            if signature != expected:
                raise ValueError("Paired sequence ID was reused with different prompt, seed or actors")
        if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
            raise ValueError("InterGen bridge returned nonfinite motion")
        for index in range(job["frames"] // HORIZON):
            if is_cancelled(job["request_id"]):
                raise GenerationCancelled("Realtime job cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError("Realtime job exceeded its time limit")
            start = job["source_start_frame"] + index * HORIZON
            payload = {"positions": positions[:, start:start + HORIZON],
                       "rotations": rotations[:, start:start + HORIZON]}
            on_chunk(index, encode_chunk(payload, request_id=job["request_id"], index=index,
                                         stage_kind="paired", actor_ids=job["actor_ids"],
                                         extra={"seed": job["seed"], "pair_sequence_id": key,
                                                "source_start_frame": start,
                                                "source_total_frames": job["source_total_frames"]}))
        return {"model": "InterGen", "research_only": True,
                "license": "CC BY-NC-SA 4.0", "source_revision": self.source_revision,
                "generation_seconds": time.perf_counter() - started,
                "bridge": bridge_metadata, "frames": job["frames"], "fps": FPS,
                "pair_sequence_id": key, "source_start_frame": job["source_start_frame"],
                "source_total_frames": job["source_total_frames"], "cache_reused": cached is not None}


@dataclass
class JobState:
    request_id: str
    stage_kind: str
    total_chunks: int
    actor_ids: tuple[str, ...] = ()
    submitted_at: float = field(default_factory=time.monotonic)
    status: str = "queued"
    chunks: dict[int, bytes] = field(default_factory=dict)
    metadata: dict | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    started_at: float | None = None
    finished_at: float | None = None

    def public(self) -> dict:
        return {"request_id": self.request_id, "stage_kind": self.stage_kind,
                "status": self.status, "produced_chunks": len(self.chunks),
                "total_chunks": self.total_chunks, "available_chunks": sorted(self.chunks),
                "submitted_at": self.submitted_at, "started_at": self.started_at,
                "finished_at": self.finished_at, "error": self.error,
                "metadata": self.metadata}


class JobManager:
    def __init__(self, core: CoreStageAdapter, *, pair_adapter=None, pair_enabled: bool = False,
                 max_queued: int = MAX_QUEUED, job_timeout: float = MAX_JOB_SECONDS):
        self.core = core
        self.pair_adapter = pair_adapter
        self.pair_enabled = pair_enabled
        self.job_timeout = job_timeout
        self.jobs: dict[str, JobState] = {}
        self._lock = threading.RLock()
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=max_queued)
        self.core.reference_provider = self._reference
        threading.Thread(target=self._worker, daemon=True, name="realtime-gpu-worker").start()

    def _reference(self, source_job_id: str, source_frame: int, hand: str,
                   target_xyz: np.ndarray) -> dict:
        """Use only a retained, completed Core pose as the coherent hand reference."""
        with self._lock:
            state = self.jobs.get(source_job_id)
            if state is None or state.status != "complete" or state.stage_kind == "paired":
                raise ValueError("Hand reference must name a completed Core job")
            if len(state.actor_ids) != 1:
                raise ValueError("Hand reference must contain one actor")
            index, within = divmod(source_frame, HORIZON)
            payload = state.chunks.get(index)
            if payload is None:
                raise ValueError("Hand reference frame is unavailable")
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                positions = np.array(data["positions"][0, within], dtype=np.float32, copy=True)
                rotations = np.array(data["rotations"][0, within], dtype=np.float32, copy=True)
        hand_index = int(self.core.runtime.model.skeleton.bone_index[hand])
        delta = target_xyz - positions[hand_index]
        if np.linalg.norm(delta) > .45 or abs(float(delta[1])) > .20:
            raise ValueError("Coherent Core hand reference is too far from the object target")
        positions += delta
        return {"positions": positions, "rotations": rotations,
                "translation_m": delta.astype(float).tolist()}

    def submit(self, body: dict) -> JobState:
        job = validate_job(body, pair_enabled=self.pair_enabled and self.pair_adapter is not None)
        with self._lock:
            if job["request_id"] in self.jobs:
                raise ValueError("request_id already exists")
            if self._queue.full():
                raise OverflowError("Realtime queue is full")
            finished = [k for k, v in self.jobs.items() if v.status in ("complete", "failed", "cancelled")]
            while len(self.jobs) >= MAX_RETAINED and finished:
                self.jobs.pop(finished.pop(0), None)
            if len(self.jobs) >= MAX_RETAINED:
                raise OverflowError("Too many retained jobs")
            state = JobState(job["request_id"], job["stage_kind"], job["frames"] // HORIZON,
                             actor_ids=job["actor_ids"])
            self.jobs[job["request_id"]] = state
            self._queue.put_nowait(job)
            return state

    def get(self, request_id: str) -> JobState | None:
        with self._lock:
            return self.jobs.get(request_id)

    def snapshot(self, request_id: str) -> dict | None:
        with self._lock:
            state = self.jobs.get(request_id)
            return None if state is None else state.public()

    def chunk(self, request_id: str, index: int) -> tuple[int, bytes | dict]:
        with self._lock:
            state = self.jobs.get(request_id)
            if state is None or not 0 <= index < state.total_chunks:
                return 404, {"error": "Chunk not found"}
            if index in state.chunks:
                return 200, state.chunks[index]
            if state.status in ("failed", "cancelled"):
                return 409, {"error": state.error or state.status}
            return 202, {"status": state.status, "chunk_index": index}

    def cancel(self, request_id: str) -> dict | None:
        with self._lock:
            state = self.jobs.get(request_id)
            if state is None:
                return None
            if state.status in ("queued", "running"):
                state.cancel_event.set()
                if state.status == "queued":
                    state.status = "cancelled"
                    state.finished_at = time.monotonic()
            return state.public()

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            state = self.get(job["request_id"])
            if state is None:
                continue
            with self._lock:
                if state.cancel_event.is_set():
                    self._queue.task_done()
                    continue
                state.status = "running"
                state.started_at = time.monotonic()
            deadline = state.started_at + self.job_timeout

            def cancelled(request_id: str) -> bool:
                return state.cancel_event.is_set()

            def publish(index: int, data: bytes) -> None:
                if state.cancel_event.is_set():
                    raise GenerationCancelled("Realtime job cancelled")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Realtime job exceeded its time limit")
                with self._lock:
                    state.chunks[index] = data

            try:
                adapter = self.pair_adapter if job["stage_kind"] == "paired" else self.core
                metadata = adapter.generate(job, publish, cancelled, deadline)
                with self._lock:
                    if state.cancel_event.is_set():
                        state.status = "cancelled"
                    else:
                        state.status = "complete"
                        state.metadata = metadata
            except (GenerationCancelled, TimeoutError) as exc:
                with self._lock:
                    state.status = "cancelled" if state.cancel_event.is_set() else "failed"
                    state.error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                with self._lock:
                    state.status = "failed"
                    state.error = f"{type(exc).__name__}: {exc}"
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            finally:
                with self._lock:
                    state.finished_at = time.monotonic()
                self._queue.task_done()


def make_handler(manager: JobManager, token: str):
    if not token:
        raise ValueError("A nonempty API token is required")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status: int, data: dict | bytes, content_type: str = "application/json"):
            payload = data if isinstance(data, bytes) else json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def authorized(self) -> bool:
            if not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.reply(401, {"error": "Unauthorized"})
                return False
            return True

        def path_parts(self):
            return [unquote(part) for part in urlsplit(self.path).path.split("/") if part]

        def do_GET(self):
            if not self.authorized():
                return
            parts = self.path_parts()
            if parts == ["health"]:
                self.reply(200, {"ready": True, "model": "ARDY-Core-RP-20FPS-Horizon40",
                                 "fps": FPS, "joints": JOINTS, "horizon": HORIZON,
                                 "pair_research_enabled": manager.pair_enabled,
                                 "queue_depth": manager._queue.qsize()})
            elif len(parts) == 4 and parts[:3] == ["v1", "realtime", "jobs"]:
                state = manager.snapshot(parts[3])
                self.reply(404, {"error": "Job not found"}) if state is None else self.reply(200, state)
            elif len(parts) == 6 and parts[:3] == ["v1", "realtime", "jobs"] and parts[4] == "chunks":
                try:
                    index = int(parts[5])
                except ValueError:
                    return self.reply(404, {"error": "Chunk not found"})
                status, value = manager.chunk(parts[3], index)
                self.reply(status, value, "application/octet-stream" if status == 200 else "application/json")
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if not self.authorized():
                return
            if self.path_parts() != ["v1", "realtime", "jobs"]:
                return self.reply(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY_BYTES:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(length))
                state = manager.submit(body)
                self.reply(202, {"request_id": state.request_id, "status": state.status,
                                 "status_url": f"/v1/realtime/jobs/{quote(state.request_id, safe='')}"})
            except OverflowError as exc:
                self.reply(429, {"error": str(exc)})
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                self.reply(400, {"error": str(exc)})

        def do_DELETE(self):
            if not self.authorized():
                return
            parts = self.path_parts()
            if len(parts) != 4 or parts[:3] != ["v1", "realtime", "jobs"]:
                return self.reply(404, {"error": "Not found"})
            state = manager.cancel(parts[3])
            self.reply(404, {"error": "Job not found"}) if state is None else self.reply(200, state)

    return Handler


def serve(manager: JobManager, token: str, *, host: str = "127.0.0.1", port: int = 8769):
    ThreadingHTTPServer((host, port), make_handler(manager, token)).serve_forever()
