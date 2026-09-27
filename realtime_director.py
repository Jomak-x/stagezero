"""Threadless buffered direction over immutable, synchronized actor clips.

The caller owns the inference executor. It claims one request at a time and
returns a complete canonical chunk; playback never waits for that executor.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import io
import json
import math
import threading
import time
import uuid
import zipfile
from typing import Any, Callable, Iterable

import numpy as np

from realtime_clip import CanonicalClip, FPS, MAX_CLIP_FRAMES


HORIZON = 40
PROJECT_VERSION = 1
MAX_ARCHIVE_BYTES = 500_000_000
MAX_CHUNKS = 500
MAX_PAIR_BOUNDARY_ROOT_STEP_M = .15
MAX_PAIR_BOUNDARY_MEAN_JOINT_STEP_M = .15
MAX_PAIR_BOUNDARY_MAX_JOINT_STEP_M = .4
KINDS = frozenset({"approach", "paired_action", "exit", "action", "release", "transition"})


class UnsupportedAction(ValueError):
    """An unavailable source or mode was requested explicitly."""


def _json_copy(value: Any) -> Any:
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("stage parameters must contain finite JSON values") from exc
    if len(encoded) > 1_000_000:
        raise ValueError("stage parameters are too large")
    return json.loads(encoded)


@dataclass(frozen=True)
class StageSpec:
    prompt: str
    kind: str = "action"
    frames: int = HORIZON
    source: str = "ardy_core"
    actor_prompts: dict[str, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not 1 <= len(self.prompt.strip()) <= 500:
            raise ValueError("stage prompt must contain 1–500 characters")
        if self.kind not in KINDS:
            raise ValueError(f"unsupported stage kind: {self.kind}")
        if type(self.frames) is not int or not HORIZON <= self.frames <= MAX_CLIP_FRAMES or self.frames % HORIZON:
            raise ValueError("stage frames must be a multiple of 40")
        if not isinstance(self.source, str) or not 1 <= len(self.source) <= 100:
            raise ValueError("stage source is required")
        prompts = self.actor_prompts
        if prompts is not None:
            if not isinstance(prompts, dict) or not prompts or any(
                not isinstance(k, str) or not isinstance(v, str) or not 1 <= len(v.strip()) <= 500
                for k, v in prompts.items()
            ):
                raise ValueError("actor prompts must be nonempty strings keyed by actor ID")
            prompts = dict(prompts)
        object.__setattr__(self, "prompt", self.prompt.strip())
        object.__setattr__(self, "actor_prompts", prompts)
        object.__setattr__(self, "metadata", _json_copy(self.metadata))


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    epoch: int
    stage_id: str
    start_frame: int
    frames: int
    stage_kind: str
    prompt: str
    source: str
    actor_ids: tuple[str, ...]
    actor_prompts: dict[str, str] | None
    metadata: dict[str, Any]
    history: CanonicalClip | None


@dataclass
class _QueuedStage:
    id: str
    spec: StageSpec
    remaining: int


class RealtimeDirector:
    """A committed timeline, bounded lookahead queue, and monotonic playhead.

    Source policy is enforced on submission. ``intergen`` remains a research
    source and cannot be scheduled by a production director. The scheduler
    never invents frames on underrun or inference failure.
    """

    def __init__(self, actor_ids: Iterable[str], *, mode: str = "production",
                 target_buffer_frames: int = 80, max_buffer_frames: int = 160,
                 clock: Callable[[], float] = time.monotonic,
                 project_metadata: dict[str, Any] | None = None):
        ids = tuple(actor_ids)
        if not 1 <= len(ids) <= 2 or any(not isinstance(x, str) or not 1 <= len(x) <= 64 for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("director requires one or two unique stable actor IDs")
        if mode not in ("production", "research"):
            raise ValueError("mode must be production or research")
        if (type(target_buffer_frames) is not int or type(max_buffer_frames) is not int
                or not HORIZON <= target_buffer_frames <= max_buffer_frames
                or max_buffer_frames > MAX_CLIP_FRAMES):
            raise ValueError("invalid buffer limits")
        self.actor_ids = ids
        self.mode = mode
        self.project_metadata = _json_copy({} if project_metadata is None else project_metadata)
        if not isinstance(self.project_metadata, dict):
            raise ValueError("project_metadata must be a JSON object")
        self.target_buffer_frames = target_buffer_frames
        self.max_buffer_frames = max_buffer_frames
        self._clock = clock
        self._lock = threading.RLock()
        self._chunks: list[CanonicalClip] = []
        self._segments: list[dict[str, Any]] = []
        self._frames = 0
        self._queue: deque[_QueuedStage] = deque()
        self._inflight: GenerationRequest | None = None
        self._cancelled: list[str] = []
        self._epoch = 0
        self._failure: str | None = None
        self._contact_open = False
        self.playhead = 0
        self.playing = False
        self._last_tick: float | None = None
        self._fraction = 0.0
        self._revision = 0

    @property
    def total_frames(self) -> int:
        with self._lock:
            return self._frames

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def segments(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(_json_copy(segment) for segment in self._segments)

    def _validate_stage(self, stage: StageSpec) -> None:
        if not isinstance(stage, StageSpec):
            raise ValueError("queue entries must be StageSpec")
        if stage.source not in ("ardy_core", "intergen"):
            raise UnsupportedAction(f"unsupported motion source: {stage.source}")
        if stage.source == "intergen" and self.mode != "research":
            raise UnsupportedAction("InterGen paired motion is available only in explicit research mode")
        if stage.kind == "paired_action" and stage.source != "intergen":
            raise UnsupportedAction("paired contact requires the research InterGen source")
        if stage.source == "intergen" and stage.kind not in ("paired_action", "exit", "release"):
            raise UnsupportedAction("InterGen is supported only for paired contact and its exit")
        if stage.actor_prompts is not None and set(stage.actor_prompts) != set(self.actor_ids):
            raise ValueError("actor prompts must cover the stable actor IDs")
        if stage.kind in ("paired_action", "exit", "release") and len(self.actor_ids) != 2:
            raise ValueError("paired and release stages require two actors")

    def queue_sequence(self, stages: Iterable[StageSpec]) -> tuple[str, ...]:
        specs = tuple(stages)
        if not specs:
            raise ValueError("sequence requires at least one stage")
        with self._lock:
            # Validate the entire sequence before changing the queue.
            pending = sum(item.remaining for item in self._queue)
            for stage in specs:
                self._validate_stage(stage)
                pending += stage.frames
                if self._frames + pending > MAX_CLIP_FRAMES:
                    raise ValueError("timeline exceeds the 15000-frame limit")
            for index, stage in enumerate(specs):
                if stage.kind in ("exit", "release") and (index == 0 or specs[index - 1].kind != "paired_action"):
                    raise UnsupportedAction("paired exit must follow its contact stage in the same sequence")
                if stage.kind != "paired_action":
                    continue
                pair_id = stage.metadata.get("pair_sequence_id")
                if not isinstance(pair_id, str) or not pair_id.strip():
                    raise UnsupportedAction("paired contact requires a cached pair_sequence_id")
                if index + 1 >= len(specs):
                    raise UnsupportedAction("paired contact requires a contiguous exit from the same sequence")
                following = specs[index + 1]
                if (following.kind not in ("exit", "release") or following.source != "intergen" or
                        following.metadata.get("pair_sequence_id") != pair_id):
                    raise UnsupportedAction("paired contact requires a contiguous InterGen exit from the same sequence")
                total = stage.frames + following.frames
                if (total > 120 or stage.prompt != following.prompt or
                        stage.metadata.get("seed") != following.metadata.get("seed") or
                        type(stage.metadata.get("seed")) is not int or
                        stage.metadata.get("source_start_frame") != 0 or
                        following.metadata.get("source_start_frame") != stage.frames or
                        stage.metadata.get("source_total_frames") != total or
                        following.metadata.get("source_total_frames") != total):
                    raise UnsupportedAction("paired action and exit must use one 120-frame-or-shorter cached sample, prompt, and seed")
                release = following.metadata.get("release_window")
                if (not isinstance(release, list) or len(release) != 2 or
                        any(type(x) is not int for x in release) or
                        not stage.frames <= release[0] < release[1] <= total):
                    raise UnsupportedAction("paired exit requires a verified release window")
            ids = tuple(uuid.uuid4().hex for _ in specs)
            self._queue.extend(_QueuedStage(stage_id, spec, spec.frames)
                               for stage_id, spec in zip(ids, specs))
            self._failure = None
            return ids

    def submit_instruction(self, prompt: str, *, kind: str = "action", frames: int = HORIZON,
                           source: str = "ardy_core", actor_prompts: dict[str, str] | None = None,
                           metadata: dict[str, Any] | None = None) -> str:
        spec = StageSpec(prompt, kind, frames, source, actor_prompts,
                         {} if metadata is None else metadata)
        return self.queue_sequence((spec,))[0]

    def _buffer_frames(self) -> int:
        return max(0, self._frames - self.playhead)

    def _history(self) -> CanonicalClip | None:
        if not self._chunks:
            return None
        needed = HORIZON
        pieces = []
        for chunk in reversed(self._chunks):
            start = max(0, chunk.frames - needed)
            pieces.append(chunk.slice_frames(start, chunk.frames))
            needed -= chunk.frames - start
            if needed <= 0:
                break
        pieces.reverse()
        if len(pieces) == 1:
            return pieces[0]
        p = np.concatenate([piece.positions for piece in pieces], axis=1)
        r = np.concatenate([piece.rotations for piece in pieces], axis=1)
        native = None if any(piece.native_features is None for piece in pieces) else np.concatenate(
            [piece.native_features for piece in pieces], axis=1)
        return CanonicalClip(p, r, FPS, self.actor_ids, "committed_history",
                             {"end_frame": self._frames}, native)

    def claim_request(self) -> GenerationRequest | None:
        """Return one bounded horizon; never call inference while holding a UI lock."""
        with self._lock:
            if self._inflight is not None or self._failure is not None or not self._queue:
                return None
            if self._buffer_frames() >= self.target_buffer_frames:
                return None
            if self._buffer_frames() + HORIZON > self.max_buffer_frames:
                return None
            stage = self._queue[0]
            metadata = _json_copy(stage.spec.metadata)
            if stage.spec.source == "intergen" and "source_start_frame" in metadata:
                metadata["source_start_frame"] += stage.spec.frames - stage.remaining
            request = GenerationRequest(
                request_id=uuid.uuid4().hex, epoch=self._epoch, stage_id=stage.id,
                start_frame=self._frames, frames=HORIZON, stage_kind=stage.spec.kind,
                prompt=stage.spec.prompt, source=stage.spec.source,
                actor_ids=self.actor_ids,
                actor_prompts=None if stage.spec.actor_prompts is None else dict(stage.spec.actor_prompts),
                metadata=metadata, history=self._history(),
            )
            self._inflight = request
            return request

    def _is_current(self, request_id: str) -> bool:
        return self._inflight is not None and self._inflight.request_id == request_id and self._inflight.epoch == self._epoch

    def complete(self, request_id: str, clip: CanonicalClip) -> bool:
        """Append only a matching completion. Stale or cancelled work is ignored."""
        with self._lock:
            if not self._is_current(request_id):
                return False
            request = self._inflight
            if not isinstance(clip, CanonicalClip) or clip.actor_ids != self.actor_ids or clip.fps != FPS or clip.frames != request.frames:
                self._failure = "Generator returned an incompatible synchronized clip"
                self._inflight = None
                raise ValueError(self._failure)
            if clip.source != request.source:
                self._failure = "Generator returned the wrong motion source"
                self._inflight = None
                raise ValueError(self._failure)
            if request.source == "intergen" and any(
                clip.metadata.get(key) != request.metadata.get(key)
                for key in ("pair_sequence_id", "source_start_frame", "source_total_frames", "seed")
            ):
                self._failure = "Generator returned a different paired sequence or source span"
                self._inflight = None
                raise ValueError(self._failure)
            if request.source == "intergen" and self._chunks:
                before = self._chunks[-1].positions[:, -1]
                after = clip.positions[:, 0]
                root_step = np.linalg.norm(after[:, 0] - before[:, 0], axis=-1).max()
                joint_steps = np.linalg.norm(after - before, axis=-1)
                mean_joint_step = joint_steps.mean()
                max_joint_step = joint_steps.max()
                if (root_step > MAX_PAIR_BOUNDARY_ROOT_STEP_M or
                        mean_joint_step > MAX_PAIR_BOUNDARY_MEAN_JOINT_STEP_M or
                        max_joint_step > MAX_PAIR_BOUNDARY_MAX_JOINT_STEP_M):
                    self._failure = ("Paired motion does not join the committed pose "
                                     f"(root {root_step:.2f} m, mean joint {mean_joint_step:.2f} m, "
                                     f"max joint {max_joint_step:.2f} m)")
                    self._inflight = None
                    raise ValueError(self._failure)
            if self._frames + clip.frames > MAX_CLIP_FRAMES:
                self._failure = "Timeline frame limit reached"
                self._inflight = None
                raise ValueError(self._failure)
            stage = self._queue[0]
            if stage.id != request.stage_id or self._frames != request.start_frame:
                self._failure = "Timeline changed while generation was in flight"
                self._inflight = None
                raise ValueError(self._failure)
            start = self._frames
            self._chunks.append(clip)
            self._frames += clip.frames
            stage.remaining -= clip.frames
            done = stage.remaining == 0
            self._segments.append({"start": start, "end": self._frames,
                                   "stage_id": stage.id, "kind": stage.spec.kind,
                                   "prompt": stage.spec.prompt, "source": clip.source,
                                   "request_id": request_id, "stage_complete": done,
                                   "metadata": _json_copy(stage.spec.metadata),
                                   "clip_metadata": _json_copy(clip.metadata)})
            if stage.spec.kind == "paired_action":
                self._contact_open = True
            if stage.spec.kind in ("exit", "release") and done:
                self._contact_open = False
            if done:
                self._queue.popleft()
            self._inflight = None
            self._revision += 1
            return True

    def fail(self, request_id: str, error: Exception | str) -> bool:
        with self._lock:
            if not self._is_current(request_id):
                return False
            message = str(error).strip()[:300]
            self._failure = message or type(error).__name__
            self._inflight = None
            return True

    def retry(self) -> bool:
        with self._lock:
            if self._failure is None:
                return False
            self._failure = None
            return bool(self._queue)

    def interrupt(self, prompt: str, *, kind: str = "action", frames: int = HORIZON,
                  source: str = "ardy_core", actor_prompts: dict[str, str] | None = None,
                  metadata: dict[str, Any] | None = None,
                  release_instruction: str = "Release contact and step apart safely.") -> str:
        """Replace uncommitted intent; preserve committed contact until release.

        If a paired chunk was committed, an existing exit stage is retained or
        a release stage is queued before the new action. No pose is teleported.
        """
        next_stage = StageSpec(prompt, kind, frames, source, actor_prompts,
                               {} if metadata is None else metadata)
        with self._lock:
            self._validate_stage(next_stage)
            retained: list[_QueuedStage] = []
            if self._contact_open:
                # A paired stage can span several horizons. Keep every
                # uncommitted horizon in that SAME cached sequence before its
                # exit; skipping straight to exit would jump over contact.
                pair_remainder = self._queue[0] if self._queue and self._queue[0].spec.kind == "paired_action" else None
                if pair_remainder is not None:
                    retained.append(pair_remainder)
                exit_stage = next((item for item in self._queue if item.spec.kind in ("exit", "release")), None)
                if exit_stage is None:
                    raise UnsupportedAction("active paired contact has no verified exit from the same sequence")
                retained.append(exit_stage)
            if self._frames + sum(item.remaining for item in retained) + next_stage.frames > MAX_CLIP_FRAMES:
                raise ValueError("timeline exceeds the 15000-frame limit")
            keep_request = self._inflight is not None and any(
                self._inflight.stage_id == item.id for item in retained)
            if self._inflight is not None and not keep_request:
                self._cancelled.append(self._inflight.request_id)
                self._epoch += 1
                self._inflight = None
            self._queue = deque(retained)
            stage_id = uuid.uuid4().hex
            self._queue.append(_QueuedStage(stage_id, next_stage, next_stage.frames))
            self._failure = None
            return stage_id

    def cancel_pending(self) -> None:
        """Discard uncommitted work, retaining the exact committed timeline."""
        with self._lock:
            if self._contact_open:
                raise UnsupportedAction("active research contact must complete its verified exit")
            if self._inflight is not None:
                self._cancelled.append(self._inflight.request_id)
            self._epoch += 1
            self._inflight = None
            self._queue.clear()
            self._failure = None

    def take_cancellations(self) -> tuple[str, ...]:
        with self._lock:
            result = tuple(self._cancelled)
            self._cancelled.clear()
            return result

    def play(self, *, now: float | None = None) -> None:
        with self._lock:
            self.playing = True
            self._last_tick = self._clock() if now is None else float(now)
            self._fraction = 0.0

    def pause(self) -> None:
        with self._lock:
            self.playing = False
            self._last_tick = None
            self._fraction = 0.0

    def seek(self, frame: int) -> None:
        with self._lock:
            if type(frame) is not int or not 0 <= frame < self._frames:
                raise ValueError("seek frame is outside committed motion")
            self.playhead = frame
            self._last_tick = self._clock()
            self._fraction = 0.0

    def tick(self, *, now: float | None = None) -> dict[str, Any]:
        """Advance from a clock without waiting for model work or inventing poses."""
        with self._lock:
            current = self._clock() if now is None else float(now)
            if not math.isfinite(current):
                raise ValueError("clock must be finite")
            if self.playing:
                if self._last_tick is None:
                    self._last_tick = current
                elapsed = max(0.0, current - self._last_tick)
                self._last_tick = current
                if self._frames:
                    advance = self._fraction + elapsed * FPS
                    whole = int(advance + 1e-9)
                    maximum = self._frames - 1
                    if self.playhead + whole >= maximum:
                        self.playhead = maximum
                        self._fraction = 0.0
                    else:
                        self.playhead += whole
                        self._fraction = advance - whole
                else:
                    self._fraction = 0.0
                if self._frames and self.playhead == self._frames - 1 and not self._queue and self._inflight is None and self._failure is None:
                    self.playing = False
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self._failure is not None:
                phase = "generation_failed"
                status = f"Generation failed: {self._failure}. Committed frames are preserved; retry or interrupt."
            elif self.playing and self._buffer_frames() <= 1 and (self._queue or self._inflight):
                phase = "buffering"
                status = "Waiting for generated frames; holding the last committed pose."
            elif self.playing and self._frames:
                phase = "playing"
                status = "Playing generated motion."
            elif self._inflight is not None or self._queue:
                phase = "generating" if self._inflight else "queued"
                status = "Generating motion." if self._inflight else "Motion queued."
            elif self._frames:
                phase = "complete"
                status = "Generated timeline complete."
            else:
                phase = "empty"
                status = "No generated motion."
            return {"phase": phase, "status": status, "frame": self.playhead,
                    "total_frames": self._frames, "buffer_frames": self._buffer_frames(),
                    "playing": self.playing, "generated": bool(self._frames),
                    "inflight_request_id": None if self._inflight is None else self._inflight.request_id,
                    "queued_stages": len(self._queue), "contact_open": self._contact_open,
                    "failure": self._failure, "revision": self._revision,
                    "mode": self.mode, "actor_ids": self.actor_ids}

    def frame_pose(self, frame: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            index = self.playhead if frame is None else frame
            if type(index) is not int or not 0 <= index < self._frames:
                raise ValueError("frame is outside committed motion")
            for chunk in self._chunks:
                if index < chunk.frames:
                    return chunk.positions[:, index].copy(), chunk.rotations[:, index].copy()
                index -= chunk.frames
            raise AssertionError("timeline index is inconsistent")

    def timeline_clip(self) -> CanonicalClip | None:
        with self._lock:
            if not self._chunks:
                return None
            if len(self._chunks) == 1:
                return self._chunks[0]
            native = None if any(c.native_features is None for c in self._chunks) else np.concatenate(
                [c.native_features for c in self._chunks], axis=1)
            return CanonicalClip(np.concatenate([c.positions for c in self._chunks], axis=1),
                                 np.concatenate([c.rotations for c in self._chunks], axis=1),
                                 FPS, self.actor_ids, "timeline", {"segments": self.segments}, native)

    def save_project(self) -> bytes:
        """Save exact committed frames and provenance; no prompt regeneration."""
        with self._lock:
            if not self._chunks:
                raise ValueError("there are no generated frames to save")
            if len(self._chunks) > MAX_CHUNKS:
                raise ValueError("project has too many motion chunks")
            project_metadata = _json_copy(self.project_metadata)
            if not isinstance(project_metadata, dict):
                raise ValueError("project_metadata must be a JSON object")
            data: dict[str, np.ndarray] = {}
            for i, chunk in enumerate(self._chunks):
                data[f"c{i}_positions"] = chunk.positions
                data[f"c{i}_rotations"] = chunk.rotations
                if chunk.native_features is not None:
                    data[f"c{i}_native_features"] = chunk.native_features
            manifest = {
                "version": PROJECT_VERSION, "schema": "Core27-20fps-shared-timeline",
                "mode": self.mode, "fps": FPS, "actor_ids": self.actor_ids,
                "target_buffer_frames": self.target_buffer_frames,
                "max_buffer_frames": self.max_buffer_frames,
                "playhead": self.playhead, "segments": self._segments,
                "chunks": [{"frames": c.frames, "source": c.source,
                            "metadata": c.metadata,
                            "native_features": c.native_features is not None}
                           for c in self._chunks],
                "queue": [{"id": q.id, "spec": asdict(q.spec), "remaining": q.remaining}
                          for q in self._queue],
                "contact_open": self._contact_open, "failure": self._failure,
                "project_metadata": project_metadata,
            }
            data["manifest"] = np.array(json.dumps(manifest, allow_nan=False))
            output = io.BytesIO()
            np.savez_compressed(output, **data)
            return output.getvalue()

    @classmethod
    def load_project(cls, content: bytes, *, clock: Callable[[], float] = time.monotonic) -> "RealtimeDirector":
        """Validate an NPZ archive without pickle or path extraction."""
        if not isinstance(content, bytes) or len(content) > MAX_ARCHIVE_BYTES:
            raise ValueError("invalid or oversized project archive")
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()
            if (len(entries) > MAX_CHUNKS * 3 + 1 or
                    sum(item.file_size for item in entries) > MAX_ARCHIVE_BYTES or
                    len({item.filename for item in entries}) != len(entries)):
                raise ValueError("invalid project archive entries")
        with np.load(io.BytesIO(content), allow_pickle=False) as data:
            if "manifest" not in data:
                raise ValueError("missing project manifest")
            raw = str(data["manifest"])
            if len(raw) > 2_000_000:
                raise ValueError("project manifest is too large")
            manifest = json.loads(raw)
            if manifest.get("version") != PROJECT_VERSION or manifest.get("schema") != "Core27-20fps-shared-timeline" or manifest.get("fps") != FPS:
                raise ValueError("unsupported project version or motion schema")
            director = cls(manifest["actor_ids"], mode=manifest["mode"],
                           target_buffer_frames=manifest["target_buffer_frames"],
                           max_buffer_frames=manifest["max_buffer_frames"], clock=clock)
            project_metadata = _json_copy(manifest.get("project_metadata", {}))
            if not isinstance(project_metadata, dict):
                raise ValueError("saved project_metadata must be a JSON object")
            director.project_metadata = project_metadata
            items = manifest["chunks"]
            if not isinstance(items, list) or not 1 <= len(items) <= MAX_CHUNKS:
                raise ValueError("invalid project chunk count")
            for i, item in enumerate(items):
                native = data[f"c{i}_native_features"] if item["native_features"] else None
                clip = CanonicalClip(data[f"c{i}_positions"], data[f"c{i}_rotations"],
                                     FPS, director.actor_ids, item["source"], item["metadata"], native)
                if clip.frames != item["frames"]:
                    raise ValueError("project chunk frame count disagrees with manifest")
                director._chunks.append(clip)
                director._frames += clip.frames
                if director._frames > MAX_CLIP_FRAMES:
                    raise ValueError("project timeline is too long")
            segments = manifest["segments"]
            if not isinstance(segments, list) or len(segments) != len(items):
                raise ValueError("invalid project provenance")
            end = 0
            contact_open = False
            for item, clip in zip(segments, director._chunks):
                if (not isinstance(item, dict) or type(item.get("start")) is not int or
                        type(item.get("end")) is not int or item["start"] != end or
                        item["end"] != end + clip.frames or item.get("source") != clip.source or
                        item.get("clip_metadata") != _json_copy(clip.metadata)):
                    raise ValueError("invalid project segment timeline")
                if item.get("kind") == "paired_action":
                    contact_open = True
                if item.get("kind") in ("exit", "release") and item.get("stage_complete") is True:
                    contact_open = False
                end = item["end"]
            director._segments = _json_copy(segments)
            frame = manifest["playhead"]
            if type(frame) is not int or not 0 <= frame < director._frames:
                raise ValueError("invalid saved playhead")
            director.playhead = frame
            for item in manifest["queue"]:
                stage = StageSpec(**item["spec"])
                director._validate_stage(stage)
                remaining = item["remaining"]
                if type(remaining) is not int or not HORIZON <= remaining <= stage.frames or remaining % HORIZON:
                    raise ValueError("invalid queued stage progress")
                director._queue.append(_QueuedStage(item["id"], stage, remaining))
            if director._frames + sum(item.remaining for item in director._queue) > MAX_CLIP_FRAMES:
                raise ValueError("saved plan exceeds the timeline frame limit")
            if type(manifest.get("contact_open")) is not bool or manifest["contact_open"] != contact_open:
                raise ValueError("saved contact state disagrees with committed motion")
            if contact_open:
                last_pair = next((item for item in reversed(segments)
                                  if item.get("kind") == "paired_action"), None)
                pair_id = None if last_pair is None else last_pair["metadata"].get("pair_sequence_id")
                exit_stage = next((item for item in director._queue
                                   if item.spec.kind in ("exit", "release")), None)
                if (not isinstance(pair_id, str) or exit_stage is None or
                        exit_stage.spec.source != "intergen" or
                        exit_stage.spec.metadata.get("pair_sequence_id") != pair_id or
                        any(item.spec.kind == "paired_action" and
                            item.spec.metadata.get("pair_sequence_id") != pair_id
                            for item in director._queue)):
                    raise ValueError("saved contact has no matching cached pair exit")
            director._contact_open = contact_open
            failure = manifest.get("failure")
            if failure is not None and not isinstance(failure, str):
                raise ValueError("invalid saved failure")
            director._failure = failure
            director._revision = 1
            return director
