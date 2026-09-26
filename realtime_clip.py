"""Validated, immutable motion on one shared Core27 / 20 fps timeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

import numpy as np


FPS = 20
JOINTS = 27
FEATURES = 330
MAX_ACTORS = 2
MAX_CLIP_FRAMES = 15_000


class FrozenJSONDict(dict):
    """JSON-compatible read-only mapping used for clip provenance."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("clip metadata is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable
    __ior__ = _immutable


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenJSONDict({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _json_copy(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("clip metadata must be a dictionary")
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("clip metadata must contain finite JSON values") from exc
    if len(encoded) > 1_000_000:
        raise ValueError("clip metadata is too large")
    return json.loads(encoded)


def _array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        result = np.array(value, dtype=np.float32, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have shape {shape} and finite values")
    # A bytes-backed array cannot be made writeable again with setflags(True).
    # This matters because committed chunks are shared with async workers.
    return np.frombuffer(result.tobytes(), dtype=np.float32).reshape(shape)


@dataclass(frozen=True)
class CanonicalClip:
    """One or two actors in the same world space and frame clock.

    Arrays are copied on construction and then read-only. ``native_features``
    is optional because imported paired motion requires model-specific inverse
    mapping before it can be used as ARDY history.
    """

    positions: np.ndarray
    rotations: np.ndarray
    fps: int
    actor_ids: tuple[str, ...]
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    native_features: np.ndarray | None = None

    def __post_init__(self) -> None:
        ids = tuple(self.actor_ids)
        if not 1 <= len(ids) <= MAX_ACTORS or any(
            not isinstance(actor_id, str) or not 1 <= len(actor_id) <= 64 for actor_id in ids
        ) or len(set(ids)) != len(ids):
            raise ValueError("actor_ids must contain one or two unique stable IDs")
        if type(self.fps) is not int or self.fps != FPS:
            raise ValueError("canonical clips must use 20 fps")
        if not isinstance(self.source, str) or not 1 <= len(self.source) <= 100:
            raise ValueError("clip source is required")
        raw = np.asarray(self.positions)
        if raw.ndim != 4 or raw.shape[0] != len(ids) or raw.shape[2:] != (JOINTS, 3):
            raise ValueError("positions must have shape [actors, frames, 27, 3]")
        frames = raw.shape[1]
        if not 1 <= frames <= MAX_CLIP_FRAMES:
            raise ValueError("clip must contain 1–15000 frames")
        positions = _array(self.positions, (len(ids), frames, JOINTS, 3), "positions")
        rotations = _array(self.rotations, (len(ids), frames, JOINTS, 3, 3), "rotations")
        orthogonal = rotations @ np.swapaxes(rotations, -1, -2)
        if not np.allclose(orthogonal, np.eye(3), atol=.03) or not np.allclose(
            np.linalg.det(rotations), 1, atol=.03
        ):
            raise ValueError("rotations must be proper orthonormal matrices")
        features = None if self.native_features is None else _array(
            self.native_features, (len(ids), frames, FEATURES), "native_features"
        )
        object.__setattr__(self, "actor_ids", ids)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "rotations", rotations)
        object.__setattr__(self, "native_features", features)
        object.__setattr__(self, "metadata", _freeze_json(_json_copy(self.metadata)))

    @property
    def frames(self) -> int:
        return self.positions.shape[1]

    @classmethod
    def from_arrays(cls, positions: np.ndarray, rotations: np.ndarray, *,
                    actor_ids: tuple[str, ...], source: str, fps: int = FPS,
                    metadata: dict[str, Any] | None = None,
                    native_features: np.ndarray | None = None) -> "CanonicalClip":
        return cls(positions, rotations, fps, actor_ids, source,
                   {} if metadata is None else metadata, native_features)

    def slice_frames(self, start: int, stop: int) -> "CanonicalClip":
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= self.frames:
            raise ValueError("invalid clip slice")
        native = None if self.native_features is None else self.native_features[:, start:stop]
        return CanonicalClip(self.positions[:, start:stop], self.rotations[:, start:stop],
                             self.fps, self.actor_ids, self.source, self.metadata, native)
