"""Deterministic, bounded particle effects for mock scene playback.

All positions and sizes are metres, with +Y up. ``evaluate_effects`` is pure:
the same effect specification and playhead time produce identical arrays even
after seeking backwards. Smoke uses dim, soft points to suggest translucency;
Viser's point clouds do not offer per-particle alpha. Cinematic kinds use a
compact descriptor rendered as shader volumes in the custom studio client.
"""

from __future__ import annotations

import math
import re
from numbers import Integral, Real

import numpy as np


MAX_EFFECTS = 8
EFFECT_KINDS = {
    "explosion": {"size": [4.0, 3.2, 4.0], "color": [255, 108, 24], "position": [0.0, 1.6, -2.5]},
    "energy_burst": {"size": [3.8, 3.8, 3.8], "color": [76, 192, 255], "position": [0.0, 1.9, -2.5]},
    "rain": {"size": [5.0, 3.2, 5.0], "color": [109, 183, 242], "position": [0.0, 1.6, 0.0]},
    "snow": {"size": [5.0, 3.2, 5.0], "color": [229, 243, 255], "position": [0.0, 1.6, 0.0]},
    "fireflies": {"size": [3.5, 1.8, 3.5], "color": [244, 224, 100], "position": [0.0, 1.3, 0.0]},
    "sparks": {"size": [2.0, 2.3, 2.0], "color": [255, 143, 49], "position": [0.0, 0.3, 0.0]},
    "smoke": {"size": [1.6, 2.7, 1.6], "color": [166, 173, 187], "position": [0.0, 1.2, 0.0]},
    "portal": {"size": [2.0, 2.8, 0.4], "color": [124, 100, 255], "position": [0.0, 1.5, 0.0]},
}
_FIELDS = frozenset(("id", "kind", "position", "size", "color", "intensity", "seed"))
_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


def _number(value: object, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{label} must be finite and between {minimum} and {maximum}")
    return number


def _vector(value: object, label: str, minimum: float, maximum: float) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain three coordinates")
    return [_number(v, f"{label}[{i}]", minimum, maximum) for i, v in enumerate(value)]


def validate_effects(value: object) -> list[dict]:
    """Copy effect JSON into a strict canonical schema; reject executable fields."""
    if not isinstance(value, list) or len(value) > MAX_EFFECTS:
        raise ValueError(f"effects must be a list of at most {MAX_EFFECTS} items")
    result: list[dict] = []
    seen: set[str] = set()
    for i, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != _FIELDS:
            raise ValueError(f"effects[{i}] has missing or unknown fields")
        identifier = item["id"]
        if not isinstance(identifier, str) or not _ID_PATTERN.fullmatch(identifier) or identifier in seen:
            raise ValueError(f"effects[{i}].id is invalid or duplicated")
        seen.add(identifier)
        kind = item["kind"]
        if not isinstance(kind, str) or kind not in EFFECT_KINDS:
            raise ValueError(f"effects[{i}].kind is unsupported")
        position = _vector(item["position"], f"effects[{i}].position", -20.0, 20.0)
        size = _vector(item["size"], f"effects[{i}].size", 0.1, 8.0)
        color = item["color"]
        if not isinstance(color, (list, tuple)) or len(color) != 3 or any(
            isinstance(c, bool) or not isinstance(c, Integral) or not 0 <= c <= 255 for c in color
        ):
            raise ValueError(f"effects[{i}].color must contain three RGB bytes")
        intensity = _number(item["intensity"], f"effects[{i}].intensity", 0.0, 1.0)
        seed = item["seed"]
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 2**31 - 1:
            raise ValueError(f"effects[{i}].seed must be a non-negative 31-bit integer")
        result.append({"id": identifier, "kind": kind, "position": position,
                       "size": size, "color": [int(c) for c in color],
                       "intensity": intensity, "seed": int(seed)})
    return result


def make_effect(kind: str, index: int, position: list[float] | tuple[float, float, float] | None = None) -> dict:
    """Create one distinctive effect with a stable ID and seed."""
    if not isinstance(kind, str) or kind not in EFFECT_KINDS:
        raise ValueError(f"unsupported effect kind: {kind!r}")
    if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
        raise ValueError("index must be a non-negative integer")
    template = EFFECT_KINDS[kind]
    return validate_effects([{"id": f"{kind}-{index}", "kind": kind,
                             "position": template["position"] if position is None else position,
                             "size": template["size"], "color": template["color"],
                             "intensity": 0.75, "seed": int(index) % (2**31)}])[0]


def _colors(base: np.ndarray, amount: np.ndarray, n: int) -> np.ndarray:
    """Vary brightness while keeping the supplied scene palette recognizable."""
    brightness = np.clip(amount.reshape(n, 1), 0.0, 1.0)
    return np.clip(base[None, :] * brightness + 255.0 * np.maximum(brightness - 0.82, 0) * 0.35,
                   0, 255).astype(np.uint8)


def evaluate_effects(effects: object, time_seconds: object) -> list[dict]:
    """Evaluate fixed-count particles and optional line geometry at playhead time.

    Each returned state has ``id``, ``kind``, ``points`` (N x 3 float32),
    ``colors`` (N x 3 uint8), ``point_size`` (metres), and ``point_shape``.
    Rain and sparks additionally include ``segments`` (N x 2 x 3). Portal
    includes ``segments`` for two closed, animated ring paths. No state is
    accumulated between calls.
    """
    canonical = validate_effects(effects)
    time = _number(time_seconds, "time_seconds", 0.0, 86400.0)
    states = []
    for effect in canonical:
        kind = effect["kind"]
        intensity = effect["intensity"]
        if intensity == 0.0:
            states.append({"id": effect["id"], "kind": kind,
                           "points": np.empty((0, 3), dtype=np.float32),
                           "colors": np.empty((0, 3), dtype=np.uint8),
                           "point_size": 0.01, "point_shape": "circle"})
            continue
        n = 128 + round(128 * intensity)
        rng = np.random.default_rng(effect["seed"])
        # Fixed random samples give continuous motion without changing particle
        # identity whenever time advances or the playhead seeks backwards.
        u = rng.random((n, 6), dtype=np.float64)
        pos = np.asarray(effect["position"], dtype=np.float64)
        size = np.asarray(effect["size"], dtype=np.float64)
        base = np.asarray(effect["color"], dtype=np.float64)
        points = np.empty((n, 3), dtype=np.float64)
        segments = None
        shape = "circle"
        point_size = 0.035
        if kind in ("explosion", "energy_burst"):
            # Point fallback for clients that do not render shader volumes.
            age = (time % 6.0) / 6.0
            theta = u[:, 0] * math.tau
            elevation = u[:, 1] * math.pi
            radius = (0.04 + 0.43 * math.sin(age * math.pi)) * (0.4 + 0.6 * u[:, 2])
            points[:, 0] = np.cos(theta) * np.sin(elevation) * radius * size[0]
            points[:, 1] = np.cos(elevation) * radius * size[1]
            points[:, 2] = np.sin(theta) * np.sin(elevation) * radius * size[2]
            amount = np.full(n, (1.0 - age) ** 2)
            point_size, shape = 0.06, "sparkle"
        elif kind == "rain":
            phase = np.mod(u[:, 0] - time * (0.48 + 0.32 * u[:, 3]), 1.0)
            points[:, 0] = (u[:, 1] - 0.5) * size[0] + 0.07 * np.sin(time * 0.5 + u[:, 4] * 3)
            points[:, 1] = (phase - 0.5) * size[1]
            points[:, 2] = (u[:, 2] - 0.5) * size[2]
            # A narrow streak follows each falling drop.
            ends = points.copy()
            ends[:, 0] -= 0.035 * size[0]
            ends[:, 1] += 0.12 + 0.07 * u[:, 4]
            segments = np.stack((points, ends), axis=1)
            amount = 0.55 + 0.45 * u[:, 5]
            point_size, shape = 0.012, "diamond"
        elif kind == "snow":
            phase = np.mod(u[:, 0] - time * (0.055 + 0.045 * u[:, 3]), 1.0)
            points[:, 0] = (u[:, 1] - 0.5) * size[0] + 0.07 * np.sin(time * 0.6 + u[:, 4] * 8)
            points[:, 1] = (phase - 0.5) * size[1]
            points[:, 2] = (u[:, 2] - 0.5) * size[2] + 0.06 * np.cos(time * 0.4 + u[:, 5] * 9)
            amount = 0.7 + 0.3 * u[:, 4]
            point_size = 0.025 + 0.012 * intensity
        elif kind == "fireflies":
            points[:, 0] = (u[:, 0] - 0.5) * size[0] + 0.13 * np.sin(time * (0.7 + u[:, 3]) + u[:, 4] * 6)
            points[:, 1] = (u[:, 1] - 0.5) * size[1] + 0.09 * np.cos(time * (0.8 + u[:, 4]) + u[:, 5] * 6)
            points[:, 2] = (u[:, 2] - 0.5) * size[2] + 0.12 * np.cos(time * (0.6 + u[:, 5]) + u[:, 3] * 6)
            amount = 0.17 + 0.83 * (0.5 + 0.5 * np.sin(time * (2.3 + u[:, 3] * 3) + u[:, 4] * math.tau)) ** 4
            point_size, shape = 0.045, "sparkle"
        elif kind == "sparks":
            age = np.mod(u[:, 0] + time * (0.5 + 0.3 * u[:, 3]), 1.0)
            angle = u[:, 1] * math.tau
            spread = (0.1 + 0.42 * u[:, 2]) * age
            points[:, 0] = np.cos(angle) * spread * size[0]
            points[:, 1] = -0.32 * size[1] + (1.55 * age - 1.2 * age * age) * size[1]
            points[:, 2] = np.sin(angle) * spread * size[2]
            tail = points.copy()
            tail[:, 0] *= 0.94
            tail[:, 1] -= 0.05 + 0.06 * age
            tail[:, 2] *= 0.94
            segments = np.stack((points, tail), axis=1)
            amount = 1.0 - 0.78 * age
            point_size, shape = 0.025, "sparkle"
        elif kind == "smoke":
            age = np.mod(u[:, 0] + time * (0.085 + 0.06 * u[:, 3]), 1.0)
            radius = (0.08 + age * 0.42) * np.sqrt(u[:, 1])
            angle = u[:, 2] * math.tau + age * 2.2
            points[:, 0] = np.cos(angle) * radius * size[0] + 0.12 * age * np.sin(time * 0.8)
            points[:, 1] = (age - 0.5) * size[1]
            points[:, 2] = np.sin(angle) * radius * size[2]
            amount = 0.36 + 0.33 * np.sin(math.pi * age)
            point_size, shape = 0.085, "rounded"
        else:  # portal: two animated rings plus a rotating interior field.
            theta = u[:, 0] * math.tau + time * (0.1 + 0.08 * u[:, 3])
            radius = 0.7 + 0.25 * u[:, 1]
            points[:, 0] = np.cos(theta) * radius * size[0] * 0.5
            points[:, 1] = np.sin(theta) * radius * size[1] * 0.5
            points[:, 2] = (u[:, 2] - 0.5) * size[2] + 0.06 * np.sin(theta * 4 + time * 2)
            amount = 0.45 + 0.55 * (0.5 + 0.5 * np.sin(theta * 8 - time * 4))
            point_size, shape = 0.04, "sparkle"
            angles = np.linspace(0.0, math.tau, 97)
            rings = []
            for layer in (0, 1):
                wobble = 1.0 + 0.025 * np.sin(angles * 7 + time * (2.2 if layer == 0 else -1.7))
                ring = np.column_stack((np.cos(angles) * size[0] * (0.48 + 0.045 * layer) * wobble,
                                        np.sin(angles) * size[1] * (0.48 + 0.045 * layer) * wobble,
                                        np.full_like(angles, (layer - 0.5) * size[2] * 0.22)))
                rings.append(np.stack((ring[:-1], ring[1:]), axis=1))
            segments = np.concatenate(rings, axis=0)

        points += pos
        if segments is not None:
            segments += pos
        point_colors = _colors(base, amount, n)
        state = {"id": effect["id"], "kind": kind, "points": points.astype(np.float32),
                 "colors": point_colors, "point_size": float(point_size), "point_shape": shape}
        if segments is not None:
            state["segments"] = segments.astype(np.float32)
            state["segment_colors"] = (base.astype(np.uint8)
                                       if kind == "portal" else np.repeat(point_colors[:, None, :], 2, axis=1))
        states.append(state)
    return states


class EffectSceneLayer:
    """Stream pure effect states into stable Viser nodes and remove stale nodes."""

    def __init__(self, server):
        self.server = server
        self.handles: dict[str, dict] = {}

    def update(self, effects: object, time_seconds: object) -> None:
        states = evaluate_effects(effects, time_seconds)
        current = {state["id"] for state in states}
        for identifier in set(self.handles) - current:
            for handle in self.handles.pop(identifier).values():
                handle.remove()
        for state in states:
            identifier = state["id"]
            old = self.handles.get(identifier)
            if state["kind"] in ("explosion", "energy_burst"):
                effect = next(e for e in effects if e["id"] == identifier)
                # 4x3 float32 descriptor travels through the point-cloud protocol.
                descriptor = np.asarray([
                    [float(time_seconds), effect["intensity"], effect["seed"] % 65536],
                    effect["size"], np.asarray(effect["color"]) / 255.0,
                    [0, 0, 0],
                ], dtype=np.float32)
                key = "cinematic_" + state["kind"]
                if old is not None and key not in old:
                    for handle in old.values():
                        handle.remove()
                    old = None
                if old is None:
                    old = {key: self.server.scene.add_point_cloud(
                        "/effects/" + identifier + "/" + key, descriptor,
                        np.zeros((4, 3), dtype=np.uint8), point_size=0.001,
                        precision="float32", position=tuple(effect["position"]))}
                    self.handles[identifier] = old
                else:
                    old[key].points = descriptor
                    old[key].position = tuple(effect["position"])
                continue
            if old is not None and "points" not in old:
                for handle in old.values():
                    handle.remove()
                old = None
            needs_segments = "segments" in state
            if old is not None and ("segments" in old) != needs_segments:
                for handle in old.values():
                    handle.remove()
                old = None
            if old is None:
                name = "/effects/" + identifier
                old = {"points": self.server.scene.add_point_cloud(
                    name + "/particles", state["points"], state["colors"],
                    point_size=state["point_size"], point_shape=state["point_shape"], precision="float32")}
                if needs_segments:
                    old["segments"] = self.server.scene.add_line_segments(
                        name + "/streaks", state["segments"], state["segment_colors"],
                        line_width=2.0 if state["kind"] == "portal" else 1.0)
                self.handles[identifier] = old
            else:
                old["points"].points = state["points"]
                old["points"].colors = state["colors"]
                old["points"].point_size = state["point_size"]
                old["points"].point_shape = state["point_shape"]
                if needs_segments:
                    old["segments"].points = state["segments"]
                    old["segments"].colors = state["segment_colors"]
                    old["segments"].line_width = 2.0 if state["kind"] == "portal" else 1.0
