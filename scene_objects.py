"""Small, UI-independent scene objects driven by a recorded joint trajectory.

Coordinates are metres, with +Y up. ``positions`` is indexed as
``[frame][joint][xyz]``. Evaluation is pure: replaying or seeking to a frame
always produces the same state from the trajectory prefix.
"""

from __future__ import annotations

import math
import re
from numbers import Integral, Real

import numpy as np


MAX_OBJECTS = 40
KINDS = {
    "door": {"action": "open", "trigger": "proximity", "size": [0.9, 2.0, 0.12], "color": [122, 91, 68], "radius": 0.8},
    "lamp": {"action": "switch_on", "trigger": "proximity", "size": [0.35, 1.4, 0.35], "color": [142, 149, 168], "radius": 0.7},
    "ball": {"action": "pick_up", "trigger": "touch", "size": [0.3, 0.3, 0.3], "color": [230, 82, 69], "radius": 0.04},
    "chair": {"action": "sit", "trigger": "proximity", "size": [0.7, 0.9, 0.7], "color": [89, 121, 151], "radius": 0.55},
    "table": {"action": "none", "trigger": "none", "size": [1.6, 0.8, 0.9], "color": [145, 105, 72], "radius": 0},
    "sofa": {"action": "none", "trigger": "none", "size": [2.2, 0.9, 0.95], "color": [99, 117, 144], "radius": 0},
    "crate": {"action": "none", "trigger": "none", "size": [0.8, 0.8, 0.8], "color": [150, 111, 70], "radius": 0},
    "barrel": {"action": "none", "trigger": "none", "size": [0.65, 0.9, 0.65], "color": [132, 88, 58], "radius": 0},
    "pillar": {"action": "none", "trigger": "none", "size": [0.65, 2.6, 0.65], "color": [170, 171, 165], "radius": 0},
    "wall": {"action": "none", "trigger": "none", "size": [5.0, 2.5, 0.24], "color": [149, 153, 158], "radius": 0},
    "arch": {"action": "none", "trigger": "none", "size": [2.8, 2.7, 0.38], "color": [177, 161, 137], "radius": 0},
    "plant": {"action": "none", "trigger": "none", "size": [0.75, 1.0, 0.75], "color": [61, 139, 83], "radius": 0},
    "tree": {"action": "none", "trigger": "none", "size": [1.8, 3.2, 1.8], "color": [68, 137, 75], "radius": 0},
    "rock": {"action": "none", "trigger": "none", "size": [1.1, 0.7, 0.9], "color": [120, 123, 124], "radius": 0},
    "console": {"action": "activate", "trigger": "proximity", "size": [1.1, 1.15, 0.65], "color": [64, 86, 104], "radius": 0.8},
    "platform": {"action": "none", "trigger": "none", "size": [3.0, 0.3, 2.0], "color": [87, 95, 105], "radius": 0},
}
_FIELDS = frozenset(("id", "name", "kind", "position", "size", "color", "interaction"))
_INTERACTION_FIELDS = frozenset(("action", "trigger", "radius"))
_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


def _real(value: object, label: str, minimum: float, maximum: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < minimum or number > maximum or (positive and number == 0):
        raise ValueError(f"{label} must be finite and between {minimum} and {maximum}")
    return number


def _vector(value: object, label: str, minimum: float, maximum: float, *, positive: bool = False) -> list[float]:
    try:
        valid = not isinstance(value, (str, bytes, dict)) and len(value) == 3
    except TypeError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must have three coordinates")
    return [_real(v, f"{label}[{i}]", minimum, maximum, positive=positive) for i, v in enumerate(value)]


def make_object(kind: str, index: int, position: list[float] | tuple[float, float, float] | None = None) -> dict:
    """Return a validated default for one of the supported object kinds.

    ``index`` distinguishes IDs when creating several objects of one kind.
    Position is the object's centre, so the default Y rests its bottom at 0.
    """
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError(f"unsupported object kind: {kind!r}")
    if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
        raise ValueError("index must be a non-negative integer")
    template = KINDS[kind]
    centre = [0.0, template["size"][1] / 2.0, 0.0] if position is None else position
    return validate_objects([{
        "id": f"{kind}-{index}",
        "name": kind.capitalize(),
        "kind": kind,
        "position": centre,
        "size": template["size"],
        "color": template["color"],
        "interaction": {
            "action": template["action"],
            "trigger": template["trigger"],
            "radius": template["radius"],
        },
    }])[0]


def validate_objects(value: object) -> list[dict]:
    """Validate and copy untrusted object JSON into the canonical schema.

    Unknown keys, duplicate IDs, unsupported actions, and implausible units
    fail closed. Nothing executable is accepted from generated content.
    """
    if not isinstance(value, list) or len(value) > MAX_OBJECTS:
        raise ValueError(f"objects must be a list of at most {MAX_OBJECTS} items")
    result = []
    seen_ids = set()
    for i, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != _FIELDS:
            raise ValueError(f"objects[{i}] has missing or unknown fields")
        kind = item["kind"]
        if not isinstance(kind, str) or kind not in KINDS:
            raise ValueError(f"objects[{i}].kind is unsupported")
        identifier = item["id"]
        if not isinstance(identifier, str) or not _ID_PATTERN.fullmatch(identifier) or identifier in seen_ids:
            raise ValueError(f"objects[{i}].id is invalid or duplicated")
        seen_ids.add(identifier)
        name = item["name"]
        if not isinstance(name, str) or not 1 <= len(name) <= 80 or not name.strip() or any(ord(c) < 32 for c in name):
            raise ValueError(f"objects[{i}].name is invalid")
        position = _vector(item["position"], f"objects[{i}].position", -20, 20)
        size = _vector(item["size"], f"objects[{i}].size", 0.05, 12)
        if kind == "ball" and max(size) - min(size) > 1e-6:
            raise ValueError(f"objects[{i}].size must be equal on all axes for a ball")
        color = item["color"]
        if not isinstance(color, (list, tuple)) or len(color) != 3 or any(
            isinstance(c, bool) or not isinstance(c, Integral) or not 0 <= c <= 255 for c in color
        ):
            raise ValueError(f"objects[{i}].color must contain three RGB bytes")
        interaction = item["interaction"]
        if not isinstance(interaction, dict) or set(interaction) != _INTERACTION_FIELDS:
            raise ValueError(f"objects[{i}].interaction has missing or unknown fields")
        if interaction["action"] != KINDS[kind]["action"]:
            raise ValueError(f"objects[{i}].interaction.action does not match kind")
        trigger = interaction["trigger"]
        allowed_triggers = ("proximity", "touch") if kind in ("door", "lamp") else (KINDS[kind]["trigger"],)
        if trigger not in allowed_triggers:
            raise ValueError(f"objects[{i}].interaction.trigger is invalid for kind")
        max_radius = 0 if trigger == "none" else (0.15 if trigger == "touch" else 5.0)
        radius = _real(interaction["radius"], f"objects[{i}].interaction.radius", 0, max_radius,
                       positive=trigger != "none")
        result.append({
            "id": identifier, "name": name.strip(), "kind": kind,
            "position": position, "size": size, "color": [int(c) for c in color],
            "interaction": {"action": interaction["action"], "trigger": trigger, "radius": radius},
        })
    return result


def evaluate_objects(objects: object, positions: object, frame: int, hand_indices: object = None) -> list[dict]:
    """Evaluate visible state at ``frame`` from all preceding motion frames.

    Touch uses only supplied hand joint indices. An acquired ball follows the
    first touching hand; a chair changes only its visual cue and makes no
    claim that the character actually sat down.
    """
    canonical = validate_objects(objects)
    if isinstance(frame, bool) or not isinstance(frame, Integral) or frame < 0:
        raise ValueError("frame must be a non-negative integer")
    try:
        length = len(positions)
    except (TypeError, ValueError):
        raise ValueError("positions must be [frame][joint][xyz]") from None
    if not length or frame >= length:
        raise ValueError("frame is outside positions")
    if hand_indices is None:
        hands = []
    elif not isinstance(hand_indices, (list, tuple)) or any(
        isinstance(index, bool) or not isinstance(index, Integral) or index < 0 for index in hand_indices
    ):
        raise ValueError("hand_indices must be non-negative joint indices")
    else:
        hands = list(dict.fromkeys(int(index) for index in hand_indices))
    if any(obj["interaction"]["trigger"] == "touch" for obj in canonical) and not hands:
        raise ValueError("touch interactions require explicit hand_indices")

    try:
        trajectory = np.asarray(positions[:frame + 1], dtype=np.float64)
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("positions must be [frame][joint][xyz]") from exc
    if trajectory.ndim != 3 or trajectory.shape[1] < 1 or trajectory.shape[2] != 3:
        raise ValueError("positions must be [frame][joint][xyz]")
    if not np.isfinite(trajectory).all() or np.any(np.abs(trajectory) > 1000):
        raise ValueError("positions must have finite coordinates within 1000 metres")
    if any(index >= trajectory.shape[1] for index in hands):
        raise ValueError("hand_indices exceed joint count")

    active = [False] * len(canonical)
    holders: list[int | None] = [None] * len(canonical)
    root_xz = trajectory[:, 0, :][:, [0, 2]]
    for i, obj in enumerate(canonical):
        if obj["interaction"]["trigger"] == "none":
            continue
        centre = np.asarray(obj["position"], dtype=np.float64)
        size = np.asarray(obj["size"], dtype=np.float64)
        radius = obj["interaction"]["radius"]
        if obj["interaction"]["trigger"] == "proximity":
            # Ground-plane footprint distance keeps tall props reachable when
            # the root is at floor height. Joint 0 is the character root.
            outside = np.maximum(np.abs(root_xz - centre[[0, 2]]) - size[[0, 2]] / 2, 0)
            active[i] = bool(np.any(np.linalg.norm(outside, axis=-1) <= radius))
        else:
            contact_points = trajectory[:, hands, :]
            if obj["kind"] == "ball":
                surface_distance = np.linalg.norm(contact_points - centre, axis=-1) - size[0] / 2
            else:
                outside = np.maximum(np.abs(contact_points - centre) - size / 2, 0)
                surface_distance = np.linalg.norm(outside, axis=-1)
            touches = np.argwhere(surface_distance <= radius)
            if touches.size:
                active[i] = True
                holders[i] = hands[int(touches[0, 1])]

    states = []
    for i, obj in enumerate(canonical):
        position = obj["position"].copy()
        color = obj["color"].copy()
        if active[i]:
            if obj["kind"] == "door":
                position[1] += obj["size"][1]
            elif obj["kind"] == "lamp":
                color = [min(255, round(0.35 * channel + 0.65 * 255)) for channel in color]
            elif obj["kind"] == "ball":
                position = trajectory[-1, holders[i], :].tolist()
            elif obj["kind"] == "chair":
                color = [min(255, round(0.65 * channel + 0.35 * 255)) for channel in color]
        states.append({"id": obj["id"], "position": position, "color": color, "active": active[i]})
    return states
