"""Small, reproducible ARDY Core action candidates for staged scenes.

These are prompts and native conditioning requests, not successful motion clips.
Only a caller that has inspected generated output may record an outcome.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping


MODEL = "ARDY-Core-RP-20FPS-Horizon40"
SOURCE = "ardy_core"
CONTACT_MODE = "staged_no_contact"
SAMPLE_FRAMES = (7, 15, 23, 31, 39)

PROMPTS = {
    "walk": "Walk naturally along the street toward the marked position, keeping a steady gait.",
    "stop": "Slow to a natural stop and settle your weight over both feet.",
    "turn": "Turn your body toward the other person while keeping your feet grounded.",
    "meet": "Notice the other person and pause facing them at a respectful distance.",
    "guard": "Raise a light sparring guard while staying at a safe distance.",
    "lunge": "Make a short controlled sparring feint toward the other person, stopping well short of contact.",
    "dodge": "Take a small sidestep away from the feint while keeping balance and distance.",
    "hit_reaction": "Act a brief startled reaction to a near miss, without being struck, then recover balance.",
    "retreat": "Step back from the other person, lower your guard, and regain a relaxed stance.",
    "depart": "Turn away and walk along the marked street route, leaving the encounter behind.",
}


def make_action_candidate(action: str, actor_id: str, *, seed: int,
                          root_targets: list[dict], conditions: Mapping | None = None,
                          prompt: str | None = None) -> dict:
    """Return JSON-safe input provenance with explicitly unobserved outcome."""
    if action not in PROMPTS:
        raise ValueError(f"Unknown Core action: {action}")
    if not isinstance(actor_id, str) or not actor_id or len(actor_id) > 64:
        raise ValueError("actor_id must be a stable nonempty ID")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a uint32")
    text = PROMPTS[action] if prompt is None else prompt
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 500:
        raise ValueError("prompt must contain 1–500 characters")
    if not isinstance(root_targets, list) or not root_targets or len(root_targets) > 24:
        raise ValueError("native root targets must contain 1–24 constraints")
    seen = set()
    normalized = []
    for target in root_targets:
        if not isinstance(target, dict) or set(target) != {"frame", "position_xz", "heading"}:
            raise ValueError("each root target needs frame, position_xz and heading")
        frame, xz, heading = target["frame"], target["position_xz"], target["heading"]
        if type(frame) is not int or not 0 <= frame < 40 or frame in seen:
            raise ValueError("root target frames must be unique local 0–39 indices")
        if (not isinstance(xz, (tuple, list)) or len(xz) != 2 or
                any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 25 for v in xz) or
                type(heading) not in (int, float) or not math.isfinite(heading) or abs(heading) > math.pi):
            raise ValueError("root target position or heading is outside native bounds")
        seen.add(frame)
        normalized.append({"frame": frame, "position_xz": [float(xz[0]), float(xz[1])],
                           "heading": float(heading)})
    if sorted(seen) != [x["frame"] for x in normalized]:
        raise ValueError("root targets must be sorted by frame")
    if conditions is not None and not isinstance(conditions, Mapping):
        raise ValueError("conditions must be a mapping")
    try:
        safe_conditions = json.loads(json.dumps(dict(conditions or {}), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("conditions must contain finite JSON values") from exc
    return {"action": action, "actor_id": actor_id, "prompt": text.strip(),
            "seed": seed, "model": MODEL, "source": SOURCE,
            "conditions": safe_conditions,
            "native_constraints": {"root_targets": normalized},
            "output": None, "measurement": None, "status": "planned_unobserved"}


def record_action_outcome(candidate: Mapping, *, output: Mapping,
                          measurement: Mapping, status: str) -> dict:
    """Attach observed evidence; never infer success from prompt or targets."""
    if status not in {"observed_pass", "observed_fail", "generation_failed"}:
        raise ValueError("outcome status must be an observed pass/fail or generation failure")
    if not isinstance(output, Mapping) or not output:
        raise ValueError("an output reference is required")
    if not isinstance(measurement, Mapping) or not measurement:
        raise ValueError("observed measurements are required")
    if status == "observed_pass" and any(measurement.get(key) is not True for key in (
            "geometry_checks_pass", "visual_review_pass", "instruction_followed")):
        raise ValueError("observed pass requires positive geometry, visual, and instruction verdicts")
    if candidate.get("status") != "planned_unobserved":
        raise ValueError("only an unobserved candidate may be recorded")
    try:
        result = json.loads(json.dumps(dict(candidate), allow_nan=False))
        result["output"] = json.loads(json.dumps(dict(output), allow_nan=False))
        result["measurement"] = json.loads(json.dumps(dict(measurement), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("outcome evidence must contain finite JSON values") from exc
    result["status"] = status
    return result
