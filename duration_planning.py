"""Deterministic duration planning for 25 fps, 104-frame motion chunks.

Auto is a rough editing convenience. It does not infer when an action is complete.
"""
from dataclasses import dataclass
import math
import re

FPS = 25
CHUNK_FRAMES = 104
MIN_FRAMES = 4
MAX_FRAMES_PER_REQUEST = 750  # 30 seconds, at most eight backend calls.


@dataclass(frozen=True)
class DurationPlan:
    frames: int
    seconds: float
    label: str
    automatic: bool


def plan_duration(prompt, seconds=None):
    """Quantize a requested length to frames, or estimate Auto from visible text.

    Auto accepts a single explicit duration such as "walk for 10 seconds".
    Otherwise the estimate uses only prompt size and sequence words.
    """
    automatic = seconds is None or seconds == 'auto'
    if automatic:
        text = str(prompt)
        matches = re.findall(r'(?<![\w.])(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b', text, re.I)
        if len(matches) == 1:
            requested = float(matches[0])
            if not math.isfinite(requested) or requested <= 0:
                raise ValueError('Prompt duration must be greater than zero')
            frames = max(MIN_FRAMES, min(MAX_FRAMES_PER_REQUEST, round(requested * FPS)))
            if requested * FPS > MAX_FRAMES_PER_REQUEST:
                label = 'Auto · prompt duration capped at 30 s'
            elif requested * FPS < MIN_FRAMES:
                label = 'Auto · prompt duration raised to 0.16 s minimum'
            else:
                label = 'Auto · duration stated in prompt'
        else:
            words = re.findall(r"\b[\w']+\b", text)
            sequence = bool(re.search(r'\b(then|afterwards|followed by|next|finally)\b', text, re.I))
            chunks = 3 if len(words) > 25 else 2 if len(words) > 10 or sequence else 1
            frames = chunks * CHUNK_FRAMES
            label = 'Auto · text-length estimate'
    else:
        if isinstance(seconds, bool):
            raise ValueError('Duration must be a number of seconds')
        try:
            requested = float(seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Duration must be a number of seconds') from exc
        # UI total-minus-prefix arithmetic can produce 0.15999999999999992.
        if not math.isfinite(requested) or not MIN_FRAMES / FPS - 1e-9 <= requested <= MAX_FRAMES_PER_REQUEST / FPS + 1e-9:
            raise ValueError('Duration must be between 0.16 and 30 seconds')
        frames = max(MIN_FRAMES, min(MAX_FRAMES_PER_REQUEST, round(requested * FPS)))
        label = 'Specified duration'
    return DurationPlan(frames, frames / FPS, label, automatic)
