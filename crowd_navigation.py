"""Deterministic disc-proxy crossing trajectories for crowd playback.

Flat square walkable proxy; this does not infer floors or obstacles from pixels.
The local solver predicts closest approach, steers right, limits acceleration
and turning, then projects residual disc overlaps. It is not an ORCA solver and
does not guarantee full-body collision avoidance.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import random
import time
from typing import Mapping

import numpy as np

WAIT, CROSS, CLEAR, STAGE = range(4)
STATES = ("curb_wait", "crossing", "clearing", "staging")
SIDES = ("north", "east", "south", "west")
GAITS = ("casual", "brisk", "relaxed")


def _weights(source: Mapping[str, float] | None, names: tuple[str, ...]) -> list[float]:
    if source is None:
        return [1 / len(names)] * len(names)
    if not isinstance(source, Mapping) or set(source) - set(names):
        raise ValueError(f"weights must use {names}")
    values = [float(source.get(n, 0)) for n in names]
    if not all(math.isfinite(v) and v >= 0 for v in values) or sum(values) <= 0:
        raise ValueError("weights must be finite, nonnegative, and nonzero")
    return [v / sum(values) for v in values]


@dataclass(frozen=True)
class CrowdConfig:
    count: int = 64
    seed: int = 42
    duration_s: float = 120.0
    sample_hz: int = 15
    speed_min_mps: float = 0.75
    speed_max_mps: float = 1.55
    diagonal_fraction: float = 0.35
    group_fraction: float = 0.25
    side_weights: Mapping[str, float] | None = None
    style_weights: Mapping[str, float] | None = None
    crossing_half_extent_m: float = 10.0
    world_half_extent_m: float = 16.0
    red_s: float = 15.0
    walk_s: float = 38.0
    clearance_s: float = 7.0

    def __post_init__(self) -> None:
        if not 1 <= self.count <= 1000 or self.duration_s <= 0:
            raise ValueError("invalid count or duration")
        if self.sample_hz < 1 or 30 % self.sample_hz:
            raise ValueError("sample_hz must divide 30")
        if not 0.1 <= self.speed_min_mps <= self.speed_max_mps <= 3:
            raise ValueError("invalid walking speeds")
        if not 0 <= self.diagonal_fraction <= 1 or not 0 <= self.group_fraction <= 1:
            raise ValueError("invalid fractions")
        if not 5 <= self.crossing_half_extent_m <= 20:
            raise ValueError("invalid crossing extent")
        if self.world_half_extent_m - self.crossing_half_extent_m < 5:
            raise ValueError("sidewalk band needs at least five meters")
        if min(self.red_s, self.walk_s, self.clearance_s) <= 0:
            raise ValueError("traffic phase durations must be positive")
        _weights(self.side_weights, SIDES)
        _weights(self.style_weights, GAITS)


def traffic_phase(t: float, cfg: CrowdConfig) -> tuple[str, int]:
    period = cfg.red_s + cfg.walk_s + cfg.clearance_s
    cycle = int(max(0, t) // period)
    clock = max(0, t) - cycle * period
    return ("red" if clock < cfg.red_s else
            "walk" if clock < cfg.red_s + cfg.walk_s else "clearance"), cycle


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-8)


def _unit_rows(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)


def _clamp(v: np.ndarray, limit: np.ndarray | float) -> np.ndarray:
    length = np.linalg.norm(v, axis=-1)
    return v * np.minimum(1, np.asarray(limit) / np.maximum(length, 1e-8))[..., None]


def _route(rng: random.Random, cfg: CrowdConfig) -> tuple[str, np.ndarray, np.ndarray]:
    c, w = cfg.crossing_half_extent_m, cfg.world_half_extent_m
    if rng.random() < cfg.diagonal_fraction:
        name = rng.choice(("southwest", "northeast", "northwest", "southeast"))
        sx, sz = {"southwest": (-1, -1), "northeast": (1, 1),
                  "northwest": (-1, 1), "southeast": (1, -1)}[name]
        near = min(w - 2.1, c + 3.1)
        a = np.array([sx * rng.uniform(c + 1.1, near),
                      sz * rng.uniform(c + 1.1, near)])
        b = np.array([-sx * rng.uniform(c + 1.1, c + 1.8),
                      -sz * rng.uniform(c + 1.1, c + 1.8)])
        return name, a, b
    name = rng.choices(SIDES, weights=_weights(cfg.side_weights, SIDES))[0]
    lane = rng.uniform(-c + 2, c - 2)
    lane += {"west": -1, "east": 1, "north": -1, "south": 1}[name]
    lane = max(-c + 1, min(c - 1, lane))
    a = rng.uniform(c + 1.2, min(w - 1.5, c + 4))
    b = rng.uniform(c + 1.1, c + 1.8)
    drift = rng.uniform(-0.8, 0.8)
    if name == "west":
        return name, np.array([-a, lane]), np.array([b, lane + drift])
    if name == "east":
        return name, np.array([a, lane]), np.array([-b, lane + drift])
    if name == "north":
        return name, np.array([lane, a]), np.array([lane + drift, -b])
    return name, np.array([lane, -a]), np.array([lane + drift, b])


def _place(cfg: CrowdConfig) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(cfg.seed)
    people: list[dict] = []
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    speeds: list[float] = []
    group = 0
    while len(people) < cfg.count:
        size = min(rng.choice((2, 2, 3)) if rng.random() < cfg.group_fraction else 1,
                   cfg.count - len(people))
        # Place a whole group together. Splitting one member to an unrelated
        # fallback route would leave a false group label in the archive.
        for _ in range(3000):
            route, anchor, far = _route(rng, cfg)
            forward = _unit(far - anchor)
            side = np.array([-forward[1], forward[0]])
            offsets = [(m - (size - 1) / 2) * 0.95 for m in range(size)]
            candidates = [anchor + side * offset for offset in offsets]
            if any(max(abs(p)) > cfg.world_half_extent_m - 0.35 for p in candidates):
                continue
            if any(np.linalg.norm(p - old) < 0.78 for p in candidates for old in starts):
                continue
            break
        else:
            raise RuntimeError("could not place crowd without initial overlap")
        group_speed = rng.uniform(cfg.speed_min_mps, cfg.speed_max_mps)
        route_length = float(np.linalg.norm(far - anchor))
        delay_ceiling = min(4.5, max(0.0, cfg.walk_s - route_length / group_speed - 3.0))
        group_delay = rng.uniform(0.0, delay_ceiling)
        for member in range(size):
            offset, candidate = offsets[member], candidates[member]
            idx = len(people)
            destination = far + side * offset * 0.65
            speed = min(cfg.speed_max_mps, max(cfg.speed_min_mps,
                group_speed + rng.uniform(-0.06, 0.06)))
            gait = rng.choices(GAITS, weights=_weights(cfg.style_weights, GAITS))[0]
            starts.append(candidate)
            ends.append(destination)
            speeds.append(speed)
            people.append({"id": idx, "group_id": group if size > 1 else None,
                           "radius": 0.29, "gait": gait, "color_index": rng.randrange(16),
                           "route": route, "speed_mps": round(speed, 3),
                           "launch_delay_s": round(group_delay, 3),
                           "start_xz": np.round(candidate, 3).tolist(),
                           "end_xz": np.round(destination, 3).tolist()})
        group += 1
    return people, np.array(starts), np.array(ends), np.array(speeds)


def _pairs(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = pos[:, None, :] - pos[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distance, np.inf)
    return delta, distance


@dataclass
class CrowdResult:
    config: CrowdConfig
    agents: list[dict]
    trajectories: np.ndarray  # samples x agents x [x,z,heading,speed,state,distance]
    metrics: dict

    def to_dict(self) -> dict:
        data = np.round(self.trajectories, 3)
        if not np.isfinite(data).all():
            raise ValueError("trajectory contains nonfinite number")
        frames = []
        for index, people in enumerate(data):
            rows = people.tolist()
            for row in rows:
                row[4] = int(row[4])
            frames.append({"t": round(index / self.config.sample_hz, 4), "people": rows})
        c, w = self.config.crossing_half_extent_m, self.config.world_half_extent_m
        return {"schema": "stagezero.crowd.v1", "units": "meters", "up_axis": "Y",
                "heading": "atan2(vx,vz), zero faces +Z", "duration": self.config.duration_s,
                "dt": 1 / self.config.sample_hz, "state_names": list(STATES),
                "person_columns": ["x", "z", "heading", "speed", "state", "distance"],
                "config": asdict(self.config), "agents": self.agents,
                "geometry": {"crossing_xz_min": [-c, -c], "crossing_xz_max": [c, c],
                             "world_xz_min": [-w, -w], "world_xz_max": [w, w],
                             "floor_y": 0, "obstacles": [],
                             "limitation": "flat proxy; background image not collision scanned"},
                "metrics": self.metrics, "frames": frames}

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), separators=(",", ":"), allow_nan=False))
        return path


def simulate_crowd(config: CrowdConfig | Mapping[str, object] = CrowdConfig()) -> CrowdResult:
    if isinstance(config, Mapping):
        unknown = set(config) - {f.name for f in fields(CrowdConfig)}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        config = CrowdConfig(**config)
    cfg = config
    began = time.perf_counter()
    agents, start, end, speed = _place(cfg)
    n = cfg.count
    grouped: dict[int, list[int]] = {}
    for agent in agents:
        if agent["group_id"] is not None:
            grouped.setdefault(agent["group_id"], []).append(agent["id"])
    dt, stride = 1 / 30, 30 // cfg.sample_hz
    ticks = round(cfg.duration_s / dt)
    trace = np.empty((ticks // stride + 1, n, 6), np.float32)
    pos, vel = start.copy(), np.zeros((n, 2))
    yaw = np.arctan2((end - start)[:, 0], (end - start)[:, 1])
    state = np.full(n, WAIT, np.int8)
    target = end.copy()
    to_end = np.ones(n, bool)
    last_launch = np.full(n, -1, int)
    distance = np.zeros(n)
    actual_speed = np.zeros(n)
    prior_actual_velocity = np.zeros((n, 2))
    completions = np.zeros(n, int)
    last_pos_3s = pos.copy()
    stalled_run_seconds = np.zeros(n)
    deadlocked = np.zeros(n, bool)
    oscillation_count = np.zeros(n, int)
    prior_turn = np.zeros(n)
    accel_values, commanded_accel_values, turn_values, step_times = [], [], [], []
    group_spacings: list[float] = []
    min_clearance = math.inf
    overlap_pair_steps = peak_overlaps = preprojection_pair_steps = 0
    projection_m = 0.0
    projection_step_max = 0.0
    red_violations = 0
    red_occupancy_steps = 0

    def capture(index: int) -> None:
        trace[index, :, :2] = pos
        trace[index, :, 2] = yaw
        trace[index, :, 3] = actual_speed
        trace[index, :, 4] = state
        trace[index, :, 5] = distance

    capture(0)
    radius = np.full(n, 0.29)
    launch_delay = np.array([a["launch_delay_s"] for a in agents])
    for step in range(ticks):
        tick_start = time.perf_counter()
        phase, cycle = traffic_phase(step * dt, cfg)
        if phase == "walk":
            period = cfg.red_s + cfg.walk_s + cfg.clearance_s
            clock = step * dt - cycle * period
            launch = ((state == WAIT) | (state == STAGE)) & (last_launch < cycle) & (
                clock >= cfg.red_s + launch_delay)
            state[launch], last_launch[launch] = CROSS, cycle
            target[launch] = np.where(to_end[launch, None], end[launch], start[launch])
        gap = np.linalg.norm(target - pos, axis=1)
        finished = (state == CROSS) & (gap < 0.5)
        if np.any(finished):
            completions[finished] += 1
            state[finished] = CLEAR
            from_point = np.where(to_end[finished, None], start[finished], end[finished])
            outward = _unit_rows(target[finished] - from_point)
            target[finished] += outward * 1.7
            to_end[finished] = ~to_end[finished]
        gap = np.linalg.norm(target - pos, axis=1)
        cleared = (state == CLEAR) & (gap < 0.4)
        if np.any(cleared):
            state[cleared] = STAGE
            upcoming = np.where(to_end[cleared, None], end[cleared], start[cleared])
            target[cleared] = pos[cleared] + _unit_rows(upcoming - pos[cleared]) * 0.7
        goal = target - pos
        gap = np.linalg.norm(goal, axis=1)
        moving = (state == CROSS) | (state == CLEAR) | ((state == STAGE) & (gap > 0.3))
        desired_speed = np.where(moving, np.minimum(speed, gap * 1.25), 0)
        desired = _unit_rows(goal) * desired_speed[:, None]
        for members in grouped.values():
            points = pos[members]
            toward = points.mean(axis=0) - points
            lengths = np.linalg.norm(toward, axis=1)
            pull = _unit_rows(toward) * np.clip((lengths - 0.7) * 0.3, 0, 0.25)[:, None]
            desired[members] += pull * moving[members, None]
        delta, pair_distance = _pairs(pos)
        safe = radius[:, None] + radius[None, :] + 0.16
        proximity = np.clip((safe + 0.65 - pair_distance) / 0.65, 0, 1)
        separation = np.sum(delta / np.maximum(pair_distance[..., None], 1e-8)
                            * proximity[..., None], axis=1) * 0.43
        relative_v = desired[:, None, :] - vel[None, :, :]
        closest_t = -np.sum(delta * relative_v, axis=2) / np.maximum(
            np.sum(relative_v**2, axis=2), 1e-6)
        near = delta + relative_v * np.clip(closest_t, 0, 1.25)[..., None]
        danger = np.clip((safe + 0.4 - np.linalg.norm(near, axis=2)) / 0.4, 0, 1)
        danger *= (closest_t > 0) & (closest_t < 1.25) & (pair_distance < 3.5)
        np.fill_diagonal(danger, 0)
        right = _unit_rows(np.column_stack((desired[:, 1], -desired[:, 0])))
        candidate = desired + separation + right * np.minimum(danger.sum(axis=1), 2.5)[:, None] * 0.46
        candidate[~moving] = 0
        candidate = _clamp(candidate, speed * 1.1)
        proposed_yaw = np.arctan2(candidate[:, 0], candidate[:, 1])
        turn = (proposed_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        turn[~moving] = 0
        yaw_step = np.clip(turn, -2.7 * dt, 2.7 * dt)
        yaw += yaw_step
        old_vel = vel.copy()
        old_pos = pos.copy()
        old_speed = np.linalg.norm(vel, axis=1)
        target_speed = np.linalg.norm(candidate, axis=1) * np.maximum(0.2, np.cos(turn))
        scalar = old_speed + np.clip(target_speed - old_speed, -2.0 * dt, 2.3 * dt)
        scalar[(~moving) & (scalar < 0.04)] = 0
        vel = np.column_stack((np.sin(yaw), np.cos(yaw))) * scalar[:, None]
        pos += vel * dt
        _, before_correction_distance = _pairs(pos)
        preprojection_pair_steps += int(np.count_nonzero(np.triu(
            before_correction_distance - radius[:, None] - radius[None, :] < -0.005, 1)))
        for _ in range(2):
            delta, pair_distance = _pairs(pos)
            penetration = np.maximum(radius[:, None] + radius[None, :] + 0.035 - pair_distance, 0)
            correction = np.sum(delta / np.maximum(pair_distance[..., None], 1e-8)
                                * penetration[..., None], axis=1) * 0.53
            correction = _clamp(correction, 0.09)
            pos += correction
            projection_m += float(np.linalg.norm(correction, axis=1).sum())
            projection_step_max = max(projection_step_max,
                                      float(np.linalg.norm(correction, axis=1).max()))
        pos = np.clip(pos, -cfg.world_half_extent_m + 0.35, cfg.world_half_extent_m - 0.35)
        actual_velocity = (pos - old_pos) / dt
        actual_speed = np.linalg.norm(actual_velocity, axis=1)
        distance += actual_speed * dt
        accel_values.extend(np.linalg.norm((actual_velocity - prior_actual_velocity) / dt, axis=1).tolist())
        commanded_accel_values.extend(np.linalg.norm((vel - old_vel) / dt, axis=1).tolist())
        prior_actual_velocity = actual_velocity
        turn_values.extend((np.abs(yaw_step) / dt).tolist())
        oscillation_count += ((np.sign(yaw_step) != np.sign(prior_turn)) &
                              (np.abs(yaw_step) > 0.018) & (np.abs(prior_turn) > 0.018)).astype(int)
        prior_turn = yaw_step
        _, final_pair_dist = _pairs(pos)
        clearance = final_pair_dist - radius[:, None] - radius[None, :]
        min_clearance = min(min_clearance, float(np.min(clearance)))
        overlaps = int(np.count_nonzero(np.triu(clearance < -0.005, 1)))
        overlap_pair_steps += overlaps
        peak_overlaps = max(peak_overlaps, overlaps)
        if phase == "red":
            inside = np.all(np.abs(pos) < cfg.crossing_half_extent_m - 0.1, axis=1)
            red_occupancy_steps += int(np.count_nonzero(inside))
            red_violations += int(np.count_nonzero(inside & (state != CROSS)))
        if (step + 1) % 90 == 0:
            progress = np.linalg.norm(pos - last_pos_3s, axis=1)
            stalled = (state == CROSS) & (gap > 2) & (progress < 0.45)
            stalled_run_seconds = np.where(stalled, stalled_run_seconds + 3, 0)
            deadlocked |= stalled_run_seconds >= 6
            last_pos_3s = pos.copy()
        if (step + 1) % stride == 0:
            capture((step + 1) // stride)
            for members in grouped.values():
                points = pos[members]
                if len(points) > 1:
                    delta_group = points[:, None, :] - points[None, :, :]
                    group_spacings.append(float(np.max(np.linalg.norm(delta_group, axis=2))))
        step_times.append(time.perf_counter() - tick_start)

    metrics = {
        "crossings_completed": int(completions.sum()),
        "agents_with_two_crossings": int(np.count_nonzero(completions >= 2)),
        "agents_with_no_crossing": int(np.count_nonzero(completions == 0)),
        "deadlock_agents": int(np.count_nonzero(deadlocked)),
        "oscillation_reversals": int(oscillation_count.sum()),
        "minimum_disc_clearance_m": round(min_clearance, 4),
        "overlapping_pair_steps": overlap_pair_steps,
        "preprojection_overlap_pair_steps": preprojection_pair_steps,
        "peak_overlapping_pairs": peak_overlaps,
        "projection_total_m": round(projection_m, 3),
        "projection_single_iteration_max_m": round(projection_step_max, 4),
        "root_boundary_violations": int(np.count_nonzero(np.abs(trace[:, :, :2]) > cfg.world_half_extent_m)),
        "rendered_floor_support": "unverified",
        "red_non_crossing_inside_samples": red_violations,
        "red_all_inside_agent_steps": red_occupancy_steps,
        "group_count": len(grouped),
        "group_max_spacing_p95_m": round(float(np.percentile(group_spacings, 95)), 3) if group_spacings else 0,
        "group_max_spacing_m": round(max(group_spacings), 3) if group_spacings else 0,
        "acceleration_p95_mps2": round(float(np.percentile(accel_values, 95)), 3),
        "acceleration_max_mps2": round(max(accel_values), 3),
        "commanded_acceleration_p95_mps2": round(float(np.percentile(commanded_accel_values, 95)), 3),
        "turn_rate_p95_radps": round(float(np.percentile(turn_values, 95)), 3),
        "solver_step_p95_ms": round(float(np.percentile(step_times, 95)) * 1000, 3),
        "solver_wall_s": round(time.perf_counter() - began, 3),
        "notes": "disc proxies only; body collisions and foot contact unverified",
    }
    return CrowdResult(cfg, agents, trace, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = simulate_crowd(CrowdConfig(count=args.count, seed=args.seed, duration_s=args.duration))
    result.write_json(args.output)
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
