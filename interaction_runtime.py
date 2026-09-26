"""Isolated, model-native ARDY Core interaction generation.

This runtime generates one or two independent Core actors in chronological
40-frame windows. Scene waypoints become official ARDY constraints in the
window that contains them; generated poses are never patched afterward.
The trusted ``condition_hook`` is reserved for experiments with hand or
full-body constraint objects and is not exposed to HTTP clients.
"""

from __future__ import annotations

import io
import json
import math
from numbers import Integral, Real
import time
from typing import Callable

import numpy as np


MODEL_NAME = "ARDY-Core-RP-20FPS-Horizon40"
FPS = 20
HORIZON = 40
MAX_FRAMES = 240
FEATURES = 330
JOINTS = 27
TOKEN_FRAMES = 4
MAX_ACTORS = 2
MAX_WAYPOINTS = 24


class GenerationCancelled(Exception):
    """A request was cancelled before a model window began."""


def _integer(value, label: str, *, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
        raise ValueError(f"{label} must be an integer from {low} to {high}")
    return int(value)


def _finite(value, label: str, *, bound: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or abs(value) > bound:
        raise ValueError(f"{label} must be a finite number within ±{bound:g}")
    return float(value)


def _position(value, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be [x, z]")
    return [_finite(value[0], f"{label} x", bound=25), _finite(value[1], f"{label} z", bound=25)]


def validate_request(body: dict, *, feature_count: int = FEATURES) -> dict:
    """Normalize untrusted JSON without needing Torch, CUDA, or the model."""
    if not isinstance(body, dict) or set(body) != {"request_id", "frames", "actors"}:
        raise ValueError("Request requires request_id, frames, and actors only")
    request_id = body["request_id"]
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 100:
        raise ValueError("request_id must contain 1–100 characters")
    frames = _integer(body["frames"], "frames", low=HORIZON, high=MAX_FRAMES)
    if frames % HORIZON:
        raise ValueError("frames must be a multiple of 40")
    supplied = body["actors"]
    if not isinstance(supplied, list) or not 1 <= len(supplied) <= MAX_ACTORS:
        raise ValueError("actors must contain one or two actors")
    actors = []
    ids = set()
    for item in supplied:
        if not isinstance(item, dict) or not {"id", "prompt", "seed"} <= set(item) or set(item) - {
            "id", "prompt", "seed", "history", "root_targets", "initial_position_xz", "initial_yaw"
        }:
            raise ValueError("Each actor requires id, prompt, seed, and optional history/placement/root_targets")
        actor_id = item["id"]
        if not isinstance(actor_id, str) or not 1 <= len(actor_id) <= 64 or actor_id in ids:
            raise ValueError("Actor IDs must be unique strings of 1–64 characters")
        ids.add(actor_id)
        prompt = item["prompt"]
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 500:
            raise ValueError("Actor prompt must contain 1–500 characters")
        seed = _integer(item["seed"], "actor seed", low=0, high=2**32 - 1)
        history = item.get("history")
        if history is not None:
            try:
                history = np.asarray(history, dtype=np.float32)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError("Core history must contain finite numeric features") from exc
            if (history.ndim != 2 or history.shape[1] != feature_count
                    or not TOKEN_FRAMES <= len(history) <= HORIZON
                    or len(history) % TOKEN_FRAMES or not np.isfinite(history).all()):
                raise ValueError(f"Core history must be 4–40 frames of {feature_count} finite features, divisible by four")
            history = history.copy()
        targets = item.get("root_targets", [])
        if not isinstance(targets, list) or len(targets) > MAX_WAYPOINTS:
            raise ValueError(f"root_targets must contain at most {MAX_WAYPOINTS} targets")
        normalized_targets = []
        target_frames = set()
        for target in targets:
            if not isinstance(target, dict) or set(target) not in (
                    {"frame", "position_xz"}, {"frame", "position_xz", "heading"}):
                raise ValueError("A root target requires frame and position_xz, with optional heading")
            frame = _integer(target["frame"], "target frame", low=0, high=frames - 1)
            if frame in target_frames:
                raise ValueError("Each actor may specify one root target per frame")
            target_frames.add(frame)
            normalized = {"frame": frame, "position_xz": _position(target["position_xz"], "target position_xz")}
            if "heading" in target:
                normalized["heading"] = _finite(target["heading"], "heading (radians)", bound=math.pi)
            normalized_targets.append(normalized)
        initial_position = None if "initial_position_xz" not in item else _position(
            item["initial_position_xz"], "initial_position_xz")
        initial_yaw = None if "initial_yaw" not in item else _finite(
            item["initial_yaw"], "initial_yaw (radians)", bound=math.pi)
        actors.append({"id": actor_id, "prompt": prompt.strip(), "seed": seed,
                       "history": history, "initial_position_xz": initial_position,
                       "initial_yaw": initial_yaw,
                       "root_targets": sorted(normalized_targets, key=lambda x: x["frame"])})
    return {"request_id": request_id, "frames": frames, "actors": actors}


def targets_in_window(actor: dict, *, generated_offset: int, history_length: int) -> list[dict]:
    """Translate output frame indices to the current ARDY history+40 window."""
    if generated_offset < 0 or generated_offset % HORIZON:
        raise ValueError("generated_offset must be a nonnegative multiple of 40")
    if history_length < 0 or history_length > HORIZON or history_length % TOKEN_FRAMES:
        raise ValueError("history_length must be 0–40 frames and divisible by four")
    return [{**target, "window_frame": history_length + target["frame"] - generated_offset}
            for target in actor["root_targets"]
            if generated_offset <= target["frame"] < generated_offset + HORIZON]


def build_conditions(model, actors: list[dict], *, generated_offset: int,
                     history_length: int, device: str, condition_hook: Callable | None = None,
                     current_history=None):
    """Create official per-actor conditions, including trusted extra constraints."""
    import torch

    per_actor = []
    for actor_index, actor in enumerate(actors):
        constraints = []
        for target in targets_in_window(actor, generated_offset=generated_offset, history_length=history_length):
            from ardy.constraints import Root2DConstraintSet

            heading = target.get("heading")
            constraints.append(Root2DConstraintSet(
                model.skeleton,
                frame_indices=torch.tensor([target["window_frame"]], dtype=torch.long, device=device),
                root_2d=torch.tensor([target["position_xz"]], dtype=torch.float32, device=device),
                global_root_heading=None if heading is None else torch.tensor([heading], dtype=torch.float32, device=device),
            ))
        if condition_hook is not None:
            extras = condition_hook(model=model, actor=actor, generated_offset=generated_offset,
                                    history_length=history_length, device=device,
                                    current_history=None if current_history is None else current_history[actor_index])
            if not isinstance(extras, list):
                raise ValueError("Trusted condition hook must return a list of ARDY constraints")
            constraints.extend(extras)
        per_actor.append(constraints)
    if not any(per_actor):
        return None, None
    total = history_length + HORIZON
    lengths = torch.full((len(actors),), total, dtype=torch.long, device=device)
    observed, mask = model.motion_rep.create_conditions_from_constraints_batched(
        per_actor, lengths, to_normalize=True, device=device)
    expected = (len(actors), total, model.motion_rep.motion_rep_dim)
    if observed.shape != expected or mask.shape != expected:
        raise ValueError("ARDY returned invalid condition shapes")
    if not torch.isfinite(observed).all().item():
        raise ValueError("ARDY returned nonfinite conditions")
    if torch.count_nonzero(mask[:, :history_length]).item():
        raise ValueError("Constraints overlap immutable history")
    if not torch.count_nonzero(mask[:, history_length:]).item():
        raise ValueError("ARDY returned empty future conditions")
    return observed, mask


def _validate_output(motion: np.ndarray, positions: np.ndarray, rotations: np.ndarray,
                     actor_count: int, frames: int, features: int) -> None:
    expected = ((actor_count, frames, features), (actor_count, frames, JOINTS, 3),
                (actor_count, frames, JOINTS, 3, 3))
    for name, array, shape in zip(("motion", "positions", "rotations"), (motion, positions, rotations), expected):
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid Core {name}: expected {shape}, got {array.shape}")
    orthogonality = rotations @ np.swapaxes(rotations, -1, -2)
    if not np.allclose(orthogonality, np.eye(3), atol=.025) or not np.allclose(
            np.linalg.det(rotations), 1, atol=.025):
        raise ValueError("Core returned non-orthonormal rotations")


class InteractionRuntime:
    """Reusable model wrapper; caller serializes access to a single GPU model."""

    def __init__(self, model, *, device: str = "cuda", condition_hook: Callable | None = None):
        if (model.skeleton.name != "cskel27" or model.skeleton.nbjoints != JOINTS
                or model.motion_rep.fps != FPS or model.gen_horizon_len != HORIZON
                or model.num_frames_per_token != TOKEN_FRAMES
                or model.motion_rep.motion_rep_dim != FEATURES):
            raise ValueError("Loaded model is not ARDY Core 27 / 20 fps / Horizon40")
        self.model = model
        self.device = device
        self.condition_hook = condition_hook

    def _placed_history(self, actor: dict):
        """Move only a copy of supplied normalized history via official Core ops."""
        import torch

        raw = torch.from_numpy(actor["history"]).unsqueeze(0).to(self.device)
        rep = self.model.motion_rep
        if actor["initial_position_xz"] is None and actor["initial_yaw"] is None:
            return raw[0]
        world = rep.unnormalize(raw)
        if actor["initial_position_xz"] is None:
            origin = rep.get_root_pos(world)[0, 0, [0, 2]].clone()
        else:
            origin = torch.tensor(actor["initial_position_xz"], dtype=torch.float32, device=self.device)
        if actor["initial_yaw"] is not None:
            yaw = torch.tensor([actor["initial_yaw"]], dtype=torch.float32, device=self.device)
            world = rep.rotate_to(world, yaw)
        world = rep.translate_2d_to(world, origin.unsqueeze(0))
        return rep.normalize(world)[0]

    def generate(self, body: dict, *, is_cancelled: Callable[[str], bool] | None = None,
                 deadline: float | None = None,
                 on_window: Callable[[int, list[int], dict[str, np.ndarray]], None] | None = None
                 ) -> tuple[dict[str, np.ndarray], dict]:
        import torch
        from ardy.tools import seed_everything

        request = validate_request(body, feature_count=self.model.motion_rep.motion_rep_dim)
        actors = request["actors"]
        output = [None] * len(actors)
        actor_metrics = [None] * len(actors)
        conditioned_frames: list[list[int]] = [[] for _ in actors]
        grouped = {}
        # A common RNG seed and history length permit one genuine model batch.
        # Distinct actor seeds remain independent by using separate model calls.
        for index, actor in enumerate(actors):
            history_length = 0 if actor["history"] is None else len(actor["history"])
            grouped.setdefault((actor["seed"], history_length), []).append(index)

        def checkpoint():
            if is_cancelled is not None and is_cancelled(request["request_id"]):
                raise GenerationCancelled("Interaction request cancelled")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Interaction request exceeded its time limit")

        started = time.perf_counter()
        group_timings = []
        for (seed, initial_history_length), indices in grouped.items():
            checkpoint()
            selected = [actors[i] for i in indices]
            seed_everything(seed)
            with torch.inference_mode():
                encode_started = time.perf_counter()
                feat, text_mask = self.model._encode_text([a["prompt"] for a in selected])
                encoding_seconds = time.perf_counter() - encode_started
                current = None
                if initial_history_length:
                    current = torch.stack([self._placed_history(a) for a in selected], dim=0)
                chunks = []
                step_times = []
                for offset in range(0, request["frames"], HORIZON):
                    checkpoint()
                    history_length = 0 if current is None else current.shape[1]
                    observed, mask = build_conditions(
                        self.model, selected, generated_offset=offset,
                        history_length=history_length, device=self.device,
                        condition_hook=self.condition_hook, current_history=current)
                    if mask is not None:
                        for row, actor_index in enumerate(indices):
                            active = torch.any(mask[row, history_length:] != 0, dim=-1)
                            conditioned_frames[actor_index].extend(
                                (torch.nonzero(active).flatten() + offset).cpu().tolist())
                    step_started = time.perf_counter()
                    init_translation = init_heading = None
                    if current is None:
                        origins = [a["initial_position_xz"] if a["initial_position_xz"] is not None
                                   else [0.0, 0.0] for a in selected]
                        init_translation = torch.tensor(
                            [[xz[0], 0.0, xz[1]] for xz in origins],
                            dtype=torch.float32, device=self.device)
                        init_heading = torch.tensor(
                            [a["initial_yaw"] if a["initial_yaw"] is not None else 0.0 for a in selected],
                            dtype=torch.float32, device=self.device)
                    result = self.model.autoregressive_step(
                        num_frames=history_length + HORIZON,
                        num_denoising_steps=self.model.diffusion.num_base_steps,
                        motion_mask=mask, observed_motion=observed,
                        cfg_weight=(2.0, 2.0), text_feat=feat, text_pad_mask=text_mask,
                        init_history_sequence=current,
                        init_global_translation=init_translation,
                        init_first_heading_angle=init_heading,
                    )
                    if result.shape != (len(selected), history_length + HORIZON, FEATURES):
                        raise ValueError("ARDY returned an invalid generation window")
                    chunks.append(result[:, history_length:history_length + HORIZON])
                    current = result[:, -TOKEN_FRAMES:]
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    step_times.append(time.perf_counter() - step_started)
                    if on_window is not None:
                        chunk = chunks[-1]
                        # Decode the chronological prefix so velocity-integrated
                        # roots keep the same origin as the final full-clip decode.
                        decoded_prefix = self.model.motion_rep.inverse(
                            torch.cat(chunks, dim=1), is_normalized=True)
                        window = {
                            "motion": chunk.detach().cpu().numpy().astype(np.float32, copy=True),
                            "positions": decoded_prefix["posed_joints"][:, -HORIZON:].detach().cpu().numpy().astype(np.float32, copy=True),
                            "rotations": decoded_prefix["global_rot_mats"][:, -HORIZON:].detach().cpu().numpy().astype(np.float32, copy=True),
                        }
                        _validate_output(window["motion"], window["positions"], window["rotations"],
                                         len(selected), HORIZON, FEATURES)
                        on_window(offset // HORIZON, indices, window)
                checkpoint()
                motion_tensor = torch.cat(chunks, dim=1)
                decoded = self.model.motion_rep.inverse(motion_tensor, is_normalized=True)
                motion = motion_tensor.detach().cpu().numpy().astype(np.float32, copy=True)
                positions = decoded["posed_joints"].detach().cpu().numpy().astype(np.float32, copy=True)
                rotations = decoded["global_rot_mats"].detach().cpu().numpy().astype(np.float32, copy=True)
            _validate_output(motion, positions, rotations, len(selected), request["frames"], FEATURES)
            for row, index in enumerate(indices):
                actor = actors[index]
                output[index] = (motion[row], positions[row], rotations[row])
                target_metrics = []
                for target in actor["root_targets"]:
                    endpoint = positions[row, target["frame"], self.model.skeleton.root_idx, [0, 2]]
                    target_metrics.append({"frame": target["frame"],
                                           "position_xz": target["position_xz"],
                                           "actual_position_xz": endpoint.tolist(),
                                           "position_error_m": float(np.linalg.norm(endpoint - target["position_xz"])),
                                           "heading": target.get("heading")})
                actor_metrics[index] = {"id": actor["id"], "prompt": actor["prompt"], "seed": seed,
                                        "history_frames": initial_history_length,
                                        "initial_position_xz": actor["initial_position_xz"],
                                        "initial_yaw": actor["initial_yaw"],
                                        "conditioned_frames": sorted(set(conditioned_frames[index])),
                                        "root_targets": target_metrics}
            group_timings.append({"actor_indices": indices, "seed": seed, "batch_size": len(indices),
                                  "encoding_seconds": encoding_seconds, "step_seconds": step_times})
        checkpoint()
        arrays = {}
        for index, values in enumerate(output):
            for key, value in zip(("motion", "positions", "rotations"), values):
                arrays[f"actor_{index}_{key}"] = value
        metadata = {"request_id": request["request_id"], "model": MODEL_NAME, "fps": FPS,
                    "frames": request["frames"], "actors": actor_metrics,
                    "groups": group_timings, "generation_seconds": time.perf_counter() - started,
                    "native_conditions": any(conditioned_frames), "post_generation_pose_edits": False}
        return arrays, metadata


def encode_npz(arrays: dict[str, np.ndarray], metadata: dict) -> bytes:
    data = io.BytesIO()
    np.savez_compressed(data, **arrays, metadata=np.array(json.dumps(metadata, allow_nan=False)))
    return data.getvalue()
