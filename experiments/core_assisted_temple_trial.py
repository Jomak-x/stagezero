"""Generate and archive a native Core terrain route for offline presentation.

This experiment has no service/network client and performs no pose assistance.
Every returned Core horizon is checkpointed before route evaluation completes,
so a later terrain gate or presentation step cannot require repeating GPU work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _embedding_key(prompt: str) -> str:
    """Return the explicit cached-prompt class used by this bounded trial."""
    text = prompt.casefold()
    if re.search(r"\b(walks?|walking|cross(?:es|ing)?|climbs?|climbing|ascends?|descends?|"
                 r"enters?|entering|moves?|moving|approach(?:es|ing)?|through)\b", text):
        return "walk"
    if re.search(r"\b(stands?|standing|still|turns?|turning|holds?|pauses?|relaxed)\b", text):
        return "stand"
    raise ValueError(f"No cached stand/walk embedding mapping for Core prompt: {prompt!r}")


class _DirectCoreClient:
    """Adapt InteractionRuntime to terrain_assisted_session's client.wait API."""

    def __init__(self, runtime, output: Path, prompt_mapping: dict[str, str], seed: int):
        self.runtime = runtime
        self.output = output
        self.prompt_mapping = prompt_mapping
        self.seed = int(seed)
        self.window_index = 0

    def wait(self, body, *, cancelled=lambda: False):
        from realtime_clip import CanonicalClip

        cancelled = cancelled or (lambda: False)
        if cancelled():
            raise RuntimeError("Generation cancelled before Core request")
        request_id = str(body.get("request_id", "request"))
        actors = []
        for actor_id in body["actor_ids"]:
            prompt = body.get("actor_prompts", {}).get(actor_id, body["prompt"])
            self.prompt_mapping[prompt] = _embedding_key(prompt)
            actor = {"id": actor_id, "prompt": prompt, "seed": self.seed,
                     "root_targets": body.get("root_targets", {}).get(actor_id, [])}
            placement = body.get("initial_placements", {}).get(actor_id, {})
            if "position_xz" in placement:
                actor["initial_position_xz"] = placement["position_xz"]
            if "yaw" in placement:
                actor["initial_yaw"] = placement["yaw"]
            if actor_id in body.get("coordinate_frames_y", {}):
                actor["coordinate_frame_y"] = body["coordinate_frames_y"][actor_id]
            if body.get("history") is not None:
                history = body["history"]
                if "native_features" not in history:
                    raise ValueError("Direct Core terrain continuation requires exact native history features")
                history_array = np.asarray(history["native_features"], dtype=np.float32)
                actor["history"] = history_array[len(actors)].tolist()
            actors.append(actor)
        core_body = {"request_id": request_id, "frames": int(body["frames"]), "actors": actors}

        def save_window(index, selected_indices, data):
            if cancelled():
                raise RuntimeError("Generation cancelled after Core window")
            target = self.output / "native_windows" / f"window-{self.window_index:03d}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, motion=data["motion"], positions=data["positions"],
                                rotations=data["rotations"])
            sidecar = {
                "window_index": self.window_index,
                "request_window_index": int(index),
                "request_id": request_id,
                "selected_actor_indices": [int(x) for x in selected_indices],
                "frames": int(data["motion"].shape[1]),
                "archive": str(target),
                "native_features_preserved": True,
            }
            (target.with_suffix(".json")).write_text(json.dumps(sidecar, indent=2) + "\n")
            self.window_index += 1

        arrays, metadata = self.runtime.generate(core_body, is_cancelled=lambda _rid: cancelled(),
                                                 on_window=save_window)
        ids = tuple(actor["id"] for actor in core_body["actors"])
        clip = CanonicalClip(
            np.stack([arrays[f"actor_{i}_positions"] for i in range(len(ids))]),
            np.stack([arrays[f"actor_{i}_rotations"] for i in range(len(ids))]),
            20, ids, "ardy_core", metadata,
            np.stack([arrays[f"actor_{i}_motion"] for i in range(len(ids))]),
        )
        return [clip]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--commands", required=True,
                        help="Ordered spatial commands, e.g. 'ascend stairs then cross bridge'")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--actor-id", default="actor_1")
    parser.add_argument("--start-x", type=float, default=0.0)
    parser.add_argument("--start-z", type=float, default=0.0)
    parser.add_argument("--start-yaw", type=float, default=float(np.pi))
    parser.add_argument("--native-only", action="store_true",
                        help="Generate/archive native Core motion only; this is the default behavior")
    args = parser.parse_args()

    import torch
    from ardy.model import load_model
    from ardy.tools import seed_everything
    from interaction_runtime import InteractionRuntime
    from terrain_assisted_session import generate_native_terrain_commands, save_native_terrain_result

    args.output.mkdir(parents=True, exist_ok=True)
    scene = json.loads(args.scene.read_text())
    (args.output / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    embedding_paths = {key: args.embeddings / f"{key}_embedding.pt" for key in ("stand", "walk")}
    for path in [*embedding_paths.values(), args.scene]:
        if not path.is_file():
            raise FileNotFoundError(path)

    torch.set_num_threads(4)
    model = load_model("ARDY-Core-RP-20FPS-Horizon40", device="cuda",
                       checkpoints_dir=str(args.checkpoints), text_encoder_mode="local",
                       text_encoder=False)
    embeddings = {
        key: tuple(tensor.to("cuda") for tensor in torch.load(path, weights_only=True))
        for key, path in embedding_paths.items()
    }
    model.text_encoder = None

    def encode_cached(prompts):
        keys = [_embedding_key(prompt) for prompt in prompts]
        return tuple(torch.cat([embeddings[key][part] for key in keys], dim=0)
                     for part in range(2))

    model._encode_text = encode_cached
    runtime = InteractionRuntime(model, device="cuda")
    mapping = {
        "walk": "cached walk_embedding.pt for supported walk/cross/climb/ascend/descend/enter/move/approach verb forms, or through; walking class takes precedence",
        "stand": "cached stand_embedding.pt for supported stand/turn/hold/pause verb forms, still, or relaxed (unless a walk-class term also occurs)",
        "fallback": "none; unmatched prompts fail closed",
    }
    observed_prompt_mapping: dict[str, str] = {}
    manifest = {
        "status": "loading", "provenance": "native Core terrain generation; presentation not attempted",
        "commands": args.commands, "scene": str(args.scene),
        "scene_sha256": _sha256(args.scene), "checkpoints": str(args.checkpoints),
        "embedding_sources": {key: {"path": str(path), "sha256": _sha256(path)}
                              for key, path in embedding_paths.items()},
        "cached_embedding_mapping": mapping,
        "seed": args.seed, "native_only": True,
        "native_archive": "native_route.npz", "window_directory": "native_windows",
    }
    manifest_path = args.output / "manifest.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    save_manifest()
    seed_everything(args.seed)
    client = _DirectCoreClient(runtime, args.output, observed_prompt_mapping, args.seed)
    started = time.monotonic()
    try:
        result = generate_native_terrain_commands(
            scene, args.commands, actor_ids=(args.actor_id,), actor_id=args.actor_id,
            initial_placements={args.actor_id: {"position_xz": [args.start_x, args.start_z],
                                                "yaw": args.start_yaw}},
            client=client,
        )
        clip = result.native_clip
        save_native_terrain_result(result, args.output / "native_terrain.npz")
        np.savez_compressed(args.output / "native_route.npz", native_features=clip.native_features,
                            positions=clip.positions, rotations=clip.rotations)
        manifest.update(
            status="native_generation_complete", duration_seconds=time.monotonic() - started,
            frames=clip.frames, fps=clip.fps, actor_ids=list(clip.actor_ids),
            action_spans=[list(span) for span in result.action_spans],
            routes=list(result.routes), measurements=list(result.measurements),
            reaction_states=list(result.reaction_states),
            observed_prompt_embedding_mapping=observed_prompt_mapping,
            committed_prefix_frames=(0 if result.committed_prefix is None else
                                     result.committed_prefix.frames),
            native_archive_sha256=_sha256(args.output / "native_route.npz"),
            presentation_status="not_run",
        )
        (args.output / "scene_evaluated.json").write_text(
            json.dumps(result.scene, indent=2, allow_nan=False) + "\n")
        save_manifest()
        print(json.dumps({"status": manifest["status"], "frames": clip.frames,
                          "action_spans": manifest["action_spans"],
                          "native_archive": str(args.output / "native_route.npz")}), flush=True)
        return 0
    except Exception as exc:
        saved = sorted((args.output / "native_windows").glob("window-*.npz"))
        if saved:
            windows = [np.load(path, allow_pickle=False) for path in saved]
            np.savez_compressed(
                args.output / "native_partial.npz",
                native_features=np.concatenate([window["motion"] for window in windows], axis=1),
                positions=np.concatenate([window["positions"] for window in windows], axis=1),
                rotations=np.concatenate([window["rotations"] for window in windows], axis=1),
            )
            for window in windows:
                window.close()
        manifest.update(status="native_generation_failed", detail=str(exc),
                        duration_seconds=time.monotonic() - started,
                        saved_native_windows=client.window_index,
                        partial_native_archive=("native_partial.npz" if saved else None),
                        partial_native_frames=(client.window_index * 40),
                        observed_prompt_embedding_mapping=observed_prompt_mapping,
                        presentation_status="not_run")
        save_manifest()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
