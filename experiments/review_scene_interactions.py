"""Review saved, model-native ARDY Core scene trials on a local CPU viewer.

Example:
  /Users/jakob/Desktop/Shellhacks/.venv/bin/python -B \
      experiments/review_scene_interactions.py \
      --input /Users/jakob/Desktop/Shellhacks/.runtime/interaction-lab --port 2341

An NPZ contains actor_0_positions/rotations and optionally actor_1_*; its
``metadata`` JSON contains the scene, prompts, plans and measured metrics.
Playback does not call a model, change poses, or claim that visual proximity
proves interaction or collision-free passage.
"""

from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/ardy"))

FPS = 20
JOINTS = 27
MAX_FRAMES = 240
MODEL = "ARDY-Core-RP-20FPS-Horizon40"
COLORS = ((104, 217, 231), (244, 177, 109))


def _metadata(value: np.ndarray) -> dict:
    if value.shape != () or value.dtype.kind not in "US":
        raise ValueError("metadata must be a scalar JSON string")
    metadata = json.loads(value.item())
    if not isinstance(metadata, dict):
        raise ValueError("metadata must contain a JSON object")
    return metadata


def load_trial(path: Path, metadata_path: Path | None = None) -> tuple[list[dict], dict]:
    """Load only decoded Core poses and JSON; reject ambiguous or broken rigs."""
    with np.load(path, allow_pickle=False) as data:
        metadata = _metadata(data["metadata"]) if "metadata" in data else {}
        if any(key.startswith("actor_2_") for key in data.files):
            raise ValueError(f"{path.name}: viewer supports at most two actors")
        actors = []
        for index in range(2):
            position_key, rotation_key = f"actor_{index}_positions", f"actor_{index}_rotations"
            if position_key not in data and rotation_key not in data:
                if index == 0:
                    raise ValueError(f"{path.name}: no Core actor poses")
                break
            if position_key not in data or rotation_key not in data:
                raise ValueError(f"{path.name}: incomplete actor {index} pose")
            positions = data[position_key].copy()
            rotations = data[rotation_key].copy()
            frames = len(positions)
            if not 1 <= frames <= MAX_FRAMES or positions.shape != (frames, JOINTS, 3) or rotations.shape != (frames, JOINTS, 3, 3):
                raise ValueError(f"{path.name}: incompatible Core 27 pose arrays for actor {index}")
            if actors and frames != len(actors[0]["positions"]):
                raise ValueError(f"{path.name}: actor timelines differ")
            if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
                raise ValueError(f"{path.name}: non-finite actor pose")
            orthogonal = rotations @ np.swapaxes(rotations, -1, -2)
            if not np.allclose(orthogonal, np.eye(3), atol=.025) or not np.allclose(np.linalg.det(rotations), 1., atol=.025):
                raise ValueError(f"{path.name}: invalid Core joint rotations")
            actors.append({"positions": positions, "rotations": rotations})
    if metadata_path is not None:
        sidecar = json.loads(metadata_path.read_text())
        if not isinstance(sidecar, dict):
            raise ValueError("metadata sidecar must be a JSON object")
        metadata.update(sidecar)
    runtime = metadata.get("runtime", metadata)
    if not isinstance(runtime, dict) or runtime.get("model", MODEL) != MODEL:
        raise ValueError(f"{path.name}: metadata does not identify ARDY Core")
    if runtime.get("fps", FPS) != FPS or runtime.get("frames", len(actors[0]["positions"])) != len(actors[0]["positions"]):
        raise ValueError(f"{path.name}: timeline metadata disagrees with poses")
    return actors, metadata


def _scene_document(metadata: dict) -> dict | None:
    scene = metadata.get("scene")
    if scene is None:
        return None
    from interaction_scene import scene_objects
    scene_objects(scene)  # Validate the same dimensions used by route planning.
    return scene


def _draw_scene(server, scene: dict | None, affordances: dict | None) -> list:
    """Use StageZero procedural geometry; make custom asset limits explicit."""
    from object_scene import ObjectSceneLayer
    from scene_objects import KINDS

    handles = []
    if scene is None:
        return handles
    procedural, states = [], []
    for index, obj in enumerate(scene["objects"]):
        kind = obj["kind"]
        x, y, z = obj["position"]
        width, height, depth = obj["size"]
        if kind != "custom":
            item = {**obj, "color": obj.get("color", KINDS[kind]["color"])}
            procedural.append(item)
            states.append({"id": obj["id"], "position": obj["position"],
                           "color": item["color"], "active": False})
            continue
        # The referenced GLB is not part of this archive. Show declared bounds,
        # not an invented solid mesh or an assumed hole.
        yaw = math.radians(obj.get("yaw", 0.))
        q = (math.cos(yaw / 2), 0., math.sin(yaw / 2), 0.)
        base = f"/declared_custom/{index}"
        handles.append(server.scene.add_box(base + "/bounds", dimensions=(width, height, depth),
                                            color=(110, 150, 180), wireframe=True,
                                            wxyz=q, position=(x, y, z), cast_shadow=False))
        handles.append(server.scene.add_label(base + "/label",
                                              text=f"{obj['name']} · declared bounds",
                                              position=(x, y + height / 2 + .22, z)))
        passage = affordances.get(obj["id"]) if isinstance(affordances, dict) else None
        if isinstance(passage, dict) and passage.get("kind") == "passage" and passage.get("verified_open") is True:
            from interaction_scene import local_axes, passage_for, scene_objects
            source = next(item for item in scene_objects(scene) if item.id == obj["id"])
            opening = passage_for(source, affordances, actor_height_m=0.)
            px, pz = opening.center_xz
            py = float(passage["floor_y_m"]) + opening.height_m / 2
            pq_yaw = math.radians(opening.yaw_degrees)
            pq = (math.cos(pq_yaw / 2), 0., math.sin(pq_yaw / 2), 0.)
            handles.append(server.scene.add_box(base + "/declared_passage",
                                                dimensions=(opening.width_m, opening.height_m, opening.depth_m),
                                                color=(88, 229, 202), wireframe=True,
                                                wxyz=pq, position=(px, py, pz), cast_shadow=False))
            width_axis, _ = local_axes(opening.yaw_degrees)
            for side, sign in (("left", -1), ("right", 1)):
                offset = sign * (opening.width_m / 2 + .04)
                handles.append(server.scene.add_box(base + f"/{side}_proxy_post",
                                                    dimensions=(.08, opening.height_m, opening.depth_m),
                                                    color=(88, 229, 202), opacity=.38, wxyz=pq,
                                                    position=(px + width_axis[0] * offset, py,
                                                              pz + width_axis[1] * offset), cast_shadow=False))
            handles.append(server.scene.add_box(base + "/proxy_lintel",
                                                dimensions=(opening.width_m + .16, .08, opening.depth_m),
                                                color=(88, 229, 202), opacity=.38, wxyz=pq,
                                                position=(px, py + opening.height_m / 2 + .04, pz),
                                                cast_shadow=False))
            handles.append(server.scene.add_label(base + "/passage_label", text="declared passage proxy",
                                                  position=(px, py + opening.height_m / 2 + .1, pz)))
    if procedural:
        layer = ObjectSceneLayer(server)
        layer.update(procedural, states)
        for parts in layer.handles.values():
            handles.extend(part[0] for part in parts)
    return handles


def _details(path: Path, metadata: dict, count: int, frames: int) -> str:
    lines = [f"**Trial:** `{escape(path.name)}`  ",
             f"**Native Core 27:** {count} actor{'s' if count != 1 else ''} · {frames} frames · 20 fps  "]
    condition = metadata.get("condition")
    if condition is not None:
        lines.append(f"**Condition:** {escape(str(condition))}  ")
    if metadata.get("seed") is not None:
        lines.append(f"**Seed:** {escape(str(metadata['seed']))}  ")
    runtime = metadata.get("runtime", metadata)
    if isinstance(runtime, dict) and isinstance(runtime.get("generation_seconds"), (int, float)):
        lines.append(f"**Generation:** {runtime['generation_seconds']:.2f} s  ")
    prompts = []
    request = metadata.get("request")
    if isinstance(request, dict) and isinstance(request.get("actors"), list):
        prompts = [a.get("prompt", "") for a in request["actors"] if isinstance(a, dict)]
    if not prompts and isinstance(runtime, dict) and isinstance(runtime.get("actors"), list):
        prompts = [a.get("prompt", "") for a in runtime["actors"] if isinstance(a, dict)]
    for i, prompt in enumerate(prompts[:count]):
        lines.append(f"**{'Cyan' if i == 0 else 'Amber'} actor:** {escape(str(prompt))}  ")
    metrics = metadata.get("metrics")
    if isinstance(metrics, dict):
        for index, result in enumerate(metrics.get("actors", [])):
            gate = result.get("gate", {})
            endpoint = result.get("endpoint", {})
            collision = result.get("scene_collision", {})
            if gate:
                lines.append(f"**Actor {index + 1} gate:** {'crossed' if gate.get('traversed_proxy') else 'missed'}  ")
            if "endpoint_error_xz_m" in endpoint:
                lines.append(f"**Target error:** {endpoint['endpoint_error_xz_m']:.3f} m  ")
            if "total_collision_frames" in collision:
                lines.append(f"**Body overlap proxy:** {collision['total_collision_frames']} frames  ")
            step = result.get("continuity", {}).get("mean_joint_peak_step_m")
            if step is not None:
                lines.append(f"**Largest pose step:** {step:.3f} m/frame  ")
        pair = metrics.get("pair", {})
        if pair:
            separation = pair.get("separation", {})
            minimum = separation.get("min_root_separation_xz_m")
            if minimum is not None:
                lines.append(f"**Minimum root separation:** {minimum:.3f} m  ")
            lines.append(f"**Pair overlap proxy:** {separation.get('root_disc_overlap_proxy_frames', 0)} frames  ")
        lines.append("Full measurements are saved in the trial archive and report.")
    lines.append("\n*Scene objects are procedural or declared geometry. Poses are saved Core outputs; visual proximity alone does not verify contact or passage.*")
    return "\n".join(lines)


def serve(input_path: Path, port: int, metadata_path: Path | None = None) -> None:
    import torch
    import viser
    from ardy.skeleton import CoreSkeleton27
    from ardy.viz.viser_utils import Character

    paths = sorted(input_path.glob("*.npz")) if input_path.is_dir() else [input_path]
    if not paths:
        raise ValueError(f"No NPZ trials in {input_path}")
    if metadata_path is not None and len(paths) != 1:
        raise ValueError("--metadata can only accompany one NPZ file")
    actors, metadata = load_trial(paths[0], metadata_path)
    scene = _scene_document(metadata)
    torch.set_num_threads(2)
    skeleton = CoreSkeleton27().to("cpu")
    server = viser.ViserServer(host="127.0.0.1", port=port, label="Core scene interaction review")
    # Viser silently picks the next free port. Keep the requested local URL
    # truthful, especially when another review session owns 2341.
    if server.get_port() != port:
        server.stop()
        raise OSError(f"Local port {port} is already in use")
    server.gui.configure_theme(dark_mode=True, control_layout="floating", control_width="medium",
                               show_logo=False, show_share_button=False)
    server.scene.set_up_direction("+y")
    server.scene.world_axes.visible = False
    server.scene.add_box("/floor", color=(26, 34, 46), dimensions=(200, .1, 200),
                         position=(0, -.1, 0), cast_shadow=False)
    characters = []
    for index in range(2):
        character = Character(f"core_scene_actor_{index}", server, skeleton,
                              create_skeleton_mesh=False, create_skinned_mesh=True,
                              mesh_mode="core_skin", show_foot_contacts=False, dark_mode=True)
        character.skinned_mesh.color = COLORS[index]
        character.skinned_mesh.visible = index < len(actors)
        characters.append(character)

    server.gui.add_markdown("# Core scene interaction review\n**Saved model outputs · local 20 fps playback**")
    chosen = server.gui.add_dropdown("Trial", tuple(p.name for p in paths), initial_value=paths[0].name)
    details = server.gui.add_markdown("")
    play = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
    pause = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
    reset = server.gui.add_button("Reset view", icon=viser.Icon.ROTATE_2)
    slider = server.gui.add_slider("Frame", min=0, max=len(actors[0]["positions"]) - 1,
                                   step=1, initial_value=0)
    status = server.gui.add_markdown("")
    lock = threading.RLock()
    state = {"actors": actors, "metadata": metadata, "path": paths[0], "scene_handles": [],
             "frame": -1, "playing": False, "started": 0., "center": np.zeros(3), "distance": 5.}

    def camera_extent() -> None:
        roots = np.concatenate([a["positions"][:, 0, :] for a in state["actors"]], axis=0)
        if (doc := _scene_document(state["metadata"])) is not None:
            object_centers = np.asarray([o["position"] for o in doc["objects"]], dtype=float)
            if len(object_centers):
                roots = np.concatenate((roots, object_centers), axis=0)
        lower, upper = roots.min(axis=0), roots.max(axis=0)
        state["center"] = (lower + upper) / 2
        span = np.linalg.norm((upper - lower)[[0, 2]])
        state["distance"] = max(4.5, float(span) * 1.25 + 2.0)

    def reset_camera(client) -> None:
        center, distance = state["center"], state["distance"]
        client.camera.position = (float(center[0] + distance * .66), 2.1,
                                  float(center[2] + distance * .8))
        client.camera.look_at = (float(center[0]), .9, float(center[2]))
        client.camera.up_direction = (0., 1., 0.)
        client.camera.fov = np.deg2rad(48)

    @server.on_client_connect
    def connected(client) -> None:
        reset_camera(client)

    def show_frame(frame: int) -> None:
        frames = len(state["actors"][0]["positions"])
        frame = max(0, min(int(frame), frames - 1))
        if frame != state["frame"]:
            with server.atomic():
                for index, actor in enumerate(state["actors"]):
                    characters[index].set_pose(torch.from_numpy(actor["positions"][frame]),
                                               torch.from_numpy(actor["rotations"][frame]))
            state["frame"] = frame
        if slider.value != frame:
            slider.value = frame
        status.content = (f"**{'Playing' if state['playing'] else 'Paused'}** · "
                          f"frame {frame}/{frames - 1} · {frame / FPS:.2f} s")

    def show_trial(path: Path, override: Path | None = None) -> None:
        new_actors, new_metadata = load_trial(path, override)
        new_scene = _scene_document(new_metadata)
        state.update(actors=new_actors, metadata=new_metadata, path=path, frame=-1, playing=False)
        for handle in state["scene_handles"]:
            handle.remove()
        state["scene_handles"] = _draw_scene(server, new_scene, new_metadata.get("affordances"))
        for index, character in enumerate(characters):
            character.skinned_mesh.visible = index < len(new_actors)
        slider.max = len(new_actors[0]["positions"]) - 1
        details.content = _details(path, new_metadata, len(new_actors), len(new_actors[0]["positions"]))
        camera_extent()
        for client in server.get_clients().values():
            reset_camera(client)
        show_frame(0)

    @chosen.on_update
    def select(_) -> None:
        with lock:
            show_trial(next(p for p in paths if p.name == chosen.value))

    @slider.on_update
    def scrub(_) -> None:
        with lock:
            if int(slider.value) != state["frame"]:
                state["playing"] = False
                show_frame(slider.value)

    @play.on_click
    def start(_) -> None:
        with lock:
            if state["frame"] >= len(state["actors"][0]["positions"]) - 1:
                show_frame(0)
            state["started"] = time.perf_counter() - state["frame"] / FPS
            state["playing"] = True
            show_frame(state["frame"])

    @pause.on_click
    def stop(_) -> None:
        with lock:
            state["playing"] = False
            show_frame(state["frame"])

    @reset.on_click
    def reset_view(_) -> None:
        with lock:
            state["playing"] = False
            show_frame(0)
            for client in server.get_clients().values():
                reset_camera(client)

    show_trial(paths[0], metadata_path)
    print(f"Core scene interaction review: http://127.0.0.1:{port}/", flush=True)
    try:
        while True:
            with lock:
                if state["playing"]:
                    last = len(state["actors"][0]["positions"]) - 1
                    frame = min(int((time.perf_counter() - state["started"]) * FPS), last)
                    if frame != state["frame"]:
                        show_frame(frame)
                    if frame == last:
                        state["playing"] = False
                        show_frame(frame)
            time.sleep(1 / (FPS * 2))
    except KeyboardInterrupt:
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="One Core NPZ trial or a directory of trials")
    parser.add_argument("--metadata", type=Path, help="Optional JSON sidecar for a single NPZ")
    parser.add_argument("--port", type=int, default=2341, help="Loopback port (default: 2341)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    serve(args.input, args.port, args.metadata)


if __name__ == "__main__":
    main()
