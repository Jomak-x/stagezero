"""Review stored G1 motion ablations in a local browser, without inference.

Example:
    python experiments/review_motion.py --directory review/motion-ablation \
        --history-project .runtime/projects/source.stagezero.npz

Each ablation NPZ supplies 104 generated frames. With --history-project, the
first take's original 104 frames are displayed before the generated branch.
"""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import torch
import viser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preview import ROOT, load_recording  # noqa: E402
from ardy.viz.viser_utils import Character  # noqa: E402

FPS = 25
GENERATED_FRAMES = 104


def load_prefix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"]))
        if manifest.get("version") != 1 or manifest.get("fps") != FPS or not manifest.get("takes"):
            raise ValueError("History project is not a compatible G1 project")
        key = manifest["takes"][0]["key"]
        positions = data[f"{key}_positions"][:GENERATED_FRAMES].copy()
        rotations = data[f"{key}_rotations"][:GENERATED_FRAMES].copy()
    validate_pose(positions, rotations, "history prefix")
    return positions, rotations


def validate_pose(positions: np.ndarray, rotations: np.ndarray, label: str) -> None:
    if positions.shape != (GENERATED_FRAMES, 34, 3) or rotations.shape != (GENERATED_FRAMES, 34, 3, 3):
        raise ValueError(f"{label}: expected 104 frames of 34-joint G1 positions and rotations")
    if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
        raise ValueError(f"{label}: non-finite pose data")


def load_case(path: Path, prefix: tuple[np.ndarray, np.ndarray] | None) -> dict:
    with np.load(path, allow_pickle=False) as data:
        positions = data["positions"].copy()
        rotations = data["rotations"].copy()
        metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
    validate_pose(positions, rotations, path.name)
    if prefix is not None:
        positions = np.concatenate((prefix[0], positions))
        rotations = np.concatenate((prefix[1], rotations))
    return {"positions": positions, "rotations": rotations, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True, help="Directory of ablation NPZ files")
    parser.add_argument("--history-project", type=Path, help="Saved project providing the original 104-frame prefix")
    parser.add_argument("--port", type=int, default=2337, help="Local viewer port (default: 2337)")
    args = parser.parse_args()
    files = sorted(args.directory.glob("*.npz"))
    if not files:
        parser.error(f"No ablation NPZ files found in {args.directory}")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    prefix = load_prefix(args.history_project) if args.history_project else None
    motion = load_case(files[0], prefix)
    torch.set_num_threads(2)
    skeleton, _, _ = load_recording(ROOT / "assets/recorded_g1.csv")
    server = viser.ViserServer(host="127.0.0.1", port=args.port, label="ARDY Ablation Review")
    server.gui.configure_theme(dark_mode=True, control_layout="floating", show_logo=False, show_share_button=False)
    server.scene.set_up_direction("+y")
    server.scene.world_axes.visible = False
    server.scene.add_box("/floor", color=(28, 36, 46), dimensions=(200, 0.1, 200), position=(0, -0.1, 0), cast_shadow=False)
    character = Character("actor", server, skeleton, create_skeleton_mesh=False,
                          create_skinned_mesh=True, mesh_mode="g1_stl", show_foot_contacts=False)
    if not character.g1_mesh_rig.mesh_handles:
        raise RuntimeError("Official G1 mesh assets are unavailable")

    lock = threading.RLock()
    state = {"motion": motion, "frame": 0, "playing": False, "started": 0.0,
             "root": motion["positions"][0, 0].copy()}
    server.gui.add_markdown("# ARDY motion ablation review\n**Stored experimental motion · no generation**")
    chosen = server.gui.add_dropdown("Case", tuple(path.name for path in files), initial_value=files[0].name)
    info = server.gui.add_markdown("")
    play = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
    pause = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
    reset = server.gui.add_button("Reset view", icon=viser.Icon.ROTATE_2)
    follow = server.gui.add_checkbox("Follow actor", initial_value=True)
    slider = server.gui.add_slider("Frame", min=0, max=len(motion["positions"]) - 1, step=1, initial_value=0)
    status = server.gui.add_markdown("")

    def reset_camera(client) -> None:
        root = state["motion"]["positions"][0, 0]
        client.camera.position = (float(root[0] + 2.6), 1.9, float(root[2] + 3.4))
        client.camera.look_at = (float(root[0]), 0.75, float(root[2]))
        client.camera.up_direction = (0, 1, 0)
        client.camera.fov = np.deg2rad(42)

    @server.on_client_connect
    def connected(client) -> None:
        reset_camera(client)

    def update_status() -> None:
        frame = state["frame"]
        boundary = " · generated branch" if prefix is not None and frame >= GENERATED_FRAMES else " · original prefix" if prefix is not None else ""
        status.content = f"**{'Playing' if state['playing'] else 'Paused'}** · {frame / FPS:.2f} s · frame {frame}{boundary}"

    def show_frame(frame: int) -> None:
        frame = max(0, min(frame, len(state["motion"]["positions"]) - 1))
        if frame == state["frame"] and slider.value == frame:
            update_status()
            return
        root = state["motion"]["positions"][frame, 0].copy()
        delta = root - state["root"]
        delta[1] = 0
        with server.atomic():
            character.set_pose(torch.from_numpy(state["motion"]["positions"][frame]),
                               torch.from_numpy(state["motion"]["rotations"][frame]))
            if follow.value and np.any(delta):
                for client in server.get_clients().values():
                    # Viser's position setter translates look_at by the same delta.
                    client.camera.position = np.asarray(client.camera.position) + delta
        state["frame"] = frame
        state["root"] = root
        slider.value = frame
        update_status()

    def show_case(path: Path) -> None:
        state["playing"] = False
        state["motion"] = load_case(path, prefix)
        state["frame"] = -1
        state["root"] = state["motion"]["positions"][0, 0].copy()
        slider.max = len(state["motion"]["positions"]) - 1
        metadata = state["motion"]["metadata"]
        fields = ("prompt", "config", "seed", "initial_history_frames", "carry_history_frames", "cfg_weight")
        details = "  \n".join(f"**{key.replace('_', ' ').title()}:** {escape(str(metadata[key]))}" for key in fields if key in metadata)
        info.content = f"**File:** {escape(path.name)}  \n{details}  \n**Playback:** {len(state['motion']['positions'])} frames at {FPS} fps; stored arrays only."
        for client in server.get_clients().values():
            reset_camera(client)
        show_frame(0)

    @chosen.on_update
    def choose_case(_) -> None:
        with lock:
            show_case(args.directory / chosen.value)

    @slider.on_update
    def scrub(_) -> None:
        with lock:
            if int(slider.value) == state["frame"]:
                return  # programmatic playback update, not a user scrub
            state["playing"] = False
            show_frame(int(slider.value))

    @play.on_click
    def start(_) -> None:
        with lock:
            if state["frame"] == slider.max:
                show_frame(0)
            state["started"] = time.perf_counter() - state["frame"] / FPS
            state["playing"] = True
            update_status()

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

    show_case(files[0])
    print(f"Ablation viewer: http://127.0.0.1:{args.port}", flush=True)
    try:
        while True:
            with lock:
                if state["playing"]:
                    next_frame = min(int((time.perf_counter() - state["started"]) * FPS), slider.max)
                    if next_frame != state["frame"]:
                        show_frame(next_frame)
                    if next_frame == slider.max:
                        state["playing"] = False
                        update_status()
            time.sleep(1 / (FPS * 2))
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
