"""Review the stored seed-1 batched G1 pair, without inference or scene effects.

Example:
    .venv/bin/python -B experiments/review_interactions.py

Both tracks are independent ARDY samples generated together in one batch. Their
shared 25 fps timeline is synchronized for visual review; this does not imply
that the model generated or enforced a person-to-person interaction.
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
FRAMES = 52
CASE = "seed1_batched"
DEFAULT_ARCHIVE = ROOT / ".runtime/motion-research/interaction-v2.npz"
DEFAULT_REPORT = ROOT / "review/motion-interaction-probe.json"


def load_pair(path: Path) -> list[dict[str, np.ndarray]]:
    actors = []
    with np.load(path, allow_pickle=False) as data:
        for i in range(2):
            prefix = f"{CASE}_actor{i}"
            positions = data[f"{prefix}_positions"].copy()
            rotations = data[f"{prefix}_rotations"].copy()
            if positions.shape != (FRAMES, 34, 3) or rotations.shape != (FRAMES, 34, 3, 3):
                raise ValueError(f"{prefix}: incompatible G1 pose arrays")
            if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
                raise ValueError(f"{prefix}: non-finite pose arrays")
            if not np.allclose(rotations @ np.swapaxes(rotations, -1, -2), np.eye(3), atol=0.02):
                raise ValueError(f"{prefix}: invalid joint rotations")
            actors.append({"positions": positions, "rotations": rotations})
    return actors


def load_metadata(path: Path) -> dict:
    report = json.loads(path.read_text())
    if report.get("model") != "ARDY-G1-RP-25FPS-Horizon52":
        raise ValueError("report has a different checkpoint")
    case = next((c for c in report.get("cases", []) if c.get("name") == CASE), None)
    if case is None or case.get("status") != "ok" or case.get("actor_count") != 2:
        raise ValueError(f"report has no successful {CASE} pair")
    return {"report": report, "case": case}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_ARCHIVE, help="Stored interaction probe archive")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Matching probe JSON report")
    parser.add_argument("--port", type=int, default=2339, help="Loopback review port (default: 2339)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    pair = load_pair(args.npz)
    metadata = load_metadata(args.report)
    report, case = metadata["report"], metadata["case"]
    torch.set_num_threads(2)
    skeleton, _, _ = load_recording(ROOT / "assets/recorded_g1.csv")

    server = viser.ViserServer(host="127.0.0.1", port=args.port, label="ARDY Independent Pair Review")
    server.gui.configure_theme(dark_mode=True, control_layout="floating", show_logo=False,
                               show_share_button=False)
    server.scene.set_up_direction("+y")
    server.scene.world_axes.visible = False
    server.scene.add_box("/floor", color=(25, 33, 45), dimensions=(200, 0.1, 200),
                         position=(0, -0.1, 0), cast_shadow=False)

    characters = []
    colors = ((93, 219, 217), (239, 181, 101))
    for i in range(2):
        character = Character(f"actor{i}", server, skeleton, create_skeleton_mesh=False,
                              create_skinned_mesh=True, mesh_mode="g1_stl", show_foot_contacts=False)
        if not character.g1_mesh_rig.mesh_handles:
            raise RuntimeError("Official G1 mesh assets are unavailable")
        for mesh in character.g1_mesh_rig.mesh_handles:
            mesh.color = colors[i]
        characters.append(character)

    all_roots = np.concatenate([actor["positions"][:, 0, :] for actor in pair], axis=0)
    center = (all_roots.min(axis=0) + all_roots.max(axis=0)) / 2
    planar_span = np.linalg.norm((all_roots.max(axis=0) - all_roots.min(axis=0))[[0, 2]])
    distance = max(4.7, float(planar_span) * 1.55 + 2.5)

    def reset_camera(client) -> None:
        client.camera.position = (float(center[0] + distance * 0.62), 2.15,
                                  float(center[2] + distance * 0.78))
        client.camera.look_at = (float(center[0]), 0.9, float(center[2]))
        client.camera.up_direction = (0, 1, 0)
        client.camera.fov = np.deg2rad(48)

    @server.on_client_connect
    def connected(client) -> None:
        reset_camera(client)

    prompts = case.get("prompts", ["", ""])
    if len(prompts) != 2:
        prompts = ["", ""]
    server.gui.add_markdown(
        "# ARDY pair review\n"
        "**Stored seed-1 batched samples · 25 fps · 52 frames**\n\n"
        "The two actors were generated independently in one model batch. "
        "Their poses were not conditioned on one another. No contact or handoff is represented."
    )
    server.gui.add_markdown(
        f"**Cyan actor:** {escape(str(prompts[0]))}  \n"
        f"**Amber actor:** {escape(str(prompts[1]))}"
    )
    server.gui.add_markdown(
        f"**Case:** `{CASE}` · **seed:** {escape(str(case.get('seed', '?')))}  \n"
        f"**Source frame:** {escape(str(report.get('source_frame', '?')))} · "
        f"**History:** {escape(str(report.get('history_frames', '?')))} frames  \n"
        f"**Batch runtime:** {case.get('total_seconds', 0):.3f} s · "
        f"**Peak allocated:** {case.get('gpu_peak_allocated_gib', 0):.3f} GiB"
    )
    play = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
    pause = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
    reset = server.gui.add_button("Reset view", icon=viser.Icon.ROTATE_2)
    slider = server.gui.add_slider("Frame", min=0, max=FRAMES - 1, step=1, initial_value=0)
    status = server.gui.add_markdown("")
    lock = threading.RLock()
    state = {"frame": -1, "playing": False, "started": 0.0}

    def update_status() -> None:
        frame = state["frame"]
        status.content = (f"**{'Playing' if state['playing'] else 'Paused'}** · "
                          f"frame {frame}/{FRAMES - 1} · {frame / FPS:.2f} s")

    def show_frame(frame: int) -> None:
        frame = max(0, min(frame, FRAMES - 1))
        if frame != state["frame"]:
            with server.atomic():
                for i in range(2):
                    characters[i].set_pose(torch.from_numpy(pair[i]["positions"][frame]),
                                           torch.from_numpy(pair[i]["rotations"][frame]))
            state["frame"] = frame
        if slider.value != frame:
            slider.value = frame
        update_status()

    @slider.on_update
    def scrub(_) -> None:
        with lock:
            if int(slider.value) == state["frame"]:
                return
            state["playing"] = False
            show_frame(int(slider.value))

    @play.on_click
    def start(_) -> None:
        with lock:
            if state["frame"] == FRAMES - 1:
                show_frame(0)
            state["started"] = time.perf_counter() - state["frame"] / FPS
            state["playing"] = True
            update_status()

    @pause.on_click
    def stop(_) -> None:
        with lock:
            state["playing"] = False
            update_status()

    @reset.on_click
    def reset_view(_) -> None:
        with lock:
            state["playing"] = False
            show_frame(0)
            for client in server.get_clients().values():
                reset_camera(client)

    show_frame(0)
    print(f"Interaction review: http://127.0.0.1:{args.port}/", flush=True)
    try:
        while True:
            with lock:
                if state["playing"]:
                    frame = min(int((time.perf_counter() - state["started"]) * FPS), FRAMES - 1)
                    if frame != state["frame"]:
                        show_frame(frame)
                    if frame == FRAMES - 1:
                        state["playing"] = False
                        update_status()
            time.sleep(1 / (FPS * 2))
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
