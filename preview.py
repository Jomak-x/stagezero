"""StageZero milestone 1: recorded playback, with no inference dependencies.

CSV conversion adapted from NVIDIA ARDY scripts/interactive_demo/motion_io.py
(Apache-2.0); character rendering is reused directly from the pinned source.
"""
from pathlib import Path
import sys
import time
import threading
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor/ardy"))

import numpy as np
import torch
import trimesh
import viser
from scipy.spatial.transform import Rotation
from ardy.assets import skeleton_asset_path
from ardy.skeleton import G1Skeleton34
from ardy.viz.viser_utils import Character

FPS = 60  # The source is 120 fps; use every second recorded frame.


def load_recording(path):
    skeleton = G1Skeleton34()
    with open(path) as f:
        columns = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)[::2, 1:]
    if data.ndim != 2 or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("Recording must contain at least two finite motion frames")
    basis = Rotation.from_euler("x", -90, degrees=True) * Rotation.from_euler("z", -90, degrees=True)
    root = basis.apply(data[:, :3]) * 0.01  # centimeters -> meters
    root[:, [0, 2]] -= root[0, [0, 2]]
    rotations = np.tile(np.eye(3), (len(data), skeleton.nbjoints, 1, 1))
    rotations[:, 0] = (basis * Rotation.from_euler("xyz", data[:, 3:6], degrees=True) * basis.inv()).as_matrix()
    xml = ET.parse(skeleton_asset_path("g1skel34", "xml", "g1.xml"))
    defaults = {e.get("class"): e.find("joint").get("axis") for e in xml.findall(".//default") if e.get("class") and e.find("joint") is not None}
    parents = {child: parent for parent in xml.iter() for child in parent}
    for joint in xml.find("worldbody").findall(".//joint"):
        name = joint.get("name").replace("_joint", "_skel")
        if name not in skeleton.bone_index:
            continue
        axis = np.fromstring(joint.get("axis") or defaults[joint.get("class")], sep=" ")
        column = columns.index(joint.get("name") + "_dof") - 1
        r = Rotation.from_rotvec(np.deg2rad(data[:, column, None]) * axis)
        if "quat" in parents[joint].attrib:
            r = Rotation.from_quat(np.fromstring(parents[joint].get("quat"), sep=" "), scalar_first=True) * r
        rotations[:, skeleton.bone_index[name]] = (basis * r * basis.inv()).as_matrix()
    with torch.inference_mode():
        global_rots, positions, _ = skeleton.fk(torch.tensor(rotations, dtype=torch.float32), torch.tensor(root, dtype=torch.float32))
    return skeleton, positions, global_rots


def main():
    torch.set_num_threads(2)
    skeleton, positions, rotations = load_recording(ROOT / "assets/recorded_g1.csv")
    server = viser.ViserServer(host="127.0.0.1", port=2334, label="StageZero")
    server.gui.configure_theme(dark_mode=True, control_layout="floating", control_width="medium", show_logo=False, show_share_button=False, brand_color=(72, 202, 183))
    server.scene.set_up_direction("+y")
    server.scene.world_axes.visible = False
    server.scene.configure_environment_map(None)
    server.scene.configure_default_lights(enabled=True, cast_shadow=True)
    server.scene.add_light_ambient("/fill", color=(191, 215, 239), intensity=0.45)
    # A quiet floor and low circular platform give scale without distracting scenery.
    server.scene.add_box("/floor", color=(24, 31, 42), dimensions=(200, 0.1, 200), position=(0, -0.17, 0), cast_shadow=False)
    stage = trimesh.creation.cylinder(radius=2.0, height=0.12, sections=96)
    stage.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, (1, 0, 0)))
    server.scene.add_mesh_simple("/stage", vertices=stage.vertices, faces=stage.faces, color=(68, 83, 99), position=(0, -0.06, 0), flat_shading=False)
    character = Character("actor", server, skeleton, create_skeleton_mesh=False, create_skinned_mesh=True, mesh_mode="g1_stl", show_foot_contacts=False)
    if not character.g1_mesh_rig.mesh_handles:
        raise RuntimeError("Supplied G1 meshes are missing; cannot show the preview")
    for mesh in character.g1_mesh_rig.mesh_handles:
        mesh.color = (202, 219, 222)
    center = positions[:, 0].mean(0).numpy()
    camera_target = (float(center[0]), 0.75, float(center[2]))
    camera_position = (float(center[0]) + 2.6, 1.9, float(center[2]) + 3.4)

    def reset_camera(client):
        client.camera.position = camera_position
        client.camera.look_at = camera_target
        client.camera.up_direction = (0, 1, 0)
        client.camera.fov = np.deg2rad(42)

    @server.on_client_connect
    def connected(client):
        reset_camera(client)

    server.gui.add_markdown("# StageZero\n**01 / Visual preview**\n\nOne actor. One small stage.")
    server.gui.add_markdown("**RECORDED PLAYBACK**\n\nNot live AI generation.\n\n**Shadow boxing** · 12 seconds\nBONES SEED motion capture, retargeted to the Unitree G1 humanoid.")
    play = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
    pause = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
    reset = server.gui.add_button("Reset", icon=viser.Icon.ROTATE_2)
    status = server.gui.add_markdown("")
    server.gui.add_markdown("**Camera**\n\nDrag to orbit · Right-drag to pan · Scroll to zoom\n\nReset returns to the first frame and initial camera.")
    server.gui.add_markdown("[Motion source](https://huggingface.co/datasets/bones-studio/seed) · [ARDY renderer](https://github.com/nv-tlabs/ardy)\n\nMilestone 1 · Local playback only")
    lock = threading.RLock()
    state = {"frame": 0, "playing": False, "started": 0.0}
    last_status = ""

    def update_status():
        nonlocal last_status
        at_end = state["frame"] == len(positions) - 1
        label = "Playing" if state["playing"] else ("Finished" if at_end else "Paused")
        content = f"**{label}** · {state['frame'] / FPS:04.1f} / {(len(positions) - 1) / FPS:.1f} s"
        play.disabled = state["playing"]
        pause.disabled = not state["playing"]
        if content != last_status:
            status.content = content
            last_status = content

    def show_frame(frame):
        state["frame"] = frame
        with server.atomic():
            character.set_pose(positions[frame], rotations[frame])

    @play.on_click
    def play_clicked(_):
        with lock:
            if state["frame"] == len(positions) - 1:
                show_frame(0)
            state["started"] = time.perf_counter() - state["frame"] / FPS
            state["playing"] = True
            update_status()

    @pause.on_click
    def pause_clicked(_):
        with lock:
            state["playing"] = False
            update_status()

    @reset.on_click
    def reset_clicked(_):
        with lock:
            state["playing"] = False
            show_frame(0)
            for client in server.get_clients().values():
                reset_camera(client)
            update_status()

    show_frame(0)
    update_status()
    print(f"StageZero ready: http://localhost:2334 — {len(positions)} recorded frames at {FPS} fps", flush=True)
    try:
        while True:
            with lock:
                if state["playing"]:
                    frame = min(int((time.perf_counter() - state["started"]) * FPS), len(positions) - 1)
                    if frame != state["frame"]:
                        show_frame(frame)
                    if frame == len(positions) - 1:
                        state["playing"] = False
                    update_status()
            time.sleep(1 / (FPS * 2))
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
