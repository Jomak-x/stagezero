#!/usr/bin/env python3
"""Check bounded AABB overlap samples against captured city and CoreSkin meshes.

This reads the actual logged 60 Hz poses. It does not replay, repair, or change
the motion controller. Sampled frames are selected from the final run audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ardy"))

from ardy.skeleton import CoreSkeleton27
from ardy.viz.core_skin import CoreSkin
from swing_scene import load_swing_scene
from experiments.verify_swing_run import box_depths, segment_clearance_samples


CASES = (
    (2100, "mj", "city-4", "sustained pre-pickup rooftop stance"),
    (2173, "mj", "city-4", "worst localized city-4 proxy overlap"),
    (5278, "spider", "city-10", "worst localized full-body proxy overlap"),
)


def _city_mesh(scene, building_id):
    vertices, faces, offset = [], [], 0
    for part in scene["meshes"]:
        if part["object_id"] != building_id:
            continue
        points = np.asarray(part["positions"], dtype=np.float64)
        triangles = np.asarray(part["indices"], dtype=np.int64).reshape(-1, 3)
        vertices.append(points)
        faces.append(triangles + offset)
        offset += len(points)
    return trimesh.Trimesh(vertices=np.concatenate(vertices),
                           faces=np.concatenate(faces), process=False)


def _logged_frames(path, frame_ids):
    required = set(frame_ids)
    found = {}
    with path.open() as stream:
        for line in stream:
            if not required:
                break
            # The first field is frame; avoid parsing the other 8,780 large rows.
            prefix = line[:40]
            if not any(f'"frame": {frame}' in prefix or f'"frame":{frame}' in prefix
                       for frame in required):
                continue
            state = json.loads(line)
            frame = int(state["frame"])
            if frame in required:
                found[frame] = state
                required.remove(frame)
    if required:
        raise ValueError(f"recording lacks frames {sorted(required)}")
    return found


def check(run_dir):
    scene = load_swing_scene()
    states = _logged_frames(run_dir / "frames.jsonl", (case[0] for case in CASES))
    report = json.loads((run_dir / "recorded-report.json").read_text())
    if report.get("scene_provenance", {}).get("sha256") != scene["provenance"]["sha256"]:
        raise ValueError("logged scene provenance differs from generated city snapshot")
    meshes = {bid: _city_mesh(scene, bid) for bid in {case[2] for case in CASES}}
    bounds = {box["id"]: box for box in scene["buildings"]}
    skin = CoreSkin(CoreSkeleton27())
    results = []
    for frame, actor_name, building_id, reason in CASES:
        state = states[frame]
        actor_index = 0 if actor_name == "spider" else 1
        actor = state["actors"][actor_index]
        positions = np.asarray([row["positions"] for row in state["actors"]], dtype=float)
        samples, radii, segments = segment_clearance_samples(positions)
        start, stop = actor_index * 130, (actor_index + 1) * 130
        proxy_depths = box_depths(samples[start:stop], radii[start:stop], bounds[building_id])
        ix = int(np.argmax(proxy_depths))
        point = samples[start + ix]
        mesh = meshes[building_id]
        sample_nearest, sample_distance, sample_triangle = trimesh.proximity.closest_point(
            mesh, point[None])
        with torch.inference_mode():
            body = skin.skin(torch.as_tensor(actor["rotations"], dtype=torch.float32)[None],
                             torch.as_tensor(actor["positions"], dtype=torch.float32)[None],
                             rot_is_global=True).cpu().numpy()[0]
        box = bounds[building_id]
        lo, hi = np.asarray(box["min"]) - .35, np.asarray(box["max"]) + .35
        nearby = np.flatnonzero(np.all((body >= lo) & (body <= hi), axis=1))
        if not len(nearby):
            raise ValueError(f"no CoreSkin vertices near {building_id} at frame {frame}")
        closest, distances, triangles = trimesh.proximity.closest_point(mesh, body[nearby])
        closest_index = int(np.argmin(distances))
        body_min_y = float(body[:, 1].min())
        building_max_y = float(mesh.vertices[:, 1].max())
        gap = body_min_y - building_max_y
        results.append({
            "frame": frame, "time_seconds": state["time"], "phase": state["phase"],
            "actor": actor_name, "building": building_id, "reason": reason,
            "proxy": {"segment": int(segments[start + ix]), "sample_point": point.tolist(),
                      "radius_m": float(radii[start + ix]),
                      "aabb_overlap_m": float(max(0, proxy_depths[ix]))},
            "source_mesh_at_proxy_sample": {
                "triangle_distance_m": float(sample_distance[0]),
                "sphere_surface_clearance_m": float(sample_distance[0] - radii[start + ix]),
                "closest_triangle": int(sample_triangle[0]),
                "closest_point": sample_nearest[0].tolist(),
            },
            "core_skin_vs_source_mesh": {
                "checked_nearby_vertices": int(len(nearby)),
                "nearest_vertex_id": int(nearby[closest_index]),
                "nearest_vertex": body[nearby[closest_index]].tolist(),
                "nearest_triangle": int(triangles[closest_index]),
                "nearest_point": closest[closest_index].tolist(),
                "vertex_surface_clearance_m": float(distances[closest_index]),
                "all_body_vertices_above_all_building_vertices_m": gap,
                "triangle_intersection_impossible_by_vertical_separation": bool(gap > 0),
            },
        })
    return {
        "run_id": report["run_id"],
        "source_scene_sha256": scene["provenance"]["sha256"],
        "recorded_report_sha256": hashlib.sha256((run_dir / "recorded-report.json").read_bytes()).hexdigest(),
        "method": "Official CoreSkin skinned from exact logged world joint poses; nearest surfaces on exact generated city triangles. Positive whole-body min-Y minus building max-Y proves no triangle intersection for that frame and building. AABB proxy spheres are reported alongside; final roof contact is intentionally separate.",
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.run)
    body = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body)
    else:
        print(body)


if __name__ == "__main__":
    main()
