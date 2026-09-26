"""Exact generated-market geometry for the grounded two-person prototype.

The source scene and geometry compiler are captured snapshots. Mesh fitting,
size, yaw, and translation match ObjectSceneLayer._build_custom in the main app.
No geometry, props, or background surfaces are invented here.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from swing_scene import _colored_meshes, _fit, _world_vertices


SOURCE_PATH = Path(__file__).with_name("grounded_assets") / "market-generated.json"
SOURCE_SHA256 = "35d841555e0c8fb822af77a5da288b24d963ac5d901afc71763be952023d6e8d"
GENERATION_METADATA_SHA256 = "76370ab0014959c23cb91891949e878a8ee484b6ee80a5feeeae8c8c348aa185"


def _captured_scene():
    source = SOURCE_PATH.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("generated market scene snapshot changed")
    scene = json.loads(source)
    if scene.get("version") != 3 or len(scene.get("objects", [])) != 31 or len(scene.get("assets", [])) != 13:
        raise ValueError("captured market scene has an unexpected schema")
    return scene


@lru_cache(maxsize=1)
def scene_payload():
    """Return original colored meshes and measured ground/obstacle bounds.

    The clear central stage contains the origin and is free of all scene props.
    Actor hip starts are hints; the native motion can remain centered at origin.
    """
    scene = _captured_scene()
    assets = {item["id"]: item for item in scene["assets"]}
    compiled = {identifier: _fit(asset) for identifier, asset in assets.items()}
    meshes, props = [], []
    ground = None
    for obj in scene["objects"]:
        if obj["kind"] != "custom":
            raise ValueError(f"unexpected non-custom market object {obj['id']}")
        if obj["asset"] not in compiled:
            raise ValueError(f"market object {obj['id']} references missing generated asset")
        vertices, faces, colors, lower, upper = compiled[obj["asset"]]
        world = _world_vertices(vertices, lower, upper, obj)
        meshes.extend(_colored_meshes(obj, world, faces, colors))
        bounds = {
            "id": obj["id"], "name": obj["name"], "asset": obj["asset"],
            "min": world.min(axis=0).tolist(), "max": world.max(axis=0).tolist(),
        }
        if obj["id"] == "set-0":
            ground = bounds
        else:
            props.append(bounds)
    if ground is None:
        raise ValueError("captured market has no paving")
    floor_y = float(ground["max"][1])
    stage = {"min": [-2.5, floor_y, -9.5], "max": [2.5, floor_y, 1.5]}
    for prop in props:
        intersects = (prop["min"][0] < stage["max"][0] and prop["max"][0] > stage["min"][0]
                      and prop["min"][2] < stage["max"][2] and prop["max"][2] > stage["min"][2])
        if intersects:
            raise ValueError(f"central actor stage intersects market prop {prop['id']}")
    source = {
        "path": "review/scene-scale/market-generated.json", "sha256": SOURCE_SHA256,
        "generation_metadata_path": "review/scene-scale/market/metadata.json",
        "generation_metadata_sha256": GENERATION_METADATA_SHA256,
        "asset_model": "gpt-5-6-sol", "generated_assets": 8,
        "layout": "captured deterministic market-set composition with model-generated props",
    }
    return {
        "name": scene["name"], "lighting": scene["lighting"],
        "meshes": meshes, "ground_y": floor_y, "floor_bounds": ground,
        "clear_stage": stage, "props": props,
        "actor_starts": [[-1.4, floor_y + .96, .5], [1.4, floor_y + .96, .5]],
        "meeting": [0., floor_y + .96, -4.],
        "camera": {"position": [3.8, 3.3, 8.0], "look_at": [0., 1.2, -4.0]},
        "source": source, "provenance": source,
    }


load_grounded_scene = scene_payload


if __name__ == "__main__":
    payload = scene_payload()
    print(json.dumps({key: value for key, value in payload.items() if key != "meshes"}, indent=2))
    print(len(payload["meshes"]), "color meshes,",
          sum(len(mesh["indices"]) // 3 for mesh in payload["meshes"]), "triangles")
