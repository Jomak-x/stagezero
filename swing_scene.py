"""Read-only adapter for the captured, model-generated city scene.

The mesh transform matches ObjectSceneLayer._build_custom: each asset is
compiled, fitted to its own mesh bounds, scaled by the scene object's size,
rotated around world Y by yaw, and translated to its scene position.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from swing_assets.asset_geometry import compile_asset


SOURCE_PATH = Path(__file__).with_name("swing_assets") / "strong-city.json"
SOURCE_SHA256 = "a478cbf6c51cba6ddd22e75fecc24203f1725f51c50bb2eadf6ccc5b886dac00"
COMPILER_SHA256 = "09881c0db7e741978090da4404848a869e26abde474ef1062581bd201a028ab6"
BUILDING_ASSETS = frozenset({
    "terracotta-mercantile", "cream-corner-bakery",
    "blue-glass-office-tower", "stepped-bronze-highrise",
})


def _source_scene():
    data = SOURCE_PATH.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"generated city snapshot changed: {digest}")
    scene = json.loads(data)
    if scene.get("version") != 3 or not scene.get("objects") or not scene.get("assets"):
        raise ValueError("captured city has an unexpected scene schema")
    return scene


def _fit(asset):
    vertices, faces, colors = compile_asset(asset)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    return vertices.astype(np.float64), faces, colors, lower, upper


def _world_vertices(vertices, lower, upper, obj):
    # Identical normalization and Y rotation to ObjectSceneLayer._build_custom.
    fitted = (vertices - (lower + upper) / 2) / np.maximum(upper - lower, 1e-6)
    scaled = fitted * np.asarray(obj["size"], dtype=np.float64)
    angle = math.radians(obj.get("yaw", 0))
    c, s = math.cos(angle), math.sin(angle)
    rotated = scaled.copy()
    rotated[:, 0] = c * scaled[:, 0] + s * scaled[:, 2]
    rotated[:, 2] = -s * scaled[:, 0] + c * scaled[:, 2]
    return rotated + np.asarray(obj["position"], dtype=np.float64)


def _roof(obj, asset, lower, upper):
    """Find a broad, upward box face from the generated building recipe."""
    candidates = []
    span = np.maximum(upper - lower, 1e-6)
    for index, part in enumerate(asset["parts"]):
        if part["shape"] != "box" or "repeat" in part:
            continue
        rotation = part.get("rotation", (0, 0, 0))
        if abs(rotation[0]) > 1e-6 or abs(rotation[2]) > 1e-6:
            continue
        top = part["position"][1] + part["size"][1] / 2
        height = obj["position"][1] + (top - (lower[1] + upper[1]) / 2) / span[1] * obj["size"][1]
        if height < obj["position"][1] + obj["size"][1] * .25:
            continue
        width = part["size"][0] / span[0] * obj["size"][0]
        depth = part["size"][2] / span[2] * obj["size"][2]
        if min(width, depth) < 1.0:
            continue
        # Higher roof planes win; narrow decorative trim loses to a deck.
        candidates.append((height, width * depth, index, width, depth, part))
    if not candidates:
        raise ValueError(f"generated building {obj['id']} has no safe roof deck")
    height, _, index, width, depth, part = max(candidates)
    center = np.asarray(part["position"], dtype=np.float64)
    center = (center - (lower + upper) / 2) / span * np.asarray(obj["size"], dtype=np.float64)
    angle = math.radians(obj.get("yaw", 0))
    c, s = math.cos(angle), math.sin(angle)
    x = obj["position"][0] + c * center[0] + s * center[2]
    z = obj["position"][2] - s * center[0] + c * center[2]
    return {
        "id": f"roof-{obj['id']}", "building_id": obj["id"],
        "position": [float(x), float(height), float(z)],
        "radius": round(min(width, depth) * .27, 3),
        "width": round(float(width), 3), "depth": round(float(depth), 3),
        "source_part": index,
    }


def _colored_meshes(obj, world, faces, colors):
    """One indexed mesh per source vertex color, retaining exact triangles."""
    rgb = colors[faces[:, 0]]
    if not np.all(colors[faces] == rgb[:, None, :]):
        raise ValueError(f"asset {obj['asset']} contains a multicolor triangle")
    for color in np.unique(rgb, axis=0):
        selected = faces[np.all(rgb == color, axis=1)]
        used = np.unique(selected)
        yield {
            "id": f"{obj['id']}-rgb-{'-'.join(map(str, color))}",
            "object_id": obj["id"],
            "positions": world[used].tolist(),
            "indices": np.searchsorted(used, selected).reshape(-1).tolist(),
            "color": color.tolist(),
        }


def _mesh_anchor(obj, world, bounds):
    """Pick an existing high mesh vertex at the approach-side facade corner.

    The roof may step inward from the full building box. A web aimed at its
    centre can therefore cross the box for a long distance before reaching the
    source surface. An upper facade vertex near the street/front box corner
    minimizes that conservative overlap while remaining on generated geometry.
    """
    high = world[world[:, 1] >= bounds["max"][1] - max(1.3, obj["size"][1] * .15)]
    if not len(high):
        raise ValueError(f"no upper vertices in generated building {obj['id']}")
    street_x = bounds["max"][0] if obj["position"][0] < 0 else bounds["min"][0]
    front_z = bounds["max"][2]  # The route advances from +Z to -Z.
    planar = np.hypot(high[:, 0] - street_x, high[:, 2] - front_z)
    best_distance = planar.min()
    near = np.flatnonzero(planar <= best_distance + 1e-6)
    chosen = high[near[np.argmax(high[near, 1])]]
    return {
        "id": f"anchor-{obj['id']}", "building_id": obj["id"],
        "position": chosen.tolist(),
        "terminal_allowance_m": round(max(.4, float(best_distance) + .15), 3),
        "source": "existing upper facade mesh vertex",
    }


@lru_cache(maxsize=1)
def load_swing_scene():
    """Return exact colored city meshes and derived world-space play geometry.

    `roof.position.y` is the physical surface. `spawn` and `mj_spawn` are hip
    positions; collision boxes use the full rendered building bounds.
    """
    scene = _source_scene()
    assets = {asset["id"]: asset for asset in scene["assets"]}
    compiled = {key: _fit(asset) for key, asset in assets.items()}
    meshes, buildings, roofs, anchors = [], [], [], []
    for obj in scene["objects"]:
        if obj["kind"] != "custom":
            raise ValueError(f"unexpected non-custom city object {obj['id']}")
        vertices, faces, colors, lower, upper = compiled[obj["asset"]]
        world = _world_vertices(vertices, lower, upper, obj)
        meshes.extend(_colored_meshes(obj, world, faces, colors))
        if obj["asset"] in BUILDING_ASSETS:
            bounds = {
                "id": obj["id"], "asset": obj["asset"],
                "min": world.min(axis=0).round(6).tolist(),
                "max": world.max(axis=0).round(6).tolist(),
            }
            buildings.append(bounds)
            roof = _roof(obj, assets[obj["asset"]], lower, upper)
            roofs.append(roof)
            if roof["position"][1] >= 4.5:
                anchors.append(_mesh_anchor(obj, world, bounds))
    roofs.sort(key=lambda item: item["position"][2], reverse=True)
    anchors.sort(key=lambda item: item["position"][2], reverse=True)
    pickup = next(roof for roof in roofs if roof["building_id"] == "city-4")
    destination = next(roof for roof in roofs if roof["building_id"] == "city-10")
    provenance = {
            "path": "review/scene-refinement/strong-city.json",
            "sha256": SOURCE_SHA256,
            "compiler_path": "asset_geometry.py",
            "compiler_sha256": COMPILER_SHA256,
            "generated_asset_model": "gpt-5-6-sol",
            "layout": "captured refined pipeline composition, with deterministic architectural staging",
        }
    return {
        "source": provenance, "provenance": provenance,
        "name": scene["name"], "lighting": scene["lighting"],
        "meshes": meshes, "buildings": buildings, "collisions": buildings,
        "roofs": roofs, "landing_zones": roofs, "anchors": anchors,
        "spawn": [0.0, .96, 3.5], "start": [0.0, .96, 3.5],
        "mj_spawn": [pickup["position"][0], pickup["position"][1] + .96,
                     pickup["position"][2]],
        "pickup_roof_id": pickup["id"], "landing_roof_id": destination["id"],
    }


if __name__ == "__main__":
    data = load_swing_scene()
    print(json.dumps({key: value for key, value in data.items() if key != "meshes"}, indent=2))
    print(f"{len(data['meshes'])} colored meshes, "
          f"{sum(len(mesh['indices']) // 3 for mesh in data['meshes'])} source triangles")
