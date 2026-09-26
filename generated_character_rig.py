"""Export the fitted generated human as a native, self-contained glTF skin.

This adapter is for locally generated catalog entries. Its marker describes a
fit, not trusted provenance: imports still pass the ordinary GLB compatibility
checks and must never opt into this adapter based on arbitrary extras alone.
"""
from __future__ import annotations

from copy import deepcopy
import json
import struct

import numpy as np

from character_actor import GeneratedCharacterActor, _BONES, _PARENTS, _smooth_normals
from character_assets import inspect_glb
from retargeting import RigMappingError, RigProfile, TargetPose, _default_skeleton, _rotation, neutral_source_pose

_MARKER = "shellhacks_generated_human"
_NAMES = ("Hips", "Spine", "Head", "LeftArm", "LeftForeArm", "LeftHand",
          "RightArm", "RightForeArm", "RightHand", "LeftUpLeg", "LeftLeg",
          "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot",
          "LeftShoulderBlend", "RightShoulderBlend")


def _pack(document, binary):
    document["buffers"] = [{"byteLength": len(binary)}]
    encoded = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary = bytes(binary) + b"\0" * (-len(binary) % 4)
    return (struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(binary))
            + struct.pack("<I4s", len(encoded), b"JSON") + encoded
            + struct.pack("<I4s", len(binary), b"BIN\0") + binary)


def _calibration(positions, rotations, skeleton):
    try:
        from character_actor import _as_numpy
        p, r = _as_numpy(positions).astype(float), _as_numpy(rotations).astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RigMappingError("Generated calibration must contain numeric arrays") from exc
    if (skeleton.nbjoints != 34 or p.shape != (34, 3) or r.shape != (34, 3, 3)
            or not np.isfinite(p).all() or abs(p).max() > 100):
        raise RigMappingError("Generated calibration needs bounded G1 positions and rotations")
    for rotation in r:
        _rotation(rotation, "Generated calibration rotation", tolerance=1e-3)
    return p, r


def export_generated_character(data, skeleton, bind_positions=None, bind_rotations=None):
    """Fit one static textured human, retaining original material/image payloads."""
    source = inspect_glb(data)
    doc = deepcopy(source.document)
    meshes = doc.get("meshes", [])
    displayed = [node for node in source.nodes if "mesh" in doc["nodes"][node.index]]
    if (source.kind != "static" or len(meshes) != 1 or len(meshes[0]["primitives"]) != 1
            or len(displayed) != 1):
        raise ValueError("Generated fitting requires one static mesh with one material")
    if (bind_positions is None) != (bind_rotations is None):
        raise ValueError("Provide both calibration positions and rotations")
    if bind_positions is None:
        bind_positions, bind_rotations = neutral_source_pose(skeleton)
    bind_positions, bind_rotations = _calibration(bind_positions, bind_rotations, skeleton)
    actor = GeneratedCharacterActor(None, data, "/generated/export", skeleton,
                                    bind_positions, bind_rotations)
    primitive = meshes[0]["primitives"][0]
    original = source.read_accessor(primitive["attributes"]["POSITION"])
    world = original @ displayed[0].world_matrix[:3, :3].T + displayed[0].world_matrix[:3, 3]
    bounds = np.array([world.min(axis=0), world.max(axis=0)])
    origin = (bounds[0] + bounds[1]) / 2
    origin[1] = bounds[0, 1]
    fitted = (world-origin) * (1.70 / (bounds[1, 1]-bounds[0, 1]))
    # Preserve accessor vertex order and UV indices exactly; never silently
    # attach fitting weights to a reordered mesh from a format importer.
    if fitted.shape != actor.vertices.shape or not np.allclose(fitted, actor.vertices, atol=1e-6):
        raise ValueError("Generated mesh vertex order changed during fitting")
    original_faces = (source.read_accessor(primitive["indices"], normalize=False).reshape(-1, 3)
                      if "indices" in primitive else np.arange(len(original)).reshape(-1, 3))
    if not np.array_equal(original_faces, actor.faces):
        raise ValueError("Generated mesh triangle order changed during fitting")
    binary = bytearray(source.binary_chunk)

    def append(values, kind, component=5126, target=None):
        array = np.asarray(values, dtype={5126: "<f4", 5123: "<u2"}[component])
        binary.extend(b"\0" * (-len(binary) % 4))
        view = {"buffer": 0, "byteOffset": len(binary), "byteLength": array.nbytes}
        if target is not None:
            view["target"] = target
        doc.setdefault("bufferViews", []).append(view)
        binary.extend(array.tobytes())
        accessor = {"bufferView": len(doc["bufferViews"])-1, "componentType": component,
                    "count": len(array), "type": kind}
        if kind == "VEC3":
            accessor.update(min=array.min(axis=0).tolist(), max=array.max(axis=0).tolist())
        doc.setdefault("accessors", []).append(accessor)
        return len(doc["accessors"])-1

    attributes = primitive["attributes"]
    attributes["POSITION"] = append(actor.vertices, "VEC3", target=34962)
    attributes["NORMAL"] = append(_smooth_normals(actor.vertices, actor.faces), "VEC3", target=34962)
    # A transformed tangent can no longer be reused after baking node axes.
    # GLTFLoader reconstructs tangent space from the preserved UVs/normals.
    attributes.pop("TANGENT", None)
    indices = np.argsort(actor.weights, axis=1)[:, -4:]
    weights = np.take_along_axis(actor.weights, indices, axis=1)
    attributes["JOINTS_0"] = append(indices, "VEC4", 5123, 34962)
    attributes["WEIGHTS_0"] = append(weights, "VEC4", target=34962)
    doc["nodes"] = [{"name": "GeneratedHuman", "mesh": 0, "skin": 0}]
    for i, name in enumerate(_NAMES):
        parent = _PARENTS[i]
        offset = actor.rest[i] if parent < 0 else actor.rest[i]-actor.rest[parent]
        node = {"name": name, "translation": offset.tolist()}
        children = (np.flatnonzero(_PARENTS == i)+1).tolist()
        if children:
            node["children"] = children
        doc["nodes"].append(node)
    inverse = np.tile(np.eye(4), (17, 1, 1))
    inverse[:, :3, 3] = -actor.rest
    doc["skins"] = [{"name": "GeneratedHumanFit", "skeleton": 1, "joints": list(range(1, 18)),
                     "inverseBindMatrices": append(inverse.transpose(0, 2, 1).reshape(17, 16), "MAT4")}]
    doc["scenes"], doc["scene"] = [{"nodes": [0, 1]}], 0
    doc.pop("animations", None)
    doc["extras"] = {_MARKER: {"version": 1, "source_bind_positions": bind_positions.tolist(),
                               "source_bind_rotations": bind_rotations.tolist()}}
    result = _pack(doc, binary)
    inspect_glb(result)
    return result


class GeneratedCharacterRetargeter:
    """Original fitted motion, represented as original glTF node matrices."""
    def __init__(self, asset, skeleton=None):
        self.asset = asset
        self.skeleton = _default_skeleton() if skeleton is None else skeleton
        doc = asset.document
        extras = doc.get("extras")
        marker = extras.get(_MARKER) if isinstance(extras, dict) else None
        if (not isinstance(marker, dict) or set(marker) != {"version", "source_bind_positions", "source_bind_rotations"}
                or type(marker["version"]) is not int or marker["version"] != 1):
            raise RigMappingError("Missing supported generated human fit")
        self.source_positions, self.source_rotations = _calibration(
            marker["source_bind_positions"], marker["source_bind_rotations"], self.skeleton)
        if (len(asset.nodes) != 18 or len(doc.get("meshes", [])) != 1 or len(doc.get("skins", [])) != 1
                or len(doc["meshes"][0]["primitives"]) != 1
                or doc.get("scenes") != [{"nodes": [0, 1]}] or doc.get("scene", 0) != 0):
            raise RigMappingError("Generated fit has unexpected scene topology")
        if doc["nodes"][0] != {"name": "GeneratedHuman", "mesh": 0, "skin": 0}:
            raise RigMappingError("Generated fit has unexpected mesh transform")
        skin = doc["skins"][0]
        if (skin.get("joints") != list(range(1, 18)) or skin.get("skeleton") != 1
                or "inverseBindMatrices" not in skin):
            raise RigMappingError("Generated fit has unexpected skin joints")
        self.rest_local = np.stack([node.local_matrix for node in asset.nodes])
        self.rest_world = np.stack([node.world_matrix for node in asset.nodes])
        for i, node in enumerate(asset.nodes[1:]):
            parent = None if _PARENTS[i] < 0 else int(_PARENTS[i]+1)
            expected_children = tuple((np.flatnonzero(_PARENTS == i)+1).tolist())
            if (node.name != _NAMES[i] or node.parent != parent or node.children != expected_children
                    or not np.allclose(node.world_matrix[:3, :3], np.eye(3), atol=1e-7)):
                raise RigMappingError("Generated fit bone hierarchy or transforms changed")
        rest = self.rest_world[1:, :3, 3].copy()
        if not np.isfinite(rest).all() or abs(rest).max() > 3 or not np.allclose(rest[[15, 16]], rest[[3, 6]]):
            raise RigMappingError("Generated fit has invalid anatomical pivots")
        inverse = asset.read_accessor(skin["inverseBindMatrices"]).reshape(17, 4, 4).transpose(0, 2, 1)
        if not np.allclose(inverse @ self.rest_world[1:], np.eye(4), atol=1e-6):
            raise RigMappingError("Generated fit inverse binds do not match the fitted bones")
        primitive = doc["meshes"][0]["primitives"][0]
        attributes = primitive["attributes"]
        vertices = asset.read_accessor(attributes["POSITION"]).astype(float)
        faces = (asset.read_accessor(primitive["indices"]).reshape(-1, 3) if "indices" in primitive
                 else np.arange(len(vertices)).reshape(-1, 3))
        if len(vertices) > 250_000 or len(faces) > 500_000 or len(vertices) < 300:
            raise RigMappingError("Generated fit geometry exceeds fitting limits")
        if (not np.isclose(np.ptp(vertices[:, 1]), 1.70, atol=1e-5)
                or not np.isclose(vertices[:, 1].min(), 0, atol=1e-5)):
            raise RigMappingError("Generated fit geometry must be normalized to standing height")
        indices = asset.read_accessor(attributes["JOINTS_0"], normalize=False)
        values = asset.read_accessor(attributes["WEIGHTS_0"])
        weights = np.zeros((len(vertices), 17), dtype=np.float32)
        rows = np.broadcast_to(np.arange(len(vertices))[:, None], indices.shape)
        np.add.at(weights, (rows, indices), values)
        self._actor = GeneratedCharacterActor.from_fitted_mesh(
            vertices, faces, weights, rest, self.skeleton, self.source_positions, self.source_rotations)
        self.profile = RigProfile("generated_human_v1", {
            ("spine" if role == "torso" else role): i+1
            for i, (role, _, _) in enumerate(_BONES[:15])})
        self.warnings = ("Generated human fitting is approximate; face and fingers are static.",)

    def neutral_source_pose(self):
        return self.source_positions.copy(), self.source_rotations.copy()

    def bind_pose(self):
        return TargetPose(self.rest_local.copy(), self.rest_world.copy(), self.rest_world[1, :3, 3].copy())

    def retarget(self, global_positions, global_rotations):
        positions, rotations = _calibration(global_positions, global_rotations, self.skeleton)
        self._actor.update(positions, rotations)
        world = self.rest_world.copy()
        world[1:, :3, :3] = self._actor.bone_matrices
        world[1:, :3, 3] = self._actor.bone_positions
        local = world.copy()
        for node in self.asset.nodes:
            if node.parent is not None:
                local[node.index] = np.linalg.inv(world[node.parent]) @ world[node.index]
        return TargetPose(local, world, world[1, :3, 3].copy())


def build_generated_retargeter(asset, skeleton=None):
    """Load an inspected, trusted generated entry without refitting its geometry."""
    return GeneratedCharacterRetargeter(asset, skeleton)
