"""Tiny synthetic GLB fixtures for offline importer and retarget tests.

These meshes are test geometry, not licensed character artwork or motion output.
"""

from __future__ import annotations

import json
import struct
import zlib


def make_glb(document: dict, binary: bytes) -> bytes:
    document = dict(document)
    original_buffers = document.get("buffers", [{}])
    document["buffers"] = [{**original_buffers[0], "byteLength": len(binary)}]
    json_chunk = json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    bin_chunk = binary + b"\0" * (-len(binary) % 4)
    length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    return (struct.pack("<4sII", b"glTF", 2, length) +
            struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk +
            struct.pack("<I4s", len(bin_chunk), b"BIN\0") + bin_chunk)


def base_document_and_binary() -> tuple[dict, bytes]:
    positions = struct.pack("<9f", -0.2, 0, 0, 0.2, 0, 0, 0, 0.4, 0)
    normals = struct.pack("<9f", *(v for _ in range(3) for v in (0, 0, 1)))
    indices = struct.pack("<3H", 0, 1, 2)
    binary = positions + normals + indices
    views = [
        {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
        {"buffer": 0, "byteOffset": len(positions), "byteLength": len(normals)},
        {"buffer": 0, "byteOffset": len(positions) + len(normals), "byteLength": len(indices)},
    ]
    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [-0.2, 0, 0], "max": [0.2, 0.4, 0]},
        {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
        {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"},
    ]
    document = {
        "asset": {"version": "2.0", "generator": "ShellHacks synthetic fixture"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "mesh", "mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "mode": 4}]}],
        "bufferViews": views, "accessors": accessors,
        "materials": [{"name": "synthetic", "pbrMetallicRoughness": {"baseColorFactor": [0.4, 0.6, 0.9, 1]}}],
    }
    document["meshes"][0]["primitives"][0]["material"] = 0
    return document, binary


def make_static_glb() -> bytes:
    document, binary = base_document_and_binary()
    return make_glb(document, binary)


_ROLE_PARENT = {
    "pelvis": None,
    "spine": "pelvis",
    "left_upper_arm": "spine", "left_forearm": "left_upper_arm",
    "right_upper_arm": "spine", "right_forearm": "right_upper_arm",
    "left_thigh": "pelvis", "left_shin": "left_thigh", "left_foot": "left_shin",
    "right_thigh": "pelvis", "right_shin": "right_thigh", "right_foot": "right_shin",
}
_NAMES = {
    "g1": {
        "pelvis": "pelvis_skel", "spine": "waist_pitch_skel",
        "left_upper_arm": "left_shoulder_yaw_skel", "left_forearm": "left_elbow_skel",
        "right_upper_arm": "right_shoulder_yaw_skel", "right_forearm": "right_elbow_skel",
        "left_thigh": "left_hip_yaw_skel", "left_shin": "left_knee_skel", "left_foot": "left_ankle_roll_skel",
        "right_thigh": "right_hip_yaw_skel", "right_shin": "right_knee_skel", "right_foot": "right_ankle_roll_skel",
    },
    "mixamo": {
        "pelvis": "mixamorig:Hips", "spine": "mixamorig:Spine",
        "left_upper_arm": "mixamorig:LeftArm", "left_forearm": "mixamorig:LeftForeArm",
        "right_upper_arm": "mixamorig:RightArm", "right_forearm": "mixamorig:RightForeArm",
        "left_thigh": "mixamorig:LeftUpLeg", "left_shin": "mixamorig:LeftLeg", "left_foot": "mixamorig:LeftFoot",
        "right_thigh": "mixamorig:RightUpLeg", "right_shin": "mixamorig:RightLeg", "right_foot": "mixamorig:RightFoot",
    },
}
_OFFSETS = {
    "pelvis": (0, 1.0, 0), "spine": (0, 0.24, 0),
    "left_upper_arm": (0.25, 0.24, 0), "left_forearm": (0.33, 0, 0),
    "right_upper_arm": (-0.25, 0.24, 0), "right_forearm": (-0.33, 0, 0),
    "left_thigh": (0.14, -0.12, 0), "left_shin": (0, -0.4, 0), "left_foot": (0, -0.4, 0.1),
    "right_thigh": (-0.14, -0.12, 0), "right_shin": (0, -0.4, 0), "right_foot": (0, -0.4, 0.1),
}


def make_humanoid_glb(name_scheme: str = "g1") -> bytes:
    if name_scheme not in _NAMES:
        raise ValueError("name_scheme must be 'g1' or 'mixamo'")
    document, binary = base_document_and_binary()
    roles = list(_ROLE_PARENT)
    nodes = document["nodes"]
    nodes[0]["skin"] = 0
    world: dict[str, tuple[float, float, float]] = {}
    scale = 1.0 if name_scheme == "g1" else 1.2
    for role in roles:
        parent = _ROLE_PARENT[role]
        offset = tuple(value * scale for value in _OFFSETS[role])
        world[role] = offset if parent is None else tuple(a + b for a, b in zip(world[parent], offset))
        nodes.append({"name": _NAMES[name_scheme][role], "translation": list(offset)})
    for role in roles:
        parent = _ROLE_PARENT[role]
        if parent is None:
            document["scenes"][0]["nodes"].append(roles.index(role) + 1)
        else:
            nodes[roles.index(parent) + 1].setdefault("children", []).append(roles.index(role) + 1)
    def append_view(content: bytes) -> int:
        nonlocal binary
        binary += b"\0" * (-len(binary) % 4)
        offset = len(binary)
        binary += content
        document["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(content)})
        return len(document["bufferViews"]) - 1
    joints = bytes([0, 0, 0, 0] * 3)
    weights = struct.pack("<12f", *(value for _ in range(3) for value in (1, 0, 0, 0)))
    jview = append_view(joints)
    wview = append_view(weights)
    document["accessors"].extend([
        {"bufferView": jview, "componentType": 5121, "count": 3, "type": "VEC4"},
        {"bufferView": wview, "componentType": 5126, "count": 3, "type": "VEC4"},
    ])
    primitive = document["meshes"][0]["primitives"][0]
    primitive["attributes"].update({"JOINTS_0": 3, "WEIGHTS_0": 4})
    inverse_binds = bytearray()
    for role in roles:
        x, y, z = world[role]
        matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -x, -y, -z, 1]
        inverse_binds.extend(struct.pack("<16f", *matrix))
    bind_view = append_view(bytes(inverse_binds))
    document["accessors"].append({"bufferView": bind_view, "componentType": 5126, "count": len(roles), "type": "MAT4"})
    document["skins"] = [{"joints": list(range(1, len(roles) + 1)), "skeleton": 1, "inverseBindMatrices": 5}]
    if name_scheme == "mixamo":
        # A real embedded PNG/material path, synthesized from four RGBA pixels.
        def chunk(kind: bytes, payload: bytes) -> bytes:
            return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
        rows = (b"\x00" + bytes.fromhex("7fa7e6ff 7fa7e6ff")) * 2
        png = (b"\x89PNG\r\n\x1a\n" +
               chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 6, 0, 0, 0)) +
               chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
        image_view = append_view(png)
        uv_view = append_view(struct.pack("<6f", 0, 0, 1, 0, 0.5, 1))
        document["images"] = [{"bufferView": image_view, "mimeType": "image/png"}]
        document["textures"] = [{"source": 0}]
        document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
        document["accessors"].append({"bufferView": uv_view, "componentType": 5126, "count": 3, "type": "VEC2"})
        primitive["attributes"]["TEXCOORD_0"] = 6
    return make_glb(document, binary)
