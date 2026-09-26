"""Generate two visible synthetic GLB avatars for local renderer QA.

Run from any directory with ``python examples/create_glb_demo_models.py``.
The output is always under this checkout's ignored ``.runtime/glb`` folder.
These are deliberately simple cuboid test shapes, not ARDY motion output or
production character art. All geometry, skinning, materials and PNG bytes are
built deterministically here; no external asset or test fixture is imported.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
import zlib


OUTPUT = Path(__file__).resolve().parents[1] / ".runtime" / "glb"
PARENTS = {
    "pelvis": None, "spine": "pelvis", "head": "spine",
    "left_upper_arm": "spine", "left_forearm": "left_upper_arm", "left_hand": "left_forearm",
    "right_upper_arm": "spine", "right_forearm": "right_upper_arm", "right_hand": "right_forearm",
    "left_thigh": "pelvis", "left_shin": "left_thigh", "left_foot": "left_shin",
    "right_thigh": "pelvis", "right_shin": "right_thigh", "right_foot": "right_shin",
}
NAMES = {
    "g1": {
        "pelvis": "pelvis_skel", "spine": "waist_pitch_skel", "head": "head_skel",
        "left_upper_arm": "left_shoulder_yaw_skel", "left_forearm": "left_elbow_skel", "left_hand": "left_wrist_yaw_skel",
        "right_upper_arm": "right_shoulder_yaw_skel", "right_forearm": "right_elbow_skel", "right_hand": "right_wrist_yaw_skel",
        "left_thigh": "left_hip_yaw_skel", "left_shin": "left_knee_skel", "left_foot": "left_ankle_roll_skel",
        "right_thigh": "right_hip_yaw_skel", "right_shin": "right_knee_skel", "right_foot": "right_ankle_roll_skel",
    },
    "mixamo": {
        "pelvis": "mixamorig:Hips", "spine": "mixamorig:Spine", "head": "mixamorig:Head",
        "left_upper_arm": "mixamorig:LeftArm", "left_forearm": "mixamorig:LeftForeArm", "left_hand": "mixamorig:LeftHand",
        "right_upper_arm": "mixamorig:RightArm", "right_forearm": "mixamorig:RightForeArm", "right_hand": "mixamorig:RightHand",
        "left_thigh": "mixamorig:LeftUpLeg", "left_shin": "mixamorig:LeftLeg", "left_foot": "mixamorig:LeftFoot",
        "right_thigh": "mixamorig:RightUpLeg", "right_shin": "mixamorig:RightLeg", "right_foot": "mixamorig:RightFoot",
    },
}
PALETTES = {
    "g1": [(40, 113, 174), (86, 171, 219), (232, 177, 75), (42, 76, 113),
           (68, 134, 184), (122, 191, 218), (236, 210, 146), (34, 66, 92)],
    "mixamo": [(140, 80, 166), (205, 143, 205), (244, 196, 100), (77, 50, 117),
               (169, 101, 180), (225, 168, 210), (253, 222, 159), (63, 41, 101)],
}


def _world_positions(style: str) -> dict[str, tuple[float, float, float]]:
    if style == "g1":
        shoulder_y, shoulder_x, elbow_x, hand_x = 1.50, 0.28, 0.63, 0.95
        pelvis_y, spine_y, head_y = 1.00, 1.24, 1.76
        thigh_y, knee_y, ankle_y = 0.91, 0.50, 0.11
        hip_x = 0.13
    else:
        shoulder_y, shoulder_x, elbow_x, hand_x = 1.66, 0.32, 0.76, 1.16
        pelvis_y, spine_y, head_y = 1.08, 1.36, 1.92
        thigh_y, knee_y, ankle_y = 0.98, 0.52, 0.12
        hip_x = 0.17
    positions = {"pelvis": (0, pelvis_y, 0), "spine": (0, spine_y, 0), "head": (0, head_y, 0)}
    for side, sign in (("left", 1), ("right", -1)):
        positions.update({
            f"{side}_upper_arm": (sign * shoulder_x, shoulder_y, 0),
            f"{side}_forearm": (sign * elbow_x, shoulder_y, 0),
            f"{side}_hand": (sign * hand_x, shoulder_y, 0),
            f"{side}_thigh": (sign * hip_x, thigh_y, 0),
            f"{side}_shin": (sign * hip_x, knee_y, 0),
            f"{side}_foot": (sign * hip_x, ankle_y, 0.05),
        })
    return positions


def _box_corners(bounds: tuple[float, float, float, float, float, float]):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return (
        ((1, 0, 0), ((xmax, ymin, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmax, ymin, zmax))),
        ((-1, 0, 0), ((xmin, ymin, zmax), (xmin, ymax, zmax), (xmin, ymax, zmin), (xmin, ymin, zmin))),
        ((0, 1, 0), ((xmin, ymax, zmin), (xmin, ymax, zmax), (xmax, ymax, zmax), (xmax, ymax, zmin))),
        ((0, -1, 0), ((xmin, ymin, zmax), (xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymin, zmax))),
        ((0, 0, 1), ((xmax, ymin, zmax), (xmax, ymax, zmax), (xmin, ymax, zmax), (xmin, ymin, zmax))),
        ((0, 0, -1), ((xmin, ymin, zmin), (xmin, ymax, zmin), (xmax, ymax, zmin), (xmax, ymin, zmin))),
    )


def _png(palette: list[tuple[int, int, int]]) -> bytes:
    # Four by four opaque RGBA PNG. Each category occupies two pixels in a row.
    pixels = [(*palette[(x // 2 + 4 * (y // 2)) % len(palette)], 255)
              for y in range(4) for x in range(4)]
    scanlines = b"".join(b"\x00" + bytes(channel for pixel in pixels[y * 4:(y + 1) * 4] for channel in pixel)
                         for y in range(4))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 6, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(scanlines, level=9)) + chunk(b"IEND", b""))


class MeshBuilder:
    def __init__(self, positions: dict[str, tuple[float, float, float]]):
        self.rest = positions
        self.roles = list(PARENTS)
        self.ordinal = {role: i for i, role in enumerate(self.roles)}
        self.positions: list[tuple[float, float, float]] = []
        self.normals: list[tuple[int, int, int]] = []
        self.uvs: list[tuple[float, float]] = []
        self.joints: list[tuple[int, int, int, int]] = []
        self.weights: list[tuple[float, float, float, float]] = []
        self.indices: list[int] = []

    def add_box(self, bounds: tuple[float, float, float, float, float, float],
                role: str, palette_slot: int, blend_to: str | None = None) -> None:
        start = self.rest[role]
        end = self.rest[blend_to] if blend_to else None
        delta = tuple(end[i] - start[i] for i in range(3)) if end else None
        denominator = sum(value * value for value in delta) if delta else 1
        # Each vertex samples the middle of one texel. Nearest sampling gives a
        # clear, solid part color while exercising embedded texture transfer.
        uv = (((palette_slot % 4) + 0.5) / 4, ((palette_slot // 4) + 0.5) / 4)
        for normal, corners in _box_corners(bounds):
            first = len(self.positions)
            for point in corners:
                blend = 0.0
                if end is not None and delta is not None:
                    projection = sum((point[i] - start[i]) * delta[i] for i in range(3)) / denominator
                    blend = 0.45 * max(0.0, min(1.0, projection))
                self.positions.append(point)
                self.normals.append(normal)
                self.uvs.append(uv)
                self.joints.append((self.ordinal[role], self.ordinal[blend_to] if blend_to else 0, 0, 0))
                self.weights.append((1.0 - blend, blend, 0.0, 0.0))
            self.indices.extend((first, first + 1, first + 2, first, first + 2, first + 3))


def _body_mesh(style: str, rest: dict[str, tuple[float, float, float]]) -> MeshBuilder:
    mesh = MeshBuilder(rest)
    factor = 1.0 if style == "g1" else 1.13
    pelvis_y = rest["pelvis"][1]
    shoulder_y = rest["left_upper_arm"][1]
    head_y = rest["head"][1]
    mesh.add_box((-0.20 * factor, 0.20 * factor, pelvis_y - 0.10 * factor,
                  pelvis_y + 0.12 * factor, -0.12 * factor, 0.12 * factor), "pelvis", 0)
    mesh.add_box((-0.26 * factor, 0.26 * factor, pelvis_y + 0.12 * factor,
                  shoulder_y + 0.09 * factor, -0.13 * factor, 0.13 * factor), "spine", 1)
    mesh.add_box((-0.13 * factor, 0.13 * factor, head_y - 0.14 * factor,
                  head_y + 0.14 * factor, -0.12 * factor, 0.12 * factor), "head", 2)
    for side, sign in (("left", 1), ("right", -1)):
        shoulder = rest[f"{side}_upper_arm"][0]
        elbow = rest[f"{side}_forearm"][0]
        hand = rest[f"{side}_hand"][0]
        upper_x = sorted((shoulder, elbow))
        lower_x = sorted((elbow, hand))
        arm_half = 0.065 * factor
        fore_half = 0.052 * factor
        mesh.add_box((upper_x[0], upper_x[1], shoulder_y - arm_half, shoulder_y + arm_half,
                      -arm_half, arm_half), f"{side}_upper_arm", 4 if sign > 0 else 5,
                     f"{side}_forearm")
        mesh.add_box((lower_x[0], lower_x[1], shoulder_y - fore_half, shoulder_y + fore_half,
                      -fore_half, fore_half), f"{side}_forearm", 4 if sign > 0 else 5)
        mesh.add_box((hand - 0.075 * factor, hand + 0.075 * factor,
                      shoulder_y - 0.07 * factor, shoulder_y + 0.07 * factor,
                      -0.065 * factor, 0.065 * factor), f"{side}_hand", 2)
        hip = rest[f"{side}_thigh"]
        knee = rest[f"{side}_shin"]
        ankle = rest[f"{side}_foot"]
        thigh_half = 0.085 * factor
        shin_half = 0.065 * factor
        mesh.add_box((hip[0] - thigh_half, hip[0] + thigh_half, knee[1], hip[1],
                      -thigh_half, thigh_half), f"{side}_thigh", 3 if sign > 0 else 6,
                     f"{side}_shin")
        mesh.add_box((knee[0] - shin_half, knee[0] + shin_half, ankle[1], knee[1],
                      -shin_half, shin_half), f"{side}_shin", 3 if sign > 0 else 6,
                     f"{side}_foot")
        mesh.add_box((ankle[0] - 0.09 * factor, ankle[0] + 0.09 * factor,
                      0.02, ankle[1] + 0.04 * factor, -0.08 * factor, 0.27 * factor),
                     f"{side}_foot", 7)
    return mesh


def _glb(style: str) -> bytes:
    rest = _world_positions(style)
    mesh = _body_mesh(style, rest)
    binary = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []

    def add_view(payload: bytes) -> int:
        binary.extend(b"\0" * (-len(binary) % 4))
        offset = len(binary)
        binary.extend(payload)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(payload)})
        return len(views) - 1

    def floats(rows) -> bytes:
        values = [value for row in rows for value in row]
        return struct.pack("<" + "f" * len(values), *values)

    positions_view = add_view(floats(mesh.positions))
    normals_view = add_view(floats(mesh.normals))
    uv_view = add_view(floats(mesh.uvs))
    joints_view = add_view(bytes(value for row in mesh.joints for value in row))
    weights_view = add_view(floats(mesh.weights))
    indices_view = add_view(struct.pack("<" + "H" * len(mesh.indices), *mesh.indices))
    inverse_binds = []
    for role in mesh.roles:
        x, y, z = rest[role]
        inverse_binds.append((1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -x, -y, -z, 1))
    bind_view = add_view(floats(inverse_binds))
    png_view = add_view(_png(PALETTES[style]))
    coordinates = tuple(zip(*mesh.positions))
    descriptors = (
        (positions_view, 5126, "VEC3", {"min": [min(axis) for axis in coordinates],
                                        "max": [max(axis) for axis in coordinates]}),
        (normals_view, 5126, "VEC3", {}),
        (uv_view, 5126, "VEC2", {}),
        (joints_view, 5121, "VEC4", {}),
        (weights_view, 5126, "VEC4", {}),
        (indices_view, 5123, "SCALAR", {}),
        (bind_view, 5126, "MAT4", {}),
    )
    for view, component, shape, extras in descriptors:
        count = len(mesh.indices) if shape == "SCALAR" else len(mesh.roles) if shape == "MAT4" else len(mesh.positions)
        accessors.append({"bufferView": view, "componentType": component, "count": count,
                          "type": shape, **extras})
    roles = mesh.roles
    indices = {role: roles.index(role) + 1 for role in roles}
    nodes = [{"name": f"Synthetic {style.upper()} technical avatar", "mesh": 0, "skin": 0}]
    for role in roles:
        parent = PARENTS[role]
        center = rest[role]
        offset = center if parent is None else tuple(center[i] - rest[parent][i] for i in range(3))
        node = {"name": NAMES[style][role], "translation": list(offset)}
        children = [indices[child] for child, ancestor in PARENTS.items() if ancestor == role]
        if children:
            node["children"] = children
        nodes.append(node)
    document = {
        "asset": {"version": "2.0", "generator": "ShellHacks synthetic technical avatar generator"},
        "scene": 0, "scenes": [{"nodes": [0, indices["pelvis"]]}],
        "nodes": nodes,
        "meshes": [{"name": f"Synthetic {style.upper()} cuboid body", "primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2, "JOINTS_0": 3, "WEIGHTS_0": 4},
            "indices": 5, "material": 0, "mode": 4,
        }]}],
        "skins": [{"name": f"Synthetic {style.upper()} humanoid rig", "joints": list(range(1, len(roles) + 1)),
                   "skeleton": indices["pelvis"], "inverseBindMatrices": 6}],
        "images": [{"bufferView": png_view, "mimeType": "image/png"}],
        "samplers": [{"magFilter": 9728, "minFilter": 9728, "wrapS": 33071, "wrapT": 33071}],
        "textures": [{"source": 0, "sampler": 0}],
        "materials": [{"name": f"Synthetic {style.upper()} colored blocks",
                       "pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
                                                 "metallicFactor": 0, "roughnessFactor": 0.9},
                       "doubleSided": False}],
        "bufferViews": views, "accessors": accessors,
        "buffers": [{"byteLength": len(binary)}],
    }
    json_chunk = json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    bin_chunk = bytes(binary) + b"\0" * (-len(binary) % 4)
    length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    return (struct.pack("<4sII", b"glTF", 2, length) + struct.pack("<I4s", len(json_chunk), b"JSON") +
            json_chunk + struct.pack("<I4s", len(bin_chunk), b"BIN\0") + bin_chunk)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for style in ("g1", "mixamo"):
        path = OUTPUT / f"demo-{style}.glb"
        data = _glb(style)
        path.write_bytes(data)
        print(f"{path} ({len(data)} bytes; synthetic technical avatar)")


if __name__ == "__main__":
    main()
