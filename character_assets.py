"""Bounded GLB 2.0 import for local character previews and rig mapping.

This module deliberately keeps the source GLB intact. Rendering and rig-profile
approval belong to their respective layers; a valid skin is only mapping-ready.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import warnings
from typing import Any

import numpy as np
from PIL import Image


class AssetValidationError(ValueError):
    """An uploaded GLB cannot safely be used as a character asset."""


@dataclass(frozen=True)
class AssetLimits:
    max_file_bytes: int = 32 * 1024 * 1024
    max_json_bytes: int = 4 * 1024 * 1024
    max_expanded_bytes: int = 256 * 1024 * 1024
    max_vertices: int = 2_000_000
    max_triangles: int = 4_000_000
    max_nodes: int = 4096
    max_buffer_views: int = 32768
    max_accessors: int = 16384
    max_meshes: int = 4096
    max_primitives: int = 16384
    max_skins: int = 16
    max_materials: int = 512
    max_textures: int = 512
    max_images: int = 64
    max_image_pixels: int = 32_000_000


DEFAULT_LIMITS = AssetLimits()
_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
_DTYPES = {
    5120: np.dtype("i1"), 5121: np.dtype("u1"),
    5122: np.dtype("<i2"), 5123: np.dtype("<u2"),
    5125: np.dtype("<u4"), 5126: np.dtype("<f4"),
}
_ID = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_NAME = re.compile(r"[^\w .()\-]+", re.UNICODE)


def _fail(message: str) -> None:
    raise AssetValidationError(message)


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _index(value: Any, length: int, label: str) -> int:
    result = _integer(value, label)
    if result >= length:
        _fail(f"{label} is out of range")
    return result


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{label} must be an array")
    return value


def _finite_vector(value: Any, length: int, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        _fail(f"{label} must have {length} numbers")
    if any(type(x) not in (int, float) for x in value):
        _fail(f"{label} contains a non-finite number")
    try:
        converted = np.array(value, dtype=np.float64)
    except (OverflowError, ValueError):
        _fail(f"{label} contains a non-finite number")
    if not np.isfinite(converted).all():
        _fail(f"{label} contains a non-finite number")
    return converted


def _finite_scalar(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        _fail(f"{label} must be a finite number")
    try:
        converted = float(value)
    except OverflowError:
        _fail(f"{label} must be a finite number")
    if not math.isfinite(converted):
        _fail(f"{label} must be a finite number")
    return converted


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_constant(value: str) -> None:
    _fail(f"invalid JSON number: {value}")


def _json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        _fail("invalid JSON number: non-finite exponent")
    return result


def _parse_glb(data: bytes, limits: AssetLimits) -> tuple[dict[str, Any], bytes]:
    if not isinstance(data, bytes):
        _fail("GLB upload must be bytes")
    if len(data) > limits.max_file_bytes:
        _fail("GLB exceeds the file size limit")
    if len(data) < 20:
        _fail("GLB header or JSON chunk is missing")
    magic, version, declared = struct.unpack_from("<4sII", data)
    if magic != b"glTF" or version != 2 or declared != len(data):
        _fail("invalid GLB 2.0 header or length")
    offset = 12
    chunks: list[tuple[bytes, bytes]] = []
    while offset < len(data):
        if len(data) - offset < 8:
            _fail("truncated GLB chunk header")
        size, kind = struct.unpack_from("<I4s", data, offset)
        offset += 8
        if size % 4 or size > len(data) - offset:
            _fail("invalid GLB chunk length")
        chunks.append((kind, data[offset:offset + size]))
        offset += size
        if len(chunks) > 2:
            _fail("unsupported GLB chunk count")
    if not chunks or chunks[0][0] != b"JSON" or len(chunks[0][1]) > limits.max_json_bytes:
        _fail("missing or oversized GLB JSON chunk")
    if len(chunks) > 1 and chunks[1][0] != b"BIN\0":
        _fail("unsupported GLB binary chunk")
    try:
        doc = json.loads(chunks[0][1].decode("utf-8"), object_pairs_hook=_pairs_without_duplicates,
                         parse_constant=_bad_constant, parse_float=_json_float)
    except AssetValidationError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        _fail(f"invalid GLB JSON: {exc}")
    _object(doc, "GLB JSON")
    return doc, chunks[1][1] if len(chunks) > 1 else b""


def _reject_extensions(doc: dict[str, Any]) -> None:
    # glTF loaders may act on an extension payload even when extensionsUsed is
    # absent. In particular, EXT_mesh_gpu_instancing multiplies GPU geometry
    # without adding mesh nodes, bypassing the normal displayed-instance cap.
    for key in ("extensionsRequired", "extensionsUsed"):
        if _array(doc.get(key, []), key):
            _fail(f"GLB {key} are unsupported")
    stack: list[Any] = [doc]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if "extensions" in item:
                _fail("GLB extension payloads are unsupported")
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def _local_matrix(node: dict[str, Any], label: str) -> np.ndarray:
    if "matrix" in node:
        if any(key in node for key in ("translation", "rotation", "scale")):
            _fail(f"{label} mixes matrix with TRS")
        matrix = _finite_vector(node["matrix"], 16, f"{label}.matrix").reshape((4, 4), order="F")
    else:
        translation = _finite_vector(node.get("translation", [0, 0, 0]), 3, f"{label}.translation")
        rotation = _finite_vector(node.get("rotation", [0, 0, 0, 1]), 4, f"{label}.rotation")
        scale = _finite_vector(node.get("scale", [1, 1, 1]), 3, f"{label}.scale")
        norm = float(np.linalg.norm(rotation))
        if norm < 1e-8 or abs(norm - 1.0) > 1e-3:
            _fail(f"{label}.rotation must be a unit quaternion")
        if np.any(np.abs(scale) < 1e-8):
            _fail(f"{label}.scale is singular")
        x, y, z, w = rotation / norm
        matrix = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), translation[0]],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), translation[1]],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y), translation[2]],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        matrix[:3, :3] *= scale[np.newaxis, :]
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6) or abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
        _fail(f"{label} has an invalid or singular transform")
    return matrix


@dataclass(frozen=True)
class NodeRest:
    index: int
    name: str
    parent: int | None
    children: tuple[int, ...]
    local_matrix: np.ndarray
    world_matrix: np.ndarray


def _nodes(doc: dict[str, Any], limits: AssetLimits) -> tuple[NodeRest, ...]:
    raw = _array(doc.get("nodes", []), "nodes")
    if len(raw) > limits.max_nodes:
        _fail("too many scene nodes")
    parents: list[int | None] = [None] * len(raw)
    children: list[tuple[int, ...]] = []
    locals_: list[np.ndarray] = []
    for index, item in enumerate(raw):
        node = _object(item, f"nodes[{index}]")
        if "weights" in node:
            _fail("morph target node weights are unsupported")
        if "name" in node and (not isinstance(node["name"], str) or len(node["name"]) > 256):
            _fail("scene node name is invalid or too long")
        listed = _array(node.get("children", []), f"nodes[{index}].children")
        kids = tuple(_index(child, len(raw), f"nodes[{index}].children") for child in listed)
        if len(set(kids)) != len(kids):
            _fail(f"nodes[{index}] repeats a child")
        for child in kids:
            if child == index or parents[child] is not None:
                _fail("scene node has multiple parents or is its own child")
            parents[child] = index
        children.append(kids)
        locals_.append(_local_matrix(node, f"nodes[{index}]"))
    worlds: list[np.ndarray | None] = [None] * len(raw)
    visiting: set[int] = set()

    def world(index: int) -> np.ndarray:
        existing = worlds[index]
        if existing is not None:
            return existing
        if index in visiting:
            _fail("scene node hierarchy contains a cycle")
        visiting.add(index)
        parent = parents[index]
        result = locals_[index] if parent is None else world(parent) @ locals_[index]
        visiting.remove(index)
        if not np.isfinite(result).all():
            _fail("scene world transform is not finite")
        worlds[index] = result
        return result

    try:
        for index in range(len(raw)):
            world(index)
    except RecursionError:
        _fail("scene node hierarchy is too deep")
    scenes = _array(doc.get("scenes", []), "scenes")
    if len(scenes) > 64:
        _fail("too many scenes")
    if not scenes:
        _fail("GLB has no scene")
    for scene_index, scene in enumerate(scenes):
        roots = _array(_object(scene, f"scenes[{scene_index}]").get("nodes", []), "scene roots")
        if len(roots) != len(set(_index(root, len(raw), "scene root") for root in roots)):
            _fail("scene repeats a root node")
        for root in roots:
            root = _index(root, len(raw), "scene root")
            if parents[root] is not None:
                _fail("scene root has a parent")
    if "scene" in doc:
        _index(doc["scene"], len(doc.get("scenes", [])), "default scene")
    return tuple(NodeRest(index, str(raw[index].get("name", "")), parents[index], children[index],
                          locals_[index], worlds[index]) for index in range(len(raw)))


def _read_accessor(doc: dict[str, Any], binary: bytes, index: int, *, normalize: bool = True) -> np.ndarray:
    accessors = doc["accessors"]
    item = accessors[_index(index, len(accessors), "accessor")]
    component_type = item["componentType"]
    dtype = _DTYPES[component_type]
    width = _COMPONENTS[item["type"]]
    view = doc["bufferViews"][_index(item["bufferView"], len(doc["bufferViews"]), "accessor bufferView")]
    stride = view.get("byteStride", dtype.itemsize * width)
    start = view.get("byteOffset", 0) + item.get("byteOffset", 0)
    count = item["count"]
    # Validation guarantees that every strided element fits inside this view.
    if count == 0:
        values = np.empty((0, width), dtype=dtype)
    else:
        values = np.ndarray((count, width), dtype=dtype, buffer=binary, offset=start,
                            strides=(stride, dtype.itemsize)).copy()
    if normalize and item.get("normalized", False):
        if component_type == 5120:
            values = np.maximum(values.astype(np.float64) / 127.0, -1.0)
        elif component_type == 5121:
            values = values.astype(np.float64) / 255.0
        elif component_type == 5122:
            values = np.maximum(values.astype(np.float64) / 32767.0, -1.0)
        elif component_type == 5123:
            values = values.astype(np.float64) / 65535.0
        elif component_type == 5125:
            values = values.astype(np.float64) / 4294967295.0
    return values[:, 0] if width == 1 else values


@dataclass(frozen=True)
class CharacterAsset:
    asset_id: str
    display_name: str
    kind: str
    status: str
    sha256: str
    glb_bytes: bytes
    document: dict[str, Any]
    binary_chunk: bytes
    nodes: tuple[NodeRest, ...]
    skins: tuple[dict[str, Any], ...]
    vertex_count: int
    triangle_count: int
    image_pixels: int
    bounds: np.ndarray

    def read_accessor(self, index: int, *, normalize: bool = True) -> np.ndarray:
        return _read_accessor(self.document, self.binary_chunk, index, normalize=normalize)


def _validate_buffers(doc: dict[str, Any], binary: bytes, limits: AssetLimits) -> None:
    buffers = _array(doc.get("buffers", []), "buffers")
    if len(buffers) != 1:
        _fail("GLB must contain exactly one embedded buffer")
    buffer = _object(buffers[0], "buffers[0]")
    if "uri" in buffer:
        _fail("external or data-URI buffers are unsupported")
    size = _integer(buffer.get("byteLength"), "buffers[0].byteLength")
    if size > len(binary) or len(binary) - size > 3:
        _fail("embedded buffer length does not match BIN chunk")
    views = _array(doc.get("bufferViews", []), "bufferViews")
    if len(views) > limits.max_buffer_views:
        _fail("too many buffer views")
    for i, item in enumerate(views):
        view = _object(item, f"bufferViews[{i}]")
        if _index(view.get("buffer"), 1, "bufferView.buffer") != 0:
            _fail("bufferView refers to an external buffer")
        start = _integer(view.get("byteOffset", 0), "bufferView.byteOffset")
        length = _integer(view.get("byteLength"), "bufferView.byteLength")
        if start > size or length > size - start:
            _fail("bufferView exceeds embedded buffer")
        if "byteStride" in view:
            stride = _integer(view["byteStride"], "bufferView.byteStride", 4)
            if stride > 252 or stride % 4:
                _fail("bufferView.byteStride is invalid")
    accessors = _array(doc.get("accessors", []), "accessors")
    if len(accessors) > limits.max_accessors:
        _fail("too many accessors")
    expanded = 0
    for i, item in enumerate(accessors):
        accessor = _object(item, f"accessors[{i}]")
        if "sparse" in accessor:
            _fail("sparse accessors are unsupported")
        component_type = accessor.get("componentType")
        accessor_type = accessor.get("type")
        if type(component_type) is not int or not isinstance(accessor_type, str):
            _fail(f"accessors[{i}] has an unsupported component type or shape")
        dtype = _DTYPES.get(component_type)
        width = _COMPONENTS.get(accessor_type)
        if dtype is None or width is None:
            _fail(f"accessors[{i}] has an unsupported component type or shape")
        count = _integer(accessor.get("count"), "accessor.count")
        if count > limits.max_vertices * 4:
            _fail("accessor element count exceeds limit")
        if type(accessor.get("normalized", False)) is not bool:
            _fail("accessor.normalized must be boolean")
        if accessor.get("normalized", False) and accessor["componentType"] == 5126:
            _fail("float accessor cannot be normalized")
        if accessor["type"] in ("MAT2", "MAT3", "MAT4") and accessor["componentType"] != 5126:
            _fail("integer matrix accessors are unsupported")
        view = views[_index(accessor.get("bufferView"), len(views), "accessor.bufferView")]
        offset = _integer(accessor.get("byteOffset", 0), "accessor.byteOffset")
        item_bytes = dtype.itemsize * width
        stride = view.get("byteStride", item_bytes)
        if stride < item_bytes or stride % dtype.itemsize:
            _fail("accessor stride is smaller than an element or misaligned")
        if offset % dtype.itemsize or (view.get("byteOffset", 0) + offset) % dtype.itemsize:
            _fail("accessor offset is misaligned")
        needed = 0 if count == 0 else (count - 1) * stride + item_bytes
        if offset > view["byteLength"] or needed > view["byteLength"] - offset:
            _fail("accessor exceeds bufferView")
        expanded += count * item_bytes
        if expanded > limits.max_expanded_bytes:
            _fail("expanded accessor data exceeds limit")


def _validate_images(doc: dict[str, Any], binary: bytes, limits: AssetLimits) -> int:
    images = _array(doc.get("images", []), "images")
    if len(images) > limits.max_images:
        _fail("too many embedded images")
    total = 0
    views = doc["bufferViews"]
    for i, item in enumerate(images):
        image = _object(item, f"images[{i}]")
        if "uri" in image:
            _fail("external or data-URI images are unsupported")
        if image.get("mimeType") not in ("image/png", "image/jpeg"):
            _fail("only embedded PNG and JPEG images are supported")
        view = views[_index(image.get("bufferView"), len(views), "image.bufferView")]
        start = view.get("byteOffset", 0)
        content = binary[start:start + view["byteLength"]]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(content)) as opened:
                    if opened.format != ("PNG" if image["mimeType"] == "image/png" else "JPEG"):
                        _fail("image MIME type does not match its bytes")
                    pixels = opened.width * opened.height
                    if pixels <= 0 or pixels > limits.max_image_pixels:
                        _fail("image dimensions exceed limit")
                    opened.verify()
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombWarning,
                Image.DecompressionBombError) as exc:
            if isinstance(exc, AssetValidationError):
                raise
            _fail(f"invalid embedded image: {exc}")
        total += pixels
        if total * 4 > limits.max_expanded_bytes:
            _fail("expanded image data exceeds limit")
    textures = _array(doc.get("textures", []), "textures")
    samplers = _array(doc.get("samplers", []), "samplers")
    if len(textures) > limits.max_textures or len(samplers) > limits.max_textures:
        _fail("too many textures or samplers")
    for i, item in enumerate(textures):
        tex = _object(item, f"textures[{i}]")
        _index(tex.get("source"), len(images), "texture.source")
        if "sampler" in tex:
            _index(tex["sampler"], len(samplers), "texture.sampler")
    for i, sampler in enumerate(samplers):
        _object(sampler, f"samplers[{i}]")
    return total


def _validate_materials(doc: dict[str, Any], limits: AssetLimits) -> None:
    materials = _array(doc.get("materials", []), "materials")
    if len(materials) > limits.max_materials:
        _fail("too many materials")
    texture_count = len(doc.get("textures", []))
    for i, raw in enumerate(materials):
        material = _object(raw, f"materials[{i}]")
        pbr = _object(material.get("pbrMetallicRoughness", {}), "pbrMetallicRoughness")
        _finite_vector(pbr.get("baseColorFactor", [1, 1, 1, 1]), 4, "baseColorFactor")
        _finite_vector(material.get("emissiveFactor", [0, 0, 0]), 3, "emissiveFactor")
        for key in ("metallicFactor", "roughnessFactor"):
            value = _finite_scalar(pbr.get(key, 1), key)
            if not 0 <= value <= 1:
                _fail(f"{key} must be between zero and one")
        for owner, names in ((pbr, ("baseColorTexture", "metallicRoughnessTexture")),
                             (material, ("normalTexture", "occlusionTexture", "emissiveTexture"))):
            for key in names:
                if key in owner:
                    info = _object(owner[key], key)
                    _index(info.get("index"), texture_count, f"{key}.index")
                    if "texCoord" in info:
                        coordinate = _integer(info["texCoord"], f"{key}.texCoord")
                        if coordinate > 7:
                            _fail("material texture coordinate set exceeds limit")
        if material.get("alphaMode", "OPAQUE") not in ("OPAQUE", "MASK", "BLEND"):
            _fail("material alphaMode is unsupported")
        cutoff = _finite_scalar(material.get("alphaCutoff", 0.5), "material alphaCutoff")
        if not 0 <= cutoff <= 1:
            _fail("material alphaCutoff is invalid")


def _validate_meshes(doc: dict[str, Any], binary: bytes, limits: AssetLimits) -> tuple[int, int]:
    meshes = _array(doc.get("meshes", []), "meshes")
    accessors = doc["accessors"]
    materials = _array(doc.get("materials", []), "materials")
    vertices = triangles = 0
    if not meshes:
        _fail("GLB contains no mesh")
    if len(meshes) > limits.max_meshes:
        _fail("too many meshes")
    primitive_total = 0
    for mi, item in enumerate(meshes):
        mesh = _object(item, f"meshes[{mi}]")
        if "weights" in mesh:
            _fail("morph target mesh weights are unsupported")
        primitives = _array(mesh.get("primitives"), "mesh.primitives")
        if not primitives:
            _fail("mesh contains no primitives")
        primitive_total += len(primitives)
        if primitive_total > limits.max_primitives:
            _fail("too many mesh primitives")
        for pi, raw in enumerate(primitives):
            primitive = _object(raw, f"meshes[{mi}].primitives[{pi}]")
            if "targets" in primitive:
                _fail("morph target primitives are unsupported")
            if primitive.get("mode", 4) != 4:
                _fail("only triangle mesh primitives are supported")
            attrs = _object(primitive.get("attributes"), "primitive.attributes")
            if "POSITION" not in attrs or "NORMAL" not in attrs:
                _fail("triangle mesh requires POSITION and NORMAL")
            position_index = _index(attrs["POSITION"], len(accessors), "POSITION accessor")
            normal_index = _index(attrs["NORMAL"], len(accessors), "NORMAL accessor")
            position = accessors[position_index]
            normal = accessors[normal_index]
            if (position["type"], position["componentType"]) != ("VEC3", 5126):
                _fail("POSITION must be VEC3 FLOAT")
            if (normal["type"], normal["componentType"]) != ("VEC3", 5126):
                _fail("NORMAL must be VEC3 FLOAT")
            count = position["count"]
            if count == 0 or normal["count"] != count:
                _fail("mesh attribute counts do not match")
            for key, index in attrs.items():
                accessor = accessors[_index(index, len(accessors), f"{key} accessor")]
                if accessor["count"] != count:
                    _fail("mesh attribute counts do not match")
            if not np.isfinite(_read_accessor(doc, binary, position_index)).all() or not np.isfinite(_read_accessor(doc, binary, normal_index)).all():
                _fail("mesh POSITION or NORMAL contains non-finite values")
            if "indices" in primitive:
                idx = _index(primitive["indices"], len(accessors), "indices accessor")
                descriptor = accessors[idx]
                if descriptor["type"] != "SCALAR" or descriptor["componentType"] not in (5121, 5123, 5125):
                    _fail("indices must be unsigned scalar integers")
                values = _read_accessor(doc, binary, idx, normalize=False)
                if descriptor["count"] % 3 or np.any(values >= count):
                    _fail("triangle indices are invalid")
                triangle_count = descriptor["count"] // 3
            else:
                if count % 3:
                    _fail("unindexed triangle vertex count must be divisible by three")
                triangle_count = count // 3
            if "material" in primitive:
                material_index = _index(primitive["material"], len(materials), "primitive.material")
                material = materials[material_index]
                pbr = material.get("pbrMetallicRoughness", {})
                for owner, keys in ((pbr, ("baseColorTexture", "metallicRoughnessTexture")),
                                    (material, ("normalTexture", "occlusionTexture", "emissiveTexture"))):
                    for key in keys:
                        if key not in owner:
                            continue
                        semantic = f"TEXCOORD_{owner[key].get('texCoord', 0)}"
                        if semantic not in attrs:
                            _fail(f"textured primitive lacks {semantic}")
                        uv = accessors[attrs[semantic]]
                        if uv["type"] != "VEC2" or uv["componentType"] not in (5121, 5123, 5126):
                            _fail(f"{semantic} has an unsupported accessor type")
                        if uv["componentType"] != 5126 and not uv.get("normalized", False):
                            _fail(f"{semantic} integer coordinates must be normalized")
            vertices += count
            triangles += triangle_count
            if vertices > limits.max_vertices or triangles > limits.max_triangles:
                _fail("mesh geometry exceeds vertex or triangle limit")
    return vertices, triangles


def _validate_skins(doc: dict[str, Any], binary: bytes, nodes: tuple[NodeRest, ...],
                    limits: AssetLimits) -> tuple[dict[str, Any], ...]:
    raw_skins = _array(doc.get("skins", []), "skins")
    if len(raw_skins) > limits.max_skins:
        _fail("too many skins")
    raw_nodes = doc.get("nodes", [])
    meshes = doc["meshes"]
    skin_sets: set[tuple[int, ...]] = set()
    for si, raw in enumerate(raw_skins):
        skin = _object(raw, f"skins[{si}]")
        joints = _array(skin.get("joints"), "skin.joints")
        if not joints:
            _fail("skin contains no joints")
        joints = tuple(_index(value, len(nodes), "skin.joint") for value in joints)
        if len(set(joints)) != len(joints):
            _fail("skin repeats a joint")
        skin_sets.add(joints)
        if "skeleton" in skin:
            _index(skin["skeleton"], len(nodes), "skin.skeleton")
        if "inverseBindMatrices" in skin:
            accessor = _index(skin["inverseBindMatrices"], len(doc["accessors"]), "skin.inverseBindMatrices")
            descriptor = doc["accessors"][accessor]
            if descriptor["type"] != "MAT4" or descriptor["componentType"] != 5126 or descriptor["count"] != len(joints):
                _fail("inverse bind matrices must be one MAT4 FLOAT per joint")
            matrices = _read_accessor(doc, binary, accessor).reshape((-1, 4, 4)).transpose((0, 2, 1))
            if not np.isfinite(matrices).all() or np.any(np.abs(np.linalg.det(matrices)) < 1e-12):
                _fail("inverse bind matrix is non-finite or singular")
    if len(skin_sets) > 1:
        _fail("multiple independent armatures require separate character assets")
    attached = 0
    for ni, raw in enumerate(raw_nodes):
        node = _object(raw, f"nodes[{ni}]")
        if "mesh" in node:
            _index(node["mesh"], len(meshes), "node.mesh")
        if "skin" not in node:
            if "mesh" in node:
                for primitive in meshes[node["mesh"]]["primitives"]:
                    if any(key.startswith("JOINTS_") or key.startswith("WEIGHTS_")
                           for key in primitive["attributes"]):
                        _fail("mesh has skin attributes but its node has no skin")
            continue
        skin_index = _index(node["skin"], len(raw_skins), "node.skin")
        if "mesh" not in node:
            _fail("skinned node has no mesh")
        attached += 1
        skin = raw_skins[skin_index]
        joints = skin["joints"]
        for primitive in meshes[node["mesh"]]["primitives"]:
            attrs = primitive["attributes"]
            if "JOINTS_0" not in attrs or "WEIGHTS_0" not in attrs:
                _fail("skinned mesh requires JOINTS_0 and WEIGHTS_0")
            ji, wi = attrs["JOINTS_0"], attrs["WEIGHTS_0"]
            jdesc, wdesc = doc["accessors"][ji], doc["accessors"][wi]
            if jdesc["type"] != "VEC4" or jdesc["componentType"] not in (5121, 5123):
                _fail("JOINTS_0 must be VEC4 unsigned byte or short")
            if jdesc.get("normalized", False):
                _fail("JOINTS_0 must not be normalized")
            if wdesc["type"] != "VEC4" or wdesc["componentType"] not in (5121, 5123, 5126):
                _fail("WEIGHTS_0 must be VEC4 float or normalized unsigned integer")
            if wdesc["componentType"] != 5126 and not wdesc.get("normalized", False):
                _fail("integer skin weights must be normalized")
            if any(key.startswith("JOINTS_") or key.startswith("WEIGHTS_") for key in attrs if key not in ("JOINTS_0", "WEIGHTS_0")):
                _fail("more than four skin influences are unsupported")
            joint_values = _read_accessor(doc, binary, ji, normalize=False)
            weights = _read_accessor(doc, binary, wi)
            if np.any(joint_values >= len(joints)):
                _fail("JOINTS_0 references a joint outside the skin")
            if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1):
                _fail("WEIGHTS_0 contains invalid weights")
            sums = weights.sum(axis=1)
            if np.any(np.abs(sums - 1.0) > 0.02):
                _fail("skin weights must sum to one")
    if raw_skins and attached == 0:
        _fail("GLB defines a skin but attaches it to no mesh")
    return tuple(raw_skins)


def _scene_bounds(doc: dict[str, Any], binary: bytes, nodes: tuple[NodeRest, ...],
                  limits: AssetLimits) -> np.ndarray:
    selected = doc.get("scene", 0)
    roots = doc["scenes"][selected].get("nodes", [])
    reachable: set[int] = set()
    pending = list(roots)
    while pending:
        index = pending.pop()
        if index in reachable:
            continue
        reachable.add(index)
        pending.extend(nodes[index].children)
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    found = False
    displayed_vertices = 0
    displayed_triangles = 0
    for node in nodes:
        if node.index not in reachable:
            continue
        raw = doc["nodes"][node.index]
        if "mesh" not in raw:
            continue
        skin_matrices = None
        if "skin" in raw:
            skin = doc["skins"][raw["skin"]]
            if any(joint not in reachable for joint in skin["joints"]):
                _fail("selected scene omits a skinned mesh joint")
            if "inverseBindMatrices" in skin:
                inverse_binds = (_read_accessor(doc, binary, skin["inverseBindMatrices"])
                                 .reshape((-1, 4, 4)).transpose((0, 2, 1)))
            else:
                inverse_binds = np.repeat(np.eye(4)[np.newaxis, :, :], len(skin["joints"]), axis=0)
            skin_matrices = np.stack([nodes[joint].world_matrix @ inverse_binds[i]
                                      for i, joint in enumerate(skin["joints"])])
        for primitive in doc["meshes"][raw["mesh"]]["primitives"]:
            position_descriptor = doc["accessors"][primitive["attributes"]["POSITION"]]
            displayed_vertices += position_descriptor["count"]
            displayed_triangles += (doc["accessors"][primitive["indices"]]["count"]
                                    if "indices" in primitive else position_descriptor["count"]) // 3
            if displayed_vertices > limits.max_vertices or displayed_triangles > limits.max_triangles:
                _fail("displayed mesh instances exceed geometry limit")
            position = _read_accessor(doc, binary, primitive["attributes"]["POSITION"])
            if skin_matrices is None:
                transformed = position @ node.world_matrix[:3, :3].T + node.world_matrix[:3, 3]
                if not np.isfinite(transformed).all():
                    _fail("transformed mesh bounds are not finite")
                minimum = np.minimum(minimum, transformed.min(axis=0))
                maximum = np.maximum(maximum, transformed.max(axis=0))
            else:
                joints = _read_accessor(doc, binary, primitive["attributes"]["JOINTS_0"], normalize=False)
                weights = _read_accessor(doc, binary, primitive["attributes"]["WEIGHTS_0"])
                # Bound temporary matrices even when the source vertex budget is
                # large. The skinned mesh node transform cancels in glTF's world
                # skinning equation; joint world × inverse bind drives the rest.
                for start in range(0, len(position), 16384):
                    stop = min(start + 16384, len(position))
                    points = position[start:stop]
                    transformed = np.zeros((len(points), 3), dtype=np.float64)
                    for influence in range(4):
                        matrices = skin_matrices[joints[start:stop, influence]]
                        moved = (matrices[:, :3, :3] * points[:, np.newaxis, :]).sum(axis=2)
                        moved += matrices[:, :3, 3]
                        transformed += weights[start:stop, influence, np.newaxis] * moved
                    if not np.isfinite(transformed).all():
                        _fail("skinned mesh bounds are not finite")
                    minimum = np.minimum(minimum, transformed.min(axis=0))
                    maximum = np.maximum(maximum, transformed.max(axis=0))
            found = True
    if not found:
        _fail("selected GLB scene has no mesh node to display")
    return np.stack((minimum, maximum))


def _display_name(value: str) -> str:
    if not isinstance(value, str):
        _fail("display name must be text")
    name = _SAFE_NAME.sub("_", value).strip(" ._")[:120]
    if not name:
        _fail("display name is empty")
    return name


def inspect_glb(data: bytes, *, display_name: str = "Imported GLB", limits: AssetLimits = DEFAULT_LIMITS) -> CharacterAsset:
    """Validate a self-contained GLB and return its source plus rest-pose data."""
    name = _display_name(display_name)
    doc, binary = _parse_glb(data, limits)
    asset = _object(doc.get("asset"), "asset")
    if asset.get("version") != "2.0":
        _fail("GLB asset.version must be 2.0")
    _reject_extensions(doc)
    for key in ("nodes", "scenes", "meshes", "bufferViews", "accessors"):
        if key not in doc:
            _fail(f"GLB is missing {key}")
    _validate_buffers(doc, binary, limits)
    nodes = _nodes(doc, limits)
    pixels = _validate_images(doc, binary, limits)
    _validate_materials(doc, limits)
    vertices, triangles = _validate_meshes(doc, binary, limits)
    skins = _validate_skins(doc, binary, nodes, limits)
    bounds = _scene_bounds(doc, binary, nodes, limits)
    if sum(item["count"] * _DTYPES[item["componentType"]].itemsize * _COMPONENTS[item["type"]]
           for item in doc["accessors"]) + pixels * 4 > limits.max_expanded_bytes:
        _fail("combined expanded geometry and images exceed limit")
    digest = hashlib.sha256(data).hexdigest()
    kind = "skinned" if skins else "static"
    return CharacterAsset(digest, name, kind, "mapping_required" if skins else "static_preview",
                          digest, data, doc, binary, nodes, skins, vertices, triangles, pixels, bounds)


def import_glb(data: bytes, storage_root: str | Path, *, display_name: str = "Imported GLB",
               limits: AssetLimits = DEFAULT_LIMITS) -> CharacterAsset:
    """Validate and atomically store the original GLB under its content hash."""
    asset = inspect_glb(data, display_name=display_name, limits=limits)
    root = Path(storage_root)
    if root.is_symlink():
        _fail("asset storage root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    folder = root / asset.asset_id
    if folder.is_symlink():
        _fail("stored asset folder must not be a symlink")
    folder.mkdir(mode=0o700, exist_ok=True)
    destination = folder / "asset.glb"
    if destination.exists():
        if (destination.is_symlink() or destination.stat().st_size > limits.max_file_bytes or
                hashlib.sha256(destination.read_bytes()).hexdigest() != asset.sha256):
            _fail("stored asset does not match its content hash")
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(destination, flags, 0o600)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    metadata = {"schema_version": 1, "asset_id": asset.asset_id, "display_name": asset.display_name,
                "kind": asset.kind, "status": asset.status, "sha256": asset.sha256,
                "vertex_count": asset.vertex_count, "triangle_count": asset.triangle_count,
                "image_pixels": asset.image_pixels}
    temporary = folder / f"manifest.{os.urandom(8).hex()}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(metadata, output, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, folder / "manifest.json")
    finally:
        temporary.unlink(missing_ok=True)
    return asset


def load_character_asset(storage_root: str | Path, asset_id: str,
                         *, limits: AssetLimits = DEFAULT_LIMITS) -> CharacterAsset:
    """Read a previously imported asset by its generated ID, never a user path."""
    if not isinstance(asset_id, str) or not _ID.fullmatch(asset_id):
        _fail("invalid asset ID")
    folder = Path(storage_root) / asset_id
    manifest_path, glb_path = folder / "manifest.json", folder / "asset.glb"
    if Path(storage_root).is_symlink() or folder.is_symlink() or manifest_path.is_symlink() or glb_path.is_symlink():
        _fail("stored asset contains a symlink")
    try:
        if glb_path.stat().st_size > limits.max_file_bytes:
            _fail("stored GLB exceeds file size limit")
        if manifest_path.stat().st_size > 8192:
            _fail("stored manifest exceeds size limit")
        metadata = _object(json.loads(manifest_path.read_text(encoding="utf-8")), "stored manifest")
        if metadata.get("schema_version") != 1 or metadata.get("asset_id") != asset_id:
            _fail("stored manifest is invalid")
        asset = inspect_glb(glb_path.read_bytes(), display_name=metadata["display_name"], limits=limits)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        _fail(f"cannot load stored asset: {exc}")
    if asset.sha256 != asset_id or metadata.get("sha256") != asset.sha256:
        _fail("stored asset hash does not match")
    return asset
