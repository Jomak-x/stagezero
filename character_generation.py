"""Bounded validation for generated, self-contained character GLBs."""
import json
import struct

MAX_GLB_BYTES = 40 * 1024 * 1024

def _data_uri(uri: object) -> bool:
    return isinstance(uri, str) and uri.lower().startswith("data:") and "," in uri


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate character GLB JSON key")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ValueError("Invalid character GLB JSON number")


def validate_glb(data: bytes) -> bytes:
    """Validate GLB framing and core resource containment, returning the original bytes.

    This is an ingestion check, not a full glTF renderer/parser. GLB JSON may include
    embedded data URIs; buffer and image references to external files are refused.
    """
    if not isinstance(data, bytes) or len(data) > MAX_GLB_BYTES or len(data) < 20:
        raise ValueError("Invalid or oversized character GLB")
    magic, version, declared = struct.unpack_from("<4sII", data)
    if magic != b"glTF" or version != 2 or declared != len(data) or declared % 4:
        raise ValueError("Invalid character GLB header")

    chunks = []
    offset = 12
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("Truncated character GLB chunk")
        length, kind = struct.unpack_from("<I4s", data, offset)
        offset += 8
        if length % 4 or offset + length > len(data):
            raise ValueError("Invalid character GLB chunk length")
        chunks.append((kind, data[offset:offset + length]))
        offset += length
    if not chunks or chunks[0][0] != b"JSON" or len(chunks) > 2 or (
        len(chunks) == 2 and chunks[1][0] != b"BIN\0"
    ):
        raise ValueError("Invalid character GLB chunk layout")
    try:
        document = json.loads(chunks[0][1].rstrip(b" ").decode("utf-8"),
                              object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (UnicodeError, json.JSONDecodeError, ValueError):
        raise ValueError("Invalid character GLB JSON") from None
    if not isinstance(document, dict) or not isinstance(document.get("asset"), dict) or (
        document["asset"].get("version") != "2.0"
    ):
        raise ValueError("Invalid character glTF asset")

    buffers = document.get("buffers", [])
    images = document.get("images", [])
    views = document.get("bufferViews", [])
    if not all(isinstance(value, list) for value in (buffers, images, views)):
        raise ValueError("Invalid character GLB resources")
    embedded_buffer_count = 0
    for buffer in buffers:
        if not isinstance(buffer, dict) or type(buffer.get("byteLength")) is not int or buffer["byteLength"] < 0:
            raise ValueError("Invalid character GLB buffer")
        if "uri" in buffer:
            if not _data_uri(buffer["uri"]):
                raise ValueError("Character GLB references an external buffer")
        else:
            embedded_buffer_count += 1
            if embedded_buffer_count > 1 or len(chunks) < 2 or buffer["byteLength"] > len(chunks[1][1]) or (
                len(chunks[1][1]) - buffer["byteLength"] > 3
            ):
                raise ValueError("Character GLB is missing embedded buffer data")
    for view in views:
        if not isinstance(view, dict) or type(view.get("buffer")) is not int or not (0 <= view["buffer"] < len(buffers)):
            raise ValueError("Invalid character GLB buffer view")
        start = view.get("byteOffset", 0)
        length = view.get("byteLength")
        if type(start) is not int or type(length) is not int or start < 0 or length < 0 or (
            start + length > buffers[view["buffer"]]["byteLength"]
        ):
            raise ValueError("Invalid character GLB buffer view")
    for image in images:
        if not isinstance(image, dict):
            raise ValueError("Invalid character GLB image")
        if "uri" in image:
            if not _data_uri(image["uri"]):
                raise ValueError("Character GLB references an external image")
        elif type(image.get("bufferView")) is not int or not (0 <= image["bufferView"] < len(views)):
            raise ValueError("Character GLB image is missing embedded data")
    return data
