"""Bounded procedural asset recipes and compact colored mesh compilation.

Recipes are data only. All coordinates are local to a unit-sized object whose
centre is the origin. This module deliberately does not evaluate Python, load
files, or depend on a general mesh library.
"""

from __future__ import annotations

import json
import math
import re
import struct
from numbers import Integral, Real

import numpy as np


MAX_ASSETS = 16
MAX_PARTS = 64
MAX_EXPANDED_SHAPES = 512
SHAPES = frozenset(('box', 'sphere', 'cylinder', 'cone'))
_ID = re.compile(r'[A-Za-z0-9_-]{1,64}\Z')


def _number(value, label, lo, hi, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{label} must be a finite number')
    value = float(value)
    if not math.isfinite(value) or value < lo or value > hi or (positive and value == 0):
        raise ValueError(f'{label} must be finite and between {lo} and {hi}')
    return value


def _vec(value, label, lo, hi, *, positive=False):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f'{label} must contain three numbers')
    return [_number(v, f'{label}[{i}]', lo, hi, positive=positive) for i, v in enumerate(value)]


def _rotation(degrees):
    x, y, z = np.radians(degrees)
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    return np.asarray(((cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
                       (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
                       (-sy, cy * sx, cy * cx)), dtype=np.float64)


def expanded_count(asset):
    """Count primitives after lattice repetition (validated or raw recipes)."""
    return sum(math.prod(part.get('repeat', {}).get('count', [1, 1, 1])) for part in asset['parts'])


def triangle_count(asset):
    """Exact triangle cost of the renderer's bounded primitive tessellation."""
    faces = {'box':12, 'sphere':168, 'cylinder':48, 'cone':24}
    return sum(faces[p['shape']] * math.prod(p.get('repeat', {}).get('count', [1,1,1]))
               for p in asset['parts'])


def validate_assets(value, *, check_bounds=True):
    """Return detached canonical recipes, rejecting unsafe or oversized data."""
    if not isinstance(value, list) or len(value) > MAX_ASSETS:
        raise ValueError(f'assets must be a list of at most {MAX_ASSETS} items')
    assets, seen = [], set()
    for ai, asset in enumerate(value):
        label = f'assets[{ai}]'
        if not isinstance(asset, dict) or set(asset) != {'id', 'name', 'parts'}:
            raise ValueError(f'{label} has missing or unknown fields')
        ident, name, parts = asset['id'], asset['name'], asset['parts']
        if not isinstance(ident, str) or not _ID.fullmatch(ident) or ident in seen:
            raise ValueError(f'{label}.id is invalid or duplicated')
        seen.add(ident)
        if not isinstance(name, str) or not 1 <= len(name) <= 80 or not name.strip() or any(ord(c) < 32 for c in name):
            raise ValueError(f'{label}.name is invalid')
        if not isinstance(parts, list) or not 1 <= len(parts) <= MAX_PARTS:
            raise ValueError(f'{label}.parts must contain 1 to {MAX_PARTS} parts')
        clean_parts, total = [], 0
        for pi, part in enumerate(parts):
            p_label = f'{label}.parts[{pi}]'
            required = {'shape', 'position', 'size', 'color'}
            if not isinstance(part, dict) or not required <= set(part) or set(part) - (required | {'rotation', 'repeat'}):
                raise ValueError(f'{p_label} has missing or unknown fields')
            shape = part['shape']
            if not isinstance(shape, str) or shape not in SHAPES:
                raise ValueError(f'{p_label}.shape is unsupported')
            position = _vec(part['position'], f'{p_label}.position', -1, 1)
            size = _vec(part['size'], f'{p_label}.size', 0.001, 1, positive=True)
            color = part['color']
            if not isinstance(color, (list, tuple)) or len(color) != 3 or any(
                isinstance(c, bool) or not isinstance(c, Integral) or not 0 <= c <= 255 for c in color
            ):
                raise ValueError(f'{p_label}.color must contain three RGB bytes')
            clean = {'shape': shape, 'position': position, 'size': size, 'color': [int(c) for c in color]}
            rotation = [0., 0., 0.]
            if 'rotation' in part:
                rotation = _vec(part['rotation'], f'{p_label}.rotation', -360, 360)
                clean['rotation'] = rotation
            count, step = [1, 1, 1], [0., 0., 0.]
            if 'repeat' in part:
                repeat = part['repeat']
                if not isinstance(repeat, dict) or set(repeat) != {'count', 'step'}:
                    raise ValueError(f'{p_label}.repeat has missing or unknown fields')
                raw_count = repeat['count']
                if not isinstance(raw_count, (list, tuple)) or len(raw_count) != 3 or any(
                    isinstance(c, bool) or not isinstance(c, Integral) or not 1 <= c <= 32 for c in raw_count
                ):
                    raise ValueError(f'{p_label}.repeat.count must have three integers from 1 to 32')
                count = [int(c) for c in raw_count]
                step = _vec(repeat['step'], f'{p_label}.repeat.step', -1, 1)
                clean['repeat'] = {'count': count, 'step': step}
            total += math.prod(count)
            if total > MAX_EXPANDED_SHAPES:
                raise ValueError(f'{label} exceeds {MAX_EXPANDED_SHAPES} expanded shapes')
            extent = np.abs(_rotation(rotation)) @ (np.asarray(size) / 2)
            first = np.asarray(position)
            last = first + (np.asarray(count) - 1) * np.asarray(step)
            lower, upper = np.minimum(first, last) - extent, np.maximum(first, last) + extent
            if check_bounds and (np.any(lower < -0.500001) or np.any(upper > 0.500001)):
                raise ValueError(f'{p_label} extends beyond the local unit bounds')
            clean_parts.append(clean)
        assets.append({'id': ident, 'name': name.strip(), 'parts': clean_parts})
    return assets


def _primitive(shape):
    """Unit vertices and outward triangles for a centred shape."""
    if shape == 'box':
        v = np.asarray([(x, y, z) for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)], dtype=np.float64)
        f = np.asarray([(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6),
                        (0, 1, 5), (0, 5, 4), (2, 6, 7), (2, 7, 3),
                        (0, 4, 6), (0, 6, 2), (1, 3, 7), (1, 7, 5)], dtype=np.uint32)
        return v, f
    sides = 12
    angles = np.arange(sides) * 2 * np.pi / sides
    if shape in ('cylinder', 'cone'):
        bottom = np.stack((.5 * np.cos(angles), np.full(sides, -.5), .5 * np.sin(angles)), axis=1)
        if shape == 'cylinder':
            top = bottom.copy(); top[:, 1] = .5
            vertices = np.vstack((bottom, top, [[0, -.5, 0], [0, .5, 0]]))
            faces = []
            for i in range(sides):
                j = (i + 1) % sides
                faces.extend(((i, sides + i, sides + j), (i, sides + j, j), (2*sides, j, i), (2*sides+1, sides+i, sides+j)))
        else:
            vertices = np.vstack((bottom, [[0, -.5, 0], [0, .5, 0]]))
            faces = [(i, sides+1, (i+1) % sides) for i in range(sides)]
            faces += [(sides, (i+1) % sides, i) for i in range(sides)]
        return vertices, np.asarray(faces, dtype=np.uint32)
    # UV sphere, 8 latitude bands and 12 longitude sectors.
    vertices = [[0, .5, 0]]
    for band in range(1, 8):
        phi = np.pi * band / 8
        vertices.extend((.5 * math.sin(phi) * math.cos(a), .5 * math.cos(phi),
                         .5 * math.sin(phi) * math.sin(a)) for a in angles)
    vertices.append((0, -.5, 0))
    faces = []
    for i in range(sides):
        faces.append((0, 1+(i+1)%sides, 1+i))
    for band in range(6):
        upper, lower = 1+band*sides, 1+(band+1)*sides
        for i in range(sides):
            j = (i+1) % sides
            faces.extend(((upper+i, upper+j, lower+i), (upper+j, lower+j, lower+i)))
    last = len(vertices)-1
    for i in range(sides):
        faces.append((last, last-sides+i, last-sides+(i+1)%sides))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.uint32)


def _outward(vertices, faces):
    """Correct triangle winding for each convex primitive."""
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    inward = np.sum(normals * triangles.mean(axis=1), axis=1) < 0
    faces = faces.copy()
    faces[inward] = faces[inward][:, [0, 2, 1]]
    return faces


def compile_asset(asset):
    """Compile a recipe to one local mesh: vertices, faces, per-vertex RGB."""
    clean = validate_assets([asset])[0]
    vertices, faces, colors = [], [], []
    for part in clean['parts']:
        base, triangles = _primitive(part['shape'])
        triangles = _outward(base, triangles)
        transformed = (base * np.asarray(part['size'])) @ _rotation(part.get('rotation', [0, 0, 0])).T
        repeat = part.get('repeat', {'count': [1, 1, 1], 'step': [0, 0, 0]})
        for ix in range(repeat['count'][0]):
            for iy in range(repeat['count'][1]):
                for iz in range(repeat['count'][2]):
                    offset = np.asarray(part['position']) + np.asarray(repeat['step']) * [ix, iy, iz]
                    start = sum(len(v) for v in vertices)
                    vertices.append((transformed + offset).astype(np.float32))
                    faces.append(triangles + start)
                    colors.append(np.tile(np.asarray(part['color'], dtype=np.uint8), (len(base), 1)))
    return np.concatenate(vertices), np.concatenate(faces), np.concatenate(colors)


def mesh_to_glb(vertices, faces, vertex_colors):
    """Encode one vertex-colored triangle mesh as a minimal binary glTF file."""
    verts = np.asarray(vertices, dtype='<f4')
    triangles = np.asarray(faces, dtype='<u4')
    colors = np.asarray(vertex_colors, dtype=np.uint8)
    if verts.ndim != 2 or verts.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3 or colors.shape != verts.shape:
        raise ValueError('mesh arrays have invalid shapes')
    if not np.isfinite(verts).all() or np.any(triangles >= len(verts)):
        raise ValueError('mesh arrays contain invalid values')
    # Duplicate triangle vertices so normals stay crisp at edges and corners.
    flat = verts[triangles.reshape(-1)]
    # Authored asset palettes use display (sRGB) bytes. glTF COLOR_0 is a
    # *linear* multiplier, so writing those bytes unchanged makes midtones
    # render far too bright (green foliage turns mint and stone turns white).
    srgb = colors[triangles.reshape(-1)].astype(np.float32) / 255.0
    rgb = np.rint(np.where(srgb <= 0.04045, srgb / 12.92,
                           ((srgb + 0.055) / 1.055) ** 2.4) * 255.0).astype(np.uint8)
    tri = flat.reshape(-1, 3, 3)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals = (normals / np.maximum(lengths[:, None], 1e-12)).repeat(3, axis=0).astype('<f4')
    indices = np.arange(len(flat), dtype='<u4')
    chunks = [flat.astype('<f4').tobytes(), normals.tobytes(), rgb.tobytes(), indices.tobytes()]
    offsets, blob = [], bytearray()
    for chunk in chunks:
        offsets.append(len(blob)); blob.extend(chunk); blob.extend(b'\0' * (-len(blob) % 4))
    views = [{'buffer': 0, 'byteOffset': off, 'byteLength': len(chunk)} for off, chunk in zip(offsets, chunks)]
    accessors = [
        {'bufferView': 0, 'componentType': 5126, 'count': len(flat), 'type': 'VEC3',
         'min': flat.min(axis=0).tolist(), 'max': flat.max(axis=0).tolist()},
        {'bufferView': 1, 'componentType': 5126, 'count': len(flat), 'type': 'VEC3'},
        {'bufferView': 2, 'componentType': 5121, 'count': len(flat), 'type': 'VEC3', 'normalized': True},
        {'bufferView': 3, 'componentType': 5125, 'count': len(indices), 'type': 'SCALAR'},
    ]
    document = {'asset': {'version': '2.0', 'generator': 'Shellhacks procedural assets'},
                'buffers': [{'byteLength': len(blob)}], 'bufferViews': views, 'accessors': accessors,
                'materials': [{'pbrMetallicRoughness': {'baseColorFactor': [1, 1, 1, 1], 'metallicFactor': 0, 'roughnessFactor': 1}, 'doubleSided': True}],
                'meshes': [{'primitives': [{'attributes': {'POSITION': 0, 'NORMAL': 1, 'COLOR_0': 2}, 'indices': 3, 'material': 0}]}],
                'nodes': [{'mesh': 0}], 'scenes': [{'nodes': [0]}], 'scene': 0}
    header = json.dumps(document, separators=(',', ':'), allow_nan=False).encode('utf8')
    header += b' ' * (-len(header) % 4)
    total = 12 + 8 + len(header) + 8 + len(blob)
    return (struct.pack('<4sII', b'glTF', 2, total) + struct.pack('<I4s', len(header), b'JSON') + header +
            struct.pack('<I4s', len(blob), b'BIN\0') + bytes(blob))
