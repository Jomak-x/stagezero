"""Conservative reference detail transfer onto an existing character UV atlas.

This is a texture operation, not a geometry or likeness reconstruction step.
The reference must be a front-facing RGBA cutout in the same pose as a Y-up
mesh whose front faces +Z. The existing bake is retained on hidden surfaces.
No background-removal heuristics are used here: callers supply a real cutout.
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image


def _raster_samples(points, faces, width, height, *, budget=700_000):
    """Yield vectorized triangle/pixel barycentrics with bounded memory."""
    triangles = np.asarray(points, dtype=np.float32)[faces]
    low = np.maximum(np.floor(triangles.min(axis=1)).astype(np.int32), 0)
    high = np.minimum(np.ceil(triangles.max(axis=1)).astype(np.int32), [width, height])
    size = np.maximum(high - low, 0)
    area = size[:, 0].astype(np.int64) * size[:, 1]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    denominator = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (
        c[:, 0] - b[:, 0]
    ) * (a[:, 1] - c[:, 1])
    area[np.abs(denominator) < 1e-8] = 0
    active = np.flatnonzero(area)
    # Even a single enormous triangle is split, rather than exceeding budget.
    cumulative = np.cumsum(area[active])
    total = int(cumulative[-1]) if len(cumulative) else 0
    for start in range(0, total, budget):
        offsets = np.arange(start, min(start + budget, total), dtype=np.int64)
        local = np.searchsorted(cumulative, offsets, side="right")
        face = active[local]
        previous = np.where(local > 0, cumulative[np.maximum(local - 1, 0)], 0)
        offsets -= previous
        x = low[face, 0] + offsets % size[face, 0]
        y = low[face, 1] + offsets // size[face, 0]
        dx, dy = x + 0.5 - c[face, 0], y + 0.5 - c[face, 1]
        w0 = ((b[face, 1] - c[face, 1]) * dx + (c[face, 0] - b[face, 0]) * dy) / denominator[face]
        w1 = ((c[face, 1] - a[face, 1]) * dx + (a[face, 0] - c[face, 0]) * dy) / denominator[face]
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        yield x[inside], y[inside], face[inside], np.stack(
            (w0[inside], w1[inside], w2[inside]), axis=1
        ).astype(np.float32)


def _bilinear(image, xy):
    """Sample pixel-center coordinates, clamped at image boundaries."""
    xy = np.asarray(xy) - 0.5
    x = np.clip(xy[:, 0], 0, image.shape[1] - 1)
    y = np.clip(xy[:, 1], 0, image.shape[0] - 1)
    x0, y0 = x.astype(np.int32), y.astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, image.shape[1] - 1), np.minimum(y0 + 1, image.shape[0] - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return ((image[y0, x0] * (1 - fx) + image[y0, x1] * fx) * (1 - fy)
            + (image[y1, x0] * (1 - fx) + image[y1, x1] * fx) * fy)


def _reference_pixels(reference):
    """Extend trusted foreground colors across small reconstruction errors.

    Fixed alpha erosion can erase narrow fingers while the reconstructed mesh
    extends a few pixels beyond the input silhouette. Use only nearly opaque
    source colors, retain them exactly, and fade a short nearest-color extension
    to zero. Transparent/background RGB is never a color source. The caller's
    face-normal and depth tests still exclude rear and occluded surfaces.
    """
    from scipy.ndimage import distance_transform_edt

    pixels = np.asarray(reference, dtype=np.float32).copy()
    trusted = pixels[:, :, 3] >= 250
    if not trusted.any():
        pixels[:, :, 3] = 0
        return pixels
    distance, nearest = distance_transform_edt(~trusted, return_indices=True)
    radius = max(1, round(max(reference.size) / 192))
    pixels[:, :, :3] = pixels[nearest[0], nearest[1], :3]
    confidence = np.clip(1 - np.maximum(distance - radius, 0) / radius, 0, 1)
    # The projection's alpha-confidence ramp starts at 224.
    pixels[:, :, 3] = 224 + 31 * confidence
    return pixels


def _pad_projected_atlas(atlas, occupied, projected, *, padding=4):
    """Extend edge colors into UV gutters without touching another surface.

    Texture filtering samples outside an island, so updating only interior
    pixels leaves visible lines of the old bake at UV seams. Grow *all* islands
    together so an enhanced island cannot claim a neighboring rear island's
    gutter. Only gutters whose closest propagated source was enhanced change.
    Alpha and every occupied texel remain unchanged.
    """
    known = occupied.copy()
    has_detail = np.asarray(projected, dtype=bool).copy() & known
    padded = np.zeros_like(known)
    height, width = known.shape
    for _ in range(padding):
        previous = known.copy()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            target = (slice(max(0, dy), min(height, height + dy)),
                      slice(max(0, dx), min(width, width + dx)))
            source = (slice(max(0, -dy), min(height, height - dy)),
                      slice(max(0, -dx), min(width, width - dx)))
            take = ~known[target] & previous[source]
            detail = take & has_detail[source]
            atlas[target][detail, :3] = atlas[source][detail, :3]
            known[target][take] = True
            has_detail[target][take] = has_detail[source][take]
            padded[target][detail] = True
    return int(np.count_nonzero(padded))


def enhance_front_texture(mesh, reference: Image.Image, *, strength=0.85,
                          visibility_size=1024, min_silhouette_iou=0.72):
    """Return ``(mesh_copy, diagnostics)`` with visible reference detail baked in.

    The mesh is never modified in place. Unsupported materials, an opaque or
    empty reference, and poor silhouette alignment safely return an unchanged
    copy with ``applied=False``. Only a single-material UV atlas is supported.
    Affine silhouette fitting cannot correct pose or perspective differences;
    the IoU guard is necessary but does not establish facial landmark alignment.
    ``strength`` controls a soft blend with the original appearance.
    """
    started = time.monotonic()
    result = mesh.copy()
    diagnostics = {"applied": False, "reason": "", "changed_texels": 0}

    def finish(reason):
        diagnostics.update(reason=reason, seconds=round(time.monotonic() - started, 3))
        return result, diagnostics

    if not 0 <= strength <= 1 or not 64 <= visibility_size <= 2048:
        raise ValueError("Invalid texture projection strength or visibility size")
    if not 0 <= min_silhouette_iou <= 1:
        raise ValueError("Invalid minimum silhouette overlap")
    if strength == 0:
        return finish("zero_strength")
    material = getattr(result.visual, "material", None)
    texture = getattr(material, "baseColorTexture", None)
    uv = getattr(result.visual, "uv", None)
    if texture is None or uv is None or len(uv) != len(result.vertices):
        return finish("requires_single_material_uv_texture")
    if reference.mode != "RGBA":
        return finish("requires_rgba_cutout")
    alpha = np.asarray(reference.getchannel("A"))
    foreground = alpha > 192
    if not foreground.any() or not (alpha < 16).any():
        return finish("requires_nonempty_transparent_cutout")
    if max(texture.size) > 4096:
        return finish("texture_exceeds_projection_limit")
    vertices = np.asarray(result.vertices, dtype=np.float32)
    faces = np.asarray(result.faces, dtype=np.int32)
    uv = np.asarray(uv, dtype=np.float32)
    if (not len(faces) or not np.isfinite(vertices).all() or not np.isfinite(uv).all()
            or uv.min() < 0 or uv.max() > 1):
        return finish("invalid_mesh_or_uv")
    ymin, xmin = np.argwhere(foreground).min(axis=0)
    ymax, xmax = np.argwhere(foreground).max(axis=0) + 1
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    extent = upper - lower
    if min(extent[:2]) < 1e-7:
        return finish("degenerate_front_bounds")
    camera = vertices[:, :2].copy()
    camera[:, 0] = xmin + (vertices[:, 0] - lower[0]) / extent[0] * (xmax - xmin)
    camera[:, 1] = ymin + (upper[1] - vertices[:, 1]) / extent[1] * (ymax - ymin)
    scale = min(1.0, visibility_size / max(reference.size))
    width, height = max(1, round(reference.width * scale)), max(1, round(reference.height * scale))
    camera_scale = np.array([width / reference.width, height / reference.height], dtype=np.float32)
    depth = np.full((height, width), -np.inf, dtype=np.float32)
    for x, y, face, weights in _raster_samples(camera * camera_scale, faces, width, height):
        z = (vertices[faces[face], 2] * weights).sum(axis=1)
        np.maximum.at(depth, (y, x), z)
    mask = np.asarray(reference.getchannel("A").resize((width, height), Image.Resampling.BILINEAR)) > 192
    silhouette = np.isfinite(depth)
    union = np.count_nonzero(mask | silhouette)
    overlap = float(np.count_nonzero(mask & silhouette) / max(union, 1))
    diagnostics["silhouette_iou"] = round(overlap, 4)
    if overlap < min_silhouette_iou:
        return finish("reference_pose_or_silhouette_mismatch")

    reference_array = _reference_pixels(reference)
    atlas = np.asarray(texture.convert("RGBA")).copy()
    original = atlas.copy()
    tex_height, tex_width = atlas.shape[:2]
    uv_pixels = uv * [tex_width, tex_height]
    uv_pixels[:, 1] = tex_height - uv_pixels[:, 1]
    normals = np.asarray(result.vertex_normals, dtype=np.float32)
    # Reject rear-facing triangles even if smoothed normals point forwards.
    front_indices = np.flatnonzero(np.asarray(result.face_normals)[:, 2] > 0.05)
    front_faces = faces[front_indices]
    applied_weight = np.zeros((tex_height, tex_width), dtype=np.float32)
    # Nearest-pixel depth can differ on a sloping surface. This small tolerance
    # covers roughly 1.5 camera pixels while preserving occluded layers.
    tolerance = max(float(extent[1]) / height * 1.5, float(extent[2]) * 0.003)
    for x, y, face, weights in _raster_samples(uv_pixels, front_faces, tex_width, tex_height):
        vertex_ids = front_faces[face]
        xyz = (vertices[vertex_ids] * weights[:, :, None]).sum(axis=1)
        screen = (camera[vertex_ids] * weights[:, :, None]).sum(axis=1)
        nz = (normals[vertex_ids, 2] * weights).sum(axis=1)
        sample = _bilinear(reference_array, screen)
        screen_depth = screen * camera_scale
        sx = np.clip(screen_depth[:, 0].astype(np.int32), 0, width - 1)
        sy = np.clip(screen_depth[:, 1].astype(np.int32), 0, height - 1)
        visible = np.isfinite(depth[sy, sx]) & (xyz[:, 2] >= depth[sy, sx] - tolerance)
        # Smoothstep blends towards the baked side view; front detail receives
        # full strength only beyond cos(theta)=0.8.
        weight = np.clip((nz - 0.35) / 0.45, 0, 1)
        weight = weight * weight * (3 - 2 * weight)
        weight *= strength * np.clip((sample[:, 3] - 224) / 31, 0, 1) * visible
        use = weight > applied_weight[y, x]
        x, y, weight, sample = x[use], y[use], weight[use], sample[use]
        # Always blend against the original atlas, never compound at seams.
        rgb = original[y, x, :3] * (1 - weight[:, None]) + sample[:, :3] * weight[:, None]
        atlas[y, x, :3] = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
        applied_weight[y, x] = weight
    occupied = np.zeros((tex_height, tex_width), dtype=bool)
    for x, y, _face, _weights in _raster_samples(uv_pixels, faces, tex_width, tex_height):
        occupied[y, x] = True
    changed_surface = np.any(atlas[:, :, :3] != original[:, :, :3], axis=2)
    diagnostics["padded_texels"] = _pad_projected_atlas(atlas, occupied, changed_surface)
    changed = int(np.count_nonzero(np.any(atlas != original, axis=2)))
    diagnostics.update(changed_texels=changed, projected_texels=int(np.count_nonzero(applied_weight)),
                       texture_size=[tex_width, tex_height], applied=changed > 0)
    if changed:
        result.visual.material.baseColorTexture = Image.fromarray(atlas, "RGBA")
    return finish("projected_front_detail" if changed else "no_visible_detail_change")
