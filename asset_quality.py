"""Conservative, deterministic checks and repairs for procedural asset recipes.

This pass handles geometric mistakes that are common in generated props while
leaving deliberate overlap (columns, trim, foliage) alone. It operates on the
bounded recipe format, before mesh compilation, so no mesh library is needed.
"""

from __future__ import annotations

import math

from asset_geometry import validate_assets


_EPS = 1e-5
_SURFACE_GAP = .002
_MAX_SUPPORT_GAP = .04


def _bounds(part):
    center, size = part['position'], part['size']
    return ([center[i] - size[i] / 2 for i in range(3)],
            [center[i] + size[i] / 2 for i in range(3)])


def _unrotated_box(part):
    return part['shape'] == 'box' and all(abs(v) <= _EPS for v in part.get('rotation', [0, 0, 0]))


def _overlap(a, b, axis):
    al, ah = _bounds(a)
    bl, bh = _bounds(b)
    other = [i for i in range(3) if i != axis]
    return [max(0., min(ah[i], bh[i]) - max(al[i], bl[i])) for i in other]


def _axis(part):
    """Normal axis of a thin unrotated rectangular detail, if recognizable."""
    if not _unrotated_box(part):
        return None
    axis = min(range(3), key=lambda i: part['size'][i])
    if part['size'][axis] > .03 or max(part['size']) < .07:
        return None
    return axis


def _repeat_overlap(part):
    axis = _axis(part)
    repeat = part.get('repeat')
    if axis is None or repeat is None:
        return []
    return [i for i in range(3) if i != axis and repeat['count'][i] > 1
            and abs(repeat['step'][i]) > _EPS
            and part['size'][i] > abs(repeat['step'][i]) * 1.05]


def _occluder(parts, index):
    pane = parts[index]
    axis = _axis(pane)
    if axis is None or 'repeat' not in pane:
        return None
    lightness = sum(pane['color'])
    plo, phi = _bounds(pane)
    side = 1 if pane['position'][axis] >= parts[0]['position'][axis] else -1
    pane_face = phi[axis] if side > 0 else plo[axis]
    for j, slab in enumerate(parts):
        if j == index or not _unrotated_box(slab) or slab.get('repeat') or _axis(slab) != axis:
            continue
        if sum(slab['color']) >= lightness * .8:
            continue
        overlap = _overlap(pane, slab, axis)
        if any(overlap[k] < pane['size'][d] * .6 for k, d in enumerate(i for i in range(3) if i != axis)):
            continue
        slo, shi = _bounds(slab)
        slab_face = shi[axis] if side > 0 else slo[axis]
        if side * (slab_face - pane_face) >= -_EPS:
            return j, axis, side, slab_face
    return None


def _coplanar(parts, index):
    detail = parts[index]
    axis = _axis(detail)
    if axis is None:
        return None
    dlo, dhi = _bounds(detail)
    for j, other in enumerate(parts):
        if j == index or not _unrotated_box(other) or other.get('repeat'):
            continue
        if math.prod(other['size']) <= math.prod(detail['size']):
            continue
        olo, ohi = _bounds(other)
        overlap = _overlap(detail, other, axis)
        if any(overlap[k] < detail['size'][d] * .5 for k, d in enumerate(i for i in range(3) if i != axis)):
            continue
        for side, own, surface in ((1, dhi[axis], ohi[axis]), (-1, dlo[axis], olo[axis])):
            if abs(own - surface) <= .0005:
                return j, axis, side
    return None


def _support_gap(parts, index):
    """Find a small vertical gap to a supporting part with horizontal overlap."""
    # The first primitive defines the main mass in generated assets. A small
    # floating detail should grow toward it; never inflate the main mass to
    # reach a detail (which could swallow road paint or façade trim).
    if index == 0:
        return None
    part = parts[index]
    if not _unrotated_box(part) or part.get('repeat') and part['repeat']['count'][1] > 1:
        return None
    lo, hi = _bounds(part)
    best = None
    for j, other in enumerate(parts):
        if j == index or not _unrotated_box(other):
            continue
        olo, ohi = _bounds(other)
        overlap_x = min(hi[0], ohi[0]) - max(lo[0], olo[0])
        overlap_z = min(hi[2], ohi[2]) - max(lo[2], olo[2])
        if overlap_x <= min(part['size'][0], other['size'][0]) * .15 or overlap_z <= min(part['size'][2], other['size'][2]) * .15:
            continue
        # An existing intersection means this part is attached already.
        if min(hi[1], ohi[1]) - max(lo[1], olo[1]) >= -_EPS:
            return None
        if lo[1] > ohi[1]:
            gap, side = lo[1] - ohi[1], -1  # extend our lower face
        else:
            gap, side = olo[1] - hi[1], 1   # extend our upper face
        if best is None or gap < best[0]:
            best = (gap, j, side)
    return best


def assess_assets(assets):
    """Return stable issue dictionaries for validated asset recipes.

    ``severity == 'error'`` identifies visible defects that should trigger a
    generation retry if they remain after ``refine_assets``. Warnings identify
    uncertain geometry that should be reviewed but may be intentional.
    """
    clean = validate_assets(assets)
    issues = []
    for asset in clean:
        parts = asset['parts']
        for i, part in enumerate(parts):
            def add(code, severity, message):
                issues.append({'asset_id': asset['id'], 'part_index': i,
                               'code': code, 'severity': severity, 'message': message})
            for axis in _repeat_overlap(part):
                add('repeat_overlap', 'error', f'repeated panels overlap on axis {axis}')
            hit = _occluder(parts, i)
            if hit:
                add('facade_occlusion', 'error', f'darker part {hit[0]} covers this repeated panel')
            hit = _coplanar(parts, i)
            if hit:
                add('coplanar_surface', 'warning', f'visible face coincides with part {hit[0]}')
            support = _support_gap(parts, i)
            if support and support[0] <= _MAX_SUPPORT_GAP:
                add('detached_detail', 'warning', f'{support[0]:.4f} local-unit gap to part {support[1]}')
    return issues


def refine_assets(assets):
    """Return a detached, bounded repair of obvious generated geometry defects.

    Pane grids are resized to their cell spacing; occluded panes are placed
    ahead of their backing; thin coplanar details are nudged outward; and
    nearby unsupported boxes are extended to meet their support. Repairs that
    would exceed the unit bounds are skipped. Calling this twice is stable.
    """
    clean = validate_assets(assets)
    for asset in clean:
        parts = asset['parts']
        for part in parts:
            for axis in _repeat_overlap(part):
                part['size'][axis] = max(.001, abs(part['repeat']['step'][axis]) * .82)
        for i, part in enumerate(parts):
            hit = _occluder(parts, i)
            if hit:
                _, axis, side, surface = hit
                new_position = surface + side * (part['size'][axis] / 2 + _SURFACE_GAP)
                old_position = part['position'][axis]
                part['position'][axis] = new_position
                try:
                    validate_assets([asset])
                except ValueError:
                    part['position'][axis] = old_position
        for i, part in enumerate(parts):
            if _occluder(parts, i):
                continue
            # Several small overlays may share the same face. Resolve the
            # resulting stack in one pass so a second call changes nothing.
            for _ in range(4):
                hit = _coplanar(parts, i)
                if not hit:
                    break
                _, axis, side = hit
                part['position'][axis] += side * _SURFACE_GAP
                try:
                    validate_assets([asset])
                except ValueError:
                    part['position'][axis] -= side * _SURFACE_GAP
                    break
        for i, part in enumerate(parts):
            hit = _support_gap(parts, i)
            if hit and _EPS < hit[0] <= _MAX_SUPPORT_GAP:
                gap, _, side = hit
                grow = gap + .001
                part['size'][1] += grow
                part['position'][1] += side * grow / 2
                try:
                    validate_assets([asset])
                except ValueError:
                    part['size'][1] -= grow
                    part['position'][1] -= side * grow / 2
    return clean
