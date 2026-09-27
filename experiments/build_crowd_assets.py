"""Reproducibly export generated motion sources and a city crossing adaptation.

The crossing keeps the generated city asset geometry.  Roads, stripes, new
placements and walkable layout are explicit authored modifications.  The
source city snapshot is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowd_motion_library import build_library


ROOT = Path(__file__).resolve().parents[1]
CITY = ROOT / 'review/prompt-scenes/backgrounds/city.json'
RIG = ROOT / 'assets/paired/Xbot.glb'


def _hex(color):
    return '#' + ''.join(f'{max(0, min(255, int(c))):02x}' for c in color)


def _box(center, size, color, *, origin, shape='box', yaw_radians=0.):
    if min(size) <= 0:
        raise ValueError('Box dimensions must be positive')
    result = {'center': [round(float(x), 4) for x in center],
            'size': [round(float(x), 4) for x in size],
            'color': _hex(color) if isinstance(color, list) else color,
            'shape': shape, 'origin': origin}
    if yaw_radians:
        result['yawRadians'] = round(float(yaw_radians), 6)
    return result


def _generated_object_boxes(source, source_id, *, x, z, yaw=None):
    original = next(item for item in source['objects'] if item['id'] == source_id)
    asset = next(item for item in source['assets'] if item['id'] == original['asset'])
    size = original['size']
    degrees = float(original.get('yaw', 0.) if yaw is None else yaw)
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    boxes = []
    for index, part in enumerate(asset['parts']):
        if part['shape'] not in ('box', 'cylinder', 'sphere'):
            continue
        repeat = part.get('repeat', {})
        counts = repeat.get('count', [1, 1, 1])
        step = repeat.get('step', [0., 0., 0.])
        for ix in range(counts[0]):
            for iy in range(counts[1]):
                for iz in range(counts[2]):
                    local = [part['position'][j] + (ix, iy, iz)[j] * step[j] for j in range(3)]
                    dx, dz = local[0] * size[0], local[2] * size[2]
                    cx, cz = x + c * dx + s * dz, z - s * dx + c * dz
                    sy = part['size'][1] * size[1]
                    box_size = [part['size'][0] * size[0], sy, part['size'][2] * size[2]]
                    if int(abs(degrees)) % 180 == 90:
                        box_size[0], box_size[2] = box_size[2], box_size[0]
                    center = [cx, size[1] / 2 + local[1] * size[1], cz]
                    boxes.append(_box(center, box_size, part['color'],
                                      origin={'kind': 'generated_city_asset', 'source_object': source_id,
                                              'asset': asset['id'], 'part': index}, shape=part['shape']))
    return boxes


def build_background(output: Path, source_path: Path = CITY) -> dict:
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    boxes = []
    def add(center, size, color, label, *, yaw_radians=0.):
        boxes.append(_box(center, size, color,
                          origin={'kind': 'authored_crossing_adaptation', 'label': label},
                          yaw_radians=yaw_radians))
    # The center remains open to the crowd solver; all surfaces share y=0.
    add([0, -.065, 0], [46, .11, 46], '#343a43', 'asphalt foundation')
    for sx in (-1, 1):
        for sz in (-1, 1):
            add([sx * 10.6, -.039, sz * 10.6], [1.4, .08, 1.4], '#bbb3a1', 'curb waiting pad')
    for side in (-1, 1):
        add([0, -.025, side * 16.5], [46, .05, 13], '#9f9b8f', 'north-south curb sidewalk')
        add([side * 16.5, -.025, 0], [13, .05, 20], '#9f9b8f', 'east-west curb sidewalk')
    # Painted orthogonal and diagonal routes echo a scramble crossing.
    for center_z in (-5.2, 5.2):
        for i in range(-9, 10, 2):
            add([i, .001, center_z], [1.15, .002, 2.2], '#e3dcc4', 'east-west zebra stripe')
    for center_x in (-5.2, 5.2):
        for i in range(-9, 10, 2):
            add([center_x, .001, i], [2.2, .002, 1.15], '#e3dcc4', 'north-south zebra stripe')
    for t in (-8, -6, -4, 4, 6, 8):
        add([t, .001, t], [2.2, .002, .55], '#eee7cf', 'diagonal zebra stripe',
            yaw_radians=math.pi / 4)
        add([t, .001, -t], [2.2, .002, .55], '#eee7cf', 'diagonal zebra stripe',
            yaw_radians=-math.pi / 4)
    # The generated city architecture is rearranged outside the walkable square.
    building_ids = ['city-4', 'city-5', 'city-6', 'city-7', 'city-8', 'city-9',
                    'city-23', 'city-24', 'city-25', 'city-26', 'city-27', 'city-28']
    placements = [(-19, -19), (-15, -19), (-19, -15), (-19, 15), (-15, 19), (-19, 19),
                  (19, -19), (15, -19), (19, -15), (19, 15), (15, 19), (19, 19)]
    for source_id, (x, z) in zip(building_ids, placements):
        boxes.extend(_generated_object_boxes(source, source_id, x=x, z=z,
                                             yaw=90 if x < 0 else -90))
    tower_ids = ['city-42', 'city-43', 'city-44', 'city-45', 'city-46', 'city-47']
    tower_placements = [(-12, -22), (0, -22), (12, -22),
                        (-12, 22), (0, 22), (12, 22)]
    for source_id, (x, z) in zip(tower_ids, tower_placements):
        boxes.extend(_generated_object_boxes(source, source_id, x=x, z=z,
                                             yaw=0 if z < 0 else 180))
    for side, source_ids in ((-1, ['city-13', 'city-14', 'city-15', 'city-16']),
                             (1, ['city-32', 'city-33', 'city-34', 'city-35'])):
        for source_id, z in zip(source_ids, [-9, -3, 3, 9]):
            boxes.extend(_generated_object_boxes(source, source_id, x=side * 18.2, z=z))
    data = {
        'version': 1, 'name': 'Generated city scramble crossing adaptation',
        'upAxis': 'y', 'ground': 'xz', 'boxes': boxes,
        'walkable': {'central_square': [-10, -10, 10, 10], 'corner_pads_to': 16},
        'source': {'path': str(source_path.relative_to(ROOT)),
                   'sha256': hashlib.sha256(source_bytes).hexdigest(),
                   'pipeline': 'StageZero generated city background snapshot',
                   'source_name': source['name'],
                   'source_objects_reused': building_ids + tower_ids + [i for pair in
                        (['city-13', 'city-14', 'city-15', 'city-16'],
                         ['city-32', 'city-33', 'city-34', 'city-35']) for i in pair]},
        'authored_modifications': [
            'Repositioned 18 generated building/cafe/tower objects and 8 generated streetlamps around a 20 m central square.',
            'Added asphalt 1 cm below path roots, four nonoverlapping sidewalk strips top at y=0, corner pads top at y=0.001, and orthogonal/diagonal paint stripes top at y=0.002.',
            'Generated asset parts retain their source colors and relative proportions; curved primitives are exported as bounding boxes for lightweight rendering.',
            'Scene metadata defines walkable layout; no visual object detection is claimed.'],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, separators=(',', ':')))
    return {'box_count': len(boxes), 'generated_boxes': sum(b['origin']['kind'] == 'generated_city_asset' for b in boxes),
            'bytes': output.stat().st_size, 'source_sha256': data['source']['sha256']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'review/crowd-crossing/assets')
    parser.add_argument('--sources', type=Path, help='JSON list of source specifications')
    args = parser.parse_args()
    output = args.output
    background = build_background(output / 'crossing_scene.json')
    if args.sources:
        sources = json.loads(args.sources.read_text())
    else:
        sources = [
            {'id': 'cached-city-walk-a', 'path': str(ROOT / 'review/main-cast-ui/generation/core-002.npz'),
             'actor': 0, 'type': 'walk', 'review': 'cached provisional walking source'},
            {'id': 'cached-city-walk-b', 'path': str(ROOT / 'review/main-cast-ui/generation/core-002.npz'),
             'actor': 1, 'type': 'walk', 'review': 'cached provisional walking source'},
            {'id': 'cached-neutral-idle', 'path': str(ROOT / 'review/prompt-scenes/market-three/sources/scene-16054804b3e34e98a810a56aaf53daef/core-000.npz'),
             'actor': 0, 'type': 'idle', 'review': 'cached provisional generated neutral pose'},
        ]
    result = build_library(sources, output, rig_path=RIG)
    print(json.dumps({'background': background,
                      'motion': {key: value for key, value in result.items() if key != 'clips'},
                      'clips': [{'id': c['id'], 'frames': c['frames'], 'speed': c['speed'],
                                 'loop_seam_rms_m': c['calibration']['loop_seam_rms_m']}
                                for c in result['clips']]}, indent=2))


if __name__ == '__main__':
    main()
