"""Reproducible server-side scene stress diagnostic; run from the repository root.

``.venv/bin/python review/scene-scale/stress_test.py`` writes stress-results.json.
The fake scene records the exact GLB payload handed to Viser. It does not
measure browser parsing, GPU memory, frame rate, or shadow rendering.
"""

import copy
import json
import statistics
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from asset_geometry import expanded_count
from scene_composition import validate_scene
from test_scene_performance import make_layer, make_session


def asset(index=0, count=187, shape='box'):
    """Rows of tiny colored box details within one local unit cube."""
    columns = {187: 11, 200: 8, 512: 16}[count]
    rows = count // columns
    assert columns * rows == count
    return {
        'id': f'stress-{index}', 'name': f'Stress facade {index}',
        'parts': [{
            'shape': shape, 'position': [-.4, 0, -.4],
            'size': [.045, .04, .045], 'color': [70 + index * 9, 115, 168],
            'repeat': {'count': [columns, 1, rows],
                       'step': [.8 / max(columns - 1, 1), 0, .8 / max(rows - 1, 1)]},
        }],
    }


def scene_document(*, distinct_assets=False, distinct_sizes=False, count=64, shapes_per_asset=187, shape='box', validate=True):
    assets = [asset(i, shapes_per_asset, shape) for i in range(16 if distinct_assets else 1)]
    objects = []
    for i in range(count):
        size = [2.4 + (i % 8) * .045, 6.0 + (i // 8) * .08, 2.0] if distinct_sizes else [2.4, 6.0, 2.0]
        objects.append({
            'id': f'stress-{i}', 'name': f'Facade {i}', 'kind': 'custom',
            'position': [float((i % 8 - 4) * 3), size[1] / 2, float((i // 8 - 4) * 3)],
            'size': size, 'color': [255, 255, 255],
            'interaction': {'action': 'none', 'trigger': 'none', 'radius': 0},
            'asset': assets[i % len(assets)]['id'],
        })
    document = {'version': 3, 'name': 'Synthetic stress facades',
                'objects': objects, 'assets': assets, 'effects': [], 'lighting': 'neutral'}
    # The sphere case remains a renderer diagnostic after a triangle cap is added.
    return validate_scene(document) if validate else document


def exact_limit_scene():
    document = scene_document()
    one_more = asset(1)
    one_more['parts'].append({'shape': 'box', 'position': [0, .2, 0],
                              'size': [.03, .03, .03], 'color': [175, 115, 168]})
    document['assets'].append(one_more)
    for obj in document['objects'][32:]:
        obj['asset'] = one_more['id']
    return validate_scene(document)


def glb_triangles(data):
    magic, version, length = struct.unpack_from('<4sII', data)
    assert (magic, version, length) == (b'glTF', 2, len(data))
    json_len, chunk_type = struct.unpack_from('<I4s', data, 12)
    assert chunk_type == b'JSON'
    header = json.loads(data[20:20 + json_len])
    indices = header['meshes'][0]['primitives'][0]['indices']
    return header['accessors'][indices]['count'] // 3


def percentile(samples, p):
    ordered = sorted(samples)
    position = (len(ordered) - 1) * p
    lower = int(position)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)


def run_case(document, frames=120):
    session = make_session(document, frames=frames)
    layer, fake_scene = make_layer()
    started = time.perf_counter()
    layer.update(document['objects'], session.object_states())
    build_ms = 1000 * (time.perf_counter() - started)
    payloads = [handle.glb_data for _, handle in fake_scene.calls]
    samples = []
    for frame in range(frames):
        session.frame = frame
        started = time.perf_counter()
        layer.update(document['objects'], session.object_states())
        samples.append(1000 * (time.perf_counter() - started))
    return {
        'objects': len(document['objects']),
        'expanded_shapes': sum(expanded_count(next(a for a in document['assets'] if a['id'] == o['asset']))
                               for o in document['objects']),
        'created_handles': len(fake_scene.calls),
        'glb_triangles_total': sum(glb_triangles(data) for data in payloads),
        'glb_payload_bytes_all_handles': sum(map(len, payloads)),
        'glb_payload_bytes_distinct_content': sum(len(data) for data in set(payloads)),
        'build_ms': round(build_ms, 2),
        'playback_server_cpu_median_ms': round(statistics.median(samples), 4),
        'playback_server_cpu_p95_ms': round(percentile(samples, .95), 4),
        'mesh_cache_entries': len(layer._mesh_cache), 'glb_cache_entries': len(layer._glb_cache),
    }


def run_cycles(document, iterations=280):
    session = make_session(document)
    layer, fake_scene = make_layer()
    layer.update(document['objects'], session.object_states())
    original = fake_scene.calls[0][1]
    samples = []
    for i in range(iterations):
        obj = document['objects'][0]
        if i % 4 == 0:
            obj['position'][0] += .03
        elif i % 4 == 1:
            obj['size'][0] = 2.41 + i * .001
        elif i % 4 == 2:
            obj['color'][0] = 180 + i % 60
        else:
            obj['yaw'] = float(i)
        started = time.perf_counter()
        layer.update(document['objects'], session.object_states())
        samples.append(1000 * (time.perf_counter() - started))
    asset_samples = []
    for i in range(40):
        document['assets'][0]['parts'][0]['color'][0] = 70 + i
        started = time.perf_counter()
        layer.update(document['objects'], session.object_states())
        asset_samples.append(1000 * (time.perf_counter() - started))
    # A removal and a replacement must leave exactly one live handle per object.
    removed = document['objects'].pop(0)
    layer.update(document['objects'], session.object_states())
    assert removed['id'] not in layer.handles
    replacement = copy.deepcopy(removed)
    replacement['id'] = 'replacement'
    document['objects'].append(replacement)
    layer.update(document['objects'], session.object_states())
    assert original.removed
    assert len(layer.handles) == len(document['objects'])
    assert sum(not handle.removed for _, handle in fake_scene.calls) == len(document['objects'])
    assert len(layer._mesh_cache) <= 32 and len(layer._glb_cache) <= 64
    return {'iterations': iterations, 'asset_edit_iterations': len(asset_samples),
            'edit_server_cpu_median_ms': round(statistics.median(samples), 3),
            'edit_server_cpu_p95_ms': round(percentile(samples, .95), 3),
            'shared_asset_edit_server_cpu_median_ms': round(statistics.median(asset_samples), 3),
            'shared_asset_edit_server_cpu_p95_ms': round(percentile(asset_samples, .95), 3),
            'created_handles_total': len(fake_scene.calls), 'live_handles_after_replace': len(layer.handles),
            'mesh_cache_entries': len(layer._mesh_cache), 'glb_cache_entries': len(layer._glb_cache),
            'stale_live_handles': 0}


def main():
    cases = {
        'shared_asset_shared_size': scene_document(),
        'shared_asset_unique_sizes': scene_document(distinct_sizes=True),
        'sixteen_assets_shared_size': scene_document(distinct_assets=True),
        'maximum_asset_512_shapes': scene_document(count=23, shapes_per_asset=512),
        'exact_scene_limit_12000_shapes': exact_limit_scene(),
        'sphere_heavy_near_limit': scene_document(shape='sphere', validate=False),
    }
    result = {'method': 'Synthetic legal scenes with 11,776 to 12,000 expanded shapes, including boxes and spheres; fake Viser handles, 120 static frames per case; warm Python process. Timings are one diagnostic run, not a device FPS claim.',
              'cases': {name: run_case(doc) for name, doc in cases.items()},
              'edit_move_delete_replace': run_cycles(scene_document()),
              'interpretation': 'The 12,000-shape cap does not bound geometry tightly: 11,968 boxes produce 143,616 triangles while the same number of spheres produce 2,010,624 triangles and 187 MB of GLB payload handed to 64 handles. Recommend retaining the shape cap and adding a scene-wide 250,000-triangle limit based on primitive face counts (box 12, sphere 168, cylinder 48, cone 24). Shared asset edits may produce frame-budget spikes. Browser rendering and interactive FPS require a separate real-client measurement.',
              'scope': 'Actual GLB payload bytes and triangles handed to fake Viser; CPU includes object_states and layer.update. Browser network/GPU/render latency excluded.'}
    destination = Path(__file__).with_name('stress-results.json')
    destination.write_text(json.dumps(result, indent=2) + '\n')
    curved = copy.deepcopy(cases['sphere_heavy_near_limit'])
    curved['camera'] = {'position': [25., 22., 26.], 'look_at': [0., 2., 0.]}
    Path(__file__).with_name('limit-curved.json').write_text(json.dumps(curved, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
