"""Scene playback regression checks and a reproducible CPU-side benchmark.

Run ``.venv/bin/python test_scene_performance.py --benchmark`` for measured
server-side costs using the generated city scene and fake Viser handles.
"""

import copy
import io
import json
import statistics
import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np
import trimesh

from object_directing import ObjectDirectorSession
from object_scene import ObjectSceneLayer
from scene_composition import validate_scene


ROOT = Path(__file__).resolve().parent
CITY = ROOT / 'review/demo/fresh-city.json'


class Handle:
    def __init__(self, data):
        self.glb_data = data
        self._position = None
        self.position_writes = 0
        self.color_writes = 0
        self._color = None
        self.wxyz = None
        self.removed = False

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self.position_writes += 1
        self._position = value

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self.color_writes += 1
        self._color = value

    def remove(self):
        self.removed = True


class Scene:
    def __init__(self):
        self.calls = []

    def add_glb(self, name, data, *, wxyz, receive_shadow=True):
        handle = Handle(data)
        handle.wxyz = wxyz
        handle.receive_shadow = receive_shadow
        self.calls.append((name, handle))
        return handle


def make_session(document, frames=300):
    # object_states needs only the stable fields installed by DirectorSession.
    session = ObjectDirectorSession.__new__(ObjectDirectorSession)
    session.lock = threading.RLock()
    session.scene = document
    session.positions = np.zeros((frames, 34, 3), dtype=np.float32)
    session.frame = 0
    session.fps = 60
    return session


def make_layer():
    scene = Scene()
    layer = ObjectSceneLayer(type('Server', (), {'scene': scene})())
    layer._atmosphere = type('Atmosphere', (), {'update': lambda self, *args: None})()
    return layer, scene


class PlaybackPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.document = json.loads(CITY.read_text())
        self.session = make_session(self.document)
        self.layer, self.scene = make_layer()

    def test_static_city_emits_no_repeated_handle_changes(self):
        self.layer.update(self.document['objects'], self.session.object_states())
        originals = [handle for _, handle in self.scene.calls]
        by_id = {name: handle for name, handle in self.scene.calls}
        self.assertFalse(by_id['/objects/city-0'].receive_shadow)
        self.assertFalse(by_id['/objects/city-3'].receive_shadow)
        self.assertTrue(by_id['/objects/city-11'].receive_shadow)
        for frame in range(1, 120):
            self.session.frame = frame
            self.layer.update(self.document['objects'], self.session.object_states())
        self.assertEqual(len(self.scene.calls), len(self.document['objects']))
        self.assertEqual([handle.position_writes for handle in originals], [1] * len(originals))
        self.assertFalse(any(handle.removed for handle in originals))

    def test_tall_thin_walls_are_broad_shadow_receivers(self):
        for size in ([0.15, 4.0, 4.0], [1.5, 3.1, 0.15]):
            with self.subTest(size=size):
                wall = copy.deepcopy(self.document['objects'][3])
                wall['id'] = 'wall-thin'
                wall['size'] = size
                wall['position'] = [0, 2, 0]
                document = dict(self.document, objects=[wall])
                session = make_session(document)
                layer, scene = make_layer()
                layer.update(document['objects'], session.object_states())
                self.assertFalse(scene.calls[0][1].receive_shadow)

    def test_transform_and_asset_edits_only_rebuild_affected_geometry(self):
        self.layer.update(self.document['objects'], self.session.object_states())
        originals = {name: handle for name, handle in self.scene.calls}
        obj = self.document['objects'][3]
        obj['position'][0] += 2
        self.layer.update(self.document['objects'], self.session.object_states())
        self.assertEqual(len(self.scene.calls), 16)
        self.assertEqual(originals['/objects/' + obj['id']].position, tuple(obj['position']))
        obj['size'][0] += .5
        self.layer.update(self.document['objects'], self.session.object_states())
        self.assertEqual(len(self.scene.calls), 17)
        self.assertTrue(originals['/objects/' + obj['id']].removed)
        self.assertFalse(originals['/objects/' + self.document['objects'][4]['id']].removed)
        before = len(self.scene.calls)
        asset = next(asset for asset in self.document['assets'] if asset['id'] == obj['asset'])
        asset['parts'][0]['color'] = [1, 2, 3]
        self.layer.update(self.document['objects'], self.session.object_states())
        users = sum(candidate['asset'] == obj['asset'] for candidate in self.document['objects'])
        self.assertEqual(len(self.scene.calls) - before, users)

    def test_public_bundle_is_detached_and_interaction_remains_dynamic(self):
        first = self.session.object_states()
        first['assets'][0]['parts'][0]['color'] = [1, 2, 3]
        first['objects'][0]['position'][0] += 10
        second = self.session.object_states()
        self.assertNotEqual(first['assets'], second['assets'])
        self.assertNotEqual(first['objects'], second['objects'])
        obj = self.document['objects'][3]
        obj['interaction'] = {'action': 'open', 'trigger': 'proximity', 'radius': 5.0}
        obj['position'] = [0, 0, 0]
        self.session.frame = 1
        dynamic = self.session.object_states()
        self.assertTrue(dynamic['objects'][3]['active'])
        self.assertEqual(dynamic['objects'][3]['position'][1], obj['size'][1])

    def test_limit_scale_scene_has_correct_live_glbs_and_no_static_churn(self):
        sys.path.insert(0, str(ROOT / 'review/scene-scale'))
        from stress_test import glb_triangles, scene_document
        document = scene_document(distinct_sizes=True)
        session = make_session(document)
        layer, scene = make_layer()
        layer.update(document['objects'], session.object_states())
        self.assertEqual(len(scene.calls), 64)
        self.assertEqual(sum(glb_triangles(handle.glb_data) for _, handle in scene.calls), 143616)
        parsed = trimesh.load(io.BytesIO(scene.calls[0][1].glb_data), file_type='glb', force='mesh')
        self.assertFalse(parsed.is_empty)
        self.assertEqual(len(parsed.faces), 187 * 12)
        for frame in (1, 30, 120, 299):
            session.frame = frame
            layer.update(document['objects'], session.object_states())
        self.assertEqual(len(scene.calls), 64)
        self.assertEqual([handle.position_writes for _, handle in scene.calls], [1] * 64)
        self.assertEqual(len(layer._mesh_cache), 1)
        self.assertEqual(len(layer._glb_cache), 64)

    def test_large_scene_edit_remove_replace_and_cache_eviction(self):
        sys.path.insert(0, str(ROOT / 'review/scene-scale'))
        from stress_test import scene_document
        document = scene_document()
        session = make_session(document)
        layer, scene = make_layer()
        layer.update(document['objects'], session.object_states())
        first = scene.calls[0][1]
        for i in range(70):
            document['objects'][0]['size'][0] = 2.4 + (i + 1) * .001
            layer.update(document['objects'], session.object_states())
        self.assertTrue(first.removed)
        self.assertLessEqual(len(layer._glb_cache), 64)
        for i in range(35):
            document['assets'][0]['parts'][0]['color'][0] = 70 + i
            layer.update(document['objects'], session.object_states())
        self.assertLessEqual(len(layer._mesh_cache), 32)
        self.assertLessEqual(len(layer._glb_cache), 64)
        removed = document['objects'].pop(0)
        layer.update(document['objects'], session.object_states())
        self.assertNotIn(removed['id'], layer.handles)
        replacement = copy.deepcopy(removed)
        replacement['id'] = 'replacement'
        document['objects'].append(replacement)
        layer.update(document['objects'], session.object_states())
        self.assertEqual(len(layer.handles), 64)
        self.assertEqual(sum(not handle.removed for _, handle in scene.calls), 64)

    def test_shape_overlimit_rejected_without_changing_live_scene(self):
        sys.path.insert(0, str(ROOT / 'review/scene-scale'))
        from stress_test import exact_limit_scene
        document = exact_limit_scene()
        session = make_session(document)
        session.scene = dict(copy.deepcopy(document), gate={'enabled': True})
        session.project_revision = 7
        before = copy.deepcopy(session.scene)
        too_many_shapes = copy.deepcopy(document)
        too_many_shapes['objects'][0]['asset'] = too_many_shapes['assets'][1]['id']
        self.assertEqual(len(too_many_shapes['objects']), 64)
        with self.assertRaisesRegex(ValueError, '12000'):
            session.set_scene(too_many_shapes)
        self.assertEqual(session.scene, before)
        self.assertEqual(session.project_revision, 7)
        with self.assertRaisesRegex(ValueError, '64'):
            validate_scene(dict(document, objects=document['objects'] + [copy.deepcopy(document['objects'][0])]))


def benchmark():
    document = json.loads(CITY.read_text())
    # Full 5 seconds of 60 Hz playback, with a stable generated city.
    runs = []
    for _ in range(5):
        session = make_session(document)
        layer, scene = make_layer()
        started = time.perf_counter()
        layer.update(document['objects'], session.object_states())
        build_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        for frame in range(300):
            session.frame = frame
            layer.update(document['objects'], session.object_states())
        playback_ms = (time.perf_counter() - started) * 1000 / 300
        runs.append((build_ms, playback_ms, len(scene.calls),
                     sum(handle.position_writes for _, handle in scene.calls)))
    result = {
        'fixture': 'review/demo/fresh-city.json',
        'fixture_objects': len(document['objects']),
        'fixture_assets': len(document['assets']),
        'sample': 'five runs of five seconds at 60 Hz; fake Viser handles isolate server CPU',
        'baseline_before_changes_ms_per_frame': 0.57,
        'baseline_measurement_note': 'single 120 frame local run, same fixture and fake Viser handles',
        'after_median_ms_per_frame': round(statistics.median(run[1] for run in runs), 4),
        'after_runs_ms_per_frame': [round(run[1], 4) for run in runs],
        'after_median_first_build_ms': round(statistics.median(run[0] for run in runs), 3),
        'after_handle_counts': {'created': runs[-1][2], 'position_writes_during_301_updates': runs[-1][3]},
        'frame_budget_ms_at_60fps': round(1000 / 60, 3),
        'scope': 'Server scene state and GLB construction only; browser GPU draw/render latency is excluded.',
    }
    destination = ROOT / 'review/scene-refinement/performance.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    if '--benchmark' in sys.argv:
        benchmark()
    else:
        unittest.main()
