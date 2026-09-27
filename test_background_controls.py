"""Background gallery and async generation workflow, without a provider/server."""
import copy
import inspect
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import viser

from object_controls import add_object_controls
from scene_composition import PRESETS, make_preset
from scene_preset_previews import preset_preview


class Handle:
    def __init__(self, gui, **values):
        self.gui, self.callbacks = gui, {}
        self.visible, self.disabled, self.closed = True, False, False
        self.__dict__.update(values)

    def __enter__(self):
        self.gui.parents.append(self)
        return self

    def __exit__(self, *_):
        self.gui.parents.pop()

    def close(self):
        self.closed = True

    def on_click(self, callback):
        self.callbacks['click'] = callback
        return callback

    def on_update(self, callback):
        self.callbacks['update'] = callback
        return callback

    def on_upload(self, callback):
        self.callbacks['upload'] = callback
        return callback


class Gui:
    def __init__(self):
        self.handles, self.parents = [], []
        self.client = SimpleNamespace(gui=self, camera=SimpleNamespace())

    def __getattr__(self, name):
        if not name.startswith('add_'):
            raise AttributeError(name)
        api = getattr(viser.GuiApi, name)

        def add(*args, **kwargs):
            # Check our UI calls against the actual installed Viser signatures.
            values = inspect.signature(api).bind(None, *args, **kwargs).arguments
            values.pop('self')
            values['kind'] = name
            values['parent'] = self.parents[-1] if self.parents else None
            if 'initial_value' in values:
                values['value'] = values['initial_value']
            elif 'options' in values:
                values['value'] = values['options'][0]
            handle = Handle(self, **values)
            self.handles.append(handle)
            return handle
        return add

    def find(self, label):
        return next(h for h in reversed(self.handles) if getattr(h, 'label', None) == label)

    def click(self, label):
        handle = self.find(label)
        handle.callbacks['click'](SimpleNamespace(client=self.client))
        return handle


class Session:
    def __init__(self):
        self.document = make_preset('Jungle temple')
        self.started, self.finish = threading.Event(), threading.Event()
        self.error = None
        self.calls = []
        self.lock = threading.RLock()
        self.project_revision = 0

    def scene_document(self):
        return copy.deepcopy(self.document)

    def set_scene(self, document, *, reset_gate=True):
        self.document = copy.deepcopy(document)
        self.project_revision += 1

    def generate_scene(self, prompt, generator, seed, *, expected_revision=None):
        self.calls.append((prompt, generator, seed, expected_revision))
        self.started.set()
        if not self.finish.wait(3):
            raise TimeoutError('Test did not release generation')
        if self.error:
            raise self.error
        if expected_revision != self.project_revision:
            raise ValueError('Project changed during generation')
        self.set_scene(make_preset('City boulevard'))
        return self.scene_document()


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('Timed out')
        time.sleep(.005)


class BackgroundControlsTests(unittest.TestCase):
    def setUp(self):
        self.gui, self.session = Gui(), Session()
        # The real upload adapter requires Viser's websocket machinery; this
        # fake exercises scene controls and gallery callbacks directly.
        with patch('object_controls.install_upload_snapshots'):
            add_object_controls(self.gui, self.session)

    def test_gallery_categories_and_one_click_apply(self):
        self.gui.click('Browse backgrounds')
        images = [h for h in self.gui.handles if h.kind == 'add_image']
        self.assertEqual(len(images), len(PRESETS))
        self.assertEqual(sum(h.visible for h in images), 3)
        categories = self.gui.find('Collection')
        categories.value = 'Everyday'
        categories.callbacks['click'](None)
        self.assertEqual(sum(h.visible for h in images), 5)
        self.gui.click('Use City boulevard')
        self.assertEqual(self.session.scene_document(), make_preset('City boulevard'))
        self.assertTrue(any(h.closed for h in self.gui.handles if h.kind == 'add_modal'))
        self.assertTrue(hasattr(self.gui.client.camera, 'look_at'))

    def test_generation_is_automatic_async_and_reports_completion(self):
        original = self.session.scene_document()
        self.gui.find('Describe a scene').value = 'A rainy futuristic city'
        self.gui.find('Generation source').value = 'Recipes · offline'
        self.gui.click('Generate background + scene · replace')
        self.assertTrue(self.session.started.wait(2))
        self.assertEqual(original, self.session.scene_document())
        self.assertTrue(self.gui.find('Generate background + scene · replace').disabled)
        self.assertIsNone(self.session.calls[0][1])
        self.assertEqual(self.session.calls[0][3], 0)
        self.session.finish.set()
        wait_for(lambda: not self.gui.find('Generate background + scene · replace').disabled)
        self.assertNotEqual(original, self.session.scene_document())
        self.assertTrue(any('Scene ready' in getattr(h, 'content', '') for h in self.gui.handles))

    def test_failure_preserves_background_and_allows_retry(self):
        original = self.session.scene_document()
        self.session.error = ValueError('Generation unavailable')
        self.gui.find('Describe a scene').value = 'A rainy city'
        self.gui.find('Generation source').value = 'Recipes · offline'
        self.gui.click('Generate background + scene · replace')
        self.assertTrue(self.session.started.wait(2))
        self.session.finish.set()
        wait_for(lambda: not self.gui.find('Generate background + scene · replace').disabled)
        self.assertEqual(original, self.session.scene_document())
        self.assertTrue(any('Scene unchanged' in getattr(h, 'content', '') for h in self.gui.handles))

    def test_gallery_busy_generation_cannot_replace_scene(self):
        self.gui.find('Generation source').value = 'Recipes · offline'
        self.gui.click('Generate background + scene · replace')
        self.assertTrue(self.session.started.wait(2))
        self.gui.click('Browse backgrounds')
        self.gui.click('Use Winter plaza')
        self.assertEqual(self.session.scene_document(), make_preset('Jungle temple'))
        self.session.finish.set()
        wait_for(lambda: not self.gui.find('Generate background + scene · replace').disabled)

    def test_cinematic_effect_uses_position_and_scale_controls(self):
        self.gui.find('Effect').value = 'energy_burst'
        self.gui.find('Effect position · m').value = (1., 2., -3.)
        self.gui.find('Effect scale').value = 1.5
        self.gui.click('Add effect')
        effect = self.session.scene_document()['effects'][-1]
        self.assertEqual(effect['kind'], 'energy_burst')
        self.assertEqual(effect['position'], [1., 2., -3.])
        np.testing.assert_allclose(effect['size'], [5.7, 5.7, 5.7])

    def test_every_preset_has_a_nonempty_real_thumbnail(self):
        for name in PRESETS:
            image = preset_preview(name)
            self.assertEqual(image.shape, (300, 560, 3))
            self.assertGreater(np.std(image.astype(float)), 15)


if __name__ == '__main__':
    unittest.main()
