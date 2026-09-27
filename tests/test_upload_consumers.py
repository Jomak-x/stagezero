"""Project and scene callbacks consume the file that completed their transfer."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from object_controls import add_object_controls
from object_directing import ObjectDirectorSession
from takes import encode_project
import test_studio_ui as fixtures
from upload_events import UploadEvent, UploadFile


class _SceneGui(fixtures.Gui):
    def add_number(self, label, initial_value=0, **kwargs):
        return self._handle(label=label, value=initial_value)

    add_slider = add_number

    def add_vector3(self, label, initial_value=(0, 0, 0), **kwargs):
        return self._handle(label=label, value=initial_value)


class UploadConsumerTests(unittest.TestCase):
    setUp = fixtures.StudioUITests.setUp
    tearDown = fixtures.StudioUITests.tearDown
    _seed_take = fixtures.StudioUITests._seed_take

    def test_project_callback_loads_snapshot_when_live_handle_has_newer_file(self):
        original = self._seed_take()
        original.segments = [dict(start=0, end=len(original.positions), prompt='First')]
        payload = encode_project(self.session.takes, self.session.active_take,
                                 self.session.frame, self.session.scene)
        self.session.rename_active_take('Changed after saved file')
        self.ui.upload.value = UploadFile('later.npz', b'not the completed project')
        event = UploadEvent(object(), 1, self.ui.upload, 'project-a',
                            UploadFile('first.npz', payload))
        self.ui.upload.callbacks['upload'](event)
        self.assertEqual(self.session.takes[original.id].name, 'First')
        self.assertNotIn('Open failed', self.session.project_status)

    def test_scene_callback_loads_snapshot_when_live_handle_has_newer_file(self):
        session = ObjectDirectorSession(self.backend, *self.session.recorded)
        gui = _SceneGui()
        with patch('object_controls.install_upload_snapshots'):
            add_object_controls(gui, session)
        handle = next(h for h in gui.handles if getattr(h, 'label', '') == 'Import scene JSON')
        scene = {'version': 2, 'name': 'First scene', 'objects': [],
                 'effects': [], 'lighting': 'neutral'}
        handle.value = UploadFile('later.json', b'invalid later contents')
        event = UploadEvent(object(), 1, handle, 'scene-a',
                            UploadFile('first.json', json.dumps(scene).encode()))
        handle.callbacks['upload'](event)
        self.assertEqual(session.scene_document(), scene)


if __name__ == '__main__':
    unittest.main()
