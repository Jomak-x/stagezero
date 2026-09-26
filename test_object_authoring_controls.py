"""Scene and object authoring controls stay distinct and bound their uploads."""

import unittest
from unittest.mock import patch

import numpy as np

from object_controls import add_object_controls
from object_directing import ObjectDirectorSession
from scene_composition import MAX_SCENE_FILE_BYTES
from test_live_motion import ControlledBackend
from tests.test_upload_consumers import _SceneGui


class ObjectAuthoringControlTests(unittest.TestCase):
    def test_offline_defaults_and_scene_upload_transfer_limit(self):
        backend = ControlledBackend()
        backend.release.set()
        session = ObjectDirectorSession(
            backend, np.zeros((20, 34, 3), dtype=np.float32),
            np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1)))
        gui = _SceneGui()
        gui._owner = object()
        with patch('object_controls.GatewayGenerator.from_env', side_effect=ValueError('No gateway')), \
             patch('object_controls.install_upload_snapshots'), \
             patch('object_controls.ScopedUploadLimits') as limits:
            add_object_controls(gui, session)
        by_label = {handle.label: handle for handle in gui.handles if hasattr(handle, 'label')}
        self.assertEqual(by_label['Describe an object'].value, 'A lamp')
        self.assertEqual(by_label['Object source'].value, 'Recipes · catalog props')
        self.assertIn('Generate background + scene · replace', by_label)
        self.assertIn('Generate object · add to scene', by_label)
        self.assertIn('Advanced · reusable props', by_label)
        limits.assert_called_once_with(gui._owner)
        self.assertIs(limits.return_value.register.call_args.args[0], by_label['Import scene JSON'])
        self.assertEqual(limits.return_value.register.call_args.kwargs['max_bytes'], MAX_SCENE_FILE_BYTES)


if __name__ == '__main__':
    unittest.main()
