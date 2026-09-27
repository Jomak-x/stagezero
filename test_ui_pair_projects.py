"""The main project entry must route native archives without corrupting G1 state."""
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
from studio_ui import StudioUI


class MainPairProjectTests(unittest.TestCase):
    def ui(self, callback):
        ui = StudioUI.__new__(StudioUI)
        ui.session = SimpleNamespace(project_status='', load_project=Mock())
        ui.on_native_open = callback
        ui.update = Mock()
        return ui

    def test_native_main_open_dispatches_without_g1_decoder(self):
        data = io.BytesIO(); np.savez_compressed(data, joints=np.zeros((4,2,22,3)))
        callback = Mock(); ui = self.ui(callback)
        ui.open_data(data.getvalue())
        callback.assert_called_once_with(data.getvalue())
        ui.session.load_project.assert_not_called()
        self.assertIn('paired scene', ui.session.project_status)

    def test_invalid_native_open_retains_main_state(self):
        data = io.BytesIO(); np.savez_compressed(data, joints=np.zeros(1))
        callback = Mock(side_effect=ValueError('invalid native archive')); ui = self.ui(callback)
        ui.open_data(data.getvalue())
        self.assertIn('current takes preserved', ui.session.project_status)
        ui.session.load_project.assert_not_called()

    def test_main_saved_picker_includes_native_folder(self):
        with tempfile.TemporaryDirectory() as root:
            ui = StudioUI.__new__(StudioUI); ui.folder = Path(root)/'projects'; ui.folder.mkdir()
            native = Path(root)/'native-pair-projects'; native.mkdir()
            (ui.folder/'solo.stagezero.npz').write_bytes(b'a')
            (native/'duet.native-pair.stagezero.npz').write_bytes(b'b')
            (native/'linked.stagezero.npz').symlink_to(ui.folder/'solo.stagezero.npz')
            ui.saved = SimpleNamespace(options=(),value='');ui.open=SimpleNamespace(disabled=True)
            ui.refresh_saved()
            self.assertEqual(set(ui.saved_map), {'solo.stagezero.npz','duet.native-pair.stagezero.npz'})
            self.assertFalse(ui.open.disabled)

if __name__ == '__main__': unittest.main()
