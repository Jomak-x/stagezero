"""Placement controls keep the floor handle and editable start pose aligned."""

import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from character_placement import CharacterPlacement


class FakeHandle:
    def __init__(self, **values):
        self.__dict__.update(values)
        self.callbacks = {}

    def on_click(self, callback):
        self.callbacks['click'] = callback
        return callback

    def on_update(self, callback):
        self.callbacks['update'] = callback
        return callback

    def click(self):
        self.callbacks['click'](SimpleNamespace(client=object()))

    def edit(self, value):
        self.value = value
        self.callbacks['update'](SimpleNamespace(client=object()))


class FakeGui:
    def __init__(self):
        self.handles = []

    def _handle(self, **values):
        handle = FakeHandle(**values)
        self.handles.append(handle)
        return handle

    def add_html(self, content):
        return self._handle(content=content)

    def add_button(self, label, **kwargs):
        return self._handle(label=label, disabled=False, visible=kwargs.get('visible', True))

    def add_number(self, label, initial_value, **_):
        return self._handle(label=label, value=initial_value, disabled=False)

    def add_folder(self, *_args, **_kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass


class FakeScene:
    def add_transform_controls(self, name, **kwargs):
        self.name = name
        self.options = kwargs
        return FakeHandle(position=(0., 0., 0.), visible=False)


class FakeSession:
    def __init__(self):
        self.lock = threading.RLock()
        self.busy = False
        self.playing = False
        self.mode = 'Recorded preview'
        self.active_take = None
        self.pose = (0., 0., 0.)
        self.seek = Mock()
        self.set_start_pose = Mock(side_effect=self._set_pose)
        self.can_undo_start_pose = False
        self.undo_start_pose = Mock(return_value=True)

    def set_mode(self, mode):
        self.mode = mode

    def get_start_pose(self):
        return self.pose

    def _set_pose(self, x, z, heading):
        self.pose = (x, z, heading)


class CharacterPlacementTests(unittest.TestCase):
    def setUp(self):
        self.scene = FakeScene()
        self.session = FakeSession()
        self.camera = SimpleNamespace(reset=Mock())
        self.gui = FakeGui()
        self.placement = CharacterPlacement(SimpleNamespace(scene=self.scene), self.session, self.camera)
        self.placement.build(self.gui)

    def test_move_handle_edits_saved_pose_and_end_hides_handle(self):
        self.placement.toggle.click()
        self.assertTrue(self.placement.handle.visible)
        self.assertEqual(self.session.mode, 'Live ARDY')
        self.session.seek.assert_called_with(0)
        self.camera.reset.assert_called_once()
        self.placement.handle.position = (3.25, 0., -1.5)
        self.placement.handle.callbacks['update'](SimpleNamespace(client=object()))
        self.session.set_start_pose.assert_called_with(3.25, -1.5, 0.)
        self.assertEqual(self.placement.x.value, 3.25)
        self.assertEqual(self.placement.z.value, -1.5)
        self.placement.toggle.click()
        self.assertFalse(self.placement.handle.visible)

    def test_exact_controls_and_take_change_keep_handle_in_sync(self):
        self.placement.x.edit(4.)
        self.placement.z.edit(2.)
        self.placement.facing.edit(90.)
        self.assertEqual(self.session.pose, (4., 2., 90.))
        self.placement.toggle.click()
        self.session.pose = (-1., 5., -45.)
        self.placement.update()
        self.assertEqual(self.placement.handle.position, (-1., 0., 5.))
        self.assertEqual((self.placement.x.value, self.placement.z.value, self.placement.facing.value), (-1., 5., -45.))

    def test_busy_does_not_allow_placement_or_keep_handle_visible(self):
        self.placement.toggle.click()
        self.session.busy = True
        self.placement.update()
        self.assertFalse(self.placement.handle.visible)
        self.assertTrue(self.placement.toggle.disabled)
        self.placement.x.edit(9.)
        self.session.set_start_pose.assert_not_called()
        self.placement.toggle.click()
        self.assertFalse(self.placement.active)

    def test_playback_or_take_switch_closes_drag_handle(self):
        self.placement.toggle.click()
        self.session.playing = True
        self.placement.update()
        self.assertFalse(self.placement.active)
        self.assertFalse(self.placement.handle.visible)
        self.session.playing = False
        self.placement.toggle.click()
        self.session.active_take = 'another take'
        self.placement.update()
        self.assertFalse(self.placement.handle.visible)

    def test_undo_move_button_uses_backend_snapshot(self):
        self.assertFalse(self.placement.undo.visible)
        self.session.can_undo_start_pose = True
        self.placement.update()
        self.assertTrue(self.placement.undo.visible)
        self.placement.undo.click()
        self.session.undo_start_pose.assert_called_once()
        self.session.can_undo_start_pose = False
        self.placement.update()
        self.assertFalse(self.placement.undo.visible)


if __name__ == '__main__':
    unittest.main()
