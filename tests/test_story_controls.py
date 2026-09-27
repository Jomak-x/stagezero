"""Popup lifecycle, scene handoff, and stale movement selection."""

from copy import deepcopy
from threading import RLock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from story_controls import StoryControls


class Handle:
    def __init__(self, **kwargs):
        self.__dict__.update(visible=True, disabled=False, **kwargs)
        self.callbacks = {}
        self.closed = False

    def on_click(self, callback):
        self.callbacks['click'] = callback
        return callback

    def on_update(self, callback):
        self.callbacks['update'] = callback
        return callback

    def click(self, client=None):
        self.callbacks['click'](SimpleNamespace(client=client))

    def edit(self, value, client=None):
        self.value = value
        self.callbacks['update'](SimpleNamespace(client=client or object()))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def close(self):
        self.closed = True


class Gui:
    def __init__(self):
        self.modals = []

    def add_modal(self, title, **kwargs):
        modal = Handle(title=title, **kwargs)
        self.modals.append(modal)
        return modal

    def add_tab_group(self):
        return self

    def add_tab(self, _name):
        return Handle()

    def add_folder(self, _name, **_kwargs):
        return Handle()

    def add_button(self, label, **_kwargs):
        return Handle(label=label)

    def add_text(self, label, initial_value='', **_kwargs):
        return Handle(label=label, value=initial_value)

    def add_dropdown(self, label, options):
        return Handle(label=label, options=options, value=options[0])

    def add_html(self, content):
        return Handle(content=content)


class Workflow:
    def __init__(self, session):
        self.session = session
        self.jobs = {}
        self.closed = False

    def submit(self, prompt, seconds=60):
        identifier = f'job-{len(self.jobs) + 1}'
        self.jobs[identifier] = {'status': 'running', 'loaded': False, 'take_id': None,
                                 'plan': None, 'progress': {'completed_beats': 0,
                                                            'total_beats': 4,
                                                            'completed_chunks': 0,
                                                            'total_chunks': 8},
                                 'prompt': prompt, 'seconds': seconds, 'error': None}
        return identifier

    def snapshot(self, identifier):
        return deepcopy(self.jobs[identifier])

    def cancel(self, identifier):
        self.jobs[identifier]['status'] = 'cancelled'

    def load(self, identifier, automatic=False):
        if self.jobs[identifier]['status'] != 'completed':
            if automatic:
                return False
            raise ValueError('Scene is still running')
        self.jobs[identifier]['loaded'] = True
        self.jobs[identifier]['take_id'] = 'scene-take'
        self.session.takes['scene-take'] = self.session.scene_take
        self.session.active_take = 'scene-take'
        self.session.project_revision += 1
        return True

    def close(self):
        self.closed = True


class Session:
    def __init__(self):
        self.lock = RLock()
        self.takes = {}
        self.scene_take = SimpleNamespace(id='scene-take', name='Saved performance', segments=[
            {'prompt': 'Walk ahead', 'start': 0, 'end': 100},
            {'prompt': 'Wave', 'start': 100, 'end': 200}])
        self.active_take = None
        self.project_revision = 0
        self.action_edit_revision = 0
        self.version = 0
        self.character_motion_enabled = True
        self.busy = False
        self.can_undo_action_edit = False
        self.playing = False
        self.frame = 0
        self.status = ''
        self.calls = []

    def set_mode(self, mode):
        self.calls.append(('mode', mode))

    def select_take(self, identifier):
        self.active_take = identifier
        self.calls.append(('select', identifier))

    def submit_action_edit(self, prompt, index, operation, seconds=None):
        self.calls.append(('edit', prompt, index, operation, seconds))
        self.version += 1
        self.busy = True
        self.status = 'Regenerating 2 actions · 0/3 chunks'
        return True

    def seek(self, frame):
        self.calls.append(('seek', frame))
        self.version += 1
        self.busy = False

    def play(self):
        self.playing = True

    def pause(self):
        self.playing = False

    def undo_action_edit(self):
        self.calls.append(('undo',))
        return True


class Core:
    def __init__(self):
        self.active = True
        self.deactivations = 0

    def snapshot(self):
        return {'active': self.active}

    def deactivate(self):
        self.active = False
        self.deactivations += 1


class StoryControlsTests(unittest.TestCase):
    def setUp(self):
        patcher = patch('story_controls.StoryWorkflow', Workflow)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.gui = Gui()
        self.client_gui = Gui()
        self.client = SimpleNamespace(client_id='viewer-1', gui=self.client_gui)
        self.session = Session()
        self.core = Core()
        self.handoffs = []
        self.controls = StoryControls(self.gui, self.session, core_session=self.core,
                                      on_story_activate=lambda: self.handoffs.append(True))

    def test_client_local_modal_and_close_preserve_running_job(self):
        view = self.controls.open(self.client)
        self.assertEqual(len(self.client_gui.modals), 1)
        self.assertEqual(self.client_gui.modals[0].size, 'xl')
        self.assertEqual(len(self.gui.modals), 0)
        view.generate.click(self.client)
        self.assertEqual(len(self.controls.ids), 1)
        view.close.click(self.client)
        self.assertTrue(view.modal.closed)
        self.assertEqual(len(self.controls.workflow.jobs), 1)
        view.generate.click(self.client)  # A delayed event from a removed modal is inert.
        self.assertEqual(len(self.controls.workflow.jobs), 1)
        reopened = self.controls.open(self.client)
        self.assertEqual(reopened.jobs.options[0], next(iter(self.controls.ids)))
        self.assertIn('Running', reopened.status.content)

    def test_core_active_scene_can_submit_but_load_hands_off(self):
        self.session.character_motion_enabled = False
        view = self.controls.open(self.client)
        self.controls.update()
        self.assertFalse(view.generate.disabled)
        view.generate.click(self.client)
        identifier = next(iter(self.controls.ids.values()))
        self.controls.workflow.jobs[identifier]['status'] = 'completed'
        self.controls.update()
        self.assertFalse(self.controls.workflow.jobs[identifier]['loaded'])
        self.assertEqual(self.core.deactivations, 0)
        view.load.click(self.client)
        self.assertTrue(self.controls.workflow.jobs[identifier]['loaded'])
        self.assertEqual(self.core.deactivations, 1)
        self.assertEqual(self.handoffs, [True])

    def test_stale_movement_cannot_edit_replacement_take(self):
        view = self.controls.open(self.client)
        view.generate.click(self.client)
        identifier = next(iter(self.controls.ids.values()))
        self.controls.workflow.jobs[identifier]['status'] = 'completed'
        view.load.click(self.client)
        self.session.character_motion_enabled = True
        view.select_action.click(self.client)
        self.assertFalse(view.edit.disabled)
        self.session.takes['scene-take'] = SimpleNamespace(id='scene-take', name='Changed performance', segments=[
            {'prompt': 'Jump ahead', 'start': 0, 'end': 100}])
        view.edit.click(self.client)
        self.assertFalse(any(call[0] == 'edit' for call in self.session.calls))
        self.assertIn('changed', view.error)
        self.controls.update()
        self.assertTrue(view.edit.disabled)

    def test_duration_validation_and_selected_movement_edit(self):
        view = self.controls.open(self.client)
        view.seconds.value = '121'
        view.generate.click(self.client)
        self.assertFalse(self.controls.ids)
        self.assertIn('120', view.error)
        view.seconds.value = '60'
        view.generate.click(self.client)
        identifier = next(iter(self.controls.ids.values()))
        self.assertEqual(self.controls.workflow.jobs[identifier]['seconds'], 60)
        self.controls.workflow.jobs[identifier]['status'] = 'completed'
        view.load.click(self.client)
        view.select_action.click(self.client)
        view.action_prompt.value = 'Walk slowly'
        view.action_seconds.value = '3.20'
        view.edit.click(self.client)
        self.assertIn(('edit', 'Walk slowly', 0, 'replace', 3.2), self.session.calls)

    def test_popup_undo_does_not_touch_another_take(self):
        view = self.controls.open(self.client)
        view.generate.click(self.client)
        identifier = next(iter(self.controls.ids.values()))
        self.controls.workflow.jobs[identifier]['status'] = 'completed'
        view.load.click(self.client)
        self.session._undo_action_edit = (SimpleNamespace(id='other-take'), object())
        self.session.can_undo_action_edit = True
        self.controls.update()
        self.assertTrue(view.undo.disabled)
        view.undo.click(self.client)
        self.assertNotIn(('undo',), self.session.calls)

    def test_reopened_project_take_requires_explicit_selection_then_refines(self):
        # Simulate a fresh studio process after the project file was reopened:
        # take data exists, but the transient scene-job history is empty.
        self.session.takes['scene-take'] = self.session.scene_take
        self.session.active_take = 'scene-take'
        view = self.controls.open(self.client)
        self.assertEqual(self.controls.ids, {})
        self.assertEqual(view.actions.options, ('No movements yet',))
        self.assertTrue(view.edit.disabled)
        view.use_current.click(self.client)
        self.assertIn('Project take: Saved performance', view.source.content)
        self.assertEqual(len(view.action_map), 2)
        view.select_action.click(self.client)
        view.action_prompt.value = 'Walk toward the door'
        view.edit.click(self.client)
        self.assertIn(('edit', 'Walk toward the door', 0, 'replace', 4.0),
                      self.session.calls)

    def test_current_take_is_not_implicit_fallback_for_unloaded_job(self):
        self.session.takes['scene-take'] = self.session.scene_take
        self.session.active_take = 'scene-take'
        view = self.controls.open(self.client)
        view.generate.click(self.client)
        self.assertEqual(view.actions.options, ('No movements yet',))
        view.select_action.click(self.client)
        self.assertIn('Load a scene', view.error)
        view.use_current.click(self.client)
        self.assertIn('Project take: Saved performance', view.source.content)

    def test_popup_movement_progress_and_failure_replace_completed_job_status(self):
        view = self.controls.open(self.client)
        view.generate.click(self.client)
        identifier = next(iter(self.controls.ids.values()))
        self.controls.workflow.jobs[identifier]['status'] = 'completed'
        view.load.click(self.client)
        view.select_action.click(self.client)
        view.edit.click(self.client)
        self.assertIn('Regenerating 2 actions', view.status.content)
        self.assertTrue(view.cancel_edit.disabled is False)
        self.session.status = 'Regenerating actions · 2/3 chunks received; holding pose'
        self.controls.update()
        self.assertIn('2/3 chunks', view.status.content)
        self.session.busy = False
        self.session.status = 'Action edit failed · backend unavailable. Original take preserved; retry.'
        self.controls.update()
        self.assertIn('Action edit failed', view.status.content)
        self.assertTrue(view.cancel_edit.disabled)
        view.edit.click(self.client)
        self.session.busy = False
        self.session.action_edit_revision += 1
        self.session.status = 'Regenerated 2 actions · Undo is available'
        self.session.takes['scene-take'] = SimpleNamespace(
            id='scene-take', name='Saved performance',
            segments=[{'prompt': 'Walk ahead', 'start': 0, 'end': 100},
                      {'prompt': 'Wave', 'start': 100, 'end': 200}])
        self.controls.update()
        self.assertIn('Regenerated 2 actions', view.status.content)
        self.controls.update()
        self.assertIn('Regenerated 2 actions', view.status.content)
        self.assertEqual(view.selected_index, 0)
        self.assertTrue(view.edit.disabled)  # The replacement take requires Select movement again.

    def test_cancel_movement_update_is_scoped_to_popup_owner_and_version(self):
        view = self.controls.open(self.client)
        view.generate.click(self.client)
        identifier = next(iter(self.controls.ids.values()))
        self.controls.workflow.jobs[identifier]['status'] = 'completed'
        view.load.click(self.client)
        view.select_action.click(self.client)
        view.edit.click(self.client)
        other = self.controls.open(SimpleNamespace(client_id='viewer-2', gui=Gui()))
        self.assertTrue(other.cancel_edit.disabled)
        other.cancel_edit.click()
        self.assertTrue(self.session.busy)
        self.assertNotIn(('seek', 0), self.session.calls)
        # A newer session operation must not be cancelled by an old popup event.
        self.session.version += 1
        self.controls.update()
        self.assertTrue(view.cancel_edit.disabled)
        view.cancel_edit.click(self.client)
        self.assertTrue(self.session.busy)
        self.assertNotIn(('seek', 0), self.session.calls)
        self.assertIn('No movement update', view.error)

    def test_cancel_own_movement_update_preserves_take(self):
        view = self.controls.open(self.client)
        view.generate.click(self.client)
        identifier = next(iter(self.controls.ids.values()))
        self.controls.workflow.jobs[identifier]['status'] = 'completed'
        view.load.click(self.client)
        view.select_action.click(self.client)
        view.edit.click(self.client)
        view.cancel_edit.click(self.client)
        self.assertFalse(self.session.busy)
        self.assertIn('Movement update cancelled', view.status.content)
        self.assertIs(self.session.takes['scene-take'], self.session.scene_take)


if __name__ == '__main__':
    unittest.main()
