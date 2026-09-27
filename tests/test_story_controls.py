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
        self.controls = []

    def add_modal(self, title, **kwargs):
        modal = Handle(title=title, **kwargs)
        self.modals.append(modal)
        return modal

    def add_tab_group(self):
        return self

    def add_tab(self, _name):
        return Handle()

    def add_folder(self, _name, **_kwargs):
        folder = Handle(label=_name)
        self.controls.append(folder)
        return folder

    def add_button(self, label, **_kwargs):
        handle = Handle(label=label)
        self.controls.append(handle)
        return handle

    def add_text(self, label, initial_value='', **_kwargs):
        handle = Handle(label=label, value=initial_value, **_kwargs)
        self.controls.append(handle)
        return handle

    def add_dropdown(self, label, options, initial_value=None, **_kwargs):
        handle = Handle(label=label, options=options,
                        value=initial_value if initial_value is not None else options[0])
        self.controls.append(handle)
        return handle

    def add_html(self, content):
        handle = Handle(content=content)
        self.controls.append(handle)
        return handle


class Workflow:
    def __init__(self, session):
        self.session = session
        self.jobs = {}
        self.closed = False

    def submit(self, prompt, seconds=None):
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

    def submit_action_edit(self, prompt, index, operation, seconds=None,
                           automatic_timing=False):
        self.calls.append(('edit', prompt, index, operation, seconds, automatic_timing))
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

    def create_scene(self, view):
        view.prompt.value = 'Walk forward, then wave.'
        view.generate.click(self.client)
        return next(iter(self.controls.ids.values()))

    def test_create_starts_with_only_essential_controls(self):
        view = self.controls.open(self.client)
        self.assertEqual(view.modal.title, 'Full scene')
        self.assertEqual(view.prompt.value, '')
        self.assertIn('Walk forward', view.prompt.hint)
        self.assertEqual(view.length.value, 'Auto')
        self.assertEqual(view.length.options[0], 'Auto')
        self.assertFalse(view.seconds.visible)
        self.assertFalse(view.jobs.visible)
        self.assertFalse(view.cancel.visible)
        self.assertFalse(view.load.visible)
        self.assertFalse(view.details.visible)
        self.assertFalse(view.actions.visible)
        self.assertFalse(view.timing.visible)
        self.assertFalse(view.undo.visible)
        self.assertFalse(view.cancel_edit.visible)
        self.assertEqual(view.status.content, '')
        self.assertEqual(view.estimate.content, '')
        view.generate.click(self.client)
        self.assertFalse(self.controls.ids)
        self.assertIn('Describe what happens', view.status.content)

    def test_movement_change_populates_fields_and_poll_preserves_draft(self):
        self.session.takes['scene-take'] = self.session.scene_take
        self.session.active_take = 'scene-take'
        view = self.controls.open(self.client)
        self.assertEqual(view.action_prompt.value, 'Walk ahead')
        self.assertEqual(view.action_seconds.value, '4.00')
        view.actions.edit(view.actions.options[1], self.client)
        self.assertEqual(view.action_prompt.value, 'Wave')
        view.action_prompt.value = 'Wave with the left hand'
        self.controls.update()
        self.assertEqual(view.action_prompt.value, 'Wave with the left hand')
        view.edit.click(self.client)
        self.assertIn(('edit', 'Wave with the left hand', 1, 'replace', None, True),
                      self.session.calls)

    def test_client_local_modal_and_close_preserve_running_job(self):
        view = self.controls.open(self.client)
        self.assertEqual(len(self.client_gui.modals), 1)
        self.assertEqual(self.client_gui.modals[0].size, 'md')
        self.assertEqual(len(self.gui.modals), 0)
        identifier = self.create_scene(view)
        self.assertIsNone(self.controls.workflow.jobs[identifier]['seconds'])
        self.assertEqual(len(self.controls.ids), 1)
        view.close.click(self.client)
        self.assertTrue(view.modal.closed)
        self.assertEqual(len(self.controls.workflow.jobs), 1)
        view.generate.click(self.client)  # A delayed event from a removed modal is inert.
        self.assertEqual(len(self.controls.workflow.jobs), 1)
        reopened = self.controls.open(self.client)
        self.assertEqual(reopened.jobs.options[0], next(iter(self.controls.ids)))
        self.assertEqual(reopened.length.value, 'Auto')
        self.assertIn('Creating scene', reopened.status.content)

    def test_open_modal_receives_other_clients_scenes_and_current_take(self):
        self.session.takes['scene-take'] = self.session.scene_take
        self.session.active_take = 'scene-take'
        first = self.controls.open(self.client)
        self.assertEqual(first.jobs.value, 'Current scene')
        self.assertIn('Saved performance', first.source.content)
        first.prompt.value = 'Keep this draft'
        second_client = SimpleNamespace(client_id='viewer-2', gui=Gui())
        second = self.controls.open(second_client)
        identifier = self.create_scene(second)
        self.controls.update()
        label = next(label for label, value in self.controls.ids.items()
                     if value == identifier)
        self.assertIn(label, first.jobs.options)
        self.assertEqual(first.jobs.value, 'Current scene')
        self.assertEqual(first.prompt.value, 'Keep this draft')
        first.jobs.edit(label, self.client)
        self.assertEqual(first.actions.options, ('No movements yet',))
        self.assertTrue(first.edit.disabled)
        first.jobs.edit('Current scene', self.client)
        self.assertEqual(len(first.action_map), 2)
        self.assertIn('Saved performance', first.source.content)

    def test_current_scene_switch_requires_fresh_movement_binding(self):
        self.session.takes['scene-take'] = self.session.scene_take
        self.session.active_take = 'scene-take'
        view = self.controls.open(self.client)
        self.assertFalse(view.edit.disabled)
        new_take = SimpleNamespace(id='new-scene', name='New scene', segments=[
            {'prompt': 'Jump', 'start': 0, 'end': 75}])
        self.session.takes['new-scene'] = new_take
        self.session.active_take = 'new-scene'
        view.edit.click(self.client)
        self.assertFalse(any(call[0] == 'edit' for call in self.session.calls))
        self.assertTrue(view.edit.disabled)
        self.assertTrue(view.refresh_action.visible)
        self.assertIn('Current scene changed', view.status.content)
        view.refresh_action.click(self.client)
        self.assertEqual(view.action_prompt.value, 'Jump')
        self.assertFalse(view.edit.disabled)

    def test_ready_actions_review_or_navigate_directly_to_refine(self):
        view = self.controls.open(self.client)
        identifier = self.create_scene(view)
        self.controls.workflow.jobs[identifier]['status'] = 'completed'
        self.controls.update()
        self.assertTrue(view.review.visible)
        self.assertTrue(view.load.visible)
        with patch('story_controls.navigate_tab') as navigate:
            view.load.click(self.client)
        navigate.assert_called_once_with(view.tabs, 1, self.client)
        self.assertTrue(self.controls.workflow.jobs[identifier]['loaded'])
        self.assertEqual(view.actions.options[0].split(' · ')[0], '01')
        self.assertFalse(view.modal.closed)
        view.review.click(self.client)
        self.assertTrue(view.modal.closed)
        self.assertIn(('select', 'scene-take'), self.session.calls)

    def test_timeline_opens_owned_exact_scene_movement_without_seek(self):
        view = self.controls.open(self.client)
        identifier = self.create_scene(view)
        self.controls.workflow.jobs[identifier]['status'] = 'completed'
        view.load.click(self.client)
        self.assertTrue(self.controls.owns_take('scene-take'))
        with patch('story_controls.navigate_tab') as navigate:
            self.assertTrue(self.controls.open_for_movement(self.client, 'scene-take', 1))
        opened = self.controls._views[self.client.client_id]
        self.assertTrue(view.modal.closed)
        self.assertEqual(opened.selected_index, 1)
        self.assertEqual(opened.action_prompt.value, 'Wave')
        self.assertFalse(any(call[0] == 'seek' for call in self.session.calls))
        navigate.assert_called_once_with(opened.tabs, 1, self.client)
        self.assertFalse(self.controls.open_for_movement(self.client, 'scene-take', 9))
        self.session.active_take = None
        self.assertFalse(self.controls.open_for_movement(self.client, 'scene-take', 0))

    def test_timeline_does_not_claim_unmarked_ordinary_take(self):
        self.session.takes['ordinary'] = SimpleNamespace(
            id='ordinary', name='Ordinary',
            segments=[{'prompt': 'Walk', 'start': 0, 'end': 50}])
        self.session.active_take = 'ordinary'
        self.assertFalse(self.controls.owns_take('ordinary'))
        self.assertFalse(self.controls.open_for_movement(self.client, 'ordinary', 0))
        self.assertEqual(self.client_gui.modals, [])
        self.session.takes['ordinary'].segments[0]['beat_id'] = 'beat-1'
        with patch('story_controls.navigate_tab'):
            self.assertTrue(self.controls.open_for_movement(self.client, 'ordinary', 0))

    def test_auto_plan_estimate_and_refine_durations(self):
        view = self.controls.open(self.client)
        identifier = self.create_scene(view)
        self.controls.workflow.jobs[identifier]['plan'] = {
            'beats': [
                {'prompt': 'Walk toward the door', 'seconds': 3.4},
                {'prompt': 'Pause and wave', 'seconds': 2.16},
                {'prompt': 'Turn and walk away', 'seconds': 4.04},
            ],
            'warnings': [],
        }
        self.controls.update()
        self.assertIn('Estimated 9.6s · 3 movements', view.estimate.content)
        self.assertIn('Pause and wave · 2.16s', view.plan.content)
        self.session.scene_take = SimpleNamespace(
            id='scene-take', name='Saved performance', segments=[
                {'prompt': 'Walk toward the door', 'start': 0, 'end': 85},
                {'prompt': 'Pause and wave', 'start': 85, 'end': 139},
                {'prompt': 'Turn and walk away', 'start': 139, 'end': 240},
            ])
        self.controls.workflow.jobs[identifier]['status'] = 'completed'
        view.load.click(self.client)
        self.assertIn('Current scene: 9.6s · 3 movements', view.estimate.content)
        self.assertIn('3.40s long · Walk toward the door', view.actions.options[0])
        self.assertIn('2.16s long · Pause and wave', view.actions.options[1])
        self.assertIn('4.04s long · Turn and walk away', view.actions.options[2])

    def test_core_active_scene_can_submit_but_load_hands_off(self):
        self.session.character_motion_enabled = False
        view = self.controls.open(self.client)
        self.controls.update()
        self.assertFalse(view.generate.disabled)
        identifier = self.create_scene(view)
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
        identifier = self.create_scene(view)
        self.controls.workflow.jobs[identifier]['status'] = 'completed'
        view.load.click(self.client)
        self.session.character_motion_enabled = True
        self.assertFalse(view.edit.disabled)
        self.session.takes['scene-take'] = SimpleNamespace(id='scene-take', name='Changed performance', segments=[
            {'prompt': 'Jump ahead', 'start': 0, 'end': 100}])
        view.edit.click(self.client)
        self.assertFalse(any(call[0] == 'edit' for call in self.session.calls))
        self.assertIn('changed', view.error)
        self.controls.update()
        self.assertTrue(view.edit.disabled)
        self.assertTrue(view.refresh_action.visible)
        view.refresh_action.click(self.client)
        self.assertFalse(view.edit.disabled)

    def test_duration_validation_and_selected_movement_edit(self):
        view = self.controls.open(self.client)
        view.length.edit('15 seconds', self.client)
        self.assertFalse(view.seconds.visible)
        identifier = self.create_scene(view)
        self.assertEqual(self.controls.workflow.jobs[identifier]['seconds'], 15)
        view.length.edit('Custom', self.client)
        self.assertTrue(view.seconds.visible)
        view.seconds.value = '121'
        view.prompt.value = 'Walk forward, then wave.'
        view.generate.click(self.client)
        self.assertEqual(len(self.controls.ids), 1)
        self.assertIn('120', view.error)
        view.seconds.value = '60'
        view.generate.click(self.client)
        identifier = list(self.controls.ids.values())[-1]
        self.assertEqual(self.controls.workflow.jobs[identifier]['seconds'], 60)
        self.controls.workflow.jobs[identifier]['status'] = 'completed'
        view.load.click(self.client)
        view.action_prompt.value = 'Walk slowly'
        view.action_length.edit('Custom', self.client)
        self.assertTrue(view.action_seconds.visible)
        view.action_seconds.value = '3.20'
        view.edit.click(self.client)
        self.assertIn(('edit', 'Walk slowly', 0, 'replace', 3.2, False), self.session.calls)

    def test_popup_undo_does_not_touch_another_take(self):
        view = self.controls.open(self.client)
        identifier = self.create_scene(view)
        self.controls.workflow.jobs[identifier]['status'] = 'completed'
        view.load.click(self.client)
        self.session._undo_action_edit = (SimpleNamespace(id='other-take'), object())
        self.session.can_undo_action_edit = True
        self.controls.update()
        self.assertTrue(view.undo.disabled)
        view.undo.click(self.client)
        self.assertNotIn(('undo',), self.session.calls)

    def test_reopened_project_take_is_ready_to_refine(self):
        # Simulate a fresh studio process after the project file was reopened:
        # take data exists, but the transient scene-job history is empty.
        self.session.takes['scene-take'] = self.session.scene_take
        self.session.active_take = 'scene-take'
        view = self.controls.open(self.client)
        self.assertEqual(self.controls.ids, {})
        self.assertIn('Saved performance', view.source.content)
        self.assertIn('Current scene: 8s · 2 movements', view.estimate.content)
        self.assertEqual(len(view.action_map), 2)
        self.assertFalse(view.edit.disabled)
        view.action_prompt.value = 'Walk toward the door'
        view.edit.click(self.client)
        self.assertIn(('edit', 'Walk toward the door', 0, 'replace', None, True),
                      self.session.calls)

    def test_current_take_is_not_implicit_fallback_for_unloaded_job(self):
        self.session.takes['scene-take'] = self.session.scene_take
        self.session.active_take = 'scene-take'
        view = self.controls.open(self.client)
        self.create_scene(view)
        self.assertEqual(view.actions.options, ('No movements yet',))
        self.assertTrue(view.edit.disabled)

    def test_popup_movement_progress_and_failure_replace_completed_job_status(self):
        view = self.controls.open(self.client)
        identifier = self.create_scene(view)
        self.controls.workflow.jobs[identifier]['status'] = 'completed'
        view.load.click(self.client)
        view.action_prompt.value = 'Walk sideways'
        view.edit.click(self.client)
        self.assertIn('Updating movement', view.status.content)
        self.assertTrue(view.cancel_edit.disabled is False)
        self.session.status = 'Regenerating actions · 2/3 chunks received; holding pose'
        self.controls.update()
        self.assertIn('Updating movement', view.status.content)
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
            segments=[{'prompt': 'Walk sideways', 'start': 0, 'end': 125},
                      {'prompt': 'Wave', 'start': 125, 'end': 220}])
        self.controls.update()
        self.assertIn('Movement updated', view.status.content)
        self.assertIn('Current scene: 8.8s · 2 movements', view.estimate.content)
        self.assertIn('5.00s long · Walk sideways', view.actions.options[0])
        self.controls.update()
        self.assertIn('Movement updated', view.status.content)
        self.assertEqual(view.selected_index, 0)
        self.assertFalse(view.edit.disabled)  # Our completed edit rebinds its movement.
        self.assertEqual(view.action_prompt.value, 'Walk sideways')

    def test_cancel_movement_update_is_scoped_to_popup_owner_and_version(self):
        view = self.controls.open(self.client)
        identifier = self.create_scene(view)
        self.controls.workflow.jobs[identifier]['status'] = 'completed'
        view.load.click(self.client)
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
        identifier = self.create_scene(view)
        self.controls.workflow.jobs[identifier]['status'] = 'completed'
        view.load.click(self.client)
        view.edit.click(self.client)
        view.cancel_edit.click(self.client)
        self.assertFalse(self.session.busy)
        self.assertIn('Update cancelled', view.status.content)
        self.assertIs(self.session.takes['scene-take'], self.session.scene_take)


if __name__ == '__main__':
    unittest.main()
