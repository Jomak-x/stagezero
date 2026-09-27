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
        self.assertIn(('edit', 'Wave with the left hand', 1, 'replace', 4.0),
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

    def test_paired_active_can_generate_without_interrupting_pair_until_explicit_load(self):
        self.core.active = False
        paired = {'active': True, 'busy': False, 'capturing': False}
        self.controls.paired_session = SimpleNamespace(snapshot=lambda: dict(paired))
        self.session.character_motion_enabled = False

        def handoff():
            self.handoffs.append(True)
            paired['active'] = False
            self.session.character_motion_enabled = True

        self.controls.on_story_activate = handoff
        view = self.controls.open(self.client)
        self.assertFalse(view.generate.disabled)
        identifier = self.create_scene(view)
        self.assertTrue(paired['active'])
        self.assertEqual(self.handoffs, [])
        self.controls.workflow.jobs[identifier]['status'] = 'completed'
        self.controls.update()
        self.assertFalse(self.controls.workflow.jobs[identifier]['loaded'])
        self.assertIsNone(self.session.active_take)
        view.load.click(self.client)
        self.assertTrue(self.controls.workflow.jobs[identifier]['loaded'])
        self.assertFalse(paired['active'])
        self.assertTrue(self.session.character_motion_enabled)
        self.assertEqual(self.core.deactivations, 0)
        self.assertEqual(self.handoffs, [True])
        self.assertFalse(view.edit.disabled)

    def test_paired_busy_or_export_blocks_load_before_g1_take_changes(self):
        self.core.active = False
        paired = {'active': True, 'busy': False, 'capturing': False}
        self.controls.paired_session = SimpleNamespace(snapshot=lambda: dict(paired))
        view = self.controls.open(self.client)
        identifier = self.create_scene(view)
        self.controls.workflow.jobs[identifier]['status'] = 'completed'
        for flag, error in (('busy', 'Finish or cancel paired generation'),
                            ('capturing', 'Wait for paired playback export')):
            with self.subTest(flag=flag):
                paired[flag] = True
                self.controls.update()
                view.load.click(self.client)
                self.assertIn(error, view.error)
                self.assertFalse(self.controls.workflow.jobs[identifier]['loaded'])
                self.assertEqual(self.session.takes, {})
                self.assertIsNone(self.session.active_take)
                self.assertEqual(self.session.project_revision, 0)
                self.assertEqual(self.handoffs, [])
                paired[flag] = False

    def test_inactive_pair_with_pending_work_still_blocks_automatic_load(self):
        self.core.active = False
        paired = {'active': False, 'busy': True, 'capturing': False}
        self.controls.paired_session = SimpleNamespace(snapshot=lambda: dict(paired))
        view = self.controls.open(self.client)
        identifier = self.create_scene(view)
        self.controls.workflow.jobs[identifier]['status'] = 'completed'
        for flag in ('busy', 'capturing'):
            paired.update(busy=flag == 'busy', capturing=flag == 'capturing')
            self.controls.update()
            self.assertFalse(self.controls.workflow.jobs[identifier]['loaded'])
            self.assertIsNone(self.session.active_take)
        paired.update(busy=False, capturing=False)
        self.controls.update()
        self.assertTrue(self.controls.workflow.jobs[identifier]['loaded'])
        self.assertEqual(self.handoffs, [])  # Automatic completion never switches modes.

    def test_explicit_activation_callback_runs_without_core_session(self):
        self.controls.core_session = None
        self.controls._activate_g1()
        self.assertEqual(self.handoffs, [True])
        self.assertEqual(self.core.deactivations, 0)

    def test_cancel_full_scene_does_not_cancel_or_switch_paired_generation(self):
        self.core.active = False
        paired = {'active': True, 'busy': True, 'capturing': False}
        self.controls.paired_session = SimpleNamespace(snapshot=lambda: dict(paired))
        self.session.character_motion_enabled = False
        view = self.controls.open(self.client)
        identifier = self.create_scene(view)
        view.cancel.click(self.client)
        self.assertEqual(self.controls.workflow.jobs[identifier]['status'], 'cancelled')
        self.assertEqual(paired, {'active': True, 'busy': True, 'capturing': False})
        self.assertEqual(self.handoffs, [])

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
        view.action_seconds.value = '3.20'
        view.edit.click(self.client)
        self.assertIn(('edit', 'Walk slowly', 0, 'replace', 3.2), self.session.calls)

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
        self.assertIn(('edit', 'Walk toward the door', 0, 'replace', 4.0),
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
