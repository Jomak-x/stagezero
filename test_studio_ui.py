"""Sidebar state and transport tests with counted Viser-style handles."""
from pathlib import Path
from functools import partial
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import socket
import threading
import unittest
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
import viser
from viser._gui_handles import GuiButtonGroupHandle

from directing import DirectorSession
from prompt_assistant import PromptAssistantResult
from prompt_assistant_ui import PromptAssistantUI
from studio_ui import StudioUI, CREATE, EXTEND, REPLACE, AUTO, SET_DURATION, TARGET_TOTAL
from takes import Take
from test_live_motion import ControlledBackend, wait_until


def ready_refiner(prompt, _answers, **_kwargs):
    """Keep Studio transport tests independent of the text provider."""
    return PromptAssistantResult(prompt, 'Direction is ready.', (), True, 'test')


class Handle:
    def __init__(self, **values):
        object.__setattr__(self, 'writes', [])
        object.__setattr__(self, 'callbacks', {})
        for key, value in values.items():
            object.__setattr__(self, key, value)

    def __setattr__(self, name, value):
        if name not in ('writes', 'callbacks'):
            self.writes.append((name, value))
        object.__setattr__(self, name, value)

    def on_click(self, callback):
        self.callbacks['click'] = callback
        return callback

    def on_update(self, callback):
        self.callbacks['update'] = callback
        return callback

    def on_upload(self, callback):
        self.callbacks['upload'] = callback
        return callback

    def click(self, value=None):
        if value is not None:
            self.value = value
        self.callbacks['click'](SimpleNamespace(client=object()))

    def edit(self, value):
        self.value = value
        self.callbacks['update'](SimpleNamespace(client=object()))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class ButtonGroupHandle(Handle):
    """Mirror Viser's button group: disabling it raises an assertion."""

    @property
    def disabled(self):
        return False

    @disabled.setter
    def disabled(self, disabled):
        assert not disabled, 'Button groups cannot be disabled.'


class Gui:
    def __init__(self):
        self.handles = []
        self.tab_labels = []

    def _handle(self, handle_type=Handle, **values):
        values.setdefault('visible', True)
        handle = handle_type(**values)
        self.handles.append(handle)
        return handle

    def add_html(self, content):
        return self._handle(content=content)

    def add_markdown(self, content):
        return self._handle(content=content)

    def add_button_group(self, label, options):
        return self._handle(handle_type=ButtonGroupHandle, label=label,
                            options=options, value=options[0], disabled=False)

    def add_text(self, label, initial_value='', **kwargs):
        return self._handle(label=label, value=initial_value, disabled=False)

    def add_button(self, label, **kwargs):
        return self._handle(label=label, disabled=False, visible=kwargs.get('visible', True))

    def add_dropdown(self, label, options, initial_value=None):
        return self._handle(label=label, options=options, value=initial_value or options[0], disabled=False)

    def add_checkbox(self, label, initial_value=False):
        return self._handle(label=label, value=initial_value, disabled=False)

    def add_upload_button(self, label, **kwargs):
        return self._handle(label=label, value=None, disabled=False)

    def add_tab_group(self):
        return self

    def add_tab(self, label):
        self.tab_labels.append(label)
        return self

    def add_folder(self, label, **kwargs):
        return self._handle(label=label)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class StudioUITests(unittest.TestCase):
    def setUp(self):
        # Transport is tested with real GuiApi handlers in test_upload_events;
        # this fixture models only the public layout and callback surface.
        upload_installer = patch('studio_ui.install_upload_snapshots')
        upload_installer.start()
        self.addCleanup(upload_installer.stop)
        assistant_factory = patch('studio_ui.PromptAssistantUI',
                                  partial(PromptAssistantUI, refiner=ready_refiner))
        assistant_factory.start()
        self.addCleanup(assistant_factory.stop)
        self.temp = TemporaryDirectory()
        self.backend = ControlledBackend()
        self.session = DirectorSession(
            self.backend,
            np.zeros((120, 34, 3), dtype=np.float32),
            np.tile(np.eye(3, dtype=np.float32), (120, 34, 1, 1)),
        )
        self.gui = Gui()
        camera = SimpleNamespace(build_gui=lambda gui: None)
        self.ui = StudioUI(SimpleNamespace(gui=self.gui), self.session, camera,
                           Path(self.temp.name), lambda gui: None)

    def tearDown(self):
        self.backend.release.set()
        self.temp.cleanup()

    def _wait_for(self, predicate):
        def after_update():
            self.ui.update()
            return predicate()
        wait_until(after_update)

    def test_idle_updates_publish_nothing_and_seek_draft_is_stable(self):
        for handle in self.gui.handles:
            handle.writes.clear()
        for _ in range(100):
            self.ui.update()
        self.assertEqual(sum(len(handle.writes) for handle in self.gui.handles), 0)

        self.ui.seek_time.value = '0.25'
        self.session.seek(30)
        for handle in self.gui.handles:
            handle.writes.clear()
        self.ui.update()
        self.assertEqual(self.ui.seek_time.value, '0.25')
        self.assertEqual(self.ui.seek_time.writes, [])
        self.assertFalse(self.ui.playhead.visible)
        self.assertIn('Paused', self.ui.status.content)
        self.assertLessEqual(sum(len(handle.writes) for handle in self.gui.handles), 3)

    def test_go_and_frame_steps_clamp_to_last_sample(self):
        self.ui.seek_time.value = '1.98'
        self.ui.seek_go.click()
        self.assertEqual(self.session.frame, 119)
        self.ui.update()
        self.assertIn('Finished', self.ui.status.content)
        self.ui.frames.click('−1 frame')
        self.assertEqual(self.session.frame, 118)
        self.ui.frames.click('+1 frame')
        self.assertEqual(self.session.frame, 119)
        self.ui.transport.click('Start')
        self.assertEqual(self.session.frame, 0)
        self.ui.seek_time.value = '2.00'  # Clip duration selects its final sample.
        self.ui.seek_go.click()
        self.assertEqual(self.session.frame, 119)
        self.ui.seek_time.value = '2.01'
        self.ui.seek_go.click()
        self.assertEqual(self.session.frame, 119)
        self.assertIn('2.00 s', self.session.status)

    def test_static_character_hides_motion_groups_then_restores_selected_take(self):
        take = self._seed_take(100)
        self.ui.edit_action.edit(EXTEND)
        self.session.seek(25)
        self.ui.update()
        source_positions = self.session.positions
        self.assertIsInstance(self.ui.transport, ButtonGroupHandle)
        self.assertIsInstance(self.ui.frames, ButtonGroupHandle)
        self.assertFalse(self.ui.transport.visible)
        self.assertTrue(self.ui.frames.visible)

        self.ui.transport.writes.clear()
        self.ui.frames.writes.clear()
        self.session.set_character_motion_enabled(False)
        self.ui.update()
        self.assertFalse(self.ui.transport.visible)
        self.assertFalse(self.ui.frames.visible)
        self.assertTrue(self.ui.generate.disabled)
        self.assertEqual(self.ui.transport.writes, [])
        self.assertEqual(self.ui.frames.writes, [('visible', False)])
        self.assertEqual(self.session.active_take, take.id)
        self.assertIs(self.session.takes[take.id], take)
        self.assertIs(self.session.positions, source_positions)
        self.assertEqual(self.session.frame, 25)

        # Delayed clicks from a hidden browser control must not move the clip.
        self.ui.transport.click('Play')
        self.ui.frames.click('End')
        self.assertFalse(self.session.playing)
        self.assertEqual(self.session.frame, 25)

        self.session.set_character_motion_enabled(True)
        self.ui.update()
        self.assertFalse(self.ui.transport.visible)
        self.assertTrue(self.ui.frames.visible)
        self.assertFalse(self.ui.generate.disabled)
        self.assertEqual(self.session.active_take, take.id)
        self.assertIs(self.session.positions, source_positions)
        self.assertEqual(self.session.frame, 25)
        self.ui.transport.click('Play')
        self.assertTrue(self.session.playing)
        self.ui.transport.click('Pause')
        self.ui.frames.click('+1 frame')
        self.assertEqual(self.session.frame, 26)

    def test_static_character_blocks_timeline_transport_and_action_generation(self):
        take = self._seed_segmented_take()
        self._timeline_edit(take.id, 0, 'replace', 'open-editor')
        self.ui.action_prompt.edit('Wave gently')
        self.session.seek(25)
        source_positions = self.session.positions
        self.session.set_character_motion_enabled(False)
        self.ui.update()
        self.assertTrue(self.ui.save_action.disabled)
        self.assertTrue(self.ui.seek_go.disabled)

        # Delayed clicks must not bypass static preview restrictions, including
        # the new timeline route and an editor opened before the model changed.
        self._timeline_edit(take.id, 0, 'play', 'static-play')
        self._timeline_edit(take.id, 0, 'start', 'static-start')
        self._timeline_edit(take.id, 1, 'replace', 'static-edit')
        self.ui.seek_time.value = '3.00'
        self.ui.seek_go.click()
        self.assertFalse(self.session.playing)
        self.assertEqual(self.session.frame, 25)
        self.assertEqual(self.ui.action_edit, (take.id, 0, 'replace'))
        with patch.object(self.session, 'submit_action_edit', autospec=True, return_value=True) as submit:
            self.ui.save_action.click()
            submit.assert_not_called()
            self.session.set_character_motion_enabled(True)
            self.ui.update()
            self.assertFalse(self.ui.save_action.disabled)
            self.assertFalse(self.ui.seek_go.disabled)
            self.assertEqual(self.session.active_take, take.id)
            self.assertIs(self.session.positions, source_positions)
            self.assertEqual(self.session.frame, 25)
            self.ui.save_action.click()
            submit.assert_called_once_with('Wave gently', 0, 'replace', seconds=2.0)

        self._timeline_edit(take.id, 0, 'play', 'animated-play')
        self.assertTrue(self.session.playing)
        self._timeline_edit(take.id, 0, 'pause', 'animated-pause')
        self.assertFalse(self.session.playing)

    def test_character_controls_preserve_tabs_and_guide_navigation(self):
        gui = Gui()
        character_guis = []
        camera = SimpleNamespace(build_gui=lambda gui: None)
        ui = StudioUI(SimpleNamespace(gui=gui), self.session, camera,
                      Path(self.temp.name), lambda gui: None,
                      character_controls=character_guis.append)
        self.assertEqual(character_guis, [gui])
        self.assertEqual(gui.tab_labels,
                         ['Motion', 'Takes', 'Scene', 'Character', 'View', 'Project', 'Guide'])
        with patch('studio_ui.navigate_tab') as navigate:
            ui.quick_actions.click('Guide')
            ui.guide_browse_takes.click()
            ui.guide_scene.click()
            self.assertEqual([gui.tab_labels[call.args[1]] for call in navigate.call_args_list],
                             ['Guide', 'Takes', 'Scene'])

    def test_create_switches_to_live_and_requires_valid_prompt(self):
        self.assertEqual(self.ui.edit_action.value, CREATE)
        self.assertEqual(self.ui.prompt.value, '')
        self.assertTrue(self.ui.generate.disabled)
        self.assertIn('Enter a direction', self.ui.prompt_count.content)
        self.ui.prompt.edit('A person waves with their right hand.')
        self.assertFalse(self.ui.generate.disabled)
        self.assertIn('4.16 seconds', self.ui.duration_preview.content)
        self.ui.prompt.edit('x' * 501)
        self.ui.update()
        self.assertTrue(self.ui.generate.disabled)
        self.assertIn('501/500', self.ui.prompt_count.content)
        self.ui.generate.click()
        self.assertFalse(self.session.busy)

        self.ui.prompt.edit('Wave gently')
        self.ui.update()
        self.assertFalse(self.ui.generate.disabled)
        self.ui.generate.click()
        self._wait_for(lambda: self.session.busy)
        self.assertEqual(self.session.mode, 'Live ARDY')
        self.assertTrue(self.session.busy)
        self.ui.update()
        self.assertIn('Generating', self.ui.status.content)
        self.assertTrue(self.ui.prompt.disabled)
        self.assertTrue(self.ui.cancel.visible)
        self.assertTrue(self.backend.started.wait(1))
        self.ui.cancel.click()
        self.assertFalse(self.session.busy)
        self.backend.release.set()
        wait_until(lambda: not self.session.busy)
        self.assertEqual(self.session.kind, 'reference')

    def test_first_motion_examples_fill_draft_without_submitting(self):
        self.assertTrue(self.ui.motion_intro.visible)
        self.assertIn('Create your first motion', self.ui.motion_intro.content)
        self.assertTrue(self.ui.ideas_folder.visible)
        self.assertLess(self.gui.handles.index(self.ui.ideas_folder),
                        self.gui.handles.index(self.ui.prompt))
        self.ui.ideas.click('Walk')
        self.assertEqual(self.ui.prompt.value, 'A person walks forward at a relaxed pace.')
        self.assertEqual(self.session.prompt, self.ui.prompt.value)
        self.assertFalse(self.session.busy)
        self.assertFalse(self.ui.generate.disabled)
        self.session.set_character_motion_enabled(False)
        self.ui.update()
        self.assertTrue(self.ui.generate.disabled)
        self.assertIn('motion-ready character', self.ui.motion_intro.content)

    def test_generation_progress_and_retry_use_current_draft(self):
        self.ui.ideas.click('Wave')
        submitted = []

        def start(prompt, **kwargs):
            submitted.append((prompt, kwargs))
            self.session.busy = True
            self.session.status = 'Generating 4.16 s new take · 0/2 chunks'

        self.session.submit = Mock(side_effect=start)
        self.ui.generate.click()
        self._wait_for(lambda: len(submitted) == 1)
        self.assertEqual(len(submitted), 1)
        self.assertIn('0/2 chunks', self.ui.motion_progress.content)
        self.assertIn('s elapsed', self.ui.motion_progress.content)
        self.assertTrue(self.ui.motion_progress.visible)
        self.session.status = 'Generating 4.16 s · 1/2 chunks received; holding pose'
        self.ui.update()
        self.assertIn('1/2 chunks received', self.ui.motion_progress.content)

        self.session.busy = False
        self.session.status = 'Generation failed · BackendError. Original take preserved; retry.'
        self.ui.update()
        self.assertEqual(self.ui.generate.label, 'Retry generation')
        self.assertEqual(self.ui.prompt.value, submitted[0][0])
        self.assertIn('Review the direction and length', self.ui.motion_progress.content)
        self.ui.generate.click()
        self._wait_for(lambda: len(submitted) == 2)
        self.assertEqual(len(submitted), 2)
        self.assertEqual(submitted[1], submitted[0])

        self.session.busy = False
        self.session.status = 'Generation failed · BackendError. Original take preserved; retry.'
        self.ui.update()
        self.ui.prompt.edit('Turn slowly')
        self.assertEqual(self.ui.generate.label, 'Generate motion')
        self.assertFalse(self.ui.motion_progress.visible)
        self.ui.prompt.edit(submitted[0][0])
        self.assertEqual(self.ui.generate.label, 'Retry generation')
        self.session.project_revision += 1
        self.ui.update()
        self.assertEqual(self.ui.generate.label, 'Generate motion')
        self.assertFalse(self.ui.motion_progress.visible)

    def test_project_save_and_real_status_are_available_above_tabs(self):
        self.assertLess(self.gui.handles.index(self.ui.project_name),
                        self.gui.handles.index(self.ui.motion_intro))
        self.assertLess(self.gui.handles.index(self.ui.save),
                        self.gui.handles.index(self.ui.motion_intro))
        self.assertIn('No project save in this session', self.ui.files.content)
        self.session.project_status = 'Unsaved changes · Save project stores every take'
        self.ui.update()
        self.assertIn('Unsaved changes', self.ui.files.content)
        client = SimpleNamespace(send_file_download=Mock())
        self.ui.save.callbacks['click'](SimpleNamespace(client=client))
        self.assertIn('Saved ', self.ui.files.content)
        self.assertNotIn('Unsaved changes', self.ui.files.content)
        client.send_file_download.assert_called_once()
        client.send_file_download.side_effect = OSError('download channel closed')
        self.ui.save.callbacks['click'](SimpleNamespace(client=client))
        self.assertIn('Saved ', self.ui.files.content)
        self.assertIn('Download failed', self.ui.files.content)
        self.assertNotIn('Save failed', self.ui.files.content)

    def _seed_take(self, length=100):
        positions = np.zeros((length, 34, 3), dtype=np.float32)
        rotations = np.tile(np.eye(3, dtype=np.float32), (length, 34, 1, 1))
        motion = np.zeros((length, 414), dtype=np.float32)
        self.session.set_mode('Live ARDY')
        take = Take('a', 'First', positions, rotations, motion)
        self.session.takes = {take.id: take}
        self.session.select_take(take.id)
        self.ui.update()
        self.ui.prompt.edit('A person waves with their right hand.')
        return take

    def _seed_segmented_take(self):
        take = self._seed_take(100)
        take.segments = [
            dict(start=0, end=50, prompt='Walk forward'),
            dict(start=50, end=100, prompt='Wave once'),
        ]
        self.ui.update()
        return take

    def _timeline_edit(self, take_id, index, operation, nonce='1'):
        self.ui.timeline_command.edit(json.dumps(dict(
            take_id=take_id, index=index, operation=operation, nonce=nonce)))

    def test_timeline_action_command_opens_focused_editor_and_routes_generation(self):
        take = self._seed_segmented_take()
        client = object()
        payload = dict(take_id=take.id, index=1, operation='replace', nonce='first')
        with patch('studio_ui.navigate_tab') as navigate:
            self.ui.timeline_command.value = json.dumps(payload)
            self.ui.timeline_command.callbacks['update'](SimpleNamespace(client=client))
            navigate.assert_called_once_with(self.ui.tabs, 0, client)
        self.assertEqual(self.ui.action_edit, (take.id, 1, 'replace'))
        self.assertIn('Edit action 2', self.ui.action_heading.content)
        self.assertIn('No following actions need regeneration', self.ui.action_note.content)
        self.assertEqual(self.ui.action_prompt.value, 'Wave once')
        self.assertEqual(self.ui.action_duration.value, '2.00')
        self.assertEqual(self.session.frame, 50)
        self.assertFalse(self.ui.prompt.visible)
        self.assertFalse(self.ui.ideas_folder.visible)
        self.assertFalse(self.ui.advanced_folder.visible)
        self.assertEqual(self.ui.save_action.label, 'Update motion')

        self.ui.action_prompt.edit('Wave twice')
        self.ui.action_duration.edit('3.00')
        self.session.submit_action_edit = Mock(return_value=True)
        self.ui.save_action.click()
        self.session.submit_action_edit.assert_called_once_with(
            'Wave twice', 1, 'replace', seconds=3.0)

    def test_action_edit_explains_exact_following_regeneration(self):
        take = self._seed_segmented_take()
        self._timeline_edit(take.id, 0, 'replace', 'replace-first')
        self.assertIn('1 following action will regenerate', self.ui.action_note.content)
        self.assertEqual(self.ui.save_action.label, 'Update motion')
        self._timeline_edit(take.id, 0, 'insert_before', 'before-first')
        self.assertIn('2 following actions will regenerate', self.ui.action_note.content)
        self._timeline_edit(take.id, 1, 'insert_after', 'after-last')
        self.assertIn('No following actions need regeneration', self.ui.action_note.content)

    def test_action_editor_rejects_replaced_source_with_same_take_id(self):
        take = self._seed_segmented_take()
        self._timeline_edit(take.id, 0, 'replace', 'original')
        self.ui.action_prompt.edit('Wave twice')
        replacement = Take(take.id, take.name, take.positions.copy(), take.rotations.copy(),
                           take.motion.copy(), segments=list(take.segments))
        self.session.takes[take.id] = replacement
        self.session.submit_action_edit = Mock(return_value=True)
        self.ui.save_action.click()
        self.session.submit_action_edit.assert_not_called()
        self.ui.update()
        self.assertIsNone(self.ui.action_edit)

    def test_timeline_command_rejects_stale_invalid_and_busy_input(self):
        take = self._seed_segmented_take()
        with patch('studio_ui.navigate_tab') as navigate:
            self._timeline_edit('other', 0, 'replace')
            self._timeline_edit(take.id, 3, 'replace')
            self._timeline_edit(take.id, True, 'replace')
            self._timeline_edit(take.id, 0, 'remove')
            self.ui.timeline_command.edit('invalid json')
            self.assertIsNone(self.ui.action_edit)
            self._timeline_edit(take.id, 0, 'insert_before', 'accepted')
            self.assertEqual(self.ui.action_edit, (take.id, 0, 'insert_before'))
            self.assertEqual(self.ui.action_prompt.value, '')
            self.assertEqual(self.ui.save_action.label, 'Add action')
            self.ui.cancel_action.click()
            self._timeline_edit(take.id, 1, 'replace', 'accepted')
            self.assertIsNone(self.ui.action_edit)
            self.session.busy = True
            self._timeline_edit(take.id, 1, 'replace', 'busy')
            self.assertIsNone(self.ui.action_edit)
            self.assertEqual(navigate.call_count, 1)
        self.session.busy = False

    def test_timeline_transport_uses_current_take_and_keeps_editor_state(self):
        take = self._seed_segmented_take()
        self._timeline_edit(take.id, 0, 'replace', 'edit')
        original_editor = self.ui.action_edit
        with patch('studio_ui.navigate_tab') as navigate:
            self._timeline_edit('old-take', 0, 'play', 'stale')
            self.assertFalse(self.session.playing)
            self._timeline_edit(take.id, 0, 'play', 'play')
            self.assertTrue(self.session.playing)
            self.assertEqual(self.ui.action_edit, original_editor)
            self._timeline_edit(take.id, 0, 'play', 'play')
            self._timeline_edit(take.id, 0, 'pause', 'pause')
            self.assertFalse(self.session.playing)
            self.session.seek(25)
            self._timeline_edit(take.id, 0, 'start', 'start')
            self.assertEqual(self.session.frame, 0)
            self.assertEqual(navigate.call_count, 0)
        self.session.busy = True
        with patch.object(self.session, 'pause') as pause:
            self._timeline_edit(take.id, 0, 'play', 'busy-play')
            self._timeline_edit(take.id, 0, 'pause', 'busy-pause')
            pause.assert_called_once_with()
        self.session.busy = False

    def test_selected_take_opens_compact_summary_and_advanced_ending(self):
        take = self._seed_segmented_take()
        self.assertIn('2 actions', self.ui.editor_context.content)
        self.assertFalse(self.ui.prompt.visible)
        self.assertFalse(self.ui.generate.visible)
        self.assertFalse(self.ui.edit_action.visible)
        self.assertTrue(self.ui.add_to_end.visible)
        self.assertTrue(self.ui.advanced_folder.visible)
        self.ui.advanced_replace.click()
        self.assertEqual(self.ui.edit_action.value, REPLACE)
        self.assertTrue(self.ui.prompt.visible)
        self.assertTrue(self.ui.cancel_alternate.visible)
        self.ui.cancel_alternate.click()
        self.assertFalse(self.ui.prompt.visible)
        self.assertEqual(self.ui.edit_action.value, EXTEND)

    def test_action_editor_validates_duration_and_clears_on_switch(self):
        take = self._seed_segmented_take()
        self._timeline_edit(take.id, 1, 'insert_after')
        self.assertIn('Add action to end', self.ui.action_heading.content)
        self.ui.action_prompt.edit('Turn around')
        self.ui.action_duration.edit('40')
        self.assertTrue(self.ui.save_action.disabled)
        self.assertIn('0.16 to 30', self.ui.action_feedback.content)
        self.ui.action_duration.edit('2.00')
        self.assertFalse(self.ui.save_action.disabled)
        self.ui.quick_actions.click('New take')
        self.assertIsNone(self.ui.action_edit)
        self.assertTrue(self.ui.prompt.visible)
        self.assertFalse(self.ui.add_to_end.visible)

    def test_action_edit_keeps_draft_on_failure_and_closes_after_commit(self):
        take = self._seed_segmented_take()
        self._timeline_edit(take.id, 0, 'replace')
        self.ui.action_prompt.edit("Step to the character's left")

        def start(*_args, **_kwargs):
            self.session.busy = True
            self.session.status = 'Regenerating 2 actions · 0/2 chunks'
            return True

        self.session.submit_action_edit = Mock(side_effect=start)
        self.ui.save_action.click()
        self.assertTrue(self.ui.save_action.disabled)
        self.assertIn('0/2 chunks', self.ui.action_progress.content)
        self.session.busy = False
        self.session.status = 'Action edit failed · BackendError. Original take preserved; retry.'
        self.ui.update()
        self.assertEqual(self.ui.action_edit, (take.id, 0, 'replace'))
        self.assertEqual(self.ui.action_prompt.value, "Step to the character's left")
        self.assertEqual(self.ui.save_action.label, 'Retry update motion')
        self.assertTrue(self.ui.action_progress.visible)

        self.ui.save_action.click()
        self.session.busy = False
        self.session.action_edit_revision += 1
        self.ui.update()
        self.assertIsNone(self.ui.action_edit)
        self.assertFalse(self.ui.prompt.visible)
        self.assertTrue(self.ui.add_to_end.visible)

    def test_add_to_end_uses_action_editor_and_undo_is_available(self):
        take = self._seed_segmented_take()
        self.assertTrue(self.ui.add_to_end.visible)
        self.ui.add_to_end.click()
        self.assertEqual(self.ui.action_edit, (take.id, 1, 'insert_after'))
        self.assertEqual(self.ui.action_prompt.value, '')
        self.ui.cancel_action.click()
        self.ui.prepare_extend.click()
        self.assertEqual(self.ui.action_edit, (take.id, 1, 'insert_after'))
        self.session.undo_action_edit = Mock(return_value=True)
        with patch.object(DirectorSession, 'can_undo_action_edit', new_callable=PropertyMock) as can_undo:
            can_undo.return_value = True
            self.ui.update()
            self.assertTrue(self.ui.undo_action.visible)
            self.ui.undo_action.click()
        self.session.undo_action_edit.assert_called_once_with()
        self.assertIsNone(self.ui.action_edit)

    def test_new_take_clears_draft_and_keeps_saved_takes(self):
        saved = self._seed_take()
        self.ui.edit_action.edit(EXTEND)
        self.ui.quick_actions.click('New take')
        self.assertEqual(set(self.session.takes), {saved.id})
        self.assertIsNone(self.session.active_take)
        self.assertEqual(self.session.kind, 'reference')
        self.assertEqual(self.ui.prompt.value, '')
        self.assertEqual(self.session.prompt, '')
        self.assertEqual(self.ui.edit_action.value, CREATE)
        self.assertTrue(self.ui.generate.disabled)
        self.assertIn('New take draft', self.ui.editor_context.content)

    def test_choosing_create_from_saved_take_starts_blank_draft(self):
        saved = self._seed_take()
        self.ui.edit_action.edit(EXTEND)
        self.ui.edit_action.edit(CREATE)
        self.assertIsNone(self.session.active_take)
        self.assertEqual(self.ui.prompt.value, '')
        self.assertIn(saved.id, self.session.takes)
        self.assertTrue(self.ui.generate.disabled)

    def test_take_rows_select_for_edit_and_context_actions(self):
        first = self._seed_take()
        positions, rotations, motion = first.positions.copy(), first.rotations.copy(), first.motion.copy()
        second = Take('b', 'Second', positions, rotations, motion)
        self.session.takes[second.id] = second
        self.ui.update()
        self.assertEqual(self.gui.tab_labels, ['Motion', 'Takes', 'Scene', 'View', 'Project', 'Guide'])
        self.assertFalse(self.ui.takes.visible)
        self.assertTrue(self.ui.take_slots[1].visible)
        self.assertIn('Edit Second', self.ui.take_slots[1].label)
        with patch('studio_ui.navigate_tab') as navigate:
            self.ui.take_slots[1].click()
            self.assertEqual(navigate.call_args.args[1], 0)
        self.assertEqual(self.session.active_take, second.id)
        self.assertFalse(self.session.playing)
        self.assertEqual(self.ui.edit_action.value, EXTEND)
        self.assertIn('Second · 4.00 s', self.ui.editor_context.content)
        with patch('studio_ui.navigate_tab') as navigate:
            self.ui.prepare_replace.click()
            self.assertEqual(navigate.call_args.args[1], 0)
        self.assertEqual(self.ui.edit_action.value, REPLACE)
        self.assertIn('Ready to change this ending', self.ui.status.content)

    def test_guide_actions_open_relevant_tabs(self):
        with patch('studio_ui.navigate_tab') as navigate:
            self.ui.quick_actions.click('Guide')
            self.ui.guide_browse_takes.click()
            self.ui.guide_scene.click()
            self.assertEqual([call.args[1] for call in navigate.call_args_list], [5, 1, 2])

    def test_remove_take_can_be_undone_from_takes(self):
        saved = self._seed_take()
        self.ui.remove_take.click()
        self.assertNotIn(saved.id, self.session.takes)
        self.assertFalse(self.ui.undo_remove.disabled)
        self.assertFalse(self.ui.save.disabled)
        self.ui.undo_remove.click()
        self.assertIn(saved.id, self.session.takes)
        self.assertEqual(self.session.active_take, saved.id)

    def test_extend_uses_selected_take_and_target_total_not_playhead(self):
        self._seed_take()
        self.session.seek(25)
        self.ui.edit_action.edit(EXTEND)
        self.ui.duration_mode.edit(TARGET_TOTAL)
        self.ui.duration_seconds.edit('9.00')
        self.assertIn('4.00 seconds → 9.00 seconds total', self.ui.duration_preview.content)
        self.session.submit = Mock()
        self.ui.generate.click()
        self._wait_for(lambda: self.session.submit.call_count == 1)
        self.session.submit.assert_called_once_with(
            'A person waves with their right hand.', seconds=5.0,
            edit_mode='extend', at_frame=None)
        self.assertEqual(self.session.frame, 25)

    def test_replace_uses_explicit_time_and_shows_original_preservation(self):
        self._seed_take()
        self.ui.edit_action.edit(REPLACE)
        self.ui.replace_time.edit('1.00')
        self.ui.duration_mode.edit(SET_DURATION)
        self.ui.duration_seconds.edit('2.00')
        self.assertIn('Keep first 1.00 seconds', self.ui.duration_preview.content)
        self.assertIn('3.00 second take', self.ui.duration_preview.content)
        self.assertIn('Original stays available', self.ui.duration_preview.content)
        self.session.submit = Mock()
        self.ui.generate.click()
        self._wait_for(lambda: self.session.submit.call_count == 1)
        self.session.submit.assert_called_once_with(
            'A person waves with their right hand.', seconds=2.0,
            edit_mode='replace', at_frame=25)

        self.ui.replace_time.edit('4.00')
        self.assertTrue(self.ui.generate.disabled)
        self.assertIn('Enter a time', self.ui.duration_preview.content)
        self.ui.replace_time.edit('0.00')
        self.assertFalse(self.ui.generate.disabled)
        self.ui.replace_time.edit('0.08')
        self.assertTrue(self.ui.generate.disabled)
        self.assertIn('0.16 seconds or later', self.ui.duration_preview.content)

    def test_prepare_edit_controls_and_invalid_total(self):
        self._seed_take()
        self.assertNotIn('no inference', self.ui.status.content)
        self.assertNotIn('Recorded generated take', self.ui.status.content)
        self.session.seek(50)
        self.ui.prepare_replace.click()
        self.ui.update()
        self.assertIn('Ready to change this ending', self.ui.status.content)
        self.assertEqual(self.ui.edit_action.value, REPLACE)
        self.assertEqual(self.ui.replace_time.value, '2.00')
        self.ui.prepare_extend.click()
        self.assertEqual(self.ui.edit_action.value, EXTEND)
        self.ui.duration_mode.edit(TARGET_TOTAL)
        self.ui.duration_seconds.edit('3.00')
        self.assertTrue(self.ui.generate.disabled)
        self.assertIn('must add 0.16–30 seconds', self.ui.duration_preview.content)
        self.ui.duration_mode.edit(AUTO)
        self.assertFalse(self.ui.duration_seconds.visible)
        self.assertFalse(self.ui.generate.disabled)

    def test_target_total_minimum_quantizes_exactly(self):
        self._seed_take(25)
        self.ui.edit_action.edit(EXTEND)
        self.ui.duration_mode.edit(TARGET_TOTAL)
        self.ui.duration_seconds.edit('1.16')
        self.assertFalse(self.ui.generate.disabled)
        self.assertIn('1.16 seconds total', self.ui.duration_preview.content)
        self.session.submit = Mock()
        self.ui.generate.click()
        self._wait_for(lambda: self.session.submit.call_count == 1)
        self.session.submit.assert_called_once_with(
            'A person waves with their right hand.', seconds=0.16,
            edit_mode='extend', at_frame=None)

    def test_auto_discloses_prompt_duration_cap(self):
        self.ui.prompt.edit('Wave for 60 seconds')
        self.assertIn('capped at 30 s', self.ui.duration_preview.content)
        self.assertIn('30.00 seconds', self.ui.duration_preview.content)

    def test_reference_and_recorded_states_have_distinct_duration(self):
        self.ui.mode.edit('Live ARDY')
        self.ui.update()
        self.assertIn('Reference pose', self.ui.status.content)
        self.assertFalse(self.ui.playhead.visible)
        self.assertFalse(self.ui.transport.visible)
        self.assertTrue(self.ui.seek_go.disabled)
        self.ui.mode.edit('Recorded preview')
        self.session.seek(119)
        self.ui.update()
        self.assertIn('Finished', self.ui.status.content)
        self.assertFalse(self.ui.playhead.visible)
        self.assertFalse(self.ui.transport.visible)

    def test_transport_visibility_during_draft_and_generation(self):
        self.assertFalse(self.ui.transport.visible)
        self.ui.quick_actions.click('New take')
        self.assertEqual(self.session.kind, 'reference')
        self.assertFalse(self.ui.transport.visible)
        self.ui.prompt.edit('Wave gently')
        self.ui.generate.click()
        self._wait_for(lambda: self.session.busy)
        self.assertTrue(self.session.busy)
        self.ui.update()
        self.assertFalse(self.ui.transport.visible)
        self.assertTrue(self.ui.seek_go.disabled)
        self.assertTrue(self.backend.started.wait(1))

    def test_real_viser_button_group_handles_draft_and_busy_updates(self):
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', 0))
            except OSError as exc:
                self.skipTest(f'localhost sockets unavailable: {exc}')
        server = viser.ViserServer(host='127.0.0.1', port=0, verbose=False)
        try:
            camera = SimpleNamespace(build_gui=lambda gui: None)
            ui = StudioUI(server, self.session, camera, Path(self.temp.name),
                          lambda gui: None)
            self.assertIsInstance(ui.transport, GuiButtonGroupHandle)
            self.assertFalse(ui.transport.visible)

            self.session.new_take()
            ui.update()
            self.assertEqual(self.session.kind, 'reference')
            self.assertFalse(ui.transport.visible)

            self.session.submit('Wave gently')
            self.assertTrue(self.session.busy)
            ui.update()
            self.assertFalse(ui.transport.visible)
            self.assertTrue(ui.seek_go.disabled)
        finally:
            self.backend.release.set()
            server.stop()

    def test_pending_take_choice_is_not_overwritten_before_callback(self):
        positions = np.zeros((4, 34, 3), dtype=np.float32)
        rotations = np.tile(np.eye(3, dtype=np.float32), (4, 34, 1, 1))
        motion = np.zeros((4, 414), dtype=np.float32)
        self.session.set_mode('Live ARDY')
        self.session.takes = {
            take_id: Take(take_id, name, positions.copy(), rotations.copy(), motion.copy())
            for take_id, name in (('a', 'First'), ('b', 'Second'))
        }
        self.session.select_take('a')
        self.ui.update()
        second = next(label for label, take_id in self.ui.take_map.items() if take_id == 'b')

        # The browser changes its local selection before Viser dispatches the
        # callback. A viewer update must leave that pending choice alone.
        self.ui.takes.value = second
        self.ui.takes.writes.clear()
        self.ui.update()
        self.assertEqual(self.ui.takes.value, second)
        self.assertFalse(any(name == 'value' for name, _ in self.ui.takes.writes))

        captured = threading.Event()

        class SignalingMap(dict):
            def get(self, key, default=None):
                value = super().get(key, default)
                captured.set()
                return value

        self.ui.take_map = SignalingMap(self.ui.take_map)
        with self.session.lock:
            thread = threading.Thread(
                target=self.ui.takes.callbacks['update'],
                args=(SimpleNamespace(client=object()),),
            )
            thread.start()
            self.assertTrue(captured.wait(1))
            # This replaces the option map while the callback waits on the
            # session lock; it must still act on the captured ID.
            self.ui.update()
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.session.active_take, 'b')
        self.ui.update()
        self.assertEqual(self.ui.takes.value, second)

    def test_paired_research_status_and_transport_take_priority_without_g1_changes(self):
        class Motion:
            def __init__(self, active, status):
                self.active = active
                self.status = status
                self.calls = []

            def snapshot(self):
                return {'active': self.active, 'status': self.status,
                        'total_frames': 120, 'frame': 7}

            def play(self): self.calls.append(('play',))
            def pause(self): self.calls.append(('pause',))
            def seek(self, frame): self.calls.append(('seek', frame))

        core = Motion(False, 'Native Core saved')
        paired = Motion(True, 'Joint pair ready')
        before = self.session.positions.copy()
        self.ui.core_session = core
        self.ui.paired_session = paired
        self.ui.transport.click('Play')
        self.assertEqual(paired.calls, [('play',)])
        self.assertEqual(core.calls, [])
        self.ui.update()
        self.assertIn('Cast performance', self.ui.status.content)
        self.assertTrue(np.array_equal(self.session.positions, before))

        paired.active = False
        core.active = True
        self.ui.transport.click('Pause')
        self.assertEqual(core.calls, [('pause',)])
        self.ui.update()
        self.assertIn('Scene direction', self.ui.status.content)

    def test_cast_uses_existing_transport_save_and_open_without_changing_g1(self):
        from cast_performance_session import CastPerformanceSession
        from cast_performance import encode_project, decode_project, cast_from_performance
        from test_cast_performance import performance
        from paired_scene import EMPTY_SCENE
        cast = CastPerformanceSession()
        clip = performance(3, frames=12)
        archive = encode_project(clip, cast_from_performance(clip), dict(EMPTY_SCENE, name='Cast background'))
        cast.load(archive)
        self.ui.cast_session = cast
        self.ui.on_native_open = cast.load
        before = self.session.positions.copy()
        original_tabs = tuple(self.gui.tab_labels)
        self.ui.transport.click('Play')
        self.assertTrue(cast.snapshot()['playing'])
        self.ui.transport.click('End')
        self.assertEqual(cast.snapshot()['frame'], 11)
        self.ui.folder = Path(self.temp.name) / 'projects'
        self.ui.folder.mkdir()
        self.ui.update()
        self.assertIn('Cast performance', self.ui.status.content)
        self.ui.save.callbacks['click'](SimpleNamespace(client=None))
        paths = list((Path(self.temp.name)/'cast-projects').glob('*.cast.stagezero.npz'))
        self.assertEqual(len(paths), 1)
        saved, _, scene, frame = decode_project(paths[0].read_bytes())
        self.assertEqual(saved.joints.tobytes(), clip.joints.tobytes())
        self.assertEqual(frame, 11)
        self.assertEqual(scene['name'], 'Cast background')
        self.assertIn(paths[0].name, self.ui.saved_map)
        cast.deactivate()
        self.ui.open_data(paths[0].read_bytes())
        self.assertTrue(cast.active)
        self.assertEqual(cast.timeline_clip().actor_ids, clip.actor_ids)
        self.assertTrue(np.array_equal(self.session.positions, before))
        self.assertEqual(tuple(self.gui.tab_labels), original_tabs)


    def test_native_playback_controls_are_visible_and_hide_for_work_capture_and_g1(self):
        from cast_performance_session import CastPerformanceSession
        from test_cast_performance import performance
        cast = CastPerformanceSession()
        self.addCleanup(cast.close)
        self.ui.cast_session = cast
        cast.activate()
        self.ui.update()
        self.assertFalse(self.ui.transport.visible)
        cast.load_performance(performance(3, frames=12))
        before = self.session.positions.copy()
        self.ui.update()
        self.assertTrue(self.ui.transport.visible)
        self.ui.transport.click('Play')
        self.assertTrue(cast.snapshot()['playing'])
        self.ui.transport.click('Pause')
        self.assertFalse(cast.snapshot()['playing'])
        cast.seek(7)
        self.ui.transport.click('Start')
        self.assertEqual(cast.snapshot()['frame'], 0)
        cast.begin_capture()
        self.ui.update()
        self.assertFalse(self.ui.transport.visible)
        self.ui.transport.click('Play')  # A stale client click stays harmless.
        self.assertFalse(cast.snapshot()['playing'])
        cast.end_capture(0)
        self.ui.update()
        self.assertTrue(self.ui.transport.visible)
        with patch.object(cast, 'snapshot', return_value=dict(cast.snapshot(), busy=True)):
            self.ui.update()
            self.assertFalse(self.ui.transport.visible)
            self.ui.transport.click('Play')
        self.assertFalse(cast.snapshot()['playing'])
        cast.deactivate()
        self.ui.update()
        self.assertFalse(self.ui.transport.visible)
        np.testing.assert_array_equal(self.session.positions, before)

    def test_new_take_waits_for_cast_work_then_uses_explicit_mode_handoff(self):
        state = {'active': True, 'busy': True, 'capturing': False,
                 'total_frames': 8, 'frame': 0, 'fps': 30, 'status': 'Generating'}
        self.ui.cast_session = SimpleNamespace(snapshot=lambda: dict(state))
        before = self.session.active_take
        calls = []
        def handoff():
            calls.append('g1')
            state['active'] = False
        self.ui.on_story_activate = handoff
        self.ui.quick_actions.click('New take')
        self.assertEqual(calls, [])
        self.assertEqual(self.session.active_take, before)
        self.assertIn('Finish generation', self.session.project_status)
        state['busy'] = False
        self.ui.quick_actions.click('New take')
        self.assertEqual(calls, ['g1'])
        self.assertFalse(state['active'])



if __name__ == '__main__':
    unittest.main()
