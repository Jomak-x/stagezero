"""Socket integration tests without provider credentials or a GPU."""
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import msgspec
import numpy as np
from viser._messages import Message
from directing import DirectorSession
from story_jobs import StoryJobQueue
from story_workflow import StoryWorkflow
from test_story_jobs import FakeBackend as BaseBackend, plan, until
from voice_directing import VoiceDirecting
from voice_commands import route_voice_command
from voice_protocol import (VoiceCommandMessage, VoiceRecordingMessage,
                            VoiceQueueMessage)


class FakeBackend(BaseBackend):
    def __init__(self, url='fake', block=False):
        super().__init__(url)
        if block:
            self.release.clear()


class Server:
    def __init__(self):
        self._websock_server = Mock()
        self.messages = []
        connection = SimpleNamespace(queue_message=self.messages.append)
        self.clients = {1: SimpleNamespace(client_id=1, _websock_connection=connection)}
    def get_clients(self):
        return self.clients
    def on_client_disconnect(self, callback):
        self.disconnect = callback


class VoiceTests(unittest.TestCase):
    def setUp(self):
        self.server = Server()
        p = np.zeros((20, 34, 3), dtype=np.float32)
        r = np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1))
        self.session = DirectorSession(Mock(), p, r)
        self.queue = StoryJobQueue([FakeBackend('fake')])
        planner = Mock()
        planner.plan.side_effect = lambda prompt, context, seconds: dict(plan(seconds), prompt=prompt)
        self.workflow = StoryWorkflow(self.session, planner=planner, queue=self.queue)
        self.bridge = VoiceDirecting(self.server, self.session, self.workflow)
    def tearDown(self):
        self.bridge.close()
        self.workflow.close()
    def complete(self):
        self.bridge.command(1, VoiceCommandMessage('request', 'submit', 'Wave'))
        until(lambda: self.workflow.snapshot(self.bridge.active['job'])['status'] == 'completed')
        self.bridge.update(force=True)
    def test_invalid_recording_never_contacts_provider(self):
        self.bridge.speech = Mock()
        self.bridge.recording(1, VoiceRecordingMessage('bad', 'text/plain', b'bad'))
        self.bridge.speech.transcribe.assert_not_called()
        self.assertEqual(self.server.messages[-1].status, 'failed')
    def test_scene_change_during_transcription_drops_late_result(self):
        entered, release = threading.Event(), threading.Event()
        def transcribe(*args):
            entered.set()
            release.wait(2)
            return 'Wave'
        self.bridge.speech = Mock(transcribe=transcribe)
        self.bridge.recording(1, VoiceRecordingMessage('audio', 'audio/webm', b'a'*200))
        self.assertTrue(entered.wait(1))
        self.session.project_revision += 1
        release.set()
        until(lambda: not self.bridge.transcribing)
        self.assertFalse(self.workflow.jobs)
        self.assertEqual(self.server.messages[-1].status, 'cancelled')
    def test_cancelled_transcription_cannot_submit(self):
        entered, release = threading.Event(), threading.Event()
        def transcribe(*args):
            entered.set(); release.wait(2)
            return 'Wave'
        self.bridge.speech = Mock(transcribe=transcribe)
        self.bridge.recording(1, VoiceRecordingMessage('audio', 'audio/webm', b'a'*200))
        self.assertTrue(entered.wait(1))
        self.bridge.command(1, VoiceCommandMessage('audio', 'cancel'))
        release.set()
        until(lambda: not self.bridge.transcribing)
        self.assertFalse(self.workflow.jobs)
    def test_protocol_round_trip_uses_binary_audio(self):
        for message in [VoiceRecordingMessage('id', 'audio/webm', b'abc'),
                        VoiceCommandMessage('id', 'submit', 'Wave'),
                        VoiceRecordingMessage('id', 'audio/webm', b'abc', 'single_action'),
                        VoiceCommandMessage('id', 'submit', 'Wave', 'single_action')]:
            decoded = Message.deserialize(msgspec.msgpack.encode(message.as_serializable_dict()))
            self.assertEqual(decoded, message)
        encoded = VoiceRecordingMessage('t', 'audio/webm', b'abc').as_serializable_dict()
        self.assertEqual(encoded['audio'], b'abc')

    def test_full_scene_completes_and_is_registered(self):
        self.bridge.on_story_submitted = Mock()
        self.complete()
        self.assertEqual(self.bridge.requests['request']['status'], 'completed')
        self.assertTrue(self.session.playing)
        self.assertEqual(len(self.session.positions), 208)
        self.bridge.on_story_submitted.assert_called_once()

    def test_core_activation_precedes_scene_character_check(self):
        self.session.character_motion_enabled = False
        self.bridge.on_motion_activate = lambda: setattr(self.session, 'character_motion_enabled', True)
        self.complete()
        self.assertEqual(self.bridge.requests['request']['status'], 'completed')

    def test_cancel_after_commit_reports_completed(self):
        self.session.backend = FakeBackend('motion')
        self.bridge.command(1, VoiceCommandMessage('motion', 'submit', 'Wave', 'single_action'))
        until(lambda: not self.session.busy)
        self.bridge.command(1, VoiceCommandMessage('motion', 'cancel'))
        self.assertEqual(self.bridge.requests['motion']['status'], 'completed')

    def test_history_remains_bounded_while_oldest_is_running(self):
        backend = FakeBackend('motion', block=True)
        self.session.backend = backend
        self.bridge.command(1, VoiceCommandMessage('oldest', 'submit', 'Wave', 'single_action'))
        self.assertTrue(backend.started.wait(1))
        for i in range(45):
            identifier = f'queued-{i}'
            self.bridge.command(1, VoiceCommandMessage(identifier, 'submit', 'Jump', 'single_action'))
            self.bridge.command(1, VoiceCommandMessage(identifier, 'cancel'))
        self.assertLessEqual(len(self.bridge.history), 32)
        self.assertLessEqual(len(self.bridge.requests), 32)
        self.assertEqual(self.bridge.active['id'], 'oldest')
        backend.release.set()

    def test_scene_queue_preserves_both_completed_takes(self):
        self.bridge.command(1, VoiceCommandMessage('one', 'submit', 'generate scene walk then wave', 'auto'))
        self.bridge.command(1, VoiceCommandMessage('two', 'submit', 'generate scene jump then bow', 'auto'))
        def finished():
            self.bridge.update(force=True)
            return self.bridge.requests['two']['status'] == 'completed'
        until(finished)
        self.assertEqual(len(self.session.takes), 2)
        self.assertEqual(self.bridge.requests['one']['status'], 'completed')

    def test_named_scene_edit_preserves_source_and_duration(self):
        self.complete()
        source = self.session.takes[self.session.active_take]
        source.name = 'Walk to door'
        self.bridge.command(1, VoiceCommandMessage('edit', 'submit',
            'edit scene "Walk to door" to walk to the window instead', 'auto'))
        def finished():
            self.bridge.update(force=True)
            return self.bridge.requests['edit']['status'] == 'completed'
        until(finished)
        self.assertIs(self.session.takes[source.id], source)
        self.assertNotEqual(self.session.active_take, source.id)
        job = self.workflow.snapshot(self.bridge.requests['edit']['job'])
        self.assertIn('Existing actions:', job['prompt'])
        self.assertEqual(job['seconds'], len(source.positions) / 25)

    def test_named_action_uses_atomic_action_api(self):
        self.complete()
        source = self.session.takes[self.session.active_take]
        source.name = 'Walk to door'
        self.session.backend = FakeBackend('edit')
        self.bridge.command(1, VoiceCommandMessage('edit', 'submit',
            'edit action 2 in "Walk to door" to bow', 'auto'))
        until(lambda: not self.session.busy)
        self.bridge.update(force=True)
        edited = self.session.takes[source.id]
        self.assertEqual(self.bridge.requests['edit']['status'], 'completed')
        self.assertEqual(edited.segments[0]['prompt'], source.segments[0]['prompt'])
        self.assertEqual(edited.segments[1]['prompt'], 'bow')
        self.assertTrue(self.session.can_undo_action_edit)

    def test_single_action_uses_native_generation_and_reports_completion(self):
        backend = FakeBackend('motion')
        self.session.backend = backend
        submitted = Mock()
        self.bridge.on_single_action_submitted = submitted
        self.bridge.command(1, VoiceCommandMessage('motion', 'submit', 'Walk forward and wave', 'single_action'))
        self.assertTrue(self.session.busy)
        self.assertEqual(self.session.mode, 'Live ARDY')
        self.assertEqual(self.workflow.jobs, {})
        submitted.assert_called_once_with('Walk forward and wave')
        until(lambda: not self.session.busy)
        self.bridge.update(force=True)
        self.assertEqual(backend.calls[0][1], 'Walk forward and wave')
        self.assertEqual(self.bridge.requests['motion']['sent_status'][0], 'completed')
        self.assertTrue(self.session.playing)

    def test_single_action_cancel_only_invalidates_its_own_generation(self):
        backend = FakeBackend('motion', block=True)
        self.session.backend = backend
        self.bridge.command(1, VoiceCommandMessage('motion', 'submit', 'Wave', 'single_action'))
        self.assertTrue(backend.started.wait(1))
        self.session.submit('Jump', edit_mode='new')
        newer_version = self.session.version
        self.bridge.command(1, VoiceCommandMessage('motion', 'cancel'))
        self.assertEqual(self.session.version, newer_version)
        self.assertTrue(self.session.busy)
        backend.release.set()
        until(lambda: not self.session.busy)

    def test_single_action_does_not_interrupt_unrelated_generation(self):
        backend = FakeBackend('motion', block=True)
        self.session.backend = backend
        self.session.set_mode('Live ARDY')
        self.session.submit('Jump', edit_mode='new')
        self.assertTrue(backend.started.wait(1))
        existing_version = self.session.version
        self.bridge.command(1, VoiceCommandMessage('motion', 'submit', 'Wave', 'single_action'))
        self.assertEqual(self.session.version, existing_version)
        self.assertTrue(self.session.busy)
        self.assertIsNone(self.bridge.active)
        self.assertEqual(self.server.messages[-1].status, 'queued')
        backend.release.set()
        until(lambda: not self.session.busy)

    def test_single_action_has_native_prompt_limit(self):
        self.bridge.command(1, VoiceCommandMessage('motion', 'submit', 'A' * 501, 'single_action'))
        self.assertEqual(self.server.messages[-1].status, 'failed')
        self.assertIn('500 characters', self.server.messages[-1].detail)
        self.assertEqual(self.workflow.jobs, {})

    def test_recording_keeps_single_action_destination_after_transcription(self):
        backend = FakeBackend('motion')
        self.session.backend = backend
        self.bridge.speech = Mock(transcribe=Mock(return_value='Wave'))
        self.bridge.recording(1, VoiceRecordingMessage('audio', 'audio/webm', b'a' * 200, 'single_action'))
        until(lambda: not self.bridge.transcribing)
        until(lambda: not self.session.busy)
        self.assertEqual(self.workflow.jobs, {})
        self.assertEqual(backend.calls[0][1], 'Wave')

    def test_transcription_permission_error_is_actionable(self):
        from speech_service import SpeechError
        self.bridge.speech = Mock(transcribe=Mock(side_effect=SpeechError('Enable Speech to Text permission')))
        self.bridge.recording(1, VoiceRecordingMessage('audio', 'audio/webm', b'a' * 200))
        until(lambda: not self.bridge.transcribing)
        self.assertEqual(self.server.messages[-1].status, 'failed')
        self.assertEqual(self.server.messages[-1].detail, 'Enable Speech to Text permission')


    def test_stale_completed_request_stops_retrying_load(self):
        self.bridge.command(1, VoiceCommandMessage('request', 'submit', 'Wave'))
        until(lambda: self.workflow.snapshot(self.bridge.active['job'])['status'] == 'completed')
        self.session.project_revision += 1
        self.bridge.update(force=True)
        self.assertTrue(self.bridge.requests['request']['terminal'])
        self.assertFalse(self.session.takes)
        self.workflow.load = Mock(side_effect=AssertionError('must not retry stale job'))
        self.bridge.update(force=True)

    def test_auto_route_requires_clear_command_and_exact_named_target(self):
        self.assertEqual(route_voice_command('generate a full scene: cross the room', 'auto', {}).target, 'full_scene')
        self.assertEqual(route_voice_command('generate a short: wave', 'auto', {}).prompt, 'wave')
        with self.assertRaisesRegex(ValueError, 'Start with'):
            route_voice_command('wave', 'auto', {})
        with self.assertRaisesRegex(ValueError, 'No saved'):
            route_voice_command('edit Alex to wave', 'auto', {})
        named = SimpleNamespace(id='take-1', name='Take 1', segments=[
            dict(start=0, end=10, prompt='Walk'), dict(start=10, end=20, prompt='Wave')])
        takes = {named.id: named}
        route = route_voice_command('edit action 2 in "Take 1" to jump', 'auto', takes)
        self.assertEqual((route.take_id, route.edit_mode, route.at_frame, route.prompt),
                         ('take-1', 'action', 10, 'jump'))
        route = route_voice_command('edit Take 1 to jump', 'auto', takes)
        self.assertEqual((route.take_id, route.edit_mode, route.at_frame), ('take-1', 'replace', 0))
        route = route_voice_command('edit scene "Take 1" to make the ending calmer', 'auto', takes)
        self.assertEqual((route.target, route.take_id, route.edit_mode), ('full_scene', 'take-1', 'scene'))
        with self.assertRaisesRegex(ValueError, 'Action number'):
            route_voice_command('edit action 3 in Take 1 to jump', 'auto', takes)

    def test_fifo_actions_wait_and_preserve_each_request(self):
        backend = FakeBackend('motion', block=True)
        self.session.backend = backend
        self.bridge.command(1, VoiceCommandMessage('first', 'submit', 'generate a short wave', 'auto'))
        self.assertTrue(backend.started.wait(1))
        self.bridge.command(1, VoiceCommandMessage('second', 'submit', 'generate a short jump', 'auto'))
        self.assertEqual(self.bridge.requests['second']['status'], 'queued')
        self.assertEqual(self.bridge.active['id'], 'first')
        snapshot = [m for m in self.server.messages if isinstance(m, VoiceQueueMessage)][-1]
        self.assertEqual([r['request_id'] for r in snapshot.requests], ['first', 'second'])
        backend.release.set()
        until(lambda: not self.session.busy)
        self.bridge.update(force=True)
        until(lambda: len(backend.calls) == 2)
        until(lambda: not self.session.busy)
        self.bridge.update(force=True)
        self.assertEqual([call[1] for call in backend.calls], ['wave', 'jump'])
        self.assertEqual(self.bridge.requests['first']['status'], 'completed')
        self.assertEqual(self.bridge.requests['second']['status'], 'completed')

    def test_cancel_queued_request_does_not_interrupt_current_action(self):
        backend = FakeBackend('motion', block=True)
        self.session.backend = backend
        self.bridge.command(1, VoiceCommandMessage('first', 'submit', 'Wave', 'single_action'))
        self.assertTrue(backend.started.wait(1))
        self.bridge.command(1, VoiceCommandMessage('second', 'submit', 'Jump', 'single_action'))
        self.bridge.command(1, VoiceCommandMessage('second', 'cancel'))
        self.assertEqual(self.bridge.requests['second']['status'], 'cancelled')
        self.assertTrue(self.session.busy)
        backend.release.set()
        until(lambda: not self.session.busy)
        self.bridge.update(force=True)
        self.assertEqual(len(backend.calls), 1)

    def test_external_change_cancels_queued_request(self):
        backend = FakeBackend('motion', block=True)
        self.session.backend = backend
        self.bridge.command(1, VoiceCommandMessage('first', 'submit', 'Wave', 'single_action'))
        self.assertTrue(backend.started.wait(1))
        self.bridge.command(1, VoiceCommandMessage('second', 'submit', 'Jump', 'single_action'))
        self.session.seek(0)
        backend.release.set()
        self.bridge.update(force=True)
        self.assertEqual(self.bridge.requests['second']['status'], 'cancelled')

    def test_external_busy_generation_completes_then_voice_runs(self):
        backend = FakeBackend('motion', block=True)
        self.session.backend = backend
        self.session.set_mode('Live ARDY')
        self.session.submit('Existing', edit_mode='new')
        self.assertTrue(backend.started.wait(1))
        self.bridge.command(1, VoiceCommandMessage('voice', 'submit', 'Wave', 'single_action'))
        self.assertEqual(self.bridge.requests['voice']['status'], 'queued')
        backend.release.set()
        until(lambda: not self.session.busy)
        self.bridge.update(force=True)
        until(lambda: len(backend.calls) == 2)
        self.assertEqual([call[1] for call in backend.calls], ['Existing', 'Wave'])

if __name__ == '__main__':
    unittest.main()
