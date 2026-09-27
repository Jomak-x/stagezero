"""Offline character-line generation, cancellation, and take ownership tests."""

from dataclasses import replace
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from dialogue_directing import DialogueDirector
from dialogue_protocol import DialogueAssetsMessage, DialogueCommandMessage, DialogueStateMessage
from speech_service import SpeechError
from takes import Take, validate_take


VOICE = 'MaleVoice00000000001'


def until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError('Timed out waiting for dialogue state')


class FakeConnection:
    def __init__(self):
        self.messages = []

    def queue_message(self, message):
        self.messages.append(message)


class FakeServer:
    def __init__(self):
        self.connection = FakeConnection()
        self.clients = {1: SimpleNamespace(client_id=1, _websock_connection=self.connection)}
        self._websock_server = SimpleNamespace(register_handler=lambda *args: None)

    def on_client_disconnect(self, callback):
        self.disconnect = callback

    def get_clients(self):
        return self.clients


class FakeSpeech:
    def __init__(self, *, fail_once=False, block=False):
        self.fail_once = fail_once
        self.block = block
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def list_voice_options(self):
        return [{'id': VOICE, 'name': 'Roger', 'gender': 'male'}]

    def synthesize(self, text, voice_id):
        self.calls += 1
        self.entered.set()
        if self.block:
            self.release.wait(2)
        if self.fail_once:
            self.fail_once = False
            raise SpeechError('Speech service request failed; retry shortly')
        return dict(audio_id='audio-1', voice_id=voice_id, audio=b'0' * 300,
                    duration_seconds=.4)


def sample_session():
    positions = np.zeros((4, 34, 3), dtype=np.float32)
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None, None], 4 * 34, axis=0).reshape(4, 34, 3, 3)
    motion = np.zeros((4, 414), dtype=np.float32)
    take = Take('take-1', 'Take 1', positions, rotations, motion,
                segments=[dict(start=0, end=4, prompt='Walk')])
    validate_take(take)
    return SimpleNamespace(lock=threading.RLock(), takes={'take-1': take},
        active_take='take-1', mode='Live ARDY', kind='generated', frame=2, fps=25,
        playing=False, playback_speed=1., clip_revision=0, project_revision=0,
        camera_project_id='project-1', project_status='', positions=positions,
        rotations=rotations, motion=motion, _undo_action_edit=None)


class DialogueDirectorTests(unittest.TestCase):
    def setUp(self):
        self.session = sample_session()
        self.server = FakeServer()
        self.speech = FakeSpeech()
        self.enabled = True
        self.director = DialogueDirector(self.session, self.server,
            speech=self.speech, enabled=lambda: self.enabled)
        until(lambda: self.director.catalog_revision > 0)

    def tearDown(self):
        self.speech.release.set()
        self.director.close()

    def submit(self, request_id='request-1', text='Hello.'):
        self.director.command(1, DialogueCommandMessage(
            request_id, 'submit', 'take-1', 2, 'main', text, VOICE))

    def test_completed_line_extends_stationary_tail_and_publishes_audio(self):
        self.submit()
        until(lambda: any(cue.get('text') == 'Hello.' for cue in self.session.takes['take-1'].dialogue))
        take = self.session.takes['take-1']
        self.assertEqual(len(take.positions), 13)
        self.assertEqual((take.dialogue[0]['start_frame'], take.dialogue[0]['end_frame']), (2, 12))
        self.assertEqual(take.segments[-1]['prompt'], 'Dialogue hold')
        validate_take(take)
        self.assertIsNone(self.director.jobs['request-1']['source_take'])
        self.director.update(force=True)
        assets = [message for message in self.server.connection.messages
                  if isinstance(message, DialogueAssetsMessage)][-1]
        self.assertEqual(assets.assets[take.dialogue[0]['audio_id']], b'0' * 300)

    def test_failed_retry_cancel_discards_late_audio_and_duplicate_retry(self):
        self.speech.fail_once = True
        self.submit()
        until(lambda: self.director.jobs['request-1']['status'] == 'failed')
        self.assertIsNone(self.director.jobs['request-1']['source_take'])
        line_id = self.director.jobs['request-1']['line_id']
        self.speech.block = True
        self.director.command(1, DialogueCommandMessage('request-2', 'retry', line_id=line_id))
        until(lambda: self.director.jobs['request-2']['status'] == 'generating')
        self.director.command(1, DialogueCommandMessage('request-3', 'retry', line_id=line_id))
        self.assertNotIn('request-3', self.director.jobs)
        self.director.command(1, DialogueCommandMessage('request-4', 'cancel', line_id=line_id))
        self.speech.release.set()
        until(lambda: self.director.active is None)
        self.assertEqual(self.director.jobs['request-2']['status'], 'cancelled')
        self.assertEqual(self.session.takes['take-1'].dialogue, [])
        self.director.update(force=True)
        state = [message for message in self.server.connection.messages
                 if isinstance(message, DialogueStateMessage)][-1]
        self.assertEqual(len([row for row in state.lines if row['line_id'] == line_id]), 1)

    def test_replaced_take_and_native_mode_cannot_receive_late_audio(self):
        self.speech.block = True
        self.submit()
        until(lambda: self.speech.entered.is_set())
        original = self.session.takes['take-1']
        self.session.takes['take-1'] = replace(original, name='Edited')
        self.speech.release.set()
        until(lambda: self.director.active is None)
        self.assertEqual(self.session.takes['take-1'].dialogue, [])
        self.enabled = False
        self.director.update(force=True)
        state = [message for message in self.server.connection.messages
                 if isinstance(message, DialogueStateMessage)][-1]
        self.assertEqual(state.take_id, '')
        self.assertFalse(state.available)

    def test_recorded_preview_mode_discards_inflight_speech(self):
        self.speech.block = True
        self.submit()
        until(lambda: self.speech.entered.is_set())
        self.session.mode = 'Recorded preview'
        self.speech.release.set()
        until(lambda: self.director.active is None)
        self.assertEqual(self.session.takes['take-1'].dialogue, [])

    def test_project_revision_change_discards_inflight_speech(self):
        self.speech.block = True
        self.submit()
        until(lambda: self.speech.entered.is_set())
        self.session.project_revision += 1
        self.speech.release.set()
        until(lambda: self.director.active is None)
        self.assertEqual(self.session.takes['take-1'].dialogue, [])

    def test_cached_duplicate_line_keeps_shared_audio_until_last_remove(self):
        self.submit('first')
        until(lambda: len(self.session.takes['take-1'].dialogue) == 1)
        self.submit('second')
        until(lambda: len(self.session.takes['take-1'].dialogue) == 2)
        self.assertEqual(self.speech.calls, 1)
        first, second = self.session.takes['take-1'].dialogue
        self.assertEqual(first['audio_id'], second['audio_id'])
        self.director.command(1, DialogueCommandMessage('remove-1', 'remove',
            take_id='take-1', line_id=first['line_id']))
        self.assertEqual(len(self.session.takes['take-1'].dialogue), 1)
        self.assertIn(second['audio_id'], self.session.takes['take-1'].audio_assets)
        self.director.command(1, DialogueCommandMessage('remove-2', 'remove',
            take_id='take-1', line_id=second['line_id']))
        self.assertEqual(self.session.takes['take-1'].dialogue, [])
        self.assertEqual(self.session.takes['take-1'].audio_assets, {})


if __name__ == '__main__':
    unittest.main()
