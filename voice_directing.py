"""Bounded FIFO speech-to-text commands over the studio socket."""
from collections import deque
import threading
import time

from story_workflow import StoryWorkflow
from voice_commands import route_voice_command
from voice_protocol import (VoiceRecordingMessage, VoiceCommandMessage,
                            VoiceStatusMessage, VoiceQueueMessage)

MAX_RECORDING_BYTES = 8 * 1024 * 1024
MAX_PENDING = 8
MAX_HISTORY = 32
ALLOWED_AUDIO = {'audio/webm', 'audio/ogg', 'audio/mp4', 'audio/wav', 'audio/mpeg'}
TERMINAL = {'completed', 'failed', 'cancelled'}


class VoiceDirecting:
    def __init__(self, server, session, workflow=None, speech=None,
                 on_single_action_submitted=None, on_motion_activate=None, on_story_submitted=None):
        self.server, self.session = server, session
        self.owns_workflow = workflow is None
        self.workflow = workflow or StoryWorkflow(session)
        self.speech = speech
        self.on_single_action_submitted = on_single_action_submitted
        self.on_motion_activate = on_motion_activate
        self.on_story_submitted = on_story_submitted
        self.lock = threading.RLock()
        self.active = None
        self.pending = deque()
        self.requests = {}
        self.history = deque()
        self.last_tick = 0.
        self.closed = False
        self.transcribing = False
        server._websock_server.register_handler(VoiceRecordingMessage, self.recording)
        server._websock_server.register_handler(VoiceCommandMessage, self.command)
        server.on_client_disconnect(self.disconnected)

    def _send(self, client_id, message):
        client = self.server.get_clients().get(client_id)
        if client is not None:
            client._websock_connection.queue_message(message)

    def _snapshot(self):
        for client_id in self.server.get_clients():
            rows = [dict(request_id=r['id'], status=r.get('status', 'queued'),
                         detail=r.get('detail', ''), transcript=r.get('transcript', ''),
                         retryable=r.get('retryable', False), target=r['target'])
                    for identifier in self.history if (r := self.requests.get(identifier))
                    and r['client'] == client_id]
            self._send(client_id, VoiceQueueMessage(rows))

    def _status(self, request, status, detail='', retryable=False):
        value = (status, detail, request.get('transcript', ''), retryable)
        if request.get('sent_status') != value:
            request.update(status=status, detail=detail, retryable=retryable, sent_status=value)
            self._snapshot()
            self._send(request['client'], VoiceStatusMessage(request['id'], *value))

    def _state(self):
        with self.session.lock:
            return self.session.scene, self.session.project_revision, self.session.version

    def _rebase_pending(self, old, new):
        if old == new:
            return
        for identifier in self.pending:
            request = self.requests[identifier]
            if request['state'] == old:
                request['state'] = new
                with self.session.lock:
                    request['clip_at_enqueue'] = self.session.clip_revision

    def _enqueue(self, client_id, request_id, target, text='', audio=None, mime=''):
        if self.closed or not isinstance(request_id, str) or not 1 <= len(request_id) <= 100:
            return None
        if request_id in self.requests:
            return None
        if audio is None and (not isinstance(text, str) or len(text) > 2000):
            self._send(client_id, VoiceStatusMessage(request_id, 'failed',
                'Enter a direction between 1 and 2000 characters', '', False))
            return None
        if target not in ('auto', 'full_scene', 'single_action'):
            self._send(client_id, VoiceStatusMessage(request_id, 'failed',
                'Choose Auto, Single action, or Full scene', '', False))
            return None
        if len(self.pending) + int(self.active is not None) >= MAX_PENDING:
            self._send(client_id, VoiceStatusMessage(request_id, 'failed',
                'Eight directions are pending; wait or cancel one', '', False))
            return None
        if len(self.history) >= MAX_HISTORY:
            oldest = next((identifier for identifier in self.history
                           if self.requests[identifier].get('status') in TERMINAL), None)
            if oldest is None:
                self._send(client_id, VoiceStatusMessage(request_id, 'failed',
                    'Direction history is full; wait or cancel one', '', False))
                return None
            self.history.remove(oldest)
            self.requests.pop(oldest, None)
        with self.session.lock:
            state = self._state()
            clip = self.session.clip_revision
            busy = self.session.busy
        request = dict(id=request_id, client=client_id, target=target, transcript=text,
                       state=state, clip_at_enqueue=clip, waiting_external=busy,
                       audio=audio, mime=mime, job=None)
        self.requests[request_id] = request
        self.history.append(request_id)
        self.pending.append(request_id)
        self._status(request, 'queued', 'Waiting for earlier directions…')
        self._pump()
        return request

    def recording(self, client_id, message):
        with self.lock:
            mime = message.mime_type.split(';')[0] if isinstance(message.mime_type, str) else ''
            if (mime not in ALLOWED_AUDIO or not isinstance(message.audio, bytes)
                    or not 0 < len(message.audio) <= MAX_RECORDING_BYTES):
                self._send(client_id, VoiceStatusMessage(message.request_id, 'failed',
                    'Record up to 30 seconds in a supported audio format', '', False))
                return
            self._enqueue(client_id, message.request_id, message.target,
                          audio=message.audio, mime=mime)

    def _transcribe(self, request):
        try:
            from speech_service import ElevenLabsSpeech
            speech = self.speech or ElevenLabsSpeech.from_env()
            text = speech.transcribe(request['audio'], request['mime'])
            with self.lock:
                if self.active is request and not self.closed:
                    request['transcript'] = text.strip() if isinstance(text, str) else ''
                    request['audio'] = None
                    self._dispatch(request)
        except Exception as exc:
            with self.lock:
                if self.active is request:
                    from speech_service import SpeechError
                    detail = (str(exc) if isinstance(exc, SpeechError) else
                        'Could not transcribe. Check ElevenLabs configuration or type your direction.')
                    self._finish(request, 'failed', detail)
        finally:
            with self.lock:
                self.transcribing = False
                self._pump()

    def _pump(self):
        if self.closed or self.active is not None or self.transcribing:
            return
        while self.pending:
            request = self.requests[self.pending[0]]
            current = self._state()
            if (request['waiting_external'] and request['state'][0] is current[0]
                    and current[1] == request['state'][1] + 1
                    and current[2] == request['state'][2]):
                with self.session.lock:
                    if (not self.session.busy and self.session.kind == 'generated'
                            and self.session.clip_revision > request['clip_at_enqueue']):
                        request['state'] = current
                        request['clip_at_enqueue'] = self.session.clip_revision
                        request['waiting_external'] = False
            if (request['state'][0] is not current[0]
                    or request['state'][1:] != current[1:]):
                self.pending.popleft()
                request['audio'] = None
                self._status(request, 'cancelled', 'Scene or selection changed while queued')
                continue
            with self.session.lock:
                if self.session.busy:
                    return
            self.pending.popleft()
            self.active = request
            if request['audio'] is not None:
                self.transcribing = True
                self._status(request, 'transcribing', 'Turning your direction into text…')
                threading.Thread(target=self._transcribe, args=(request,),
                                 daemon=True, name='voice-transcription').start()
            else:
                self._dispatch(request)
            return

    def _dispatch(self, request):
        # StoryWorkflow.load uses this same ordering. Socket callbacks and the
        # viewer tick must never acquire these two locks in opposite orders.
        with self.workflow.lock, self.session.lock:
            current = self._state()
            if (request['state'][0] is not current[0]
                    or request['state'][1:] != current[1:]):
                self._finish(request, 'cancelled', 'Scene or selection changed before direction started')
                return
            try:
                route = route_voice_command(request['transcript'], request['target'], self.session.takes)
            except ValueError as exc:
                self._finish(request, 'failed', str(exc))
                return
            request['transcript'] = request['transcript'].strip()
            request['route'] = route
            request['target'] = route.target
            if self.on_motion_activate is not None:
                try:
                    self.on_motion_activate()
                except Exception:
                    self._finish(request, 'failed', 'Could not activate motion generation')
                    return
            self._rebase_pending(current, self._state())
            current = self._state()
            if route.target == 'full_scene' and not getattr(self.session, 'character_motion_enabled', True):
                self._finish(request, 'failed', 'Choose a motion-ready character before generating a scene')
                return
            if route.target == 'single_action':
                before = current
                if self.session.mode != 'Live ARDY':
                    self.session.set_mode('Live ARDY')
                if route.take_id:
                    if route.take_id not in self.session.takes:
                        self._finish(request, 'cancelled', 'Named take was removed')
                        return
                    self.session.select_take(route.take_id)
                previous_revision = self.session.clip_revision
                try:
                    if route.edit_mode == 'action':
                        take = self.session.takes[route.take_id]
                        index = next(i for i, segment in enumerate(take.segments)
                                     if segment['start'] == route.at_frame)
                        self.session.submit_action_edit(route.prompt, index, 'replace')
                    else:
                        self.session.submit(route.prompt, edit_mode=route.edit_mode or 'new',
                                            at_frame=route.at_frame)
                except Exception:
                    self._finish(request, 'failed', 'Could not start the named motion edit')
                    return
                if not self.session.busy:
                    self._finish(request, 'failed', self.session.status)
                    return
                request['generation_version'] = self.session.version
                request['clip_revision'] = previous_revision
                request['dispatch_state'] = self._state()
                self._rebase_pending(before, request['dispatch_state'])
                if self.on_single_action_submitted is not None:
                    try:
                        self.on_single_action_submitted(route.prompt)
                    except Exception:
                        pass
                self._status(request, 'running', 'Editing named motion…' if route.take_id else 'Generating one action…')
            else:
                self.session.pause()
                try:
                    story_prompt = route.prompt
                    if route.edit_mode == 'scene':
                        source = self.session.takes.get(route.take_id)
                        if source is None:
                            self._finish(request, 'cancelled', 'Named scene was removed')
                            return
                        existing = '; '.join(str(segment.get('prompt', ''))[:160]
                                             for segment in source.segments)
                        context = (f'Revise saved scene "{source.name}". Preserve its action '
                                   f'sequence where the requested edit allows. Existing actions: '
                                   f'{existing}. Requested change: ')
                        story_prompt = context[:max(0, 2000 - len(route.prompt))] + route.prompt
                        request['source_take_id'] = source.id
                    seconds = (min(120, max(.16, len(source.positions) / 25))
                               if route.edit_mode == 'scene' else 30)
                    request['job'] = self.workflow.submit(story_prompt, seconds=seconds)
                    request['story_state'] = self._state()
                    if self.on_story_submitted is not None:
                        self.on_story_submitted(request['job'])
                    self._status(request, 'planning', 'Planning the scene…')
                except Exception:
                    self._finish(request, 'failed', 'Could not start this direction; wait for pending scenes and retry')

    def _finish(self, request, status, detail, retryable=False):
        request['terminal'] = status in TERMINAL
        if request['terminal']:
            request['audio'] = None
        self._status(request, status, detail, retryable)
        if self.active is request:
            self.active = None
        self._pump()

    def _cancel(self, request):
        if request['id'] in self.pending:
            self.pending.remove(request['id'])
        request['audio'] = None
        if request.get('job'):
            self.workflow.cancel(request['job'])
        if self.active is request and request.get('generation_version') is not None:
            with self.session.lock:
                if self.session.busy and self.session.version == request['generation_version']:
                    before = self._state()
                    self.session._invalidate()
                    self.session.status = 'Voice direction cancelled · current take preserved'
                    self._rebase_pending(before, self._state())
        self._finish(request, 'cancelled', 'Direction cancelled')

    def command(self, client_id, message):
        with self.lock:
            if message.command == 'submit':
                self._enqueue(client_id, message.request_id, message.target, text=message.text)
                return
            request = self.requests.get(message.request_id)
            if request is None or request['client'] != client_id:
                return
            if (message.command == 'cancel' and self.active is request
                    and request.get('generation_version') is not None
                    and not self.session.busy):
                # A result may have committed since the last UI tick. Report
                # its true outcome rather than claiming it was cancelled.
                self.update(force=True)
            if message.command == 'cancel' and request.get('status') not in TERMINAL:
                self._cancel(request)

    def update(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_tick < .05:
            return
        self.last_tick = now
        with self.lock, self.workflow.lock:
            request = self.active
            if request and request.get('generation_version') is not None:
                with self.session.lock:
                    if self.session.version != request['generation_version']:
                        self._finish(request, 'cancelled', 'Action changed while generating')
                    elif not self.session.busy:
                        completed = (self.session.clip_revision > request['clip_revision']
                                     and self.session.kind == 'generated')
                        current = self._state()
                        if (current[0] is request['dispatch_state'][0]
                                and current[1] == request['dispatch_state'][1] + int(completed)):
                            self._rebase_pending(request['dispatch_state'], current)
                        if completed and current[1] == request['dispatch_state'][1] + 1:
                            self._finish(request, 'completed', 'Playing your action')
                        else:
                            self._finish(request, 'failed', self.session.status)
                    else:
                        self._status(request, 'running', self.session.status)
            if request and self.active is request and request.get('job') and not request.get('take_id'):
                data = self.workflow.snapshot(request['job'])
                status = data['status']
                if status == 'completed':
                    with self.session.lock:
                        current = self._state()
                        if data.get('loaded'):
                            self._finish(request, 'completed', 'Scene already opened in the editor')
                        elif (current[0] is not request['story_state'][0]
                              or current[1:] != request['story_state'][1:]):
                            self._finish(request, 'cancelled', 'Editor changed; result remains available in Full scene')
                        elif self.session.busy:
                            self._status(request, 'queued', 'Waiting for current motion before loading…')
                        else:
                            before = self._state()
                            try:
                                if self.on_motion_activate is not None:
                                    self.on_motion_activate()
                                self.workflow.load(request['job'], automatic=True)
                                data = self.workflow.snapshot(request['job'])
                                if data.get('loaded'):
                                    request['take_id'] = self.session.active_take
                                    self._rebase_pending(before, self._state())
                                    self.session.play()
                                    self._finish(request, 'completed', 'Scene ready in Full scene and saved takes')
                                else:
                                    self._finish(request, 'cancelled', 'Scene changed; direction kept without replacing your work')
                            except (ValueError, RuntimeError):
                                self._finish(request, 'failed', 'Could not load this take into the current scene')
                elif status in ('failed', 'cancelled'):
                    self._finish(request, status, data.get('error') or status.capitalize())
                else:
                    progress = data.get('progress') or {}
                    detail = (f'Generating actions: {progress.get("completed_beats", 0)}/{progress.get("total_beats", 0)}'
                              if status == 'running' else 'Planning the scene…' if status == 'planning'
                              else 'Waiting for generation…')
                    self._status(request, status, detail)
            self._pump()

    def disconnected(self, client):
        with self.lock:
            # Remove all queued work first, so cancelling the active request
            # cannot dispatch another request from a client that has gone away.
            for identifier in list(self.pending):
                request = self.requests[identifier]
                if request['client'] == client.client_id:
                    self.pending.remove(identifier)
                    request['audio'] = None
                    self._status(request, 'cancelled', 'Client disconnected')
            for request in list(self.requests.values()):
                if (request['client'] == client.client_id and request.get('status') not in TERMINAL
                        and request['id'] not in self.pending):
                    self._cancel(request)

    def close(self):
        with self.lock:
            self.closed = True
            for request in list(self.requests.values()):
                if request.get('status') not in TERMINAL:
                    self._cancel(request)
            if self.owns_workflow:
                self.workflow.close()
