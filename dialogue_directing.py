"""Bounded, cancellable character lines attached to an existing G1 take."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import replace
import math
import threading
import uuid

from dialogue_protocol import (DialogueAssetsMessage, DialogueCommandMessage,
                               DialoguePlaybackMessage, DialogueStateMessage,
                               DialogueStatusMessage, DialogueVoicesMessage)
from speech_service import ElevenLabsSpeech, SpeechError
from take_editing import extend_take_hold
from takes import MAX_AUDIO_ASSETS, MAX_DIALOGUE_CUES, MAX_PROJECT_AUDIO_BYTES, validate_take


MAX_PENDING = 8
MAX_HISTORY = 64
MAX_CACHE_BYTES = 16 * 1024 * 1024


class DialogueDirector:
    def __init__(self, session, server, *, speech=None, enabled=None):
        self.session, self.server = session, server
        self.speech = speech if speech is not None else ElevenLabsSpeech.from_env()
        self.enabled = enabled if enabled is not None else (lambda: True)
        self.lock = threading.RLock()
        self.jobs: OrderedDict[str, dict] = OrderedDict()
        self.pending: deque[str] = deque()
        self.active: str | None = None
        self.catalog: list[dict[str, str]] = []
        self.catalog_detail = 'Loading character voices…'
        self.catalog_revision = 0
        self.sent_catalog: dict[int, int] = {}
        self.sent_assets: dict[int, tuple] = {}
        self.sent_state: dict[int, tuple] = {}
        self.cache: OrderedDict[tuple[str, str], dict] = OrderedDict()
        self.cache_bytes = 0
        self.closed = False
        server._websock_server.register_handler(DialogueCommandMessage, self.command)
        server.on_client_disconnect(self.disconnected)
        self._catalog_thread = threading.Thread(target=self._load_catalog,
            name='stagezero-dialogue-catalog', daemon=True)
        self._catalog_thread.start()

    def _send(self, client_id, message):
        client = self.server.get_clients().get(client_id)
        if client is not None:
            client._websock_connection.queue_message(message)

    def _load_catalog(self):
        try:
            voices = self.speech.list_voice_options()
            detail = '' if voices else 'No labeled male or female voices are available to this account'
        except SpeechError as exc:
            voices, detail = [], str(exc)
        except Exception:
            voices, detail = [], 'Voice catalog request failed; retry shortly'
        with self.lock:
            if self.closed:
                return
            self.catalog, self.catalog_detail = voices, detail
            self.catalog_revision += 1

    def refresh_catalog(self):
        with self.lock:
            if self.closed or self._catalog_thread.is_alive():
                return
            self.catalog_detail = 'Loading character voices…'
            self.catalog_revision += 1
            self._catalog_thread = threading.Thread(target=self._load_catalog,
                name='stagezero-dialogue-catalog', daemon=True)
            self._catalog_thread.start()

    def _context(self):
        mode_enabled = bool(self.enabled())
        with self.session.lock:
            take = self.session.takes.get(self.session.active_take)
            selected = (take if mode_enabled and self.session.mode == 'Live ARDY'
                        and self.session.kind == 'generated' else None)
            take_id = selected.id if selected is not None else ''
            take_name = selected.name if selected is not None else ''
            frame = int(self.session.frame)
            clock = DialoguePlaybackMessage(take_id, frame, float(self.session.fps),
                bool(self.session.playing) if selected is not None else False,
                float(self.session.playback_speed), int(self.session.clip_revision))
            assets = dict(selected.audio_assets) if selected is not None else {}
            cues = [dict(cue) for cue in selected.dialogue] if selected is not None else []
            asset_key = (take_id, id(selected), self.session.project_revision, mode_enabled)
        if not mode_enabled:
            detail = 'Character lines are available on a selected G1 take'
        elif selected is None:
            detail = 'Select a generated G1 take to add a character line'
        elif not self.catalog:
            detail = self.catalog_detail
        else:
            detail = ''
        available = bool(mode_enabled and selected is not None and self.catalog)
        rows = []
        for cue in cues:
            rows.append(dict(line_id=cue.get('line_id', cue['audio_id']), request_id='',
                take_id=take_id, start_frame=cue['start_frame'], end_frame=cue['end_frame'],
                character_id=cue.get('character_id', 'main'), text=cue['text'],
                voice_id=cue['voice_id'], audio_id=cue['audio_id'], status='completed',
                detail='', retryable=False))
        latest_attempts = {job['line_id']: job for job in self.jobs.values()}
        for job in latest_attempts.values():
            if job['take_id'] != take_id or job['status'] == 'completed':
                continue
            rows.append({name: job[name] for name in ('line_id', 'request_id', 'take_id',
                'start_frame', 'character_id', 'text', 'voice_id', 'status', 'detail', 'retryable')})
        state = DialogueStateMessage(take_id, take_name, frame,
            [{'id': 'main', 'name': 'Character'}] if selected is not None else [],
            rows, available, detail)
        return state, clock, asset_key, assets, cues

    def connected(self, client):
        self.update(force=True, client_ids=[client.client_id])

    def update(self, *, force=False, client_ids=None):
        with self.lock:
            if self.closed:
                return
            state, clock, asset_key, assets, cues = self._context()
            recipients = client_ids if client_ids is not None else list(self.server.get_clients())
            for client_id in recipients:
                if force or self.sent_catalog.get(client_id) != self.catalog_revision:
                    self._send(client_id, DialogueVoicesMessage(
                        list(self.catalog), bool(self.catalog), self.catalog_detail))
                    self.sent_catalog[client_id] = self.catalog_revision
                if force or self.sent_assets.get(client_id) != asset_key:
                    self._send(client_id, DialogueAssetsMessage(state.take_id, assets, cues))
                    self.sent_assets[client_id] = asset_key
                state_key = (state.take_id, state.frame, state.available, state.detail,
                             tuple((row['line_id'], row['status'], row.get('detail', '')) for row in state.lines))
                if force or self.sent_state.get(client_id) != state_key:
                    self._send(client_id, state)
                    self.sent_state[client_id] = state_key
                self._send(client_id, clock)

    def _status(self, job, status, detail='', retryable=False):
        job.update(status=status, detail=detail, retryable=retryable)
        if status in ('completed', 'failed', 'cancelled'):
            job['source_take'] = None
        self.sent_state.clear()
        self._send(job['client_id'], DialogueStatusMessage(
            job['request_id'], job['line_id'], status, detail, retryable))

    def _reject(self, client_id, request_id, detail):
        self._send(client_id, DialogueStatusMessage(request_id, '', 'failed', detail, False))

    def _find_job(self, line_id):
        return next((job for job in reversed(list(self.jobs.values()))
                     if job['line_id'] == line_id), None)

    def command(self, client_id, message: DialogueCommandMessage):
        with self.lock:
            request_id = message.request_id
            if self.closed or not isinstance(request_id, str) or not 1 <= len(request_id) <= 100:
                return
            if message.command == 'catalog':
                self.refresh_catalog()
                self.update(force=True, client_ids=[client_id])
                return
            if message.command == 'submit':
                self._submit(client_id, message)
            elif message.command == 'retry':
                self._retry(client_id, message)
            elif message.command == 'cancel':
                self._cancel(client_id, message)
            elif message.command == 'remove':
                self._remove(client_id, message)
            else:
                self._reject(client_id, request_id, 'Unknown dialogue command')

    def _submit(self, client_id, message):
        if message.request_id in self.jobs:
            return
        if len(self.pending) + (self.active is not None) >= MAX_PENDING:
            return self._reject(client_id, message.request_id, 'Voice queue is full; wait for a line to finish')
        if not isinstance(message.text, str) or not 1 <= len(message.text.strip()) <= 1000:
            return self._reject(client_id, message.request_id, 'Dialogue must be 1–1000 characters')
        if message.character_id != 'main':
            return self._reject(client_id, message.request_id, 'Select the G1 character')
        if message.voice_id not in {voice['id'] for voice in self.catalog}:
            return self._reject(client_id, message.request_id, 'Choose an available voice')
        if not self.enabled():
            return self._reject(client_id, message.request_id, 'Select a generated G1 take first')
        with self.session.lock:
            take = self.session.takes.get(self.session.active_take)
            if (take is None or self.session.mode != 'Live ARDY' or self.session.kind != 'generated'
                    or message.take_id != take.id):
                return self._reject(client_id, message.request_id, 'The selected take changed; select it again')
            if type(message.start_frame) is not int or not 0 <= message.start_frame < len(take.positions):
                return self._reject(client_id, message.request_id, 'Choose a frame within the selected take')
            project_id = self.session.camera_project_id
            source_revision = self.session.project_revision
        job = dict(line_id=str(uuid.uuid4()), request_id=message.request_id, client_id=client_id,
            take_id=message.take_id, project_id=project_id, start_frame=message.start_frame,
            character_id='main', text=message.text.strip(), voice_id=message.voice_id,
            status='queued', detail='', retryable=False, cancelled=False,
            source_take=take, source_revision=source_revision)
        self.jobs[message.request_id] = job
        self.pending.append(message.request_id)
        self._status(job, 'queued', 'Waiting to generate voice')
        self._trim_history()
        self._pump()

    def _retry(self, client_id, message):
        previous = self._find_job(message.line_id)
        if previous is None or previous['status'] != 'failed' or not previous['retryable']:
            return self._reject(client_id, message.request_id, 'This line cannot be retried')
        if message.request_id in self.jobs:
            return
        if len(self.pending) + (self.active is not None) >= MAX_PENDING:
            return self._reject(client_id, message.request_id, 'Voice queue is full')
        if not self.enabled():
            return self._reject(client_id, message.request_id, 'Select the original G1 take to retry')
        with self.session.lock:
            if (self.session.camera_project_id != previous['project_id'] or
                    self.session.active_take != previous['take_id'] or
                    previous['take_id'] not in self.session.takes or
                    self.session.mode != 'Live ARDY' or self.session.kind != 'generated'):
                return self._reject(client_id, message.request_id, 'The original take is no longer selected')
            current_take = self.session.takes[previous['take_id']]
            source_revision = self.session.project_revision
        job = dict(previous, request_id=message.request_id, client_id=client_id,
            status='queued', detail='', retryable=False, cancelled=False,
            source_take=current_take, source_revision=source_revision)
        self.jobs[message.request_id] = job
        self.pending.append(message.request_id)
        self._status(job, 'queued', 'Waiting to retry voice')
        self._trim_history()
        self._pump()

    def _cancel(self, client_id, message):
        job = self._find_job(message.line_id)
        if job is None or job['status'] not in ('queued', 'generating'):
            return self._reject(client_id, message.request_id, 'This line is no longer pending')
        job['cancelled'] = True
        if job['request_id'] in self.pending:
            self.pending.remove(job['request_id'])
        self._status(job, 'cancelled', 'Voice generation cancelled')
        self._send(client_id, DialogueStatusMessage(message.request_id,
            job['line_id'], 'cancelled', 'Voice generation cancelled', False))

    def _remove(self, client_id, message):
        if not self.enabled():
            return self._reject(client_id, message.request_id, 'Select the original G1 take to remove a line')
        with self.session.lock:
            take = self.session.takes.get(self.session.active_take)
            if (take is None or self.session.active_take != message.take_id or
                    self.session.mode != 'Live ARDY' or self.session.kind != 'generated'):
                return self._reject(client_id, message.request_id, 'The selected take changed')
            cue = next((cue for cue in take.dialogue if cue.get('line_id', cue['audio_id']) == message.line_id), None)
            if cue is None:
                return self._reject(client_id, message.request_id, 'Line is no longer in the selected take')
            cues = [item for item in take.dialogue if item is not cue]
            assets = dict(take.audio_assets)
            if not any(item['audio_id'] == cue['audio_id'] for item in cues):
                assets.pop(cue['audio_id'], None)
            candidate = replace(take, dialogue=cues, audio_assets=assets)
            validate_take(candidate)
            self.session.takes[take.id] = candidate
            self.session.project_revision += 1
            self.session.project_status = 'Unsaved changes · Save project stores every take'
        self.sent_assets.clear()
        self.sent_state.clear()
        self._send(client_id, DialogueStatusMessage(message.request_id,
            message.line_id, 'removed', 'Character line removed', False))

    def _trim_history(self):
        while len(self.jobs) > MAX_HISTORY:
            removable = next((request_id for request_id in self.jobs
                              if request_id != self.active and request_id not in self.pending), None)
            if removable is None:
                break
            self.jobs.pop(removable)

    def _pump(self):
        if self.active is not None or self.closed:
            return
        while self.pending:
            request_id = self.pending.popleft()
            job = self.jobs.get(request_id)
            if job is None or job['cancelled']:
                continue
            with self.session.lock:
                take = self.session.takes.get(job['take_id'])
                valid = (self.enabled() and take is not None and
                    take is job['source_take'] and
                    self.session.active_take == job['take_id'] and
                    self.session.camera_project_id == job['project_id'] and
                    self.session.project_revision == job['source_revision'] and
                    self.session.mode == 'Live ARDY' and self.session.kind == 'generated')
            if not valid:
                self._status(job, 'cancelled', 'Take or project changed before voice generation')
                continue
            self.active = request_id
            self._status(job, 'generating', 'Generating character voice…')
            thread = threading.Thread(target=self._run, args=(request_id, take),
                name='stagezero-dialogue-tts', daemon=True)
            thread.start()
            return

    def _cached_speech(self, job):
        key = (job['voice_id'], job['text'])
        if key in self.cache:
            self.cache.move_to_end(key)
            return dict(self.cache[key])
        result = self.speech.synthesize(job['text'], job['voice_id'])
        with self.lock:
            audio = result['audio']
            if len(audio) <= MAX_CACHE_BYTES:
                self.cache[key] = dict(result)
                self.cache_bytes += len(audio)
                while self.cache_bytes > MAX_CACHE_BYTES:
                    _, old = self.cache.popitem(last=False)
                    self.cache_bytes -= len(old['audio'])
        return result

    def _run(self, request_id, take_at_start):
        job = self.jobs[request_id]
        try:
            generated = self._cached_speech(job)
            with self.lock, self.session.lock:
                if job['cancelled'] or self.closed:
                    return
                take = self.session.takes.get(job['take_id'])
                if (take is not take_at_start or self.session.active_take != job['take_id'] or
                        self.session.camera_project_id != job['project_id'] or not self.enabled() or
                        self.session.project_revision != job['source_revision'] or
                        self.session.mode != 'Live ARDY' or self.session.kind != 'generated'):
                    self._status(job, 'cancelled', 'Take or project changed during voice generation')
                    return
                audio = generated['audio']
                duration = generated['duration_seconds']
                if (not isinstance(audio, bytes) or not math.isfinite(duration) or
                        not 0 < duration <= 30):
                    raise SpeechError('Speech service returned unusable audio')
                audio_id = generated['audio_id']
                current_assets = dict(take.audio_assets)
                if audio_id in current_assets and current_assets[audio_id] != audio:
                    audio_id = str(uuid.uuid4())
                current_assets[audio_id] = audio
                other_audio = sum(len(content) for item in self.session.takes.values()
                                  if item.id != take.id for content in item.audio_assets.values())
                if (len(take.dialogue) >= MAX_DIALOGUE_CUES or len(current_assets) > MAX_AUDIO_ASSETS or
                        other_audio + sum(map(len, current_assets.values())) > MAX_PROJECT_AUDIO_BYTES):
                    raise ValueError('This take has reached its voice audio budget')
                end_frame = job['start_frame'] + math.ceil(duration * 25)
                required_frames = end_frame + 1
                cue = dict(line_id=job['line_id'], character_id=job['character_id'],
                    text=job['text'], voice_id=job['voice_id'], audio_id=audio_id,
                    start_frame=job['start_frame'], end_frame=end_frame)
                completed = extend_take_hold(self.session, take.id, required_frames,
                    dialogue=[*take.dialogue, cue], audio_assets=current_assets)
                for pending_id in self.pending:
                    pending_job = self.jobs.get(pending_id)
                    if (pending_job is not None and pending_job['take_id'] == take.id and
                            pending_job['source_take'] is take):
                        pending_job['source_take'] = completed
                        pending_job['source_revision'] = self.session.project_revision
                self.sent_assets.clear()
                self.sent_state.clear()
                self._status(job, 'completed', 'Character line ready')
        except SpeechError as exc:
            with self.lock:
                if not job['cancelled'] and not self.closed:
                    self._status(job, 'failed', str(exc), True)
        except ValueError as exc:
            with self.lock:
                if not job['cancelled'] and not self.closed:
                    self._status(job, 'failed', str(exc), False)
        except Exception:
            with self.lock:
                if not job['cancelled'] and not self.closed:
                    self._status(job, 'failed', 'Voice generation failed; retry shortly', True)
        finally:
            with self.lock:
                if self.active == request_id:
                    self.active = None
                self._trim_history()
                self._pump()

    def disconnected(self, client):
        client_id = client.client_id
        with self.lock:
            self.sent_catalog.pop(client_id, None)
            self.sent_assets.pop(client_id, None)
            self.sent_state.pop(client_id, None)
            for request_id in list(self.pending):
                job = self.jobs[request_id]
                if job['client_id'] == client_id:
                    self.pending.remove(request_id)
                    job['cancelled'] = True
                    job['status'] = 'cancelled'

    def close(self):
        with self.lock:
            self.closed = True
            self.pending.clear()
            if self.active in self.jobs:
                self.jobs[self.active]['cancelled'] = True
            self.cache.clear()
            self.cache_bytes = 0
