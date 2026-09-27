"""Native paired studio state: one learned pair, a stable cast, exact 30 fps."""
from __future__ import annotations

import math
from pathlib import Path
import threading
import time
import uuid

from native_pair_clip import (NativePairClip, FPS, SCHEMA, MAX_CAST, encode_project,
                              decode_project, load_source, validate_cast, validate_placement)
from paired_scene import EMPTY_SCENE, scene_copy, json_copy

COLORS = ((52, 209, 220), (250, 178, 78), (182, 129, 242), (247, 124, 146),
          (126, 220, 147), (121, 161, 250), (243, 220, 123), (219, 160, 121))
REVIEWED_HANDSHAKE = Path(__file__).parent / 'review/two-character/native-recovery/originals/handshake_seed42.npz'
REVIEWED_SPARRING = Path(__file__).parent / 'review/two-character/platform-integration/block37/block37.native.npz'


class NativePairSession:
    def __init__(self, provider=None, *, clock=time.monotonic):
        self._lock = threading.RLock()
        self._provider, self._clock = provider, clock
        self._cast = [{'id': f'actor_{i+1}', 'name': f'Actor {i+1}', 'color': list(COLORS[i])} for i in range(2)]
        self._pair = ('actor_1', 'actor_2')
        self._placement = {'x': 0., 'z': 0., 'yaw_degrees': 0.}
        self._active = self._closed = self._busy = self._playing = self._capturing = False
        self._clip = self._pending = self._failure = self._thread = self._last_request = None
        self._last_context = None
        self._last_performance = None
        self._progress = None
        self._context_source = None
        self._scene = scene_copy(EMPTY_SCENE)
        self._frame = self._revision = self._epoch = self._transport_revision = 0
        self._play_started = None

    def _open(self):
        if self._closed:
            raise RuntimeError('Native paired session is closed')
        if self._capturing:
            raise RuntimeError('Playback export is running; wait before changing the interaction')

    def _require_active(self):
        self._open()
        if not self._active:
            raise RuntimeError('Activate native paired interaction first')

    def _invalidate(self, *, clear_retry=True):
        self._epoch += 1
        self._pending = None
        self._progress = None
        if clear_retry:
            self._last_request = self._last_context = self._last_performance = None
            self._failure = None

    def _idle_edit(self):
        self._open()
        if self._busy:
            raise RuntimeError('Finish or cancel generation before editing the cast')

    @staticmethod
    def _check_geometry(clip, scene, placement, pair):
        if clip is None:
            return
        from native_pair_geometry import check_native_pair_geometry
        from native_pair_transition import shared_place_pair
        world = shared_place_pair(clip.joints, yaw=math.radians(placement['yaw_degrees']),
                                  translation=(placement['x'], 0., placement['z']))
        check_native_pair_geometry(world, scene, actor_ids=pair)

    @property
    def active(self):
        with self._lock:
            return self._active

    @property
    def available(self):
        return self._provider is not None and not self._closed

    @property
    def revision(self):
        return self._revision

    @property
    def scene_document(self):
        with self._lock:
            return json_copy(self._scene)

    def activate(self, active=True):
        if type(active) is not bool:
            raise ValueError('Activation must be a boolean')
        if not active:
            return self.deactivate()
        with self._lock:
            self._open()
            self._active = True
        return self.snapshot()

    set_active = activate

    def deactivate(self):
        with self._lock:
            self._open()
            self._invalidate()
            self._active = self._playing = False
            self._play_started = None
        return self.snapshot()

    stop = deactivate

    def configure_provider(self, provider):
        with self._lock:
            self._open()
            self._invalidate()
            self._provider = provider
        return self.snapshot()

    configure_client = configure_provider

    def add_actor(self, name=None, color=None):
        with self._lock:
            self._idle_edit()
            if len(self._cast) >= MAX_CAST:
                raise ValueError('The native cast supports at most 8 actors')
            identifier = 'actor_' + uuid.uuid4().hex[:12]
            actor = {'id': identifier, 'name': name or f'Actor {len(self._cast)+1}',
                     'color': list(color if color is not None else COLORS[len(self._cast)])}
            cast, _ = validate_cast(self._cast + [actor], self._pair)
            self._cast = cast
            self._invalidate()
            self._revision += 1
            return identifier

    def rename_actor(self, identifier, name):
        with self._lock:
            self._idle_edit()
            cast = json_copy(self._cast)
            actor = next((a for a in cast if a['id'] == identifier), None)
            if actor is None:
                raise ValueError('Unknown cast actor')
            actor['name'] = name
            self._cast, _ = validate_cast(cast, self._pair)
            self._invalidate()
            self._revision += 1
        return self.snapshot()

    def remove_actor(self, identifier):
        with self._lock:
            self._idle_edit()
            if identifier not in [a['id'] for a in self._cast]:
                raise ValueError('Unknown cast actor')
            if identifier in self._pair:
                raise ValueError('Choose another interaction pair before removing a selected actor')
            self._cast = [a for a in self._cast if a['id'] != identifier]
            self._invalidate()
            self._revision += 1
        return self.snapshot()

    def select_pair(self, first, second=None):
        pair = first if second is None else (first, second)
        with self._lock:
            self._open()
            _, pair = validate_cast(self._cast, pair)
            if pair != self._pair:
                self._invalidate()
                self._pair = pair
                self._playing = False
                self._frame = 0
                self._play_started = None
                self._revision += 1
        return self.snapshot()

    def set_hand_pose(self, hand_pose):
        if hand_pose not in ('relaxed', 'fists'):
            raise ValueError('Choose relaxed hands or explicitly authored fists')
        with self._lock:
            self._open()
            if self._busy:
                raise RuntimeError('Finish or cancel generation before changing the hand pose')
            if self._clip is None:
                raise ValueError('Load an interaction before choosing its hand pose')
            if self._clip.metadata.get('render_hand_pose', 'relaxed') != hand_pose:
                self._invalidate()
                self._clip = NativePairClip(self._clip.joints, self._clip.features,
                    dict(self._clip.metadata, render_hand_pose=hand_pose))
                self._revision += 1
        return self.snapshot()

    def set_placement(self, x=0., z=0., yaw_degrees=0.):
        placement = validate_placement({'x': x, 'z': z, 'yaw_degrees': yaw_degrees})
        with self._lock:
            self._open()
            if self._busy:
                raise RuntimeError('Finish or cancel generation before changing pair placement')
            if placement != self._placement:
                self._check_geometry(self._clip, self._scene, placement, self._pair)
                self._invalidate()
                self._placement = placement
                self._revision += 1
        return self.snapshot()

    def update_scene(self, scene_document):
        scene = scene_copy(scene_document)
        with self._lock:
            self._open()
            if scene != self._scene:
                self._check_geometry(self._clip, scene, self._placement, self._pair)
                self._invalidate()
                self._scene = scene
                self._revision += 1
        return self.snapshot()

    def load_source(self, source, scene_document=None):
        clip = source if isinstance(source, NativePairClip) else load_source(source)
        with self._lock:
            self._open()
            scene = scene_copy(self._scene if scene_document is None else scene_document)
            self._check_geometry(clip, scene, self._placement, self._pair)
            self._invalidate()
            self._context_source = None
            self._clip, self._scene = clip, scene
            self._frame = 0
            self._active = True
            self._playing = False
            self._play_started = None
            self._failure = None
            self._revision += 1
        return self.snapshot()

    def load_reviewed_handshake(self, scene_document=None):
        return self.load_source(REVIEWED_HANDSHAKE, scene_document)

    def load_sparring_preview(self, scene_document=None):
        source = load_source(REVIEWED_SPARRING)
        clip = NativePairClip(source.joints, source.features, dict(source.metadata,
            render_hand_pose='fists', reviewed_semantics='sparring preview; physical strikes not verified'))
        return self.load_source(clip, scene_document)

    def generate(self, prompt, seed, frames=210, scene_document=None):
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 500:
            raise ValueError('Describe the interaction in 1–500 characters')
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Seed must be a uint32')
        if type(frames) is not int or not 30 <= frames <= 210:
            raise ValueError('Generate 30–210 native frames')
        with self._lock:
            self._require_active()
            if self._provider is None:
                raise RuntimeError('Native paired generation is not configured; the reviewed library is available')
            if self._busy:
                raise RuntimeError('A native paired request is still running or cancelling')
            scene = scene_copy(self._scene if scene_document is None else scene_document)
            self._invalidate()
            epoch, provider = self._epoch, self._provider
            request = {'prompt': prompt.strip(), 'seed': seed, 'frames': frames}
            self._last_request = (request.copy(), scene)
            self._last_context = None
            self._pending = dict(request, request_id='native-' + uuid.uuid4().hex)
            self._busy, self._failure = True, None
            self._thread = threading.Thread(target=self._worker, args=(provider, request, scene, epoch),
                                            daemon=True, name='native-paired-studio')
            self._thread.start()
        return self.snapshot()

    def build_context(self, builder, scene_document=None):
        """Build an explicitly composed ARDY context without replacing good output early."""
        if not callable(builder):
            raise ValueError('ARDY context builder must be callable')
        with self._lock:
            self._require_active()
            if self._busy:
                raise RuntimeError('A native paired request is still running or cancelling')
            if self._clip is None:
                raise ValueError('Load or generate a paired interaction before building ARDY context')
            if self._clip.segments is not None:
                raise ValueError('This take is already composed; choose an original pair before adding ARDY context')
            scene = scene_copy(self._scene if scene_document is None else scene_document)
            # Give builders their own metadata object and immutable array owners.
            # A failed builder cannot mutate the retained source in place.
            base = NativePairClip(self._clip.joints, self._clip.features, self._clip.metadata)
            self._invalidate()
            epoch = self._epoch
            self._last_context = (builder, scene)
            self._pending = {'operation': 'context', 'request_id': 'context-' + uuid.uuid4().hex,
                             'prompt': base.metadata.get('prompt', ''), 'frames': 0}
            self._busy, self._failure = True, None
            self._thread = threading.Thread(target=self._context_worker, args=(builder, base, scene, epoch, self._clip),
                                            daemon=True, name='native-paired-context')
            self._thread.start()
        return self.snapshot()

    def build_performance(self, builder, scene_document=None, *, request=None):
        """Build a complete performance atomically, preserving the last good take.

        The captured builder owns generation and planning. It receives an isolated
        scene, cancellation predicate, and a progress callback. Retry reuses that
        builder and request only until the scene, cast, placement or source changes.
        """
        if not callable(builder):
            raise ValueError('Performance builder must be callable')
        captured_request = json_copy(request)
        with self._lock:
            self._require_active()
            if self._busy:
                raise RuntimeError('A native paired request is still running or cancelling')
            scene = scene_copy(self._scene if scene_document is None else scene_document)
            self._invalidate()
            epoch = self._epoch
            self._last_performance = (builder, scene, captured_request)
            self._pending = {'operation': 'performance', 'request_id': 'performance-' + uuid.uuid4().hex,
                             'request': captured_request, 'frames': 0}
            self._busy, self._failure = True, None
            self._thread = threading.Thread(target=self._performance_worker,
                args=(builder, scene, epoch), daemon=True, name='native-paired-performance')
            self._thread.start()
        return self.snapshot()

    def _performance_worker(self, builder, scene, epoch):
        def progress(value):
            value = json_copy(value, limit=8192)
            with self._lock:
                if not self._cancelled(epoch):
                    self._progress = value
        try:
            result = builder(json_copy(scene), cancelled=lambda: self._cancelled(epoch),
                             on_progress=progress)
            if not isinstance(result, dict) or not isinstance(result.get('clip'), NativePairClip):
                raise ValueError('Performance builder must return a complete native pair clip and placement')
            clip = result['clip']
            placement = validate_placement(result.get('placement'))
            with self._lock:
                if not self._cancelled(epoch):
                    self._check_geometry(clip, scene, placement, self._pair)
                    self._clip, self._scene, self._placement = clip, scene, placement
                    self._context_source = None
                    self._frame = 0
                    self._playing = True
                    self._play_started = self._clock()
                    self._revision += 1
        except Exception as exc:
            detail = str(exc)
            token = getattr(builder, 'token', None)
            if isinstance(token, str) and token:
                detail = detail.replace(token, '[redacted]')
            with self._lock:
                if self._epoch == epoch and not self._closed:
                    self._failure = detail[:500]
        finally:
            with self._lock:
                if self._epoch == epoch:
                    self._pending = None
                    self._progress = None
                self._busy = False

    def _context_worker(self, builder, base, scene, epoch, original):
        try:
            clip = builder(base, json_copy(scene), cancelled=lambda: self._cancelled(epoch))
            if not isinstance(clip, NativePairClip) or clip.segments is None:
                raise ValueError('ARDY context builder must return a complete clip with disclosed source segments')
            with self._lock:
                if not self._cancelled(epoch):
                    self._check_geometry(clip, scene, self._placement, self._pair)
                    self._context_source = original
                    self._clip, self._scene = clip, scene
                    self._frame = 0
                    self._playing = True
                    self._play_started = self._clock()
                    self._revision += 1
        except Exception as exc:
            detail = str(exc)
            token = getattr(builder, 'token', None)
            if isinstance(token, str) and token:
                detail = detail.replace(token, '[redacted]')
            with self._lock:
                if self._epoch == epoch and not self._closed:
                    self._failure = detail[:500]
        finally:
            with self._lock:
                if self._epoch == epoch:
                    self._pending = None
                self._busy = False

    def _cancelled(self, epoch):
        with self._lock:
            return self._closed or not self._active or self._epoch != epoch

    def _worker(self, provider, request, scene, epoch):
        try:
            clip = provider.generate(**request, cancelled=lambda: self._cancelled(epoch))
            if not isinstance(clip, NativePairClip) or clip.frames != request['frames']:
                raise ValueError('Provider must return a complete exact native pair at the requested length')
            with self._lock:
                if not self._cancelled(epoch):
                    self._check_geometry(clip, scene, self._placement, self._pair)
                    self._context_source = None
                    self._clip, self._scene = clip, scene
                    self._frame = 0
                    self._playing = True
                    self._play_started = self._clock()
                    self._revision += 1
        except Exception as exc:
            detail = str(exc)
            token = getattr(provider, 'token', None)
            if isinstance(token, str) and token:
                detail = detail.replace(token, '[redacted]')
            with self._lock:
                if self._epoch == epoch and not self._closed:
                    self._failure = detail[:500]
        finally:
            with self._lock:
                if self._epoch == epoch:
                    self._pending = None
                self._busy = False

    def cancel(self):
        with self._lock:
            self._open()
            self._invalidate()
            self._failure = None
        return self.snapshot()

    def retry(self):
        with self._lock:
            performance = self._last_performance
            if performance is not None:
                builder, scene, request = performance
                return self.build_performance(builder, scene_document=scene, request=request)
            context = self._last_context
            if context is not None:
                builder, scene = context
                return self.build_context(builder, scene_document=scene)
            if self._last_request is None:
                raise ValueError('No generation request to retry')
            request, scene = self._last_request
            return self.generate(**request, scene_document=scene)

    def original_pair_clip(self):
        """Last context source retained in memory; source files remain separately archived."""
        with self._lock:
            return self._context_source

    def timeline_clip(self):
        with self._lock:
            return self._clip

    def frame_pose(self, frame=None):
        with self._lock:
            if self._clip is None:
                return None
            index = self._frame if frame is None else frame
            if type(index) is not int or not 0 <= index < self._clip.frames:
                raise ValueError('Native frame is outside the clip')
            return self._clip.joints[index]

    def play(self):
        with self._lock:
            self._require_active()
            if self._clip is None:
                raise ValueError('Load or generate native motion first')
            if self._frame == self._clip.frames - 1:
                self._frame = 0
            self._playing = True
            self._play_started = self._clock() - self._frame / FPS
            self._transport_revision += 1
        return self.snapshot()

    def pause(self):
        with self._lock:
            self._open()
            self._tick(self._clock())
            self._playing = False
            self._play_started = None
            self._transport_revision += 1
        return self.snapshot()

    def restart(self):
        self.seek(0)
        return self.play()

    def seek(self, frame):
        with self._lock:
            self._open()
            if self._clip is None or type(frame) is not int or not 0 <= frame < self._clip.frames:
                raise ValueError('Native frame is outside the clip')
            self._frame = frame
            if self._playing:
                self._play_started = self._clock() - frame / FPS
            self._transport_revision += 1
        return self.snapshot()

    def _tick(self, now):
        if self._capturing:
            return
        if self._active and self._playing and self._clip is not None:
            self._frame = min(self._clip.frames - 1, max(0, int((now - self._play_started) * FPS + 1e-7)))
            if self._frame == self._clip.frames - 1:
                self._playing = False
                self._play_started = None

    def tick(self, now=None):
        now = self._clock() if now is None else now
        if type(now) not in (float, int) or not math.isfinite(now):
            raise ValueError('Native playback clock must be finite')
        with self._lock:
            self._tick(now)
        return self.snapshot()

    def snapshot(self):
        with self._lock:
            frames = self._clip.frames if self._clip is not None else 0
            phase = ('inactive' if not self._active else
                     'generating' if self._busy and self._pending else 'cancelling' if self._busy else
                     'generation_failed' if self._failure else 'playing' if self._playing else
                     'paused' if frames else 'ready')
            messages = {'inactive': 'Native paired interactions inactive.', 'generating': 'Generating both actors together; current motion retained.',
                        'cancelling': 'Cancelling; late output will be ignored.', 'generation_failed': f'Generation failed: {self._failure}',
                        'playing': 'Playing native paired motion at 30 fps.', 'paused': 'Native paired motion paused.',
                        'ready': 'Choose two actors and play the reviewed handshake or generate an interaction.'}
            pending = self._pending or {}
            if phase == 'generating' and pending.get('operation') == 'performance':
                messages[phase] = 'Building the complete cast performance; current motion retained.'
            if phase == 'generating' and pending.get('operation') == 'context':
                messages[phase] = 'Building ARDY approach and departure around the preserved paired motion.'
            metadata = self._clip.metadata if self._clip is not None else {}
            segments = self._clip.segments if self._clip is not None else None
            composed = segments is not None
            if composed and phase in ('playing', 'paused'):
                messages[phase] = ('Playing' if phase == 'playing' else 'Paused') + ' composed motion at 30 fps; source and authored segments are labeled.'
            if segments is None:
                segments = [{'start': 0, 'end': frames, 'prompt': metadata.get('prompt', ''),
                             'source': self._clip.source, 'kind': 'paired_action'}] if frames else []
            return {'active': self._active, 'available': self.available, 'initialized': bool(frames),
                    'placement': dict(self._placement), 'cast': json_copy(self._cast), 'selected_pair': list(self._pair), 'actor_ids': list(self._pair),
                    'phase': phase, 'status': messages[phase], 'total_frames': frames, 'frame': self._frame,
                    'transport_revision': self._transport_revision,
                    'playing': self._playing, 'revision': self._revision, 'epoch': self._epoch, 'fps': FPS,
                    'source': self._clip.source if self._clip is not None else 'InterGen', 'schema': SCHEMA,
                    'composed': composed, 'hand_pose': metadata.get('render_hand_pose', 'relaxed'),
                    'research_only': True, 'license': metadata.get('license', 'CC BY-NC-SA 4.0'),
                    'physical_contact_verified': False, 'scene_conditioned': False, 'native_features_available':
                    self._clip is not None and self._clip.features is not None,
                    'capturing': self._capturing, 'pending': self._pending is not None, 'busy': self._busy, 'failure': self._failure,
                    'request_id': pending.get('request_id'), 'requested_frames': pending.get('frames', 0),
                    'pending_operation': pending.get('operation', 'generate') if pending else None,
                    'progress': json_copy(self._progress),
                    'generated': bool(frames), 'buffer_frames': max(0, frames-self._frame),
                    'segments': segments}

    def begin_capture(self):
        with self._lock:
            self._require_active()
            if self._busy or self._clip is None:
                raise ValueError('Finish generation and load a complete interaction before export')
            self._tick(self._clock())
            self._playing = False
            self._play_started = None
            self._capturing = True
            self._transport_revision += 1
        return self.snapshot()

    def capture_seek(self, frame):
        with self._lock:
            if not self._capturing or self._closed:
                raise RuntimeError('No active native playback export')
            if type(frame) is not int or not 0 <= frame < self._clip.frames:
                raise ValueError('Native capture frame is outside the clip')
            self._frame = frame
            self._transport_revision += 1
        return self.snapshot()

    def end_capture(self, frame=0):
        with self._lock:
            if not self._capturing:
                return self.snapshot()
            self._capturing = False
            self._frame = max(0, min(int(frame), self._clip.frames - 1))
            self._playing = False
            self._play_started = None
            self._transport_revision += 1
        return self.snapshot()

    def save(self):
        with self._lock:
            if self._closed:
                raise RuntimeError('Native paired session is closed')
            return encode_project(self._clip, self._cast, self._pair, self._scene, self._frame, self._placement)

    save_project = save

    def load(self, content):
        clip, cast, pair, scene, frame, placement = decode_project(content)
        with self._lock:
            self._open()
            self._check_geometry(clip, scene, placement, pair)
            self._invalidate()
            self._context_source = None
            self._clip, self._cast, self._pair, self._scene, self._frame = clip, cast, pair, scene, frame
            self._placement = placement
            self._active, self._playing, self._failure = True, False, None
            self._play_started = None
            self._revision += 1
        return self.snapshot()

    load_project = load

    def close(self, timeout=1.):
        with self._lock:
            self._invalidate()
            self._active = self._playing = False
            self._closed = True
            self._capturing = False
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(max(0., min(float(timeout), 2.)))
