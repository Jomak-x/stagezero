"""Motion-tab controls for native 30 fps paired performances.

The panel talks only to NativePairStudioSession. G1, Core, and their archives
remain owned by their existing controls.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
import re
import threading
import time


MAX_CAST = 6
MAX_ARCHIVE_BYTES = 8_000_000
MAX_VIDEO_BYTES = 64_000_000
ARCHIVE_SUFFIX = ".native-pair.stagezero.npz"
HANDSHAKE_PROMPT = "Two people shake hands and step apart."
HANDSHAKE_SEED = 42
HAND_POSES = {"Relaxed": "relaxed", "Closed fists": "fists"}


class NativePairControls:
    """Build and refresh the native pair controls inside Studio's Motion tab."""

    def __init__(self, gui, session, studio_session, *, on_active, project_folder,
                 on_capture=None, on_frame_cast=None, on_build_context=None, on_open=None):
        self.session = session
        self.studio = studio_session
        self.on_active = on_active
        self.on_capture = on_capture
        self.on_frame_cast = on_frame_cast
        self.on_build_context = on_build_context
        self.on_open = on_open
        self.folder = Path(project_folder)
        self._notice = ""
        self._syncing = False
        self._cast_labels = {}
        self._saved_map = {}
        self._saved_checked_at = 0.0
        self._last_selected_actor = None
        self._last_pair = None
        self._last_placement = None
        self._preferred_actor_id = None
        self._capture_lock = threading.Lock()
        self._capture_pending = False
        self._capture_message = ""

        with gui.add_folder("Cast & paired motion · native", expand_by_default=True):
            gui.add_markdown(
                "Pair interactions use InterGen; ARDY approach and exit segments are labeled separately. "
                "The reviewed handshake shows partner timing. Hand contact and foot grounding "
                "are not physically verified."
            )
            self.view = gui.add_button("Show native cast", color="green")
            self.frame_cast = gui.add_button("Frame cast", color="gray")
            self.back = gui.add_button("Return to G1", color="gray")
            with gui.add_folder("Characters", expand_by_default=True):
                self.actor = gui.add_dropdown("Selected character", ("Loading cast",))
                self.actor_name = gui.add_text("Character name", initial_value="")
                self.rename = gui.add_button("Rename selected character", color="gray")
                self.remove = gui.add_button("Remove selected character", color="gray")
                self.new_name = gui.add_text("New character name", initial_value="")
                self.add = gui.add_button("Add character", color="gray")
            with gui.add_folder("Interaction", expand_by_default=True):
                self.first = gui.add_dropdown("First character", ("Loading cast",))
                self.second = gui.add_dropdown("Second character", ("Loading cast",))
                self.choose_pair = gui.add_button("Use this pair", color="gray")
                with gui.add_folder("Place the pair in the scene", expand_by_default=False):
                    gui.add_markdown("Move and turn both characters together while preserving their interaction.")
                    self.position_x = gui.add_number("Pair X · m", initial_value=0., min=-100., max=100., step=.1)
                    self.position_z = gui.add_number("Pair Z · m", initial_value=0., min=-100., max=100., step=.1)
                    self.yaw = gui.add_number("Pair direction · degrees", initial_value=0., min=-180., max=180., step=5.)
                    self.place = gui.add_button("Apply pair placement", color="gray")
                self.reviewed = gui.add_button("Play reviewed handshake", color="green")
                self.sparring = gui.add_button("Play sparring preview", color="gray")
                gui.add_markdown("Sparring preview shows non-contact movement; strikes are not verified.")
                self.prompt = gui.add_text("What should they do together?", initial_value=HANDSHAKE_PROMPT,
                                           multiline=True)
                self.seed = gui.add_text("Variation seed", initial_value=str(HANDSHAKE_SEED))
                self.generate = gui.add_button("Generate pair motion")
                self.build_context = gui.add_button("Add ARDY approach and exit · experimental", color="gray")
                gui.add_markdown("ARDY segments and authored transitions are labeled on the timeline; the interaction source stays unchanged.")
                self.cancel = gui.add_button("Cancel generation", color="gray")
                self.retry = gui.add_button("Retry", color="gray")
            with gui.add_folder("Playback", expand_by_default=True):
                self.play = gui.add_button("Play", color="gray")
                self.pause = gui.add_button("Pause", color="gray")
                self.restart = gui.add_button("Start over", color="gray")
                self.frame = gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
                self.hand_pose = gui.add_dropdown("Hand pose", tuple(HAND_POSES), initial_value="Relaxed")
                gui.add_markdown("Finger poses are authored display controls.")
                self.capture = gui.add_button("Export playback video", color="gray")
            with gui.add_folder("Saved native performances", expand_by_default=False):
                self.save = gui.add_button("Save performance + download", color="gray")
                self.saved = gui.add_dropdown("Saved performances", ("No saved performances",))
                self.open = gui.add_button("Open saved performance", color="gray")
                self.upload = gui.add_upload_button("Open native performance file", mime_type=".npz")
            self.status = gui.add_markdown("Preparing native cast…")
        self._bind()
        self.tick()

    @staticmethod
    def _set(handle, name, value):
        if getattr(handle, name, None) != value:
            setattr(handle, name, value)

    @staticmethod
    def _safe_text(value):
        # Viser's markdown is MDX. Escape user/model text before displaying it.
        safe = re.sub(r"([\\`*_\[\]()#+.!|~-])", r"\\\1", str(value))
        return escape(safe, quote=False).replace("{", "&#123;").replace("}", "&#125;")

    def _scene_document(self):
        return self.studio.scene_document()

    def _cast(self, snapshot):
        actors = snapshot.get("cast") or ()
        result = []
        for actor in actors:
            if isinstance(actor, dict) and isinstance(actor.get("id"), str):
                result.append((actor["id"], str(actor.get("name") or actor["id"])))
        return result

    def _selected_id(self, handle):
        return self._cast_labels.get(handle.value)

    def _pair_ids(self):
        first, second = self._selected_id(self.first), self._selected_id(self.second)
        if not first or not second or first == second:
            raise ValueError("Choose two different characters for the interaction.")
        return first, second

    def _run(self, action):
        self._notice = ""
        with self._capture_lock:
            if not self._capture_pending:
                self._capture_message = ""
        try:
            action()
        except Exception as exc:
            self._notice = str(exc)[:240]
        self.tick()

    def _activate(self):
        self.on_active(True)

    def _validate_name(self, value):
        name = value.strip()
        if not 1 <= len(name) <= 40:
            raise ValueError("Enter a character name from 1 to 40 characters.")
        return name

    def _generation_args(self):
        prompt = self.prompt.value.strip()
        if not 1 <= len(prompt) <= 500:
            raise ValueError("Describe the shared action in 1–500 characters.")
        try:
            seed = int(self.seed.value)
        except (TypeError, ValueError):
            raise ValueError("Variation seed must be an integer from 0 to 4294967295.") from None
        if not 0 <= seed < 2**32:
            raise ValueError("Variation seed must be an integer from 0 to 4294967295.")
        return prompt, seed

    def refresh_saved(self, *, force=False):
        now = time.monotonic()
        if not force and now - self._saved_checked_at < 1.0:
            return
        self._saved_checked_at = now
        root = self.folder.resolve()
        candidates = []
        try:
            for path in self.folder.glob("*" + ARCHIVE_SUFFIX):
                try:
                    if path.is_symlink() or path.resolve().parent != root or not path.is_file():
                        continue
                    stat = path.stat()
                    if 0 < stat.st_size <= MAX_ARCHIVE_BYTES:
                        candidates.append((stat.st_mtime_ns, path.name, path))
                except OSError:
                    continue
        except OSError:
            candidates = []
        candidates.sort(reverse=True)
        self._saved_map = {name: path for _, name, path in candidates[:100]}
        options = tuple(self._saved_map) or ("No saved performances",)
        self._set(self.saved, "options", options)
        if self.saved.value not in options:
            self._set(self.saved, "value", options[0])

    def _open_bytes(self, data):
        if not isinstance(data, bytes) or not 0 < len(data) <= MAX_ARCHIVE_BYTES:
            raise ValueError("Choose a native performance archive under 8 MB.")
        if self.session.snapshot().get("busy"):
            raise ValueError("Wait for generation to finish before opening another performance.")
        # Decoding is owned by the session; it validates before replacing its clip.
        if self.on_open is not None:
            self.on_open(data)
        else:
            self.session.load(data)
            self._activate()
        self._notice = "Opened native paired performance."

    def _bind(self):
        @self.view.on_click
        def view_clicked(_):
            self._run(self._activate)

        @self.frame_cast.on_click
        def frame_cast_clicked(event):
            def action():
                if self.on_frame_cast is None:
                    raise RuntimeError("Cast framing is not available in this viewer.")
                self.on_frame_cast(getattr(event, "client", None))
                self._notice = "Camera framed around the cast."
            self._run(action)

        @self.back.on_click
        def back_clicked(_):
            self._run(lambda: self.on_active(False))

        @self.actor.on_update
        def actor_changed(_):
            if self._syncing:
                return
            actor_id = self._selected_id(self.actor)
            if actor_id:
                name = dict(self._cast(self.session.snapshot())).get(actor_id, "")
                self._set(self.actor_name, "value", name)

        @self.add.on_click
        def add_clicked(_):
            def action():
                if len(self._cast(self.session.snapshot())) >= MAX_CAST:
                    raise ValueError("A cast can contain up to six characters.")
                self._preferred_actor_id = self.session.add_actor(self._validate_name(self.new_name.value))
                self._set(self.new_name, "value", "")
                self._notice = "Character added. Choose a pair to direct an interaction."
            self._run(action)

        @self.rename.on_click
        def rename_clicked(_):
            def action():
                actor_id = self._selected_id(self.actor)
                if actor_id is None:
                    raise ValueError("Choose a character to rename.")
                self.session.rename_actor(actor_id, self._validate_name(self.actor_name.value))
                self._notice = "Character renamed."
            self._run(action)

        @self.remove.on_click
        def remove_clicked(_):
            def action():
                actor_id = self._selected_id(self.actor)
                if actor_id is None:
                    raise ValueError("Choose a character to remove.")
                self.session.remove_actor(actor_id)
                self._notice = "Character removed."
            self._run(action)

        @self.choose_pair.on_click
        def pair_clicked(_):
            def action():
                self.session.select_pair(*self._pair_ids())
                self._notice = "Pair selected. Give them an interaction or play the reviewed handshake."
            self._run(action)

        @self.place.on_click
        def place_clicked(_):
            def action():
                self.session.set_placement(x=float(self.position_x.value), z=float(self.position_z.value),
                                           yaw_degrees=float(self.yaw.value))
                self._notice = "Pair placement updated."
            self._run(action)

        @self.reviewed.on_click
        def reviewed_clicked(_):
            def action():
                first, second = self._pair_ids()
                self._activate()
                self.session.select_pair(first, second)
                self.session.load_reviewed_handshake(scene_document=self._scene_document())
                self.session.play()
                self._set(self.prompt, "value", HANDSHAKE_PROMPT)
                self._set(self.seed, "value", str(HANDSHAKE_SEED))
                self._notice = "Loaded the reviewed 7-second handshake source."
            self._run(action)

        @self.sparring.on_click
        def sparring_clicked(_):
            def action():
                first, second = self._pair_ids()
                self._activate()
                self.session.select_pair(first, second)
                self.session.load_sparring_preview(scene_document=self._scene_document())
                self.session.play()
                self._notice = "Playing the curated non-contact sparring preview."
            self._run(action)

        @self.generate.on_click
        def generate_clicked(_):
            def action():
                prompt, seed = self._generation_args()
                first, second = self._pair_ids()
                self._activate()
                self.session.select_pair(first, second)
                self.session.generate(prompt, seed, frames=210, scene_document=self._scene_document())
                self._notice = "Generating a complete 7-second paired performance."
            self._run(action)

        @self.build_context.on_click
        def build_context_clicked(_):
            def action():
                if self.on_build_context is None:
                    raise RuntimeError("ARDY approach and exit are not available in this viewer.")
                self.on_build_context()
                self._notice = "Building ARDY approach and exit around the paired interaction."
            self._run(action)

        @self.cancel.on_click
        def cancel_clicked(_):
            self._run(self.session.cancel)

        @self.retry.on_click
        def retry_clicked(_):
            self._run(self.session.retry)

        @self.play.on_click
        def play_clicked(_):
            self._run(lambda: (self._activate(), self.session.play()))

        @self.pause.on_click
        def pause_clicked(_):
            self._run(self.session.pause)

        @self.restart.on_click
        def restart_clicked(_):
            self._run(self.session.restart)

        @self.frame.on_update
        def frame_changed(_):
            if not self._syncing:
                self._run(lambda: self.session.seek(int(self.frame.value)))

        @self.hand_pose.on_update
        def hand_pose_changed(_):
            if not self._syncing:
                self._run(lambda: self.session.set_hand_pose(HAND_POSES[self.hand_pose.value]))

        @self.capture.on_click
        def capture_clicked(event):
            self._run(lambda: self.start_capture(getattr(event, 'client', None)))

        @self.save.on_click
        def save_clicked(event):
            def action():
                data = self.session.save()
                if not isinstance(data, bytes) or not 0 < len(data) <= MAX_ARCHIVE_BYTES:
                    raise ValueError("Native performance exceeds the 8 MB archive limit.")
                self.folder.mkdir(parents=True, exist_ok=True)
                path = self.folder / f"native-pair-{time.time_ns()}{ARCHIVE_SUFFIX}"
                temporary = path.with_suffix(".tmp")
                try:
                    temporary.write_bytes(data)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                if getattr(event, "client", None) is not None:
                    event.client.send_file_download(path.name, data)
                self.refresh_saved(force=True)
                self._set(self.saved, "value", path.name)
                self._notice = "Saved native paired performance."
            self._run(action)

        @self.open.on_click
        def open_clicked(_):
            def action():
                path = self._saved_map.get(self.saved.value)
                root = self.folder.resolve()
                if (path is None or path.name != self.saved.value or path.is_symlink()
                        or path.resolve().parent != root or not path.is_file()):
                    raise ValueError("Choose a saved native performance from this folder.")
                if not 0 < path.stat().st_size <= MAX_ARCHIVE_BYTES:
                    raise ValueError("Native performance archive exceeds the 8 MB limit.")
                self._open_bytes(path.read_bytes())
            self._run(action)

        @self.upload.on_upload
        def upload_clicked(event):
            self._run(lambda: self._open_bytes(event.file.content))

    def tick(self):
        """Refresh from the existing Studio UI cadence without polling inference."""
        snapshot = self.session.snapshot()
        self.refresh_saved()
        cast = self._cast(snapshot)
        old_labels = self._cast_labels
        labels = {f"{name} · {actor_id}": actor_id for actor_id, name in cast}
        self._cast_labels = labels
        options = tuple(labels) or ("No characters",)
        pair = tuple(snapshot.get("selected_pair") or ())
        placement = snapshot.get("placement") or {}
        placement_key = (placement.get("x", 0.), placement.get("z", 0.), placement.get("yaw_degrees", 0.))
        inverse = {actor_id: label for label, actor_id in labels.items()}
        frames = int(snapshot.get("total_frames") or 0)
        current = int(snapshot.get("frame") or 0)
        active = bool(snapshot.get("active"))
        busy = bool(snapshot.get("busy"))
        available = bool(snapshot.get("available"))
        with self._capture_lock:
            capture_pending, capture_message = self._capture_pending, self._capture_message
        operating = busy or capture_pending or bool(snapshot.get("capturing"))
        self._syncing = True
        try:
            for handle in (self.actor, self.first, self.second):
                old_id = old_labels.get(handle.value)
                self._set(handle, "options", options)
                if handle.value not in options or (handle is self.actor and self._preferred_actor_id):
                    preferred = pair[0] if handle is self.first and len(pair) == 2 else (
                        pair[1] if handle is self.second and len(pair) == 2 else self._preferred_actor_id or old_id)
                    choice = inverse.get(preferred) or options[min(1, len(options)-1) if handle is self.second else 0]
                    self._set(handle, "value", choice)
            self._preferred_actor_id = None
            if pair != self._last_pair:
                if len(pair) == 2:
                    self._set(self.first, "value", inverse.get(pair[0], self.first.value))
                    self._set(self.second, "value", inverse.get(pair[1], self.second.value))
                self._last_pair = pair
            if placement_key != self._last_placement:
                self._set(self.position_x, "value", placement_key[0])
                self._set(self.position_z, "value", placement_key[1])
                self._set(self.yaw, "value", placement_key[2])
                self._last_placement = placement_key
            selected = self._selected_id(self.actor)
            if selected != self._last_selected_actor:
                self._set(self.actor_name, "value", dict(cast).get(selected, ""))
                self._last_selected_actor = selected
            self._set(self.add, "disabled", operating or len(cast) >= MAX_CAST)
            self._set(self.rename, "disabled", operating or not selected)
            self._set(self.remove, "disabled", operating or not selected or selected in pair or len(cast) <= 2)
            pair_ready = len(cast) >= 2 and not operating
            self._set(self.choose_pair, "disabled", not pair_ready)
            self._set(self.place, "disabled", operating)
            self._set(self.reviewed, "disabled", not pair_ready)
            self._set(self.sparring, "disabled", not pair_ready)
            self._set(self.generate, "disabled", not pair_ready or not available)
            self._set(self.build_context, "disabled", operating or frames <= 0 or
                      bool(snapshot.get("composed")) or self.on_build_context is None)
            self._set(self.cancel, "disabled", not busy or capture_pending)
            self._set(self.retry, "disabled", operating or snapshot.get("phase") != "generation_failed")
            self._set(self.view, "disabled", active or operating)
            self._set(self.frame_cast, "disabled", operating or self.on_frame_cast is None)
            self._set(self.back, "disabled", not active or operating)
            for handle in (self.play, self.pause, self.restart):
                self._set(handle, "disabled", operating or frames <= 0)
            self._set(self.frame, "max", max(frames - 1, 1))
            self._set(self.frame, "value", min(current, max(frames - 1, 1)))
            self._set(self.frame, "disabled", operating or frames <= 0)
            pose = snapshot.get("hand_pose", "relaxed")
            pose_label = next((label for label, value in HAND_POSES.items() if value == pose), "Relaxed")
            self._set(self.hand_pose, "value", pose_label)
            self._set(self.hand_pose, "disabled", operating or frames <= 0)
            self._set(self.capture, "disabled", operating or frames <= 0 or self.on_capture is None)
            self._set(self.save, "disabled", operating or frames <= 0)
            self._set(self.open, "disabled", operating or not self._saved_map)
            self._set(self.upload, "disabled", operating)
            status = str(snapshot.get("status") or "Native pair ready.")
            if frames:
                status += f" {current / 30:.1f}s / {frames / 30:.1f}s."
            if not busy and self._notice.startswith(("Building ARDY", "Generating a complete")):
                self._notice = ""
            if self._notice:
                status = self._notice + " " + status
            if capture_message:
                status = capture_message + " " + status
            self._set(self.status, "content", self._safe_text(status))
        finally:
            self._syncing = False

    def start_capture(self, client):
        if self.on_capture is None:
            raise RuntimeError('Video export is not available in this viewer.')
        with self._capture_lock:
            if self._capture_pending:
                raise RuntimeError('A video export is already running.')
            self._capture_pending = True
            self._capture_message = 'Exporting all native frames to MP4…'
        threading.Thread(target=self._capture_worker, args=(client,),
                         name='native-pair-video-export', daemon=True).start()

    def _capture_worker(self, client):
        try:
            result = self.on_capture(client)
            path = Path(result)
            trusted_root = (self.folder.parent / "native-pair-videos").resolve()
            try:
                relative = path.resolve().relative_to(trusted_root)
            except ValueError:
                raise ValueError("Video export returned an unexpected file location.") from None
            if len(relative.parts) != 2 or relative.name != "playback.mp4":
                raise ValueError("Video export returned an unexpected file location.")
            if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
                raise ValueError("Exported video is unavailable.")
            if not 8 <= path.stat().st_size <= MAX_VIDEO_BYTES:
                raise ValueError("Exported video is empty or exceeds 64 MB.")
            with path.open("rb") as stream:
                if stream.read(8)[4:8] != b"ftyp":
                    raise ValueError("Exported file is not an MP4 video.")
            if client is not None:
                client.send_file_download(path.name, path.read_bytes())
            message = f"Playback video exported: {path}"
        except Exception as exc:
            message = f"Video export failed: {str(exc)[:240]}"
        with self._capture_lock:
            self._capture_pending = False
            self._capture_message = message
