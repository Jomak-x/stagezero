"""Small Viser controls for the optional Core scene director.

The ordinary G1 motion, takes, scene authoring, and project controls remain
owned by StudioUI.  This panel only operates on CoreStudioSession.
"""

from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path
import time

from studio_interaction_scene import adapt_studio_scene, recommend_placements

MAX_STUDIO_ARCHIVE_BYTES = 64_000_000
CITY_EXAMPLES = {
    "Downtown boulevard": Path(__file__).parent / "examples/scenes/city-boulevard.json",
    "Generated city": Path(__file__).parent / "review/demo/fresh-city.json",
}
CITY_LAYOUTS = ("Downtown boulevard", "Generated city", "Current Studio scene")
CITY_ROUTES = ("direct", "west", "east")
CITY_TIMINGS = ("measured", "brisk")
CITY_SEEDS = ("33", "42", "103")
CITY_STARTS = ("opposite ends", "swapped ends", "turn into route")


EXAMPLES = {
    "Dance": ("A person dances with clear, rhythmic whole-body movement.",
              "A second person dances nearby with their own rhythm."),
    "Martial arts": ("A person performs a solo martial arts form with controlled steps and turns.",
                     "A second person performs a separate martial arts form nearby."),
    "Celebration": ("A person celebrates with joyful gestures and light steps.",
                    "A second person celebrates nearby with joyful gestures."),
    "Conversation": ("A person speaks with natural standing gestures and attentive posture.",
                     "A second person listens and replies with natural standing gestures."),
    "Approach": ("A person walks forward at a comfortable pace.",
                 "A second person walks forward at a comfortable pace."),
}
DURATIONS = {"2 seconds": 2, "6 seconds": 6, "12 seconds": 12}
ACTOR_COUNTS = {"One actor": 1, "Two actors": 2}


class CoreStudioControls:
    """Build and refresh the optional native Core panel inside Motion."""

    def __init__(self, gui, core_session, studio_session, *, on_active, project_folder):
        self.core = core_session
        self.studio = studio_session
        self.on_active = on_active
        self.folder = Path(project_folder)
        self._target_ids = {}
        self._target_actions = {}
        self._target_cache_key = None
        self._notice = ""
        self._syncing = False
        self._last_actor_count = None
        self._saved_map = {}
        self._saved_checked_at = 0.0
        with gui.add_folder("Scene direction · Core", expand_by_default=False):
            self.enabled = gui.add_checkbox("Use Core scene direction", initial_value=False)
            self.status = gui.add_markdown("Core scene direction is off.")
            self.cast = gui.add_dropdown("Cast", tuple(ACTOR_COUNTS), initial_value="One actor")
            self.start = gui.add_button("Start / change cast", color="gray")
            self.example = gui.add_dropdown("Direction idea", tuple(EXAMPLES))
            self.use_example = gui.add_button("Use idea", color="gray")
            self.actor_one = gui.add_text("Actor 1 direction", initial_value=EXAMPLES["Dance"][0], multiline=True)
            self.actor_two = gui.add_text("Actor 2 direction", initial_value=EXAMPLES["Dance"][1], multiline=True)
            self.duration = gui.add_dropdown("Length", tuple(DURATIONS), initial_value="6 seconds")
            self.generate = gui.add_button("Generate / redirect")
            with gui.add_folder("City encounter · staged/no-contact fight", expand_by_default=False):
                self.city_layout = gui.add_dropdown("City layout", CITY_LAYOUTS)
                self.city_starts = gui.add_dropdown("City starts", CITY_STARTS)
                self.city_route = gui.add_dropdown("City route", CITY_ROUTES)
                self.city_timing = gui.add_dropdown("City timing", CITY_TIMINGS)
                self.city_seed = gui.add_dropdown("City seed", CITY_SEEDS)
                self.city_generate = gui.add_button("Generate city encounter")
                gui.add_markdown("Two native Core actors share a route and event clock. The fight is staged without controlled contact; review the generated motion before use.")
            self.play = gui.add_button("Play", color="gray")
            self.pause = gui.add_button("Pause", color="gray")
            self.restart = gui.add_button("Restart", color="gray")
            self.frame = gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
            with gui.add_folder("Move to a scene object", expand_by_default=False):
                self.nav_actor = gui.add_dropdown("Actor", ("Actor 1",), initial_value="Actor 1")
                self.target = gui.add_dropdown("Object", ("No scene objects",), initial_value="No scene objects")
                self.verb = gui.add_dropdown("Action", ("approach", "go_through"), initial_value="approach")
                self.navigate = gui.add_button("Navigate to object")
            self.retry = gui.add_button("Retry failed generation", color="gray")
            self.cancel = gui.add_button("Cancel pending motion", color="gray")
            with gui.add_folder("Core projects", expand_by_default=False):
                self.save = gui.add_button("Save exact Core project + download", color="gray")
                self.saved = gui.add_dropdown("Saved Core projects", ("No saved Core projects",))
                self.open_saved = gui.add_button("Open saved Core project", color="gray")
                self.upload = gui.add_upload_button("Open Core project file", mime_type=".npz")
                self.example_motion = gui.add_button("Play included motion example", color="gray")
            gui.add_markdown("The timeline shows Core motion while this mode is on; your G1 takes stay saved.")
        self._bind()
        self.tick()

    @staticmethod
    def _set(handle, property_name, value):
        if getattr(handle, property_name, None) != value:
            setattr(handle, property_name, value)

    def _snapshot(self):
        return self.core.snapshot()

    def _scene_document(self):
        return self.studio.scene_document()

    def _scene_targets(self, snapshot):
        initialized = bool(snapshot.get("initialized"))
        key = (initialized, snapshot.get("epoch") if initialized
               else getattr(self.studio, "project_revision", None))
        if key == self._target_cache_key:
            return self._target_ids
        try:
            current = getattr(self.core, "scene_document", None)
            document = (current if initialized and isinstance(current, dict)
                        else self._scene_document())
            objects = adapt_studio_scene(document)["objects"]
        except Exception:
            objects = []
        labels, actions = {}, {}
        for obj in objects:
            identifier = obj.get("id")
            name = obj.get("name")
            if isinstance(identifier, str) and isinstance(name, str):
                label = f"{name} · {identifier}"
                labels[label] = identifier
                actions[label] = tuple(obj.get("actions", ()))
        self._target_actions = actions
        self._target_ids = labels
        self._target_cache_key = key
        return labels

    def _placements(self, count):
        points = recommend_placements(self._scene_document(), count)
        return {f"actor_{index + 1}": point for index, point in enumerate(points)}

    def _backup(self):
        if self._snapshot().get("total_frames", 0) <= 0:
            return None
        data = self.core.save()
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"before-core-change-{time.time_ns()}.core.stagezero.npz"
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_bytes(data)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        self.refresh_saved(force=True)
        return path

    def refresh_saved(self, *, force=False):
        """Publish a bounded local catalog without accepting arbitrary paths."""
        now = time.monotonic()
        if not force and now - self._saved_checked_at < 1.0:
            return
        self._saved_checked_at = now
        root = self.folder.resolve()
        candidates = []
        try:
            paths = self.folder.glob("*.core.stagezero.npz")
            for path in paths:
                try:
                    if path.is_symlink() or path.resolve().parent != root or not path.is_file():
                        continue
                    info = path.stat()
                    if 0 < info.st_size <= MAX_STUDIO_ARCHIVE_BYTES:
                        candidates.append((info.st_mtime_ns, path.name, path))
                except OSError:
                    continue
        except OSError:
            candidates = []
        candidates.sort(reverse=True)
        self._saved_map = {name: path for _, name, path in candidates[:100]}
        options = tuple(self._saved_map) or ("No saved Core projects",)
        self._set(self.saved, "options", options)
        if self.saved.value not in options:
            self._set(self.saved, "value", options[0])

    def _open_bytes(self, content):
        if not isinstance(content, bytes) or len(content) > MAX_STUDIO_ARCHIVE_BYTES:
            raise ValueError("Choose an exact Core project under 64 MB.")
        self._backup()
        was_active = bool(self._snapshot().get("active"))
        self.on_active(True)
        try:
            self.core.load(content)
        except Exception:
            self.on_active(was_active)
            raise
        self._notice = "Opened exact Core motion."

    def _activate(self):
        was_active = bool(self._snapshot().get("active"))
        self.on_active(True)
        try:
            if self.core.snapshot().get("initialized"):
                self.core.activate()
            else:
                count = ACTOR_COUNTS[self.cast.value]
                self.core.start(actor_count=count, scene_document=self._scene_document(),
                                placements=self._placements(count))
        except Exception:
            self.on_active(was_active)
            raise

    def _run(self, action):
        self._notice = ""
        try:
            action()
        except Exception as exc:
            self._notice = str(exc)[:240]
        self.tick()

    def _bind(self):
        @self.enabled.on_update
        def enabled_changed(event):
            if event.client is None or self._syncing:
                return
            if self.enabled.value:
                def turn_on():
                    self._activate()
                self._run(turn_on)
            else:
                def turn_off():
                    self.core.deactivate()
                    self.on_active(False)
                self._run(turn_off)

        @self.start.on_click
        def change_cast(_):
            def action():
                self._backup()
                was_active = bool(self._snapshot().get("active"))
                self.on_active(True)
                try:
                    count = ACTOR_COUNTS[self.cast.value]
                    self.core.reset(actor_count=count, scene_document=self._scene_document(),
                                    placements=self._placements(count))
                except Exception:
                    self.on_active(was_active)
                    raise
            self._run(action)

        @self.use_example.on_click
        def example_clicked(_):
            prompts = EXAMPLES[self.example.value]
            self._set(self.actor_one, "value", prompts[0])
            self._set(self.actor_two, "value", prompts[1])

        @self.generate.on_click
        def generate_clicked(_):
            def action():
                ids = self._snapshot().get("actor_ids", ())
                prompts = {"actor_1": self.actor_one.value.strip()}
                if "actor_2" in ids:
                    prompts["actor_2"] = self.actor_two.value.strip()
                self.core.direct(prompts, DURATIONS[self.duration.value])
            self._run(action)

        @self.city_generate.on_click
        def city_generate_clicked(_):
            def action():
                scene = (self._scene_document() if self.city_layout.value == "Current Studio scene"
                         else json.loads(CITY_EXAMPLES[self.city_layout.value].read_text()))
                placements = None
                if self.city_starts.value != "opposite ends":
                    from core_city_encounter import build_city_encounter
                    placements = build_city_encounter(
                        scene, route_variant=self.city_route.value,
                        timing_variant=self.city_timing.value,
                        seed=int(self.city_seed.value))[0]
                    placements = {actor: dict(item) for actor, item in placements.items()}
                    if self.city_starts.value == "swapped ends":
                        placements = {"actor_1": placements["actor_2"],
                                      "actor_2": placements["actor_1"]}
                    else:
                        for item in placements.values():
                            item["yaw"] = ((item["yaw"] + 2 * math.pi) % (2 * math.pi)) - math.pi
                self._backup()
                was_active = bool(self._snapshot().get("active"))
                self.on_active(True)
                try:
                    self.core.run_city_encounter(
                        scene, placements=placements, route_variant=self.city_route.value,
                        timing_variant=self.city_timing.value,
                        seed=int(self.city_seed.value))
                except Exception:
                    self.on_active(was_active)
                    raise
                self._notice = "City encounter queued: staged/no-contact fight. Planned motion is not a verified result."
            self._run(action)

        @self.play.on_click
        def play_clicked(_):
            self._run(self.core.play)

        @self.pause.on_click
        def pause_clicked(_):
            self._run(self.core.pause)

        @self.restart.on_click
        def restart_clicked(_):
            self._run(lambda: (self.core.seek(0), self.core.play()))

        @self.frame.on_update
        def frame_changed(event):
            if event.client is not None and not self._syncing:
                self._run(lambda: self.core.seek(int(self.frame.value)))

        @self.navigate.on_click
        def navigate_clicked(_):
            def action():
                actor_id = "actor_1" if self.nav_actor.value == "Actor 1" else "actor_2"
                target_id = self._target_ids.get(self.target.value)
                if target_id is None:
                    raise ValueError("Choose a scene object first.")
                if self.verb.value not in self._target_actions.get(self.target.value, ()):
                    raise ValueError("That action is unavailable for this object.")
                self.core.navigate(actor_id, target_id, self.verb.value)
            self._run(action)

        @self.target.on_update
        def target_changed(event):
            if event.client is not None and not self._syncing:
                self.tick()

        @self.retry.on_click
        def retry_clicked(_):
            self._run(self.core.retry)

        @self.cancel.on_click
        def cancel_clicked(_):
            self._run(self.core.cancel)

        @self.save.on_click
        def save_clicked(event):
            def action():
                data = self.core.save()
                self.folder.mkdir(parents=True, exist_ok=True)
                path = self.folder / f"core-{time.time_ns()}.core.stagezero.npz"
                temporary = path.with_suffix(".tmp")
                try:
                    temporary.write_bytes(data)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                if event.client is not None:
                    event.client.send_file_download(path.name, data)
                self.refresh_saved(force=True)
                self._set(self.saved, "value", path.name)
                self._notice = "Saved exact Core motion."
            self._run(action)

        @self.open_saved.on_click
        def open_saved_clicked(_):
            def action():
                path = self._saved_map.get(self.saved.value)
                root = self.folder.resolve()
                if (path is None or path.name != self.saved.value or path.is_symlink()
                        or path.resolve().parent != root or not path.is_file()):
                    raise ValueError("Select a saved Core project from this folder.")
                size = path.stat().st_size
                if not 0 < size <= MAX_STUDIO_ARCHIVE_BYTES:
                    raise ValueError("Saved Core project exceeds the 64 MB limit.")
                self._open_bytes(path.read_bytes())
            self._run(action)

        @self.upload.on_upload
        def upload_project(event):
            self._run(lambda: self._open_bytes(event.file.content))

        @self.example_motion.on_click
        def example_motion_clicked(_):
            def action():
                self._backup()
                was_active = bool(self._snapshot().get("active"))
                self.on_active(True)
                try:
                    self.core.load_example()
                except Exception:
                    self.on_active(was_active)
                    raise
            self._run(action)

    def tick(self):
        """Refresh at the studio's existing UI cadence without polling inference."""
        snapshot = self._snapshot()
        self.refresh_saved()
        active = bool(snapshot.get("active"))
        available = bool(snapshot.get("available"))
        ids = tuple(snapshot.get("actor_ids") or ())
        count = len(ids)
        frames = int(snapshot.get("total_frames") or 0)
        current = int(snapshot.get("frame") or 0)
        phase = snapshot.get("phase")
        self._syncing = True
        try:
            self._set(self.enabled, "value", active)
            if count != self._last_actor_count:
                self._set(self.cast, "value", "Two actors" if count == 2 else "One actor")
                self._last_actor_count = count
            self._set(self.actor_two, "visible", count == 2)
            self._set(self.nav_actor, "options", ("Actor 1", "Actor 2") if count == 2 else ("Actor 1",))
            if self.nav_actor.value not in self.nav_actor.options:
                self._set(self.nav_actor, "value", "Actor 1")
            targets = self._scene_targets(snapshot)
            self._set(self.target, "options", tuple(targets) or ("No scene objects",))
            if self.target.value not in self.target.options:
                self._set(self.target, "value", self.target.options[0])
            verbs = self._target_actions.get(self.target.value, ())
            self._set(self.verb, "options", verbs or ("approach",))
            if self.verb.value not in self.verb.options:
                self._set(self.verb, "value", self.verb.options[0])
            self._set(self.frame, "max", max(frames - 1, 1))
            self._set(self.frame, "value", min(current, max(frames - 1, 1)))
            self._set(self.start, "disabled", not active)
            self._set(self.generate, "disabled", not active or not available or not ids)
            self._set(self.city_generate, "disabled", not available or not all(
                path.is_file() for path in CITY_EXAMPLES.values()))
            self._set(self.navigate, "disabled", not active or not available or not targets or not verbs or not ids)
            self._set(self.play, "disabled", not active or frames == 0)
            self._set(self.pause, "disabled", not active or frames == 0)
            self._set(self.restart, "disabled", not active or frames == 0)
            self._set(self.frame, "disabled", not active or frames == 0)
            self._set(self.retry, "disabled", not active or not available or phase != "generation_failed")
            self._set(self.cancel, "disabled", not active or phase not in ("generating", "queued", "buffering"))
            self._set(self.save, "disabled", frames == 0)
            self._set(self.open_saved, "disabled", not bool(self._saved_map))
            self._set(self.example_motion, "visible", bool(snapshot.get("example_available", False)))
            status = str(snapshot.get("status") or "Ready.")
            if not available:
                status = ("Core service unavailable. Saved Core motion and the included example can still play."
                          if active else "Core service unavailable. Open a saved Core project to play it.")
            if active and frames:
                status += f" {current / 20:.1f}s / {frames / 20:.1f}s."
            if self._notice:
                status = self._notice + " " + status
            self._set(self.status, "content", escape(status))
        finally:
            self._syncing = False
