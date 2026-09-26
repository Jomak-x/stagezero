"""Small Viser controls for the optional Core scene director.

The ordinary G1 motion, takes, scene authoring, and project controls remain
owned by StudioUI.  This panel only operates on CoreStudioSession.
"""

from __future__ import annotations

from html import escape
import math
from pathlib import Path
import re
import threading
import time

from core_choreography import choreography_preset, validate_plan
from studio_interaction_scene import adapt_studio_scene, recommend_placements

MAX_STUDIO_ARCHIVE_BYTES = 64_000_000
MAX_PAIR_ARCHIVE_BYTES = 8_000_000
PAIR_EXAMPLE_PROMPT = ("Two people perform a choreographed martial arts exchange: sidestep dodge, "
                       "forearm block, controlled push, then step apart.")
PAIR_IDEAS = {
    "Close exchange": (PAIR_EXAMPLE_PROMPT, 42, "6 seconds"),
    "Partner dance": ("Two people hold hands and dance together, stepping sideways and turning around each other.",
                      7302, "6 seconds"),
}


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
TOGETHER_PRESETS = {
    "Feint and dodge": "feint_dodge",
    "Dance and answer": "dance_response",
    "Surprise and celebrate": "surprise_celebration",
    "Pose duet": "pose_duet",
}


class CoreStudioControls:
    """Build and refresh the optional native Core panel inside Motion."""

    def __init__(self, gui, core_session, studio_session, *, on_active, project_folder,
                 paired_session=None, on_paired_active=None):
        self.core = core_session
        self.paired = paired_session
        self.on_paired_active = on_paired_active
        self.studio = studio_session
        self.on_active = on_active
        self.folder = Path(project_folder)
        self.pair_folder = self.folder.parent / "paired-research"
        self._target_ids = {}
        self._target_actions = {}
        self._target_cache_key = None
        self._notice = ""
        self._syncing = False
        self._last_actor_count = None
        self._saved_map = {}
        self._saved_checked_at = 0.0
        self._pair_saved_map = {}
        self._pair_saved_checked_at = 0.0
        self._pair_notice = ""
        self._together_plan = None
        self._together_key = None
        self._ai_pending = False
        self._ai_result = None
        self._ai_request = 0
        self._ai_state = None
        self._ai_lock = threading.Lock()
        self._together_message = "Choose a preset or describe a shared scene for AI planning."
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
            with gui.add_folder("Together · experimental", expand_by_default=False):
                gui.add_markdown("Two actors required. A new performance places them at least 2.25 m apart. Shared beats coordinate timing and directions; physical interaction quality is still experimental.")
                self.together_start = gui.add_button("New two-person performance", color="gray")
                self.together_preset = gui.add_dropdown("Shared preset", tuple(TOGETHER_PRESETS),
                                                        initial_value="Pose duet")
                self.together_swap = gui.add_checkbox("Swap actor roles", initial_value=False)
                self.together_preview_preset = gui.add_button("Preview shared preset", color="gray")
                self.together_direction = gui.add_text("Shared direction for AI", initial_value="", multiline=True)
                self.together_plan_ai = gui.add_button("Plan shared scene with AI", color="gray")
                self.together_preview = gui.add_markdown("Choose a preset or describe a shared scene for AI planning.")
                self.together_generate = gui.add_button("Generate shared sequence")
                if self.paired is not None:
                    with gui.add_folder("Joint pair · InterGen research", expand_by_default=False):
                        gui.add_markdown("Experimental joint generation · InterGen research model (CC BY-NC-SA 4.0). The scene is a playback backdrop, not a generation constraint. Contact is not verified. Saved clips are separate from Native Core projects and G1 takes.")
                        self.pair_idea = gui.add_dropdown("Joint pair idea", tuple(PAIR_IDEAS), initial_value="Close exchange")
                        self.pair_example = gui.add_button("Use pair idea", color="gray")
                        self.pair_prompt = gui.add_text("Joint pair prompt", initial_value=PAIR_EXAMPLE_PROMPT, multiline=True)
                        self.pair_seed = gui.add_text("Pair seed", initial_value="42")
                        self.pair_length = gui.add_dropdown("Pair length", ("2 seconds", "4 seconds", "6 seconds"), initial_value="6 seconds")
                        self.pair_generate = gui.add_button("Generate joint pair · research")
                        self.pair_status = gui.add_markdown("No paired research clip yet.")
                        self.pair_view = gui.add_button("View paired research", color="gray")
                        self.pair_back = gui.add_button("Return to G1", color="gray")
                        self.pair_play = gui.add_button("Play paired clip", color="gray")
                        self.pair_pause = gui.add_button("Pause paired clip", color="gray")
                        self.pair_restart = gui.add_button("Restart paired clip", color="gray")
                        self.pair_cancel = gui.add_button("Cancel paired generation", color="gray")
                        with gui.add_folder("Paired research archives", expand_by_default=False):
                            self.pair_save = gui.add_button("Save paired research clip + download", color="gray")
                            self.pair_saved = gui.add_dropdown("Saved paired research clips", ("No paired research clips",))
                            self.pair_open = gui.add_button("Open paired research clip", color="gray")
                            self.pair_upload = gui.add_upload_button("Open paired research archive file", mime_type=".npz")
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

    @staticmethod
    def _face_each_other(placements):
        """Orient two existing anchors toward one another without moving them."""
        first, second = placements["actor_1"], placements["actor_2"]
        ax, az = first["position_xz"]
        bx, bz = second["position_xz"]
        return {"actor_1": {**first, "yaw": math.atan2(bx - ax, bz - az)},
                "actor_2": {**second, "yaw": math.atan2(ax - bx, az - bz)}}

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

    def refresh_pair_saved(self, *, force=False):
        if self.paired is None:
            return
        now = time.monotonic()
        if not force and now - self._pair_saved_checked_at < 1.0:
            return
        self._pair_saved_checked_at = now
        root = self.pair_folder.resolve()
        candidates = []
        try:
            for path in self.pair_folder.glob("*.paired.stagezero.npz"):
                try:
                    if path.is_symlink() or path.resolve().parent != root or not path.is_file():
                        continue
                    info = path.stat()
                    if 0 < info.st_size <= MAX_PAIR_ARCHIVE_BYTES:
                        candidates.append((info.st_mtime_ns, path.name, path))
                except OSError:
                    continue
        except OSError:
            candidates = []
        candidates.sort(reverse=True)
        self._pair_saved_map = {name: path for _, name, path in candidates[:100]}
        options = tuple(self._pair_saved_map) or ("No paired research clips",)
        self._set(self.pair_saved, "options", options)
        if self.pair_saved.value not in options:
            self._set(self.pair_saved, "value", options[0])

    def _open_pair_bytes(self, content):
        if not isinstance(content, bytes) or not 0 < len(content) <= MAX_PAIR_ARCHIVE_BYTES:
            raise ValueError("Choose a paired research archive under 8 MB.")
        if self.paired.snapshot().get("busy"):
            raise ValueError("Cancel paired generation before opening another clip.")
        self.paired.load(content)
        self.on_paired_active(True)
        self._pair_notice = "Opened paired InterGen research clip."

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

    def _run_pair(self, action):
        self._pair_notice = ""
        try:
            action()
        except Exception as exc:
            self._pair_notice = str(exc)[:240]
        self.tick()

    def _together_state(self, snapshot=None):
        snapshot = self._snapshot() if snapshot is None else snapshot
        return (bool(snapshot.get("active")), bool(snapshot.get("initialized")),
                tuple(snapshot.get("actor_ids") or ()), snapshot.get("epoch"),
                self.together_preset.value, bool(self.together_swap.value),
                self.together_direction.value.strip())

    @staticmethod
    def _swap_plan(plan, ids):
        result = {**plan, "beats": []}
        for beat in plan["beats"]:
            swapped = {**beat, "actor_prompts": {
                ids[0]: beat["actor_prompts"][ids[1]],
                ids[1]: beat["actor_prompts"][ids[0]]}}
            if "root_offsets" in beat:
                swapped["root_offsets"] = {
                    ids[0]: beat["root_offsets"][ids[1]],
                    ids[1]: beat["root_offsets"][ids[0]]}
            if "headings" in beat:
                swapped["headings"] = {
                    ids[0]: beat["headings"][ids[1]],
                    ids[1]: beat["headings"][ids[0]]}
            result["beats"].append(swapped)
        return validate_plan(result, ids)

    def _set_together_plan(self, plan, state):
        ids = state[2]
        normalized = validate_plan(plan, ids)
        self._together_plan = self._swap_plan(normalized, ids) if state[5] else normalized
        self._together_key = state
        self._together_message = "Candidate plan ready. Generate to try it in Core."

    @staticmethod
    def _mdx_text(value):
        """Keep model/user text literal in the studio client's MDX renderer."""
        safe = re.sub(r"([\\`*_\[\]()#+.!|~-])", r"\\\1", str(value))
        return escape(safe, quote=False).replace("{", "&#123;").replace("}", "&#125;")

    def _together_preview_text(self):
        plan = self._together_plan
        if plan is None:
            return self._mdx_text(self._together_message)
        rows = [f"**{self._mdx_text(plan['name'])} · {sum(b['seconds'] for b in plan['beats'])} seconds**",
                "Candidate directions. Motion has not been generated or visually verified."]
        for index, beat in enumerate(plan["beats"], 1):
            rows.append(f"**{index}. {self._mdx_text(beat['name'])} · {beat['seconds']} s**\n\n"
                        f"**Actor 1:** {self._mdx_text(beat['actor_prompts']['actor_1'])}\n\n"
                        f"**Actor 2:** {self._mdx_text(beat['actor_prompts']['actor_2'])}")
        return "\n\n".join(rows)

    def _start_ai_plan(self, intent, ids, state):
        with self._ai_lock:
            if self._ai_pending:
                raise ValueError("AI planning is still running. Wait for it to finish before retrying.")
            self._ai_request += 1
            request = self._ai_request
            self._ai_pending = True
            self._ai_state = state
            self._ai_result = None
        self._together_plan = None
        self._together_key = None
        self._together_message = "AI is drafting a shared plan…"

        def work():
            try:
                from core_choreography_ai import ChoreographyPlanner
                plan = ChoreographyPlanner.from_env().generate(
                    intent, ids, seed=int(time.time_ns() % (2**31)))
                result = (request, state, plan, None)
            except Exception as exc:
                result = (request, state, None, str(exc)[:240])
            with self._ai_lock:
                self._ai_pending = False
                self._ai_state = None
                if self._ai_request == request:
                    self._ai_result = result

        threading.Thread(target=work, name="core-choreography-plan", daemon=True).start()

    def _collect_ai_plan(self, state):
        with self._ai_lock:
            if self._ai_pending and self._ai_state is not None and self._ai_state != state:
                self._ai_request += 1
                self._ai_state = None
                self._ai_result = None
                self._together_message = "Scene or direction changed; the old AI plan will be ignored when planning finishes."
            result, self._ai_result = self._ai_result, None
        if result is None:
            return
        _, submitted_state, plan, error = result
        if submitted_state != state:
            self._together_message = "Scene or direction changed; the old AI plan was ignored."
            return
        if error:
            self._together_message = f"AI planning failed: {error}"
            return
        try:
            self._set_together_plan(plan, state)
        except Exception as exc:
            self._together_message = f"AI plan rejected: {str(exc)[:200]}"

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

        @self.together_preview_preset.on_click
        def preset_clicked(_):
            def action():
                state = self._together_state()
                if not state[0] or len(state[2]) != 2:
                    raise ValueError("Start a two-actor Core cast to preview shared choreography.")
                with self._ai_lock:
                    self._ai_request += 1
                    self._ai_state = None
                    self._ai_result = None
                name = TOGETHER_PRESETS[self.together_preset.value]
                self._set_together_plan(choreography_preset(name, state[2]), state)
            self._run(action)

        @self.together_start.on_click
        def start_together_clicked(_):
            def action():
                snapshot = self._snapshot()
                self._backup()
                was_active = bool(snapshot.get("active"))
                self.on_active(True)
                try:
                    points = sorted(recommend_placements(self._scene_document(), 2,
                                                          minimum_separation_m=2.25),
                                    key=lambda point: tuple(point["position_xz"]))
                    placements = {f"actor_{index + 1}": point
                                  for index, point in enumerate(points)}
                    if TOGETHER_PRESETS[self.together_preset.value] in ("feint_dodge", "pose_duet"):
                        placements = self._face_each_other(placements)
                    self.core.reset(actor_count=2, scene_document=self._scene_document(),
                                    placements=placements)
                except Exception:
                    self.on_active(was_active)
                    raise
            self._run(action)

        @self.together_plan_ai.on_click
        def ai_clicked(_):
            def action():
                state = self._together_state()
                if not state[0] or len(state[2]) != 2:
                    raise ValueError("Start a two-actor Core cast to plan shared choreography.")
                intent = state[6]
                if not 1 <= len(intent) <= 1600:
                    raise ValueError("Describe a shared scene in 1–1600 characters.")
                self._start_ai_plan(intent, state[2], state)
            self._run(action)

        @self.together_generate.on_click
        def together_generate_clicked(_):
            def action():
                state = self._together_state()
                if self._together_plan is None or self._together_key != state:
                    raise ValueError("Preview a current shared plan first.")
                if len(state[2]) != 2:
                    raise ValueError("Shared choreography requires two actors.")
                self._backup()
                self.core.choreograph(self._together_plan)
                self._together_plan = None
                self._together_key = None
                self._together_message = "Shared sequence submitted to Core. Review the actual motion."
            self._run(action)

        def together_input_changed(event):
            if event.client is not None and not self._syncing:
                self.tick()

        self.together_preset.on_update(together_input_changed)
        self.together_swap.on_update(together_input_changed)
        self.together_direction.on_update(together_input_changed)
        self.cast.on_update(together_input_changed)

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

        if self.paired is not None:
            @self.pair_example.on_click
            def pair_example_clicked(_):
                prompt, seed, length = PAIR_IDEAS[self.pair_idea.value]
                self._set(self.pair_prompt, "value", prompt)
                self._set(self.pair_seed, "value", str(seed))
                self._set(self.pair_length, "value", length)

            @self.pair_generate.on_click
            def pair_generate_clicked(_):
                def action():
                    prompt = self.pair_prompt.value.strip()
                    if not 1 <= len(prompt) <= 500:
                        raise ValueError("Describe a joint pair in 1–500 characters.")
                    try:
                        seed = int(self.pair_seed.value)
                    except (TypeError, ValueError):
                        raise ValueError("Pair seed must be an integer from 0 to 4294967295.") from None
                    if not 0 <= seed < 2**32:
                        raise ValueError("Pair seed must be an integer from 0 to 4294967295.")
                    frames = {"2 seconds": 40, "4 seconds": 80, "6 seconds": 120}[self.pair_length.value]
                    self.on_paired_active(True)
                    self.paired.generate(prompt, seed, frames=frames,
                                         scene_document=self._scene_document())
                    self._pair_notice = "Joint InterGen research job submitted. Existing motion remains saved."
                self._run_pair(action)

            @self.pair_view.on_click
            def pair_view_clicked(_):
                self._run_pair(lambda: self.on_paired_active(True))

            @self.pair_back.on_click
            def pair_back_clicked(_):
                self._run_pair(lambda: self.on_paired_active(False))

            @self.pair_play.on_click
            def pair_play_clicked(_):
                self._run_pair(self.paired.play)

            @self.pair_pause.on_click
            def pair_pause_clicked(_):
                self._run_pair(self.paired.pause)

            @self.pair_restart.on_click
            def pair_restart_clicked(_):
                self._run_pair(self.paired.restart)

            @self.pair_cancel.on_click
            def pair_cancel_clicked(_):
                self._run_pair(self.paired.cancel)

            @self.pair_save.on_click
            def pair_save_clicked(event):
                def action():
                    data = self.paired.save()
                    self.pair_folder.mkdir(parents=True, exist_ok=True)
                    path = self.pair_folder / f"paired-{time.time_ns()}.paired.stagezero.npz"
                    temporary = path.with_suffix(".tmp")
                    try:
                        temporary.write_bytes(data)
                        temporary.replace(path)
                    finally:
                        temporary.unlink(missing_ok=True)
                    if event.client is not None:
                        event.client.send_file_download(path.name, data)
                    self.refresh_pair_saved(force=True)
                    self._set(self.pair_saved, "value", path.name)
                    self._pair_notice = "Saved separate InterGen research archive."
                self._run_pair(action)

            @self.pair_open.on_click
            def pair_open_clicked(_):
                def action():
                    path = self._pair_saved_map.get(self.pair_saved.value)
                    root = self.pair_folder.resolve()
                    if (path is None or path.name != self.pair_saved.value or path.is_symlink()
                            or path.resolve().parent != root or not path.is_file()):
                        raise ValueError("Choose a saved paired research archive from this folder.")
                    if not 0 < path.stat().st_size <= MAX_PAIR_ARCHIVE_BYTES:
                        raise ValueError("Paired research archive exceeds the 8 MB limit.")
                    self._open_pair_bytes(path.read_bytes())
                self._run_pair(action)

            @self.pair_upload.on_upload
            def pair_upload_clicked(event):
                self._run_pair(lambda: self._open_pair_bytes(event.file.content))

    def tick(self):
        """Refresh at the studio's existing UI cadence without polling inference."""
        snapshot = self._snapshot()
        self.refresh_saved()
        self.refresh_pair_saved()
        active = bool(snapshot.get("active"))
        available = bool(snapshot.get("available"))
        ids = tuple(snapshot.get("actor_ids") or ())
        count = len(ids)
        frames = int(snapshot.get("total_frames") or 0)
        current = int(snapshot.get("frame") or 0)
        phase = snapshot.get("phase")
        together_state = self._together_state(snapshot)
        if self._together_key is not None and self._together_key != together_state:
            self._together_plan = None
            self._together_key = None
            self._together_message = "Scene or direction changed. Preview a shared plan again."
        self._collect_ai_plan(together_state)
        with self._ai_lock:
            ai_pending = self._ai_pending
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
            together_ready = active and available and len(ids) == 2
            self._set(self.together_start, "disabled", False)
            self._set(self.together_preview_preset, "disabled", not together_ready)
            self._set(self.together_plan_ai, "disabled", not together_ready or ai_pending)
            self._set(self.together_generate, "disabled", not together_ready or
                      self._together_plan is None or self._together_key != together_state)
            self._set(self.together_preview, "content", self._together_preview_text())
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
            if self.paired is not None:
                paired = self.paired.snapshot()
                pair_frames = int(paired.get("total_frames") or 0)
                pair_active = bool(paired.get("active"))
                pair_busy = bool(paired.get("busy"))
                self._set(self.pair_generate, "disabled", pair_busy or not paired.get("available"))
                self._set(self.pair_view, "disabled", pair_busy or not pair_frames or pair_active)
                self._set(self.pair_back, "disabled", pair_busy or not pair_active)
                self._set(self.pair_play, "disabled", not pair_active or not pair_frames)
                self._set(self.pair_pause, "disabled", not pair_active or not pair_frames)
                self._set(self.pair_restart, "disabled", not pair_active or not pair_frames)
                self._set(self.pair_cancel, "disabled", not pair_busy)
                self._set(self.pair_save, "disabled", pair_busy or not pair_frames)
                self._set(self.pair_open, "disabled", pair_busy or not self._pair_saved_map)
                pair_status = str(paired.get("status") or "No paired research clip yet.")
                if self._pair_notice:
                    pair_status = self._pair_notice + " " + pair_status
                self._set(self.pair_status, "content", self._mdx_text(pair_status))
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
