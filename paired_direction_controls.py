"""A single, scene-level direction form for the native two-person workflow.

The UI collects intent and stable scene references. The caller owns planning,
route checks, generation, and publication; changing a field never starts work.
"""

from __future__ import annotations

from html import escape
import json
import math
import re


SOURCES = {
    "Handshake": "handshake",
    "Sparring": "sparring",
    "Describe another interaction": "generate",
}
DEFAULT_PROMPTS = {
    "handshake": "They shake hands and step apart.",
    "sparring": "They perform a controlled, non-contact sparring exchange and step apart.",
    "generate": "They greet each other, celebrate together, then step apart.",
}
CUSTOM = "A point I choose"
MAX_PROMPT = 2000


def _safe_markdown(value):
    # Viser markdown is MDX; status may contain user/model text.
    escaped = re.sub(r"([\\`*_\[\]()#+.!|~-])", r"\\\1", str(value))
    return escape(escaped, quote=False).replace("{", "&#123;").replace("}", "&#125;")


def _number(value, name, bound=24.):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number between −{bound:g} and {bound:g} metres.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a number between −{bound:g} and {bound:g} metres.") from None
    if not math.isfinite(number) or abs(number) > bound:
        raise ValueError(f"{name} must be a number between −{bound:g} and {bound:g} metres.")
    return number


class PairedDirectionControls:
    """Collect one whole-scene request; callbacks supply the generation engine.

    on_generate(request, client) may return a job with snapshot/cancel/retry.
    on_preview(request, client) may return {summary, request}; its normalized
    request is applied to the form so the next Generate uses the shown marks.
    """

    def __init__(self, gui, session, *, on_generate, on_preview, on_frame_cast,
                 on_active, scene_provider=None, on_export=None, on_edit=None):
        self.session = session
        self.on_edit = on_edit or (lambda: None)
        self.on_generate = on_generate
        self.on_preview = on_preview
        self.on_frame_cast = on_frame_cast
        self.on_active = on_active
        self.on_export = on_export
        native_scene = session.scene_document
        self.scene_provider = (scene_provider if scene_provider is not None else
                               native_scene if callable(native_scene) else
                               lambda: session.scene_document)
        self._cast_labels = {}
        self._place_labels = {}
        self._scene_key = None
        self._notice = ""
        self._preview_summary = ""
        self._job = None
        self._syncing = False
        self._selected_pair = None
        self._previous_source = "handshake"
        self._custom_prompt = DEFAULT_PROMPTS["generate"]
        with gui.add_folder("Direct a scene with two people", expand_by_default=True):
            gui.add_markdown("Place each start and the meeting point. The scene follows those marks; choose their interaction below.")
            self.first = gui.add_dropdown("First person", ("Loading cast",))
            self.second = gui.add_dropdown("Second person", ("Loading cast",))
            self.prompt = gui.add_text("Interaction direction",
                                       initial_value=DEFAULT_PROMPTS["handshake"], multiline=True)
            self.source = gui.add_dropdown("When they meet", tuple(SOURCES), initial_value="Handshake")
            self.source_note = gui.add_markdown("")
            with gui.add_folder("Starting places", expand_by_default=True):
                self.first_x = gui.add_number("First person X · m", initial_value=-2., min=-24., max=24., step=.1)
                self.first_z = gui.add_number("First person Z · m", initial_value=-1., min=-24., max=24., step=.1)
                self.second_x = gui.add_number("Second person X · m", initial_value=2., min=-24., max=24., step=.1)
                self.second_z = gui.add_number("Second person Z · m", initial_value=-1., min=-24., max=24., step=.1)
            with gui.add_folder("Meeting place", expand_by_default=True):
                self.place = gui.add_dropdown("Meet near", (CUSTOM,), initial_value=CUSTOM)
                self.meet_x = gui.add_number("Meeting X · m", initial_value=0., min=-24., max=24., step=.1)
                self.meet_z = gui.add_number("Meeting Z · m", initial_value=0., min=-24., max=24., step=.1)
                self.meet_yaw = gui.add_number("Facing · degrees", initial_value=0., min=-180., max=180., step=5.)
            with gui.add_folder("Advanced", expand_by_default=False):
                self.seed = gui.add_text("Variation seed", initial_value="42")
                gui.add_markdown("Paired interaction generation uses a research model. Contact and landings still need review.")
            self.preview = gui.add_button("Preview places", color="gray")
            self.frame_cast = gui.add_button("Frame people", color="gray")
            self.generate = gui.add_button("Generate scene", color="green")
            self.cancel = gui.add_button("Cancel scene", color="gray")
            self.retry = gui.add_button("Retry scene", color="gray")
            with gui.add_folder("Watch the scene", expand_by_default=True):
                self.play = gui.add_button("Play scene", color="green")
                self.pause = gui.add_button("Pause", color="gray")
                self.export = gui.add_button("Export video", color="gray")
            self.status = gui.add_markdown("Choose two people and describe their scene.")
        self._bind()
        self.tick()

    @staticmethod
    def _set(handle, name, value):
        if getattr(handle, name, None) != value:
            setattr(handle, name, value)

    def _scene_places(self, scene):
        places = {}
        # Scene targets can be attachment points on props, while objects may
        # be solids. Their IDs are references only; the planner finds safe
        # nearby ground before generation.
        for item in scene.get("objects", ()):
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                name = str(item.get("name") or item["id"])
                places[f"Near {name} · {item['id']}"] = item["id"]
        for item in scene.get("targets", ()):
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                name = str(item.get("name") or item["id"])
                places[f"Near {name} · {item['id']}"] = item["id"]
        return places

    def _request(self):
        first = self._cast_labels.get(self.first.value)
        second = self._cast_labels.get(self.second.value)
        if first is None or second is None or first == second:
            raise ValueError("Choose two different people for this scene.")
        prompt = self.prompt.value.strip()
        if not 1 <= len(prompt) <= MAX_PROMPT:
            raise ValueError("Describe the scene in 1–2000 characters.")
        source = SOURCES.get(self.source.value)
        if source is None:
            raise ValueError("Choose what happens when they meet.")
        try:
            seed = int(str(self.seed.value).strip())
        except (TypeError, ValueError):
            raise ValueError("Variation seed must be an integer from 0 to 4294967295.") from None
        if not 0 <= seed < 2**32:
            raise ValueError("Variation seed must be an integer from 0 to 4294967295.")
        starts = [
            {"x": _number(self.first_x.value, "First person's X"),
             "z": _number(self.first_z.value, "First person's Z")},
            {"x": _number(self.second_x.value, "Second person's X"),
             "z": _number(self.second_z.value, "Second person's Z")},
        ]
        if math.dist((starts[0]["x"], starts[0]["z"]),
                     (starts[1]["x"], starts[1]["z"])) < 1.5:
            raise ValueError("Place the two starting marks at least 1.5 m apart.")
        yaw = _number(self.meet_yaw.value, "Facing angle", 180.)
        target_id = self._place_labels.get(self.place.value)
        if self.place.value != CUSTOM and target_id is None:
            raise ValueError("That meeting place changed. Choose a place again.")
        meeting = {"x": _number(self.meet_x.value, "Meeting X"),
                   "z": _number(self.meet_z.value, "Meeting Z"),
                   "yaw_degrees": yaw}
        return {"actor_ids": [first, second], "prompt": prompt, "source": source,
                "starts": starts, "meeting": meeting, "target_id": target_id,
                "seed": seed}

    def set_marks(self, request):
        """Receive reviewed 3D marks or a normalized preview without generating."""
        previous_syncing = self._syncing
        self._syncing = True
        try:
            starts, meeting = request["starts"], request["meeting"]
            if not isinstance(starts, (list, tuple)) or len(starts) != 2 or not isinstance(meeting, dict):
                raise ValueError("Expected two starts and one meeting place.")
            values = [
                _number(starts[0]["x"], "First person's X"),
                _number(starts[0]["z"], "First person's Z"),
                _number(starts[1]["x"], "Second person's X"),
                _number(starts[1]["z"], "Second person's Z"),
                _number(meeting["x"], "Meeting X"),
                _number(meeting["z"], "Meeting Z"),
                _number(meeting["yaw_degrees"], "Facing angle", 180.),
            ]
            for handle, value in zip((self.first_x, self.first_z, self.second_x,
                                      self.second_z, self.meet_x, self.meet_z,
                                      self.meet_yaw), values):
                self._set(handle, "value", value)
            label = next((label for label, identifier in self._place_labels.items()
                          if identifier == request.get("target_id")), CUSTOM)
            self._set(self.place, "value", label)
            self._preview_summary = ""
        finally:
            self._syncing = previous_syncing

    def restore_request(self, request):
        """Restore an archived direction without triggering generation."""
        from paired_direction import validate_request
        request = validate_request(request)
        self.tick()
        self._syncing = True
        try:
            self.set_marks(request)
            inverse = {value: label for label, value in self._cast_labels.items()}
            for handle, actor_id in zip((self.first, self.second), request['actor_ids']):
                if actor_id in inverse:
                    self._set(handle, 'value', inverse[actor_id])
            source_label = next(label for label, value in SOURCES.items() if value == request['source'])
            self._set(self.source, 'value', source_label)
            self._previous_source = request['source']
            if request['source'] == 'generate':
                self._custom_prompt = request['prompt']
            self._set(self.prompt, 'value', request['prompt'])
            self._set(self.seed, 'value', str(request['seed']))
            self._notice = 'Restored the saved scene direction.'
        finally:
            self._syncing = False
        self.tick()

    def _run(self, callback):
        self._notice = ""
        try:
            callback()
        except Exception as exc:
            self._notice = str(exc)[:240]
        self.tick()

    def _bind(self):
        @self.preview.on_click
        def preview_clicked(event):
            def action():
                request = self._request()
                result = self.on_preview(request, getattr(event, "client", None))
                if isinstance(result, dict):
                    if isinstance(result.get("request"), dict):
                        self.set_marks(result["request"])
                    self._preview_summary = str(result.get("summary") or "Places are ready for review.")[:300]
                else:
                    self._preview_summary = "Places are ready for review."
            self._run(action)

        @self.frame_cast.on_click
        def frame_clicked(event):
            self._run(lambda: self.on_frame_cast(getattr(event, "client", None)))

        @self.generate.on_click
        def generate_clicked(event):
            def action():
                request = self._request()
                self._job = self.on_generate(request, getattr(event, "client", None))
                self._notice = "Scene submitted."
            self._run(action)

        @self.cancel.on_click
        def cancel_clicked(_):
            def action():
                if self._job is not None and callable(getattr(self._job, "cancel", None)):
                    self._job.cancel()
                else:
                    self.session.cancel()
                self._notice = "Scene cancelled."
            self._run(action)

        @self.retry.on_click
        def retry_clicked(_):
            def action():
                if self._job is not None and callable(getattr(self._job, "retry", None)):
                    self._job.retry()
                else:
                    self.session.retry()
                self._notice = "Retrying scene."
            self._run(action)

        @self.play.on_click
        def play_clicked(_):
            self._run(lambda: (self.session.seek(0), self.session.play()))

        @self.pause.on_click
        def pause_clicked(_):
            self._run(self.session.pause)

        @self.export.on_click
        def export_clicked(event):
            def action():
                if self.on_export is None:
                    raise ValueError("Video export is unavailable in this viewer.")
                self.on_export(getattr(event, "client", None))
            self._run(action)

        @self.source.on_update
        def source_changed(_):
            if self._syncing:
                return
            current = SOURCES.get(self.source.value)
            if self._previous_source == "generate":
                self._custom_prompt = self.prompt.value
            if current == "generate":
                self._set(self.prompt, "value", self._custom_prompt)
            elif current in DEFAULT_PROMPTS:
                self._set(self.prompt, "value", DEFAULT_PROMPTS[current])
            self._previous_source = current
            self.on_edit()
            self._preview_summary = ""
            self.tick()

        for handle in (self.first, self.second, self.place, self.prompt,
                       self.first_x, self.first_z, self.second_x, self.second_z,
                       self.meet_x, self.meet_z, self.meet_yaw, self.seed):
            @handle.on_update
            def changed(_):
                if not self._syncing:
                    self.on_edit()
                    self._preview_summary = ""

    def tick(self):
        """Refresh cast and scene inventory at the host Studio UI cadence."""
        self._syncing = True
        try:
            state = self.session.snapshot()
            cast = tuple((actor["id"], str(actor.get("name") or actor["id"]))
                         for actor in state.get("cast", ()) if isinstance(actor, dict)
                         and isinstance(actor.get("id"), str))
            labels = {f"{name} · {identifier}": identifier for identifier, name in cast}
            old = self._cast_labels
            self._cast_labels = labels
            options = tuple(labels) or ("No people yet",)
            pair = tuple(state.get("selected_pair") or ())
            for handle, preferred in ((self.first, pair[0] if len(pair) == 2 else None),
                                      (self.second, pair[1] if len(pair) == 2 else None)):
                prior = old.get(handle.value)
                self._set(handle, "options", options)
                if pair != self._selected_pair or handle.value not in options:
                    identifier = (preferred if pair != self._selected_pair else prior)
                    if identifier not in labels.values():
                        identifier = preferred
                    choice = next((label for label, aid in labels.items() if aid == identifier), None)
                    self._set(handle, "value", choice or options[min(1, len(options)-1) if handle is self.second else 0])
            self._selected_pair = pair
            scene = self.scene_provider()
            key = json.dumps(scene, sort_keys=True, allow_nan=False)
            if key != self._scene_key:
                if self._scene_key is not None:
                    self.on_edit()
                places = self._scene_places(scene)
                selected = self._place_labels.get(self.place.value)
                self._place_labels = places
                place_options = (CUSTOM, *places)
                self._set(self.place, "options", place_options)
                label = next((label for label, aid in places.items() if aid == selected), CUSTOM)
                self._set(self.place, "value", label)
                self._scene_key = key
            job_state = None
            if self._job is not None and callable(getattr(self._job, "snapshot", None)):
                job_state = self._job.snapshot()
            if not isinstance(job_state, dict):
                job_state = state
            phase = str(job_state.get("phase") or job_state.get("status") or "ready")
            busy = bool(job_state.get("busy") or phase in ("planning", "routing", "queued", "running", "generating", "composing", "validating", "cancelling"))
            capturing = bool(job_state.get("capturing"))
            operating = busy or capturing
            frames = int(job_state.get("total_frames") or 0)
            failed = phase in ("failed", "generation_failed") or bool(job_state.get("error"))
            if not operating and frames and self._notice == "Scene submitted.":
                self._notice = ""
                self._preview_summary = ""
            self._set(self.generate, "disabled", operating or len(cast) < 2)
            self._set(self.preview, "disabled", operating or len(cast) < 2)
            self._set(self.cancel, "disabled", not busy)
            self._set(self.retry, "disabled", operating or not failed)
            self._set(self.frame_cast, "disabled", capturing)
            for handle in (self.play, self.pause):
                self._set(handle, "disabled", operating or frames <= 0)
            self._set(self.export, "disabled", operating or frames <= 0 or self.on_export is None)
            source = SOURCES.get(self.source.value)
            source_note = (
                "The handshake is a fixed reviewed interaction. Start and meeting marks control the approach."
                if source == "handshake" else
                "The sparring is a fixed reviewed, non-contact interaction. Start and meeting marks control the approach."
                if source == "sparring" else
                "Describe what they do at the meeting point. Start and meeting marks control the approach."
            )
            self._set(self.source_note, "content", source_note)
            self._set(self.prompt, "disabled", operating or source != "generate")
            status_text = str(job_state.get("status") or "")
            detail = self._notice or self._preview_summary or str(job_state.get("error") or "")
            if status_text and status_text != phase:
                detail = (detail + " · " if detail else "") + status_text
            progress = job_state.get("progress")
            if isinstance(progress, str) and progress:
                detail = progress
            if isinstance(progress, dict) and progress.get("total_steps"):
                detail += f" · {progress.get('completed_steps', 0)}/{progress['total_steps']} steps"
            message = f"**{_safe_markdown(phase.capitalize())}**"
            if detail:
                message += " · " + _safe_markdown(detail)
            self._set(self.status, "content", message)
        finally:
            self._syncing = False
