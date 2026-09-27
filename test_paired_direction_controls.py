"""Contract checks for the two-person scene form; no model or viewer needed."""

from types import SimpleNamespace
import unittest

from paired_direction_controls import PairedDirectionControls, CUSTOM


class Handle:
    def __init__(self, **fields):
        self.callbacks = {}
        self.disabled = False
        self.content = ""
        for name, value in fields.items():
            setattr(self, name, value)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def on_click(self, callback):
        self.callbacks["click"] = callback

    def on_update(self, callback):
        self.callbacks["update"] = callback

    def click(self, client=None):
        self.callbacks["click"](SimpleNamespace(client=client))

    def edit(self, value):
        self.value = value
        if "update" in self.callbacks:
            self.callbacks["update"](SimpleNamespace(client=None))


class Gui:
    def __init__(self):
        self.handles = {}

    def _new(self, label, **fields):
        result = Handle(**fields)
        self.handles[label] = result
        return result

    def add_folder(self, label, **_):
        return self._new(label)

    def add_markdown(self, content):
        return self._new("Markdown", content=content)

    def add_dropdown(self, label, options, initial_value=None):
        return self._new(label, options=tuple(options), value=initial_value or tuple(options)[0])

    def add_text(self, label, initial_value="", **_):
        return self._new(label, value=initial_value)

    def add_number(self, label, initial_value=0., **_):
        return self._new(label, value=initial_value)

    def add_button(self, label, **_):
        return self._new(label)


class Session:
    def __init__(self):
        self.cast = [{"id": "actor_1", "name": "Ava"}, {"id": "actor_2", "name": "Ben"}]
        self.pair = ["actor_1", "actor_2"]
        self.scene = {"objects": [{"id": "plaza", "name": "Plaza"}], "targets": []}
        self.phase = "ready"
        self.busy = False
        self.cancelled = False
        self.retried = False
        self.frames = 0
        self.capturing = False
        self.calls = []

    def snapshot(self):
        return {"cast": self.cast, "selected_pair": self.pair,
                "phase": self.phase, "busy": self.busy,
                "total_frames": self.frames, "capturing": self.capturing,
                "status": "Scene ready to play."}

    def scene_document(self):
        return self.scene

    def cancel(self):
        self.cancelled = True

    def retry(self):
        self.retried = True

    def seek(self, frame):
        self.calls.append(("seek", frame))

    def play(self):
        self.calls.append(("play",))

    def pause(self):
        self.calls.append(("pause",))


class Job:
    def __init__(self):
        self.state = {"status": "planning", "progress": {"completed_steps": 1, "total_steps": 4}}
        self.cancelled = False
        self.retried = False

    def snapshot(self):
        return self.state

    def cancel(self):
        self.cancelled = True
        self.state = {"status": "cancelled"}

    def retry(self):
        self.retried = True


class PairedDirectionControlsTests(unittest.TestCase):
    def setUp(self):
        self.session = Session()
        self.gui = Gui()
        self.generated = []
        self.previewed = []
        self.framed = []
        self.active = []
        self.job = Job()

        def generate(request, client):
            self.generated.append((request, client))
            return self.job

        def preview(request, client):
            self.previewed.append((request, client))
            return {"summary": "Routes checked; both can reach the plaza.",
                    "request": dict(request, meeting={"x": 1., "z": 2., "yaw_degrees": 45.})}

        self.controls = PairedDirectionControls(self.gui, self.session,
            on_generate=generate, on_preview=preview,
            on_frame_cast=self.framed.append, on_active=self.active.append)

    def test_city_scene_is_one_validated_request(self):
        ui = self.controls
        self.assertIn("Near Plaza · plaza", ui.place.options)
        ui.place.edit("Near Plaza · plaza")
        ui.source.edit("Handshake")
        ui.generate.click(client="browser")
        self.assertEqual(len(self.generated), 1)
        request, client = self.generated[0]
        self.assertEqual(client, "browser")
        self.assertEqual(request["actor_ids"], ["actor_1", "actor_2"])
        self.assertEqual(request["starts"], [{"x": -2., "z": -1.}, {"x": 2., "z": -1.}])
        self.assertEqual(request["target_id"], "plaza")
        self.assertEqual(request["source"], "handshake")
        self.assertEqual(request["prompt"], "They shake hands and step apart.")
        self.assertEqual(request["seed"], 42)
        self.assertIn("1/4 steps", ui.status.content)

    def test_preview_normalizes_marks_without_generating(self):
        ui = self.controls
        ui.preview.click(client="viewer")
        self.assertEqual(len(self.previewed), 1)
        self.assertEqual(self.generated, [])
        self.assertEqual(ui.meet_x.value, 1.)
        self.assertEqual(ui.meet_z.value, 2.)
        self.assertEqual(ui.meet_yaw.value, 45.)
        self.assertIn("Routes checked", ui.status.content)
        ui.set_marks({"starts": [{"x": -4, "z": -1}, {"x": 3, "z": -2}],
                      "meeting": {"x": 0, "z": 0, "yaw_degrees": 0}})
        self.assertEqual(ui.first_x.value, -4.)
        self.assertEqual(ui.place.value, CUSTOM)
        self.assertEqual(self.generated, [])
        ui.place.value = "Near Plaza · plaza"
        ui.set_marks({"starts": [{"x": -4, "z": -1}, {"x": 3, "z": -2}],
                      "meeting": {"x": 0, "z": 0, "yaw_degrees": 0}})
        self.assertEqual(ui.place.value, CUSTOM)

    def test_duplicate_actor_and_invalid_marks_block_callbacks(self):
        ui = self.controls
        ui.second.value = ui.first.value
        ui.generate.click()
        self.assertEqual(self.generated, [])
        self.assertIn("two different", ui.status.content)
        ui.second.value = next(label for label in ui.second.options if "actor_2" in label)
        ui.second_x.value = -1.9
        ui.generate.click()
        self.assertIn("1\\.5 m apart", ui.status.content)
        ui.second_x.value = 2.
        ui.meet_x.value = float("nan")
        ui.generate.click()
        self.assertIn("Meeting X", ui.status.content)
        self.assertEqual(self.generated, [])

    def test_cast_and_scene_inventory_refresh_preserves_ids(self):
        ui = self.controls
        ui.first.value = next(label for label in ui.first.options if "actor_2" in label)
        ui.second.value = next(label for label in ui.second.options if "actor_1" in label)
        ui.place.value = "Near Plaza · plaza"
        self.session.cast[0]["name"] = "Ava Prime"
        self.session.scene["objects"][0]["name"] = "Central Plaza"
        ui.tick()
        self.assertEqual(ui._cast_labels[ui.first.value], "actor_2")
        self.assertEqual(ui._cast_labels[ui.second.value], "actor_1")
        self.assertEqual(ui._place_labels[ui.place.value], "plaza")

    def test_cancel_uses_scene_job_and_retry_uses_failure_state(self):
        ui = self.controls
        ui.generate.click()
        self.assertTrue(ui.generate.disabled)
        ui.cancel.click()
        self.assertTrue(self.job.cancelled)
        self.assertFalse(self.session.cancelled)
        self.job.state = {"status": "failed", "error": "Try another route"}
        ui.tick()
        self.assertFalse(ui.retry.disabled)
        ui.retry.click()
        self.assertTrue(self.job.retried)

    def test_failed_job_error_replaces_submission_notice(self):
        ui = self.controls
        ui.generate.click()
        self.assertIn("Scene submitted", ui.status.content)
        self.job.state = {"status": "failed", "error": "Native provider unavailable",
                          "progress": "old progress"}
        ui.tick()
        self.assertIn("Native provider unavailable", ui.status.content)
        self.assertNotIn("Scene submitted", ui.status.content)
        self.assertNotIn("old progress", ui.status.content)

    def test_readiness_callback_blocks_only_the_route_it_reports(self):
        ui = PairedDirectionControls(Gui(), self.session,
            on_generate=lambda request, _client: self.generated.append(request),
            on_preview=lambda *_: None,
            on_frame_cast=lambda *_: None, on_active=lambda *_: None,
            generation_readiness=lambda source: (
                "Configure the native pair provider" if source == "generate" else None))
        ui.source.edit("Describe another interaction")
        self.assertTrue(ui.generate.disabled)
        self.assertIn("Configure the native pair provider", ui.source_note.content)
        ui.generate.click()
        self.assertEqual(self.generated, [])
        self.assertIn("Configure the native pair provider", ui.status.content)
        ui.source.edit("Handshake")
        self.assertFalse(ui.generate.disabled)
        self.assertNotIn("Configure the native pair provider", ui.status.content)
        ui.generate.click()
        self.assertEqual(self.generated[0]["source"], "handshake")

    def test_request_fields_freeze_while_scene_is_running(self):
        ui = self.controls
        ui.generate.click()
        for handle in (ui.first, ui.second, ui.source, ui.place, ui.prompt,
                       ui.first_x, ui.first_z, ui.second_x, ui.second_z,
                       ui.meet_x, ui.meet_z, ui.meet_yaw, ui.seed):
            self.assertTrue(handle.disabled)
        self.job.state = {"status": "ready", "total_frames": 210}
        ui.tick()
        self.assertFalse(ui.first.disabled)
        self.assertFalse(ui.source.disabled)
        self.assertFalse(ui.seed.disabled)

    def test_source_choice_keeps_custom_story_text_visible(self):
        ui = self.controls
        self.assertTrue(ui.prompt.disabled)
        ui.source.edit("Describe another interaction")
        self.assertFalse(ui.prompt.disabled)
        ui.prompt.edit("They begin a choreographed duel.")
        ui.generate.click()
        self.assertEqual(self.generated[0][0]["source"], "generate")
        self.assertEqual(self.generated[0][0]["prompt"], "They begin a choreographed duel.")
        ui.source.edit("Sparring")
        self.assertTrue(ui.prompt.disabled)
        self.assertIn("non-contact", ui.prompt.value)
        ui.source.edit("Describe another interaction")
        self.assertEqual(ui.prompt.value, "They begin a choreographed duel.")

    def test_editor_scene_provider_updates_places_without_mutating_native_scene(self):
        editor_scene = {"objects": [{"id": "market", "name": "Market"}], "targets": []}
        ui = PairedDirectionControls(Gui(), self.session,
            on_generate=lambda *_: None, on_preview=lambda *_: None,
            on_frame_cast=lambda *_: None, on_active=lambda *_: None,
            scene_provider=lambda: editor_scene)
        self.assertIn("Near Market · market", ui.place.options)
        self.assertNotIn("Near Plaza · plaza", ui.place.options)
        editor_scene["objects"][0]["name"] = "Old Market"
        ui.tick()
        self.assertIn("Near Old Market · market", ui.place.options)
        self.assertEqual(self.session.scene["objects"][0]["name"], "Plaza")

    def test_native_scene_property_is_accepted_when_no_provider(self):
        class PropertySession(Session):
            @property
            def scene_document(self):
                return self.scene

        ui = PairedDirectionControls(Gui(), PropertySession(),
            on_generate=lambda *_: None, on_preview=lambda *_: None,
            on_frame_cast=lambda *_: None, on_active=lambda *_: None)
        self.assertIn("Near Plaza · plaza", ui.place.options)

    def test_watch_and_export_controls_follow_busy_and_capture_state(self):
        exports = []
        ui = PairedDirectionControls(Gui(), self.session,
            on_generate=lambda *_: self.session, on_preview=lambda *_: None,
            on_frame_cast=lambda *_: None, on_active=lambda *_: None,
            on_export=exports.append)
        self.assertTrue(ui.play.disabled)
        self.session.frames = 400
        ui.tick()
        self.assertFalse(ui.play.disabled)
        ui.play.click()
        ui.pause.click()
        ui.export.click(client="viewer")
        self.assertEqual(self.session.calls, [("seek", 0), ("play",), ("pause",)])
        self.assertEqual(exports, ["viewer"])
        self.session.capturing = True
        ui.tick()
        self.assertTrue(ui.play.disabled)
        self.assertTrue(ui.export.disabled)
        self.assertTrue(ui.frame_cast.disabled)
        self.assertTrue(ui.cancel.disabled)

    def test_restore_custom_direction_round_trips_without_submission_or_edit(self):
        edits = []
        ui = PairedDirectionControls(Gui(), self.session,
            on_generate=lambda *_: self.fail("restore must not generate"),
            on_preview=lambda *_: self.fail("restore must not preview"),
            on_frame_cast=lambda *_: None, on_active=lambda *_: None,
            on_edit=lambda: edits.append("edit"))
        archived = {
            "actor_ids": ["actor_2", "actor_1"],
            "prompt": "They perform a playful synchronized dance and step apart.",
            "source": "generate", "seed": 7302,
            "starts": [{"x": 4.5, "z": -3.25}, {"x": -5., "z": 1.75}],
            "meeting": {"x": 2.25, "z": 1.5, "yaw_degrees": -35.},
            "target_id": "plaza",
        }
        ui.restore_request(archived)
        self.assertEqual(ui._request(), archived)
        self.assertEqual(ui.source.value, "Describe another interaction")
        self.assertFalse(ui.prompt.disabled)
        self.assertEqual(edits, [])
        ui.source.edit("Sparring")
        ui.source.edit("Describe another interaction")
        self.assertEqual(ui.prompt.value, archived["prompt"])

    def test_manual_edits_invalidate_but_programmatic_marks_do_not(self):
        edits = []
        ui = PairedDirectionControls(Gui(), self.session,
            on_generate=lambda *_: None, on_preview=lambda *_: None,
            on_frame_cast=lambda *_: None, on_active=lambda *_: None,
            on_edit=lambda: edits.append("edit"))
        original_set = ui._set

        def setter_with_visor_event(handle, name, value):
            changed = getattr(handle, name, None) != value
            original_set(handle, name, value)
            if changed and name == "value" and "update" in handle.callbacks:
                handle.callbacks["update"](SimpleNamespace(client=None))

        # Viser can echo a programmatic value assignment as an update event.
        # set_marks must suppress that event, including a target reset to Custom.
        ui._set = setter_with_visor_event
        ui.place.value = "Near Plaza · plaza"
        ui.set_marks({"starts": [{"x": -3., "z": -2.}, {"x": 3., "z": -2.}],
                      "meeting": {"x": 1., "z": 0., "yaw_degrees": 25.}})
        self.assertEqual(ui.place.value, CUSTOM)
        self.assertEqual(edits, [])
        self.assertEqual(self.generated, [])

        for handle, value in ((ui.first_x, -4.), (ui.meet_yaw, 35.),
                              (ui.seed, "99"), (ui.place, "Near Plaza · plaza"),
                              (ui.source, "Describe another interaction"),
                              (ui.prompt, "They turn and wave.")):
            with self.subTest(handle=handle):
                before = len(edits)
                ui._preview_summary = "Old route preview"
                handle.edit(value)
                self.assertGreater(len(edits), before)
                self.assertEqual(ui._preview_summary, "")
        self.assertEqual(self.generated, [])


if __name__ == "__main__":
    unittest.main()
