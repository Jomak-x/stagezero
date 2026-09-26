"""Focused user-action checks for the native pair Motion panel."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest

from native_pair_controls import NativePairControls, HANDSHAKE_PROMPT


class Handle:
    def __init__(self, **fields):
        self.callbacks = {}
        self.disabled = False
        for name, value in fields.items():
            setattr(self, name, value)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def on_click(self, callback):
        self.callbacks["click"] = callback
        return callback

    def on_update(self, callback):
        self.callbacks["update"] = callback
        return callback

    def on_upload(self, callback):
        self.callbacks["upload"] = callback
        return callback

    def click(self, client=None):
        self.callbacks["click"](SimpleNamespace(client=client))

    def edit(self, value):
        self.value = value
        if "update" in self.callbacks:
            self.callbacks["update"](SimpleNamespace(client=None))

    def upload(self, content):
        self.callbacks["upload"](SimpleNamespace(file=SimpleNamespace(content=content), client=None))


class Gui:
    def __init__(self):
        self.handles = {}

    def _new(self, label, **fields):
        handle = Handle(**fields)
        self.handles[label] = handle
        return handle

    def add_folder(self, label, **_):
        return self._new(label)

    def add_markdown(self, content):
        return self._new("status", content=content)

    def add_button(self, label, **_):
        return self._new(label)

    def add_upload_button(self, label, **_):
        return self._new(label)

    def add_dropdown(self, label, options, initial_value=None):
        return self._new(label, options=options, value=initial_value or options[0])

    def add_text(self, label, initial_value="", **_):
        return self._new(label, value=initial_value)

    def add_slider(self, label, *, min, max, step, initial_value):
        return self._new(label, min=min, max=max, step=step, value=initial_value)

    add_number = add_slider


class Session:
    def __init__(self):
        self.cast = [{"id": "actor_1", "name": "Ava"}, {"id": "actor_2", "name": "Ben"}]
        self.pair = ("actor_1", "actor_2")
        self.active = False
        self.available = True
        self.busy = False
        self.frames = 0
        self.frame = 0
        self.phase = "ready"
        self.composed = False
        self.hand_pose = "relaxed"
        self.capturing = False
        self.placement = {"x": 0., "z": 0., "yaw_degrees": 0.}
        self.calls = []

    def snapshot(self):
        return dict(cast=[dict(item) for item in self.cast], selected_pair=self.pair,
                    active=self.active, available=self.available, busy=self.busy,
                    total_frames=self.frames, frame=self.frame, phase=self.phase, status="Ready",
                    placement=dict(self.placement), composed=self.composed, hand_pose=self.hand_pose,
                    capturing=self.capturing)

    def add_actor(self, name):
        actor_id = f"actor_{len(self.cast) + 1}"
        self.cast.append({"id": actor_id, "name": name})
        self.calls.append(("add", name))
        return actor_id

    def rename_actor(self, actor_id, name):
        next(item for item in self.cast if item["id"] == actor_id)["name"] = name
        self.calls.append(("rename", actor_id, name))

    def remove_actor(self, actor_id):
        self.cast = [item for item in self.cast if item["id"] != actor_id]
        self.calls.append(("remove", actor_id))

    def select_pair(self, first, second):
        self.pair = first, second
        self.calls.append(("pair", first, second))

    def set_placement(self, *, x, z, yaw_degrees):
        self.placement = {"x": x, "z": z, "yaw_degrees": yaw_degrees}
        self.calls.append(("place", x, z, yaw_degrees))

    def set_hand_pose(self, pose):
        self.hand_pose = pose
        self.calls.append(("hand_pose", pose))

    def load_reviewed_handshake(self, *, scene_document):
        self.calls.append(("reviewed", scene_document))
        self.frames = 210

    def load_sparring_preview(self, *, scene_document):
        self.calls.append(("sparring", scene_document))
        self.frames = 210

    def generate(self, prompt, seed, *, frames, scene_document):
        self.calls.append(("generate", prompt, seed, frames, scene_document))

    def play(self):
        self.calls.append(("play",))

    def pause(self):
        self.calls.append(("pause",))

    def restart(self):
        self.calls.append(("restart",))

    def seek(self, frame):
        self.calls.append(("seek", frame))
        self.frame = frame

    def cancel(self):
        self.calls.append(("cancel",))

    def retry(self):
        self.calls.append(("retry",))

    def save(self):
        self.calls.append(("save",))
        return b"native-source"

    def load(self, content):
        if content != b"native-source":
            raise ValueError("Bad native archive")
        self.calls.append(("load", content))
        self.frames = 210


class Studio:
    def scene_document(self):
        return {"version": 2, "objects": []}


class NativePairControlsTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.gui = Gui()
        self.session = Session()
        self.active_calls = []
        self.project_folder = Path(self.temp.name) / "native-pair-projects"

        def activate(value):
            self.active_calls.append(value)
            self.session.active = value

        self.controls = NativePairControls(self.gui, self.session, Studio(), on_active=activate,
                                           project_folder=self.project_folder)

    def test_add_rename_and_select_pair_then_generate(self):
        ui = self.controls
        self.assertEqual(ui._selected_id(ui.first), "actor_1")
        self.assertEqual(ui._selected_id(ui.second), "actor_2")
        ui.new_name.value = "Cory"
        ui.add.click()
        self.assertEqual(len(self.session.cast), 3)
        cory = next(label for label in ui.actor.options if "Cory" in label)
        ui.actor.edit(cory)
        ui.actor_name.value = "Casey"
        ui.rename.click()
        self.assertEqual(self.session.cast[2]["name"], "Casey")
        ui.second.edit(next(label for label in ui.second.options if "Casey" in label))
        ui.choose_pair.click()
        self.assertEqual(self.session.pair, ("actor_1", "actor_3"))
        ui.prompt.value = "Two people circle one another."
        ui.seed.value = "99"
        ui.generate.click()
        self.assertEqual(self.session.calls[-1][:4],
                         ("generate", "Two people circle one another.", 99, 210))
        self.assertEqual(self.active_calls[-1], True)

    def test_reviewed_handshake_and_scrub(self):
        ui = self.controls
        ui.reviewed.click()
        self.assertIn(("reviewed", {"version": 2, "objects": []}), self.session.calls)
        self.assertIn(("play",), self.session.calls)
        self.assertEqual(ui.prompt.value, HANDSHAKE_PROMPT)
        ui.frame.edit(90)
        self.assertIn(("seek", 90), self.session.calls)
        self.assertIn("3\\.0s", ui.status.content)

    def test_curated_sparring_preview_plays_and_disables_during_capture(self):
        ui = self.controls
        ui.sparring.click()
        self.assertIn(("sparring", {"version": 2, "objects": []}), self.session.calls)
        self.assertEqual(self.session.calls[-1], ("play",))
        self.assertIn("non\\-contact", ui.status.content)
        self.session.capturing = True
        ui.tick()
        self.assertTrue(ui.sparring.disabled)

    def test_pair_placement_applies_one_shared_transform(self):
        ui = self.controls
        ui.position_x.value = 2.5
        ui.position_z.value = -1.0
        ui.yaw.value = 90.
        ui.place.click()
        self.assertIn(("place", 2.5, -1., 90.), self.session.calls)

    def test_hand_pose_is_explicit_and_disabled_without_clip_or_during_capture(self):
        ui = self.controls
        self.assertTrue(ui.hand_pose.disabled)
        self.session.frames = 210
        ui.tick()
        self.assertFalse(ui.hand_pose.disabled)
        ui.hand_pose.edit("Closed fists")
        self.assertIn(("hand_pose", "fists"), self.session.calls)
        self.session.capturing = True
        ui.tick()
        self.assertTrue(ui.hand_pose.disabled)

    def test_frame_cast_and_ardy_context_callbacks(self):
        framed = []
        builds = []
        self.controls.on_frame_cast = lambda client: framed.append(client)
        self.controls.on_build_context = lambda: builds.append(True)
        self.controls.tick()
        client = object()
        self.controls.frame_cast.click(client)
        self.assertEqual(framed, [client])
        self.assertTrue(self.controls.build_context.disabled)
        self.session.frames = 210
        self.controls.tick()
        self.assertFalse(self.controls.build_context.disabled)
        self.controls.build_context.click()
        self.assertEqual(builds, [True])
        self.session.composed = True
        self.controls.tick()
        self.assertTrue(self.controls.build_context.disabled)

    def test_invalid_prompt_seed_or_pair_does_not_generate(self):
        ui = self.controls
        ui.prompt.value = ""
        ui.generate.click()
        self.assertIn("Describe", ui.status.content)
        ui.prompt.value = "Together"
        ui.seed.value = "not-an-int"
        ui.generate.click()
        self.assertIn("seed", ui.status.content)
        ui.seed.value = "42"
        ui.second.value = ui.first.value
        ui.generate.click()
        self.assertIn("different characters", ui.status.content)
        self.assertFalse(any(call[0] == "generate" for call in self.session.calls))

    def test_save_open_and_reject_invalid_archive(self):
        ui = self.controls
        ui.reviewed.click()
        ui.save.click()
        self.assertEqual(len(list(self.project_folder.glob("*.native-pair.stagezero.npz"))), 1)
        ui.open.click()
        self.assertIn(("load", b"native-source"), self.session.calls)
        previous_loads = sum(call[0] == "load" for call in self.session.calls)
        ui.upload.upload(b"bad")
        self.assertEqual(sum(call[0] == "load" for call in self.session.calls), previous_loads)
        self.assertIn("Bad native archive", ui.status.content)

    def test_capture_runs_off_callback_thread(self):
        from threading import Event
        started, release = Event(), Event()
        path = Path(self.temp.name) / "native-pair-videos" / "capture-1" / "playback.mp4"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        self.controls.on_capture = lambda _client: (started.set(), release.wait(1), path)[-1]
        downloads = []
        client = SimpleNamespace(send_file_download=lambda name, data: downloads.append((name, data)))
        self.session.frames = 210
        self.controls.tick()
        before = time.monotonic()
        self.controls.capture.click(client)
        self.assertLess(time.monotonic() - before, 0.2)
        self.assertTrue(started.wait(1))
        self.assertTrue(self.controls.capture.disabled)
        release.set()
        for _ in range(100):
            self.controls.tick()
            if not self.controls._capture_pending:
                break
            time.sleep(0.005)
        self.assertIn("playback\\.mp4", self.controls.status.content)
        self.assertEqual(downloads, [("playback.mp4", path.read_bytes())])

    def test_capture_rejects_path_outside_trusted_video_directory(self):
        path = Path(self.temp.name) / "private.mp4"
        path.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        self.controls.on_capture = lambda _client: path
        self.session.frames = 210
        self.controls.tick()
        downloads = []
        self.controls.capture.click(SimpleNamespace(send_file_download=lambda *args: downloads.append(args)))
        for _ in range(100):
            self.controls.tick()
            if not self.controls._capture_pending:
                break
            time.sleep(0.005)
        self.assertEqual(downloads, [])
        self.assertIn("unexpected file location", self.controls.status.content)

    def test_real_session_reviewed_handshake_and_archive_contract(self):
        from native_pair_session import NativePairSession
        from paired_scene import EMPTY_SCENE

        actual = NativePairSession()
        gui = Gui()
        folder = Path(self.temp.name) / "real-projects"
        controls = NativePairControls(gui, actual,
                                      SimpleNamespace(scene_document=lambda: EMPTY_SCENE),
                                      on_active=actual.activate, project_folder=folder)
        controls.reviewed.click()
        self.assertEqual(actual.snapshot()["total_frames"], 210)
        self.assertEqual(actual.snapshot()["fps"], 30)
        self.assertEqual(actual.snapshot()["selected_pair"], ["actor_1", "actor_2"])
        controls.pause.click()
        controls.save.click()
        archive = next(folder.glob("*.native-pair.stagezero.npz"))
        controls.upload.upload(archive.read_bytes())
        self.assertEqual(actual.snapshot()["total_frames"], 210)


if __name__ == "__main__":
    unittest.main()
