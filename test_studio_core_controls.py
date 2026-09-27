"""Focused checks for the optional Core panel's user actions."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from studio_core_controls import CoreStudioControls
from scene_objects import make_object


class Handle:
    def __init__(self, **fields):
        self.callbacks = {}
        self.visible = True
        self.disabled = False
        for key, value in fields.items():
            setattr(self, key, value)

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

    def edit(self, value, client=None):
        self.value = value
        self.callbacks["update"](SimpleNamespace(client=client))

    def upload(self, content):
        self.callbacks["upload"](SimpleNamespace(file=SimpleNamespace(content=content), client=None))


class Gui:
    def __init__(self):
        self.labels = {}

    def _add(self, label, **fields):
        handle = Handle(**fields)
        self.labels[label] = handle
        return handle

    def add_folder(self, label, **kwargs):
        return self._add(label, expanded=kwargs.get("expand_by_default"))

    def add_checkbox(self, label, initial_value=False):
        return self._add(label, value=initial_value)

    def add_markdown(self, content):
        return self._add("status", content=content)

    def add_dropdown(self, label, options, initial_value=None):
        return self._add(label, options=options, value=initial_value or options[0])

    def add_button(self, label, **_):
        return self._add(label)

    def add_text(self, label, initial_value="", **_):
        return self._add(label, value=initial_value)

    def add_slider(self, label, *, min, max, step, initial_value):
        return self._add(label, min=min, max=max, step=step, value=initial_value)

    def add_upload_button(self, label, **_):
        return self._add(label)


class Core:
    def __init__(self, available=True):
        self.active = False
        self.available = available
        self.actor_ids = ()
        self.initialized = False
        self.total_frames = 0
        self.frame = 0
        self.phase = "empty"
        self.epoch = 0
        self.calls = []

    def snapshot(self):
        return dict(active=self.active, available=self.available, actor_ids=self.actor_ids,
                    total_frames=self.total_frames, frame=self.frame, phase=self.phase,
                    status="Ready", example_available=True, initialized=self.initialized,
                    epoch=self.epoch)

    def start(self, *, actor_count, scene_document, placements):
        self.calls.append(("start", actor_count, scene_document))
        self.active = True
        self.initialized = True
        self.epoch += 1
        self.actor_ids = tuple(f"actor_{i}" for i in range(1, actor_count + 1))

    reset = start

    def activate(self):
        self.calls.append(("activate",))
        self.active = True

    def deactivate(self):
        self.calls.append(("deactivate",))
        self.active = False

    def direct(self, prompts, seconds):
        self.calls.append(("direct", prompts, seconds))

    def navigate(self, actor_id, target_id, verb):
        self.calls.append(("navigate", actor_id, target_id, verb))

    def spatial_commands(self, actor_id, text):
        self.calls.append(("spatial_commands", actor_id, text))

    def play(self):
        self.calls.append(("play",))

    def pause(self):
        self.calls.append(("pause",))

    def seek(self, frame):
        self.calls.append(("seek", frame))

    def retry(self):
        self.calls.append(("retry",))

    def cancel(self):
        self.calls.append(("cancel",))

    def save(self):
        self.calls.append(("save",))
        return b"exact-native-frames"

    def load(self, content):
        self.calls.append(("load", content))
        self.active = True
        self.initialized = True
        self.epoch += 1
        self.actor_ids = ("actor_1",)
        self.total_frames = 60

    def load_example(self):
        self.calls.append(("load_example",))
        self.active = True
        self.initialized = True
        self.epoch += 1
        self.actor_ids = ("actor_1",)
        self.total_frames = 60


class Studio:
    project_revision = 0

    def scene_document(self):
        gate = make_object("arch", 0, position=[0, 1.6, 2])
        gate.update(id="gate-1", name="Gate", size=[3, 3.2, .4])
        floor = make_object("platform", 0, position=[0, -.05, 0])
        floor.update(name="Floor", size=[8, .1, 8])
        table = make_object("table", 0, position=[3, .4, 0])
        table.update(name="Table")
        return {"version": 2, "name": "Stage", "objects": [gate, floor, table],
            "effects": [], "lighting": "neutral"}


class CoreStudioControlsTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.gui = Gui()
        self.core = Core()
        self.active_requests = []
        self.controls = CoreStudioControls(
            self.gui, self.core, Studio(), on_active=self.active_requests.append,
            project_folder=Path(self.temp.name))

    def test_stays_collapsed_and_does_not_start_until_enabled(self):
        self.assertFalse(self.gui.labels["Scene direction · Core"].expanded)
        self.assertEqual(self.core.calls, [])
        self.assertTrue(self.controls.generate.disabled)
        self.controls.enabled.edit(True, client=object())
        self.assertEqual(self.active_requests, [True])
        self.assertEqual(self.core.calls[0][:2], ("start", 1))
        self.assertFalse(self.controls.generate.disabled)

    def test_cast_change_backs_up_exact_frames_before_reset(self):
        self.controls.enabled.edit(True, client=object())
        self.core.total_frames = 40
        self.controls.tick()
        self.controls.cast.value = "Two actors"
        self.controls.start.click()
        self.assertEqual(self.core.calls[-2][0], "save")
        self.assertEqual(self.core.calls[-1][:2], ("start", 2))
        backups = list(Path(self.temp.name).glob("before-core-change-*.npz"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"exact-native-frames")

    def test_actor_prompts_and_exact_scene_target_ids(self):
        self.controls.enabled.edit(True, client=object())
        self.controls.cast.value = "Two actors"
        self.controls.start.click()
        self.controls.actor_one.value = "Actor one dances"
        self.controls.actor_two.value = "Actor two celebrates"
        self.controls.duration.value = "12 seconds"
        self.controls.generate.click()
        self.assertEqual(self.core.calls[-1], ("direct", {
            "actor_1": "Actor one dances", "actor_2": "Actor two celebrates"}, 12))
        self.controls.nav_actor.value = "Actor 2"
        self.controls.verb.value = "go_through"
        self.controls.navigate.click()
        self.assertEqual(self.core.calls[-1], ("navigate", "actor_2", "gate-1", "go_through"))

    def test_spatial_command_button_uses_selected_actor_and_preserves_prompt(self):
        self.assertTrue(self.controls.spatial_run.disabled)
        self.controls.enabled.edit(True, client=object())
        self.controls.cast.value = "Two actors"
        self.controls.start.click()
        self.controls.nav_actor.value = "Actor 2"
        self.controls.spatial_text.value = "walk two metres forward then approach gate-1"
        self.controls.spatial_run.click()
        self.assertEqual(self.core.calls[-1], ("spatial_commands", "actor_2", self.controls.spatial_text.value))

    def test_catalog_excludes_floor_and_rejects_unverified_passage(self):
        self.controls.enabled.edit(True, client=object())
        self.assertEqual(len(self.controls.target.options), 2)
        self.assertFalse(any("Floor" in label for label in self.controls.target.options))
        self.controls.target.edit("Table · table-0", client=object())
        self.assertEqual(self.controls.verb.options, ("approach",))
        self.controls.verb.value = "go_through"
        previous = len(self.core.calls)
        self.controls.navigate.click()
        self.assertEqual(len(self.core.calls), previous)
        self.assertIn("unavailable", self.controls.status.content)

    def test_offline_exact_replay_and_save(self):
        self.core.available = False
        self.controls.tick()
        self.controls.upload.upload(b"saved-project")
        self.assertEqual(self.core.calls[-1], ("load", b"saved-project"))
        self.assertFalse(self.controls.play.disabled)
        self.assertTrue(self.controls.generate.disabled)
        downloads = []
        client = SimpleNamespace(send_file_download=lambda name, data: downloads.append((name, data)))
        self.controls.save.click(client)
        self.assertEqual(downloads[0][1], b"exact-native-frames")
        self.assertTrue(list(Path(self.temp.name).glob("core-*.npz")))

    def test_saved_catalog_opens_exact_local_file_with_backup(self):
        self.controls.enabled.edit(True, client=object())
        self.core.total_frames = 40
        saved = Path(self.temp.name) / "city-arch.core.stagezero.npz"
        saved.write_bytes(b"archived-city-arch")
        self.controls.refresh_saved(force=True)
        self.assertIn(saved.name, self.controls.saved.options)
        self.controls.saved.value = saved.name
        self.controls.open_saved.click()
        self.assertIn(("load", b"archived-city-arch"), self.core.calls)
        backups = list(Path(self.temp.name).glob("before-core-change-*.npz"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"exact-native-frames")

    def test_catalog_skips_symlinks_and_oversized_files(self):
        root = Path(self.temp.name)
        (root / "valid.core.stagezero.npz").write_bytes(b"valid")
        (root / "huge.core.stagezero.npz").touch()
        with (root / "huge.core.stagezero.npz").open("wb") as stream:
            stream.truncate(64_000_001)
        outside = root.parent / "outside.core.stagezero.npz"
        (root / "link.core.stagezero.npz").symlink_to(outside)
        self.controls.refresh_saved(force=True)
        self.assertEqual(self.controls.saved.options, ("valid.core.stagezero.npz",))

    def test_example_load_backs_up_prior_native_project(self):
        self.controls.enabled.edit(True, client=object())
        self.core.total_frames = 40
        self.controls.example_motion.click()
        self.assertEqual(self.core.calls[-2][0], "save")
        self.assertEqual(self.core.calls[-1], ("load_example",))
        backups = list(Path(self.temp.name).glob("before-core-change-*.npz"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"exact-native-frames")
        self.assertIn(backups[0].name, self.controls.saved.options)


if __name__ == "__main__":
    unittest.main()
