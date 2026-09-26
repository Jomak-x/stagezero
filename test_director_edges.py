"""Regression coverage for project boundaries and in-flight controller work."""
from pathlib import Path
import threading
import time
import tempfile
import unittest
from unittest import mock

import numpy as np

import directing
from directing import DirectorSession
from takes import decode_project
from test_live_motion import ControlledBackend, wait_until


class DirectorEdgeTests(unittest.TestCase):
    def setUp(self):
        self.backend = ControlledBackend()
        self.backend.release.set()
        self.session = DirectorSession(
            self.backend,
            np.zeros((20, 34, 3), dtype=np.float32),
            np.tile(np.eye(3, dtype=np.float32), (20, 34, 1, 1)),
        )
        self.session.set_mode("Live ARDY")

    def generate(self, prompt="old"):
        self.session.submit(prompt)
        wait_until(lambda: not self.session.busy)
        self.session.pause()
        return self.session.takes[self.session.active_take]

    def start_pending_branch(self):
        original = (self.session.takes[self.session.active_take]
                    if self.session.active_take in self.session.takes else self.generate())
        self.backend.started.clear()
        self.backend.release.clear()
        self.session.seek(40)
        self.session.submit("late branch")
        self.assertTrue(self.backend.started.wait(1))
        return original

    def test_new_project_backs_up_all_takes_before_reset(self):
        original = self.generate()
        self.session.seek(40)
        branch = self.generate("alternate")
        self.session.seek(40)
        with tempfile.TemporaryDirectory() as directory:
            backup, _ = self.session.save_project(directory, "automatic-backup")
            self.session.new_project(directory)

            backups = list(Path(directory).glob("automatic-backup-*.stagezero.npz"))
            self.assertTrue(backup.exists())
            self.assertEqual(len(backups), 2)
            newest = max(backups, key=lambda path: path.stat().st_mtime_ns)
            restored, active, frame, _scene = decode_project(newest.read_bytes())
            self.assertEqual(set(restored), {original.id, branch.id})
            self.assertEqual(active, branch.id)
            self.assertEqual(frame, 40)
            self.assertEqual(self.session.takes, {})
            self.assertIsNone(self.session.active_take)
            self.assertIn("backed up", self.session.project_status)

    def test_new_project_backup_failure_retains_current_project(self):
        take = self.generate()
        self.session.seek(17)
        original_positions = self.session.positions
        original_revision = self.session.project_revision
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(self.session, "save_project", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    self.session.new_project(directory)
        self.assertIs(self.session.takes[take.id], take)
        self.assertEqual(self.session.active_take, take.id)
        self.assertEqual(self.session.frame, 17)
        self.assertIs(self.session.positions, original_positions)
        self.assertEqual(self.session.project_revision, original_revision)

    def test_new_project_rejects_late_generation_completion(self):
        self.start_pending_branch()
        with tempfile.TemporaryDirectory() as directory:
            self.session.new_project(directory)
            self.backend.release.set()
            wait_until(lambda: not self.session.busy)
            time.sleep(0.05)
        self.assertEqual(self.session.takes, {})
        self.assertIsNone(self.session.active_take)
        self.assertIsNone(self.session.motion)
        self.assertEqual(self.session.kind, "reference")
        self.assertIn("backed up", self.session.project_status)

    def test_load_rejects_late_generation_completion(self):
        saved_take = self.generate()
        with tempfile.TemporaryDirectory() as directory:
            data = self.session.save_project(directory, "load-source")[1]
        self.start_pending_branch()
        self.session.load_project(data)
        loaded_positions = self.session.positions
        self.backend.release.set()
        wait_until(lambda: not self.session.busy)
        time.sleep(0.05)
        self.assertEqual(set(self.session.takes), {saved_take.id})
        self.assertEqual(self.session.active_take, saved_take.id)
        self.assertIs(self.session.positions, loaded_positions)
        self.assertEqual(self.session.positions[0, 0, 0], 1)

    def test_seek_rejects_late_branch_completion(self):
        original = self.start_pending_branch()
        self.session.seek(20)
        self.backend.release.set()
        wait_until(lambda: not self.session.busy)
        time.sleep(0.05)
        self.assertEqual(set(self.session.takes), {original.id})
        self.assertEqual(self.session.active_take, original.id)
        self.assertEqual(self.session.frame, 20)
        self.assertIs(self.session.positions, original.positions)

    def test_concurrent_save_reports_snapshot_revision(self):
        take = self.generate()
        self.session.seek(103)
        self.backend.started.clear()
        self.backend.release.clear()
        entered_write = threading.Event()
        release_write = threading.Event()
        original_write_bytes = Path.write_bytes
        saved = []
        failures = []

        def delayed_write(path, data):
            if path.suffix == ".tmp":
                entered_write.set()
                if not release_write.wait(3):
                    raise TimeoutError("test did not release snapshot write")
            return original_write_bytes(path, data)

        def save():
            try:
                saved.append(self.session.save_project(save_directory, "snapshot"))
            except Exception as exc:  # surface worker-thread errors in the test
                failures.append(exc)

        with tempfile.TemporaryDirectory() as save_directory:
            with mock.patch.object(Path, "write_bytes", delayed_write):
                thread = threading.Thread(target=save)
                thread.start()
                self.assertTrue(entered_write.wait(1), "save did not reach its unlocked write")
                self.session.submit("concurrent extension")
                self.assertTrue(self.backend.started.wait(1))
                self.backend.release.set()
                wait_until(lambda: not self.session.busy)
                self.assertIn(take.id, self.session.takes)
                release_write.set()
                thread.join(2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(len(saved), 1)
        _path, data = saved[0]
        snapshot, _active, _frame, _scene = decode_project(data)
        self.assertEqual(len(snapshot[take.id].positions), 104)
        self.assertEqual(len(self.session.takes[take.id].positions), 208)
        self.assertIn("newer changes remain unsaved", self.session.project_status)


if __name__ == "__main__":
    unittest.main()
