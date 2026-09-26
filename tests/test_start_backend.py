"""Behavior tests for the backend launcher without starting the real backend."""

import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest


LAUNCHER = Path(__file__).resolve().parents[1] / "start-backend.sh"
SCRATCH = LAUNCHER.parent / ".runtime" / "cache-startup-test"
UNSET = "<unset>"


class StartBackendTests(unittest.TestCase):
    def _launch(self, *, cache=False, token=False, hf_home=None, running=False):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temporary:
            workspace = Path(temporary)
            project = workspace / "stagezero"
            runtime = project / ".runtime"
            runtime.mkdir(parents=True)
            (runtime / "api-token").write_text("test-token")
            if token:
                (runtime / "hf-token").write_text("test-hf-token")
            cache_path = workspace / ".cache" / "huggingface"
            if cache:
                cache_path.mkdir(parents=True)

            bin_dir = workspace / "bin"
            bin_dir.mkdir()
            pgrep = bin_dir / "pgrep"
            pgrep.write_text("#!/bin/sh\nexit \"${STUB_PGREP_EXIT:-1}\"\n")
            pgrep.chmod(0o755)
            nohup = bin_dir / "nohup"
            nohup.write_text("#!/bin/sh\nexec \"$@\"\n")
            nohup.chmod(0o755)

            python_stub = project / ".venv" / "bin" / "python"
            python_stub.parent.mkdir(parents=True)
            python_stub.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"${HF_HOME-<unset>}\" "
                "\"${HF_TOKEN_PATH-<unset>}\" "
                "\"${PYTHONPATH-<unset>}\" "
                "\"${HF_XET_CHUNK_CACHE_SIZE_BYTES-<unset>}\" "
                '> "$CAPTURE_PATH.tmp"\n'
                'mv "$CAPTURE_PATH.tmp" "$CAPTURE_PATH"\n'
            )
            python_stub.chmod(0o755)

            # The launcher has fixed Pod paths. Substitute only those paths in a
            # temporary copy so every filesystem and process effect stays local.
            source = LAUNCHER.read_text()
            source = source.replace("/workspace/stagezero", str(project))
            source = source.replace("/workspace/.cache/huggingface", str(cache_path))
            source = source.replace("/workspace/hf", str(workspace / "hf"))
            launcher = workspace / "start-backend.sh"
            launcher.write_text(source)

            capture = workspace / "launched-environment.txt"
            environment = os.environ.copy()
            environment.pop("HF_HOME", None)
            environment.pop("HF_TOKEN_PATH", None)
            environment.update(
                PATH=f"{bin_dir}:{environment['PATH']}",
                CAPTURE_PATH=str(capture),
                STUB_PGREP_EXIT="0" if running else "1",
            )
            if hf_home is not None:
                environment["HF_HOME"] = hf_home
            result = subprocess.run(
                ["bash", str(launcher)],
                cwd=workspace,
                env=environment,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            if running:
                self.assertFalse(capture.exists())
                self.assertFalse((runtime / "backend.pid").exists())
                return result.stdout, None, project, cache_path

            deadline = time.monotonic() + 2
            while not capture.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(capture.exists(), result.stderr)
            self.assertTrue((runtime / "backend.pid").exists())
            return result.stdout, capture.read_text().splitlines(), project, cache_path

    def test_existing_cache_is_inherited_when_hf_home_is_unset(self):
        _, launched, _, cache_path = self._launch(cache=True)
        self.assertEqual(launched[0], str(cache_path))
        self.assertEqual(launched[1], UNSET)

    def test_empty_hf_home_uses_existing_cache(self):
        _, launched, _, cache_path = self._launch(cache=True, hf_home="")
        self.assertEqual(launched[0], str(cache_path))

    def test_missing_cache_leaves_hf_home_unset(self):
        _, launched, _, _ = self._launch()
        self.assertEqual(launched[0], UNSET)

    def test_explicit_hf_home_wins_over_existing_cache(self):
        _, launched, _, _ = self._launch(cache=True, hf_home="/explicit/cache")
        self.assertEqual(launched[0], "/explicit/cache")

    def test_hf_token_keeps_legacy_cache_default(self):
        _, launched, project, _ = self._launch(cache=True, token=True)
        self.assertEqual(launched[0], str(project.parent / "hf"))
        self.assertEqual(launched[1], str(project / ".runtime" / "hf-token"))

    def test_explicit_hf_home_with_token_is_preserved(self):
        _, launched, project, _ = self._launch(token=True, hf_home="/explicit/cache")
        self.assertEqual(launched[0], "/explicit/cache")
        self.assertEqual(launched[1], str(project / ".runtime" / "hf-token"))

    def test_launch_keeps_other_required_environment(self):
        _, launched, project, _ = self._launch()
        self.assertEqual(launched[2:], [str(project / "ardy"), "0"])

    def test_running_backend_exits_without_launching(self):
        stdout, launched, _, _ = self._launch(running=True)
        self.assertIsNone(launched)
        self.assertIn("already running", stdout)


if __name__ == "__main__":
    unittest.main()
