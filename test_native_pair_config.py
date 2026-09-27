"""Local provider discovery never needs a remote model or GPU."""

import json
from pathlib import Path
import tempfile
import unittest

from native_pair_config import ENV_NAME, LOCAL_CONFIG, resolve_native_pair_provider


CONFIG = {"ssh_host": "root@example.org", "ssh_port": 22,
          "known_hosts": "/tmp/known_hosts"}


class NativePairConfigResolutionTests(unittest.TestCase):
    def test_missing_implicit_config_keeps_viewer_available_with_setup_message(self):
        with tempfile.TemporaryDirectory() as directory:
            result = resolve_native_pair_provider(repo_root=directory, environ={})
            self.assertIsNone(result.provider)
            self.assertEqual(result.source, "missing")
            self.assertEqual(result.path, Path(directory) / LOCAL_CONFIG)
            self.assertIn("STAGEZERO_NATIVE_PAIR_CONFIG", result.status)

    def test_precedence_cli_then_environment_then_repo_local(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / LOCAL_CONFIG
            local.parent.mkdir()
            local.write_text(json.dumps(CONFIG))
            env_path = root / "env.json"
            env_path.write_text(json.dumps(dict(CONFIG, ssh_port=2222)))
            cli_path = root / "cli.json"
            cli_path.write_text(json.dumps(dict(CONFIG, ssh_port=3333)))
            environment = {ENV_NAME: str(env_path)}
            local_result = resolve_native_pair_provider(repo_root=root, environ={})
            self.assertEqual(local_result.source, "local")
            self.assertEqual(local_result.provider.config.ssh_port, 22)
            env_result = resolve_native_pair_provider(repo_root=root, environ=environment)
            self.assertEqual(env_result.source, "environment")
            self.assertEqual(env_result.provider.config.ssh_port, 2222)
            cli_result = resolve_native_pair_provider(cli_path, repo_root=root, environ=environment)
            self.assertEqual(cli_result.source, "cli")
            self.assertEqual(cli_result.provider.config.ssh_port, 3333)

    def test_selected_invalid_config_fails_visibly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "from cli"):
                resolve_native_pair_provider(root / "absent.json", repo_root=root, environ={})
            with self.assertRaisesRegex(ValueError, "STAGEZERO_NATIVE_PAIR_CONFIG is empty"):
                resolve_native_pair_provider(repo_root=root, environ={ENV_NAME: " "})
            bad = root / LOCAL_CONFIG
            bad.parent.mkdir()
            bad.write_text('{"unknown":true}')
            with self.assertRaisesRegex(ValueError, "from local"):
                resolve_native_pair_provider(repo_root=root, environ={})


if __name__ == "__main__":
    unittest.main()
