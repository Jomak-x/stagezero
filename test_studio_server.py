"""Verify the client adapter doesn't alter other Viser servers."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import studio_server

class StudioServerTests(unittest.TestCase):
    def test_only_studio_constructor_uses_custom_assets_and_restores_factory(self):
        with tempfile.TemporaryDirectory() as temp:
            build = Path(temp)
            (build / 'index.html').write_text('studio')
            original = studio_server.infra.WebsockServer
            calls = []
            def socket(**kwargs):
                calls.append(kwargs)
                return object()
            def server(**kwargs):
                return studio_server.infra.WebsockServer(http_server_root=Path('/upstream'), **kwargs)
            with patch.object(studio_server, '_BUILD', build), patch.object(studio_server.infra, 'WebsockServer', socket), patch.object(studio_server.viser, 'ViserServer', server):
                studio_server.create_studio_server(port=2337)
                self.assertIs(studio_server.infra.WebsockServer, socket)
            self.assertIs(studio_server.infra.WebsockServer, original)
            self.assertEqual(calls[0]['http_server_root'], build)

    def test_missing_build_fails_with_actionable_message(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(studio_server, '_BUILD', Path(temp)):
            with self.assertRaisesRegex(RuntimeError, 'npm run build'):
                studio_server.create_studio_server()
