"""Serve the versioned StageZero browser client through Viser's existing socket.

The pinned Viser version does not expose a custom HTTP asset directory. We adapt
only WebsockServer construction, restoring the module immediately afterward;
no installed dependency files are modified. Preview/live viewers keep upstream UI.
"""
from pathlib import Path
from threading import RLock
from unittest.mock import patch
import viser
from viser import infra

_BUILD = Path(__file__).resolve().parent / 'studio_client' / 'build'
_CONSTRUCTION_LOCK = RLock()


def create_studio_server(**kwargs):
    if not (_BUILD / 'index.html').is_file():
        raise RuntimeError('Build the studio client first: cd studio_client && npm ci && npm run build')
    with _CONSTRUCTION_LOCK:
        upstream = infra.WebsockServer

        def studio_socket(*args, **socket_kwargs):
            socket_kwargs['http_server_root'] = _BUILD
            return upstream(*args, **socket_kwargs)

        with patch.object(infra, 'WebsockServer', studio_socket):
            return viser.ViserServer(**kwargs)
