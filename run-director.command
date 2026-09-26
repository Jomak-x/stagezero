#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f studio_client/build/index.html ]]; then
  (cd studio_client && npm ci && npm run build)
fi
test -s .runtime/api-token
if [[ -f .runtime/pod.env ]]; then source .runtime/pod.env; fi
: "${STAGEZERO_SSH_HOST:?Set STAGEZERO_SSH_HOST in .runtime/pod.env (see pod.env.example)}"
: "${STAGEZERO_SSH_PORT:=22}"
: "${STAGEZERO_SSH_KEY:=$HOME/.ssh/id_ed25519}"
SSH_ARGS=(-o BatchMode=yes -o ConnectTimeout=10 -o UserKnownHostsFile="$PWD/.runtime/known_hosts" -i "$STAGEZERO_SSH_KEY" -p "$STAGEZERO_SSH_PORT")
if ssh "${SSH_ARGS[@]}" "$STAGEZERO_SSH_HOST" 'bash /workspace/stagezero/start-backend.sh'; then
  if ! .venv/bin/python -c 'import socket; socket.create_connection(("127.0.0.1",8765),1).close()' 2>/dev/null; then
    ssh "${SSH_ARGS[@]}" -M -S "$PWD/.runtime/pod-ssh" -fN \
      -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
      -L 127.0.0.1:8765:127.0.0.1:8765 "$STAGEZERO_SSH_HOST" || echo 'Tunnel unavailable; recorded preview still works.'
  fi
else
  echo 'Pod unavailable; opening the viewer with recorded fallback available.'
fi
echo 'Viewer: http://127.0.0.1:2336'
echo "Remote access: point your private Tailscale Serve proxy at localhost:2336 after review."
echo 'The Pod may still be loading. Recorded preview remains available.'
if .venv/bin/python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:2336",timeout=2)' 2>/dev/null; then
  echo 'Viewer is already running; open the address above.'
else
  exec .venv/bin/python director_viewer.py
fi
