#!/usr/bin/env bash
# Reuse the existing Pod; never provisions, resizes, or stops GPU resources.
set -euo pipefail
cd "$(dirname "$0")"
source .runtime/pod.env
: "${STAGEZERO_SSH_HOST:?Set STAGEZERO_SSH_HOST in .runtime/pod.env}"
: "${STAGEZERO_SSH_PORT:=22}"
: "${STAGEZERO_SSH_KEY:=$HOME/.ssh/id_ed25519}"
SSH_ARGS=(-o BatchMode=yes -o ConnectTimeout=10 -o UserKnownHostsFile="$PWD/.runtime/known_hosts" -i "$STAGEZERO_SSH_KEY" -p "$STAGEZERO_SSH_PORT")
ssh "${SSH_ARGS[@]}" "$STAGEZERO_SSH_HOST" 'test -x /workspace/stagezero-characters/.venv/bin/python && test -d /workspace/stagezero-characters/TRELLIS' || {
  echo 'Character model environment is not installed on the existing Pod. See docs/CHARACTERS.md.'
  exit 1
}
# Copy only the worker source; credentials already exist privately on the Pod.
tar --no-xattrs -cf - character_gpu_server.py character_worker_trellis.py character_texture_detail.py character_generation.py start-character-worker.sh warm-character-worker.sh |
  ssh "${SSH_ARGS[@]}" "$STAGEZERO_SSH_HOST" 'tar -xf - -C /workspace/stagezero-characters'
ssh "${SSH_ARGS[@]}" "$STAGEZERO_SSH_HOST" 'bash /workspace/stagezero-characters/start-character-worker.sh'
if ! .venv/bin/python -c 'import socket; socket.create_connection(("127.0.0.1",8770),1).close()' 2>/dev/null; then
  ssh "${SSH_ARGS[@]}" -M -S "$PWD/.runtime/character-ssh" -fN \
    -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -L 127.0.0.1:8770:127.0.0.1:8770 "$STAGEZERO_SSH_HOST"
fi
echo 'Character GPU tunnel is available on localhost:8770.'
