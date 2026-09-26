#!/bin/zsh
set -eu
cd "$(dirname "$0")"
MOTION_PYTHON="${STAGEZERO_PYTHON:-$PWD/.venv/bin/python}"
MOTION_THREE="${STAGEZERO_THREE_DIR:-$PWD/studio_client/node_modules/three}"
MOTION_TOKEN="${STAGEZERO_TOKEN_FILE:-$PWD/.runtime/api-token}"
if [[ ! -x "$MOTION_PYTHON" ]]; then
  print -u2 'Set STAGEZERO_PYTHON to your installed StageZero Python environment.'
  exit 1
fi
if [[ ! -f "$MOTION_THREE/build/three.module.js" ]]; then
  print -u2 'Run cd studio_client && npm ci, or set STAGEZERO_THREE_DIR to node_modules/three.'
  exit 1
fi
exec "$MOTION_PYTHON" -u grounded_server.py --port "${STAGEZERO_MOTION_PORT:-2361}" \
  --realtime-url "${STAGEZERO_REALTIME_URL:-http://127.0.0.1:8769}" \
  --token-file "$MOTION_TOKEN" --three-dir "$MOTION_THREE" "$@"
