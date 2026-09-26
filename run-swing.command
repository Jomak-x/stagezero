#!/bin/zsh
set -eu
cd "$(dirname "$0")"
PYTHON="${STAGEZERO_PYTHON:-$PWD/.venv/bin/python}"
THREE="${STAGEZERO_THREE_DIR:-$PWD/studio_client/node_modules/three/build}"
TOKEN="${STAGEZERO_TOKEN_FILE:-$PWD/.runtime/api-token}"
if [[ ! -x "$PYTHON" ]]; then
  print -u2 'Set STAGEZERO_PYTHON to the installed StageZero Python environment.'
  exit 1
fi
if [[ ! -f "$THREE/three.module.js" ]]; then
  print -u2 'Install the existing studio client dependencies (cd studio_client && npm ci), or set STAGEZERO_THREE_DIR.'
  exit 1
fi
if [[ ! -f "$TOKEN" ]]; then
  print -u2 'Set STAGEZERO_TOKEN_FILE to the existing private motion service bearer token.'
  exit 1
fi
exec "$PYTHON" -u swing_server.py --port "${STAGEZERO_SWING_PORT:-2360}" --backend "${STAGEZERO_REALTIME_URL:-http://127.0.0.1:8769}" --token-file "$TOKEN" --three-dir "$THREE"
