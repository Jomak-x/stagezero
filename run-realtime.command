#!/bin/zsh
set -eu
cd "${0:A:h}"
PYTHON_BIN="${STAGEZERO_PYTHON:-$PWD/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  print -u2 'Set STAGEZERO_PYTHON to the Python environment with requirements-live.txt installed.'
  exit 1
fi
TOKEN_FILE="${STAGEZERO_TOKEN_FILE:-$PWD/.runtime/api-token}"
if [[ ! -s "$TOKEN_FILE" ]]; then
  print -u2 'Set STAGEZERO_TOKEN_FILE to your existing private backend token file.'
  exit 1
fi
exec "$PYTHON_BIN" realtime_viewer.py --token-file "$TOKEN_FILE" \
  --url "${STAGEZERO_REALTIME_URL:-http://127.0.0.1:8769}" \
  --port "${STAGEZERO_REALTIME_PORT:-2350}" "$@"
