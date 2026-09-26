#!/usr/bin/env bash
set -euo pipefail
cd /workspace/stagezero
if pgrep -f '^.venv/bin/python -u pod_backend.py$' >/dev/null; then
  echo 'StageZero backend is already running.'
  exit 0
fi
test -s .runtime/api-token
nohup env PYTHONPATH=/workspace/stagezero/ardy HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
  .venv/bin/python -u pod_backend.py >> backend.log 2>&1 </dev/null &
echo $! > .runtime/backend.pid
echo 'StageZero backend started. Cached model loading takes about 70 seconds.'
