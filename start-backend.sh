#!/usr/bin/env bash
set -euo pipefail
cd /workspace/stagezero
if pgrep -f '^.venv/bin/python -u pod_backend.py$' >/dev/null; then
  echo 'StageZero backend is already running.'
  exit 0
fi
test -s .runtime/api-token
if [[ -f .runtime/hf-token ]]; then
  export HF_TOKEN_PATH="$PWD/.runtime/hf-token"
  if [[ -z "${HF_HOME:-}" ]]; then
    export HF_HOME=/workspace/hf
  fi
elif [[ -z "${HF_HOME:-}" && -d /workspace/.cache/huggingface ]]; then
  export HF_HOME=/workspace/.cache/huggingface
fi
nohup env PYTHONPATH=/workspace/stagezero/ardy HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
  .venv/bin/python -u pod_backend.py >> backend.log 2>&1 </dev/null &
echo $! > .runtime/backend.pid
echo 'StageZero backend started. Cached model loading takes about 70 seconds.'
