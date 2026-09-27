#!/usr/bin/env bash
set -euo pipefail
cd /workspace/stagezero-characters
export HF_HOME=/dev/shm/stagezero-characters/hf
export TORCH_HOME=/dev/shm/stagezero-characters/torch
export U2NET_HOME=/dev/shm/stagezero-characters/u2net
export PYTHONPATH="$PWD/TRELLIS${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
if [[ ! -f /dev/shm/stagezero-characters/ready && -f jobs/smoke/reference.png ]]; then
  nohup bash "$PWD/warm-character-worker.sh" > warmup.log 2>&1 < /dev/null &
  echo 'Character model warmup requested; the worker accepts jobs after its smoke check passes.'
fi
if .venv/bin/python -c 'import socket; socket.create_connection(("127.0.0.1",8770),1).close()' 2>/dev/null; then
  echo 'Character worker already listening.'
  exit 0
fi
mkdir -p jobs
nohup .venv/bin/python -u character_gpu_server.py \
  --token-file /workspace/stagezero/.runtime/api-token \
  --jobs-dir "$PWD/jobs" \
  --worker-python "$PWD/.venv/bin/python" \
  --worker-script "$PWD/character_worker_trellis.py" \
  --ready-file /dev/shm/stagezero-characters/ready \
  --worker-cwd "$PWD/TRELLIS" > worker.log 2>&1 < /dev/null &
echo "$!" > worker.pid
echo 'Started private character worker. Log: /workspace/stagezero-characters/worker.log'
