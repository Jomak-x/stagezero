#!/usr/bin/env bash
# Called only by the private backend launcher after an initial verified install.
set -euo pipefail
cd /workspace/stagezero-characters
exec 9> warmup.lock
flock -n 9 || exit 0
test ! -f /dev/shm/stagezero-characters/ready || exit 0
test -f jobs/smoke/reference.png || { echo 'Initial character smoke reference is missing.'; exit 1; }
mkdir -p /dev/shm/stagezero-characters
.venv/bin/python -u character_worker_trellis.py \
  --input "$PWD/jobs/smoke/reference.png" --output "$PWD/jobs/smoke/warmup.glb"
printf 'TRELLIS character inference and GLB export verified\n' > /dev/shm/stagezero-characters/ready
