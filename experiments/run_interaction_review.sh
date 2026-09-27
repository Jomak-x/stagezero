#!/bin/sh
# Offline exact archive replay in main's existing UI; no Pod or worker changes.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
PYTHON_BIN=${STAGEZERO_PYTHON:-$ROOT/.venv/bin/python}
PROJECT=${1:-$ROOT/review/interaction-quality/final/market-three-s48-fixed-plan/scene.cast.stagezero.npz}
PORT=${2:-24971}
exec "$PYTHON_BIN" director_viewer.py --reference-only --port "$PORT" --native-project "$PROJECT"
