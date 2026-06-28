#!/usr/bin/env bash
set -euo pipefail

# Use $LUDUS_PYTHON if set, else whatever `python` is active (e.g. the .venv).
PYTHON="${LUDUS_PYTHON:-python}"
STEPS="${STEPS:-12}"

for game in tetris wolf3d; do
  for mode in baseline memory; do
    echo "========================================"
    echo "  LUDUS EXPERIMENT: game=$game mode=$mode"
    echo "========================================"
    "$PYTHON" -m ludus.cli "$game" "$mode" --provider fallback --steps "$STEPS"
  done
done
