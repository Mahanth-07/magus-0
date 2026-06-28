#!/usr/bin/env bash
set -euo pipefail

PYTHON=/opt/anaconda3/bin/python
STEPS=12

for game in tetris wolf3d; do
  for mode in baseline memory; do
    echo "========================================"
    echo "  LUDUS EXPERIMENT: game=$game mode=$mode"
    echo "========================================"
    "$PYTHON" -m ludus.cli "$game" "$mode" --provider fallback --steps "$STEPS"
  done
done
