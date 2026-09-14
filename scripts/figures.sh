#!/usr/bin/env bash
# Rebuild every figure and table from a finished results directory.
set -euo pipefail
SCALE="${1:-${SCALE:-paper}}"
export PYTHONPATH="${PYTHONPATH:-/work}:/work/experiments" MPLBACKEND=Agg
cd /work
python figures/fig_benchmark.py  "$SCALE"
python figures/fig_assignment.py "$SCALE"
python figures/make_tables.py    "$SCALE"
