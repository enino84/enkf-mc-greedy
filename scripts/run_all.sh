#!/usr/bin/env bash
# The whole suite at one scale: diagnostics, benchmark, figures, tables.
#   bash scripts/run_all.sh paper
# Shards: SHARD_INDEX / SHARD_COUNT split the benchmark runs; figures and
# tables are only built when SHARD_COUNT is 1 (run `make figures` afterwards).
set -euo pipefail
SCALE="${1:-${SCALE:-smoke}}"
export SCALE PYTHONPATH="${PYTHONPATH:-/work}:/work/experiments" MPLBACKEND=Agg
RESULTS_DIR="${RESULTS_DIR:-/work/results}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/run_${SCALE}_shard${SHARD_INDEX:-0}.log"
cd /work

echo "== qgloc suite  scale=$SCALE  shard=${SHARD_INDEX:-0}/${SHARD_COUNT:-1}  $(date -u +%FT%TZ)" | tee -a "$LOG"
python experiments/exp05_first_analysis.py "$SCALE" ${E1_ARGS:-} 2>&1 | tee -a "$LOG"
python experiments/exp02_benchmark.py   "$SCALE" ${TANDA:+--tanda "$TANDA"} 2>&1 | tee -a "$LOG"
if [ "${SHARD_COUNT:-1}" = "1" ]; then
  python figures/fig_benchmark.py  "$SCALE" 2>&1 | tee -a "$LOG"
  python figures/fig_assignment.py "$SCALE" 2>&1 | tee -a "$LOG"
  python figures/make_tables.py    "$SCALE" 2>&1 | tee -a "$LOG"
fi
echo "== done $(date -u +%FT%TZ)" | tee -a "$LOG"
