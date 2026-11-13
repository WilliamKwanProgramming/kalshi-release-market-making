#!/usr/bin/env bash
# Regenerate every result (all on synthetic data) and the README tables. Single-threaded.
# Usage: scripts/reproduce.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PYTHON:-python3}
start=$(date +%s)
commands=()
run() {
  commands+=("$PY $*")
  echo "+ $PY $*"
  "$PY" "$@"
}
# Old outputs are removed first so that a result that is no longer produced cannot linger.
rm -f results/markouts/*.csv results/markouts/*.json results/markouts/*.manifest.txt \
  results/markouts/tables.md
rm -f results/backtest/*.csv results/backtest/*.json results/backtest/*.manifest.txt \
  results/backtest/tables.md
rm -f results/calibration/*.csv results/calibration/*.json results/calibration/*.manifest.txt \
  results/calibration/tables.md
rm -f results/rehearsal/*.csv results/rehearsal/*.json results/rehearsal/*.manifest.txt \
  results/rehearsal/tables.md results/rehearsal/DEVIATIONS.md
run -m kalshi_mm.experiments.recovery --out results/markouts
run -m kalshi_mm.experiments.backtest_study --out results/backtest
run -m kalshi_mm.experiments.calibration_study --out results/calibration
run -m kalshi_mm.experiments.rehearsal --out results/rehearsal
run scripts/make_tables.py
end=$(date +%s)
# User plus system CPU seconds of all child processes: the second line of the bash builtin
# "times". It has to run in this shell, not in a subshell, so it goes through a file.
times > results/.cpu_times
cpu=$(tail -1 results/.cpu_times | awk '{s = 0; for (i = 1; i <= 2; i++) {split($i, a, "m");
  sub("s", "", a[2]); s += a[1] * 60 + a[2]}; printf "%.0f", s}')
rm -f results/.cpu_times
"$PY" scripts/write_manifest.py --seconds $((end - start)) --cpu-seconds "$cpu" \
  --out results/MANIFEST.txt -- "${commands[@]}"
echo "done in $((end - start)) s wall, $cpu s CPU of child processes; see results/MANIFEST.txt"
