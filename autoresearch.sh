#!/usr/bin/env bash
# Autoresearch benchmark: policies/ctrl_x.py:CtrlX on the fixed workload in
# sbx/arbench.py (10 scenarios x 7 days, corrected tree, occupancy seed 12345,
# exact C step). Prints METRIC lines. Exits non-zero on a failed scenario, a frost
# violation, or a policy that leaves numpy/stdlib or names a scenario.
set -euo pipefail
cd "$(dirname "$0")"
# One BLAS thread per process: two workers on a fanless laptop, no oversubscription.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec nice -n 5 starter-kit/.venv/bin/python -m sbx.arbench
