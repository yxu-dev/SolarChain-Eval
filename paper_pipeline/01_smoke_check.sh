#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SolarChain-rl

mkdir -p outputs/paper_runs

echo "[1/4] Python compile check"
python -m compileall -q src scripts tests

echo "[2/4] Built-in policy evaluation smoke"
python scripts/evaluate.py \
  --policies "static,random,myopic" \
  --episodes 1 \
  --run-name smoke_builtin

SMOKE_RUN="$(ls -td outputs/runs/*_smoke_builtin | head -n 1)"
echo "$SMOKE_RUN" > outputs/paper_runs/latest_smoke_run.txt

echo "[3/4] DQN training smoke"
python scripts/train.py \
  --algo dqn \
  --timesteps 5 \
  --run-name smoke_dqn

echo "[4/4] Figure generation smoke"
python scripts/make_figures.py \
  --run-dir "$SMOKE_RUN" \
  --figures-dir outputs/paper_runs/smoke/figures

echo "Smoke check passed."
echo "Smoke run: $SMOKE_RUN"
echo "Smoke figures: outputs/paper_runs/smoke/figures"

