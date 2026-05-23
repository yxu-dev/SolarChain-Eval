#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash paper_pipeline/03_make_figures_for_run.sh <run_dir> [figures_dir]" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SolarChain-rl

RUN_DIR="$1"
if [[ $# -ge 2 ]]; then
  FIGURES_DIR="$2"
elif [[ "$(basename "$(dirname "$RUN_DIR")")" == "runs" ]]; then
  FIGURES_DIR="$(dirname "$(dirname "$RUN_DIR")")/figures/$(basename "$RUN_DIR")"
else
  FIGURES_DIR="$RUN_DIR/figures"
fi
CONFIG="${CONFIG:-configs/month_2026_04.yaml}"

python scripts/make_figures.py \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --figures-dir "$FIGURES_DIR"

echo "Config: $CONFIG"
echo "Figures written to $FIGURES_DIR"

