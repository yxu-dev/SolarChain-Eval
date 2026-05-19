#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SolarChain-rl

CONFIG="${CONFIG:-configs/default.yaml}"
TIMESTEPS="${TIMESTEPS:-100000}"
EPISODES="${EPISODES:-10}"
PAPER_RUN_ID="${PAPER_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_paper}"
PAPER_RUN_DIR="outputs/paper_runs/$PAPER_RUN_ID"
MAIN_RUN_NAME="${PAPER_RUN_ID}_main"
ABLATION_RUN_NAME="${PAPER_RUN_ID}_no_physics_penalty"
MAIN_FIGURES_DIR="$PAPER_RUN_DIR/figures/main"
ABLATION_FIGURES_DIR="$PAPER_RUN_DIR/figures/ablation_no_physics_penalty"
mkdir -p "$PAPER_RUN_DIR" "$MAIN_FIGURES_DIR" "$ABLATION_FIGURES_DIR"

echo "Running SolarChain-Eval paper experiments"
echo "PAPER_RUN_ID=$PAPER_RUN_ID"
echo "PAPER_RUN_DIR=$PAPER_RUN_DIR"
echo "CONFIG=$CONFIG"
echo "TIMESTEPS=$TIMESTEPS"
echo "EPISODES=$EPISODES"

cat > "$PAPER_RUN_DIR/paper_run_metadata.json" <<EOF
{
  "paper_run_id": "$PAPER_RUN_ID",
  "created_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "config": "$CONFIG",
  "timesteps": $TIMESTEPS,
  "episodes": $EPISODES,
  "main_run_name": "$MAIN_RUN_NAME",
  "ablation_run_name": "$ABLATION_RUN_NAME",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || true)",
  "git_status_short": $(python - <<'PY'
import json
import subprocess

try:
    status = subprocess.run(["git", "status", "--short"], check=True, capture_output=True, text=True).stdout
except Exception:
    status = ""
print(json.dumps(status))
PY
)
}
EOF

echo "[1/5] Main six-baseline experiment"
python scripts/run_all_baselines.py \
  --config "$CONFIG" \
  --timesteps "$TIMESTEPS" \
  --episodes "$EPISODES" \
  --run-name "$MAIN_RUN_NAME"

MAIN_RUN="outputs/runs/$MAIN_RUN_NAME"
echo "$MAIN_RUN" > "$PAPER_RUN_DIR/main_run.txt"
echo "$MAIN_RUN" > outputs/paper_runs/latest_main_run.txt
echo "Main run: $MAIN_RUN"

echo "[2/5] No-physics-penalty ablation"
python scripts/run_all_baselines.py \
  --config "$CONFIG" \
  --timesteps "$TIMESTEPS" \
  --episodes "$EPISODES" \
  --run-name "$ABLATION_RUN_NAME" \
  --no-physics-penalty

ABLATION_RUN="outputs/runs/$ABLATION_RUN_NAME"
echo "$ABLATION_RUN" > "$PAPER_RUN_DIR/ablation_run.txt"
echo "$ABLATION_RUN" > outputs/paper_runs/latest_ablation_run.txt
echo "Ablation run: $ABLATION_RUN"

echo "[3/5] Main figures"
python scripts/make_figures.py \
  --config "$CONFIG" \
  --run-dir "$MAIN_RUN" \
  --figures-dir "$MAIN_FIGURES_DIR"

echo "[4/5] Ablation figures"
python scripts/make_figures.py \
  --config "$CONFIG" \
  --run-dir "$ABLATION_RUN" \
  --figures-dir "$ABLATION_FIGURES_DIR"

echo "[5/5] Paper results manifest"
bash paper_pipeline/04_write_results_manifest.sh "$MAIN_RUN" "$ABLATION_RUN" "$PAPER_RUN_DIR"

echo "Paper experiment pipeline complete."
echo "Paper run directory: $PAPER_RUN_DIR"
echo "Main run: $MAIN_RUN"
echo "Ablation run: $ABLATION_RUN"
echo "Manifest: $PAPER_RUN_DIR/PAPER_RESULTS.md"

