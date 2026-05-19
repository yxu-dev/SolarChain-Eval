#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MAIN_RUN="${1:-$(cat outputs/paper_runs/latest_main_run.txt)}"
ABLATION_RUN="${2:-$(cat outputs/paper_runs/latest_ablation_run.txt)}"
PAPER_RUN_DIR="${3:-outputs/paper_runs/manual_$(date -u +%Y%m%d_%H%M%S)}"
MANIFEST="$PAPER_RUN_DIR/PAPER_RESULTS.md"

mkdir -p "$PAPER_RUN_DIR"

cat > "$MANIFEST" <<EOF
# SolarChain-Eval Paper Results

Generated at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")

## Main Experiment

Run directory:

\`\`\`text
$MAIN_RUN
\`\`\`

Primary data files:

- \`$MAIN_RUN/summary.json\`
- \`$MAIN_RUN/metrics.csv\`
- \`$MAIN_RUN/actions.csv\`
- \`$MAIN_RUN/city_hour_policy.csv\`
- \`$MAIN_RUN/config_snapshot.json\`
- \`$MAIN_RUN/run_metadata.json\`

Trained models:

- \`$MAIN_RUN/models/ppo/ppo_model.zip\`
- \`$MAIN_RUN/models/sac/sac_model.zip\`
- \`$MAIN_RUN/models/dqn/dqn_model.zip\`

Main figures:

- \`$PAPER_RUN_DIR/figures/main/learning_curves.png\`
- \`$PAPER_RUN_DIR/figures/main/safety_utility_frontier.png\`
- \`$PAPER_RUN_DIR/figures/main/city_hour_liquidity_heatmap.png\`

## No-Physics-Penalty Ablation

Run directory:

\`\`\`text
$ABLATION_RUN
\`\`\`

Primary data files:

- \`$ABLATION_RUN/summary.json\`
- \`$ABLATION_RUN/metrics.csv\`
- \`$ABLATION_RUN/actions.csv\`
- \`$ABLATION_RUN/city_hour_policy.csv\`
- \`$ABLATION_RUN/config_snapshot.json\`
- \`$ABLATION_RUN/run_metadata.json\`

Ablation figures:

- \`$PAPER_RUN_DIR/figures/ablation_no_physics_penalty/learning_curves.png\`
- \`$PAPER_RUN_DIR/figures/ablation_no_physics_penalty/safety_utility_frontier.png\`
- \`$PAPER_RUN_DIR/figures/ablation_no_physics_penalty/city_hour_liquidity_heatmap.png\`

## Metrics To Report

Batch-level metadata:

- \`$PAPER_RUN_DIR/paper_run_metadata.json\`
- \`$PAPER_RUN_DIR/PAPER_RESULTS.md\`

Use \`summary.json\` for the main comparison table:

- cumulative_reward
- physics_violation_rate
- max_drawdown
- action_jitter
- slippage_reduction_vs_static
- spatial_fairness_index
- artificial_liquidity_MWh

Use the ablation \`summary.json\` to compare whether removing the \`P_max\` penalty increases \`physics_violation_rate\` and \`artificial_liquidity_MWh\`.

## Figure Mapping

- Learning curves: reward vs. episode for PPO, SAC, DQN, Static 1:3, Random, Myopic Greedy.
- Safety-utility frontier: physics violation rate vs. cumulative reward.
- Spatio-temporal heatmap: RL liquidity split by city/hour compared to Static 1:3.

EOF

echo "Wrote $MANIFEST"

