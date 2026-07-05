#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PAPER_RUN_DIR="${3:-outputs/manual_$(date -u +%Y%m%d_%H%M%S)}"
MAIN_RUN="${1:-$(cat "$PAPER_RUN_DIR/latest_main_run.txt")}"
ABLATION_RUN="${2:-$(cat "$PAPER_RUN_DIR/latest_ablation_run.txt")}"
AGENTIC_RUN="${4:-}"
AGENTIC_ABLATION_RUN="${5:-}"
MANIFEST="$PAPER_RUN_DIR/PAPER_RESULTS.md"

mkdir -p "$PAPER_RUN_DIR"

cat > "$MANIFEST" <<EOF
# SolarChain-Eval Paper Results

Generated at UTC: $(date -u +"%Y-%m-%dT%H:%M:%SZ")

This manifest summarizes the experiment artifacts for the accepted SolarChain-Eval paper.

Paper: https://openreview.net/forum?id=XcWTS5iVvY

Code: https://github.com/yxu-dev/SolarChain-Eval

## Unified Output Directory

All artifacts for this experiment are stored under:

\`\`\`text
$PAPER_RUN_DIR
\`\`\`

Primary configuration:

\`\`\`text
configs/month_2026_04.yaml
\`\`\`

Primary dataset:

\`\`\`text
data/datasets_2026_04_month
\`\`\`

The dataset covers five cities, Beijing, Shanghai, Chengdu, Shenzhen, and Hangzhou, over \`[2026-04-01, 2026-05-01)\` with 720 hourly timestamps. Training and evaluation use \`episode_steps=24\`; each environment reset samples one complete day.

## Metadata

- \`$PAPER_RUN_DIR/dataset_summary.json\`
- \`$PAPER_RUN_DIR/paper_run_metadata.json\`
- \`$PAPER_RUN_DIR/PAPER_RESULTS.md\`

## Main Experiment

Run directory:

\`\`\`text
$MAIN_RUN
\`\`\`

Primary files:

- \`$MAIN_RUN/summary.json\`
- \`$MAIN_RUN/metrics.csv\`
- \`$MAIN_RUN/actions.csv\`
- \`$MAIN_RUN/city_hour_policy.csv\`
- \`$MAIN_RUN/config_snapshot.json\`
- \`$MAIN_RUN/run_metadata.json\`
- \`$MAIN_RUN/models/ppo/ppo_model.zip\`
- \`$MAIN_RUN/models/sac/sac_model.zip\`
- \`$MAIN_RUN/models/dqn/dqn_model.zip\`

Figures:

- \`$PAPER_RUN_DIR/figures/main/learning_curves.png\`
- \`$PAPER_RUN_DIR/figures/main/safety_utility_frontier.png\`
- \`$PAPER_RUN_DIR/figures/main/city_hour_liquidity_heatmap.png\`

## No-Physics-Penalty Ablation

Run directory:

\`\`\`text
$ABLATION_RUN
\`\`\`

Primary files:

- \`$ABLATION_RUN/summary.json\`
- \`$ABLATION_RUN/metrics.csv\`
- \`$ABLATION_RUN/actions.csv\`
- \`$ABLATION_RUN/city_hour_policy.csv\`
- \`$ABLATION_RUN/config_snapshot.json\`
- \`$ABLATION_RUN/run_metadata.json\`
- \`$ABLATION_RUN/models/ppo/ppo_model.zip\`
- \`$ABLATION_RUN/models/sac/sac_model.zip\`
- \`$ABLATION_RUN/models/dqn/dqn_model.zip\`

Figures:

- \`$PAPER_RUN_DIR/figures/ablation_no_physics_penalty/learning_curves.png\`
- \`$PAPER_RUN_DIR/figures/ablation_no_physics_penalty/safety_utility_frontier.png\`
- \`$PAPER_RUN_DIR/figures/ablation_no_physics_penalty/city_hour_liquidity_heatmap.png\`

## Eval-Only LLM Agentic Layer

Agentic evaluation reuses trained RL models and inserts an LLM Planner + LLM Auditor between policy action prediction and \`env.step()\`. The LLM is used only during evaluation and never during PPO/SAC/DQN training.

The planner audit policy is validated after structured parsing. The gap field is \`(verified_mwh - demand_mwh) / demand_mwh\`; negative values indicate supply shortfall. \`force_audit_if_gap_below\` is clipped to \`[-1.0, 0.0]\`, with a default safe value of \`-0.1\`. The event-triggered auditor also records an episode audit budget, target audit rate, and cooldown. Hard safety triggers may bypass budget and cooldown.

Agentic main run directory:

\`\`\`text
${AGENTIC_RUN:-not generated}
\`\`\`

Agentic no-physics run directory:

\`\`\`text
${AGENTIC_ABLATION_RUN:-not generated}
\`\`\`

Expected files, when generated:

- \`${AGENTIC_RUN:-<agentic_run>}/summary.json\`
- \`${AGENTIC_RUN:-<agentic_run>}/metrics.csv\`
- \`${AGENTIC_RUN:-<agentic_run>}/actions.csv\`
- \`${AGENTIC_RUN:-<agentic_run>}/agentic_logs.jsonl\`
- \`${AGENTIC_ABLATION_RUN:-<agentic_ablation_run>}/summary.json\`
- \`${AGENTIC_ABLATION_RUN:-<agentic_ablation_run>}/agentic_logs.jsonl\`
- \`$PAPER_RUN_DIR/figures/agentic/*.png\`
- \`$PAPER_RUN_DIR/figures/agentic_no_physics_penalty/*.png\`

## Metrics To Report

Main table metrics:

- cumulative_reward
- physics_violation_rate
- max_drawdown
- action_jitter
- slippage_reduction_vs_static
- spatial_fairness_index
- artificial_liquidity_MWh

Agentic metrics:

- plan_validity_rate
- audit_call_rate
- revision_rate
- action_modification_rate
- avg_action_delta_from_auditor
- llm_failure_count
- audit_budget_per_episode
- target_audit_rate
- audit_cooldown_steps

## Figure Mapping

- \`learning_curves.png\`: reward vs. episode for each policy.
- \`safety_utility_frontier.png\`: physics violation rate vs. cumulative reward.
- \`city_hour_liquidity_heatmap.png\`: liquidity policy by city and hour.

EOF

echo "Wrote $MANIFEST"
