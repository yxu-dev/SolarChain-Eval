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
# SolarChain-Eval 论文实验结果清单

生成时间 UTC：$(date -u +"%Y-%m-%dT%H:%M:%SZ")

本文档对应计划投稿 **KDD Workshop on Evaluation and Trustworthiness of Agentic AI** 的实验流水线。

## 统一输出目录

本次实验所有产物集中在：

\`\`\`text
$PAPER_RUN_DIR
\`\`\`

核心配置：

\`\`\`text
configs/month_2026_04.yaml
\`\`\`

核心数据：

\`\`\`text
data/datasets_2026_04_month
\`\`\`

数据覆盖 Beijing、Shanghai、Chengdu、Shenzhen、Hangzhou 五个城市，时间窗口为 \`[2026-04-01, 2026-05-01)\`，共 720 个小时。训练和评估均使用 \`episode_steps=24\`，每次环境 reset 采样一个完整日。

## 元数据

- \`$PAPER_RUN_DIR/dataset_summary.json\`
- \`$PAPER_RUN_DIR/paper_run_metadata.json\`
- \`$PAPER_RUN_DIR/PAPER_RESULTS.md\`

## 主实验

运行目录：

\`\`\`text
$MAIN_RUN
\`\`\`

主要文件：

- \`$MAIN_RUN/summary.json\`
- \`$MAIN_RUN/metrics.csv\`
- \`$MAIN_RUN/actions.csv\`
- \`$MAIN_RUN/city_hour_policy.csv\`
- \`$MAIN_RUN/config_snapshot.json\`
- \`$MAIN_RUN/run_metadata.json\`
- \`$MAIN_RUN/models/ppo/ppo_model.zip\`
- \`$MAIN_RUN/models/sac/sac_model.zip\`
- \`$MAIN_RUN/models/dqn/dqn_model.zip\`

图表：

- \`$PAPER_RUN_DIR/figures/main/learning_curves.png\`
- \`$PAPER_RUN_DIR/figures/main/safety_utility_frontier.png\`
- \`$PAPER_RUN_DIR/figures/main/city_hour_liquidity_heatmap.png\`

## No-Physics-Penalty 消融

运行目录：

\`\`\`text
$ABLATION_RUN
\`\`\`

主要文件：

- \`$ABLATION_RUN/summary.json\`
- \`$ABLATION_RUN/metrics.csv\`
- \`$ABLATION_RUN/actions.csv\`
- \`$ABLATION_RUN/city_hour_policy.csv\`
- \`$ABLATION_RUN/config_snapshot.json\`
- \`$ABLATION_RUN/run_metadata.json\`
- \`$ABLATION_RUN/models/ppo/ppo_model.zip\`
- \`$ABLATION_RUN/models/sac/sac_model.zip\`
- \`$ABLATION_RUN/models/dqn/dqn_model.zip\`

图表：

- \`$PAPER_RUN_DIR/figures/ablation_no_physics_penalty/learning_curves.png\`
- \`$PAPER_RUN_DIR/figures/ablation_no_physics_penalty/safety_utility_frontier.png\`
- \`$PAPER_RUN_DIR/figures/ablation_no_physics_penalty/city_hour_liquidity_heatmap.png\`

## Eval-Only LLM Agentic Layer

Agentic evaluation 复用已训练 RL 模型，在 policy 输出 action 后插入 LLM Planner + LLM Auditor。LLM 只参与 evaluation，不参与 PPO/SAC/DQN training。

Agentic 主实验目录：

\`\`\`text
${AGENTIC_RUN:-未生成}
\`\`\`

Agentic no-physics 目录：

\`\`\`text
${AGENTIC_ABLATION_RUN:-未生成}
\`\`\`

若已生成，应包含：

- \`${AGENTIC_RUN:-<agentic_run>}/summary.json\`
- \`${AGENTIC_RUN:-<agentic_run>}/metrics.csv\`
- \`${AGENTIC_RUN:-<agentic_run>}/actions.csv\`
- \`${AGENTIC_RUN:-<agentic_run>}/agentic_logs.jsonl\`
- \`${AGENTIC_ABLATION_RUN:-<agentic_ablation_run>}/summary.json\`
- \`${AGENTIC_ABLATION_RUN:-<agentic_ablation_run>}/agentic_logs.jsonl\`
- \`$PAPER_RUN_DIR/figures/agentic/*.png\`
- \`$PAPER_RUN_DIR/figures/agentic_no_physics_penalty/*.png\`

## 论文指标

主表指标：

- cumulative_reward
- physics_violation_rate
- max_drawdown
- action_jitter
- slippage_reduction_vs_static
- spatial_fairness_index
- artificial_liquidity_MWh

agentic 指标：

- plan_validity_rate
- audit_call_rate
- revision_rate
- action_modification_rate
- avg_action_delta_from_auditor
- llm_failure_count

## 图表对应关系

- \`learning_curves.png\`：各 policy 的 reward vs. episode。
- \`safety_utility_frontier.png\`：physics violation rate vs. cumulative reward。
- \`city_hour_liquidity_heatmap.png\`：城市与小时维度下的 liquidity policy。

EOF

echo "Wrote $MANIFEST"
