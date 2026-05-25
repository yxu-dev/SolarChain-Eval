# SolarChain-Eval 正式实验中断恢复指南

本文档用于处理正式实验在 Linux 服务器上运行 `paper_final_seed_20260511` 时，已经完成前两步训练，但在 `[3/9] Eval-only LLM/agentic layer on trained RL policies` 附近中断的情况。

典型场景：

```text
[1/9] Main six-baseline experiment 已完成
[2/9] No-physics-penalty ablation 已完成
[3/9] Eval-only LLM/agentic layer on trained RL policies 中断
```

此时不建议重新运行完整 `paper_pipeline/02_run_paper_experiments.sh`，因为它会重新训练 PPO/SAC/DQN。正确做法是复用已经训练好的模型，从 agentic evaluation 继续执行。

## 1. 进入仓库和环境

在新的 bash 窗口中进入 `SolarChain-Eval` 根目录：

```bash
cd /path/to/SolarChain-Eval

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SolarChain-rl
```

确认 LLM endpoint 环境变量仍然存在。如果你有单独的环境变量脚本，先 source 它，例如：

```bash
source /path/to/openai_dku.sh
```

检查 key、model 和 base url：

```bash
python - <<'PY'
import os
print("has_key:", bool(os.getenv("SOLARCHAIN_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")))
print("model:", os.getenv("SOLARCHAIN_LLM_MODEL") or os.getenv("OPENAI_MODEL"))
print("base_url:", os.getenv("SOLARCHAIN_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
PY
```

## 2. 设置恢复变量

以下变量对应 `paper_final_seed_20260511`：

```bash
CONFIG=outputs/multiseed_configs/month_2026_04_seed_20260511.yaml
PAPER_RUN_ID=paper_final_seed_20260511
PAPER_RUN_DIR=outputs/paper_final_seed_20260511
RUNS_DIR=outputs/paper_final_seed_20260511/runs

MAIN_RUN=outputs/paper_final_seed_20260511/runs/main
ABLATION_RUN=outputs/paper_final_seed_20260511/runs/no_physics_penalty
AGENTIC_RUN=outputs/paper_final_seed_20260511/runs/agentic_llm_llm
AGENTIC_ABLATION_RUN=outputs/paper_final_seed_20260511/runs/agentic_llm_llm_no_physics_penalty

MAIN_FIGURES_DIR=outputs/paper_final_seed_20260511/figures/main
ABLATION_FIGURES_DIR=outputs/paper_final_seed_20260511/figures/ablation_no_physics_penalty
AGENTIC_FIGURES_DIR=outputs/paper_final_seed_20260511/figures/agentic
AGENTIC_ABLATION_FIGURES_DIR=outputs/paper_final_seed_20260511/figures/agentic_no_physics_penalty

mkdir -p "$RUNS_DIR" "$MAIN_FIGURES_DIR" "$ABLATION_FIGURES_DIR" "$AGENTIC_FIGURES_DIR" "$AGENTIC_ABLATION_FIGURES_DIR"
```

## 3. 确认训练模型已经存在

继续前先检查 main 和 no-physics 的 PPO/SAC/DQN 模型是否存在：

```bash
test -s "$MAIN_RUN/models/ppo/ppo_model.zip"
test -s "$MAIN_RUN/models/sac/sac_model.zip"
test -s "$MAIN_RUN/models/dqn/dqn_model.zip"

test -s "$ABLATION_RUN/models/ppo/ppo_model.zip"
test -s "$ABLATION_RUN/models/sac/sac_model.zip"
test -s "$ABLATION_RUN/models/dqn/dqn_model.zip"
```

如果以上命令没有输出，表示模型文件存在，可以继续。

## 4. 处理可能存在的半成品 agentic 目录

如果 `[3/9]` 已经开始写入 `agentic_llm_llm` 目录，但中途断开，建议先把半成品目录改名保留，避免新结果和旧残留混在一起。

```bash
if [ -d "$AGENTIC_RUN" ]; then
  mv "$AGENTIC_RUN" "${AGENTIC_RUN}.partial_$(date -u +%Y%m%d_%H%M%S)"
fi

if [ -d "$AGENTIC_ABLATION_RUN" ]; then
  mv "$AGENTIC_ABLATION_RUN" "${AGENTIC_ABLATION_RUN}.partial_$(date -u +%Y%m%d_%H%M%S)"
fi
```

这一步不会删除半成品，只是改名归档。

## 5. 继续执行 `[3/9]`：LLM agentic main

```bash
python scripts/evaluate.py \
  --config "$CONFIG" \
  --policies ppo,sac,dqn \
  --episodes 30 \
  --ppo-model "$MAIN_RUN/models/ppo/ppo_model.zip" \
  --sac-model "$MAIN_RUN/models/sac/sac_model.zip" \
  --dqn-model "$MAIN_RUN/models/dqn/dqn_model.zip" \
  --output-dir "$RUNS_DIR" \
  --run-name agentic_llm_llm \
  --no-timestamp \
  --agentic-mode planner_auditor \
  --planner llm \
  --auditor llm \
  --audit-trigger event \
  --save-agentic-logs

echo "$AGENTIC_RUN" > "$PAPER_RUN_DIR/agentic_run.txt"
echo "$AGENTIC_RUN" > "$PAPER_RUN_DIR/latest_agentic_run.txt"
```

## 6. 继续执行 `[4/9]`：LLM agentic no-physics

```bash
python scripts/evaluate.py \
  --config "$CONFIG" \
  --policies ppo,sac,dqn \
  --episodes 30 \
  --ppo-model "$ABLATION_RUN/models/ppo/ppo_model.zip" \
  --sac-model "$ABLATION_RUN/models/sac/sac_model.zip" \
  --dqn-model "$ABLATION_RUN/models/dqn/dqn_model.zip" \
  --output-dir "$RUNS_DIR" \
  --run-name agentic_llm_llm_no_physics_penalty \
  --no-timestamp \
  --no-physics-penalty \
  --agentic-mode planner_auditor \
  --planner llm \
  --auditor llm \
  --audit-trigger event \
  --save-agentic-logs

echo "$AGENTIC_ABLATION_RUN" > "$PAPER_RUN_DIR/agentic_ablation_run.txt"
echo "$AGENTIC_ABLATION_RUN" > "$PAPER_RUN_DIR/latest_agentic_ablation_run.txt"
```

## 7. 重新生成四组图表

如果中断发生在 `[3/9]`，后续图表通常还没有完整生成。建议四组图表都重新生成一次：

```bash
python scripts/make_figures.py \
  --config "$CONFIG" \
  --run-dir "$MAIN_RUN" \
  --figures-dir "$MAIN_FIGURES_DIR"

python scripts/make_figures.py \
  --config "$CONFIG" \
  --run-dir "$ABLATION_RUN" \
  --figures-dir "$ABLATION_FIGURES_DIR"

python scripts/make_figures.py \
  --config "$CONFIG" \
  --run-dir "$AGENTIC_RUN" \
  --figures-dir "$AGENTIC_FIGURES_DIR"

python scripts/make_figures.py \
  --config "$CONFIG" \
  --run-dir "$AGENTIC_ABLATION_RUN" \
  --figures-dir "$AGENTIC_ABLATION_FIGURES_DIR"
```

## 8. 重写结果清单

最后重新写出 `PAPER_RESULTS.md`：

```bash
bash paper_pipeline/04_write_results_manifest.sh \
  "$MAIN_RUN" \
  "$ABLATION_RUN" \
  "$PAPER_RUN_DIR" \
  "$AGENTIC_RUN" \
  "$AGENTIC_ABLATION_RUN"
```

## 9. 验收检查

恢复完成后检查关键文件：

```bash
test -s "$PAPER_RUN_DIR/PAPER_RESULTS.md"
test -s "$AGENTIC_RUN/summary.json"
test -s "$AGENTIC_RUN/agentic_logs.jsonl"
test -s "$AGENTIC_ABLATION_RUN/summary.json"
test -s "$AGENTIC_ABLATION_RUN/agentic_logs.jsonl"

find "$PAPER_RUN_DIR/figures" -name "*.png" -type f
```

如果这些文件都存在，说明 `paper_final_seed_20260511` 已经从 `[3/9]` 中断处恢复完成。

## 10. 注意事项

- 不要重新运行完整 `paper_pipeline/02_run_paper_experiments.sh`，否则会重新训练。
- 不要把恢复输出写到新的 `PAPER_RUN_ID`，否则会和原始 seed 结果分裂。
- 如果 LLM endpoint 在恢复过程中再次中断，只需要重复第 4 到第 9 节。
- 如果只完成了 main agentic，没有完成 no-physics agentic，可以从第 6 节继续。
- 如果已经完成 agentic evaluation，只是图表或 manifest 缺失，可以只执行第 7 和第 8 节。
