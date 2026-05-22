# SolarChain-Eval Linux 正式实验实施计划

本文档面向 Linux 服务器上的正式实验，用于计划投稿 **KDD Workshop on Evaluation and Trustworthiness of Agentic AI**。

目标是产出可直接用于论文表格、图表和实验记录的完整结果：

```text
baseline main
baseline no-physics-penalty
LLM agentic main
LLM agentic no-physics-penalty
```

## 1. 实验问题

本文评估的不是“给 SolarChain 加 RL”本身，而是一个更通用的问题：

> 自主经济治理 agent 能否在没有人工干预的情况下，安全管理一个受物理约束的去中心化能源市场？

核心设定：

- RL policy 控制三个宏观经济治理变量：`reward_ratio`、`liquidity_ratio`、`burn_rate`。
- 物理约束来自 PV 最大可发电能力、FDIA 检测和 verified generation。
- LLM Planner/Auditor 只在 evaluation 阶段使用，不参与 PPO/SAC/DQN training。

agentic evaluation 流程：

```text
trained RL policy
  -> proposed action
  -> LLM Planner action bounds
  -> LLM Auditor event-triggered review
  -> final action
  -> env.step()
```

## 2. 服务器环境准备

从仓库根目录运行：

```bash
cd /path/to/SolarChain-Eval
conda create -n SolarChain-rl python=3.10 -y
conda activate SolarChain-rl
python -m pip install -r requirements.txt
python -m pip install -e .
```

检查关键依赖：

```bash
python - <<'PY'
import sys
import gymnasium
import openai
import pydantic
import stable_baselines3

print("Python:", sys.executable)
print("Gymnasium:", gymnasium.__version__)
print("OpenAI:", openai.__version__)
print("Pydantic:", pydantic.__version__)
print("Stable-Baselines3:", stable_baselines3.__version__)
PY
```

## 3. LLM Endpoint 配置

正式 LLM 实验必须使用真实 OpenAI-compatible endpoint。不要在代码或 md 中写入明文 key。

推荐使用 SolarChain 专用变量：

```bash
export SOLARCHAIN_LLM_API_KEY="..."
export SOLARCHAIN_LLM_BASE_URL="https://..."
export SOLARCHAIN_LLM_MODEL="..."
```

也可以使用 OpenAI 标准变量：

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://..."
export OPENAI_MODEL="..."
```

运行正式实验前确认：

```bash
python - <<'PY'
import os
print("has_key:", bool(os.getenv("SOLARCHAIN_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")))
print("model:", os.getenv("SOLARCHAIN_LLM_MODEL") or os.getenv("OPENAI_MODEL"))
print("base_url:", os.getenv("SOLARCHAIN_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
PY
```

如果 key 缺失，LLM mode 会自动使用 mock client，并在 `summary.json` 写出 `mock_llm_used=true`。这只能用于 smoke，不能用于最终论文 LLM 结果。

## 4. 数据准备

主实验数据：

```bash
python scripts/generate_monthly_datasets.py \
  --start-date 2026-04-01 \
  --end-date 2026-05-01 \
  --output-dir data/datasets_2026_04_month \
  --seed 20260511
```

验证数据：

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

data_dir = Path("data/datasets_2026_04_month")
nodes = pd.read_csv(data_dir / "urban_energy_nodes.csv")
generation = pd.read_csv(data_dir / "spatiotemporal_generation.csv")
market = pd.read_csv(data_dir / "market_liquidity.csv")

assert len(nodes) == 50
assert len(generation) == 36000
assert generation["timestamp"].nunique() == 720
assert len(market) == 720
assert sorted(generation["city"].unique()) == sorted(["Beijing", "Shanghai", "Chengdu", "Shenzhen", "Hangzhou"])
print("dataset ok")
PY
```

## 5. 正式实验命令

推荐最终 run：

```bash
PAPER_RUN_ID=paper_final \
TIMESTEPS=300000 \
EPISODES=30 \
RUN_AGENTIC=1 \
AGENTIC_POLICIES=ppo,sac,dqn \
AGENTIC_PLANNER=llm \
AGENTIC_AUDITOR=llm \
AGENTIC_AUDIT_TRIGGER=event \
bash paper_pipeline/02_run_paper_experiments.sh
```

如果要先做低成本正式结构检查：

```bash
PAPER_RUN_ID=paper_debug \
TIMESTEPS=2048 \
EPISODES=2 \
AGENTIC_PLANNER=rule \
AGENTIC_AUDITOR=rule \
bash paper_pipeline/02_run_paper_experiments.sh
```

## 6. Pipeline 实际执行内容

`02_run_paper_experiments.sh` 会执行：

1. 检查或生成 2026-04 五城市月度数据。
2. 写出 `dataset_summary.json`。
3. 主实验训练 PPO/SAC/DQN，并评估 Static、Random、Myopic、PPO、SAC、DQN。
4. no-physics-penalty 消融训练 PPO/SAC/DQN，并评估同一组 baseline。
5. 使用主实验训练好的 PPO/SAC/DQN 模型，运行 eval-only LLM Planner/Auditor。
6. 使用 no-physics 消融训练好的 PPO/SAC/DQN 模型，运行 no-physics agentic eval。
7. 为四个 run 生成图表。
8. 写出 `PAPER_RESULTS.md`。

训练命令不会调用 LLM。只有第 5 和第 6 步的 `scripts/evaluate.py --agentic-mode planner_auditor` 会调用 LLM。

## 7. 预期输出

设：

```text
PAPER_RUN_ID=paper_final
```

则 paper batch 目录为：

```text
outputs/paper_runs/paper_final/
```

预期包含：

```text
dataset_summary.json
paper_run_metadata.json
main_run.txt
ablation_run.txt
agentic_run.txt
agentic_ablation_run.txt
PAPER_RESULTS.md
figures/main/learning_curves.png
figures/main/safety_utility_frontier.png
figures/main/city_hour_liquidity_heatmap.png
figures/ablation_no_physics_penalty/learning_curves.png
figures/ablation_no_physics_penalty/safety_utility_frontier.png
figures/ablation_no_physics_penalty/city_hour_liquidity_heatmap.png
figures/agentic/learning_curves.png
figures/agentic/safety_utility_frontier.png
figures/agentic/city_hour_liquidity_heatmap.png
figures/agentic_no_physics_penalty/learning_curves.png
figures/agentic_no_physics_penalty/safety_utility_frontier.png
figures/agentic_no_physics_penalty/city_hour_liquidity_heatmap.png
```

主实验 run：

```text
outputs/runs/paper_final_main/
```

预期包含：

```text
metrics.csv
summary.json
actions.csv
city_hour_policy.csv
config_snapshot.json
run_metadata.json
models/ppo/ppo_model.zip
models/sac/sac_model.zip
models/dqn/dqn_model.zip
```

no-physics run：

```text
outputs/runs/paper_final_no_physics_penalty/
```

预期包含同样文件，并且 `run_metadata.json` 中 `no_physics_penalty=true`。

agentic run：

```text
outputs/runs/paper_final_agentic_llm_llm/
```

预期包含：

```text
metrics.csv
summary.json
actions.csv
city_hour_policy.csv
config_snapshot.json
run_metadata.json
agentic_logs.jsonl
```

agentic no-physics run：

```text
outputs/runs/paper_final_agentic_llm_llm_no_physics_penalty/
```

预期包含同样 agentic 文件。

## 8. 论文中使用的核心指标

baseline 主表：

- `cumulative_reward`
- `physics_violation_rate`
- `max_drawdown`
- `max_token_drawdown`
- `action_jitter`
- `mean_slippage`
- `slippage_reduction_vs_static`
- `spatial_fairness_index`
- `artificial_liquidity_MWh`

agentic 表：

- `plan_validity_rate`
- `audit_call_rate`
- `revision_rate`
- `action_modification_rate`
- `avg_action_delta_from_auditor`
- `llm_failure_count`
- `mock_llm_used`

安全消融：

- 对比 main vs. no-physics 的 `physics_violation_rate`。
- 对比 main vs. no-physics 的 `artificial_liquidity_MWh`。
- 对比 RL-only vs. RL + Planner/Auditor 在 no-physics 条件下是否降低 unsafe backing。

## 9. 验收标准

正式实验完成后应满足：

- `outputs/paper_runs/<paper_run_id>/PAPER_RESULTS.md` 存在。
- 四组 figure 目录都存在 PNG。
- main 和 ablation run 都有 PPO/SAC/DQN 模型。
- agentic runs 都有 `agentic_logs.jsonl`。
- 真实 LLM run 的 `summary.json` 中 `mock_llm_used=false`。
- `llm_failure_count` 应接近 0；如果非 0，需要检查 endpoint、schema 兼容性或 refusal。

## 10. 常见变体

只跑 rule agentic 对照：

```bash
AGENTIC_PLANNER=rule AGENTIC_AUDITOR=rule bash paper_pipeline/02_run_paper_experiments.sh
```

跳过 agentic：

```bash
RUN_AGENTIC=0 bash paper_pipeline/02_run_paper_experiments.sh
```

缩短调试：

```bash
PAPER_RUN_ID=debug TIMESTEPS=2048 EPISODES=2 bash paper_pipeline/02_run_paper_experiments.sh
```
