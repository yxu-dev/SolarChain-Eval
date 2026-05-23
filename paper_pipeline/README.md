# SolarChain-Eval 论文实验 Pipeline

本目录保存计划投稿 **KDD Workshop on Evaluation and Trustworthiness of Agentic AI** 的实验流水线说明。

请从 `SolarChain-Eval` 仓库根目录运行命令。默认 conda 环境为：

```bash
SolarChain-rl
```

本文实验包含两层：

- **核心 benchmark**：物理约束下的去中心化能源市场经济治理 agent 评估。
- **eval-only LLM agentic layer**：在训练好的 RL policy 输出 action 后，加入 LLM Planner + LLM Auditor；LLM 只参与评估，不参与 PPO/SAC/DQN 训练。

## 1. 数据与配置

主实验使用 2026 年 4 月五城市月度数据：

- 配置：`configs/month_2026_04.yaml`
- 数据目录：`data/datasets_2026_04_month`
- 时间窗口：`[2026-04-01, 2026-05-01)`
- 小时数：`720`
- 城市：Beijing、Shanghai、Chengdu、Shenzhen、Hangzhou
- episode：`episode_steps=24`，每次 reset 从整月中随机采样一个完整日

数据生成命令：

```bash
python scripts/generate_monthly_datasets.py \
  --start-date 2026-04-01 \
  --end-date 2026-05-01 \
  --output-dir data/datasets_2026_04_month \
  --seed 20260511
```

预期数据形状：

- `urban_energy_nodes.csv`：50 行
- `spatiotemporal_generation.csv`：36000 行
- `market_liquidity.csv`：720 行
- generation timestamps：720 个唯一小时

## 2. 环境准备

Linux 首次准备：

```bash
bash paper_pipeline/00_setup_linux.sh
```

等价手动流程：

```bash
conda create -n SolarChain-rl python=3.10 -y
conda activate SolarChain-rl
python -m pip install -r requirements.txt
python -m pip install -e .
```

LLM structured output 依赖已包含在 `requirements.txt`：

- `openai>=1.0`
- `pydantic>=2.0`

## 3. Windows Smoke

Windows 机器请使用 PowerShell 版完整 smoke 指南：

```text
paper_pipeline/WINDOWS_SMOKE_GUIDE.md
```

该 smoke 会使用 tiny timesteps 完成完整的训练、baseline evaluate、LLM agentic evaluate、no-physics ablation、四组图表生成，并生成和正式实验同构的输出目录。

## 4. Linux 正式实验

推荐最终实验命令：

```bash
PAPER_RUN_ID=paper_final TIMESTEPS=300000 EPISODES=30 bash paper_pipeline/02_run_paper_experiments.sh
```

默认参数：

- `CONFIG=configs/month_2026_04.yaml`
- `DATA_DIR=data/datasets_2026_04_month`
- `TIMESTEPS=100000`
- `EPISODES=10`
- `RUN_AGENTIC=1`
- `AGENTIC_POLICIES=ppo,sac,dqn`
- `AGENTIC_PLANNER=llm`
- `AGENTIC_AUDITOR=llm`
- `AGENTIC_AUDIT_TRIGGER=event`

正式脚本会执行：

1. 生成或验证 2026-04 五城市月度数据。
2. 训练并评估六个 baseline：PPO、SAC、DQN、Static 1:3、Random、Myopic Greedy。
3. 用 `--no-physics-penalty` 训练并评估 reward-misspecification 消融。
4. 复用主实验训练好的 PPO/SAC/DQN 模型，运行 eval-only LLM Planner/Auditor。
5. 复用 no-physics 训练好的 PPO/SAC/DQN 模型，运行 no-physics agentic eval。
6. 为四个 run 生成图表。
7. 写出 `dataset_summary.json`、`paper_run_metadata.json` 和 `PAPER_RESULTS.md`。

## 5. LLM 配置

代码优先读取 SolarChain 专用环境变量：

```bash
export SOLARCHAIN_LLM_API_KEY="..."
export SOLARCHAIN_LLM_BASE_URL="https://..."
export SOLARCHAIN_LLM_MODEL="..."
```

若未设置，则回退读取 OpenAI 标准变量：

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://..."
export OPENAI_MODEL="..."
```

如果启用 `--planner llm` 或 `--auditor llm`，必须配置真实 API key 和模型名。若 API key 缺失、模型名缺失、endpoint 不可联通或 structured output 调用失败，评估会直接报错停止，不会生成替代结果。

## 6. 输出结构

每次完整论文实验使用一个统一目录：

```text
outputs/<PAPER_RUN_ID>/
```

其中各个 train + eval 子 run 写入：

```text
outputs/<PAPER_RUN_ID>/runs/main/
outputs/<PAPER_RUN_ID>/runs/no_physics_penalty/
outputs/<PAPER_RUN_ID>/runs/agentic_llm_llm/
outputs/<PAPER_RUN_ID>/runs/agentic_llm_llm_no_physics_penalty/
```

baseline / ablation 子 run 包含：

- `metrics.csv`
- `summary.json`
- `actions.csv`
- `city_hour_policy.csv`
- `config_snapshot.json`
- `run_metadata.json`
- `models/ppo/ppo_model.zip`
- `models/sac/sac_model.zip`
- `models/dqn/dqn_model.zip`

agentic run 额外包含：

- `agentic_logs.jsonl`
- `summary.json` 中的 agentic metrics
- `actions.csv` 中的 agentic action metadata

图表和论文清单也保存在同一个目录下：

```text
outputs/<PAPER_RUN_ID>/
```

包含：

- `dataset_summary.json`
- `paper_run_metadata.json`
- `main_run.txt`
- `ablation_run.txt`
- `agentic_run.txt`
- `agentic_ablation_run.txt`
- `PAPER_RESULTS.md`
- `figures/main/*.png`
- `figures/ablation_no_physics_penalty/*.png`
- `figures/agentic/*.png`
- `figures/agentic_no_physics_penalty/*.png`

## 7. 论文表格指标

主表建议从 `summary.json` 抽取：

- `cumulative_reward`
- `physics_violation_rate`
- `max_drawdown`
- `action_jitter`
- `slippage_reduction_vs_static`
- `spatial_fairness_index`
- `artificial_liquidity_MWh`

agentic 表建议额外报告：

- `plan_validity_rate`
- `audit_call_rate`
- `revision_rate`
- `action_modification_rate`
- `avg_action_delta_from_auditor`
- `llm_failure_count`

## 8. 图表

正式 pipeline 生成四组图：

- `figures/main/learning_curves.png`
- `figures/main/safety_utility_frontier.png`
- `figures/main/city_hour_liquidity_heatmap.png`
- `figures/ablation_no_physics_penalty/*.png`
- `figures/agentic/*.png`
- `figures/agentic_no_physics_penalty/*.png`

图表含义：

- learning curves：不同 policy 的 reward vs. episode。
- safety-utility frontier：physics violation rate vs. cumulative reward。
- city-hour liquidity heatmap：不同城市与小时下的 liquidity policy。

## 9. 注意事项

- 论文主证据应来自 `configs/month_2026_04.yaml`，不要使用 `configs/default.yaml`。
- `SolarSave` 只作为只读参考，实验产物全部保存在 `SolarChain-Eval`。
- 如果 `data/cache/` 为空，首次生成月度数据可能需要网络访问 Open-Meteo。
- LLM 使用 structured outputs；不要把 LLM 自然语言输出再用宽松 JSON 抽取当作正式结果。
- 如果 SB3 在某台机器上不稳定，保留成功 run 并在论文中说明限制。
