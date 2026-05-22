# Windows 完整 Smoke 流程

本文档用于在当前 Windows 设备上跑一个**完整但 tiny** 的论文实验流程。

目标是产出和 Linux 正式实验同构的数据与图表：

```text
baseline main
baseline no-physics-penalty
LLM agentic main
LLM agentic no-physics-penalty
四组 figures
paper_run_metadata.json
dataset_summary.json
PAPER_RESULTS.md
```

区别仅在于训练步数很小，结果只用于工程 smoke，不用于论文结论。

## 1. 进入仓库与环境

在 PowerShell 中运行：

```powershell
cd D:\Documents\SolarChain\SolarChain-Eval
conda activate SolarChain-rl
```

安装或刷新依赖：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

确认依赖：

```powershell
python -c "import openai, pydantic, stable_baselines3; print(openai.__version__); print(pydantic.__version__); print(stable_baselines3.__version__)"
```

## 2. 配置真实 LLM Key

本 smoke 需要测试真实 LLM 配置路径。推荐加载仓库外层的本地 PowerShell env 文件：

```powershell
. ..\openai_dku.ps1
```

或者手动设置：

```powershell
$env:OPENAI_API_KEY="你的 key"
$env:OPENAI_BASE_URL="你的 OpenAI-compatible endpoint"
$env:OPENAI_MODEL="你的模型名"
```

也可以使用 SolarChain 专用变量：

```powershell
$env:SOLARCHAIN_LLM_API_KEY="你的 key"
$env:SOLARCHAIN_LLM_BASE_URL="你的 OpenAI-compatible endpoint"
$env:SOLARCHAIN_LLM_MODEL="你的模型名"
```

确认配置：

```powershell
python -c "import os; print(bool(os.getenv('SOLARCHAIN_LLM_API_KEY') or os.getenv('OPENAI_API_KEY'))); print(os.getenv('SOLARCHAIN_LLM_MODEL') or os.getenv('OPENAI_MODEL')); print(os.getenv('SOLARCHAIN_LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL'))"
```

注意：不要把明文 key 写入 md、py 或提交到 Git。

## 3. 设置 Smoke Run ID

```powershell
$RunId = "win_smoke_tiny"
$PaperDir = "outputs\paper_runs\$RunId"
$MainRun = "outputs\runs\${RunId}_main"
$AblationRun = "outputs\runs\${RunId}_no_physics_penalty"
$AgenticRun = "outputs\runs\${RunId}_agentic_llm_llm"
$AgenticAblationRun = "outputs\runs\${RunId}_agentic_llm_llm_no_physics_penalty"

New-Item -ItemType Directory -Force $PaperDir | Out-Null
New-Item -ItemType Directory -Force "$PaperDir\figures\main" | Out-Null
New-Item -ItemType Directory -Force "$PaperDir\figures\ablation_no_physics_penalty" | Out-Null
New-Item -ItemType Directory -Force "$PaperDir\figures\agentic" | Out-Null
New-Item -ItemType Directory -Force "$PaperDir\figures\agentic_no_physics_penalty" | Out-Null
```

## 4. 生成或验证月度数据

```powershell
python scripts\generate_monthly_datasets.py `
  --start-date 2026-04-01 `
  --end-date 2026-05-01 `
  --output-dir data\datasets_2026_04_month `
  --seed 20260511
```

写出 `dataset_summary.json`：

```powershell
python -c "import json, pandas as pd; from pathlib import Path; d=Path('data/datasets_2026_04_month'); n=pd.read_csv(d/'urban_energy_nodes.csv'); g=pd.read_csv(d/'spatiotemporal_generation.csv'); m=pd.read_csv(d/'market_liquidity.csv'); t=pd.read_csv(d/'p2p_trades.csv'); s={'data_dir':str(d),'start_date':'2026-04-01','end_date_exclusive':'2026-05-01','cities':sorted(g['city'].unique().tolist()),'node_rows':len(n),'generation_rows':len(g),'unique_timestamps':g['timestamp'].nunique(),'market_rows':len(m),'trade_rows':len(t),'fdia_rows':int(g['fdia_detected'].astype(bool).sum())}; assert s['node_rows']==50 and s['generation_rows']==36000 and s['unique_timestamps']==720 and s['market_rows']==720; Path(r'$PaperDir').mkdir(parents=True, exist_ok=True); (Path(r'$PaperDir')/'dataset_summary.json').write_text(json.dumps(s, indent=2), encoding='utf-8'); print(json.dumps(s, indent=2))"
```

## 5. 运行测试

```powershell
python -m pytest -q
```

预期：全部测试通过。

## 6. Tiny 主实验：train + evaluate

这个命令会 tiny 训练 PPO/SAC/DQN，并同时评估 Static、Random、Myopic、PPO、SAC、DQN。

```powershell
python scripts\run_all_baselines.py `
  --config configs\month_2026_04.yaml `
  --timesteps 8 `
  --episodes 1 `
  --run-name "${RunId}_main"
```

记录 run 路径：

```powershell
Set-Content -Path "$PaperDir\main_run.txt" -Value $MainRun
```

预期模型：

```powershell
Get-ChildItem "$MainRun\models" -Recurse
```

应看到：

- `models\ppo\ppo_model.zip`
- `models\sac\sac_model.zip`
- `models\dqn\dqn_model.zip`

## 7. Tiny No-Physics 消融：train + evaluate

```powershell
python scripts\run_all_baselines.py `
  --config configs\month_2026_04.yaml `
  --timesteps 8 `
  --episodes 1 `
  --run-name "${RunId}_no_physics_penalty" `
  --no-physics-penalty
```

```powershell
Set-Content -Path "$PaperDir\ablation_run.txt" -Value $AblationRun
```

## 8. 真实 LLM Agentic Evaluation

复用主实验 tiny 模型，运行 eval-only LLM Planner/Auditor：

```powershell
python scripts\evaluate.py `
  --config configs\month_2026_04.yaml `
  --policies "ppo,sac,dqn" `
  --episodes 1 `
  --ppo-model "$MainRun\models\ppo\ppo_model.zip" `
  --sac-model "$MainRun\models\sac\sac_model.zip" `
  --dqn-model "$MainRun\models\dqn\dqn_model.zip" `
  --run-name "${RunId}_agentic_llm_llm" `
  --no-timestamp `
  --agentic-mode planner_auditor `
  --planner llm `
  --auditor llm `
  --audit-trigger event `
  --save-agentic-logs
```

```powershell
Set-Content -Path "$PaperDir\agentic_run.txt" -Value $AgenticRun
```

检查是否真的调用了真实 endpoint：

```powershell
Get-Content "$AgenticRun\summary.json"
```

正式 LLM smoke 中应看到：

```json
"mock_llm_used": false
```

如果是 `true`，说明 key 没有配置成功，当前只是 mock LLM smoke。

## 9. 真实 LLM Agentic No-Physics Evaluation

复用 no-physics tiny 模型：

```powershell
python scripts\evaluate.py `
  --config configs\month_2026_04.yaml `
  --policies "ppo,sac,dqn" `
  --episodes 1 `
  --ppo-model "$AblationRun\models\ppo\ppo_model.zip" `
  --sac-model "$AblationRun\models\sac\sac_model.zip" `
  --dqn-model "$AblationRun\models\dqn\dqn_model.zip" `
  --run-name "${RunId}_agentic_llm_llm_no_physics_penalty" `
  --no-timestamp `
  --no-physics-penalty `
  --agentic-mode planner_auditor `
  --planner llm `
  --auditor llm `
  --audit-trigger event `
  --save-agentic-logs
```

```powershell
Set-Content -Path "$PaperDir\agentic_ablation_run.txt" -Value $AgenticAblationRun
```

## 10. 生成四组图表

主实验图：

```powershell
python scripts\make_figures.py `
  --config configs\month_2026_04.yaml `
  --run-dir "$MainRun" `
  --figures-dir "$PaperDir\figures\main"
```

no-physics 图：

```powershell
python scripts\make_figures.py `
  --config configs\month_2026_04.yaml `
  --run-dir "$AblationRun" `
  --figures-dir "$PaperDir\figures\ablation_no_physics_penalty"
```

agentic 图：

```powershell
python scripts\make_figures.py `
  --config configs\month_2026_04.yaml `
  --run-dir "$AgenticRun" `
  --figures-dir "$PaperDir\figures\agentic"
```

agentic no-physics 图：

```powershell
python scripts\make_figures.py `
  --config configs\month_2026_04.yaml `
  --run-dir "$AgenticAblationRun" `
  --figures-dir "$PaperDir\figures\agentic_no_physics_penalty"
```

预期每个 figures 子目录都有：

- `learning_curves.png`
- `safety_utility_frontier.png`
- `city_hour_liquidity_heatmap.png`

## 11. 写出 Paper Metadata 和 Manifest

写出 `paper_run_metadata.json`：

```powershell
python -c "import json, os; from pathlib import Path; p=Path(r'$PaperDir'); meta={'paper_run_id':'$RunId','config':'configs/month_2026_04.yaml','timesteps':8,'episodes':1,'main_run':r'$MainRun','ablation_run':r'$AblationRun','agentic_run':r'$AgenticRun','agentic_ablation_run':r'$AgenticAblationRun','llm_model':os.getenv('SOLARCHAIN_LLM_MODEL') or os.getenv('OPENAI_MODEL'),'llm_base_url':os.getenv('SOLARCHAIN_LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL'),'has_llm_key':bool(os.getenv('SOLARCHAIN_LLM_API_KEY') or os.getenv('OPENAI_API_KEY'))}; (p/'paper_run_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8'); print(json.dumps(meta, indent=2))"
```

写出简版 `PAPER_RESULTS.md`：

```powershell
@"
# Windows Tiny Smoke Results

用途：KDD Workshop on Evaluation and Trustworthiness of Agentic AI 投稿前的 Windows 工程 smoke。

## Runs

- Main: $MainRun
- No-physics: $AblationRun
- Agentic LLM: $AgenticRun
- Agentic LLM no-physics: $AgenticAblationRun

## Key Files

- $PaperDir\dataset_summary.json
- $PaperDir\paper_run_metadata.json
- $MainRun\summary.json
- $AblationRun\summary.json
- $AgenticRun\summary.json
- $AgenticRun\agentic_logs.jsonl
- $AgenticAblationRun\summary.json
- $AgenticAblationRun\agentic_logs.jsonl

## Figures

- $PaperDir\figures\main
- $PaperDir\figures\ablation_no_physics_penalty
- $PaperDir\figures\agentic
- $PaperDir\figures\agentic_no_physics_penalty

说明：本 run 使用 tiny timesteps，仅验证完整流程和产物结构，不作为论文数值结论。
"@ | Set-Content -Path "$PaperDir\PAPER_RESULTS.md" -Encoding UTF8
```

## 12. 最终验收清单

检查 run 文件：

```powershell
Get-ChildItem $PaperDir -Recurse
```

应至少存在：

- `$PaperDir\dataset_summary.json`
- `$PaperDir\paper_run_metadata.json`
- `$PaperDir\PAPER_RESULTS.md`
- `$PaperDir\figures\main\learning_curves.png`
- `$PaperDir\figures\ablation_no_physics_penalty\learning_curves.png`
- `$PaperDir\figures\agentic\learning_curves.png`
- `$PaperDir\figures\agentic_no_physics_penalty\learning_curves.png`
- `$MainRun\models\ppo\ppo_model.zip`
- `$MainRun\models\sac\sac_model.zip`
- `$MainRun\models\dqn\dqn_model.zip`
- `$AblationRun\models\ppo\ppo_model.zip`
- `$AgenticRun\agentic_logs.jsonl`
- `$AgenticAblationRun\agentic_logs.jsonl`

检查 LLM 状态：

```powershell
Get-Content "$AgenticRun\summary.json"
Get-Content "$AgenticAblationRun\summary.json"
```

若要验证真实 LLM，`mock_llm_used` 必须为 `false`。如果为 `true`，请回到第 2 步重新配置 key、base url 和 model。

## 13. 说明

- 本 Windows smoke 与正式 Linux pipeline 的产物结构保持一致。
- 本 smoke 的训练步数是 `8`，只验证流程，不验证性能。
- 论文正式结果应使用 Linux 正式流程：`TIMESTEPS=300000 EPISODES=30`。
- 如果真实 LLM endpoint 不稳定，可先用 `--planner rule --auditor rule` 验证 agentic wrapper，再切回 `llm`。
