# SolarChain-Eval Artifact Pipeline

This directory contains the official reproducibility workflow for **SolarChain-Eval: A Physics-Constrained Benchmark for Trustworthy Economic Agents in Decentralized Energy Markets**.

The root `README.md` gives a concise project overview. This file is the source of truth for experiment execution, output layout, resume guidance, and artifact-to-paper mapping.

## Benchmark Story

SolarChain-Eval evaluates autonomous economic governors for decentralized peer-to-peer solar energy markets. The agent controls hourly reward allocation, liquidity provisioning, and token burns. The evaluation goes beyond reward maximization by measuring whether an agent respects physical constraints, avoids artificial liquidity, maintains market stability, and distributes rewards fairly across urban energy nodes.

The main cyber-physical risk is that a reward-maximizing policy can learn to back invalid or physically impossible generation, especially when the physics-informed reward penalty is removed. The no-physics-penalty ablation tests this risk directly.

## Environment

The artifact was prepared for a Conda environment named `SolarChain-rl`.

```bash
conda create -n SolarChain-rl python=3.10 -y
conda activate SolarChain-rl
python -m pip install -r requirements.txt
python -m pip install -e .
```

For a quick local check:

```bash
python -m pytest -q
bash artifact_pipeline/01_smoke_check.sh
```

LLM evaluations require a real OpenAI-compatible endpoint:

```bash
export SOLARCHAIN_LLM_API_KEY="..."
export SOLARCHAIN_LLM_BASE_URL="https://..."
export SOLARCHAIN_LLM_MODEL="..."
```

OpenAI-compatible fallback variables are also supported:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://..."
export OPENAI_MODEL="..."
```

If `--planner llm` or `--auditor llm` is enabled and the endpoint is missing or incompatible with structured outputs, evaluation fails instead of silently producing substitute results.

## Dataset

The hosted dataset is available at:

```text
https://huggingface.co/datasets/ThomasXu/solarchain-eval
```

The GitHub repository retains a lightweight data mirror in `data/`. The Hugging Face package contains `dataset_summary.json` and `checksums.sha256`; use those files to verify that the hosted dataset, GitHub mirror, and local data are consistent.

Accepted-paper configuration:

```text
configs/month_2026_04.yaml
data/datasets_2026_04_month/
```

Dataset window:

```text
[2026-04-01, 2026-05-01)
```

Expected data shape:

- five cities: Beijing, Shanghai, Chengdu, Shenzhen, Hangzhou
- 50 PV nodes
- 720 hourly timestamps
- 36,000 generation rows
- 720 market rows

## Official Pipeline

Run all accepted-paper artifact stages from the repository root:

```bash
PAPER_RUN_ID=paper_final_seed_20260511 \
TIMESTEPS=300000 \
EPISODES=30 \
RUN_AGENTIC=1 \
AGENTIC_POLICIES=ppo,sac,dqn \
AGENTIC_PLANNER=llm \
AGENTIC_AUDITOR=llm \
AGENTIC_AUDIT_TRIGGER=event \
bash artifact_pipeline/02_run_paper_experiments.sh
```

The pipeline executes:

```text
[0/9] dataset/config/metadata preparation
[1/9] main six-baseline run
[2/9] no-physics ablation
[3/9] LLM agentic main evaluation
[4/9] LLM agentic no-physics evaluation
[5/9] main figures
[6/9] ablation figures
[7/9] agentic figures
[8/9] agentic no-physics figures
[9/9] final manifest
```

All artifacts for one paper run are written under:

```text
outputs/<PAPER_RUN_ID>/
```

For `PAPER_RUN_ID=paper_final_seed_20260511`, the expected layout is:

```text
outputs/paper_final_seed_20260511/
  dataset_summary.json
  paper_run_metadata.json
  main_run.txt
  ablation_run.txt
  agentic_run.txt
  agentic_ablation_run.txt
  PAPER_RESULTS.md
  runs/
    main/
    no_physics_penalty/
    agentic_llm_llm/
    agentic_llm_llm_no_physics_penalty/
  figures/
    main/
    ablation_no_physics_penalty/
    agentic/
    agentic_no_physics_penalty/
```

Each evaluation run contains:

- `metrics.csv`
- `summary.json`
- `actions.csv`
- `city_hour_policy.csv`
- `config_snapshot.json`
- `run_metadata.json`

Agentic evaluation additionally writes `agentic_logs.jsonl`.

## Resume Guidance

Do not rerun completed training unless its output is missing or known to be invalid. If a stage has partial output, archive the partial run directory first, then rerun only that stage.

Completion evidence by stage:

| Stage | Completion evidence |
|---|---|
| `[0/9]` dataset/config/metadata preparation | config file, `dataset_summary.json`, `paper_run_metadata.json` |
| `[1/9]` main six-baseline run | `runs/main/summary.json`, PPO/SAC/DQN model files |
| `[2/9]` no-physics ablation | `runs/no_physics_penalty/summary.json`, PPO/SAC/DQN model files |
| `[3/9]` LLM agentic main evaluation | `runs/agentic_llm_llm/summary.json`, `agentic_logs.jsonl` |
| `[4/9]` LLM agentic no-physics evaluation | `runs/agentic_llm_llm_no_physics_penalty/summary.json`, `agentic_logs.jsonl` |
| `[5/9]` main figures | PNG files under `figures/main/` |
| `[6/9]` ablation figures | PNG files under `figures/ablation_no_physics_penalty/` |
| `[7/9]` agentic figures | PNG files under `figures/agentic/` |
| `[8/9]` agentic no-physics figures | PNG files under `figures/agentic_no_physics_penalty/` |
| `[9/9]` final manifest | `PAPER_RESULTS.md` |

The direct stage commands are the commands used inside `artifact_pipeline/02_run_paper_experiments.sh`. Use the same environment variables and run names shown above when recovering a run manually.

## Metrics And Figures

Use `summary.json` and `metrics.csv` from each run for policy-level tables.

Primary trustworthiness and utility metrics:

- `cumulative_reward`
- `physics_violation_rate`
- `max_drawdown`
- `max_token_drawdown`
- `action_jitter`
- `mean_slippage`
- `slippage_reduction_vs_static`
- `spatial_fairness_index`
- `artificial_liquidity_MWh`

Agentic extension metrics:

- `plan_validity_rate`
- `audit_call_rate`
- `revision_rate`
- `action_modification_rate`
- `avg_action_delta_from_auditor`
- `llm_failure_count`
- `audit_budget_per_episode`
- `target_audit_rate`
- `audit_cooldown_steps`

Each figure directory contains:

- `learning_curves.png`: reward by episode and policy.
- `safety_utility_frontier.png`: physics violation rate versus cumulative reward.
- `city_hour_liquidity_heatmap.png`: liquidity policy over city-hour records.

Recommended paper mapping:

- Main benchmark figure: `figures/main/`
- Reward-misspecification ablation: `figures/ablation_no_physics_penalty/`
- Evaluation-only agentic extension: `figures/agentic/`
- Agentic no-physics comparison: `figures/agentic_no_physics_penalty/`

## Acceptance Checks

```bash
python -m pytest -q
test -s outputs/paper_final_seed_20260511/PAPER_RESULTS.md
test -s outputs/paper_final_seed_20260511/runs/main/models/ppo/ppo_model.zip
test -s outputs/paper_final_seed_20260511/runs/main/models/sac/sac_model.zip
test -s outputs/paper_final_seed_20260511/runs/main/models/dqn/dqn_model.zip
test -s outputs/paper_final_seed_20260511/runs/agentic_llm_llm/agentic_logs.jsonl
find outputs/paper_final_seed_20260511/figures -name "*.png" -type f
```

LLM structured-output failures should stop the run. Formal LLM results must not be replaced by mock outputs. If an endpoint fails, fix the API key, model, base URL, or structured-output compatibility and rerun the interrupted stage.
