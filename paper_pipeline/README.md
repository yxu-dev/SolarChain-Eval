# Linux Paper Experiment Pipeline

This folder contains the Linux bash workflow for producing the full paper data and figures. Run all commands from the `SolarChain-Eval` repository root. The expected conda environment is `SolarChain-rl`.

## 0. Output Layout

The pipeline separates two levels of output:

- `outputs/runs/<run_id>/`: one concrete benchmark run, such as a main six-baseline run or a no-physics-penalty ablation run.
- `outputs/paper_runs/<paper_run_id>/`: one paper experiment batch that groups the main run, ablation run, figures, metadata, and result manifest.

Every invocation of `02_run_paper_experiments.sh` creates a new `paper_run_id` by default. You can also set it manually for hyperparameter testing.

Example layout:

```text
outputs/
  runs/
    physics_penalty_2p0_main/
      metrics.csv
      summary.json
      actions.csv
      city_hour_policy.csv
      config_snapshot.json
      run_metadata.json
      models/
    physics_penalty_2p0_no_physics_penalty/
      ...
  paper_runs/
    physics_penalty_2p0/
      paper_run_metadata.json
      main_run.txt
      ablation_run.txt
      PAPER_RESULTS.md
      figures/
        main/
        ablation_no_physics_penalty/
```

Each run directory records:

- `metrics.csv`: per-policy, per-episode metrics.
- `summary.json`: policy-level averages, including `slippage_reduction_vs_static`.
- `actions.csv`: hourly actions, liquidity, slippage, physical violation, and market state.
- `city_hour_policy.csv`: city-hour reward and liquidity split.
- `config_snapshot.json`: resolved benchmark config.
- `run_metadata.json`: command, CLI arguments, git commit, git status, model paths, and resolved config.
- `models/<algo>/<algo>_model.zip`: trained PPO/SAC/DQN models for full baseline runs.

## 1. Setup

First-time setup:

```bash
bash paper_pipeline/00_setup_linux.sh
```

If the Linux machine does not have the conda environment yet:

```bash
conda create -n SolarChain-rl python=3.10 -y
conda activate SolarChain-rl
pip install -r requirements.txt
pip install -e .
```

## 2. Smoke Check

Before any long experiment:

```bash
bash paper_pipeline/01_smoke_check.sh
```

This checks Python compilation, built-in policy evaluation, short DQN training, and figure generation.

## 3. Full Paper Experiment

Default run:

```bash
bash paper_pipeline/02_run_paper_experiments.sh
```

Default parameters:

- `CONFIG=configs/default.yaml`
- `TIMESTEPS=100000`
- `EPISODES=10`
- `PAPER_RUN_ID=<utc_timestamp>_paper`

Recommended final paper run:

```bash
PAPER_RUN_ID=paper_final TIMESTEPS=300000 EPISODES=20 bash paper_pipeline/02_run_paper_experiments.sh
```

The script runs:

1. Main six-baseline benchmark: PPO, SAC, DQN, Static 1:3, Random, Myopic Greedy.
2. No-physics-penalty ablation with the same baseline set.
3. Main figures.
4. Ablation figures.
5. `PAPER_RESULTS.md` manifest for paper tables and figures.

## 4. Hyperparameter Testing

Use a unique `PAPER_RUN_ID` for every parameter test:

```bash
PAPER_RUN_ID=lr_1e-4 TIMESTEPS=100000 EPISODES=10 bash paper_pipeline/02_run_paper_experiments.sh
PAPER_RUN_ID=lr_3e-4 TIMESTEPS=100000 EPISODES=10 bash paper_pipeline/02_run_paper_experiments.sh
PAPER_RUN_ID=gamma_0p995 TIMESTEPS=100000 EPISODES=10 bash paper_pipeline/02_run_paper_experiments.sh
```

If a parameter lives in YAML, copy the default config and edit the copy:

```bash
cp configs/default.yaml configs/physics_penalty_4p0.yaml
# Edit configs/physics_penalty_4p0.yaml: reward.physics_penalty = 4.0
PAPER_RUN_ID=physics_penalty_4p0 CONFIG=configs/physics_penalty_4p0.yaml bash paper_pipeline/02_run_paper_experiments.sh
```

Recommended low-cost sensitivity checks:

- `reward.physics_penalty`: `1.0`, `2.0`, `4.0`
- `training.learning_rate`: `0.0001`, `0.0003`, `0.001`
- `training.gamma`: `0.95`, `0.98`, `0.995`

The most important sweep is `reward.physics_penalty`, because it directly supports the trustworthiness and safety-utility frontier claims.

## 5. Regenerate Figures For Existing Runs

```bash
bash paper_pipeline/03_make_figures_for_run.sh outputs/runs/<run_id> outputs/paper_runs/<paper_run_id>/figures/custom
```

## 6. Paper Tables And Figures

Use the main run `summary.json` for the main comparison table:

- `cumulative_reward`
- `physics_violation_rate`
- `max_drawdown`
- `action_jitter`
- `slippage_reduction_vs_static`
- `spatial_fairness_index`
- `artificial_liquidity_MWh`

Use the main and ablation `summary.json` files to compare whether removing the `P_max` penalty increases:

- `physics_violation_rate`
- `artificial_liquidity_MWh`

Use these figures from each `paper_run_id`:

- `outputs/paper_runs/<paper_run_id>/figures/main/learning_curves.png`
- `outputs/paper_runs/<paper_run_id>/figures/main/safety_utility_frontier.png`
- `outputs/paper_runs/<paper_run_id>/figures/main/city_hour_liquidity_heatmap.png`
- `outputs/paper_runs/<paper_run_id>/figures/ablation_no_physics_penalty/*.png`

## 7. Logging

For long runs, keep terminal logs inside the same paper run directory:

```bash
mkdir -p outputs/paper_runs/paper_final
PAPER_RUN_ID=paper_final TIMESTEPS=300000 EPISODES=20 bash paper_pipeline/02_run_paper_experiments.sh 2>&1 | tee outputs/paper_runs/paper_final/full_run.log
```

## 8. Notes

- `SolarSave` is not used by this pipeline and should remain unchanged.
- Scripts use `set -euo pipefail`, so any failed step stops the pipeline.
- If an SB3 baseline is unstable on a target machine, keep the successful outputs and document the limitation in the paper.

