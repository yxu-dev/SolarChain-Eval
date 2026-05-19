# SolarChain-Eval

SolarChain-Eval: A Physics-Constrained Benchmark for Trustworthy Economic Agents in Decentralized Energy Markets.

This repository is an independent benchmark implementation for evaluating autonomous economic governors in decentralized energy markets. It does not modify `SolarSave`; any reusable prototype logic or data from that project is copied here first and evolved inside this repository.

## Setup

```powershell
conda activate SolarChain-rl
pip install -e .
```

## Quick Smoke Run

```powershell
python scripts\evaluate.py --policies "static,random,myopic" --episodes 1
python scripts\make_figures.py --run-dir outputs\runs\<timestamp>_eval
```

## Full Baselines

```powershell
python scripts\run_all_baselines.py --timesteps 2048 --episodes 2
```

For longer paper runs, increase `--timesteps` and `--episodes`.

## Linux Paper Pipeline

For a Linux machine, use the bash workflow in `paper_pipeline/`:

```bash
bash paper_pipeline/00_setup_linux.sh
bash paper_pipeline/01_smoke_check.sh
PAPER_RUN_ID=paper_final TIMESTEPS=300000 EPISODES=20 bash paper_pipeline/02_run_paper_experiments.sh
```

Each invocation of `02_run_paper_experiments.sh` creates a separate `outputs/paper_runs/<paper_run_id>/` directory with metadata, run pointers, figures, and a `PAPER_RESULTS.md` manifest. See `paper_pipeline/README.md` for the full process and the mapping from output files to paper tables and figures.

## Outputs

Single-model training writes timestamped directories by default:

```text
outputs/runs/<timestamp>_ppo_train/ppo_model.zip
outputs/runs/<timestamp>_sac_train/sac_model.zip
outputs/runs/<timestamp>_dqn_train/dqn_model.zip
```

Use `--no-timestamp --run-name <name>` only when you intentionally want a stable output path.

`run_all_baselines.py` writes a complete timestamped benchmark bundle:

```text
outputs/runs/<timestamp>/
```

Each evaluation bundle writes:

- `metrics.csv`
- `summary.json`
- `actions.csv`
- `city_hour_policy.csv`
- `config_snapshot.json`

Key reported metrics include `physics_violation_rate`, liquidity `max_drawdown`, `action_jitter`, `slippage_reduction_vs_static`, `spatial_fairness_index`, and `artificial_liquidity_MWh`.

The figure script writes:

- `learning_curves.png`
- `safety_utility_frontier.png`
- `city_hour_liquidity_heatmap.png`

In the end, this repository produces reproducible benchmark evidence for the paper: trained RL policies, six-policy comparison metrics, trustworthiness metrics, ablation-ready runs with `--no-physics-penalty`, and the three core workshop figures.

The `--no-physics-penalty` ablation keeps logging unsafe backed supply while removing the reward penalty. This is intended to expose whether a policy exploits rejected or above-`P_max` generation to create artificial liquidity.

See `IMPLEMENTATION_PLAN.md` for the Chinese implementation plan and experiment scope.
