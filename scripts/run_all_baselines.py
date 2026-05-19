from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stable_baselines3 import DQN, PPO, SAC

from solarchain_eval.config import load_config
from solarchain_eval.evaluate import SB3Policy, evaluate_policies
from solarchain_eval.policies import make_builtin_policy
from solarchain_eval.run_metadata import write_run_metadata
from solarchain_eval.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all SolarChain-Eval baselines")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--timesteps", type=int, default=2048)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--no-physics-penalty", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.no_physics_penalty:
        config.no_physics_penalty = True
    run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir or config.output_dir) / run_id
    model_paths = {}

    if not args.skip_rl:
        for algo in ["ppo", "sac", "dqn"]:
            model_paths[algo] = train_model(algo, config, args.timesteps, run_dir / "models" / algo)

    policies = [
        make_builtin_policy("static", config),
        make_builtin_policy("random", config),
        make_builtin_policy("myopic", config),
    ]
    if not args.skip_rl:
        config.action_mode = "continuous"
        policies.append(SB3Policy(PPO.load(model_paths["ppo"]), "ppo"))
        policies.append(SB3Policy(SAC.load(model_paths["sac"]), "sac"))
        config.action_mode = "discrete"
        policies.append(SB3Policy(DQN.load(model_paths["dqn"]), "dqn"))

    config.action_mode = "continuous"
    evaluate_policies(policies, config, args.episodes, run_dir)
    write_run_metadata(
        run_dir,
        run_type="run_all_baselines",
        args={**vars(args), "run_id": run_id, "run_dir": str(run_dir)},
        config=config,
        extra={
            "model_paths": {key: str(value) for key, value in model_paths.items()},
            "policies": [getattr(policy, "name", "policy") for policy in policies],
        },
    )
    print(f"Wrote benchmark run to {run_dir}")


if __name__ == "__main__":
    main()
