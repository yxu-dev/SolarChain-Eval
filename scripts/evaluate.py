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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SolarChain-Eval policies")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--policies", default="static,random,myopic")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ppo-model", default=None)
    parser.add_argument("--sac-model", default=None)
    parser.add_argument("--dqn-model", default=None)
    parser.add_argument("--run-name", default="eval")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--no-physics-penalty", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.no_physics_penalty:
        config.no_physics_penalty = True
    policies = []
    for name in [item.strip() for item in args.policies.split(",") if item.strip()]:
        if name == "ppo":
            config.action_mode = "continuous"
            policies.append(SB3Policy(PPO.load(args.ppo_model), "ppo"))
        elif name == "sac":
            config.action_mode = "continuous"
            policies.append(SB3Policy(SAC.load(args.sac_model), "sac"))
        elif name == "dqn":
            config.action_mode = "discrete"
            policies.append(SB3Policy(DQN.load(args.dqn_model), "dqn"))
        else:
            config.action_mode = "continuous"
            policies.append(make_builtin_policy(name, config))

    base_output = Path(args.output_dir or config.output_dir)
    if args.no_timestamp:
        output_dir = base_output / args.run_name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_output / f"{stamp}_{args.run_name}"
    episodes = args.episodes or config.evaluation.episodes
    evaluate_policies(policies, config, episodes, output_dir)
    write_run_metadata(
        output_dir,
        run_type="evaluate",
        args={**vars(args), "resolved_episodes": episodes, "output_dir": str(output_dir)},
        config=config,
        extra={"policies": [getattr(policy, "name", "policy") for policy in policies]},
    )
    print(f"Wrote evaluation outputs to {output_dir}")


if __name__ == "__main__":
    main()
