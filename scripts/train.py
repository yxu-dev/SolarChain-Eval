from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solarchain_eval.config import load_config
from solarchain_eval.run_metadata import write_run_metadata
from solarchain_eval.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SolarChain-Eval RL baselines")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--algo", choices=["ppo", "sac", "dqn"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--no-physics-penalty", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.no_physics_penalty:
        config.no_physics_penalty = True
    timesteps = args.timesteps or config.training.timesteps
    base_output = Path(args.output_dir or config.output_dir)
    run_name = args.run_name or f"{args.algo}_train"
    if args.no_timestamp:
        output_dir = base_output / run_name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_output / f"{stamp}_{run_name}"
    model_path = train_model(args.algo, config, timesteps, output_dir)
    write_run_metadata(
        output_dir,
        run_type="train",
        args={**vars(args), "resolved_timesteps": timesteps, "output_dir": str(output_dir)},
        config=config,
        extra={"model_path": str(model_path)},
    )
    print(f"Saved {args.algo.upper()} model to {model_path}")
    print(f"Run output directory: {output_dir}")


if __name__ == "__main__":
    main()
