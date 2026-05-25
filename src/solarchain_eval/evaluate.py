from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .actions import decode_continuous_action, decode_discrete_action
from .agent.llm_client import make_llm_client
from .agent.wrappers import AgenticConfig, agentic_metrics, make_agentic_processor
from .config import BenchmarkConfig
from .data import load_benchmark_data
from .env import SolarChainBenchmarkEnv
from .metrics import summarize_episode
from .policies import Policy


def run_episode(
    policy: Policy,
    config: BenchmarkConfig,
    seed: int,
    episode: int,
    policy_name: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics, steps, _ = _run_episode_impl(policy, config, seed, episode, policy_name)
    return metrics, steps


def _run_episode_impl(
    policy: Policy,
    config: BenchmarkConfig,
    seed: int,
    episode: int,
    policy_name: str | None = None,
    agentic_config: AgenticConfig | None = None,
    llm_client: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    local_config = deepcopy(config)
    name = policy_name or getattr(policy, "name", "policy")
    local_config.action_mode = "discrete" if name == "dqn" else "continuous"
    if hasattr(policy, "config"):
        policy.config = local_config
    env_config = deepcopy(local_config)
    agentic_enabled = bool(agentic_config and agentic_config.agentic_mode != "none")
    if agentic_enabled:
        env_config.action_mode = "continuous"
    data = load_benchmark_data(env_config.data_dir)
    env = SolarChainBenchmarkEnv(config=env_config, data=data)
    obs, _ = env.reset(seed=seed)
    processor = None
    if agentic_enabled:
        needs_llm = agentic_config.planner == "llm" or agentic_config.auditor == "llm"
        processor = make_agentic_processor(
            config=env_config,
            agentic_config=agentic_config,
            llm_client=(llm_client or make_llm_client()) if needs_llm else None,
        )
        processor.reset(env)
    done = False
    step_rows: list[dict[str, Any]] = []
    agentic_logs: list[dict[str, Any]] = []

    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        agentic_info: dict[str, Any] = {}
        if processor is not None:
            proposed_actual = _decode_policy_action(action, local_config)
            previous_action = np.asarray(env._prev_action, dtype=np.float32)
            action, agentic_info, log_row = processor.process(
                obs=obs,
                proposed_actual_action=proposed_actual,
                previous_action=previous_action,
                latest_info=env.latest_info(),
            )
            log_row.update({"policy": name, "episode": episode, "step": len(step_rows)})
            agentic_logs.append(log_row)
        obs, reward, terminated, truncated, info = env.step(action)
        info.update(agentic_info)
        row = {
            "policy": name,
            "episode": episode,
            "step": len(step_rows),
            "reward": float(reward),
            **{key: value for key, value in info.items() if key != "city_rewards"},
            "city_rewards": info.get("city_rewards", {}),
        }
        step_rows.append(row)
        done = terminated or truncated

    metrics = summarize_episode(step_rows)
    metrics.update({"policy": name, "episode": episode, "seed": seed})
    if processor is not None:
        metrics.update(agentic_metrics(processor.stats, len(step_rows)))
    return metrics, step_rows, agentic_logs


def evaluate_policies(
    policies: list[Policy],
    config: BenchmarkConfig,
    episodes: int,
    output_dir: str | Path,
    agentic_config: AgenticConfig | None = None,
    llm_client: Any | None = None,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    city_hour_rows: list[dict[str, Any]] = []
    agentic_log_rows: list[dict[str, Any]] = []

    total_episodes = len(policies) * episodes
    progress = tqdm(
        total=total_episodes,
        desc="Evaluating policies",
        unit="episode",
        disable=not show_progress,
        dynamic_ncols=True,
    )
    try:
        for policy in policies:
            name = getattr(policy, "name", "policy")
            for episode in range(episodes):
                progress.set_postfix(policy=name, episode=f"{episode + 1}/{episodes}")
                metrics, steps, agentic_logs = _run_episode_impl(
                    policy,
                    config,
                    config.seed + episode,
                    episode,
                    name,
                    agentic_config,
                    llm_client,
                )
                metric_rows.append(metrics)
                agentic_log_rows.extend(agentic_logs)
                for row in steps:
                    action_rows.append({key: value for key, value in row.items() if key != "city_rewards"})
                    for city, value in row.get("city_rewards", {}).items():
                        city_hour_rows.append(
                            {
                                "policy": name,
                                "episode": episode,
                                "hour": row["hour"],
                                "city": city,
                                "city_reward": float(value),
                                "reward_ratio": row["reward_ratio"],
                                "liquidity_ratio": row["liquidity_ratio"],
                                "burn_rate": row["burn_rate"],
                            }
                        )
                progress.update(1)
    finally:
        progress.close()

    metrics_frame = pd.DataFrame(metric_rows)
    actions_frame = pd.DataFrame(action_rows)
    city_hour_frame = pd.DataFrame(city_hour_rows)
    metrics_frame.to_csv(output / "metrics.csv", index=False)
    actions_frame.to_csv(output / "actions.csv", index=False)
    city_hour_frame.to_csv(output / "city_hour_policy.csv", index=False)

    summary = metrics_frame.groupby("policy", as_index=False).mean(numeric_only=True)
    if "static" in set(summary["policy"]):
        static_slippage = float(summary.loc[summary["policy"].eq("static"), "mean_slippage"].iloc[0])
        summary["slippage_reduction_vs_static"] = (static_slippage - summary["mean_slippage"]) / max(static_slippage, 1e-9)
    summary.to_json(output / "summary.json", orient="records", indent=2)
    (output / "config_snapshot.json").write_text(_json_dumps_dataclass(config), encoding="utf-8")
    if agentic_config and agentic_config.save_agentic_logs:
        with (output / "agentic_logs.jsonl").open("w", encoding="utf-8") as handle:
            for row in agentic_log_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return metrics_frame, actions_frame, city_hour_frame


def _json_dumps_dataclass(config: BenchmarkConfig) -> str:
    import json

    return json.dumps(asdict(config), indent=2)


class SB3Policy:
    def __init__(self, model, name: str):
        self.model = model
        self.name = name

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self.model.predict(obs, deterministic=deterministic)


def _decode_policy_action(action: Any, config: BenchmarkConfig) -> np.ndarray:
    if config.action_mode == "discrete":
        return decode_discrete_action(int(action), config)
    return decode_continuous_action(np.asarray(action, dtype=np.float32), config)
