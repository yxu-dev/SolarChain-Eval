from __future__ import annotations

from typing import Any

import numpy as np

from .schemas import PlannerOutput, action_dict


OBSERVATION_FIELDS = [
    "sin_hour",
    "cos_hour",
    "verified_mwh",
    "reported_mwh",
    "pmax_mwh",
    "gap",
    "liquidity",
    "token_price",
    "violation_rate",
    "static_slippage",
    "prev_reward_ratio",
    "prev_liquidity_ratio",
]


def build_episode_context(env: Any) -> dict[str, Any]:
    base_env = getattr(env, "unwrapped", env)
    config = base_env.config
    market = config.market
    context: dict[str, Any] = {
        "episode_steps": int(config.episode_steps),
        "no_physics_penalty": bool(config.no_physics_penalty),
        "action_bounds_from_config": {
            "reward_ratio": [market.min_reward_ratio, market.max_reward_ratio],
            "liquidity_ratio": [market.min_liquidity_ratio, market.max_liquidity_ratio],
            "burn_rate": [0.0, market.max_burn_rate],
            "max_total_allocation": market.max_total_allocation,
        },
    }
    try:
        start = int(base_env._episode_start_hour)
        stop = start + int(config.episode_steps)
        rows = base_env.data.city_hour[
            base_env.data.city_hour["absolute_hour"].between(start, stop - 1, inclusive="both")
        ]
        market_rows = base_env.data.market[
            base_env.data.market["absolute_hour"].between(start, stop - 1, inclusive="both")
        ]
        trades = base_env.data.trades[
            base_env.data.trades["absolute_hour"].between(start, stop - 1, inclusive="both")
        ]
        hourly = rows.groupby("absolute_hour", as_index=False).agg(
            verified_W=("verified_W", "sum"),
            reported_W=("reported_W", "sum"),
            pmax_W=("pmax_W", "sum"),
            violation_count=("violation_count", "sum"),
            record_count=("record_count", "sum"),
        )
        hourly["violation_rate"] = hourly["violation_count"] / hourly["record_count"].clip(lower=1)
        demand = trades.groupby("absolute_hour", as_index=False).agg(demand_MWh=("energy_purchased_MW", "sum"))
        hourly = hourly.merge(demand, on="absolute_hour", how="left").fillna({"demand_MWh": 0.0})
        hourly["verified_mwh"] = hourly["verified_W"] / 1_000_000.0
        hourly["reported_mwh"] = hourly["reported_W"] / 1_000_000.0
        hourly["pmax_mwh"] = hourly["pmax_W"] / 1_000_000.0
        hourly["gap"] = (hourly["verified_mwh"] - hourly["demand_MWh"]) / hourly["demand_MWh"].clip(lower=1e-6)
        context.update(
            {
                "episode_start_hour": start,
                "mean_verified_mwh": _mean(hourly["verified_mwh"]),
                "min_verified_mwh": _min(hourly["verified_mwh"]),
                "max_verified_mwh": _max(hourly["verified_mwh"]),
                "mean_reported_mwh": _mean(hourly["reported_mwh"]),
                "mean_pmax_mwh": _mean(hourly["pmax_mwh"]),
                "mean_gap": _mean(hourly["gap"]),
                "min_gap": _min(hourly["gap"]),
                "mean_violation_rate": _mean(hourly["violation_rate"]),
                "max_violation_rate": _max(hourly["violation_rate"]),
                "mean_static_slippage": _mean(market_rows.get("slippage_solarchain_pct", [])),
            }
        )
    except Exception:
        obs = base_env._observation()
        context.update(_observation_context(obs))
        context["fallback_context"] = True
    return context


def build_step_context(
    obs: np.ndarray,
    proposed_action: np.ndarray,
    previous_action: np.ndarray,
    plan: PlannerOutput,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    proposed = np.asarray(proposed_action, dtype=np.float32)
    previous = np.asarray(previous_action, dtype=np.float32)
    return {
        "observation": _observation_context(obs),
        "proposed_action": action_dict(proposed),
        "previous_action": action_dict(previous),
        "action_jitter": float(np.linalg.norm(proposed - previous, ord=1)),
        "plan": plan.model_dump(),
        "latest_info": info or {},
    }


def _observation_context(obs: np.ndarray) -> dict[str, float]:
    arr = np.asarray(obs, dtype=np.float32)
    return {field: float(arr[index]) for index, field in enumerate(OBSERVATION_FIELDS)}


def _mean(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)) if arr.size else 0.0


def _min(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.min(arr)) if arr.size else 0.0


def _max(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.max(arr)) if arr.size else 0.0
