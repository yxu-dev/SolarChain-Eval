from __future__ import annotations

from typing import Any

from solarchain_eval.config import BenchmarkConfig

from .prompts import planner_messages
from .schemas import (
    ActionBounds,
    AuditPolicy,
    PlannerOutput,
    safe_default_plan,
    validate_planner_output,
)


class SafeDefaultPlanner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.last_valid = True
        self.failure_count = 0

    def plan(self, episode_context: dict[str, Any]) -> PlannerOutput:
        self.last_valid = True
        return safe_default_plan(self.config)


class RulePlanner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.last_valid = True
        self.failure_count = 0

    def plan(self, episode_context: dict[str, Any]) -> PlannerOutput:
        market = self.config.market
        max_violation = float(episode_context.get("max_violation_rate", 0.0))
        min_gap = float(episode_context.get("min_gap", 0.0))
        mean_slippage = float(episode_context.get("mean_static_slippage", 0.0))

        risk_high = max_violation > 0.01 or min_gap < -0.10 or mean_slippage > 0.75
        if risk_high:
            raw = PlannerOutput(
                governance_mode="rule_conservative",
                action_bounds=ActionBounds(
                    reward_ratio=[market.min_reward_ratio, min(0.45, market.max_reward_ratio)],
                    liquidity_ratio=[max(0.35, market.min_liquidity_ratio), market.max_liquidity_ratio],
                    burn_rate=[0.02, min(0.18, market.max_burn_rate)],
                ),
                audit_policy=AuditPolicy(
                    force_audit_if_violation_rate_above=0.005,
                    force_audit_if_gap_below=-0.05,
                    force_audit_if_action_jitter_above=0.18,
                    force_audit_if_static_slippage_above=0.60,
                ),
                rationale="Rule planner selected conservative bounds due to episode-level risk signals.",
            )
        else:
            raw = PlannerOutput(
                governance_mode="rule_balanced",
                action_bounds=ActionBounds(
                    reward_ratio=[market.min_reward_ratio, market.max_reward_ratio],
                    liquidity_ratio=[market.min_liquidity_ratio, market.max_liquidity_ratio],
                    burn_rate=[0.0, market.max_burn_rate],
                ),
                audit_policy=AuditPolicy(
                    force_audit_if_violation_rate_above=0.01,
                    force_audit_if_gap_below=-0.10,
                    force_audit_if_action_jitter_above=0.25,
                    force_audit_if_static_slippage_above=0.75,
                ),
                rationale="Rule planner selected balanced bounds for a lower-risk episode.",
            )
        plan, valid, _ = validate_planner_output(raw, self.config)
        self.last_valid = valid
        return plan


class LLMPlanner:
    def __init__(self, llm_client: Any, config: BenchmarkConfig):
        self.llm_client = llm_client
        self.config = config
        self.last_valid = True
        self.failure_count = 0

    def plan(self, episode_context: dict[str, Any]) -> PlannerOutput:
        payload = self.llm_client.plan_structured(planner_messages(episode_context))
        plan, valid, error = validate_planner_output(payload, self.config)
        self.last_valid = valid
        if not valid:
            self.failure_count += 1
            self.last_valid = False
            raise RuntimeError(f"LLM planner structured output validation failed: {error}")
        return plan
