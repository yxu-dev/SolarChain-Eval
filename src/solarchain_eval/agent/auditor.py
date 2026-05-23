from __future__ import annotations

from typing import Any

import numpy as np

from solarchain_eval.actions import sanitize_actual_action
from solarchain_eval.config import BenchmarkConfig

from .context import build_step_context
from .prompts import auditor_messages
from .schemas import (
    ActionValues,
    AuditorOutput,
    PlannerOutput,
    RiskAssessment,
    approve_original_action,
    validate_auditor_output,
)


class NoOpAuditor:
    def __init__(self, config: BenchmarkConfig, audit_trigger: str = "event"):
        self.config = config
        self.audit_trigger = audit_trigger
        self.last_valid = True
        self.failure_count = 0

    def should_audit(
        self,
        obs: np.ndarray,
        proposed_action: np.ndarray,
        previous_action: np.ndarray,
        plan: PlannerOutput,
    ) -> bool:
        return False

    def audit(self, step_context: dict[str, Any]) -> AuditorOutput:
        return approve_original_action(_action_from_context(step_context), self.config, reason="NoOpAuditor approved.")


class RuleAuditor:
    def __init__(self, config: BenchmarkConfig, audit_trigger: str = "event"):
        self.config = config
        self.audit_trigger = audit_trigger
        self.last_valid = True
        self.failure_count = 0

    def should_audit(
        self,
        obs: np.ndarray,
        proposed_action: np.ndarray,
        previous_action: np.ndarray,
        plan: PlannerOutput,
    ) -> bool:
        if self.audit_trigger == "always":
            return True
        return _event_triggered(obs, proposed_action, previous_action, plan)

    def audit(self, step_context: dict[str, Any]) -> AuditorOutput:
        obs = step_context["observation"]
        action = _action_from_context(step_context)
        plan = PlannerOutput.model_validate(step_context["plan"])
        physics_risky = obs["violation_rate"] > plan.audit_policy.force_audit_if_violation_rate_above
        liquidity_risky = obs["gap"] < plan.audit_policy.force_audit_if_gap_below
        jitter_risky = step_context["action_jitter"] > plan.audit_policy.force_audit_if_action_jitter_above

        revised = np.asarray(action, dtype=np.float32).copy()
        if physics_risky:
            revised[0] *= 0.85
            revised[1] *= 0.90
            revised[2] = max(revised[2], min(0.12, self.config.market.max_burn_rate))
        if liquidity_risky:
            revised[1] = max(revised[1], min(0.80, self.config.market.max_liquidity_ratio))
            revised[0] *= 0.90
        if jitter_risky:
            previous = _previous_action_from_context(step_context)
            revised = 0.60 * previous + 0.40 * revised

        revised = sanitize_actual_action(revised, self.config)
        changed = bool(np.linalg.norm(revised - action, ord=1) > 1e-6)
        return AuditorOutput(
            decision="revise" if changed else "approve",
            final_action=ActionValues(
                reward_ratio=float(revised[0]),
                liquidity_ratio=float(revised[1]),
                burn_rate=float(revised[2]),
            ),
            risk_assessment=RiskAssessment(
                physics_risk="high" if physics_risky else "low",
                liquidity_risk="high" if liquidity_risky else "low",
                jitter_risk="high" if jitter_risky else "low",
                fairness_risk="unknown",
            ),
            reason="Rule auditor revised risky action." if changed else "Rule auditor approved bounded action.",
        )


class LLMAuditor:
    def __init__(self, llm_client: Any, config: BenchmarkConfig, audit_trigger: str = "event"):
        self.llm_client = llm_client
        self.config = config
        self.audit_trigger = audit_trigger
        self.last_valid = True
        self.failure_count = 0

    def should_audit(
        self,
        obs: np.ndarray,
        proposed_action: np.ndarray,
        previous_action: np.ndarray,
        plan: PlannerOutput,
    ) -> bool:
        if self.audit_trigger == "always":
            return True
        return _event_triggered(obs, proposed_action, previous_action, plan)

    def audit(self, step_context: dict[str, Any]) -> AuditorOutput:
        original = _action_from_context(step_context)
        payload = self.llm_client.audit_structured(auditor_messages(step_context))
        audit, valid, error = validate_auditor_output(payload, self.config, original)
        self.last_valid = valid
        if not valid:
            self.failure_count += 1
            self.last_valid = False
            raise RuntimeError(f"LLM auditor structured output validation failed: {error}")
        return audit


def should_audit_context(
    auditor: NoOpAuditor | RuleAuditor | LLMAuditor,
    obs: np.ndarray,
    proposed_action: np.ndarray,
    previous_action: np.ndarray,
    plan: PlannerOutput,
) -> bool:
    return auditor.should_audit(obs, proposed_action, previous_action, plan)


def build_audit_context(
    obs: np.ndarray,
    proposed_action: np.ndarray,
    previous_action: np.ndarray,
    plan: PlannerOutput,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_step_context(obs, proposed_action, previous_action, plan, info)


def _event_triggered(
    obs: np.ndarray,
    proposed_action: np.ndarray,
    previous_action: np.ndarray,
    plan: PlannerOutput,
) -> bool:
    arr = np.asarray(obs, dtype=np.float32)
    action = np.asarray(proposed_action, dtype=np.float32)
    previous = np.asarray(previous_action, dtype=np.float32)
    policy = plan.audit_policy
    return bool(
        arr[8] > policy.force_audit_if_violation_rate_above
        or arr[5] < policy.force_audit_if_gap_below
        or np.linalg.norm(action - previous, ord=1) > policy.force_audit_if_action_jitter_above
        or arr[9] > policy.force_audit_if_static_slippage_above
    )


def _action_from_context(step_context: dict[str, Any]) -> np.ndarray:
    action = step_context["proposed_action"]
    return np.array([action["reward_ratio"], action["liquidity_ratio"], action["burn_rate"]], dtype=np.float32)


def _previous_action_from_context(step_context: dict[str, Any]) -> np.ndarray:
    action = step_context["previous_action"]
    return np.array([action["reward_ratio"], action["liquidity_ratio"], action["burn_rate"]], dtype=np.float32)
