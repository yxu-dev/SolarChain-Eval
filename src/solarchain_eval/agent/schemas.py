from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from solarchain_eval.actions import sanitize_actual_action
from solarchain_eval.config import BenchmarkConfig


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ActionBounds(StrictModel):
    reward_ratio: list[float] = Field(min_length=2, max_length=2)
    liquidity_ratio: list[float] = Field(min_length=2, max_length=2)
    burn_rate: list[float] = Field(min_length=2, max_length=2)

    @field_validator("reward_ratio", "liquidity_ratio", "burn_rate")
    @classmethod
    def bounds_must_be_orderable(cls, value: list[float]) -> list[float]:
        if len(value) != 2:
            raise ValueError("bounds must contain [min, max]")
        if not np.isfinite(value).all():
            raise ValueError("bounds must be finite")
        return [float(value[0]), float(value[1])]


class AuditPolicy(StrictModel):
    force_audit_if_violation_rate_above: float
    force_audit_if_gap_below: float
    force_audit_if_action_jitter_above: float
    force_audit_if_static_slippage_above: float


class PlannerOutput(StrictModel):
    governance_mode: str
    action_bounds: ActionBounds
    audit_policy: AuditPolicy
    rationale: str


class ActionValues(StrictModel):
    reward_ratio: float
    liquidity_ratio: float
    burn_rate: float

    @field_validator("reward_ratio", "liquidity_ratio", "burn_rate")
    @classmethod
    def values_must_be_finite(cls, value: float) -> float:
        if not np.isfinite(value):
            raise ValueError("action values must be finite")
        return float(value)


class RiskAssessment(StrictModel):
    physics_risk: str
    liquidity_risk: str
    jitter_risk: str
    fairness_risk: str


class AuditorOutput(StrictModel):
    decision: Literal["approve", "revise"]
    final_action: ActionValues
    risk_assessment: RiskAssessment
    reason: str


def planner_json_schema() -> dict[str, Any]:
    return _strict_json_schema(PlannerOutput)


def auditor_json_schema() -> dict[str, Any]:
    return _strict_json_schema(AuditorOutput)


def _strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    _forbid_extra_properties(schema)
    return schema


def _forbid_extra_properties(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            node["additionalProperties"] = False
        for value in node.values():
            _forbid_extra_properties(value)
    elif isinstance(node, list):
        for value in node:
            _forbid_extra_properties(value)


def validate_planner_output(payload: Any, config: BenchmarkConfig) -> tuple[PlannerOutput, bool, str]:
    try:
        plan = payload if isinstance(payload, PlannerOutput) else PlannerOutput.model_validate(payload)
        return clip_plan_to_config(plan, config), True, ""
    except (TypeError, ValueError, ValidationError) as exc:
        return safe_default_plan(config), False, str(exc)


def validate_auditor_output(
    payload: Any,
    config: BenchmarkConfig,
    original_action: np.ndarray,
) -> tuple[AuditorOutput, bool, str]:
    try:
        audit = payload if isinstance(payload, AuditorOutput) else AuditorOutput.model_validate(payload)
        return clip_auditor_to_config(audit, config), True, ""
    except (TypeError, ValueError, ValidationError) as exc:
        return approve_original_action(original_action, config, reason=f"auditor fallback: {exc}"), False, str(exc)


def clip_plan_to_config(plan: PlannerOutput, config: BenchmarkConfig) -> PlannerOutput:
    market = config.market
    return PlannerOutput(
        governance_mode=plan.governance_mode.strip() or "bounded_governance",
        action_bounds=ActionBounds(
            reward_ratio=_clip_bounds(plan.action_bounds.reward_ratio, market.min_reward_ratio, market.max_reward_ratio),
            liquidity_ratio=_clip_bounds(
                plan.action_bounds.liquidity_ratio,
                market.min_liquidity_ratio,
                market.max_liquidity_ratio,
            ),
            burn_rate=_clip_bounds(plan.action_bounds.burn_rate, 0.0, market.max_burn_rate),
        ),
        audit_policy=AuditPolicy(
            force_audit_if_violation_rate_above=float(
                np.clip(plan.audit_policy.force_audit_if_violation_rate_above, 0.0, 1.0)
            ),
            force_audit_if_gap_below=float(np.clip(plan.audit_policy.force_audit_if_gap_below, -10.0, 10.0)),
            force_audit_if_action_jitter_above=float(
                np.clip(plan.audit_policy.force_audit_if_action_jitter_above, 0.0, 3.0)
            ),
            force_audit_if_static_slippage_above=float(
                np.clip(plan.audit_policy.force_audit_if_static_slippage_above, 0.0, 10.0)
            ),
        ),
        rationale=plan.rationale.strip() or "Structured plan validated and clipped to benchmark constraints.",
    )


def clip_auditor_to_config(audit: AuditorOutput, config: BenchmarkConfig) -> AuditorOutput:
    actual = action_values_to_array(audit.final_action)
    clipped = sanitize_actual_action(actual, config)
    return AuditorOutput(
        decision=audit.decision,
        final_action=array_to_action_values(clipped),
        risk_assessment=audit.risk_assessment,
        reason=audit.reason.strip() or "Structured audit validated and clipped to benchmark constraints.",
    )


def safe_default_plan(config: BenchmarkConfig) -> PlannerOutput:
    market = config.market
    return PlannerOutput(
        governance_mode="safe_default",
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
        rationale="Safe fallback plan using benchmark action bounds and conservative audit thresholds.",
    )


def approve_original_action(original_action: np.ndarray, config: BenchmarkConfig, reason: str = "approved") -> AuditorOutput:
    actual = sanitize_actual_action(np.asarray(original_action, dtype=np.float32), config)
    return AuditorOutput(
        decision="approve",
        final_action=array_to_action_values(actual),
        risk_assessment=RiskAssessment(
            physics_risk="unknown",
            liquidity_risk="unknown",
            jitter_risk="unknown",
            fairness_risk="unknown",
        ),
        reason=reason,
    )


def action_values_to_array(values: ActionValues | dict[str, float]) -> np.ndarray:
    if isinstance(values, ActionValues):
        return np.array([values.reward_ratio, values.liquidity_ratio, values.burn_rate], dtype=np.float32)
    return np.array([values["reward_ratio"], values["liquidity_ratio"], values["burn_rate"]], dtype=np.float32)


def array_to_action_values(action: np.ndarray) -> ActionValues:
    arr = np.asarray(action, dtype=np.float32)
    return ActionValues(reward_ratio=float(arr[0]), liquidity_ratio=float(arr[1]), burn_rate=float(arr[2]))


def action_dict(action: np.ndarray) -> dict[str, float]:
    values = array_to_action_values(np.asarray(action, dtype=np.float32))
    return values.model_dump()


def apply_plan_bounds(action: np.ndarray, plan: PlannerOutput, config: BenchmarkConfig) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    bounds = plan.action_bounds
    clipped = np.array(
        [
            np.clip(arr[0], bounds.reward_ratio[0], bounds.reward_ratio[1]),
            np.clip(arr[1], bounds.liquidity_ratio[0], bounds.liquidity_ratio[1]),
            np.clip(arr[2], bounds.burn_rate[0], bounds.burn_rate[1]),
        ],
        dtype=np.float32,
    )
    return sanitize_actual_action(clipped, config)


def _clip_bounds(bounds: list[float], low: float, high: float) -> list[float]:
    left, right = sorted([float(bounds[0]), float(bounds[1])])
    left = float(np.clip(left, low, high))
    right = float(np.clip(right, low, high))
    if right < left:
        right = left
    return [left, right]
