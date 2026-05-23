from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solarchain_eval.agent.auditor import LLMAuditor
from solarchain_eval.agent.context import build_step_context
from solarchain_eval.agent.llm_client import make_llm_client
from solarchain_eval.agent.planner import LLMPlanner, RulePlanner
from solarchain_eval.agent.schemas import (
    ActionBounds,
    ActionValues,
    AuditPolicy,
    AuditorOutput,
    PlannerOutput,
    RiskAssessment,
    action_values_to_array,
    validate_auditor_output,
    validate_planner_output,
)
from solarchain_eval.agent.wrappers import AgenticConfig
from solarchain_eval.config import load_config
from solarchain_eval.evaluate import evaluate_policies
from solarchain_eval.policies import make_builtin_policy


def test_planner_schema_validation_and_clipping():
    config = load_config()
    raw = PlannerOutput(
        governance_mode="test",
        action_bounds=ActionBounds(
            reward_ratio=[-1.0, 2.0],
            liquidity_ratio=[2.0, -1.0],
            burn_rate=[-1.0, 1.0],
        ),
        audit_policy=AuditPolicy(
            force_audit_if_violation_rate_above=2.0,
            force_audit_if_gap_below=-20.0,
            force_audit_if_action_jitter_above=9.0,
            force_audit_if_static_slippage_above=99.0,
        ),
        rationale="clip me",
    )
    plan, valid, error = validate_planner_output(raw, config)

    assert valid, error
    assert plan.action_bounds.reward_ratio == [config.market.min_reward_ratio, config.market.max_reward_ratio]
    assert plan.action_bounds.burn_rate == [0.0, config.market.max_burn_rate]
    assert plan.audit_policy.force_audit_if_violation_rate_above == 1.0


def test_auditor_schema_validation_and_action_sanitization():
    config = load_config()
    raw = AuditorOutput(
        decision="revise",
        final_action=ActionValues(reward_ratio=1.0, liquidity_ratio=1.0, burn_rate=1.0),
        risk_assessment=RiskAssessment(
            physics_risk="high",
            liquidity_risk="high",
            jitter_risk="high",
            fairness_risk="unknown",
        ),
        reason="sanitize me",
    )
    audit, valid, error = validate_auditor_output(raw, config, np.array([0.2, 0.7, 0.02], dtype=np.float32))
    final = action_values_to_array(audit.final_action)

    assert valid, error
    assert final[0] + final[1] <= config.market.max_total_allocation + 1e-6
    assert final[2] <= config.market.max_burn_rate


class FakeStructuredClient:
    def plan_structured(self, messages):
        return PlannerOutput(
            governance_mode="fake_api",
            action_bounds=ActionBounds(
                reward_ratio=[0.08, 0.50],
                liquidity_ratio=[0.30, 0.85],
                burn_rate=[0.0, 0.16],
            ),
            audit_policy=AuditPolicy(
                force_audit_if_violation_rate_above=0.01,
                force_audit_if_gap_below=-0.10,
                force_audit_if_action_jitter_above=0.25,
                force_audit_if_static_slippage_above=0.75,
            ),
            rationale="Fake structured API plan for tests.",
        )

    def audit_structured(self, messages):
        return AuditorOutput(
            decision="approve",
            final_action=ActionValues(reward_ratio=0.25, liquidity_ratio=0.75, burn_rate=0.02),
            risk_assessment=RiskAssessment(
                physics_risk="low",
                liquidity_risk="low",
                jitter_risk="low",
                fairness_risk="low",
            ),
            reason="Fake structured API audit approves.",
        )


def test_fake_structured_client_returns_schema_outputs():
    client = FakeStructuredClient()
    plan = client.plan_structured([])
    audit = client.audit_structured([])

    assert isinstance(plan, PlannerOutput)
    assert isinstance(audit, AuditorOutput)
    assert audit.decision in {"approve", "revise"}


def test_make_llm_client_requires_key_and_model(monkeypatch):
    monkeypatch.delenv("SOLARCHAIN_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SOLARCHAIN_LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="API key is required"):
        make_llm_client()


def test_llm_errors_propagate_without_safe_fallback():
    class FailingClient:
        def plan_structured(self, messages):
            raise RuntimeError("planner failed")

        def audit_structured(self, messages):
            raise RuntimeError("auditor failed")

    config = load_config()
    planner = LLMPlanner(FailingClient(), config)
    with pytest.raises(RuntimeError, match="planner failed"):
        planner.plan({})

    plan = RulePlanner(config).plan({})
    auditor = LLMAuditor(FailingClient(), config)
    step_context = build_step_context(
        np.zeros(12, dtype=np.float32),
        np.array([0.25, 0.75, 0.02], dtype=np.float32),
        np.array([0.25, 0.75, 0.02], dtype=np.float32),
        plan,
    )
    with pytest.raises(RuntimeError, match="auditor failed"):
        auditor.audit(step_context)


def test_rule_agentic_eval_smoke(tmp_path):
    config = load_config()
    config.episode_steps = 3
    policy = make_builtin_policy("static", config)
    agentic_config = AgenticConfig(
        agentic_mode="planner_auditor",
        planner="rule",
        auditor="rule",
        audit_trigger="event",
        save_agentic_logs=True,
    )

    metrics, actions, _ = evaluate_policies([policy], config, 1, tmp_path, agentic_config=agentic_config)

    assert "plan_validity_rate" in metrics.columns
    assert "agentic_action_modified" in actions.columns
    log_path = tmp_path / "agentic_logs.jsonl"
    assert log_path.exists()
    assert json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])["policy"] == "static"


def test_agentic_eval_with_no_physics_penalty(tmp_path):
    config = load_config()
    config.episode_steps = 2
    config.no_physics_penalty = True
    policy = make_builtin_policy("myopic", config)
    agentic_config = AgenticConfig(agentic_mode="planner", planner="rule")

    metrics, _, _ = evaluate_policies([policy], config, 1, tmp_path, agentic_config=agentic_config)

    assert bool(config.no_physics_penalty)
    assert float(metrics["plan_validity_rate"].iloc[0]) == 1.0
