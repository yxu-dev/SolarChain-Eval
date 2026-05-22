from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from solarchain_eval.actions import encode_actual_action
from solarchain_eval.config import BenchmarkConfig

from .auditor import LLMAuditor, NoOpAuditor, RuleAuditor, build_audit_context
from .context import build_episode_context
from .planner import LLMPlanner, RulePlanner, SafeDefaultPlanner
from .schemas import PlannerOutput, action_dict, action_values_to_array, apply_plan_bounds


@dataclass
class AgenticConfig:
    agentic_mode: str = "none"
    planner: str = "none"
    auditor: str = "none"
    audit_trigger: str = "event"
    save_agentic_logs: bool = False


class AgenticActionProcessor:
    def __init__(
        self,
        *,
        config: BenchmarkConfig,
        planner: SafeDefaultPlanner | RulePlanner | LLMPlanner,
        auditor: NoOpAuditor | RuleAuditor | LLMAuditor,
        agentic_config: AgenticConfig,
    ) -> None:
        self.config = config
        self.planner = planner
        self.auditor = auditor
        self.agentic_config = agentic_config
        self.current_plan: PlannerOutput | None = None
        self.stats = {
            "plan_count": 0,
            "plan_valid_count": 0,
            "audit_call_count": 0,
            "revision_count": 0,
            "action_modification_count": 0,
            "action_delta_sum": 0.0,
            "llm_failure_count": 0,
            "mock_llm_used": False,
        }
        self._planner_failure_seen = 0
        self._auditor_failure_seen = 0

    def reset(self, env: Any) -> PlannerOutput:
        episode_context = build_episode_context(env)
        self.current_plan = self.planner.plan(episode_context)
        self.stats["plan_count"] += 1
        if getattr(self.planner, "last_valid", True):
            self.stats["plan_valid_count"] += 1
        planner_failures = int(getattr(self.planner, "failure_count", 0))
        self.stats["llm_failure_count"] += max(planner_failures - self._planner_failure_seen, 0)
        self._planner_failure_seen = planner_failures
        return self.current_plan

    def process(
        self,
        *,
        obs: np.ndarray,
        proposed_actual_action: np.ndarray,
        previous_action: np.ndarray,
        latest_info: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        if self.current_plan is None:
            raise RuntimeError("AgenticActionProcessor.reset() must be called before process().")

        original = np.asarray(proposed_actual_action, dtype=np.float32)
        bounded = apply_plan_bounds(original, self.current_plan, self.config)
        audit_called = False
        audit_decision = "not_called"
        audit_reason = "Audit trigger not reached."
        audit_payload: dict[str, Any] | None = None
        final = bounded

        if self.agentic_config.agentic_mode == "planner_auditor" and self.auditor.should_audit(
            obs,
            bounded,
            previous_action,
            self.current_plan,
        ):
            audit_called = True
            step_context = build_audit_context(obs, bounded, previous_action, self.current_plan, latest_info)
            audit = self.auditor.audit(step_context)
            audit_payload = audit.model_dump()
            audit_decision = audit.decision
            audit_reason = audit.reason
            self.stats["audit_call_count"] += 1
            if audit.decision == "revise":
                final = action_values_to_array(audit.final_action)
                self.stats["revision_count"] += 1

        delta = float(np.linalg.norm(final - original, ord=1))
        modified = bool(delta > 1e-6)
        if modified:
            self.stats["action_modification_count"] += 1
        self.stats["action_delta_sum"] += delta
        auditor_failures = int(getattr(self.auditor, "failure_count", 0))
        self.stats["llm_failure_count"] += max(auditor_failures - self._auditor_failure_seen, 0)
        self._auditor_failure_seen = auditor_failures

        info = {
            "llm_plan": self.current_plan.model_dump(),
            "llm_audit": audit_payload,
            "agentic_action_modified": modified,
            "agentic_original_action": action_dict(original),
            "agentic_final_action": action_dict(final),
        }
        log_row = {
            "plan": self.current_plan.model_dump(),
            "audit_called": audit_called,
            "audit_decision": audit_decision,
            "original_action": action_dict(original),
            "final_action": action_dict(final),
            "action_modified": modified,
            "action_delta": delta,
            "reason": audit_reason,
        }
        return encode_actual_action(final, self.config), info, log_row


def make_agentic_processor(
    *,
    config: BenchmarkConfig,
    agentic_config: AgenticConfig,
    llm_client: Any,
) -> AgenticActionProcessor:
    if agentic_config.planner == "rule":
        planner = RulePlanner(config)
    elif agentic_config.planner == "llm":
        planner = LLMPlanner(llm_client, config)
    else:
        planner = SafeDefaultPlanner(config)

    if agentic_config.auditor == "rule":
        auditor = RuleAuditor(config, agentic_config.audit_trigger)
    elif agentic_config.auditor == "llm":
        auditor = LLMAuditor(llm_client, config, agentic_config.audit_trigger)
    else:
        auditor = NoOpAuditor(config, agentic_config.audit_trigger)

    processor = AgenticActionProcessor(
        config=config,
        planner=planner,
        auditor=auditor,
        agentic_config=agentic_config,
    )
    processor.stats["mock_llm_used"] = bool(getattr(llm_client, "mock_used", False))
    return processor


def agentic_metrics(stats: dict[str, Any], step_count: int) -> dict[str, Any]:
    plan_count = int(stats.get("plan_count", 0))
    audit_count = int(stats.get("audit_call_count", 0))
    revision_count = int(stats.get("revision_count", 0))
    modification_count = int(stats.get("action_modification_count", 0))
    return {
        "plan_validity_rate": float(stats.get("plan_valid_count", 0) / max(plan_count, 1)),
        "audit_call_rate": float(audit_count / max(step_count, 1)),
        "revision_rate": float(revision_count / max(audit_count, 1)),
        "action_modification_rate": float(modification_count / max(step_count, 1)),
        "avg_action_delta_from_auditor": float(stats.get("action_delta_sum", 0.0) / max(step_count, 1)),
        "llm_failure_count": int(stats.get("llm_failure_count", 0)),
        "mock_llm_used": bool(stats.get("mock_llm_used", False)),
    }
