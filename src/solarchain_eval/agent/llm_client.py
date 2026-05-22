from __future__ import annotations

import json
import os
from typing import Any

from .schemas import (
    ActionBounds,
    ActionValues,
    AuditPolicy,
    AuditorOutput,
    PlannerOutput,
    RiskAssessment,
    auditor_json_schema,
    planner_json_schema,
)


class MockLLMClient:
    """Deterministic schema-compatible client for local tests and missing credentials."""

    mock_used = True
    failure_count = 0

    def plan_structured(self, messages: list[dict[str, str]]) -> PlannerOutput:
        return PlannerOutput(
            governance_mode="mock_risk_bounded",
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
            rationale="Mock structured plan for deterministic local evaluation.",
        )

    def audit_structured(self, messages: list[dict[str, str]]) -> AuditorOutput:
        return AuditorOutput(
            decision="approve",
            final_action=ActionValues(reward_ratio=0.25, liquidity_ratio=0.75, burn_rate=0.02),
            risk_assessment=RiskAssessment(
                physics_risk="low",
                liquidity_risk="low",
                jitter_risk="low",
                fairness_risk="low",
            ),
            reason="Mock structured audit approves by default.",
        )


class OpenAICompatibleClient:
    """OpenAI SDK client using strict structured outputs with chat fallback."""

    mock_used = False

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        provider: str = "openai",
        timeout: float = 60.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime environment.
            raise RuntimeError("The openai package is required for OpenAICompatibleClient.") from exc

        self.provider = provider
        self.model = model
        self.failure_count = 0
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def plan_structured(self, messages: list[dict[str, str]]) -> PlannerOutput:
        try:
            return self._responses_parse(messages, PlannerOutput)
        except Exception:
            try:
                return PlannerOutput.model_validate(
                    self._chat_json_schema(messages, "planner_output", planner_json_schema())
                )
            except Exception:
                self.failure_count += 1
                raise

    def audit_structured(self, messages: list[dict[str, str]]) -> AuditorOutput:
        try:
            return self._responses_parse(messages, AuditorOutput)
        except Exception:
            try:
                return AuditorOutput.model_validate(
                    self._chat_json_schema(messages, "auditor_output", auditor_json_schema())
                )
            except Exception:
                self.failure_count += 1
                raise

    def _responses_parse(self, messages: list[dict[str, str]], model_type: type[Any]) -> Any:
        response = self._client.responses.parse(
            model=self.model,
            input=messages,
            text_format=model_type,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise ValueError("Responses structured parse returned no parsed output.")
        return parsed

    def _chat_json_schema(self, messages: list[dict[str, str]], schema_name: str, schema: dict[str, Any]) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
            temperature=0,
        )
        message = response.choices[0].message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise ValueError(f"Structured output refusal: {refusal}")
        content = message.content
        if not content:
            raise ValueError("Structured output response was empty.")
        return json.loads(content)


def make_llm_client(
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> MockLLMClient | OpenAICompatibleClient:
    resolved_provider = provider or os.getenv("SOLARCHAIN_LLM_PROVIDER") or "openai"
    resolved_api_key = api_key or os.getenv("SOLARCHAIN_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    resolved_base_url = base_url or os.getenv("SOLARCHAIN_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    resolved_model = model or os.getenv("SOLARCHAIN_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    if not resolved_api_key:
        return MockLLMClient()

    try:
        return OpenAICompatibleClient(
            api_key=resolved_api_key,
            model=resolved_model,
            base_url=resolved_base_url,
            provider=resolved_provider,
        )
    except RuntimeError:
        return MockLLMClient()
