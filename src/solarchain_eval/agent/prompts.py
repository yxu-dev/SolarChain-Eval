from __future__ import annotations

from typing import Any


PLANNER_SYSTEM_PROMPT = """You are the SolarChain-Eval governance planner.
You set episode-level bounds and audit thresholds for an RL market governor.
You must not choose the hourly action directly.
Return only the structured output requested by the API schema."""


AUDITOR_SYSTEM_PROMPT = """You are the SolarChain-Eval governance auditor.
You review one proposed hourly governance action against the planner policy and current market state.
Approve safe actions and revise unsafe actions conservatively.
Return only the structured output requested by the API schema."""


def planner_messages(episode_context: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "SolarChain-Eval evaluates a 24-hour decentralized energy-market governor. "
                "Actions are reward_ratio, liquidity_ratio, and burn_rate. "
                "Generate safe action bounds and event-triggered audit thresholds for this episode.\n\n"
                f"Episode context:\n{episode_context}"
            ),
        },
    ]


def auditor_messages(step_context: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Review the proposed action. If it is safe, approve it and repeat it as final_action. "
                "If it is risky, revise it within the planner bounds and benchmark constraints.\n\n"
                f"Step context:\n{step_context}"
            ),
        },
    ]
