"""Order-operations agent: no retrieval, only tools that act on an in-memory database."""

from __future__ import annotations

import time
from typing import Any

from langchain.agents import create_agent
from langgraph.errors import GraphRecursionError

from ..agent.graph import MAX_AGENT_STEPS, summarize
from ..config import get_settings
from ..guard import REFUSAL, InputGuard
from ..llms import get_app_model
from .env import OrdersEnv

OPS_PROMPT_VERSION = "v1"


def load_ops_prompt(version: str = OPS_PROMPT_VERSION) -> str:
    return (get_settings().prompts_dir / f"ops_{version}.txt").read_text(encoding="utf-8").strip()


def run_ops_task(
    inputs: dict[str, Any], model=None, guard: InputGuard | None = None, prompt_version: str = OPS_PROMPT_VERSION
) -> dict[str, Any]:
    """Run one task against a fresh environment and return the outputs evaluators need, including the end state."""
    env = OrdersEnv(inputs["customer_id"], faults=dict(inputs.get("faults") or {}))
    request = inputs["request"]
    start = time.perf_counter()

    decision = guard.check(request) if guard else None
    if decision and decision.blocked:
        out = {"answer": REFUSAL, "tool_calls": [], "messages": [], "terminated_early": False, "guard_blocked": True}
    else:
        agent = create_agent(model or get_app_model(), env.tools(), system_prompt=load_ops_prompt(prompt_version))
        try:
            state = agent.invoke(
                {"messages": [{"role": "user", "content": decision.text if decision else request}]},
                config={"recursion_limit": MAX_AGENT_STEPS},
            )
            out = summarize(state["messages"], 0.0).to_outputs()
        except GraphRecursionError:
            out = {"answer": "", "tool_calls": [], "messages": [], "terminated_early": True}
    out["latency_s"] = round(time.perf_counter() - start, 3)
    out["guard_flagged"] = bool(decision and decision.flagged)
    out["final_state"] = env.snapshot()
    out["env_calls"] = env.calls
    return out
