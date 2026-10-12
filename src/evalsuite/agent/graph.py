"""LangGraph agent wiring and a normalized result record used by every evaluation."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langsmith import traceable

from ..config import get_settings
from ..guard import REFUSAL, InputGuard
from ..llms import get_app_model
from .tools import ALL_TOOLS, DOC_MARKER, SEARCH_TOOL


# Graph supersteps allowed per question (each tool round costs two). Bounds a looping model, which would
# otherwise burn tokens for as long as the provider keeps answering.
MAX_AGENT_STEPS = 14


def load_prompt(version: str | None = None) -> str:
    s = get_settings()
    return (s.prompts_dir / f"agent_{version or s.agent_prompt_version}.txt").read_text(encoding="utf-8").strip()


def build_agent(prompt_version: str | None = None, model=None):
    return create_agent(model or get_app_model(), ALL_TOOLS, system_prompt=load_prompt(prompt_version))


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieved_context: str = ""
    latency_s: float = 0.0
    total_tokens: int = 0
    terminated_early: bool = False
    guard_flagged: bool = False
    guard_blocked: bool = False
    guard_error: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)

    def to_outputs(self) -> dict[str, Any]:
        return asdict(self)


def message_text(content: Any) -> str:
    """Visible text of a message. Reasoning models return a list of typed blocks; only `text` blocks are the reply."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def to_openai_messages(messages) -> list[dict[str, Any]]:
    """Convert LangChain messages to the OpenAI dict format that agentevals consumes."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.type == "human":
            out.append({"role": "user", "content": str(m.content)})
        elif isinstance(m, AIMessage):
            msg: dict[str, Any] = {"role": "assistant", "content": message_text(m.content)}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                    for c in m.tool_calls
                ]
            out.append(msg)
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "content": str(m.content), "tool_call_id": m.tool_call_id})
    return out


def summarize(messages, latency_s: float) -> AgentResult:
    calls, doc_ids, context = [], [], []
    tokens = 0
    for m in messages:
        if isinstance(m, AIMessage):
            calls.extend({"name": c["name"], "args": c["args"]} for c in m.tool_calls)
            tokens += (m.usage_metadata or {}).get("total_tokens", 0)
        elif isinstance(m, ToolMessage):
            # Evidence documents come from search results and from tools that cite their source document.
            if m.name == SEARCH_TOOL:
                context.append(str(m.content))
            for d in DOC_MARKER.findall(str(m.content)):
                if d not in doc_ids:
                    doc_ids.append(d)
    final = next((message_text(m.content) for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), "")
    return AgentResult(
        answer=final,
        tool_calls=calls,
        retrieved_doc_ids=doc_ids,
        retrieved_context="\n\n".join(context),
        latency_s=round(latency_s, 3),
        total_tokens=tokens,
        messages=to_openai_messages(messages),
    )


@traceable(name="support_agent")
def run_agent(question: str, prompt_version: str | None = None, agent=None, guard: InputGuard | None = None) -> AgentResult:
    """Answer one question. With a guard, injected instructions are removed before the agent sees the message."""
    start = time.perf_counter()
    decision = guard.check(question) if guard else None
    if decision and decision.blocked:
        return AgentResult(
            answer=REFUSAL, latency_s=round(time.perf_counter() - start, 3), guard_flagged=decision.flagged,
            guard_blocked=True, guard_error=decision.error, messages=[{"role": "user", "content": question}],
        )
    agent = agent or build_agent(prompt_version)
    agent_input = decision.text if decision else question
    try:
        state = agent.invoke(
            {"messages": [{"role": "user", "content": agent_input}]},
            config={"recursion_limit": MAX_AGENT_STEPS, "configurable": {"user_question": question}},
        )
    except GraphRecursionError:
        return AgentResult(
            answer="", latency_s=round(time.perf_counter() - start, 3), terminated_early=True,
            guard_flagged=bool(decision and decision.flagged), messages=[{"role": "user", "content": question}],
        )
    result = summarize(state["messages"], time.perf_counter() - start)
    result.guard_flagged = bool(decision and decision.flagged)
    return result
