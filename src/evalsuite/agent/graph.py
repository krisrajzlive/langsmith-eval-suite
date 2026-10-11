"""LangGraph agent wiring and a normalized result record used by every evaluation."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langsmith import traceable

from ..config import get_settings
from ..llms import get_app_model
from .tools import ALL_TOOLS, DOC_MARKER, SEARCH_TOOL


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
    messages: list[dict[str, Any]] = field(default_factory=list)

    def to_outputs(self) -> dict[str, Any]:
        return asdict(self)


def to_openai_messages(messages) -> list[dict[str, Any]]:
    """Convert LangChain messages to the OpenAI dict format that agentevals consumes."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.type == "human":
            out.append({"role": "user", "content": str(m.content)})
        elif isinstance(m, AIMessage):
            msg: dict[str, Any] = {"role": "assistant", "content": str(m.content or "")}
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
    final = next((str(m.content) for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), "")
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
def run_agent(question: str, prompt_version: str | None = None, agent=None) -> AgentResult:
    agent = agent or build_agent(prompt_version)
    start = time.perf_counter()
    state = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return summarize(state["messages"], time.perf_counter() - start)
