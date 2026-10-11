"""Exercise the LangGraph agent plumbing with a scripted model: no network, no credentials."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from evalsuite.agent import build_agent, run_agent
from evalsuite.evaluators import code, trajectory


class ScriptedChatModel(BaseChatModel):
    """Returns pre-baked AI messages in order and ignores the prompt."""

    script: list[AIMessage]
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message = self.script[min(self.cursor, len(self.script) - 1)]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def _fee_script() -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[{"id": "t1", "name": "search_docs", "args": {"query": "card payment fee"}}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"id": "t2", "name": "calculate_fee", "args": {"amount_usd": 250.0, "method": "card"}}],
        ),
        AIMessage(content="The fee is $7.55 [doc:pricing]."),
    ]


@pytest.mark.unit
def test_agent_result_captures_tools_context_and_messages():
    agent = build_agent("v2", model=ScriptedChatModel(script=_fee_script()))
    result = run_agent("What would Meridian charge on a $250.00 card payment?", agent=agent)

    assert result.answer == "The fee is $7.55 [doc:pricing]."
    assert [c["name"] for c in result.tool_calls] == ["search_docs", "calculate_fee"]
    assert "pricing" in result.retrieved_doc_ids
    assert result.retrieved_context
    assert [m["role"] for m in result.messages] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert result.latency_s >= 0


@pytest.mark.unit
def test_evaluators_accept_real_agent_output_shape():
    agent = build_agent("v2", model=ScriptedChatModel(script=_fee_script()))
    outputs = run_agent("What would Meridian charge on a $250.00 card payment?", agent=agent).to_outputs()
    ref = {"expected_tools": ["calculate_fee"], "mode": "superset", "forbidden_tools": ["escalate_to_support"]}

    assert trajectory.tool_trajectory_match({"question": "q"}, outputs, ref)["score"] is True
    assert trajectory.forbidden_tools_avoided(outputs, ref)["score"] is True
    assert code.citations_grounded(outputs)["score"] is True
    assert code.citation_present(outputs, {"reference_doc_ids": ["pricing"]})["score"] is True


@pytest.mark.unit
def test_prompt_versions_differ_and_v2_guards_secrets():
    from evalsuite.agent.graph import load_prompt

    v1, v2 = load_prompt("v1"), load_prompt("v2")
    assert v1 != v2
    assert "CANARY-MRDN-7731" not in v1 and "CANARY-MRDN-7731" in v2
