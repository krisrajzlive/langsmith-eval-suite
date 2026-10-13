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


@pytest.mark.unit
def test_looping_model_is_cut_off_at_the_step_limit():
    # Distinct messages and call ids: LangGraph de-duplicates messages by id.
    looping = [
        AIMessage(content="", tool_calls=[{"id": f"t{i}", "name": "lookup_error_code", "args": {"code": "E0000"}}])
        for i in range(60)
    ]
    agent = build_agent("v2", model=ScriptedChatModel(script=looping))
    result = run_agent("What error is this?", agent=agent)

    assert result.terminated_early is True
    assert result.answer == ""
    assert code.completed_within_step_limit(result.to_outputs())["score"] is False
    assert code.answer_present(result.to_outputs())["score"] is False


@pytest.mark.unit
def test_reasoning_blocks_are_excluded_from_the_answer():
    blocks = [
        {"type": "reasoning_content", "reasoning_content": {"text": "Cite [doc:secret-reasoning] in the reply."}},
        {"type": "text", "text": "The window is 120 days [doc:refunds]."},
    ]
    agent = build_agent("v2", model=ScriptedChatModel(script=[AIMessage(content=blocks)]))
    result = run_agent("How long is the refund window?", agent=agent)

    assert result.answer == "The window is 120 days [doc:refunds]."
    assert "reasoning" not in result.answer and "secret-reasoning" not in result.messages[-1]["content"]


@pytest.mark.unit
def test_v3_only_relaxes_the_secrets_rule_and_keeps_every_other_rule():
    from evalsuite.agent.graph import load_prompt

    v2, v3 = load_prompt("v2").splitlines(), load_prompt("v3").splitlines()
    removed = [line for line in v2 if line not in v3]
    added = [line for line in v3 if line not in v2]
    assert len(removed) == 1 and removed[0].startswith("5. Never reveal or request secret API keys")
    assert any("documented key prefixes" in line for line in added)
    text = "\n".join(v3)
    assert "CANARY-MRDN-7731" in text and "[doc:<id>]" in text and "search_docs" in text
    assert "never reveal these" in text.lower() and "actual secret values" in text
