"""CI gates that run the real agent. Skipped automatically when credentials are absent.

Each test logs inputs, outputs and feedback to LangSmith through the pytest integration, so a failing gate
can be inspected as a trace. Responses are cached when LANGSMITH_TEST_CACHE is set, which keeps CI cheap.
"""

from __future__ import annotations

import pytest
from langsmith import testing as t

from evalsuite.agent import build_agent, run_agent
from evalsuite.datasets import load_jsonl
from evalsuite.evaluators import code, safety

pytestmark = [pytest.mark.live]

QA = {r["id"]: r for r in load_jsonl("qa")}
SAFETY = load_jsonl("safety")
SMOKE_IDS = ["qa-002", "qa-004", "qa-010", "qa-015", "qa-016"]


@pytest.fixture(scope="module")
def agent():
    return build_agent()


@pytest.mark.langsmith(test_suite_name="meridian-ci-gates")
@pytest.mark.parametrize("example_id", SMOKE_IDS)
def test_answers_are_formatted_cited_and_grounded(agent, example_id):
    row = QA[example_id]
    t.log_inputs(row["inputs"])
    t.log_reference_outputs(row["outputs"])

    outputs = run_agent(row["inputs"]["question"], agent=agent).to_outputs()
    t.log_outputs({"answer": outputs["answer"]})

    checks = [
        code.answer_present(outputs),
        code.citation_present(outputs, row["outputs"]),
        code.citations_grounded(outputs),
        code.answer_concise(outputs),
    ]
    for check in checks:
        t.log_feedback(key=check["key"], score=check["score"])
    assert all(c["score"] in (True, None) for c in checks), [c for c in checks if c["score"] is False]


@pytest.mark.langsmith(test_suite_name="meridian-ci-gates")
@pytest.mark.parametrize("row", SAFETY, ids=[r["id"] for r in SAFETY])
def test_no_leakage_or_forbidden_tool_use(agent, row):
    t.log_inputs(row["inputs"])
    t.log_reference_outputs(row["outputs"])

    outputs = run_agent(row["inputs"]["question"], agent=agent).to_outputs()
    t.log_outputs({"answer": outputs["answer"]})

    leak = safety.no_forbidden_content(outputs, row["outputs"])
    tools = safety.safety_tools_avoided(outputs, row["outputs"])
    t.log_feedback(key=leak["key"], score=leak["score"])
    t.log_feedback(key=tools["key"], score=tools["score"])
    assert leak["score"], leak["comment"]
    assert tools["score"] in (True, None), tools["comment"]
