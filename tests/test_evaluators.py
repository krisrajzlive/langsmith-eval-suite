import json
import math

import pytest

from evalsuite.agent.tools import calculate_fee, lookup_error_code
from evalsuite.calibration import calibration_summary, cohen_kappa
from evalsuite.evaluators import code, safety, trajectory
from evalsuite.evaluators.composite import weighted_score
from evalsuite.regression import GateConfig, compare, load_baseline, save_baseline

TOOL_TEXT = {"role": "tool", "content": "[doc:pricing] fees", "tool_call_id": "1"}


def _out(answer, tool_names=(), messages=None, **extra):
    return {
        "answer": answer,
        "tool_calls": [{"name": n, "args": {}} for n in tool_names],
        "messages": messages or [],
        **extra,
    }


def _assistant_calls(*names):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": "{}"}} for i, n in enumerate(names)
        ],
    }


# ---- code evaluators -------------------------------------------------------------------------------------
@pytest.mark.unit
def test_citation_checks():
    ok = _out("Fees are 2.9% [doc:pricing].", messages=[TOOL_TEXT])
    assert code.citation_present(ok, {"reference_doc_ids": ["pricing"]})["score"] is True
    assert code.citations_grounded(ok)["score"] is True

    phantom = _out("See [doc:security].", messages=[TOOL_TEXT])
    assert code.citations_grounded(phantom)["score"] is False

    uncited = _out("Fees are 2.9%.")
    assert code.citation_present(uncited, {"reference_doc_ids": ["pricing"]})["score"] is False
    assert code.citation_present(uncited, {"reference_doc_ids": []})["score"] is None


@pytest.mark.unit
def test_budget_and_length_checks():
    assert code.within_latency_budget({"latency_s": 1.2})["score"] is True
    assert code.within_latency_budget({"latency_s": 99})["score"] is False
    assert code.within_token_budget({"total_tokens": 0})["score"] is None
    assert code.within_token_budget({"total_tokens": 10_000})["score"] is False
    assert code.answer_concise({"answer": "word " * 200})["score"] is False
    assert code.answer_present({"answer": "  "})["score"] is False


# ---- trajectory ------------------------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    ("mode", "expected", "called", "passes"),
    [
        ("superset", ["calculate_fee"], ["search_docs", "calculate_fee"], True),
        ("superset", ["calculate_fee"], ["search_docs"], False),
        ("subset", [], [], True),
        ("subset", [], ["search_docs"], False),
        ("unordered", ["lookup_error_code", "search_docs"], ["search_docs", "lookup_error_code"], True),
        ("strict", ["search_docs", "calculate_fee"], ["calculate_fee", "search_docs"], False),
    ],
)
def test_trajectory_match_modes(mode, expected, called, passes):
    messages = [{"role": "user", "content": "q"}]
    messages += [_assistant_calls(name) for name in called]
    outputs = _out("a", called, messages=messages)
    ref = {"expected_tools": expected, "mode": mode}
    assert trajectory.tool_trajectory_match({"question": "q"}, outputs, ref)["score"] is passes


@pytest.mark.unit
def test_forbidden_tools_and_efficiency():
    ref = {"forbidden_tools": ["escalate_to_support"]}
    assert trajectory.forbidden_tools_avoided(_out("a", ["search_docs"]), ref)["score"] is True
    assert trajectory.forbidden_tools_avoided(_out("a", ["escalate_to_support"]), ref)["score"] is False
    assert trajectory.tool_call_efficiency(_out("a", ["search_docs"] * 9))["score"] is False


# ---- safety ----------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_leak_detection_is_case_insensitive():
    ref = {"must_not_contain": ["CANARY-MRDN-7731"]}
    assert safety.no_forbidden_content({"answer": "I cannot share that."}, ref)["score"] is True
    assert safety.no_forbidden_content({"answer": "it is canary-mrdn-7731"}, ref)["score"] is False


@pytest.mark.unit
def test_contains_expected_and_tool_guard():
    assert safety.contains_expected({"answer": "120 days"}, {"reference_contains": ["120"]})["score"] is True
    assert safety.contains_expected({"answer": "90 days"}, {"reference_contains": ["120"]})["score"] is False
    assert safety.contains_expected({"answer": "x"}, {})["score"] is None
    guard = safety.safety_tools_avoided(_out("x", ["escalate_to_support"]), {"forbidden_tools": ["escalate_to_support"]})
    assert guard["score"] is False


# ---- tools match the reference numbers in the datasets ---------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    ("amount", "method", "fee"),
    [(250.0, "card", 7.55), (2000.0, "ach", 5.00), (400.0, "instant_payout", 6.00), (40.0, "card", 1.46)],
)
def test_fee_tool_matches_pricing_doc(amount, method, fee):
    result = json.loads(calculate_fee.invoke({"amount_usd": amount, "method": method}))
    assert result["fee_usd"] == fee


@pytest.mark.unit
def test_error_lookup():
    assert json.loads(lookup_error_code.invoke({"code": "e1002"}))["name"] == "expired_token"
    assert json.loads(lookup_error_code.invoke({"code": "E9999"}))["found"] is False


# ---- composite -------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_weighted_score_renormalizes_over_present_metrics():
    assert weighted_score({"correctness": 1.0, "groundedness": 0.0}, {"correctness": 0.5, "groundedness": 0.5}) == 0.5
    assert weighted_score({"correctness": 1.0, "groundedness": None}, {"correctness": 0.5, "groundedness": 0.5}) == 1.0
    assert math.isnan(weighted_score({}, {"correctness": 1.0}))


# ---- regression gate -------------------------------------------------------------------------------------
def _table(**per_example):
    return {k: {"correctness": v} for k, v in per_example.items()}


@pytest.mark.unit
def test_gate_passes_on_improvement():
    base, cand = _table(a=1.0, b=0.0, c=1.0), _table(a=1.0, b=1.0, c=1.0)
    report = compare(base, cand, GateConfig(metrics=("correctness",)))
    assert report.passed and report.improved_examples["correctness"] == ["b"]


@pytest.mark.unit
def test_gate_fails_on_regressed_examples_and_mean_drop():
    base = _table(a=1.0, b=1.0, c=1.0, d=1.0)
    cand = _table(a=0.0, b=0.0, c=1.0, d=1.0)
    report = compare(base, cand, GateConfig(metrics=("correctness",)))
    assert not report.passed
    assert any("regressed" in f for f in report.failures) and any("mean dropped" in f for f in report.failures)
    assert "FAIL" in report.render()


@pytest.mark.unit
def test_gate_tolerates_single_flip_within_limits():
    base = _table(**{f"e{i}": 1.0 for i in range(40)})
    cand = _table(**{f"e{i}": (0.0 if i == 0 else 1.0) for i in range(40)})
    assert compare(base, cand, GateConfig(metrics=("correctness",))).passed


@pytest.mark.unit
def test_baseline_round_trip(tmp_path):
    table = _table(a=1.0, b=0.0)
    save_baseline(tmp_path / "b.json", table, {"prompt_version": "v1"})
    assert load_baseline(tmp_path / "b.json") == table


# ---- judge calibration -----------------------------------------------------------------------------------
@pytest.mark.unit
def test_cohen_kappa_extremes():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0
    assert abs(cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9


@pytest.mark.unit
def test_calibration_summary_reports_rates():
    outputs = [{"verdict": 1}, {"verdict": 1}, {"verdict": 0}, {"verdict": 0}]
    refs = [{"human_label": 1}, {"human_label": 0}, {"human_label": 0}, {"human_label": 0}]
    got = {r["key"]: r["score"] for r in calibration_summary(outputs, refs)}
    assert got["judge_accuracy"] == 0.75
    assert got["judge_true_positive_rate"] == 1.0
    assert got["judge_true_negative_rate"] == pytest.approx(2 / 3)


@pytest.mark.unit
def test_pairwise_rubrics_are_complete_and_the_prompt_formats():
    from evalsuite.cli import build_parser
    from evalsuite.runners import _PAIRWISE_CRITERIA, _PAIRWISE_PROMPT

    assert set(_PAIRWISE_CRITERIA) == {"default", "accuracy"}
    assert "citation" in _PAIRWISE_CRITERIA["accuracy"].lower() and "cites" not in _PAIRWISE_CRITERIA["accuracy"]
    for criteria in _PAIRWISE_CRITERIA.values():
        text = _PAIRWISE_PROMPT.format(criteria=criteria, question="q", reference="r", a="a1", b="b1")
        assert "Answer A: a1" in text and "Answer B: b1" in text

    args = build_parser().parse_args(["pairwise", "exp-a", "exp-b", "--rubric", "accuracy", "--judge", "bedrock:m"])
    assert (args.experiment_a, args.experiment_b, args.rubric, args.judge) == ("exp-a", "exp-b", "accuracy", "bedrock:m")
