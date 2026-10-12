"""Deterministic code evaluators. These double as unit-test assertions in CI."""

from __future__ import annotations

from .common import DOC_CITATION, result, skip, tool_message_text

LATENCY_BUDGET_S = 30.0
TOKEN_BUDGET = 6000
MAX_WORDS = 150


def answer_present(outputs: dict, **_) -> dict:
    answer = (outputs.get("answer") or "").strip()
    return result("answer_present", bool(answer), "" if answer else "empty final answer")


def citation_present(outputs: dict, reference_outputs: dict | None = None, **_) -> dict:
    """Answerable questions must carry at least one [doc:<id>] citation."""
    if reference_outputs is not None and not reference_outputs.get("reference_doc_ids"):
        return skip("citation_present", "unanswerable example; citations not required")
    cited = DOC_CITATION.findall(outputs.get("answer", ""))
    return result("citation_present", bool(cited), f"cited={cited}")


def citations_grounded(outputs: dict, **_) -> dict:
    """Every cited document id must appear in a tool result, so the agent cannot cite what it never saw."""
    cited = set(DOC_CITATION.findall(outputs.get("answer", "")))
    if not cited:
        return skip("citations_grounded", "no citations to verify")
    seen = set(DOC_CITATION.findall(tool_message_text(outputs)))
    phantom = sorted(cited - seen)
    return result("citations_grounded", not phantom, f"uncited-in-tools={phantom}" if phantom else "")


def within_latency_budget(outputs: dict, **_) -> dict:
    latency = float(outputs.get("latency_s", 0.0))
    return result("within_latency_budget", latency <= LATENCY_BUDGET_S, f"{latency:.2f}s")


def within_token_budget(outputs: dict, **_) -> dict:
    tokens = int(outputs.get("total_tokens", 0))
    if tokens == 0:
        return skip("within_token_budget", "provider did not report usage")
    return result("within_token_budget", tokens <= TOKEN_BUDGET, f"{tokens} tokens")


def completed_within_step_limit(outputs: dict, **_) -> dict:
    """Fails when the agent was cut off for looping instead of reaching a final answer."""
    cut = bool(outputs.get("terminated_early"))
    return result("completed_within_step_limit", not cut, "terminated by step limit" if cut else "")


def not_blocked_by_guard(outputs: dict, **_) -> dict:
    """Legitimate questions must reach the agent: a blocked benign request is a guard false positive."""
    blocked = bool(outputs.get("guard_blocked"))
    return result("not_blocked_by_guard", not blocked, outputs.get("guard_error", "") if blocked else "")


def answer_concise(outputs: dict, **_) -> dict:
    words = len((outputs.get("answer") or "").split())
    return result("answer_concise", words <= MAX_WORDS, f"{words} words")


CODE_EVALUATORS = [
    answer_present,
    completed_within_step_limit,
    not_blocked_by_guard,
    citation_present,
    citations_grounded,
    within_latency_budget,
    within_token_budget,
    answer_concise,
]
