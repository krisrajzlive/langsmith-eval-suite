"""Evaluators for the order-operations agent. Outcome is judged on the end state of the environment."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

from ..evaluators.code import completed_within_step_limit, not_blocked_by_guard
from ..evaluators.common import result, skip
from .env import LOOKUP_TOOLS, MUTATING_TOOLS, seed_snapshot

_REJECTION_PREFIXES = (
    "not_authorized", "not_cancellable", "not_refundable", "not_editable", "invalid_amount", "confirmation_required",
)
_MUTATIONS_WITH_RULES = ("cancel_order", "refund_order", "update_address")


# ---- state comparison ----------------------------------------------------------------------------------
def _norm_refunds(refunds) -> list[list]:
    return sorted([str(oid), round(float(amt), 2)] for oid, amt in refunds)


def expected_snapshot(expected: dict[str, Any]) -> dict[str, Any]:
    """Seed state with the task's expected changes applied; anything not mentioned must stay untouched."""
    snap = copy.deepcopy(seed_snapshot())
    for order_id, fields in expected.get("orders", {}).items():
        snap["orders"][order_id].update(fields)
    snap["refunds"] = _norm_refunds(expected.get("refunds", []))
    snap["escalations"] = int(expected.get("escalations", 0))
    return snap


def _final(outputs: dict) -> dict[str, Any]:
    final = copy.deepcopy(outputs["final_state"])
    final["refunds"] = _norm_refunds(final.get("refunds", []))
    return final


def state_matches(outputs: dict, expected: dict[str, Any]) -> bool:
    return _final(outputs) == expected_snapshot(expected)


def unintended_changes(outputs: dict, expected: dict[str, Any]) -> list[str]:
    final, seed, allowed = _final(outputs), seed_snapshot(), expected_snapshot(expected)
    problems = [
        f"order {oid} changed"
        for oid, state in final["orders"].items()
        if oid not in expected.get("orders", {}) and state != seed["orders"][oid]
    ]
    problems += [f"unexpected refund {r}" for r in final["refunds"] if r not in allowed["refunds"]]
    if final["escalations"] > allowed["escalations"]:
        problems.append("unexpected escalation")
    return problems


# ---- per-example evaluators ----------------------------------------------------------------------------
def task_success(outputs: dict, reference_outputs: dict, **_) -> dict:
    ok = state_matches(outputs, reference_outputs["expected"])
    return result("task_success", ok, "" if ok else f"final={outputs['final_state']}")


def no_side_effects(outputs: dict, reference_outputs: dict, **_) -> dict:
    problems = unintended_changes(outputs, reference_outputs["expected"])
    return result("no_side_effects", not problems, "; ".join(problems))


def tool_selection(outputs: dict, reference_outputs: dict, **_) -> dict:
    """Every tool the reference needs was used, and no state-changing tool outside the reference was called."""
    wanted = {c["name"] for c in reference_outputs["reference_calls"]}
    called = {c["name"] for c in outputs.get("env_calls", [])}
    missing = wanted - called
    extra_mutations = (called & set(MUTATING_TOOLS)) - wanted
    return result("tool_selection", not missing and not extra_mutations, f"missing={sorted(missing)} extra={sorted(extra_mutations)}")


def no_rejected_mutations(outputs: dict, **_) -> dict:
    """The agent should read state first and not attempt changes the environment rejects."""
    rejected = [
        c for c in outputs.get("env_calls", [])
        if c["name"] in _MUTATIONS_WITH_RULES and c["error"].startswith(_REJECTION_PREFIXES)
    ]
    return result("no_rejected_mutations", not rejected, f"{[(c['name'], c['error'][:30]) for c in rejected]}")


def lookup_before_mutation(outputs: dict, **_) -> dict:
    names = [c["name"] for c in outputs.get("env_calls", [])]
    mutation_idx = next((i for i, n in enumerate(names) if n in _MUTATIONS_WITH_RULES), None)
    if mutation_idx is None:
        return skip("lookup_before_mutation", "no mutation attempted")
    looked_up = any(n in LOOKUP_TOOLS for n in names[:mutation_idx])
    return result("lookup_before_mutation", looked_up)


def efficiency(outputs: dict, reference_outputs: dict, **_) -> dict:
    n, cap = len(outputs.get("env_calls", [])), int(reference_outputs["max_calls"])
    return result("efficiency", n <= cap, f"{n} calls (cap {cap})")


def asked_for_clarification(outputs: dict, reference_outputs: dict, **_) -> dict:
    if not reference_outputs.get("must_ask"):
        return skip("asked_for_clarification", "not an ambiguous or unconfirmed request")
    asked = "?" in (outputs.get("answer") or "")
    return result("asked_for_clarification", asked)


def response_covers_facts(outputs: dict, reference_outputs: dict, **_) -> dict:
    needed = reference_outputs.get("response_contains", [])
    if not needed:
        return skip("response_covers_facts", "no required facts")
    answer = (outputs.get("answer") or "").lower()
    missing = [s for s in needed if s.lower() not in answer]
    return result("response_covers_facts", not missing, f"missing={missing}")


def retried_after_error(outputs: dict, reference_outputs: dict, **_) -> dict:
    if not reference_outputs.get("retry_required"):
        return skip("retried_after_error", "no fault injected")
    calls = outputs.get("env_calls", [])
    for i, c in enumerate(calls):
        if not c["ok"] and any(d["name"] == c["name"] and d["args"] == c["args"] for d in calls[i + 1:]):
            return result("retried_after_error", True)
    return result("retried_after_error", False, "never retried the failed call")


OPS_EVALUATORS = [
    task_success,
    no_side_effects,
    tool_selection,
    no_rejected_mutations,
    lookup_before_mutation,
    efficiency,
    asked_for_clarification,
    response_covers_facts,
    retried_after_error,
    completed_within_step_limit,
    not_blocked_by_guard,
]


# ---- reliability across repetitions --------------------------------------------------------------------
def reliability(pairs: list[tuple[str, bool]]) -> dict[str, float]:
    """pass@1 is the mean success rate; pass^k is the share of tasks that succeeded in every repetition."""
    by_task: dict[str, list[bool]] = defaultdict(list)
    for task_id, ok in pairs:
        by_task[task_id].append(bool(ok))
    if not by_task:
        return {}
    k = min(len(v) for v in by_task.values())
    return {
        "pass_at_1": sum(sum(v) / len(v) for v in by_task.values()) / len(by_task),
        "pass_hat_k": sum(all(v) for v in by_task.values()) / len(by_task),
        "flaky_rate": sum(0 < sum(v) < len(v) for v in by_task.values()) / len(by_task),
        "repetitions": float(k),
    }


def reliability_summary(runs, examples) -> list[dict]:
    """LangSmith summary evaluator: consistency across `num_repetitions` of every task."""
    pairs = [
        (str(ex.id), state_matches(run.outputs or {"final_state": {"orders": {}, "refunds": [], "escalations": 0}}, ex.outputs["expected"]))
        for run, ex in zip(runs, examples)
    ]
    stats = reliability(pairs)
    return [{"key": f"ops_{k}", "score": v} for k, v in stats.items()]
