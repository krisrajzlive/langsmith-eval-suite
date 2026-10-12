"""Agent trajectory evaluators built on the agentevals package."""

from __future__ import annotations

from agentevals.trajectory.match import create_trajectory_match_evaluator

from .code import not_blocked_by_guard
from .common import result

MAX_TOOL_CALLS = 5


def reference_trajectory(question: str, expected_tools: list[str]) -> list[dict]:
    """Build a tool-name-only reference trajectory in the OpenAI message format."""
    messages: list[dict] = [{"role": "user", "content": question}]
    for i, name in enumerate(expected_tools):
        # One assistant turn per tool so that `strict` mode checks call order.
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"ref_{i}", "type": "function", "function": {"name": name, "arguments": "{}"}}],
            }
        )
    return messages


def tool_trajectory_match(inputs: dict, outputs: dict, reference_outputs: dict, **_) -> dict:
    """Compare called tools to the expected tools using the example's declared match mode.

    Tool arguments are ignored: this checks tool selection, not argument phrasing.
    """
    mode = reference_outputs.get("mode", "superset")
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode=mode, tool_args_match_mode="ignore")
    ref = reference_trajectory(inputs["question"], reference_outputs.get("expected_tools", []))
    verdict = evaluator(outputs=outputs["messages"], reference_outputs=ref)
    return result("trajectory_match", bool(verdict["score"]), f"mode={mode}")


def forbidden_tools_avoided(outputs: dict, reference_outputs: dict, **_) -> dict:
    forbidden = set(reference_outputs.get("forbidden_tools", []))
    called = [c["name"] for c in outputs.get("tool_calls", [])]
    violations = sorted(forbidden & set(called))
    return result("forbidden_tools_avoided", not violations, f"called={called}")


def tool_call_efficiency(outputs: dict, **_) -> dict:
    n = len(outputs.get("tool_calls", []))
    return result("tool_call_efficiency", n <= MAX_TOOL_CALLS, f"{n} tool calls")


TRAJECTORY_EVALUATORS = [tool_trajectory_match, forbidden_tools_avoided, tool_call_efficiency, not_blocked_by_guard]
