"""Shared helpers for evaluator implementations."""

from __future__ import annotations

import re
from typing import Any

DOC_CITATION = re.compile(r"\[doc:([a-z0-9-]+)\]")


def skip(key: str, reason: str) -> dict[str, Any]:
    """Return a feedback record with no score, used when an evaluator does not apply to an example."""
    return {"key": key, "score": None, "comment": reason}


def result(key: str, passed: bool | float, comment: str = "") -> dict[str, Any]:
    return {"key": key, "score": passed, "comment": comment}


def tool_message_text(outputs: dict) -> str:
    return "\n".join(m.get("content", "") for m in outputs.get("messages", []) if m.get("role") == "tool")
