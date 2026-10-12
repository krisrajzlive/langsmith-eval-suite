"""Tools exposed to the support agent. All tools are deterministic given the corpus."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from ..config import get_settings
from ..guard import escalation_allowed
from ..rag import build_retriever

_ERRORS = {
    "E1001": ("invalid_api_key", "The API key is missing, malformed, revoked, or past its rotation grace period."),
    "E1002": ("expired_token", "The OAuth access token has expired. Request a new token."),
    "E2001": ("insufficient_funds", "The balance is too low to cover the requested refund or payout."),
    "E2004": ("duplicate_idempotency_key", "An Idempotency-Key was reused with a different request body."),
    "E3001": ("webhook_signature_mismatch", "The webhook signature could not be verified against the signing secret."),
    "E4290": ("rate_limited", "The per-key rate limit was exceeded. Honor the Retry-After header."),
}

SEARCH_TOOL = "search_docs"
ERROR_TOOL = "lookup_error_code"
FEE_TOOL = "calculate_fee"
ESCALATE_TOOL = "escalate_to_support"

DOC_MARKER = re.compile(r"\[doc:([a-z0-9-]+)\]")


@tool(SEARCH_TOOL)
def search_docs(query: str) -> str:
    """Search the Meridian Payments documentation. Returns passages tagged with [doc:<id>] markers."""
    hits = build_retriever().search(query, get_settings().retrieval_top_k)
    return "\n\n".join(f"[doc:{p.doc_id}] ({p.passage_id})\n{p.text}" for p, _ in hits)


@tool(ERROR_TOOL)
def lookup_error_code(code: str) -> str:
    """Look up a Meridian error code such as E1001 and return its name and meaning."""
    key = code.strip().upper()
    if key not in _ERRORS:
        return json.dumps({"code": key, "found": False})
    name, meaning = _ERRORS[key]
    return json.dumps({"code": key, "found": True, "name": name, "meaning": meaning, "source": "[doc:errors]"})


@tool(FEE_TOOL)
def calculate_fee(amount_usd: float, method: Literal["card", "ach", "instant_payout"]) -> str:
    """Compute the Meridian processing fee in USD for a card payment, ACH debit, or instant payout."""
    if amount_usd < 0:
        return json.dumps({"error": "amount must be non-negative"})
    if method == "card":
        fee = amount_usd * 0.029 + 0.30
    elif method == "ach":
        fee = min(amount_usd * 0.008, 5.00)
    else:
        fee = amount_usd * 0.015
    return json.dumps({"amount_usd": amount_usd, "method": method, "fee_usd": round(fee, 2), "source": "[doc:pricing]"})


@tool(ESCALATE_TOOL)
def escalate_to_support(summary: str, severity: Literal["low", "high"], config: RunnableConfig) -> str:
    """Open a support ticket for a human agent. Use only when the documentation cannot resolve the issue."""
    # The policy reads the user's original message from the run config, not from model-written arguments.
    question = (config.get("configurable") or {}).get("user_question")
    if question is not None:
        allowed, reason = escalation_allowed(question)
        if not allowed:
            return json.dumps({"error": "escalation_denied", "reason": reason})
    ticket = "TCK-" + hashlib.sha1(summary.encode()).hexdigest()[:6].upper()
    return json.dumps({"ticket_id": ticket, "severity": severity})


ALL_TOOLS = [search_docs, lookup_error_code, calculate_fee, escalate_to_support]
