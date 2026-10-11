"""Retrieval metrics computed from document ids, with no model calls."""

from __future__ import annotations

from .common import result, skip


def _ranked(outputs: dict) -> list[str]:
    return list(outputs.get("retrieved_doc_ids", []))


def recall_at_k(outputs: dict, reference_outputs: dict, **_) -> dict:
    """Fraction of reference documents that were retrieved."""
    ref = set(reference_outputs.get("reference_doc_ids", []))
    if not ref:
        return skip("retrieval_recall", "no reference documents")
    hit = ref & set(_ranked(outputs))
    return result("retrieval_recall", len(hit) / len(ref), f"missing={sorted(ref - hit)}")


def precision_at_k(outputs: dict, reference_outputs: dict, **_) -> dict:
    """Fraction of retrieved documents that are in the reference set."""
    ref = set(reference_outputs.get("reference_doc_ids", []))
    got = _ranked(outputs)
    if not ref:
        return skip("retrieval_precision", "no reference documents")
    if not got:
        return result("retrieval_precision", 0.0, "nothing retrieved")
    return result("retrieval_precision", len([d for d in got if d in ref]) / len(got))


def reciprocal_rank(outputs: dict, reference_outputs: dict, **_) -> dict:
    ref = set(reference_outputs.get("reference_doc_ids", []))
    if not ref:
        return skip("retrieval_rr", "no reference documents")
    for rank, doc in enumerate(_ranked(outputs), start=1):
        if doc in ref:
            return result("retrieval_rr", 1.0 / rank, f"first relevant at rank {rank}")
    return result("retrieval_rr", 0.0, "no relevant document retrieved")


def retrieval_summary(outputs: list[dict], reference_outputs: list[dict]) -> list[dict]:
    """Experiment-level summary evaluator: mean reciprocal rank and hit rate over answerable examples."""
    rr, hits = [], []
    for out, ref in zip(outputs, reference_outputs):
        refs = set((ref or {}).get("reference_doc_ids", []))
        if not refs:
            continue
        got = _ranked(out or {})
        hits.append(1.0 if refs & set(got) else 0.0)
        rr.append(next((1.0 / i for i, d in enumerate(got, 1) if d in refs), 0.0))
    if not rr:
        return [skip("mrr", "no answerable examples")]
    return [
        {"key": "mrr", "score": sum(rr) / len(rr)},
        {"key": "hit_rate", "score": sum(hits) / len(hits)},
    ]


RETRIEVAL_EVALUATORS = [recall_at_k, precision_at_k, reciprocal_rank]
