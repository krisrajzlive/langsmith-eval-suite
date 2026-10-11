"""Offline retrieval quality: the lexical retriever must surface reference documents for the QA set."""

import pytest

from evalsuite.datasets import load_jsonl
from evalsuite.evaluators.retrieval import recall_at_k, reciprocal_rank, retrieval_summary
from evalsuite.rag import build_retriever, load_passages


@pytest.mark.unit
def test_passages_are_split_per_section():
    passages = load_passages()
    ids = [p.passage_id for p in passages]
    assert len(ids) == len(set(ids))
    assert {"refunds", "pricing", "errors"} <= {p.doc_id for p in passages}
    assert all(p.text.strip() for p in passages)


@pytest.mark.unit
def test_corpus_retrieval_hit_rate_and_mrr():
    retriever, outputs, refs = build_retriever(), [], []
    for row in load_jsonl("qa"):
        if not row["outputs"]["reference_doc_ids"]:
            continue
        hits = retriever.search(row["inputs"]["question"], 4)
        seen = []
        for passage, _ in hits:
            if passage.doc_id not in seen:
                seen.append(passage.doc_id)
        outputs.append({"retrieved_doc_ids": seen})
        refs.append(row["outputs"])
    summary = {r["key"]: r["score"] for r in retrieval_summary(outputs, refs)}
    assert summary["hit_rate"] >= 0.95, summary
    assert summary["mrr"] >= 0.70, summary


@pytest.mark.unit
def test_recall_and_rank_metrics():
    ref = {"reference_doc_ids": ["a", "b"]}
    out = {"retrieved_doc_ids": ["x", "b"]}
    assert recall_at_k(out, ref)["score"] == 0.5
    assert reciprocal_rank(out, ref)["score"] == 0.5
    assert recall_at_k(out, {"reference_doc_ids": []})["score"] is None
