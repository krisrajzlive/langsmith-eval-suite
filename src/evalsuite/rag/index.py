"""Corpus loading and retrieval.

Two interchangeable backends share one interface:
  * lexical - offline TF-IDF cosine similarity (deterministic, no network)
  * dense   - FAISS over embeddings from Hugging Face or OpenAI
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..llms import get_embeddings

_TOKEN = re.compile(r"[a-z0-9_$%.]+")
_STOP = frozenset(
    "a an and are as at be by can do does for from how i in is it of on or the to what when which with you your".split()
)


@dataclass(frozen=True)
class Passage:
    doc_id: str  # e.g. "rate-limits"
    passage_id: str  # e.g. "rate-limits#2"
    text: str


def _tokens(text: str) -> list[str]:
    return [t.strip(".") for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def load_passages(corpus_dir: Path | None = None) -> list[Passage]:
    """Split each markdown file into passages at level-2 headings.

    Documents without sub-sections become a single passage. The title is prepended to every passage.
    """
    corpus_dir = corpus_dir or get_settings().corpus_dir
    passages: list[Passage] = []
    for path in sorted(corpus_dir.glob("*.md")):
        doc_id = path.stem
        sections = re.split(r"\n(?=## )", path.read_text(encoding="utf-8").strip())
        title = sections[0].splitlines()[0].lstrip("# ").strip()
        intro = "\n".join(sections[0].splitlines()[1:]).strip()
        chunks = [intro] if intro else []
        chunks += sections[1:]
        for i, chunk in enumerate(chunks):
            passages.append(Passage(doc_id, f"{doc_id}#{i}", f"{title}\n{chunk.strip()}"))
    return passages


class Retriever:
    def __init__(self, passages: list[Passage]):
        self.passages = passages

    def search(self, query: str, k: int) -> list[tuple[Passage, float]]:
        raise NotImplementedError


class LexicalRetriever(Retriever):
    def __init__(self, passages: list[Passage]):
        super().__init__(passages)
        docs = [Counter(_tokens(p.text)) for p in passages]
        df: Counter[str] = Counter()
        for d in docs:
            df.update(d.keys())
        n = len(passages)
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self._vecs = [self._vectorize(d) for d in docs]

    def _vectorize(self, counts: Counter[str]) -> dict[str, float]:
        vec = {t: (1 + math.log(c)) * self._idf.get(t, 0.0) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, k: int) -> list[tuple[Passage, float]]:
        q = self._vectorize(Counter(_tokens(query)))
        scored = [(p, sum(w * v.get(t, 0.0) for t, w in q.items())) for p, v in zip(self.passages, self._vecs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


class DenseRetriever(Retriever):
    def __init__(self, passages: list[Passage], embeddings):
        import faiss

        super().__init__(passages)
        self._emb = embeddings
        matrix = np.array(embeddings.embed_documents([p.text for p in passages]), dtype="float32")
        faiss.normalize_L2(matrix)
        self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._index.add(matrix)
        self._faiss = faiss

    def search(self, query: str, k: int) -> list[tuple[Passage, float]]:
        q = np.array([self._emb.embed_query(query)], dtype="float32")
        self._faiss.normalize_L2(q)
        scores, idx = self._index.search(q, k)
        return [(self.passages[i], float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]


@lru_cache
def build_retriever() -> Retriever:
    passages = load_passages()
    embeddings = get_embeddings()
    return LexicalRetriever(passages) if embeddings is None else DenseRetriever(passages, embeddings)
