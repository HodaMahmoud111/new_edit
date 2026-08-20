"""Small deterministic helpers used by the RAG retrieval pipeline.

Keeping these functions model-agnostic makes the critical ranking behaviour easy
to unit-test without downloading embedding or reranking models.
"""

from __future__ import annotations

import re
from typing import Any


def clean_rewritten_query(original_question: str, model_text: str) -> str:
    """Return a single safe retrieval query, falling back to the original wording."""
    candidate = model_text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("```")[1] if "```" in candidate[3:] else candidate
    candidate = re.sub(r"^(query|rewritten query)\s*:\s*", "", candidate, flags=re.I)
    candidate = re.sub(r"\s+", " ", candidate).strip(" \"'`")
    # A very long answer is not a focused retrieval expansion; never use it.
    if 3 <= len(candidate) <= 400:
        return candidate
    return original_question


def rerank_documents(
    query: str,
    candidates: list[tuple[dict[str, Any], float]],
    reranker: Any,
    top_k: int,
) -> list[tuple[dict[str, Any], float]]:
    """Reorder candidate chunks with a CrossEncoder while retaining all metadata."""
    if not candidates:
        return []
    pairs = [(query, doc["original_text"]) for doc, _ in candidates]
    scores = reranker.predict(pairs, batch_size=8, show_progress_bar=False)
    ranked = sorted(
        ((doc, float(score)) for (doc, _), score in zip(candidates, scores)),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:top_k]
