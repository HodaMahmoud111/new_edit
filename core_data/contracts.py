"""Dependency-free public response contracts shared by the service and local tests."""

from __future__ import annotations

from typing import Any


def verify_evidence(answer: dict[str, Any], results: list[tuple[dict[str, Any], float]]) -> dict[str, Any]:
    """Verify every returned quote and metadata tuple against retrieved source data."""
    lookup = {doc["metadata"]["content_hash"]: doc for doc, _ in results}
    checks = []
    for excerpt in answer.get("evidence_excerpts", []):
        source = lookup.get(excerpt.get("content_hash"))
        quote = excerpt.get("quote", "").strip()
        metadata_matches = bool(source) and all(
            excerpt.get(key) == source["metadata"].get(key)
            for key in ("document", "section", "page")
        )
        checks.append({
            "content_hash": excerpt.get("content_hash"),
            "quote_grounded": bool(source and quote and quote in source["original_text"]),
            "metadata_matches": metadata_matches,
        })
    return {
        "checks": checks,
        "passed": bool(checks)
        and all(check["quote_grounded"] and check["metadata_matches"] for check in checks),
    }


def safe_refusal(message: str) -> dict[str, Any]:
    """Return the required public fields for an intentionally withheld answer."""
    return {
        "recommendation": message,
        "evidence_excerpts": [],
        "confidence": "insufficient_evidence",
        "refusal": True,
        "safety_check": {"checks": [], "passed": False},
        "retriever": "scope_guard",
    }
