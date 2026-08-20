"""Lightweight checks for the Final.ipynb-aligned backend contract."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_data.engine import RagEngine
from main import app


def test_structured_presentation() -> None:
    answer = {
        "recommendation": "Summary supported by the guideline.\n- First supported point\n- Second supported point",
        "evidence_excerpts": [
            {
                "document": "Patellofemoral-Pain.pdf",
                "section": "Exercise Therapy",
                "page": 71,
                "quote": "Exercise therapy is recommended.",
            }
        ],
    }
    presentation = RagEngine._presentation_from_answer(answer)
    assert presentation["summary"] == "Summary supported by the guideline."
    assert presentation["key_points"] == ["First supported point", "Second supported point"]
    assert presentation["citations"][0]["page"] == 71
    assert "• First supported point" in presentation["display_text"]


def test_required_routes_exist() -> None:
    routes = {route.path for route in app.routes}
    assert {"/api/ask", "/api/source-pdf", "/api/health", "/api/metrics"}.issubset(routes)


if __name__ == "__main__":
    test_structured_presentation()
    test_required_routes_exist()
    print("Backend contract checks passed.")
