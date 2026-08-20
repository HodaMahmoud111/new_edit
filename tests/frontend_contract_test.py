"""Automated verification for frontend contract integrity and inspector requirements."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_routes_and_pages_exist():
    # التحقق من أن المسارات العامة والتقنية تعمل وتعيد 200
    assert client.get("/").status_code == 200
    assert client.get("/guide").status_code == 200
    assert client.get("/technical").status_code == 200

def test_ask_contract_response_keys():
    # التحقق من أن مسار الأسئلة يرجع الحقول المعتمدة دون أي تزييف
    res = client.post("/api/ask", json={"question": "What is the recommended exercise therapy for PFP?"})
    assert res.status_code == 200
    data = res.json()
    assert "recommendation" in data
    assert "evidence_excerpts" in data
    assert "confidence" in data
    assert "refusal" in data
    assert "safety_check" in data
    assert "diagnostic" in data

if __name__ == "__main__":
    test_routes_and_pages_exist()
    test_ask_contract_response_keys()
    print("All frontend contract checks passed successfully.")