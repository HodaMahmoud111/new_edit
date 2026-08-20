"""RAG engine for the approved PFP Clinical Practice Guideline.

The clinical workflow intentionally mirrors ``Final.ipynb``:
1. validate the active PDF source
2. classify safety risk with rules plus the lightweight provider
3. retrieve three unique passages with Hybrid RRF by default
4. apply the lightweight answerability gate
5. use Gemini only for the final evidence-grounded answer
6. verify every displayed quotation locally before returning it

The API exposes a small, safe ``diagnostic`` object.  It identifies the failed
decision layer for the UI without returning exception text, API keys, prompts,
or stack traces.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import pickle
import re
import shutil
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pymupdf
import requests
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .contracts import verify_evidence
from .retrieval_utils import clean_rewritten_query, rerank_documents


LOGGER = logging.getLogger("pfp_rag_service")

DOC_NAME = "Patellofemoral Pain CPG (APTA/JOSPT 2019)"
PDF_URL = "https://ifspt.org/wp-content/uploads/2025/05/Patellofemoral-Pain.pdf"
CHUNK_SIZE = 850
CHUNK_OVERLAP = 150
TOP_K = 3
WIDE_TOP_K = 15
RRF_K = 60
RERANK_CANDIDATE_K = int(os.getenv("PFP_RERANK_CANDIDATE_K", "30"))
EMBED_MODEL = os.getenv("PFP_EMBED_MODEL", "BAAI/bge-m3")
FINAL_GEMINI_MODEL = os.getenv("FINAL_GEMINI_MODEL", "gemini-3.5-flash")
LIGHT_LLM_BASE_URL = os.getenv("LIGHT_LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
LIGHT_LLM_MODEL_REQUESTED = os.getenv("LIGHT_LLM_MODEL", "openai/gpt-oss-20b")
PREFERRED_LIGHT_MODELS = (
    "openai/gpt-oss-20b",
    "groq/compound-mini",
    "llama-3.3-70b-versatile",
)
RERANK_MODEL = os.getenv("PFP_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
EMBED_BATCH_SIZE = int(os.getenv("PFP_EMBED_BATCH_SIZE", "4"))
MAX_SOURCE_PDF_BYTES = 15 * 1024 * 1024
DATA_DIR = Path(os.getenv("PFP_RAG_DATA_DIR", str(Path(__file__).resolve().parent / "data")))
PDF_PATH = DATA_DIR / "patellofemoral_pain_cpg.pdf"
CHUNKS_PATH = DATA_DIR / "chunks.json"
VECTORS_PATH = DATA_DIR / "embeddings.npz"
BM25_PATH = DATA_DIR / "bm25.pkl"
METRICS_PATH = Path(__file__).parent / "evaluation_metrics.json"
SERVICE_TOKEN = os.getenv("PFP_RAG_SERVICE_TOKEN")
# Support the explicit Gemini variable used in the local setup while remaining
# backwards-compatible with Google SDK examples that use GOOGLE_API_KEY.
# The value is never returned, logged, or included in a diagnostic payload.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
LIGHT_LLM_API_KEY = os.getenv("LIGHT_LLM_API_KEY")

HEADING_PATTERN = re.compile(
    r"^(SUMMARY OF RECOMMENDATIONS|INTRODUCTION|METHODS|CLINICAL COURSE|RISK FACTORS?|"
    r"PATHOANATOMICAL FEATURES|EXAMINATION(?: -[A-Z ]+)?|DIAGNOSIS/CLASSIFICATION|"
    r"DIFFERENTIAL DIAGNOSIS|INTERVENTIONS?|EXERCISE THERAPY|REFERENCES|APPENDICES.*)$",
    re.IGNORECASE,
)
MEDICATION_PATTERN = re.compile(
    r"\b(dosage|dose|mg\b|milligram|ibuprofen|naproxen|paracetamol|acetaminophen|nsaid|medication|prescription)\b"
    r"|(?:جرعة|جرعات)\s*(?:دواء|دوائية|مسكن)",
    re.I,
)
PFP_PATTERN = re.compile(
    r"\b(patellofemoral|pfp|anterior knee pain|patellar)\b"
    r"|ألم\s+(?:المفصل\s+)?الرضفي\s+الفخذي|ألم\s+أمام\s+الركبة|الرضفة",
    re.I,
)
OUT_OF_SCOPE_PATTERN = re.compile(
    r"\b(acl|meniscus|knee replacement|arthroplasty|post[-\s]?operative|post[-\s]?surgery)\b"
    r"|الرباط\s+الصليبي|الغضروف\s+الهلالي|استبدال\s+الركبة|بعد\s+الجراحة",
    re.I,
)
PERSONAL_PLAN_PATTERN = re.compile(
    r"\b(diagnose me|my diagnosis|for me|my patient|should i|what should i do|treatment plan|personalized)\b"
    r"|شخصي(?:ة)?|تشخيصي|شخ[ّ]?صيني|لحالتي|حالتي|جرعات\s+التمارين|تمارين\s+مناسبة",
    re.I,
)
EMERGENCY_PATTERN = re.compile(
    r"\b(emergency|cannot bear weight|severe trauma|fever|red hot swollen)\b"
    r"|إصابة\s+شديدة|ألم\s+شديد\s+بعد\s+إصابة|تورم\s+سريع|احمرار|سخونة|عدم\s+(?:القدرة|قدرتي)\s+على\s+(?:المشي|تحمل\s+الوزن)",
    re.I,
)
ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF]")
RECOMMENDATION_PATTERN = re.compile(
    r"\b(?:clinicians?\s+(?:should|may)|recommend(?:ed|ation)?|should\s+(?:not|include|prescribe|be used)|may\s+use)\b",
    re.I,
)
DOCUMENT_HEADER_PATTERN = re.compile(
    r"\b(?:journal of orthopaedic|volume\s+\d+|september\s+2019|clinical practice guidelines|cpg\d*|doi\s*:)\b",
    re.I,
)
RETRIEVAL_STOPWORDS = frozenset(
    {
        "about", "according", "approach", "approaches", "based", "does", "evidence", "for",
        "from", "guideline", "guidelines", "have", "what", "which", "with", "would", "2019",
        "patellofemoral", "pain", "pfp", "patient", "patients", "people", "recommend", "support",
        "therapy", "treatment",
    }
)


Layer = Literal["input", "risk_policy", "retrieval", "answerability", "generation", "citation", "document", "complete"]


@dataclass(frozen=True)
class LayerFailure(Exception):
    layer: Layer
    code: str
    public_message: str
    recoverable: bool = False

    def __str__(self) -> str:
        return f"{self.layer}:{self.code}"


def diagnostic(layer: Layer, code: str, recoverable: bool = False) -> dict[str, Any]:
    """Return only UI-safe status metadata; detailed exceptions stay in server logs."""
    return {"layer": layer, "code": code, "recoverable": recoverable}


def _safe_provider_status(error: Exception) -> str:
    """Map a provider failure to a stable, secret-free diagnostic code.

    Provider exception text may include request identifiers, endpoint details, or
    implementation-specific information.  The public health result intentionally
    exposes only one of these short categories.
    """
    status_code = getattr(error, "code", None) or getattr(error, "status_code", None)
    try:
        status_code = int(status_code)
    except (TypeError, ValueError):
        status_code = None
    if status_code in {401, 403}:
        return "credentials_rejected"
    if status_code == 404:
        return "model_unavailable"
    if status_code == 429:
        return "rate_limited"
    if status_code is not None and status_code >= 500:
        return "provider_unreachable"
    if isinstance(error, (requests.ConnectionError, requests.Timeout)):
        return "endpoint_unreachable"
    if isinstance(error, ValueError):
        return "invalid_provider_response"
    return "provider_check_failed"


def detect_response_language(question: str, preferred_language: str | None = None) -> str:
    if preferred_language in {"Arabic", "English"}:
        return preferred_language
    return "Arabic" if ARABIC_PATTERN.search(question) else "English"


def localize_response_message(message: str, response_language: str) -> str:
    translations = {
        "scope": "هذا النظام يعتمد فقط على دليل ألم المفصل الرضفي الفخذي ولا يمكنه تقديم تشخيص أو خطة علاج شخصية أو إرشاد لحالة مختلفة.",
        "broad": "هذا السؤال أوسع من نطاق دليل ألم المفصل الرضفي الفخذي. اطرحي سؤالًا محددًا عن PFP للحصول على إجابة مستندة إلى الدليل.",
        "insufficient": "المقاطع المسترجعة من دليل PFP لا تكفي لدعم إجابة محددة عن هذا السؤال.",
        "generation": "تعذر إنشاء الإجابة المستندة إلى الدليل بأمان في هذه اللحظة. حاولي مرة أخرى لاحقًا.",
        "citation": "لم تُعرض الإجابة لأن الاستشهادات لم يمكن التحقق منها مقابل المقاطع المسترجعة من الدليل.",
        "risk": "لم يمكن التحقق من ملاءمة هذا السؤال للدليل بأمان. اطرحي سؤالًا محددًا عن PFP أو حاولي مرة أخرى لاحقًا.",
        "personal": "لا يمكن للنظام تقديم تشخيص فردي أو جرعات تمارين مخصصة. يمكنه عرض معلومات عامة من دليل ألم المفصل الرضفي الفخذي، وللتقييم المناسب يُرجى مراجعة مختص مؤهل.",
        "urgent": "الأعراض المذكورة قد تحتاج إلى تقييم طبي عاجل اليوم، خصوصًا بعد إصابة مع تورم سريع أو احمرار أو سخونة أو صعوبة في المشي. لا تعتمدي على هذه المحادثة للتشخيص أو لتأخير طلب الرعاية العاجلة.",
    }
    if response_language == "Arabic":
        return translations.get(message, message)
    messages = {
        "scope": "This system uses only the Patellofemoral Pain guideline and cannot provide diagnosis, patient-specific management, or guidance for a different condition.",
        "broad": "This question is broader than the Patellofemoral Pain guideline. Ask a PFP-specific evidence question for a guideline-grounded response.",
        "insufficient": "The retrieved PFP guideline passages are not sufficient to support a specific answer to this question.",
        "generation": "The evidence answer could not be generated safely at this time. Please try again shortly.",
        "citation": "The answer was withheld because the guideline citations could not be verified against the retrieved passages.",
        "risk": "The question could not be assessed safely for this guideline. Ask a PFP-specific question or try again shortly.",
        "personal": "This system cannot provide an individual diagnosis or personalised exercise dosage. It can provide general PFP-guideline information; please consult a qualified clinician for an appropriate assessment.",
        "urgent": "The symptoms described may need urgent in-person medical assessment today, particularly after an injury with rapid swelling, redness, warmth, or difficulty walking. Do not use this conversation to diagnose the problem or delay urgent care.",
    }
    return messages.get(message, message)


def clean_source_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_retrieval(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower()


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= size:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        parts.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return [part for part in parts if part]


def request_pdf() -> None:
    if PDF_PATH.exists() and PDF_PATH.stat().st_size > 100_000:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(PDF_URL, timeout=90)
    response.raise_for_status()
    PDF_PATH.write_bytes(response.content)


def parse_chunks() -> list[dict[str, Any]]:
    pdf_hash = hashlib.sha256(PDF_PATH.read_bytes()).hexdigest()
    if CHUNKS_PATH.exists():
        cached = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        if cached.get("pdf_hash") == pdf_hash and cached.get("size") == CHUNK_SIZE and cached.get("overlap") == CHUNK_OVERLAP:
            return cached["chunks"]

    current_section = "GENERAL"
    chunks: list[dict[str, Any]] = []
    with pymupdf.open(PDF_PATH) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            original_page = clean_source_text(page.get_text("text"))
            for sentence in original_page.split(". "):
                if HEADING_PATTERN.match(sentence.strip()):
                    current_section = sentence.strip().title()
            for original_text in split_text(original_page):
                normalized = normalize_for_retrieval(original_text)
                chunks.append(
                    {
                        "original_text": original_text,
                        "normalized_text": normalized,
                        "retrieval_tokens": re.findall(r"[\w-]+", normalized),
                        "metadata": {
                            "content_hash": stable_id(normalized),
                            "document": DOC_NAME,
                            "section": current_section,
                            "page": page_number,
                        },
                    }
                )
    chunks = list({chunk["metadata"]["content_hash"]: chunk for chunk in chunks}.values())
    CHUNKS_PATH.write_text(
        json.dumps({"pdf_hash": pdf_hash, "size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP, "chunks": chunks}),
        encoding="utf-8",
    )
    return chunks


class LightProvider:
    """OpenAI-compatible client used only for lightweight notebook tasks."""

    def __init__(self) -> None:
        if not LIGHT_LLM_API_KEY:
            raise RuntimeError("LIGHT_LLM_API_KEY is required for risk, answerability, and PDF validation.")
        self.headers = {"Authorization": f"Bearer {LIGHT_LLM_API_KEY}", "Content-Type": "application/json"}
        self.model, self.selection_note = self._choose_live_model()

    def _choose_live_model(self) -> tuple[str, str]:
        try:
            response = requests.get(f"{LIGHT_LLM_BASE_URL}/models", headers=self.headers, timeout=20)
            response.raise_for_status()
            available = {str(item.get("id")) for item in response.json().get("data", []) if item.get("id")}
        except Exception as exc:
            raise RuntimeError("The lightweight provider could not list active models.") from exc
        if LIGHT_LLM_MODEL_REQUESTED in available:
            return LIGHT_LLM_MODEL_REQUESTED, "requested_model_available"
        for candidate in PREFERRED_LIGHT_MODELS:
            if candidate in available:
                return candidate, "preferred_fallback_selected"
        raise RuntimeError("No general text model is available for the lightweight tasks.")

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        content = text.strip()
        if content.startswith("```"):
            content = content.split("```", 2)[1] if content.count("```") >= 2 else content
            content = re.sub(r"^json", "", content, flags=re.I).strip()
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            content = content[start : end + 1]
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        return payload

    def json(self, prompt: str, max_retries: int = 3) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{LIGHT_LLM_BASE_URL}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "Return exactly one valid JSON object. Do not include markdown, explanations, or text outside JSON.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.01,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=45,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return self._extract_json(str(content))
            except Exception as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(1 + attempt)
        if isinstance(last_error, ValueError):
            raise ValueError("Lightweight provider did not return the required JSON object.") from last_error
        raise RuntimeError("Lightweight JSON call failed after retries.") from last_error


class UnavailableLightProvider:
    """Keeps the local safety fallbacks usable when the optional provider is down."""

    model = "unavailable"
    selection_note = "provider_unavailable"

    def __init__(self, status: str) -> None:
        self.status = status

    def json(self, _: str, max_retries: int = 3) -> dict[str, Any]:
        del max_retries
        raise RuntimeError("Lightweight provider is unavailable.")


class RagEngine:
    def __init__(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        request_pdf()
        self.chunks = parse_chunks()
        self.recommendation_index = self._build_recommendation_index()
        self.bm25 = self._load_or_create_bm25()
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.embeddings = self._load_or_create_embeddings()
        self.collection = f"pfp_guideline_{stable_id(EMBED_MODEL + ''.join(c['metadata']['content_hash'] for c in self.chunks))}"
        self.qdrant = QdrantClient(path=str(DATA_DIR / "qdrant"))
        self._ensure_collection()
        try:
            self.light: LightProvider | UnavailableLightProvider = LightProvider()
        except Exception as exc:
            light_status = "key_missing" if not LIGHT_LLM_API_KEY else _safe_provider_status(exc)
            self.light = UnavailableLightProvider(light_status)
            LOGGER.warning("LIGHT_LLM startup connectivity: %s", light_status)

        self.gemini = None
        self._gemini_client_status = "key_missing" if not GEMINI_API_KEY else "client_not_checked"
        if GEMINI_API_KEY:
            try:
                self.gemini = genai.Client(api_key=GEMINI_API_KEY)
            except Exception as exc:
                self._gemini_client_status = _safe_provider_status(exc)
                LOGGER.warning("GEMINI client initialization: %s", self._gemini_client_status)

        self.provider_connectivity = self._check_provider_connectivity()
        self.reranker: CrossEncoder | None = None
        self.metrics = self._load_metrics()

    def _check_provider_connectivity(self) -> dict[str, str]:
        """Perform startup-only, secret-free provider checks for local setup support.

        The checks deliberately validate account access to the configured model,
        rather than consuming a clinical-answer generation request.  A failed
        check never prevents Hybrid RRF or local safety rules from serving a safe
        fallback response.
        """
        light_status = getattr(self.light, "status", "ready")
        gemini_status = self._gemini_client_status
        if self.gemini is not None:
            try:
                self.gemini.models.get(model=FINAL_GEMINI_MODEL)
                gemini_status = "ready"
            except Exception as exc:
                gemini_status = _safe_provider_status(exc)
                LOGGER.warning("GEMINI startup connectivity: %s", gemini_status)
        return {"gemini": gemini_status, "light_llm": light_status}

    def _load_metrics(self) -> dict[str, Any]:
        try:
            persisted = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            persisted = {}
        persisted.update(
            {
                "selected_retriever": "hybrid_rrf",
                "top_k": TOP_K,
                "retrieval_pipeline": {
                    "default": "hybrid_rrf",
                    "experimental_only": "query_rewrite_hybrid_rrf_cross_encoder",
                    "lightweight_model": self.light.model,
                    "final_answer_model": FINAL_GEMINI_MODEL,
                },
            }
        )
        return persisted

    def _load_or_create_embeddings(self) -> np.ndarray:
        corpus_hash = stable_id("".join(chunk["metadata"]["content_hash"] for chunk in self.chunks))
        if VECTORS_PATH.exists():
            loaded = np.load(VECTORS_PATH, allow_pickle=False)
            if str(loaded["corpus_hash"].item()) == corpus_hash and str(loaded["model"].item()) == EMBED_MODEL:
                return loaded["embeddings"]
        vectors = self.embedder.encode(
            [chunk["normalized_text"] for chunk in self.chunks],
            normalize_embeddings=True,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=True,
        ).astype("float32")
        np.savez_compressed(VECTORS_PATH, corpus_hash=np.array(corpus_hash), model=np.array(EMBED_MODEL), embeddings=vectors)
        return vectors

    def _load_or_create_bm25(self) -> BM25Okapi:
        corpus_hash = stable_id("".join(chunk["metadata"]["content_hash"] for chunk in self.chunks))
        if BM25_PATH.exists():
            try:
                cached = pickle.loads(BM25_PATH.read_bytes())
                if cached.get("corpus_hash") == corpus_hash and isinstance(cached.get("bm25"), BM25Okapi):
                    return cached["bm25"]
            except (OSError, EOFError, pickle.UnpicklingError, AttributeError):
                pass
        bm25 = BM25Okapi([chunk["retrieval_tokens"] for chunk in self.chunks])
        BM25_PATH.write_bytes(pickle.dumps({"corpus_hash": corpus_hash, "bm25": bm25}, protocol=pickle.HIGHEST_PROTOCOL))
        return bm25

    def _ensure_collection(self) -> None:
        if self.qdrant.collection_exists(self.collection):
            return
        self.qdrant.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=int(self.embeddings.shape[1]), distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=index, vector=vector.tolist(), payload=chunk)
            for index, (chunk, vector) in enumerate(zip(self.chunks, self.embeddings))
        ]
        self.qdrant.upsert(collection_name=self.collection, points=points)

    def dense_search(self, question: str, k: int) -> list[tuple[dict[str, Any], float]]:
        vector = self.embedder.encode([normalize_for_retrieval(question)], normalize_embeddings=True)[0].tolist()
        result = self.qdrant.query_points(collection_name=self.collection, query=vector, limit=k, with_payload=True)
        return [(point.payload, float(point.score)) for point in result.points]

    def bm25_search(self, question: str, k: int) -> list[tuple[dict[str, Any], float]]:
        scores = self.bm25.get_scores(re.findall(r"[\w-]+", normalize_for_retrieval(question)))
        indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:k]
        return [(self.chunks[index], float(scores[index])) for index in indices]

    def hybrid_search(self, question: str, k: int = WIDE_TOP_K) -> list[tuple[dict[str, Any], float]]:
        fused: dict[str, float] = {}
        documents: dict[str, dict[str, Any]] = {}
        for candidates in (self.dense_search(question, k), self.bm25_search(question, k)):
            for rank, (document, _score) in enumerate(candidates, start=1):
                content_hash = document["metadata"]["content_hash"]
                fused[content_hash] = fused.get(content_hash, 0.0) + 1 / (RRF_K + rank)
                documents[content_hash] = document
        ordered = sorted(fused, key=fused.get, reverse=True)[:k]
        return [(documents[content_hash], fused[content_hash]) for content_hash in ordered]

    @staticmethod
    def _clinical_retrieval_query(question: str) -> str:
        """Add general guideline wording while retaining the user's clinical terms.

        This is a deterministic retrieval anchor, not an LLM rewrite. The
        specific topic is derived from the question and matched against a
        recommendation index built from the active PFP document.
        """
        return f"{question} clinical guideline recommendation clinicians"

    def clinical_retrieval(self, question: str, k: int = TOP_K) -> tuple[list[tuple[dict[str, Any], float]], str]:
        """The notebook-approved default: deterministic anchor -> Hybrid RRF -> top three.

        Hybrid RRF always runs. A document-derived recommendation index then
        fills any missing directly matching recommendation, protecting every
        section from PDF headers without replacing the Hybrid RRF route.
        """
        retrieval_query = self._clinical_retrieval_query(question)
        hybrid_results = self.hybrid_search(retrieval_query, k=k)
        direct_support = self._corpus_direct_guideline_support(question)
        if not direct_support:
            return hybrid_results, retrieval_query

        seen_hashes: set[str] = set()
        covered_results: list[tuple[dict[str, Any], float]] = []
        for document, score in [*direct_support, *hybrid_results]:
            content_hash = document["metadata"]["content_hash"]
            if content_hash not in seen_hashes:
                covered_results.append((document, score))
                seen_hashes.add(content_hash)
        return covered_results[:k], retrieval_query

    def _corpus_direct_guideline_support(self, question: str) -> list[tuple[dict[str, Any], float]]:
        """Return only document-native recommendations matching the question topic.

        The index is built once from the current PDF's own normative statements.
        A candidate must match two meaningful question terms when available, so
        generic PFP wording and journal headers cannot be promoted.
        """
        question_terms = self._guideline_question_terms(question)
        if not question_terms:
            return []
        index = getattr(self, "recommendation_index", self._build_recommendation_index())
        scored: list[tuple[dict[str, Any], float]] = []
        minimum_overlap = 1 if len(question_terms) == 1 else 2
        for chunk in index:
            source = normalize_for_retrieval(chunk["original_text"])
            overlap = sum(self._guideline_term_matches_source(term, source) for term in question_terms)
            if overlap < minimum_overlap:
                continue
            score = 1.0 + (overlap / len(question_terms))
            scored.append((chunk, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:TOP_K]

    def _build_recommendation_index(self) -> list[dict[str, Any]]:
        """Index recommendation passages from the active PFP document only."""
        return [
            chunk
            for chunk in self.chunks
            if self._is_guideline_recommendation(chunk["original_text"])
        ]

    @staticmethod
    def _guideline_question_terms(question: str) -> tuple[str, ...]:
        """Keep only terms that can distinguish one guideline topic from another."""
        terms = {
            token
            for token in re.findall(r"[A-Za-z0-9\u0600-\u06FF-]+", normalize_for_retrieval(question))
            if len(token) >= 4 and token not in RETRIEVAL_STOPWORDS
        }
        return tuple(sorted(terms))

    @staticmethod
    def _guideline_term_matches_source(term: str, source: str) -> bool:
        """Match exact terms first, then only a conservative shared word stem.

        This keeps ``diagnostic`` aligned with ``diagnosis`` and similar wording
        variants without broad semantic guessing or an external vocabulary.
        """
        if term in source:
            return True
        if len(term) < 6:
            return False
        stem = term[:6]
        return any(token.startswith(stem) or term.startswith(token[:6]) for token in re.findall(r"[a-z0-9-]+", source))

    @staticmethod
    def _is_generic_document_sentence(sentence: str) -> bool:
        return bool(DOCUMENT_HEADER_PATTERN.search(sentence)) and not bool(RECOMMENDATION_PATTERN.search(sentence))

    @classmethod
    def _is_guideline_recommendation(cls, text: str) -> bool:
        return bool(RECOMMENDATION_PATTERN.search(text)) and not cls._is_generic_document_sentence(text)

    def experimental_rewrite_rerank(self, question: str, k: int = TOP_K) -> tuple[list[tuple[dict[str, Any], float]], str]:
        """Optional comparison path only; never used by the public clinical endpoint."""
        rewrite_prompt = f"""Return JSON only: {{\"query\": \"...\"}}.
Rewrite this PFP guideline question as a short English retrieval query. Do not answer it.
QUESTION: {question}"""
        rewritten = clean_rewritten_query(question, str(self.light.json(rewrite_prompt).get("query", question)))
        candidates = self.hybrid_search(rewritten, k=RERANK_CANDIDATE_K)
        try:
            if self.reranker is None:
                self.reranker = CrossEncoder(RERANK_MODEL, device="cpu")
            return rerank_documents(f"{question}\n{rewritten}", candidates, self.reranker, k), rewritten
        except Exception:
            return candidates[:k], rewritten

    @staticmethod
    def _context(results: list[tuple[dict[str, Any], float]]) -> str:
        return "\n\n".join(
            f"[{document['metadata']['content_hash']}] document={document['metadata']['document']} | "
            f"section={document['metadata']['section']} | page={document['metadata']['page']}\n{document['original_text']}"
            for document, _ in results
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        return LightProvider._extract_json(text)

    def _final_gemini_json(self, prompt: str, max_retries: int = 4) -> dict[str, Any]:
        """Generate the final answer with bounded retries for transient provider failures.

        The retry only improves transport and malformed-output resilience.  It
        does not change the model, retrieved context, retrieval route, safety
        policy, or citation verification requirements.
        """
        if self.gemini is None:
            raise LayerFailure("generation", "final_generation_unavailable", "generation", True)
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self.gemini.models.generate_content(
                    model=FINAL_GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=2048,
                        response_mime_type="application/json",
                    ),
                )
                response_text = str(getattr(response, "text", "") or "").strip()
                if not response_text:
                    raise ValueError("Gemini returned an empty final-answer response.")
                parsed = self._extract_json(response_text)
                if not isinstance(parsed, dict):
                    raise ValueError("Gemini final-answer output was not a JSON object.")
                return parsed
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "Final Gemini generation attempt %d/%d failed: %s",
                    attempt + 1,
                    max_retries,
                    _safe_provider_status(exc),
                )
                if hasattr(self, "provider_connectivity"):
                    self.provider_connectivity["gemini"] = _safe_provider_status(exc)
                if attempt < max_retries - 1:
                    time.sleep(min(4, 1 + attempt))
        raise LayerFailure("generation", "final_generation_unavailable", "generation", True) from last_error

    @staticmethod
    def _rule_risk(question: str) -> dict[str, Any] | None:
        if MEDICATION_PATTERN.search(question):
            return {"category": "REFUSE", "confidence": 1.0, "reason": "medication_request"}
        if EMERGENCY_PATTERN.search(question):
            return {"category": "REFUSE", "confidence": 1.0, "reason": "urgent_symptoms"}
        if OUT_OF_SCOPE_PATTERN.search(question):
            return {"category": "REFUSE", "confidence": 0.98, "reason": "out_of_scope"}
        if PERSONAL_PLAN_PATTERN.search(question):
            return {"category": "REFUSE", "confidence": 0.98, "reason": "personal_request"}
        if re.search(r"\bknee pain\b", question, re.I) and not PFP_PATTERN.search(question):
            return {"category": "EVIDENCE_ONLY", "confidence": 0.95, "reason": "Broad knee-pain question"}
        return None

    @staticmethod
    def _is_clearly_in_scope_question(question: str) -> bool:
        """Allow a constrained local fallback only for explicit general PFP questions.

        This does not bypass the later answerability, Gemini, or citation checks. It
        only prevents an unavailable lightweight risk-classification request from
        blocking a plainly in-scope evidence query before retrieval starts.
        """
        return bool(PFP_PATTERN.search(question)) and not any(
            pattern.search(question)
            for pattern in (MEDICATION_PATTERN, EMERGENCY_PATTERN, OUT_OF_SCOPE_PATTERN, PERSONAL_PLAN_PATTERN)
        )

    def classify_query_risk(self, question: str) -> dict[str, Any]:
        rule_result = self._rule_risk(question)
        if rule_result:
            return rule_result
        prompt = f"""Return JSON only with category, confidence, and reason.
You are a safety classifier for a single-source Patellofemoral Pain guideline.
Categories: ALLOW for a general PFP evidence question; EVIDENCE_ONLY for broad or ambiguous scope;
REFUSE for medication, diagnosis, emergency, patient-specific management, or another condition.
QUESTION: {question}"""
        try:
            result = self.light.json(prompt)
            category = str(result.get("category", "EVIDENCE_ONLY")).upper()
            if category not in {"ALLOW", "EVIDENCE_ONLY", "REFUSE"}:
                category = "EVIDENCE_ONLY"
            return {
                "category": category,
                "confidence": max(0.0, min(float(result.get("confidence", 0.75)), 1.0)),
                "reason": str(result.get("reason", "Lightweight safety classification"))[:240],
            }
        except ValueError:
            # A malformed JSON result must not prevent deterministic local safety
            # triage for an otherwise clearly in-scope PFP guideline question.
            LOGGER.warning("Risk classification failed: invalid_provider_response")
            if self._is_clearly_in_scope_question(question):
                return {
                    "category": "ALLOW",
                    "confidence": 0.72,
                    "reason": "local_in_scope_fallback_after_classifier_unavailable",
                }
            return {
                "category": "EVIDENCE_ONLY",
                "confidence": 0.0,
                "reason": "local_cautious_fallback_after_classifier_unavailable",
            }
        except Exception as exc:
            LOGGER.warning("Risk classification failed: %s", type(exc).__name__)
            if self._is_clearly_in_scope_question(question):
                return {
                    "category": "ALLOW",
                    "confidence": 0.72,
                    "reason": "local_in_scope_fallback_after_classifier_unavailable",
                }
            return {
                "category": "EVIDENCE_ONLY",
                "confidence": 0.0,
                "reason": "local_cautious_fallback_after_classifier_unavailable",
            }

    @staticmethod
    def response_policy(risk: dict[str, Any]) -> str:
        return "full" if risk["category"] == "ALLOW" else "evidence_only" if risk["category"] == "EVIDENCE_ONLY" else "refuse"

    def answerability_gate(self, question: str, results: list[tuple[dict[str, Any], float]]) -> dict[str, Any]:
        if not results:
            return {"answerable": False, "confidence": 1.0, "reason": "No retrieved PFP evidence.", "supporting_hashes": []}
        allowed_hashes = [document["metadata"]["content_hash"] for document, _ in results]
        prompt = f"""Return JSON only with answerable (boolean), confidence (0 to 1), reason, and supporting_hashes.
Judge only whether the retrieved PFP guideline passages directly support a specific answer. Do not use outside knowledge.
Only list hashes from: {allowed_hashes}
QUESTION: {question}
EVIDENCE:\n{self._context(results)}"""
        try:
            result = self.light.json(prompt)
            hashes = [value for value in result.get("supporting_hashes", []) if value in allowed_hashes]
            assessment = {
                "answerable": bool(result.get("answerable", False)),
                "confidence": max(0.0, min(float(result.get("confidence", 0.0)), 1.0)),
                "reason": str(result.get("reason", "Lightweight evidence assessment"))[:300],
                "supporting_hashes": hashes,
            }
            # A lightweight model can occasionally reject a broad recommendation
            # question even when the retrieved text itself contains a direct
            # guideline recommendation.  This narrow check is intentionally
            # extractive: it only corrects false refusals for a recognised topic
            # when both a topic phrase and recommendation wording occur in the
            # current Top-3 passages.  It neither answers the question nor
            # bypasses Gemini/citation verification.
            direct_support = self._direct_guideline_support(question, results)
            if direct_support and (not assessment["answerable"] or assessment["confidence"] < 0.70):
                assessment.update(
                    {
                        "answerable": True,
                        "confidence": max(assessment["confidence"], 0.78),
                        "reason": "local_direct_guideline_support_after_conservative_answerability_assessment",
                        "supporting_hashes": [document["metadata"]["content_hash"] for document, _ in direct_support],
                    }
                )
            return assessment
        except Exception as exc:
            LOGGER.warning("Answerability check failed: %s", type(exc).__name__)
            if results and self._is_clearly_in_scope_question(question):
                return {
                    "answerable": True,
                    "confidence": 0.70,
                    "reason": "local_in_scope_fallback_after_answerability_unavailable",
                    "supporting_hashes": [document["metadata"]["content_hash"] for document, _ in results],
                }
            raise LayerFailure("answerability", "answerability_unavailable", "insufficient", True) from exc

    def generate_grounded_answer(
        self,
        question: str,
        results: list[tuple[dict[str, Any], float]],
        response_language: str,
    ) -> dict[str, Any]:
        schema = {
            "recommendation": "string",
            "evidence_excerpts": [{"quote": "verbatim string", "content_hash": "string", "document": "string", "section": "string", "page": "integer"}],
            "confidence": "high | medium | low | insufficient_evidence",
            "refusal": "boolean",
        }
        prompt = f"""ROLE:
You are a clinical evidence assistant for the Patellofemoral Pain Clinical Practice Guideline.

CONTEXT:
The retrieved PFP passages below are the only source of truth.

TASK:
Answer the user question only when directly supported. Write the recommendation and all non-verbatim explanatory text in {response_language}.

CONSTRAINTS:
- Do not use outside medical knowledge or infer missing facts.
- Do not diagnose, prescribe, provide medication dosage, or create a patient-specific plan.
- Every evidence quote must be copied verbatim and keep the exact supplied hash, document, section, and page.
- Return exactly 1 or 2 evidence excerpts. Each quote must be one complete source sentence no longer than 400 characters.
- If direct support is missing, set refusal=true.
- Return JSON only and no markdown.

OUTPUT SHAPE:
{json.dumps(schema, ensure_ascii=False)}

RETRIEVED EXCERPTS:
{self._context(results)}

USER QUESTION:
{question}"""
        answer = self._final_gemini_json(prompt)
        answer["evidence_excerpts"] = self._deduplicate_excerpts(answer.get("evidence_excerpts", []))
        return answer

    @staticmethod
    def _deduplicate_excerpts(excerpts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str]] = set()
        unique: list[dict[str, Any]] = []
        for excerpt in excerpts:
            key = (str(excerpt.get("content_hash", "")), str(excerpt.get("quote", "")).strip())
            if key not in seen:
                seen.add(key)
                unique.append(excerpt)
        return unique

    @staticmethod
    def _best_verbatim_sentence(source_text: str, question: str) -> str:
        """Return one exact source sentence that is most relevant to the question.

        Gemini may preserve the meaning of a source quote but change whitespace,
        punctuation, or surrounding text.  This method deliberately returns a
        substring of the original retrieved passage, never a regenerated quote.
        """
        question_terms = {
            token
            for token in re.findall(r"[A-Za-z0-9\u0600-\u06FF]+", question.lower())
            if len(token) >= 3
        }
        candidates = [match.group(0).strip() for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", source_text, flags=re.S)]
        candidates = [candidate for candidate in candidates if candidate]
        if not candidates:
            return source_text.strip()
        non_header_candidates = [candidate for candidate in candidates if not RagEngine._is_generic_document_sentence(candidate)]
        if non_header_candidates:
            candidates = non_header_candidates
        return max(
            candidates,
            key=lambda candidate: (
                sum(token in candidate.lower() for token in question_terms),
                -len(candidate),
            ),
        )

    @staticmethod
    def _best_recommendation_sentence(source_text: str, question: str) -> str:
        """Prefer the exact recommendation sentence over PDF headers or titles.

        PDF extraction frequently places a journal header before the actual
        paragraph.  A general similarity-only sentence selector can therefore
        return that header for a guideline recommendation question.  This
        selector first seeks a sentence that contains both the queried
        intervention and explicit recommendation wording, and only then falls
        back to the general verbatim selector.
        """
        question_terms = RagEngine._guideline_question_terms(question)
        candidates = [match.group(0).strip() for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", source_text, flags=re.S)]
        direct_candidates = [candidate for candidate in candidates if RagEngine._is_guideline_recommendation(candidate)]
        if direct_candidates:
            return max(
                direct_candidates,
                key=lambda candidate: (
                    sum(token in normalize_for_retrieval(candidate) for token in question_terms),
                    -len(candidate),
                ),
            )
        return RagEngine._best_verbatim_sentence(source_text, question)

    @classmethod
    def _repair_generated_evidence_excerpts(
        cls,
        answer: dict[str, Any],
        results: list[tuple[dict[str, Any], float]],
        question: str,
    ) -> list[dict[str, Any]]:
        """Repair formatting drift without allowing ungrounded citations through.

        A repair is permitted only when Gemini selected a `content_hash` that is
        present in the current retrieval result.  The quote and its metadata are
        then rebuilt directly from that retrieved source.  A missing or invented
        hash remains unverified and therefore still triggers the citation guard.
        """
        lookup = {document["metadata"]["content_hash"]: document for document, _ in results}
        repaired: list[dict[str, Any]] = []
        seen_hashes: set[str] = set()

        for excerpt in answer.get("evidence_excerpts", []):
            if not isinstance(excerpt, dict):
                continue
            content_hash = str(excerpt.get("content_hash", ""))
            source = lookup.get(content_hash)
            if source is None or content_hash in seen_hashes:
                continue

            original_text = source["original_text"]
            proposed_quote = str(excerpt.get("quote", "")).strip()
            quote = (
                proposed_quote
                if proposed_quote and proposed_quote in original_text
                else cls._best_recommendation_sentence(original_text, question)
            )
            if not quote or quote not in original_text:
                continue

            repaired.append({"quote": quote, **source["metadata"]})
            seen_hashes.add(content_hash)

        return repaired

    @staticmethod
    def _presentation_from_answer(answer: dict[str, Any]) -> dict[str, Any]:
        """Create display-ready fields without changing clinical claims or quotes."""
        raw_text = str(answer.get("recommendation", "")).replace("\r", "\n")
        raw_text = re.sub(r"[`*_#]+", "", raw_text)
        parsed_lines = [
            (bool(re.match(r"^\s*(?:[-•]+|\d+[.)])\s*", line)), re.sub(r"^\s*(?:[-•]+|\d+[.)])\s*", "", line).strip())
            for line in raw_text.splitlines()
        ]
        paragraphs: list[str] = []
        key_points: list[str] = []
        for is_bullet, line in parsed_lines:
            if not line:
                continue
            if is_bullet:
                key_points.append(line)
            elif len(paragraphs) == 0:
                paragraphs.append(line)
            elif len(line) <= 260:
                key_points.append(line)
            else:
                paragraphs.append(line)
        summary = " ".join(paragraphs).strip() or " ".join(key_points[:1]).strip()
        if not summary:
            summary = "No evidence summary is available."
        if not key_points and len(paragraphs) > 1:
            key_points = paragraphs[1:]
        citations = [
            {
                "document": excerpt.get("document", DOC_NAME),
                "section": excerpt.get("section", ""),
                "page": excerpt.get("page"),
                "quote": excerpt.get("quote", ""),
            }
            for excerpt in answer.get("evidence_excerpts", [])
        ]
        display_text = summary
        if key_points:
            display_text += "\n\n" + "\n".join(f"• {point}" for point in key_points[:5])
        return {
            "summary": summary,
            "key_points": key_points[:5],
            "citations": citations,
            "display_text": display_text,
        }

    @classmethod
    def _with_presentation(cls, answer: dict[str, Any]) -> dict[str, Any]:
        answer["presentation"] = cls._presentation_from_answer(answer)
        answer["recommendation"] = answer["presentation"]["display_text"]
        return answer

    @staticmethod
    def _retrieved_payload(results: list[tuple[dict[str, Any], float]]) -> list[dict[str, Any]]:
        return [
            {
                "content_hash": document["metadata"]["content_hash"],
                "document": document["metadata"]["document"],
                "section": document["metadata"]["section"],
                "page": document["metadata"]["page"],
                "original_text": document["original_text"],
            }
            for document, _ in results
        ]

    @staticmethod
    def _refusal(
        message: str,
        layer: Layer,
        code: str,
        response_language: str,
        risk: dict[str, Any] | None = None,
        results: list[tuple[dict[str, Any], float]] | None = None,
        recoverable: bool = False,
    ) -> dict[str, Any]:
        answer = {
            "recommendation": localize_response_message(message, response_language),
            "evidence_excerpts": [],
            "confidence": "insufficient_evidence",
            "refusal": True,
            "response_mode": "refuse",
            "risk": risk or {"category": "UNKNOWN", "confidence": 0.0, "reason": "Not evaluated"},
            "safety_check": {"checks": [], "passed": True, "claim_free_scope_message": True},
            "retriever": "hybrid_rrf" if results else "scope_guard",
            "diagnostic": diagnostic(layer, code, recoverable),
        }
        if results:
            answer["_retrieved"] = RagEngine._retrieved_payload(results)
        return RagEngine._with_presentation(answer)

    def _direct_taping_support(self, question: str) -> list[tuple[dict[str, Any], float]]:
        if "patellar taping" not in normalize_for_retrieval(question):
            return []
        direct = [
            chunk
            for chunk in self.chunks
            if "clinicians may use tailored patellar taping" in chunk["original_text"].lower()
            or "tailored patellar taping in combination with exercise therapy" in chunk["original_text"].lower()
        ]
        return [(chunk, 1.0) for chunk in direct[:TOP_K]]

    @staticmethod
    def _direct_guideline_support(
        question: str,
        results: list[tuple[dict[str, Any], float]],
    ) -> list[tuple[dict[str, Any], float]]:
        """Identify explicit recommendation passages without inventing a claim.

        This is deliberately narrower than answerability.  It is used only to
        recover from an overly conservative lightweight decision or a final
        provider-format failure.  A passage must name the queried intervention
        and include normative guideline wording, so an unrelated discussion of
        PFP cannot be promoted to a clinical answer.
        """
        question_terms = RagEngine._guideline_question_terms(question)
        if not question_terms:
            return []
        minimum_overlap = 1 if len(question_terms) == 1 else 2
        return [
            (document, score)
            for document, score in results
            if RagEngine._is_guideline_recommendation(document["original_text"])
            and sum(
                RagEngine._guideline_term_matches_source(term, normalize_for_retrieval(document["original_text"]))
                for term in question_terms
            ) >= minimum_overlap
        ]

    def _verified_extractive_fallback(
        self,
        question: str,
        results: list[tuple[dict[str, Any], float]],
        response_language: str,
        risk: dict[str, Any],
        answerability: dict[str, Any],
        diagnostic_code: str,
    ) -> dict[str, Any] | None:
        """Return source text verbatim if final structured generation is unavailable.

        The fallback is not a second medical generator.  It exposes only one or
        two exact sentences from directly supportive retrieved passages along
        with their source metadata.  The usual local citation verifier still
        decides whether the result can be returned.
        """
        supporting_results = self._direct_guideline_support(question, results)
        if not supporting_results:
            return None

        excerpts: list[dict[str, Any]] = []
        for document, _ in supporting_results[:2]:
            quote = self._best_recommendation_sentence(document["original_text"], question)
            if quote and quote in document["original_text"]:
                excerpts.append({"quote": quote, **document["metadata"]})
        excerpts = self._deduplicate_excerpts(excerpts)
        if not excerpts:
            return None

        prefix = "وفقًا للنص المسترجع من دليل ألم المفصل الرضفي الفخذي:" if response_language == "Arabic" else "According to the retrieved PFP guideline evidence:"
        answer = {
            "recommendation": f"{prefix}\n{excerpts[0]['quote']}",
            "evidence_excerpts": excerpts,
            "confidence": "medium",
            "refusal": False,
            "response_mode": "evidence_only",
            "risk": risk,
            "answerability": answerability,
            "retriever": "hybrid_rrf",
            "_retrieved": self._retrieved_payload(results),
            "diagnostic": diagnostic("complete", diagnostic_code, True),
        }
        answer["safety_check"] = verify_evidence(answer, results)
        return RagEngine._with_presentation(answer) if answer["safety_check"]["passed"] else None

    def _verified_taping_fallback(
        self,
        question: str,
        response_language: str,
        risk: dict[str, Any],
        answerability: dict[str, Any],
    ) -> dict[str, Any] | None:
        results = self._direct_taping_support(question)
        if not results:
            return None
        document, _ = results[0]
        quote = next(
            (sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", document["original_text"]) if "tailored patellar taping" in sentence.lower()),
            "",
        )
        if not quote:
            return None
        message = (
            "بالنسبة لألم المفصل الرضفي الفخذي، يدعم الدليل استخدام التثبيت المخصص للرضفة مع العلاج بالتمارين للمساعدة في خفض الألم الفوري وتحسين نتائج التمارين قصيرة المدى. هذه معلومة من الدليل وليست خطة علاج شخصية."
            if response_language == "Arabic"
            else "For patellofemoral pain, the guideline supports tailored patellar taping in combination with exercise therapy to assist immediate pain reduction and enhance short-term exercise outcomes. This is guideline evidence, not a personal treatment plan."
        )
        answer = {
            "recommendation": message,
            "evidence_excerpts": [{"quote": quote, **document["metadata"]}],
            "confidence": "medium",
            "refusal": False,
            "response_mode": "full",
            "risk": risk,
            "answerability": answerability,
            "retriever": "hybrid_rrf",
            "_retrieved": self._retrieved_payload(results),
            "diagnostic": diagnostic("complete", "verified_taping_fallback"),
        }
        answer["safety_check"] = verify_evidence(answer, results)
        return RagEngine._with_presentation(answer) if answer["safety_check"]["passed"] else None

    def ask(self, question: str) -> dict[str, Any]:
        response_language = detect_response_language(question)
        try:
            risk = self.classify_query_risk(question)
        except LayerFailure as failure:
            return self._refusal(failure.public_message, failure.layer, failure.code, response_language, recoverable=failure.recoverable)

        policy = self.response_policy(risk)
        if policy == "refuse":
            reason = str(risk.get("reason", ""))
            if reason == "urgent_symptoms":
                return self._refusal("urgent", "risk_policy", "urgent_triage", response_language, risk)
            if reason == "personal_request":
                return self._refusal("personal", "risk_policy", "personal_request", response_language, risk)
            return self._refusal("scope", "risk_policy", "policy_refusal", response_language, risk)

        try:
            results, retrieval_query = self.clinical_retrieval(question, k=TOP_K)
        except Exception as exc:
            LOGGER.exception("Hybrid retrieval failed: %s", type(exc).__name__)
            return self._refusal("insufficient", "retrieval", "hybrid_retrieval_failed", response_language, risk, recoverable=True)

        if policy == "evidence_only":
            return self._refusal("broad", "risk_policy", "evidence_only_scope", response_language, risk, results)

        try:
            answerability = self.answerability_gate(question, results)
        except LayerFailure as failure:
            return self._refusal(failure.public_message, failure.layer, failure.code, response_language, risk, results, failure.recoverable)

        if not answerability["answerable"] or answerability["confidence"] < 0.70:
            fallback = self._verified_taping_fallback(question, response_language, risk, answerability)
            if fallback:
                return fallback
            fallback = self._verified_extractive_fallback(
                question,
                results,
                response_language,
                risk,
                answerability,
                "answerability_recovered_by_verified_extractive_fallback",
            )
            if fallback:
                return fallback
            answer = self._refusal("insufficient", "answerability", "insufficient_guideline_support", response_language, risk, results)
            answer.update({"answerability": answerability, "retrieval_query": retrieval_query})
            return answer

        try:
            answer = self.generate_grounded_answer(question, results, response_language)
        except LayerFailure as failure:
            fallback = self._verified_taping_fallback(question, response_language, risk, answerability)
            if fallback:
                fallback["diagnostic"] = diagnostic("generation", "generation_recovered_by_taping_fallback", True)
                return fallback
            fallback = self._verified_extractive_fallback(
                question,
                results,
                response_language,
                risk,
                answerability,
                "generation_recovered_by_verified_extractive_fallback",
            )
            if fallback:
                return fallback
            answer = self._refusal(failure.public_message, failure.layer, failure.code, response_language, risk, results, failure.recoverable)
            answer.update({"answerability": answerability, "retrieval_query": retrieval_query})
            return answer

        answer.update(
            {
                "response_mode": "full",
                "risk": risk,
                "answerability": answerability,
                "retrieval_query": retrieval_query,
                "retriever": "hybrid_rrf",
                "_retrieved": self._retrieved_payload(results),
                "diagnostic": diagnostic("complete", "answer_ready"),
            }
        )
        fallback_used = any(
            "fallback_after_" in str(component.get("reason", ""))
            for component in (risk, answerability)
        )
        if fallback_used:
            answer["diagnostic"] = diagnostic("complete", "answer_ready_with_local_safety_fallback", True)
        answer["evidence_excerpts"] = self._repair_generated_evidence_excerpts(answer, results, question)
        answer["safety_check"] = verify_evidence(answer, results)
        if answer.get("refusal") or not answer["safety_check"]["passed"]:
            fallback = self._verified_taping_fallback(question, response_language, risk, answerability)
            if fallback:
                return fallback
            fallback = self._verified_extractive_fallback(
                question,
                results,
                response_language,
                risk,
                answerability,
                "citation_recovered_by_verified_extractive_fallback",
            )
            if fallback:
                return fallback
            return self._refusal("citation", "citation", "citation_verification_failed", response_language, risk, results)
        return self._with_presentation(answer)

    def validate_document_topic(self, preview: str) -> bool:
        """Use the lightweight boundary only when a live service engine is available."""
        prompt = f"""Return JSON only: {{\"is_pfp_topic\": true or false}}.
Does this short PDF preview describe Patellofemoral Pain or its clinical practice guideline?
PREVIEW: {preview[:5000]}"""
        result = self.light.json(prompt)
        return bool(result.get("is_pfp_topic", False))


engine: RagEngine | None = None
engine_lock = threading.RLock()


def ensure_engine() -> RagEngine:
    global engine
    with engine_lock:
        if engine is None:
            engine = RagEngine()
    return engine


def reset_source_index() -> None:
    for artifact in (CHUNKS_PATH, VECTORS_PATH, BM25_PATH):
        artifact.unlink(missing_ok=True)
    shutil.rmtree(DATA_DIR / "qdrant", ignore_errors=True)


def validate_source_pdf(filename: str, content_base64: str) -> tuple[bytes | None, dict[str, Any] | None]:
    if not filename.lower().endswith(".pdf"):
        return None, {"accepted": False, "code": "invalid_pdf", "message": "Please choose a PDF document.", "diagnostic": diagnostic("document", "file_extension_invalid")}
    try:
        raw_pdf = base64.b64decode(content_base64, validate=True)
    except (ValueError, base64.binascii.Error):
        return None, {"accepted": False, "code": "invalid_pdf", "message": "The selected file could not be read as a PDF.", "diagnostic": diagnostic("document", "base64_invalid")}
    if len(raw_pdf) > MAX_SOURCE_PDF_BYTES or not raw_pdf.startswith(b"%PDF"):
        return None, {"accepted": False, "code": "invalid_pdf", "message": "Please choose a valid PDF smaller than 15 MB.", "diagnostic": diagnostic("document", "pdf_invalid")}
    try:
        with pymupdf.open(stream=raw_pdf, filetype="pdf") as document:
            preview = clean_source_text(" ".join(page.get_text("text") for page in document[: min(6, len(document))]))
    except (RuntimeError, ValueError):
        return None, {"accepted": False, "code": "invalid_pdf", "message": "The selected file could not be opened as a PDF.", "diagnostic": diagnostic("document", "pdf_unreadable")}
    if not PFP_PATTERN.search(preview):
        return None, {"accepted": False, "code": "not_pfp_topic", "message": "This document does not appear to be about patellofemoral pain.", "diagnostic": diagnostic("document", "topic_rejected")}
    active_engine = engine
    if active_engine is not None and hasattr(active_engine, "validate_document_topic"):
        try:
            if not active_engine.validate_document_topic(preview):
                return None, {"accepted": False, "code": "not_pfp_topic", "message": "This document does not appear to be about patellofemoral pain.", "diagnostic": diagnostic("document", "topic_rejected")}
        except Exception as exc:
            LOGGER.warning("PDF topic validation failed: %s", type(exc).__name__)
            return None, {"accepted": False, "code": "processing_failed", "message": "The document could not be validated safely right now.", "diagnostic": diagnostic("document", "topic_validation_unavailable", True)}
    return raw_pdf, None


def health_status() -> dict[str, Any]:
    service = ensure_engine()
    return {
        "status": "ready",
        "chunks": len(service.chunks),
        "embedding_model": EMBED_MODEL,
        "final_answer_model": FINAL_GEMINI_MODEL,
        "lightweight_model": service.light.model,
        "provider_connectivity": service.provider_connectivity,
        "default_retriever": "hybrid_rrf",
        "top_k": TOP_K,
    }


def evaluation_metrics() -> dict[str, Any]:
    return ensure_engine().metrics


def ask_safely(question: str) -> dict[str, Any]:
    try:
        return ensure_engine().ask(question)
    except Exception as exc:
        LOGGER.exception("Unexpected answer pipeline failure: %s", type(exc).__name__)
        language = detect_response_language(question)
        return RagEngine._refusal("generation", "generation", "unexpected_service_failure", language, recoverable=True)


def replace_source_pdf(filename: str, content_base64: str) -> dict[str, Any]:
    raw_pdf, rejection = validate_source_pdf(filename, content_base64)
    if rejection:
        return rejection
    global engine
    with engine_lock:
        previous_engine = engine
        if previous_engine is not None:
            close = getattr(previous_engine.qdrant, "close", None)
            if callable(close):
                close()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        PDF_PATH.write_bytes(raw_pdf)
        reset_source_index()
        engine = None
        try:
            active_engine = ensure_engine()
        except Exception as exc:
            LOGGER.exception("Uploaded source processing failed: %s", type(exc).__name__)
            engine = previous_engine
            return {
                "accepted": False,
                "code": "processing_failed",
                "message": "The document could not be prepared for questions.",
                "diagnostic": diagnostic("document", "indexing_failed", True),
            }
    return {
        "accepted": True,
        "code": "source_ready",
        "message": "The document has been prepared and is ready for questions.",
        "document": filename,
        "chunks": len(active_engine.chunks),
        "diagnostic": diagnostic("document", "source_ready"),
    }
