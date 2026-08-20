import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Optional convenience only; production may provide environment variables directly.
    def load_dotenv() -> bool:
        return False
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load local secrets before importing the engine; no secrets are returned by endpoints.
load_dotenv()
from core_data.engine import ask_safely, evaluation_metrics, health_status, replace_source_pdf

BASE_DIR = Path(__file__).resolve().parent
allowed_origins = [origin.strip() for origin in os.getenv("CORS_ALLOW_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000").split(",") if origin.strip()]

app = FastAPI(
    title="Ortho RAG Service API",
    description="Backend API for Patellofemoral Pain CPG RAG System",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if (BASE_DIR / "static").is_dir():
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class QueryRequest(BaseModel):
    question: str = Field(min_length=4, max_length=1000)


class SourcePdfRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=180)
    contentBase64: str = Field(min_length=32, max_length=21_000_000)


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, __: Exception):
    """Never return stack traces, keys, or provider internals to the front end."""
    return JSONResponse(
        status_code=500,
        content={
            "recommendation": "The evidence service is temporarily unavailable. Please try again shortly.",
            "evidence_excerpts": [],
            "confidence": "insufficient_evidence",
            "refusal": True,
            "response_mode": "refuse",
            "diagnostic": {"layer": "service", "code": "unexpected_service_failure", "recoverable": True},
            "presentation": {"summary": "The evidence service is temporarily unavailable. Please try again shortly.", "key_points": [], "citations": [], "display_text": "The evidence service is temporarily unavailable. Please try again shortly."},
        },
    )


@app.get("/")
def read_root():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/guide")
def read_guide():
    return FileResponse(BASE_DIR / "templates" / "chat.html")


@app.get("/technical")
def read_technical():
    return FileResponse(BASE_DIR / "templates" / "technical.html")


@app.get("/api/health")
def read_health():
    return health_status()

@app.get("/api/metrics")
def read_metrics():
    return evaluation_metrics()


@app.post("/api/ask")
def ask_rag_endpoint(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return ask_safely(request.question.strip())


@app.post("/api/source-pdf")
def source_pdf_endpoint(request: SourcePdfRequest):
    return replace_source_pdf(request.filename, request.contentBase64)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
