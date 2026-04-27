from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from src.core import config as core_config
from src.core import document_loader
from src.core.vector_store import create_vector_store, load_vector_store
from src.rag_types.agentic_rag import get_agentic_rag_chain

_DB: Any | None = None
_RAG: Any | None = None
_ACTIVE_PDF: str | None = None
_ACTIVE_TOP_K: int | None = None


def _set_active_pdf_path(pdf_path: str) -> None:
    core_config.PDF_PATH = pdf_path
    document_loader.PDF_PATH = pdf_path


def _build_rag(top_k: int | None = None) -> Any:
    global _DB, _RAG, _ACTIVE_TOP_K
    if _DB is None:
        _DB = load_vector_store()
    if _RAG is None or _ACTIVE_TOP_K != top_k:
        _RAG = get_agentic_rag_chain(_DB, top_k=top_k)
        _ACTIVE_TOP_K = top_k
    return _RAG


def process_pdf(file_path: str) -> dict[str, Any]:
    global _DB, _RAG, _ACTIVE_PDF, _ACTIVE_TOP_K

    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    _set_active_pdf_path(str(pdf_path.resolve()))

    if Path(core_config.DB_DIR).exists():
        shutil.rmtree(core_config.DB_DIR, ignore_errors=True)

    _DB = create_vector_store()
    _RAG = None
    _ACTIVE_TOP_K = None
    _ACTIVE_PDF = str(pdf_path.resolve())

    return {
        "status": "success",
        "message": "Document processed successfully",
        "active_pdf": _ACTIVE_PDF,
    }


def query_rag(question: str, top_k: int | None = None) -> dict[str, Any]:
    clean_question = (question or "").strip()
    if not clean_question:
        raise ValueError("Question cannot be empty")

    rag = _build_rag(top_k=top_k)
    answer, docs, _context, latency = rag(clean_question)

    contexts = [getattr(doc, "page_content", "") for doc in docs]
    metadata = [getattr(doc, "metadata", {}) or {} for doc in docs]

    return {
        "answer": answer if isinstance(answer, str) else str(answer),
        "context": contexts,
        "metadata": metadata,
        "latency": latency,
        "active_pdf": _ACTIVE_PDF,
    }


def get_status() -> dict[str, Any]:
    return {
        "active_pdf": _ACTIVE_PDF,
        "db_loaded": _DB is not None,
        "rag_loaded": _RAG is not None,
        "top_k": _ACTIVE_TOP_K,
    }


class RagService:
    def ask(self, question: str, top_k: int | None = None) -> dict:
        return query_rag(question=question, top_k=top_k)
