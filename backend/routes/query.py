from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from backend.services.rag_service import RagService

router = APIRouter(prefix="/query", tags=["query"])
service = RagService()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1)


@router.post("")
def ask_question(payload: QueryRequest) -> dict:
    try:
        result = service.ask(question=payload.question, top_k=payload.top_k)
        return {
            "answer": result.get("answer", ""),
            "context": result.get("context", []),
            "metadata": result.get("metadata", []),
            "latency": result.get("latency"),
            "active_pdf": result.get("active_pdf"),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
