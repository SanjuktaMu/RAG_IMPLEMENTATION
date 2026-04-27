from __future__ import annotations

import re
from typing import Any

import numpy as np
from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sklearn.metrics.pairwise import cosine_similarity

from src.core.config import TOP_K
from src.core.embeddings import LocalEmbedding

_EMBEDDING_MODEL: LocalEmbedding | None = None


NUMERIC_HINTS = {
    "how many",
    "total",
    "revenue",
    "income",
    "expenses",
    "profit",
    "amount",
    "value",
    "trend",
    "growth",
    "decrease",
    "increase",
    "graph",
    "chart",
}

TABLE_INTENT_HINTS = {
    "revenue",
    "total",
    "segment",
    "highest",
}

IMAGE_INTENT_HINTS = {
    "trend",
    "growth",
    "increase",
    "decline",
}


def _normalize_doc_metadata(doc: Document) -> Document:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    doc_type = str(metadata.get("type", "")).strip().lower()
    if doc_type not in {"text", "table", "image"}:
        doc_type = "text"

    page = metadata.get("page", metadata.get("page_number", -1))
    try:
        page = int(page)
    except Exception:  # noqa: BLE001
        page = -1

    metadata["type"] = doc_type
    metadata["page"] = page
    doc.metadata = metadata
    return doc


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "")).strip()


def _dedupe_docs(docs):
    unique_docs = []
    seen = set()
    for doc in docs:
        text = (getattr(doc, "page_content", "") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique_docs.append(_normalize_doc_metadata(doc))
    return unique_docs


def _is_numeric_or_chart_query(query: str) -> bool:
    lower = query.lower()
    return any(hint in lower for hint in NUMERIC_HINTS)


def _has_table_intent(query: str) -> bool:
    lower = query.lower()
    return any(hint in lower for hint in TABLE_INTENT_HINTS)


def _has_image_intent(query: str) -> bool:
    lower = query.lower()
    return any(hint in lower for hint in IMAGE_INTENT_HINTS)


def _rerank_documents(query: str, docs, limit: int):
    if not docs:
        return []

    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = LocalEmbedding()

    embedding = _EMBEDDING_MODEL
    doc_texts = [doc.page_content for doc in docs]

    query_vector = np.array([embedding.embed_query(query)])
    doc_vectors = np.array(embedding.embed_documents(doc_texts))
    similarities = cosine_similarity(query_vector, doc_vectors)[0]

    prioritized = []
    prefer_structured = _is_numeric_or_chart_query(query)
    prefer_table = _has_table_intent(query)
    prefer_image = _has_image_intent(query)
    for doc, score in zip(docs, similarities.tolist()):
        metadata = getattr(doc, "metadata", {}) or {}
        doc_type = metadata.get("type", "text")
        type_bonus = 0.0
        if prefer_structured and doc_type == "table":
            type_bonus += 0.08
        if prefer_structured and doc_type == "image":
            type_bonus += 0.04
        if prefer_table and doc_type == "table":
            type_bonus += 0.12
        if prefer_image and doc_type == "image":
            type_bonus += 0.12
        prioritized.append((doc, float(score + type_bonus)))

    prioritized.sort(key=lambda item: item[1], reverse=True)
    return [doc for doc, _ in prioritized[:limit]]


class EnhancedRetriever(BaseRetriever):
    db: Any
    base_retriever: Any
    top_k: int

    model_config = {"arbitrary_types_allowed": True}

    def _retrieve(self, query: str) -> list[Document]:
        resolved_query = _normalize_query(query)
        docs = self.base_retriever.invoke(resolved_query) or []
        docs = _dedupe_docs(docs)

        if not docs:
            docs = self.db.similarity_search(resolved_query, k=max(self.top_k * 3, 10))
            docs = _dedupe_docs(docs)

        reranked = _rerank_documents(resolved_query, docs, limit=max(self.top_k * 2, self.top_k))
        return reranked[: self.top_k]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        _ = run_manager
        return self._retrieve(query)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        _ = run_manager
        return self._retrieve(query)


def get_retriever(db, top_k=None, search_type="mmr", fetch_k=None):
    k = top_k if top_k is not None else TOP_K
    resolved_fetch_k = fetch_k if fetch_k is not None else max(20, k * 5)

    search_kwargs = {"k": k}
    if search_type == "mmr":
        search_kwargs["fetch_k"] = resolved_fetch_k

    base_retriever = db.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    return EnhancedRetriever(db=db, base_retriever=base_retriever, top_k=k)