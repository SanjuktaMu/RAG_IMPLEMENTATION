import time
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.core.config import MODEL_NAME, OLLAMA_BASE_URL, TOP_K
from src.core.retriever import get_retriever


FINAL_PROMPT = """
You are a document QA assistant.

Rules:
- Answer ONLY from the provided context
- If partial information is available, answer as much as possible
- If completely missing, return exactly: Not found in document
- Do NOT use external knowledge
- Prefer concise, fact-only answers

Context:
{context}

Question:
{question}

Answer:
"""


def _is_boilerplate_chunk(text: str) -> bool:
    lower = text.lower()
    signals = [
        "registered office",
        "scan code to visit",
        "cin:",
        "ph.no",
        "corporate & marketing office",
    ]
    hits = sum(1 for s in signals if s in lower)
    return hits >= 2


def _normalize_query(question: str) -> str:
    q = re.sub(r"\bbtl\s*epc\s*(limited|ltd\.?)\b", "", question, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip(" ?")
    return q if q else question


def get_simple_rag_chain(db, top_k: int | None = None):
    k = top_k if top_k is not None else TOP_K
    retriever = get_retriever(db, top_k=k)

    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(FINAL_PROMPT)

    def rag(question: str):
        start = time.perf_counter()

        query = _normalize_query(question)

        # Basic pipeline: Query -> Retrieve top-k chunks.
        docs = retriever.invoke(query)

        print("\n----- RETRIEVED DOCS -----")
        if not docs:
            print("No documents retrieved!")
        for i, d in enumerate(docs):
            meta = d.metadata or {}
            print(
                f"\nDoc {i+1} | type={meta.get('type', 'unknown')} | "
                f"page={meta.get('page', 'unknown')}:\n{d.page_content[:300]}"
            )

        # Filter repeated low-information footer/header chunks.
        filtered_docs = [d for d in docs if d.page_content and not _is_boilerplate_chunk(d.page_content)]
        if filtered_docs:
            docs = filtered_docs

        # Fallback: fetch a broader set and keep non-boilerplate chunks.
        if not docs:
            broad_docs = db.similarity_search(query, k=max(k * 3, 10))
            docs = [d for d in broad_docs if d.page_content and not _is_boilerplate_chunk(d.page_content)][:k]

        context = "\n\n".join(d.page_content for d in docs if d.page_content.strip())

        if not context.strip():
            return "Not found in document", docs, context, 0

        messages = prompt.format_messages(
            context=context,
            question=question,
        )
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        latency = time.perf_counter() - start
        return answer, docs, context, latency

    return rag