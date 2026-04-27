import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.core.config import MODEL_NAME, OLLAMA_BASE_URL, TOP_K
from src.core.retriever import get_retriever

STRICT_QA_PROMPT = """
You are an AI assistant for document question answering.

Rules:
- Answer ONLY using the provided context.
- Do NOT use external knowledge.
- If the answer is missing in the context, return exactly: Not found in document
- Keep the answer concise and grounded in facts from context.
- If the question involves comparison, analyze the values in the context and infer the correct answer.

Context:
{context}

Question:
{question}

Answer:
""".strip()


def _response_text(response: Any) -> str:
    content = response.content if hasattr(response, "content") else response
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def _print_retrieved_docs(docs) -> None:
    print("\n----- RETRIEVED DOCS (MULTIMODAL) -----")
    if not docs:
        print("No documents retrieved")
        return

    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        print(
            f"\nDoc {i} | type={meta.get('type', 'unknown')} | "
            f"page={meta.get('page', 'unknown')}:\n{doc.page_content[:350]}"
        )


def _used_table_data_heuristic(answer: str, docs) -> bool:
    if not answer.strip() or answer.strip().lower() == "not found in document":
        return False

    has_table_doc = any((d.metadata or {}).get("type") == "table" for d in docs)
    if not has_table_doc:
        return False

    has_numbers = any(ch.isdigit() for ch in answer)
    comparison_tokens = ("highest", "lowest", "more", "less", "increase", "decrease", "segment")
    mentions_comparison = any(token in answer.lower() for token in comparison_tokens)
    return has_numbers or mentions_comparison


def get_multimodal_rag_chain(db, top_k: int | None = None):
    k = top_k if top_k is not None else max(TOP_K, 8)
    retriever = get_retriever(db, top_k=k)

    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_template(STRICT_QA_PROMPT)

    rag_handle: Any = None

    def rag(question: str):
        start = time.perf_counter()

        docs = retriever.invoke(question)
        _print_retrieved_docs(docs)

        context = "\n\n".join(d.page_content for d in docs if (d.page_content or "").strip())
        if not context.strip():
            latency = time.perf_counter() - start
            rag_handle.last_debug = {
                "question": question,
                "retrieved_doc_count": 0,
                "final_context_length": 0,
            }
            return "Not found in document", [], "", latency

        messages = prompt.format_messages(context=context, question=question)
        response = llm.invoke(messages)
        answer = _response_text(response).strip() or "Not found in document"
        used_table_data = _used_table_data_heuristic(answer, docs)
        print(f"\nTable data used (heuristic): {used_table_data}")

        latency = time.perf_counter() - start
        rag_handle.last_debug = {
            "question": question,
            "retrieved_doc_count": len(docs),
            "final_context_length": len(context),
            "used_table_data": used_table_data,
        }
        return answer, docs, context, latency

    rag_handle = rag
    rag_handle.last_debug = {
        "retrieved_doc_count": 0,
        "final_context_length": 0,
        "used_table_data": False,
    }
    return rag
