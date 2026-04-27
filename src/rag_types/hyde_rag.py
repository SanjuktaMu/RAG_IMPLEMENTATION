import time
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from src.core.retriever import get_retriever
from src.core.config import MODEL_NAME, OLLAMA_BASE_URL


# 🧠 PROMPT 1: GENERATE HYPOTHETICAL ANSWER
HYDE_PROMPT = """
You are generating a retrieval-oriented hypothesis for a document search.

Use only the entities, dates, numbers, and terminology that are already implied by the question.
Do not invent company names, people, locations, or financial figures.
Write a short, keyword-rich passage that could plausibly appear in the source document.
Keep it focused on the likely wording of the report, not on a polished explanation.

Question:
{question}

Hypothetical Answer:
"""


# 🧠 PROMPT 2: FINAL ANSWER (BALANCED)
FINAL_PROMPT = """
You are a document QA assistant.

Rules:
- Answer ONLY from the provided context
- If partial information is available, answer as much as possible
- If completely missing → say "Not found in document"
- Do NOT use external knowledge
- Prefer concise, fact-only answers
- For numeric/entity questions, return the exact value/name from context

Context:
{context}

Question:
{question}

Answer:
"""


# 🧠 PROMPT 3: STRICT EXTRACTION RETRY
STRICT_FINAL_PROMPT = """
You are a strict information extractor.

Rules:
- Use ONLY the context
- For numeric or named-entity questions, return exact values/names from context
- Keep answer very short (one line)
- If exact value/name is not in context, return exactly: Not found in document
- Do not explain your reasoning

Context:
{context}

Question:
{question}

Answer:
"""


NUMERIC_HINTS = {
    "how many", "total", "revenue", "income", "expenses", "profit", "difference",
    "value", "amount", "fy", "year", "age", "strength", "tax", "crore", "lacs"
}

ENTITY_HINTS = {
    "who", "which", "name", "client", "company", "director", "mission", "vision",
    "founded", "location", "sectors", "industries"
}

VAGUE_MARKERS = {
    "does not explicitly", "does not contain", "not explicitly", "cannot be determined",
    "provided context does not", "based on the context"
}


def _is_numeric_or_entity_question(question: str) -> bool:
    q = question.lower()
    return any(h in q for h in NUMERIC_HINTS) or any(h in q for h in ENTITY_HINTS)


def _looks_vague(answer: str) -> bool:
    lower = answer.lower().strip()
    if lower == "not found in document":
        return False
    if any(marker in lower for marker in VAGUE_MARKERS):
        return True
    return False


def _needs_retry(question: str, answer: str) -> bool:
    if not _is_numeric_or_entity_question(question):
        return _looks_vague(answer)

    lower = answer.lower().strip()
    if lower == "not found in document":
        return False

    if _looks_vague(answer):
        return True

    has_number = bool(re.search(r"\d", answer))
    numeric_expected = any(h in question.lower() for h in NUMERIC_HINTS)
    if numeric_expected and not has_number:
        return True

    # Entity answers should stay concise and direct.
    if len(answer.split()) > 50:
        return True

    return False


def get_hyde_rag_chain(db):
    retriever = get_retriever(db)

    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0
    )

    hyde_prompt = ChatPromptTemplate.from_template(HYDE_PROMPT)
    final_prompt = ChatPromptTemplate.from_template(FINAL_PROMPT)
    strict_final_prompt = ChatPromptTemplate.from_template(STRICT_FINAL_PROMPT)

    def rag(question: str):
        start = time.perf_counter()

        # 🔥 STEP 1: GENERATE HYPOTHETICAL ANSWER
        hyde_messages = hyde_prompt.format_messages(question=question)
        hypothetical_answer = llm.invoke(hyde_messages).content

        print("\n===== HYPOTHETICAL ANSWER (HyDE) =====")
        print(hypothetical_answer[:500])

        # 🔥 STEP 2: USE IT FOR RETRIEVAL
        retrieval_query = f"{question}\n\n{hypothetical_answer}"
        docs = retriever.invoke(retrieval_query)

        # 🔁 FALLBACK: if retrieval fails, use original question
        if not docs:
            print("\n⚠️ HyDE failed → fallback to normal query")
            docs = retriever.invoke(question)

        print("\n===== RETRIEVED CONTEXT =====")
        for i, doc in enumerate(docs):
            print(f"\n--- Chunk {i+1} ---")
            print(doc.page_content[:300])

        context = "\n\n".join([d.page_content for d in docs])

        # 🔥 STEP 3: FINAL ANSWER
        final_messages = final_prompt.format_messages(
            context=context,
            question=question
        )

        response = llm.invoke(final_messages)
        answer = response.content if isinstance(response.content, str) else str(response.content)

        if _needs_retry(question, answer):
            strict_messages = strict_final_prompt.format_messages(
                context=context,
                question=question,
            )
            strict_response = llm.invoke(strict_messages)
            answer = strict_response.content if isinstance(strict_response.content, str) else str(strict_response.content)

        latency = time.perf_counter() - start

        return answer, docs, context, latency

    return rag