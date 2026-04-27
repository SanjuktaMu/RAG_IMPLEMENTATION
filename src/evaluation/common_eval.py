from __future__ import annotations

from typing import Any, Iterable

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.core.config import MODEL_NAME, OLLAMA_BASE_URL
from src.evaluation.llm_judge_eval import coerce_result, parse_judge_response

JUDGE_PROMPT = """
You are an impartial evaluator for a RAG system.

Evaluate the model answer using ONLY these fields:
- Question
- Ground Truth
- Retrieved Contexts
- Model Answer

Scoring rubric:
- correct: Answer matches the ground truth factually. Small wording differences are acceptable.
- partially_correct: Answer contains part of the ground truth, but misses important details or includes uncertain/conflicting content.
- incorrect: Answer is wrong, contradictory, unsupported by context, or says not found when the ground truth expects a concrete answer.

Special rule for numeric questions:
- Treat comma/space/currency symbol formatting as equivalent.
- Core numeric value must match.

Return STRICT JSON only with this schema:
{{
    "verdict": "correct|partially_correct|incorrect",
    "score": 0.0,
    "reason": "short explanation"
}}

Question: {question}
Ground Truth: {ground_truth}
Retrieved Contexts: {contexts}
Model Answer: {answer}
""".strip()


def _make_llm() -> ChatOllama:
    return ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


def _stringify_contexts(contexts: Iterable[Any]) -> str:
    return "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in contexts)


def judge_rag_output(
    question: str,
    answer: str,
    contexts: Iterable[Any],
    ground_truth: str = "",
    llm: ChatOllama | None = None,
) -> dict[str, Any]:
    judge_llm = llm or _make_llm()
    prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)
    messages = prompt.format_messages(
        question=question,
        ground_truth=ground_truth,
        contexts=_stringify_contexts(contexts),
        answer=answer,
    )
    raw = judge_llm.invoke(messages).content
    parsed = parse_judge_response(raw)
    return coerce_result(parsed)


def standardize_rag_result(
    question: str,
    answer: str,
    contexts: Iterable[Any],
    latency: float,
    rag_type: str,
    ground_truth: str = "",
    llm: ChatOllama | None = None,
) -> dict[str, Any]:
    evaluation = judge_rag_output(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
        llm=llm,
    )

    return {
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer,
        "contexts": [getattr(doc, "page_content", str(doc)) for doc in contexts],
        "rag_type": rag_type,
        "latency": latency,
        "evaluation": evaluation,
    }


def run_with_evaluation(
    rag_function,
    query: str,
    ground_truth: str = "",
    rag_type: str = "unknown",
    llm: ChatOllama | None = None,
) -> dict[str, Any]:
    answer, docs, _context, latency = rag_function(query)
    return standardize_rag_result(
        question=query,
        answer=answer,
        contexts=docs,
        latency=latency,
        rag_type=rag_type,
        ground_truth=ground_truth,
        llm=llm,
    )
