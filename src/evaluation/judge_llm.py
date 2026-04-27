import json
import os
import importlib
from typing import Any

REQUIRED_SCORE_KEYS = (
    "context_relevance",
    "answer_relevance",
    "groundedness",
    "correctness",
)

JUDGE_PROMPT_TEMPLATE = """
You are a strict evaluator for a Retrieval-Augmented Generation (RAG) system.

Evaluate the candidate answer on these dimensions (1 to 5):
1) context_relevance: How relevant the retrieved context is to the user question.
2) answer_relevance: How well the answer addresses the user question.
3) groundedness: How well the answer is supported by the provided context.
4) correctness: How correct the answer is with respect to the ground truth (if provided).

Scoring scale:
1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Rules:
- Return ONLY valid JSON.
- Do not include markdown.
- Do not include explanations.
- Use exactly these keys: context_relevance, answer_relevance, groundedness, correctness.
- Values must be integers between 1 and 5.

Input:
Question: {question}
Context: {context}
Answer: {answer}
Ground Truth: {ground_truth}
""".strip()


class JudgeLLMError(RuntimeError):
    """Raised when judge LLM execution fails."""


def _coerce_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 1
    return max(1, min(5, score))


def _extract_json_block(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return None

    return None


def _normalize_scores(parsed: dict[str, Any] | None) -> dict[str, int]:
    # Safe fallback if model output is invalid.
    if not isinstance(parsed, dict):
        return {key: 1 for key in REQUIRED_SCORE_KEYS}

    return {key: _coerce_score(parsed.get(key, 1)) for key in REQUIRED_SCORE_KEYS}


def _build_messages(question: str, context: str, answer: str, ground_truth: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a strict JSON scoring engine. Output JSON only.",
        },
        {
            "role": "user",
            "content": JUDGE_PROMPT_TEMPLATE.format(
                question=question,
                context=context,
                answer=answer,
                ground_truth=ground_truth or "N/A",
            ),
        },
    ]


def judge_llm(
    question: str,
    context: str | list[str],
    answer: str,
    ground_truth: str = "",
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    timeout: int = 45,
) -> dict[str, int]:
    """Score a single RAG output with an LLM judge.

    Returns only 1-5 integer scores for:
    context_relevance, answer_relevance, groundedness, correctness.
    """
    provider_name = provider.strip().lower()

    context_text = "\n\n".join(context) if isinstance(context, list) else str(context)
    messages = _build_messages(question, context_text, answer, ground_truth)

    try:
        if provider_name == "groq":
            groq_module = importlib.import_module("groq")
            Groq = getattr(groq_module, "Groq")

            client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=messages,
                timeout=timeout,
            )
            raw_text = response.choices[0].message.content or ""

        elif provider_name == "openai":
            openai_module = importlib.import_module("openai")
            OpenAI = getattr(openai_module, "OpenAI")

            client = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                base_url=base_url,
                timeout=timeout,
            )
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=messages,
            )
            raw_text = response.choices[0].message.content or ""

        else:
            raise JudgeLLMError("Unsupported provider. Use 'openai' or 'groq'.")

    except Exception as exc:
        raise JudgeLLMError(f"judge_llm failed: {exc}") from exc

    parsed = _extract_json_block(raw_text)
    return _normalize_scores(parsed)


def append_judge_scores(
    results: list[dict[str, Any]],
    *,
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str = "",
    latency: float | None = None,
    rag_type: str = "unknown",
    **judge_kwargs: Any,
) -> dict[str, Any]:
    """Run judge_llm and append a standardized row to a results list."""
    scores = judge_llm(
        question=question,
        context=contexts,
        answer=answer,
        ground_truth=ground_truth,
        **judge_kwargs,
    )

    row = {
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer,
        "contexts": contexts,
        "rag_type": rag_type,
        "latency": latency,
        "judge_scores": scores,
    }
    results.append(row)
    return row
