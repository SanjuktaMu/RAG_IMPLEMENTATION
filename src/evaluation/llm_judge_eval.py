import argparse
import json
import os
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.core.config import EVAL_DIR, MODEL_NAME, OLLAMA_BASE_URL

RESULTS_DIR = Path(EVAL_DIR) / "results"
SCORES_DIR = Path(EVAL_DIR) / "scores"

JUDGE_PROMPT = """
You are an impartial evaluator for RAG answers.

Evaluate the model answer using ONLY these fields:
- Question
- Ground Truth
- Model Answer

Scoring rubric:
- correct: Answer matches the ground truth factually. Small wording differences are acceptable.
- partially_correct: Answer contains a part of the ground truth, but misses important details or includes uncertain/conflicting content.
- incorrect: Answer is wrong, contradictory, or says not found when the ground truth expects a concrete answer.

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
Model Answer: {answer}
""".strip()


def normalize_number_text(text: str) -> str:
    return text.replace(",", "").replace("H", "").strip().lower()


def parse_judge_response(raw: str) -> dict[str, Any]:
    raw = raw.strip()

    # Best case: pure JSON.
    try:
        data = json.loads(raw)
        return data
    except json.JSONDecodeError:
        pass

    # Fallback: extract first JSON object block.
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start : end + 1]
        try:
            data = json.loads(candidate)
            return data
        except json.JSONDecodeError:
            pass

    return {
        "verdict": "incorrect",
        "score": 0.0,
        "reason": f"Unparseable judge output: {raw[:180]}"
    }


def coerce_result(data: dict[str, Any]) -> dict[str, Any]:
    verdict = str(data.get("verdict", "incorrect")).strip().lower()
    if verdict not in {"correct", "partially_correct", "incorrect"}:
        verdict = "incorrect"

    try:
        score = float(data.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0

    score = max(0.0, min(1.0, score))
    reason = str(data.get("reason", ""))[:500]

    return {
        "verdict": verdict,
        "score": score,
        "reason": reason,
    }


def extract_existing_evaluation(row: dict[str, Any]) -> dict[str, Any] | None:
    evaluation = row.get("evaluation")
    if not isinstance(evaluation, dict):
        return None

    if "verdict" not in evaluation or "score" not in evaluation:
        return None

    return coerce_result(evaluation)


def evaluate_file(file_path: Path, llm: ChatOllama) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = json.loads(file_path.read_text(encoding="utf-8"))
    prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)

    judged_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, 1):
        judged = extract_existing_evaluation(row)
        if judged is None:
            messages = prompt.format_messages(
                question=row.get("question", ""),
                ground_truth=row.get("ground_truth", ""),
                answer=row.get("answer", ""),
            )
            raw = llm.invoke(messages).content
            parsed = parse_judge_response(raw)
            judged = coerce_result(parsed)

        judged_rows.append(
            {
                "id": idx,
                "question": row.get("question", ""),
                "ground_truth": row.get("ground_truth", ""),
                "answer": row.get("answer", ""),
                "verdict": judged["verdict"],
                "score": judged["score"],
                "reason": judged["reason"],
            }
        )

    total = len(judged_rows)
    correct = sum(r["verdict"] == "correct" for r in judged_rows)
    partial = sum(r["verdict"] == "partially_correct" for r in judged_rows)
    incorrect = sum(r["verdict"] == "incorrect" for r in judged_rows)
    avg_score = (sum(r["score"] for r in judged_rows) / total) if total else 0.0

    summary = {
        "file": file_path.name,
        "total": total,
        "correct": correct,
        "partially_correct": partial,
        "incorrect": incorrect,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "partial_credit_accuracy": round((correct + 0.5 * partial) / total, 4) if total else 0.0,
        "avg_score": round(avg_score, 4),
        "model": MODEL_NAME,
        "base_url": OLLAMA_BASE_URL,
    }

    return summary, judged_rows


def output_dir_for(file_path: Path) -> Path:
    rag_type = file_path.stem.replace("_results", "")
    return SCORES_DIR / rag_type


def find_result_files(single_file: str | None) -> list[Path]:
    if single_file:
        path = Path(single_file)
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {single_file}")
        return [path]

    if not RESULTS_DIR.exists():
        return []

    return sorted(RESULTS_DIR.glob("*_results.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM judge evaluation for RAG result files")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a single result json file. If omitted, evaluates all *_results.json in evaluation/results.",
    )
    args = parser.parse_args()

    files = find_result_files(args.file)
    if not files:
        print("No result files found to evaluate.")
        return

    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    all_summaries: list[dict[str, Any]] = []

    for file_path in files:
        print(f"\nEvaluating {file_path.as_posix()} with LLM judge...")
        summary, judged_rows = evaluate_file(file_path, llm)
        all_summaries.append(summary)

        out_dir = output_dir_for(file_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        details_path = out_dir / "llm_judge_details.json"
        summary_path = out_dir / "llm_judge_summary.json"

        details_path.write_text(json.dumps(judged_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Saved details: {details_path.as_posix()}")
        print(f"Saved summary: {summary_path.as_posix()}")

    aggregate_path = SCORES_DIR / "llm_judge_overall_summary.json"
    aggregate_path.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved aggregate summary: {aggregate_path.as_posix()}")


if __name__ == "__main__":
    main()
