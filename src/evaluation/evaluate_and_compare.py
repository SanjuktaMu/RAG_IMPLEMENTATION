import argparse
import json
import os
from pathlib import Path
from typing import Any

from langchain_ollama import ChatOllama

from src.core.config import EVAL_DIR, MODEL_NAME, OLLAMA_BASE_URL
from src.evaluation.llm_judge_eval import evaluate_file as llm_judge_evaluate_file
from src.evaluation.llm_judge_eval import find_result_files
from src.evaluation.ragas_eval import evaluate_file_safe as ragas_evaluate_file

SCORES_DIR = Path(EVAL_DIR) / "scores"


def rag_type_from_result_file(file_name: str) -> str:
    if file_name.endswith("_results.json"):
        return file_name[: -len("_results.json")]
    return file_name.replace(".json", "")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def output_dir_for(file_path: Path) -> Path:
    rag_type = rag_type_from_result_file(file_path.name)
    return SCORES_DIR / rag_type


def write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    header = "| rag_type | total | faithfulness | answer_relevancy | context_precision | llm_correct | llm_partial | llm_incorrect | llm_accuracy | llm_partial_credit_accuracy | llm_avg_score |\n"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"

    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {rag_type} | {total} | {faithfulness:.4f} | {answer_relevancy:.4f} | {context_precision:.4f} | {correct} | {partially_correct} | {incorrect} | {accuracy:.4f} | {partial_credit_accuracy:.4f} | {avg_score:.4f} |\n".format(
                **row
            )
        )

    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS + LLM Judge and generate side-by-side comparison")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a single result json file. If omitted, evaluates all *_results.json in evaluation/results.",
    )
    parser.add_argument(
        "--with-ragas",
        action="store_true",
        help="Enable RAGAS evaluation. Disabled by default to avoid external API dependency/noisy retries.",
    )
    args = parser.parse_args()

    if not args.with_ragas:
        # Ensure local-only judge mode does not accidentally use OpenAI env settings.
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_API_BASE", None)
        print("Running in local LLM Judge mode (Ollama only). RAGAS is skipped.")

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

    comparison_rows: list[dict[str, Any]] = []

    for file_path in files:
        print(f"\nEvaluating {file_path.as_posix()}...")

        if args.with_ragas:
            ragas_summary = ragas_evaluate_file(file_path)
        else:
            ragas_summary = {
                "file": file_path.name,
                "total": len(json.loads(file_path.read_text(encoding="utf-8"))),
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "status": "skipped",
                "error": "RAGAS skipped (run with --with-ragas to enable)",
            }
        llm_summary, llm_details = llm_judge_evaluate_file(file_path, llm)

        out_dir = output_dir_for(file_path)
        write_json(out_dir / "ragas_summary.json", ragas_summary)
        write_json(out_dir / "llm_judge_summary.json", llm_summary)
        write_json(out_dir / "llm_judge_details.json", llm_details)

        merged = {
            "rag_type": rag_type_from_result_file(file_path.name),
            "file": file_path.name,
            "total": ragas_summary["total"],
            "faithfulness": ragas_summary["faithfulness"] if ragas_summary["faithfulness"] is not None else 0.0,
            "answer_relevancy": ragas_summary["answer_relevancy"] if ragas_summary["answer_relevancy"] is not None else 0.0,
            "context_precision": ragas_summary["context_precision"] if ragas_summary["context_precision"] is not None else 0.0,
            "ragas_status": ragas_summary.get("status", "ok"),
            "ragas_error": ragas_summary.get("error", ""),
            "correct": llm_summary["correct"],
            "partially_correct": llm_summary["partially_correct"],
            "incorrect": llm_summary["incorrect"],
            "accuracy": llm_summary["accuracy"],
            "partial_credit_accuracy": llm_summary["partial_credit_accuracy"],
            "avg_score": llm_summary["avg_score"],
            "judge_model": llm_summary["model"],
        }
        comparison_rows.append(merged)

        print(json.dumps(merged, indent=2, ensure_ascii=False))

    comparison_rows.sort(key=lambda r: r["rag_type"])

    write_json(SCORES_DIR / "metrics_comparison.json", comparison_rows)
    write_markdown_table(SCORES_DIR / "metrics_comparison.md", comparison_rows)

    print(f"\nSaved comparison JSON: {(SCORES_DIR / 'metrics_comparison.json').as_posix()}")
    print(f"Saved comparison table: {(SCORES_DIR / 'metrics_comparison.md').as_posix()}")


if __name__ == "__main__":
    main()
