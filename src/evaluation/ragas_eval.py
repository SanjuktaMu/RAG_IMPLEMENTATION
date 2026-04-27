import argparse
import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from src.core.config import EVAL_DIR

RESULTS_DIR = Path(EVAL_DIR) / "results"
SCORES_DIR = Path(EVAL_DIR) / "scores"


def find_result_files(single_file: str | None) -> list[Path]:
    if single_file:
        path = Path(single_file)
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {single_file}")
        return [path]

    if not RESULTS_DIR.exists():
        return []

    return sorted(RESULTS_DIR.glob("*_results.json"))


def result_to_summary(result: Any, file_name: str, total: int) -> dict[str, Any]:
    # RAGAS returns a Result-like object that is usually dict-like.
    if hasattr(result, "to_dict"):
        metrics = result.to_dict()
    elif isinstance(result, dict):
        metrics = result
    else:
        metrics = {}

    faithfulness_value = metrics.get("faithfulness")
    answer_relevancy_value = metrics.get("answer_relevancy")
    context_precision_value = metrics.get("context_precision")

    has_all_metrics = all(
        value is not None
        for value in [faithfulness_value, answer_relevancy_value, context_precision_value]
    )

    return {
        "file": file_name,
        "total": total,
        "faithfulness": float(faithfulness_value) if faithfulness_value is not None else None,
        "answer_relevancy": float(answer_relevancy_value) if answer_relevancy_value is not None else None,
        "context_precision": float(context_precision_value) if context_precision_value is not None else None,
        "status": "ok" if has_all_metrics else "error",
        "error": "RAGAS metrics missing in result output" if not has_all_metrics else "",
    }


def evaluate_file(file_path: Path) -> dict[str, Any]:
    # Reduce noisy retry/error logs from third-party clients during metric evaluation.
    for logger_name in ["ragas", "instructor", "openai", "httpx", "httpcore"]:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = Dataset.from_list([
        {
            "question": d["question"],
            "answer": d["answer"],
            "contexts": d["contexts"],
            "ground_truth": d["ground_truth"]
        }
        for d in data
    ])

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision
        ],
        raise_exceptions=False,
        show_progress=False,
    )

    summary = result_to_summary(result, file_path.name, len(data))
    return summary


def summary_output_path(file_path: Path) -> Path:
    rag_type = file_path.stem.replace("_results", "")
    return SCORES_DIR / rag_type / "ragas_summary.json"


def evaluate_file_safe(file_path: Path) -> dict[str, Any]:
    try:
        return evaluate_file(file_path)
    except Exception as exc:  # noqa: BLE001
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            "file": file_path.name,
            "total": len(data),
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "status": "error",
            "error": str(exc)[:500],
        }


def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation for RAG result files")
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

    all_summaries: list[dict[str, Any]] = []

    for file_path in files:
        print(f"\n📊 RAGAS RESULTS: {file_path.as_posix()}\n")
        summary = evaluate_file_safe(file_path)
        all_summaries.append(summary)
        print(json.dumps(summary, indent=2))

        summary_path = summary_output_path(file_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary: {summary_path.as_posix()}")

    overall_path = SCORES_DIR / "ragas_overall_summary.json"
    overall_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"\nSaved aggregate summary: {overall_path.as_posix()}")


if __name__ == "__main__":
    main()