import argparse
from importlib import import_module
import json
from pathlib import Path

from src.core.config import EVAL_DIR
from src.core.vector_store import load_vector_store
from src.evaluation.common_eval import standardize_rag_result

DATASET_PATH = Path(EVAL_DIR) / "dataset.json"
OUTPUT_PATH = Path(EVAL_DIR) / "results" / "multimodal_results.json"
LOG_PATH = Path(EVAL_DIR) / "results" / "multimodal_logs.json"
CUSTOM_OUTPUT_PATH = Path(EVAL_DIR) / "results" / "multimodal_custom_results.json"
CUSTOM_LOG_PATH = Path(EVAL_DIR) / "results" / "multimodal_custom_logs.json"


def _resolve_multimodal_chain(db, top_k: int | None):
    module = import_module("src.rag_types.multimodal_rag")
    get_multimodal_rag_chain = getattr(module, "get_multimodal_rag_chain")
    return get_multimodal_rag_chain(db, top_k=top_k)


def parse_args():
    parser = argparse.ArgumentParser(description="Run and evaluate the Multimodal RAG pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of dataset rows to process")
    parser.add_argument("--top-k", type=int, default=None, help="Retriever top-k")
    parser.add_argument("--results-file", type=str, default=str(OUTPUT_PATH), help="Output JSON results file")
    parser.add_argument("--log-file", type=str, default=str(LOG_PATH), help="Output JSON debug log file")
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Optional custom query. Can be used multiple times. Saved separately from dataset results.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for custom queries interactively until you submit a blank line.",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip the dataset evaluation and run only custom queries.",
    )
    return parser.parse_args()


def _run_query(
    rag,
    question: str,
    ground_truth: str = "",
) -> tuple[str, list[str], float, dict[str, object]]:
    answer, docs, _context, latency = rag(question)
    answer_text = answer if isinstance(answer, str) else str(answer)
    debug_info = getattr(rag, "last_debug", {}) or {}

    print("\n===== FINAL ANSWER =====")
    print(answer_text)

    contexts = [d.page_content for d in docs]
    log_entry = {
        "question": question,
        "ground_truth": ground_truth,
        "retrieved_doc_count": debug_info.get("retrieved_doc_count"),
        "final_context_length": debug_info.get("final_context_length"),
        "used_table_data": debug_info.get("used_table_data"),
        "latency": latency,
        "answer": answer_text,
    }

    return answer_text, contexts, float(latency), log_entry


def main():
    args = parse_args()
    results_path = Path(args.results_file)
    log_path = Path(args.log_file)

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.limit is not None:
        dataset = dataset[: max(0, args.limit)]

    db = load_vector_store()
    if db is None:
        raise ValueError("Vector store not loaded properly.")

    rag = _resolve_multimodal_chain(db, top_k=args.top_k)
    results = []
    logs = []

    if not args.skip_dataset:
        for i, item in enumerate(dataset):
            question = item.get("question", "")
            ground_truth = item.get("ground_truth", "")

            print("\n==============================")
            print(f"Q{i + 1}: {question}")

            answer_text, contexts, latency, log_entry = _run_query(rag, question, ground_truth=ground_truth)

            results.append(
                standardize_rag_result(
                    question=question,
                    answer=answer_text,
                    contexts=contexts,
                    latency=latency,
                    rag_type="multimodal",
                    ground_truth=ground_truth,
                )
            )
            logs.append(log_entry)

    custom_queries = list(args.query)
    if args.interactive:
        print("\nEnter custom queries one by one. Submit a blank line to finish.")
        while True:
            user_query = input("Custom query: ").strip()
            if not user_query:
                break
            custom_queries.append(user_query)

    custom_results = []
    custom_logs = []
    for i, question in enumerate(custom_queries, start=1):
        print("\n==============================")
        print(f"Custom Q{i}: {question}")
        answer_text, contexts, latency, log_entry = _run_query(rag, question)
        custom_results.append(
            {
                "question": question,
                "ground_truth": "",
                "answer": answer_text,
                "contexts": contexts,
                "rag_type": "multimodal_custom",
                "latency": latency,
            }
        )
        custom_logs.append(log_entry)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    if custom_results:
        CUSTOM_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CUSTOM_OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(custom_results, f, indent=2, ensure_ascii=False)

        CUSTOM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CUSTOM_LOG_PATH.open("w", encoding="utf-8") as f:
            json.dump(custom_logs, f, indent=2, ensure_ascii=False)

    print("\n✅ Multimodal RAG results saved successfully!")
    print(f"✅ Multimodal RAG logs saved successfully: {log_path.as_posix()}")
    if custom_results:
        print(f"✅ Custom multimodal queries saved successfully: {CUSTOM_OUTPUT_PATH.as_posix()}")
        print(f"✅ Custom multimodal logs saved successfully: {CUSTOM_LOG_PATH.as_posix()}")


if __name__ == "__main__":
    main()
