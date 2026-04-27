import argparse
import json
from pathlib import Path

from src.core.config import EVAL_DIR
from src.core.vector_store import load_vector_store
from src.evaluation.common_eval import standardize_rag_result
from src.rag_types.fusion_rag import get_fusion_rag_chain

DATASET_PATH = Path(EVAL_DIR) / "dataset.json"
OUTPUT_PATH = Path(EVAL_DIR) / "results" / "fusion_results.json"
LOG_PATH = Path(EVAL_DIR) / "results" / "fusion_logs.json"


def parse_args():
	parser = argparse.ArgumentParser(description="Run and evaluate the Fusion RAG pipeline")
	parser.add_argument("--limit", type=int, default=None, help="Optional number of dataset rows to process")
	parser.add_argument("--top-k", type=int, default=None, help="Retriever top-k after fusion ranking")
	parser.add_argument("--alpha", type=float, default=0.5, help="Weight for vector scores; BM25 weight is 1-alpha")
	parser.add_argument("--results-file", type=str, default=str(OUTPUT_PATH), help="Path to the JSON results file")
	parser.add_argument("--log-file", type=str, default=str(LOG_PATH), help="Path to the JSON trace log file")
	return parser.parse_args()


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

	rag = get_fusion_rag_chain(db, top_k=args.top_k, alpha=args.alpha)
	results = []
	logs = []

	for i, item in enumerate(dataset):
		question = item.get("question", "")
		ground_truth = item.get("ground_truth", "")

		print("\n==============================")
		print(f"Q{i + 1}: {question}")

		answer, docs, context, latency = rag(question)
		answer_text = answer if isinstance(answer, str) else str(answer)
		debug_info = getattr(rag, "last_debug", {}) or {}

		print("\n===== FINAL ANSWER =====")
		print(answer_text)

		contexts = [d.page_content for d in docs]
		results.append(
			standardize_rag_result(
				question=question,
				answer=answer_text,
				contexts=contexts,
				latency=latency,
				rag_type="fusion",
				ground_truth=ground_truth,
			)
		)

		logs.append(
			{
				"question": question,
				"ground_truth": ground_truth,
				"alpha": debug_info.get("alpha"),
				"retrieved_doc_count": debug_info.get("retrieved_doc_count"),
				"final_doc_count": debug_info.get("final_doc_count"),
				"top_vector_scores": debug_info.get("top_vector_scores", []),
				"top_bm25_scores": debug_info.get("top_bm25_scores", []),
				"top_combined_scores": debug_info.get("top_combined_scores", []),
				"answer": answer_text,
				"latency": latency,
			}
		)

	results_path.parent.mkdir(parents=True, exist_ok=True)
	with results_path.open("w", encoding="utf-8") as f:
		json.dump(results, f, indent=2, ensure_ascii=False)

	log_path.parent.mkdir(parents=True, exist_ok=True)
	with log_path.open("w", encoding="utf-8") as f:
		json.dump(logs, f, indent=2, ensure_ascii=False)

	print("\n✅ Fusion RAG results saved successfully!")
	print(f"✅ Fusion RAG logs saved successfully: {log_path.as_posix()}")


if __name__ == "__main__":
	main()
