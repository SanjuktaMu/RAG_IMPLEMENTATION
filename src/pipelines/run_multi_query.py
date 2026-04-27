import json
import argparse
from pathlib import Path

from src.core.config import EVAL_DIR
from src.core.vector_store import load_vector_store
from src.evaluation.common_eval import standardize_rag_result
from src.rag_types.multi_query_rag import get_multi_query_rag_chain

DATASET_PATH = Path(EVAL_DIR) / "dataset.json"
OUTPUT_PATH = Path(EVAL_DIR) / "results" / "multi_query_results.json"
LOG_PATH = Path(EVAL_DIR) / "results" / "multi_query_logs.json"


def parse_args():
	parser = argparse.ArgumentParser(description="Run and evaluate the Multi Query RAG pipeline")
	parser.add_argument("--limit", type=int, default=None, help="Optional number of dataset rows to process")
	parser.add_argument("--query-count", type=int, default=4, help="Number of query reformulations to generate")
	parser.add_argument("--top-k", type=int, default=None, help="Retriever top-k for each reformulated query")
	parser.add_argument("--final-limit", type=int, default=None, help="Maximum number of merged docs kept for the answer")
	parser.add_argument("--max-workers", type=int, default=4, help="Maximum parallel retrieval workers")
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

	rag = get_multi_query_rag_chain(
		db,
		top_k=args.top_k,
		query_count=max(1, args.query_count),
		final_limit=args.final_limit,
		max_workers=max(1, args.max_workers),
	)
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
				rag_type="multi_query",
				ground_truth=ground_truth,
			)
		)

		logs.append(
			{
				"question": question,
				"ground_truth": ground_truth,
				"generated_queries": debug_info.get("generated_queries", []),
				"retrievals": debug_info.get("retrievals", []),
				"final_doc_count": debug_info.get("final_doc_count", len(docs)),
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

	print("\n✅ Multi Query RAG results saved successfully!")
	print(f"✅ Multi Query logs saved successfully: {log_path.as_posix()}")


if __name__ == "__main__":
	main()
