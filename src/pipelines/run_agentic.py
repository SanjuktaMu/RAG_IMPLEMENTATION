import argparse
import json
from pathlib import Path

from src.core.config import EVAL_DIR
from src.core.vector_store import load_vector_store
from src.evaluation.common_eval import standardize_rag_result
from src.rag_types.agentic_rag import get_agentic_rag_chain

DATASET_PATH = Path(EVAL_DIR) / "dataset.json"
OUTPUT_PATH = Path(EVAL_DIR) / "results" / "agentic_results.json"
LOG_PATH = Path(EVAL_DIR) / "results" / "agentic_logs.json"


def parse_args():
	parser = argparse.ArgumentParser(description="Run and evaluate the Agentic RAG pipeline")
	parser.add_argument("--limit", type=int, default=None, help="Optional number of dataset rows to process")
	parser.add_argument("--top-k", type=int, default=None, help="Retriever top-k for each retrieval query")
	parser.add_argument("--query-count", type=int, default=4, help="Number of generated queries for expand/decompose")
	parser.add_argument("--history-turns", type=int, default=3, help="History window used for contextual query handling")
	parser.add_argument("--final-limit", type=int, default=None, help="Maximum docs kept after reranking")
	parser.add_argument("--disable-rerank", action="store_true", help="Disable reranking step")
	parser.add_argument("--carry-history", action="store_true", help="Carry conversation history across dataset rows")
	parser.add_argument("--results-file", type=str, default=str(OUTPUT_PATH), help="Path to the JSON results file")
	parser.add_argument("--log-file", type=str, default=str(LOG_PATH), help="Path to the JSON debug log file")
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

	rag = get_agentic_rag_chain(
		db,
		top_k=args.top_k,
		query_count=max(1, args.query_count),
		history_turns=max(0, args.history_turns),
		final_limit=args.final_limit,
		enable_rerank=not args.disable_rerank,
	)

	results = []
	logs = []

	for i, item in enumerate(dataset):
		if not args.carry_history:
			reset_history = getattr(rag, "reset_history", None)
			if callable(reset_history):
				reset_history()

		question = item.get("question", "")
		ground_truth = item.get("ground_truth", "")

		print("\n==============================")
		print(f"Q{i + 1}: {question}")

		answer, docs, context, latency = rag(question)
		answer_text = answer if isinstance(answer, str) else str(answer)
		debug_info = getattr(rag, "last_debug", {}) or {}

		print("\n===== AGENTIC STRATEGY =====")
		print(f"Strategy: {debug_info.get('selected_strategy', 'SIMPLE')}")
		print("\n===== FINAL ANSWER =====")
		print(answer_text)

		contexts = [d.page_content for d in docs]
		results.append(
			standardize_rag_result(
				question=question,
				answer=answer_text,
				contexts=contexts,
				latency=latency,
				rag_type="agentic",
				ground_truth=ground_truth,
			)
		)

		logs.append(
			{
				"question": question,
				"ground_truth": ground_truth,
				"selected_strategy": debug_info.get("selected_strategy"),
				"rewritten_query": debug_info.get("rewritten_query"),
				"generated_queries": debug_info.get("generated_queries", []),
				"sub_queries": debug_info.get("sub_queries", []),
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

	print("\n✅ Agentic RAG results saved successfully!")
	print(f"✅ Agentic logs saved successfully: {log_path.as_posix()}")


if __name__ == "__main__":
	main()
