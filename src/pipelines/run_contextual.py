import argparse
import json
from pathlib import Path

from src.core.config import EVAL_DIR
from src.core.vector_store import load_vector_store
from src.evaluation.common_eval import standardize_rag_result
from src.rag_types.contextual_rag import get_contextual_rag_chain

DATASET_PATH = Path(EVAL_DIR) / "dataset.json"
OUTPUT_PATH = Path(EVAL_DIR) / "results" / "contextual_results.json"


def parse_args():
	parser = argparse.ArgumentParser(description="Run and evaluate the Contextual RAG pipeline")
	parser.add_argument("--limit", type=int, default=None, help="Optional number of dataset rows to process")
	parser.add_argument("--history-turns", type=int, default=3, help="Number of prior turns kept for query rewriting")
	parser.add_argument("--carry-history", action="store_true", help="Carry chat history across dataset rows")
	parser.add_argument("--top-k", type=int, default=None, help="Retriever top-k for each query")
	parser.add_argument("--results-file", type=str, default=str(OUTPUT_PATH), help="Path to the JSON results file")
	return parser.parse_args()


def main():
	args = parse_args()
	results_path = Path(args.results_file)

	with DATASET_PATH.open("r", encoding="utf-8") as f:
		dataset = json.load(f)

	if args.limit is not None:
		dataset = dataset[: max(0, args.limit)]

	db = load_vector_store()
	if db is None:
		raise ValueError("Vector store not loaded properly.")

	rag = get_contextual_rag_chain(
		db,
		top_k=args.top_k,
		history_turns=max(0, args.history_turns),
	)
	results = []

	for i, item in enumerate(dataset):
		if not args.carry_history and hasattr(rag, "reset_history"):
			rag.reset_history()

		question = item.get("question", "")
		ground_truth = item.get("ground_truth", "")

		print("\n==============================")
		print(f"Q{i + 1}: {question}")

		answer, docs, context, latency = rag(question)
		answer_text = answer if isinstance(answer, str) else str(answer)
		debug_info = getattr(rag, "last_debug", {}) or {}

		print("\n===== REWRITTEN QUERY =====")
		print(debug_info.get("rewritten_query", question))
		print("\n===== FINAL ANSWER =====")
		print(answer_text)

		contexts = [d.page_content for d in docs]
		results.append(
			standardize_rag_result(
				question=question,
				answer=answer_text,
				contexts=contexts,
				latency=latency,
				rag_type="contextual",
				ground_truth=ground_truth,
			)
		)

	results_path.parent.mkdir(parents=True, exist_ok=True)
	with results_path.open("w", encoding="utf-8") as f:
		json.dump(results, f, indent=2, ensure_ascii=False)

	print("\n✅ Contextual RAG results saved successfully!")


if __name__ == "__main__":
	main()
