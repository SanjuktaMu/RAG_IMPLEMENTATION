import argparse
from importlib import import_module
import json
from pathlib import Path

from src.core.config import EVAL_DIR
from src.core.vector_store import load_vector_store
from src.evaluation.common_eval import standardize_rag_result

DATASET_PATH = Path(EVAL_DIR) / "dataset.json"
OUTPUT_PATH = Path(EVAL_DIR) / "results" / "memo_results.json"
LOG_PATH = Path(EVAL_DIR) / "results" / "memo_logs.json"


def _resolve_memo_chain(db, top_k: int | None, history_turns: int, memory_snippets: int):
	module = import_module("src.rag_types.memo_rag")
	get_memo_rag_chain = getattr(module, "get_memo_rag_chain")
	return get_memo_rag_chain(
		db,
		top_k=top_k,
		history_turns=history_turns,
		memory_snippet_count=memory_snippets,
	)


def parse_args():
	parser = argparse.ArgumentParser(description="Run and evaluate the Memo RAG pipeline")
	parser.add_argument("--limit", type=int, default=None, help="Optional number of dataset rows to process")
	parser.add_argument("--top-k", type=int, default=None, help="Retriever top-k")
	parser.add_argument("--history-turns", type=int, default=4, help="Number of turns retained in chat memory")
	parser.add_argument("--memory-snippets", type=int, default=2, help="Number of memory snippets merged into context")
	parser.add_argument("--carry-history", action="store_true", help="Carry memory across dataset rows")
	parser.add_argument("--results-file", type=str, default=str(OUTPUT_PATH), help="Path to output JSON results file")
	parser.add_argument("--log-file", type=str, default=str(LOG_PATH), help="Path to output JSON debug log file")
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

	rag = _resolve_memo_chain(
		db,
		top_k=args.top_k,
		history_turns=max(0, args.history_turns),
		memory_snippets=max(0, args.memory_snippets),
	)

	results = []
	logs = []

	for i, item in enumerate(dataset):
		if not args.carry_history and hasattr(rag, "reset_history"):
			rag.reset_history()

		question = item.get("question", "")
		ground_truth = item.get("ground_truth", "")

		print("\n==============================")
		print(f"Q{i + 1}: {question}")

		answer, docs, _context, latency = rag(question)
		answer_text = answer if isinstance(answer, str) else str(answer)
		debug_info = getattr(rag, "last_debug", {}) or {}

		print("\n===== MEMO QUERY =====")
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
				rag_type="memo",
				ground_truth=ground_truth,
			)
		)

		logs.append(
			{
				"question": question,
				"ground_truth": ground_truth,
				"rewritten_query": debug_info.get("rewritten_query"),
				"retrieved_doc_count": debug_info.get("retrieved_doc_count"),
				"memory_doc_count": debug_info.get("memory_doc_count"),
				"memory_bank_size": debug_info.get("memory_bank_size"),
				"latency": latency,
				"answer": answer_text,
			}
		)

	results_path.parent.mkdir(parents=True, exist_ok=True)
	with results_path.open("w", encoding="utf-8") as f:
		json.dump(results, f, indent=2, ensure_ascii=False)

	log_path.parent.mkdir(parents=True, exist_ok=True)
	with log_path.open("w", encoding="utf-8") as f:
		json.dump(logs, f, indent=2, ensure_ascii=False)

	print("\n✅ Memo RAG results saved successfully!")
	print(f"✅ Memo RAG logs saved successfully: {log_path.as_posix()}")


if __name__ == "__main__":
	main()
