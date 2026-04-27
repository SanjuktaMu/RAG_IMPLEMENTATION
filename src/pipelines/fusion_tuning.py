"""
Fusion RAG Tuning Script
Tests different alpha values and top-k settings to optimize accuracy
"""

import json
import time
from pathlib import Path

from src.core.config import EVAL_DIR
from src.core.vector_store import load_vector_store
from src.evaluation.common_eval import standardize_rag_result
from src.rag_types.fusion_rag import get_fusion_rag_chain

DATASET_PATH = Path(EVAL_DIR) / "dataset.json"


def evaluate_with_config(alpha: float, top_k: int, limit: int = 10):
	"""Run evaluation with specific alpha and top_k configuration"""
	with DATASET_PATH.open("r", encoding="utf-8") as f:
		dataset = json.load(f)[:limit]

	db = load_vector_store()
	if db is None:
		raise ValueError("Vector store not loaded properly.")

	rag = get_fusion_rag_chain(db, top_k=top_k, alpha=alpha)
	results = []

	for i, item in enumerate(dataset):
		question = item.get("question", "")
		ground_truth = item.get("ground_truth", "")

		try:
			answer, docs, context, latency = rag(question)
			answer_text = answer if isinstance(answer, str) else str(answer)

			evaluation = standardize_rag_result(
				question=question,
				answer=answer_text,
				contexts=docs,
				latency=latency,
				rag_type="fusion",
				ground_truth=ground_truth,
			)
			results.append(evaluation)
		except Exception as e:
			print(f"Error on Q{i + 1}: {e}")
			results.append({
				"question": question,
				"error": str(e),
			})

	# Calculate metrics
	correct = sum(1 for r in results if r.get("evaluation", {}).get("verdict") == "correct")
	partial = sum(1 for r in results if r.get("evaluation", {}).get("verdict") == "partially_correct")
	incorrect = sum(1 for r in results if r.get("evaluation", {}).get("verdict") == "incorrect")
	total = len([r for r in results if "evaluation" in r])

	accuracy = correct / total if total > 0 else 0.0
	partial_accuracy = (correct + partial) / total if total > 0 else 0.0
	avg_score = (sum(r.get("evaluation", {}).get("score", 0) for r in results) / total) if total > 0 else 0.0

	return {
		"alpha": alpha,
		"top_k": top_k,
		"total": total,
		"correct": correct,
		"partial": partial,
		"incorrect": incorrect,
		"accuracy": round(accuracy, 4),
		"partial_accuracy": round(partial_accuracy, 4),
		"avg_score": round(avg_score, 4),
	}


def main():
	print("\n🔥 FUSION RAG TUNING\n")

	# Test different alpha values
	alpha_values = [0.2, 0.3, 0.5, 0.7, 0.8]  # BM25-heavy to semantic-heavy
	top_k_values = [4, 6, 8]

	results = []

	for alpha in alpha_values:
		for top_k in top_k_values:
			print(f"Testing alpha={alpha}, top_k={top_k}...")
			config_result = evaluate_with_config(alpha, top_k, limit=10)
			results.append(config_result)
			print(
				f"  ✅ Accuracy: {config_result['accuracy']:.2%} | "
				f"Partial: {config_result['partial_accuracy']:.2%} | "
				f"Avg: {config_result['avg_score']:.4f}\n"
			)

	# Find best configuration
	best = max(results, key=lambda x: (x["accuracy"], x["avg_score"]))

	print("\n" + "="*60)
	print("TUNING SUMMARY")
	print("="*60)
	print(f"Best Configuration: alpha={best['alpha']}, top_k={best['top_k']}")
	print(f"Accuracy: {best['accuracy']:.2%}")
	print(f"Partial Accuracy: {best['partial_accuracy']:.2%}")
	print(f"Average Score: {best['avg_score']:.4f}")
	print("="*60 + "\n")

	# Save results
	output_path = Path(EVAL_DIR) / "fusion_tuning_results.json"
	with output_path.open("w") as f:
		json.dump(results, f, indent=2)
	print(f"✅ Tuning results saved to {output_path}")

	print("\n💡 RECOMMENDATIONS:")
	print(f"1. Use alpha={best['alpha']} (0 = more BM25, 1 = more semantic)")
	if best['top_k'] > 4:
		print(f"2. Increase top-k from 4 to {best['top_k']} to retrieve more context")
	else:
		print(f"2. Current top-k={best['top_k']} is optimal")


if __name__ == "__main__":
	main()
