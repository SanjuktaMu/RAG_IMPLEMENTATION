import json
from pathlib import Path

from src.core.config import BASE_DIR, EVAL_DIR
from src.core.vector_store import load_vector_store
from src.rag_types.hyde_rag import get_hyde_rag_chain
from src.evaluation.common_eval import standardize_rag_result

DATASET_PATH = Path(EVAL_DIR) / "dataset.json"
OUTPUT_PATH = Path(EVAL_DIR) / "results" / "hyde_results.json"


def main():
    # Load dataset
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    db = load_vector_store()
    rag = get_hyde_rag_chain(db)

    results = []

    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n==============================")
        print(f"Q{i+1}: {question}")

        answer, docs, context, latency = rag(question)

        print("\n===== FINAL ANSWER =====")
        print(answer)

        results.append(
            standardize_rag_result(
                question=question,
                answer=answer,
                contexts=docs,
                latency=latency,
                rag_type="hyde",
                ground_truth=ground_truth,
            )
        )

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n✅ HyDE results saved!")


if __name__ == "__main__":
    main()