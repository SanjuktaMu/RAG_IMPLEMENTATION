import json
from pathlib import Path

from src.core.config import EVAL_DIR
from src.core.vector_store import load_vector_store
from src.evaluation.common_eval import standardize_rag_result
from src.rag_types.simple_rag import get_simple_rag_chain

DATASET_PATH = Path(EVAL_DIR) / "dataset.json"
OUTPUT_PATH = Path(EVAL_DIR) / "results" / "simple_results.json"


def main():
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    db = load_vector_store()
    if db is None:
        raise ValueError("Vector store not loaded properly.")

    rag = get_simple_rag_chain(db)
    results = []

    for i, item in enumerate(dataset):
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")

        print("\n==============================")
        print(f"Q{i + 1}: {question}")

        answer, docs, context, latency = rag(question)

        answer_text = answer if isinstance(answer, str) else str(answer)

        print("\n===== FINAL ANSWER =====")
        print(answer_text)

        # ✅ Store clean contexts
        contexts = [d.page_content for d in docs]

        results.append(
            standardize_rag_result(
                question=question,
                answer=answer_text,
                contexts=contexts,
                latency=latency,
                rag_type="simple",
                ground_truth=ground_truth,
            )
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n✅ Simple RAG results saved successfully!")


if __name__ == "__main__":
    main()