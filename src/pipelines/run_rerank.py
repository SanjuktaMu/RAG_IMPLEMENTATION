import json
from pathlib import Path

from src.core.config import EVAL_DIR
from src.core.vector_store import load_vector_store
from src.evaluation.common_eval import standardize_rag_result
from src.rag_types.rerank_rag import get_rerank_rag_chain

DATASET_PATH = Path(EVAL_DIR) / "dataset.json"
OUTPUT_PATH = Path(EVAL_DIR) / "results" / "rerank_results.json"


def main():
    # 🔹 Load dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 🔹 Load vector DB
    db = load_vector_store()
    if db is None:
        raise ValueError("❌ Vector store not loaded properly!")

    # 🔹 Initialize RAG
    rag = get_rerank_rag_chain(db)

    results = []

    for i, item in enumerate(dataset):
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")

        print("\n==============================")
        print(f"Q{i + 1}: {question}")

        try:
            # 🔹 Run RAG
            answer, docs, context, latency = rag(question)

            # 🔹 Ensure answer is string
            answer_text = answer if isinstance(answer, str) else str(answer)

            print("\n===== FINAL ANSWER =====")
            print(answer_text)

            # 🔹 Convert Document objects → strings (IMPORTANT FIX)
            context_texts = []
            for doc in docs:
                if hasattr(doc, "page_content"):
                    context_texts.append(doc.page_content)
                else:
                    context_texts.append(str(doc))

            results.append(
                standardize_rag_result(
                    question=question,
                    answer=answer_text,
                    contexts=context_texts,  # ✅ FIXED
                    latency=latency,
                    rag_type="rerank",
                    ground_truth=ground_truth,
                )
            )

        except Exception as e:
            print(f"❌ Error processing question {i + 1}: {e}")

    # 🔹 Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n✅ Rerank results saved successfully!")


if __name__ == "__main__":
    main()