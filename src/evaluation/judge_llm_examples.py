from __future__ import annotations

import pandas as pd

from src.evaluation.judge_llm import append_judge_scores, judge_llm


def example_after_rag_execution() -> dict:
    # Pretend these values came from any RAG pipeline.
    question = "What is BTL EPC Limited?"
    answer = "BTL EPC Limited is a heavy engineering EPC company."
    contexts = [
        "BTL EPC Limited is an engineering, procurement and construction company in the heavy engineering sector.",
        "The company delivers integrated industrial EPC solutions.",
    ]
    ground_truth = "BTL EPC Limited is an engineering, procurement and construction company in the heavy engineering sector."

    scores = judge_llm(
        question=question,
        context=contexts,
        answer=answer,
        ground_truth=ground_truth,
        provider="openai",
        model="gpt-4o-mini",
        temperature=0,
    )

    row = {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
        "judge_scores": scores,
    }
    return row


def example_store_in_results_and_dataframe() -> pd.DataFrame:
    results: list[dict] = []

    append_judge_scores(
        results,
        question="Who is the Managing Director of BTL EPC Limited?",
        answer="Ravi Todi",
        contexts=["For BTL EPC Limited ... Ravi Todi ... Managing Director"],
        ground_truth="Ravi Todi is the Managing Director.",
        latency=0.84,
        rag_type="hyde",
        provider="groq",
        model="llama-3.1-70b-versatile",
        temperature=0,
    )

    rows_for_df = []
    for item in results:
        rows_for_df.append(
            {
                "question": item["question"],
                "rag_type": item["rag_type"],
                "latency": item["latency"],
                "context_relevance": item["judge_scores"]["context_relevance"],
                "answer_relevance": item["judge_scores"]["answer_relevance"],
                "groundedness": item["judge_scores"]["groundedness"],
                "correctness": item["judge_scores"]["correctness"],
            }
        )

    return pd.DataFrame(rows_for_df)


def streamlit_button_click_example() -> str:
    return """
import streamlit as st
from src.evaluation.judge_llm import judge_llm

st.title("RAG + LLM Judge")

rag_type = st.selectbox("Select RAG", ["naive", "hyde", "fusion"])
query = st.text_input("Enter query")

if st.button("Run"):
    # Replace with your real dispatcher.
    answer, docs, context, latency = run_selected_rag(rag_type, query)
    contexts = [getattr(d, "page_content", str(d)) for d in docs]

    scores = judge_llm(
        question=query,
        context=contexts,
        answer=answer,
        ground_truth="",  # optional
        provider="openai",  # or "groq"
        model="gpt-4o-mini",
        temperature=0,
    )

    st.subheader("Generated Answer")
    st.write(answer)

    st.subheader("Retrieved Context")
    for i, c in enumerate(contexts, 1):
        st.markdown(f"**Chunk {i}:** {c[:500]}")

    st.subheader("LLM Judge Scores (1-5)")
    st.json(scores)

    st.caption(f"Latency: {latency:.2f}s")
""".strip()
