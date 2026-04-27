import time
from typing import List

import numpy as np
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

from src.core.config import MODEL_NAME, OLLAMA_BASE_URL
from src.core.embeddings import LocalEmbedding

FINAL_PROMPT = """
You are a document QA assistant.

Rules:
- Answer ONLY from the provided context
- If partial information is available, answer as much as possible
- If completely missing, return exactly: Not found in document
- Do NOT use external knowledge
- Keep the answer concise and factual
- Prefer a one-line clear definition if possible

Context:
{context}

Question:
{question}

Answer:
""".strip()


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.embedding = LocalEmbedding()
        self.cross_encoder = None

        try:
            self.cross_encoder = CrossEncoder(model_name)
            print(f"✅ Loaded cross-encoder reranker: {model_name}")
        except Exception as e:
            print(f"⚠️ Cross-encoder unavailable ({e}) → using cosine similarity fallback")

    def rerank(self, query: str, docs: List[Document], top_n: int) -> List[Document]:
        if not docs:
            return []

        # 🔥 Cross-encoder reranking (best)
        if self.cross_encoder:
            pairs = [(query, d.page_content) for d in docs]
            scores = self.cross_encoder.predict(pairs)

            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in ranked[:top_n]]

        # 🔁 Fallback: cosine similarity
        query_vec = np.array(self.embedding.embed_query(query)).reshape(1, -1)
        doc_vecs = np.array(self.embedding.embed_documents([d.page_content for d in docs]))

        scores = cosine_similarity(query_vec, doc_vecs)[0]
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked[:top_n]]


def get_rerank_rag_chain(db, candidate_k: int = 12, top_n: int = 5):
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(FINAL_PROMPT)
    reranker = Reranker()

    # Build a stable in-memory corpus so candidate retrieval does not collapse
    # to a repeated low-information footer chunk.
    raw = db._collection.get(include=["documents", "metadatas"])
    all_docs: List[Document] = [
        Document(page_content=text, metadata=(meta or {}))
        for text, meta in zip(raw.get("documents", []), raw.get("metadatas", []))
        if text and text.strip()
    ]

    if all_docs:
        corpus_embeddings = np.array(
            reranker.embedding.embed_documents([d.page_content for d in all_docs])
        )
    else:
        corpus_embeddings = np.empty((0, 0))

    def retrieve_candidates(question: str, k: int) -> List[Document]:
        if not all_docs:
            return []

        query_vec = np.array(reranker.embedding.embed_query(question)).reshape(1, -1)
        sims = cosine_similarity(query_vec, corpus_embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:k]
        return [all_docs[i] for i in top_idx]

    def rag(question: str):
        start = time.perf_counter()

        # 🔹 Step 1: Retrieve broad candidate docs
        candidate_docs = retrieve_candidates(question, k=candidate_k)

        # 🔹 Step 2: Re-rank
        docs = reranker.rerank(question, candidate_docs, top_n=top_n)

        if not docs:
            docs = candidate_docs[:top_n]

        # 🔹 Step 3: Build context
        context = "\n\n".join(d.page_content for d in docs)

        # 🔹 Step 4: LLM call
        messages = prompt.format_messages(context=context, question=question)
        response = llm.invoke(messages)

        answer = response.content if hasattr(response, "content") else str(response)

        latency = time.perf_counter() - start

        return answer, docs, context, latency

    return rag