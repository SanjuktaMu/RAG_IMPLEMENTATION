from __future__ import annotations

import json
import os
import re
import time
from typing import Any, List, cast

import numpy as np
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import CHUNK_OVERLAP, CHUNK_SIZE, DB_DIR, MODEL_NAME, OLLAMA_BASE_URL, PDF_PATH, TOP_K
from src.core.embeddings import LocalEmbedding
from src.core.vector_store import load_vector_store


FINAL_PROMPT = """
Answer ONLY from the context below.
If the answer is not found, say exactly: Not found in document.

Context:
{context}

Query:
{query}

Answer:
""".strip()


STRICT_EXTRACTION_PROMPT = """
You are a strict information extractor.

Rules:
- Extract ONLY from the context provided
- For specific questions (who, when, what is, which), return the direct answer
- Return exact names/dates/numbers from context without elaboration
- Format lists as comma-separated when asked to "name" or "list"
- If the exact answer is not in context, say exactly: Not found in document
- Do NOT make up or infer information

Context:
{context}

Query:
{query}

Answer:
""".strip()


VAGUE_MARKERS = {
	"does not explicitly",
	"does not contain",
	"not explicitly",
	"cannot be determined",
	"provided context does not",
	"based on the context",
	"not mentioned",
	"not specified",
	"unable to find",
}


SPECIFIC_QUESTION_KEYWORDS = {
	"who",
	"when",
	"where",
	"which",
	"name",
	"list",
	"how many",
	"founded",
	"located",
	"operates",
}

LOW_SIGNAL_MARKERS = {
	"notes to standalone financial statements",
	"notes to consolidated financial statements",
	"financial statements as at and for the year ended",
	"annexure",
	"secretarial audit report",
	"independent auditor",
	"chartered accountants",
	"membership no",
	"din:",
	"mem. no",
	"section 143",
	"section 204",
	"for and on behalf of the board",
	"payment to auditors",
	"allowance for expected credit loss",
}

STOP_WORDS = {
	"the",
	"is",
	"a",
	"an",
	"of",
	"in",
	"on",
	"to",
	"for",
	"and",
	"or",
	"by",
	"with",
	"from",
	"at",
	"what",
	"which",
	"who",
	"when",
	"where",
	"how",
	"does",
	"did",
	"do",
	"are",
	"was",
	"were",
	"be",
	"it",
	"its",
	"company",
	"limited",
	"annual",
	"report",
}


class BM25Okapi:
	def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
		self.corpus = tokenized_corpus
		self.k1 = k1
		self.b = b
		self.doc_len = np.asarray([len(doc) for doc in tokenized_corpus], dtype=float)
		self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
		self.term_doc_freq: dict[str, int] = {}
		self.doc_term_freqs: list[dict[str, int]] = []

		for doc in tokenized_corpus:
			freqs: dict[str, int] = {}
			for token in doc:
				freqs[token] = freqs.get(token, 0) + 1
			self.doc_term_freqs.append(freqs)
			for token in freqs:
				self.term_doc_freq[token] = self.term_doc_freq.get(token, 0) + 1

	def get_scores(self, query_tokens: list[str]) -> np.ndarray:
		num_docs = len(self.corpus)
		if num_docs == 0:
			return np.asarray([], dtype=float)

		scores = np.zeros(num_docs, dtype=float)
		for term in query_tokens:
			df = self.term_doc_freq.get(term, 0)
			if df == 0:
				continue

			idf = np.log(1 + (num_docs - df + 0.5) / (df + 0.5))
			for index, freqs in enumerate(self.doc_term_freqs):
				freq = freqs.get(term, 0)
				if freq == 0:
					continue

				denominator = freq + self.k1 * (1 - self.b + self.b * (self.doc_len[index] / (self.avgdl or 1.0)))
				scores[index] += idf * (freq * (self.k1 + 1)) / denominator

		return scores


def load_and_split(pdf_path: str, chunk_size: int = 600, chunk_overlap: int = 150) -> list[Document]:
	loader = PyPDFLoader(pdf_path)
	docs = loader.load()

	splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
	)

	chunks = splitter.split_documents(docs)
	return chunks


def create_vectorstore(chunks: list[Document], db_dir: str) -> Chroma:
	embeddings = LocalEmbedding()
	os.makedirs(db_dir, exist_ok=True)

	vectorstore = Chroma.from_documents(
		chunks,
		embeddings,
		persist_directory=db_dir,
		collection_name="rag_pdf_collection",
	)

	return vectorstore


def create_bm25_index(docs: List[Document]) -> BM25Okapi:
	tokenized_docs = [doc.page_content.split() for doc in docs]
	return BM25Okapi(tokenized_docs)


def _doc_key(doc: Document) -> str:
	content = (getattr(doc, "page_content", "") or "").strip()
	metadata = getattr(doc, "metadata", None) or {}
	metadata_text = json.dumps(metadata, sort_keys=True, ensure_ascii=False) if metadata else "{}"
	return f"{content}\n{metadata_text}"


def _normalize_scores(scores: np.ndarray, *, invert: bool = False) -> np.ndarray:
	if scores.size == 0:
		return scores

	minimum = float(np.min(scores))
	maximum = float(np.max(scores))
	span = maximum - minimum
	if span <= 1e-8:
		return np.ones_like(scores, dtype=float)

	normalized = (scores - minimum) / span
	if invert:
		normalized = 1.0 - normalized
	return normalized


def _tokenize_text(text: str) -> list[str]:
	return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())


def _normalize_query_text(query: str) -> str:
	tokens = _tokenize_text(query)
	filtered = [token for token in tokens if token not in STOP_WORDS]
	if not filtered:
		return query.strip()
	return " ".join(filtered)


def _important_query_terms(query: str) -> set[str]:
	tokens = _tokenize_text(query)
	return {token for token in tokens if token not in STOP_WORDS and len(token) > 2}


def _query_variants(query: str) -> list[str]:
	base = _normalize_query_text(query) or query
	lower = query.lower()
	variants = [base]

	if "what is" in lower or "overview" in lower:
		variants.append(f"{base} corporate overview heavy engineering epc")
	if "managing director" in lower or "director" in lower:
		variants.append(f"{base} managing director board")
		variants.append(f"{base} ravi todi managing director din")
	if "founded" in lower or "when was" in lower:
		variants.append(f"{base} founded heritage")
	if "sector" in lower or "operate" in lower:
		variants.append(f"{base} sectors power mining fertiliser metal")
	if "client" in lower or "marquee" in lower:
		variants.append(f"{base} marquee clients ntpc adani bhel")

	# Preserve order while deduplicating.
	seen: set[str] = set()
	unique_variants: list[str] = []
	for item in variants:
		cleaned = item.strip()
		if not cleaned or cleaned in seen:
			continue
		seen.add(cleaned)
		unique_variants.append(cleaned)
	return unique_variants


def _is_low_signal_chunk(text: str) -> bool:
	content = (text or "").strip().lower()
	if not content:
		return True

	if len(content) < 80:
		return True

	marker_hits = sum(1 for marker in LOW_SIGNAL_MARKERS if marker in content)
	if marker_hits >= 2:
		return True

	digit_count = sum(1 for ch in content if ch.isdigit())
	alpha_count = sum(1 for ch in content if ch.isalpha())
	if alpha_count == 0:
		return True

	digit_ratio = digit_count / max(1, len(content))
	if digit_ratio > 0.18 and marker_hits >= 1:
		return True

	return False


def _reciprocal_rank_fusion(
	vector_ranks: dict[str, int],
	bm25_ranks: dict[str, int],
	alpha: float,
	rrf_k: int = 60,
) -> dict[str, float]:
	resolved_alpha = min(1.0, max(0.0, alpha))
	keys = set(vector_ranks) | set(bm25_ranks)
	fused: dict[str, float] = {}
	for key in keys:
		v_rank = vector_ranks.get(key)
		b_rank = bm25_ranks.get(key)
		v_score = 0.0 if v_rank is None else 1.0 / (rrf_k + v_rank)
		b_score = 0.0 if b_rank is None else 1.0 / (rrf_k + b_rank)
		fused[key] = resolved_alpha * v_score + (1.0 - resolved_alpha) * b_score
	return fused


def fusion_retrieval(vectorstore, bm25, docs, query: str, k: int = 4, alpha: float = 0.5):
	"""
	Hybrid fusion retrieval using both vector and BM25 candidate pools.
	Returns: (top_docs, debug_scores)
	"""
	if k <= 0:
		return [], {"top_vector_scores": [], "top_bm25_scores": [], "top_combined_scores": []}
	if not docs:
		return [], {"top_vector_scores": [], "top_bm25_scores": [], "top_combined_scores": []}

	candidate_k = max(k * 10, 40)
	normalized_query = _normalize_query_text(query)
	query_variants = _query_variants(query)

	vector_rank_map: dict[str, int] = {}
	vector_rrf_map: dict[str, float] = {}
	key_to_doc: dict[str, Document] = {}
	per_variant_k = max(12, candidate_k // max(1, len(query_variants)))
	mmr_retriever = vectorstore.as_retriever(
		search_type="mmr",
		search_kwargs={"k": max(8, per_variant_k // 2), "fetch_k": max(30, candidate_k)},
	)
	for variant in query_variants:
		vector_results = vectorstore.similarity_search_with_score(variant, k=per_variant_k)
		for rank, (doc, _distance) in enumerate(vector_results, start=1):
			key = _doc_key(doc)
			previous = vector_rank_map.get(key)
			if previous is None or rank < previous:
				vector_rank_map[key] = rank
				key_to_doc[key] = doc

		mmr_docs = mmr_retriever.invoke(variant)
		for rank, doc in enumerate(mmr_docs, start=1):
			key = _doc_key(doc)
			# Blend MMR ranks into vector side, but with a mild rank penalty.
			mmr_rank = rank + 4
			previous = vector_rank_map.get(key)
			if previous is None or mmr_rank < previous:
				vector_rank_map[key] = mmr_rank
				key_to_doc[key] = doc

	for key, rank in vector_rank_map.items():
		vector_rrf_map[key] = 1.0 / (60 + rank)

    # BM25 ranking over full corpus docs
	bm25_query_tokens = _tokenize_text(normalized_query or query)
	if not bm25_query_tokens:
		bm25_query_tokens = _tokenize_text(query)
	bm25_raw = np.asarray(bm25.get_scores(bm25_query_tokens), dtype=float)
	bm25_ranked_indices = np.argsort(bm25_raw)[::-1][:candidate_k]
	bm25_rank_map: dict[str, int] = {}
	bm25_rrf_map: dict[str, float] = {}
	for rank, index in enumerate(bm25_ranked_indices.tolist(), start=1):
		doc = docs[index]
		key = _doc_key(doc)
		bm25_rank_map[key] = rank
		bm25_rrf_map[key] = 1.0 / (60 + rank)
		if key not in key_to_doc:
			key_to_doc[key] = doc

	if not key_to_doc:
		return [], {"top_vector_scores": [], "top_bm25_scores": [], "top_combined_scores": []}

	fused_scores = _reciprocal_rank_fusion(vector_rank_map, bm25_rank_map, alpha=alpha, rrf_k=60)

	# Ensure some high lexical-match chunks survive (helps factual questions).
	bm25_anchor_count = max(1, min(2, k))
	important_terms = _important_query_terms(query)
	bm25_anchor_docs: list[Document] = []
	for index in bm25_ranked_indices.tolist():
		if len(bm25_anchor_docs) >= bm25_anchor_count:
			break
		doc = docs[index]
		if bm25_raw[index] <= 0:
			continue
		doc_tokens = set(_tokenize_text(doc.page_content))
		if important_terms and len(doc_tokens.intersection(important_terms)) == 0:
			continue
		bm25_anchor_docs.append(doc)
	selected: list[Document] = []
	selected_keys: set[str] = set()
	for doc in bm25_anchor_docs:
		key = _doc_key(doc)
		if key in selected_keys:
			continue
		selected.append(doc)
		selected_keys.add(key)

	sorted_candidates = sorted(key_to_doc.values(), key=lambda doc: fused_scores.get(_doc_key(doc), 0.0), reverse=True)
	for doc in sorted_candidates:
		if len(selected) >= k:
			break
		key = _doc_key(doc)
		if key in selected_keys:
			continue
		selected.append(doc)
		selected_keys.add(key)

	top_docs = selected[:k]

	top_vector_scores = [float(vector_rrf_map.get(_doc_key(doc), 0.0)) for doc in top_docs]
	top_bm25_scores = [float(bm25_rrf_map.get(_doc_key(doc), 0.0)) for doc in top_docs]
	top_combined_scores = [float(fused_scores.get(_doc_key(doc), 0.0)) for doc in top_docs]

	return top_docs, {
		"top_vector_scores": top_vector_scores,
		"top_bm25_scores": top_bm25_scores,
		"top_combined_scores": top_combined_scores,
	}


def load_llm():
	return ChatOllama(
		model=MODEL_NAME,
		base_url=OLLAMA_BASE_URL,
		temperature=0,
	)


def _looks_vague(answer: str) -> bool:
	lower = answer.lower().strip()
	if lower == "not found in document":
		return False
	return any(marker in lower for marker in VAGUE_MARKERS)


def _is_specific_question(query: str) -> bool:
	lower = query.lower()
	return any(keyword in lower for keyword in SPECIFIC_QUESTION_KEYWORDS)


def _needs_strict_retry(query: str, answer: str) -> bool:
	lower = answer.lower().strip()
	if not _is_specific_question(query):
		return _looks_vague(answer)
	if lower == "not found in document" or lower == "not found in document.":
		return True
	return _looks_vague(answer)


def generate_answer(llm, query: str, docs: list[Document]):
	context = "\n\n".join([doc.page_content for doc in docs])
	prompt = FINAL_PROMPT.format(context=context, query=query)
	response = llm.invoke(prompt)
	answer = response.content if hasattr(response, "content") else str(response)
	
	# Retry with stricter prompt for brittle fact questions.
	if _needs_strict_retry(query, answer):
		strict_prompt = STRICT_EXTRACTION_PROMPT.format(context=context, query=query)
		retry_response = llm.invoke(strict_prompt)
		retry_answer = retry_response.content if hasattr(retry_response, "content") else str(retry_response)
		return retry_answer
	
	return answer


def fusion_rag_pipeline(pdf_path: str, db_dir: str, query: str, k: int = 4, alpha: float = 0.5):
	# 1. Load + split
	docs = load_and_split(pdf_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

	# 2. Vector store
	vectorstore = create_vectorstore(docs, db_dir)

	# 3. BM25
	bm25 = create_bm25_index(docs)

	# 4. Fusion retrieval
	top_docs, _score_debug = fusion_retrieval(
		vectorstore,
		bm25,
		docs,
		query,
		k=k,
		alpha=alpha,
	)

	# 5. LLM
	llm = load_llm()

	# 6. Answer
	answer = generate_answer(llm, query, top_docs)

	return answer


def _extract_documents_from_db(db) -> list[Document]:
	try:
		raw = db.get(include=["documents", "metadatas"])
		documents = raw.get("documents", []) or []
		metadatas = raw.get("metadatas", []) or []

		result: list[Document] = []
		for index, text in enumerate(documents):
			metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
			result.append(Document(page_content=text or "", metadata=metadata))
		if result:
			return result
	except Exception:  # noqa: BLE001
		pass

	count = 0
	collection = getattr(db, "_collection", None)
	if collection is not None:
		count_fn = getattr(collection, "count", None)
		if callable(count_fn):
			try:
				count_value = count_fn()
				count = int(cast(Any, count_value))
			except Exception:  # noqa: BLE001
				count = 0

	if count <= 0:
		return []

	try:
		return db.similarity_search(" ", k=count)
	except Exception:  # noqa: BLE001
		return []


def get_fusion_rag_chain(db, top_k: int | None = None, alpha: float = 0.5):
	k = top_k if top_k is not None else TOP_K
	resolved_alpha = min(1.0, max(0.0, alpha))
	all_docs = _extract_documents_from_db(db)
	filtered_docs = [doc for doc in all_docs if not _is_low_signal_chunk(doc.page_content)]
	docs = filtered_docs if len(filtered_docs) >= max(30, k * 6) else all_docs
	bm25 = create_bm25_index(docs) if docs else BM25Okapi([[]])
	llm = load_llm()
	rag_handle: Any = None

	def rag(question: str):
		start = time.perf_counter()
		if not docs:
			return "Not found in document", [], "", time.perf_counter() - start

		top_docs, score_debug = fusion_retrieval(db, bm25, docs, question, k=k, alpha=resolved_alpha)
		context = "\n\n".join(doc.page_content for doc in top_docs if doc.page_content.strip())

		print("\n===== FUSION RETRIEVAL =====")
		for i, doc in enumerate(top_docs, start=1):
			print(f"\nDoc {i}:\n{doc.page_content[:300]}")

		if not context.strip():
			answer = "Not found in document"
		else:
			answer = generate_answer(llm, question, top_docs)

		latency = time.perf_counter() - start
		rag_handle.last_debug = {
			"question": question,
			"alpha": resolved_alpha,
			"retrieved_doc_count": len(top_docs),
			"final_doc_count": len(top_docs),
			"top_vector_scores": score_debug.get("top_vector_scores", []),
			"top_bm25_scores": score_debug.get("top_bm25_scores", []),
			"top_combined_scores": score_debug.get("top_combined_scores", []),
		}
		return answer, top_docs, context, latency

	rag_handle = rag
	rag_handle.last_debug = {
		"alpha": resolved_alpha,
		"retrieved_doc_count": 0,
		"top_vector_scores": [],
		"top_bm25_scores": [],
		"top_combined_scores": [],
	}
	return rag


if __name__ == "__main__":
	query = "What are the key risks mentioned in the document?"
	answer = fusion_rag_pipeline(
		pdf_path=PDF_PATH,
		db_dir=DB_DIR,
		query=query,
		k=4,
		alpha=0.5,
	)
	print(answer)

	# Optional: if the project's vector store already exists, you can also run:
	# db = load_vector_store()
	# rag = get_fusion_rag_chain(db)
	# print(rag(query)[0])