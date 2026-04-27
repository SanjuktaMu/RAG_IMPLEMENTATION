import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from sklearn.metrics.pairwise import cosine_similarity

from src.core.config import MODEL_NAME, OLLAMA_BASE_URL, TOP_K
from src.core.embeddings import LocalEmbedding
from src.core.retriever import get_retriever


QUERY_EXPANSION_PROMPT = """
You are helping a retrieval system search a document corpus.

Generate {query_count} diverse search queries that express the same intent as the user's question.

Rules:
- Keep every query short and retrieval-oriented
- Use different wording and angles, but do not change the meaning
- Do not add explanations, numbering, bullets, or extra text
- Output one query per line only

User question:
{question}

Queries:
""".strip()


FINAL_PROMPT = """
You are a document QA assistant.

Rules:
- Answer ONLY from the provided context
- If partial information is available, answer as much as possible
- If completely missing, return exactly: Not found in document
- Do NOT use external knowledge
- Prefer concise, fact-only answers

Context:
{context}

Question:
{question}

Answer:
""".strip()


STRICT_FINAL_PROMPT = """
You are a strict information extractor.

Rules:
- Use ONLY the context
- Keep the answer short and direct
- For numeric or named-entity questions, return exact values/names from context
- If exact value/name is not in context, return exactly: Not found in document
- Do not explain your reasoning

Context:
{context}

Question:
{question}

Answer:
""".strip()


NUMERIC_HINTS = {
	"how many", "total", "revenue", "income", "expenses", "profit", "difference",
	"value", "amount", "fy", "year", "age", "strength", "tax", "crore", "lacs"
}

ENTITY_HINTS = {
	"who", "which", "name", "client", "company", "director", "mission", "vision",
	"founded", "location", "sectors", "industries"
}

VAGUE_MARKERS = {
	"does not explicitly", "does not contain", "not explicitly", "cannot be determined",
	"provided context does not", "based on the context"
}


def _is_boilerplate_chunk(text: str) -> bool:
	lower = text.lower()
	signals = [
		"registered office",
		"scan code to visit",
		"cin:",
		"ph.no",
		"corporate & marketing office",
	]
	hits = sum(1 for signal in signals if signal in lower)
	return hits >= 2


def _normalize_query(question: str) -> str:
	query = re.sub(r"\bbtl\s*epc\s*(limited|ltd\.?)\b", "", question, flags=re.IGNORECASE)
	query = re.sub(r"\s+", " ", query).strip(" ?")
	return query if query else question


def _is_numeric_or_entity_question(question: str) -> bool:
	query = question.lower()
	return any(hint in query for hint in NUMERIC_HINTS) or any(hint in query for hint in ENTITY_HINTS)


def _looks_vague(answer: str) -> bool:
	lower = answer.lower().strip()
	if lower == "not found in document":
		return False
	return any(marker in lower for marker in VAGUE_MARKERS)


def _needs_retry(question: str, answer: str) -> bool:
	if not _is_numeric_or_entity_question(question):
		return _looks_vague(answer)

	lower = answer.lower().strip()
	if lower == "not found in document":
		return False

	if _looks_vague(answer):
		return True

	if any(char.isdigit() for char in answer):
		return False

	return True


def _parse_generated_queries(raw_text: str, original_query: str, limit: int = 4) -> list[str]:
	queries: list[str] = []

	for line in raw_text.splitlines():
		cleaned = line.strip()
		if not cleaned:
			continue

		cleaned = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", cleaned)
		cleaned = cleaned.strip('"\'`')
		cleaned = re.sub(r"^queries?\s*:\s*", "", cleaned, flags=re.IGNORECASE)
		cleaned = cleaned.strip()

		if not cleaned:
			continue

		if cleaned.lower() == original_query.lower():
			continue

		if cleaned not in queries:
			queries.append(cleaned)

		if len(queries) >= limit:
			break

	return queries


def _build_query_variations(
	llm: ChatOllama,
	prompt: ChatPromptTemplate,
	question: str,
	query_count: int,
) -> list[str]:
	messages = prompt.format_messages(question=question, query_count=query_count)
	raw = llm.invoke(messages)
	content = raw.content if hasattr(raw, "content") else raw
	if isinstance(content, str):
		text = content
	elif isinstance(content, list):
		text = "\n".join(str(item) for item in content)
	else:
		text = str(content)

	queries = _parse_generated_queries(text, question, limit=query_count)

	if not queries:
		base_query = _normalize_query(question)
		fallback_templates = [
			"{base} details",
			"{base} information",
			"{base} explanation",
			"{base} summary",
			"{base} key facts",
			"{base} report",
			"{base} specific numbers",
		]
		queries = [base_query]
		for template in fallback_templates:
			if len(queries) >= query_count:
				break
			queries.append(template.format(base=base_query))

	expanded = [_normalize_query(question)]
	for query in queries:
		normalized = _normalize_query(query)
		if normalized and normalized not in expanded:
			expanded.append(normalized)

	return expanded[: max(1, query_count + 1)]


def _dedupe_docs(docs):
	unique_docs = []
	seen = set()

	for doc in docs:
		content = getattr(doc, "page_content", "") or ""
		normalized = content.strip()
		if not normalized or normalized in seen:
			continue
		seen.add(normalized)
		unique_docs.append(doc)

	return unique_docs


def _rank_docs(question: str, queries: list[str], docs, limit: int):
	if not docs:
		return [], []

	embedding = LocalEmbedding()
	doc_texts = [doc.page_content for doc in docs]
	doc_vectors = np.array(embedding.embed_documents(doc_texts))

	sources = [_normalize_query(question)]
	for query in queries:
		normalized = _normalize_query(query)
		if normalized and normalized not in sources:
			sources.append(normalized)

	query_vectors = np.array(embedding.embed_documents(sources))
	similarities = cosine_similarity(query_vectors, doc_vectors)
	scores = similarities.max(axis=0)

	ranked = sorted(zip(docs, scores.tolist()), key=lambda item: item[1], reverse=True)
	selected_docs = [doc for doc, _ in ranked[:limit]]
	selected_scores = [float(score) for _, score in ranked[:limit]]
	return selected_docs, selected_scores


def _response_text(response: Any) -> str:
	content = response.content if hasattr(response, "content") else response
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		return "\n".join(str(item) for item in content)
	return str(content)


def get_multi_query_rag_chain(
	db,
	top_k: int | None = None,
	query_count: int = 4,
	final_limit: int | None = None,
	max_workers: int = 4,
):
	k = top_k if top_k is not None else TOP_K
	per_query_k = max(k, 4)
	resolved_final_limit = final_limit if final_limit is not None else max(k * 2, 8)
	worker_count = max(1, min(max_workers, query_count + 1))

	retriever = get_retriever(db, top_k=per_query_k)

	llm = ChatOllama(
		model=MODEL_NAME,
		base_url=OLLAMA_BASE_URL,
		temperature=0,
	)

	query_prompt = ChatPromptTemplate.from_template(QUERY_EXPANSION_PROMPT)
	answer_prompt = ChatPromptTemplate.from_template(FINAL_PROMPT)
	rag_handle: Any = None

	def rag(question: str):
		start = time.perf_counter()
		debug_info = {
			"question": question,
			"generated_queries": [],
			"retrievals": [],
			"ranked_docs": [],
			"final_doc_count": 0,
		}

		# Step 1: expand the user question into multiple retrieval queries.
		queries = _build_query_variations(llm, query_prompt, question, query_count=query_count)
		debug_info["generated_queries"] = queries

		print("\n===== MULTI-QUERY VARIATIONS =====")
		for index, query in enumerate(queries, start=1):
			print(f"Query {index}: {query}")

		# Step 2: run the retrievals in parallel and merge unique documents.
		all_docs = []
		with ThreadPoolExecutor(max_workers=min(len(queries), worker_count)) as executor:
			future_map = {
				executor.submit(retriever.invoke, query): query
				for query in queries
			}

			for future in as_completed(future_map):
				query = future_map[future]
				try:
					docs = future.result() or []
				except Exception as exc:  # noqa: BLE001
					print(f"⚠️ Retrieval failed for query '{query}': {exc}")
					docs = []

				retrieval_entry = {
					"query": query,
					"doc_count": len(docs),
					"docs": [
						{
							"rank": i + 1,
							"content": doc.page_content,
						}
						for i, doc in enumerate(docs)
					],
				}
				debug_info["retrievals"].append(retrieval_entry)

				print(f"\n----- RETRIEVED DOCS FOR: {query} -----")
				if not docs:
					print("No documents retrieved!")
				for i, doc in enumerate(docs):
					meta = doc.metadata or {}
					print(
						f"\nDoc {i + 1} | type={meta.get('type', 'unknown')} | "
						f"page={meta.get('page', 'unknown')}:\n{doc.page_content[:300]}"
					)

				all_docs.extend(docs)

		docs = _dedupe_docs(all_docs)
		docs = [doc for doc in docs if doc.page_content and not _is_boilerplate_chunk(doc.page_content)]

		if not docs:
			docs = db.similarity_search(question, k=max(resolved_final_limit * 2, 10))
			docs = [doc for doc in docs if doc.page_content and not _is_boilerplate_chunk(doc.page_content)]

		docs, ranked_scores = _rank_docs(question, queries, docs, resolved_final_limit)
		debug_info["ranked_docs"] = [
			{
				"rank": index + 1,
				"score": round(score, 4),
				"content": doc.page_content,
			}
			for index, (doc, score) in enumerate(zip(docs, ranked_scores))
		]
		debug_info["final_doc_count"] = len(docs)

		print("\n===== FINAL MERGED CONTEXT =====")
		for i, doc in enumerate(docs):
			meta = doc.metadata or {}
			print(f"\n--- Chunk {i + 1} ---")
			print(f"type={meta.get('type', 'unknown')} | page={meta.get('page', 'unknown')}")
			print(doc.page_content[:300])

		context = "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())

		if not context.strip():
			return "Not found in document", docs, context, 0

		# Step 3: answer using the merged evidence.
		messages = answer_prompt.format_messages(context=context, question=question)
		response = llm.invoke(messages)
		answer = _response_text(response)

		if _needs_retry(question, answer):
			strict_prompt = ChatPromptTemplate.from_template(STRICT_FINAL_PROMPT)
			strict_messages = strict_prompt.format_messages(context=context, question=question)
			strict_response = llm.invoke(strict_messages)
			strict_answer = _response_text(strict_response)
			if strict_answer.strip():
				answer = strict_answer

		latency = time.perf_counter() - start
		rag_handle.last_debug = debug_info
		return answer, docs, context, latency

	rag_handle = rag
	rag_handle.last_debug = {}
	return rag
