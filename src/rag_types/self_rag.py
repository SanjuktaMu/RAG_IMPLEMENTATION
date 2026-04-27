import json
import re
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.core.config import MODEL_NAME, OLLAMA_BASE_URL, TOP_K
from src.core.retriever import get_retriever


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


REFLECTION_PROMPT = """
You are reviewing a draft answer from a retrieval-augmented QA system.

Given question, context, and draft answer, decide if another retrieval attempt is needed.

Output STRICT JSON only:
{{
  "needs_retry": true or false,
  "follow_up_query": "short retrieval query string",
  "critique": "short reason"
}}

Rules:
- needs_retry=true when answer is vague, unsupported, or misses key specifics
- follow_up_query must be empty when needs_retry=false
- keep critique short

Question:
{question}

Context:
{context}

Draft answer:
{draft_answer}
""".strip()


STRICT_FINAL_PROMPT = """
You are a strict information extractor.

Rules:
- Use ONLY the context
- Return direct facts, names, or numbers
- If the answer is missing in context, return exactly: Not found in document
- No extra commentary

Context:
{context}

Question:
{question}

Previous draft answer:
{draft_answer}

Critique:
{critique}

Final answer:
""".strip()


VAGUE_MARKERS = {
	"not explicitly",
	"cannot be determined",
	"provided context does not",
	"based on the context",
	"appears to",
	"likely",
	"possibly",
}


def _normalize_query(question: str) -> str:
	query = re.sub(r"\bbtl\s*epc\s*(limited|ltd\.?)\b", "", question, flags=re.IGNORECASE)
	query = re.sub(r"\s+", " ", query).strip(" ?")
	return query if query else question


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


def _response_text(response: Any) -> str:
	content = response.content if hasattr(response, "content") else response
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		return "\n".join(str(item) for item in content)
	return str(content)


def _looks_vague(answer: str) -> bool:
	lower = answer.lower().strip()
	if lower == "not found in document":
		return False
	return any(marker in lower for marker in VAGUE_MARKERS)


def _parse_reflection(raw_text: str) -> dict[str, Any]:
	default = {
		"needs_retry": False,
		"follow_up_query": "",
		"critique": "",
	}

	text = raw_text.strip()
	if not text:
		return default

	try:
		start = text.find("{")
		end = text.rfind("}")
		if start != -1 and end != -1 and end > start:
			text = text[start : end + 1]
		parsed = json.loads(text)
		return {
			"needs_retry": bool(parsed.get("needs_retry", False)),
			"follow_up_query": str(parsed.get("follow_up_query", "") or "").strip(),
			"critique": str(parsed.get("critique", "") or "").strip(),
		}
	except Exception:  # noqa: BLE001
		pass

	lower = text.lower()
	needs_retry = "needs_retry" in lower and "true" in lower
	follow_up_query = ""
	if "follow_up_query" in lower:
		parts = re.split(r"follow_up_query\s*[:=]\s*", text, flags=re.IGNORECASE)
		if len(parts) > 1:
			follow_up_query = parts[1].splitlines()[0].strip().strip('"\'`')

	critique = ""
	if "critique" in lower:
		parts = re.split(r"critique\s*[:=]\s*", text, flags=re.IGNORECASE)
		if len(parts) > 1:
			critique = parts[1].splitlines()[0].strip().strip('"\'`')

	return {
		"needs_retry": needs_retry,
		"follow_up_query": follow_up_query,
		"critique": critique,
	}


def _dedupe_docs(docs):
	unique_docs = []
	seen = set()
	for doc in docs:
		content = getattr(doc, "page_content", "") or ""
		clean = content.strip()
		if not clean or clean in seen:
			continue
		seen.add(clean)
		unique_docs.append(doc)
	return unique_docs


def _retrieve_docs(db, retriever, query: str, k: int):
	docs = retriever.invoke(query)
	if docs:
		print("\n----- SELF-RAG RETRIEVED DOCS -----")
		for i, doc in enumerate(docs, start=1):
			meta = doc.metadata or {}
			print(
				f"Doc {i} | type={meta.get('type', 'unknown')} | "
				f"page={meta.get('page', 'unknown')}: {doc.page_content[:220]}"
			)

	docs = [d for d in docs if d.page_content and not _is_boilerplate_chunk(d.page_content)]
	if not docs:
		fallback_docs = db.similarity_search(query, k=max(10, k * 3))
		docs = [
			d
			for d in fallback_docs
			if d.page_content and not _is_boilerplate_chunk(d.page_content)
		][:k]
	return docs


def get_self_rag_chain(db, top_k: int | None = None):
	k = top_k if top_k is not None else TOP_K
	retriever = get_retriever(db, top_k=k)

	llm = ChatOllama(
		model=MODEL_NAME,
		base_url=OLLAMA_BASE_URL,
		temperature=0,
	)

	answer_prompt = ChatPromptTemplate.from_template(FINAL_PROMPT)
	reflection_prompt = ChatPromptTemplate.from_template(REFLECTION_PROMPT)
	strict_prompt = ChatPromptTemplate.from_template(STRICT_FINAL_PROMPT)
	rag_handle: Any = None

	def rag(question: str):
		start = time.perf_counter()
		normalized_question = _normalize_query(question)

		docs_round_1 = _retrieve_docs(db, retriever, normalized_question, k)
		context_round_1 = "\n\n".join(doc.page_content for doc in docs_round_1 if doc.page_content.strip())

		if not context_round_1.strip():
			latency = time.perf_counter() - start
			rag_handle.last_debug = {
				"question": question,
				"initial_query": normalized_question,
				"retry_used": False,
				"retrieved_doc_count": 0,
			}
			return "Not found in document", [], "", latency

		draft_messages = answer_prompt.format_messages(context=context_round_1, question=question)
		draft_answer = _response_text(llm.invoke(draft_messages)).strip() or "Not found in document"

		reflection_messages = reflection_prompt.format_messages(
			question=question,
			context=context_round_1,
			draft_answer=draft_answer,
		)
		reflection_raw = _response_text(llm.invoke(reflection_messages))
		reflection = _parse_reflection(reflection_raw)

		needs_retry = bool(reflection.get("needs_retry", False)) or _looks_vague(draft_answer)
		follow_up_query = str(reflection.get("follow_up_query", "") or "").strip()
		critique = str(reflection.get("critique", "") or "").strip()

		final_answer = draft_answer
		final_docs = docs_round_1
		final_context = context_round_1

		if needs_retry:
			resolved_follow_up = _normalize_query(follow_up_query) if follow_up_query else normalized_question
			docs_round_2 = _retrieve_docs(db, retriever, resolved_follow_up, k)
			merged_docs = _dedupe_docs([*docs_round_1, *docs_round_2])
			merged_docs = merged_docs[: max(k * 2, k)]

			merged_context = "\n\n".join(doc.page_content for doc in merged_docs if doc.page_content.strip())
			if merged_context.strip():
				strict_messages = strict_prompt.format_messages(
					context=merged_context,
					question=question,
					draft_answer=draft_answer,
					critique=critique,
				)
				refined_answer = _response_text(llm.invoke(strict_messages)).strip()
				if refined_answer:
					final_answer = refined_answer
				final_docs = merged_docs
				final_context = merged_context

		latency = time.perf_counter() - start
		rag_handle.last_debug = {
			"question": question,
			"initial_query": normalized_question,
			"retry_used": needs_retry,
			"follow_up_query": follow_up_query,
			"critique": critique,
			"retrieved_doc_count": len(final_docs),
		}

		return final_answer or "Not found in document", final_docs, final_context, latency

	rag_handle = rag
	rag_handle.last_debug = {
		"retry_used": False,
		"retrieved_doc_count": 0,
	}
	return rag
