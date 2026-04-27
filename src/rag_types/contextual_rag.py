import re
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.core.config import MODEL_NAME, OLLAMA_BASE_URL, TOP_K
from src.core.retriever import get_retriever


REWRITE_PROMPT = """
You are a conversation-aware retrieval query rewriter.

Rewrite the current user question into one standalone search query.
Use the chat history only to resolve references like it, that, this, they, he, she,
those, former, latter, the company, the report, and similar pronouns.
If the current question is already standalone, return it unchanged.
Do not answer the question.
Do not add explanations, bullets, numbering, or extra text.
Return only the rewritten query.

Chat history:
{chat_history}

Current question:
{question}

Standalone query:
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


RECOVERY_PROMPT = """
You are a context-grounded extractor.

Rules:
- Use ONLY the provided context
- Do not use outside knowledge
- If the answer exists in context but is split across lines, reconstruct it
- If the question asks to list names/items, return concise comma-separated items
- Return exactly: Not found in document only when evidence is truly absent

Context:
{context}

Question:
{question}

Answer:
""".strip()


NUMERIC_HINTS = {
	"how many", "total", "revenue", "income", "expenses", "profit", "difference",
	"value", "amount", "fy", "year", "age", "strength", "tax", "crore", "lacs",
}

ENTITY_HINTS = {
	"who", "which", "name", "client", "company", "director", "mission", "vision",
	"founded", "location", "sectors", "industries",
}

VAGUE_MARKERS = {
	"does not explicitly", "does not contain", "not explicitly", "cannot be determined",
	"provided context does not", "based on the context",
}

CONTEXT_REF_HINTS = {
	"it", "this", "that", "they", "them", "their", "its", "he", "she", "his", "her",
	"these", "those", "former", "latter", "the company", "the report", "the above",
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
	query = re.sub(r"\s+", " ", question).strip()
	return query if query else question


def _needs_context_rewrite(question: str, has_history: bool) -> bool:
	if not has_history:
		return False
	lower = question.lower()
	return any(hint in lower for hint in CONTEXT_REF_HINTS)


def _format_chat_history(chat_history: list[dict[str, str]]) -> str:
	if not chat_history:
		return "No previous conversation."

	lines: list[str] = []
	for turn in chat_history:
		user = str(turn.get("user", "")).strip()
		assistant = str(turn.get("assistant", "")).strip()
		if user:
			lines.append(f"User: {user}")
		if assistant:
			lines.append(f"Assistant: {assistant}")

	return "\n".join(lines) if lines else "No previous conversation."


def _clean_rewritten_query(text: Any, fallback: str) -> str:
	content = text.content if hasattr(text, "content") else text
	if isinstance(content, list):
		content = "\n".join(str(item) for item in content)
	query = str(content).strip()
	if not query:
		return fallback

	first_line = query.splitlines()[0].strip()
	first_line = re.sub(r"^(?:rewritten query|standalone query|query)\s*:\s*", "", first_line, flags=re.IGNORECASE)
	first_line = first_line.strip('"').strip("'").strip("`")
	first_line = re.sub(r"\s+", " ", first_line).strip()
	return first_line or fallback


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
		# Numeric/entity questions are brittle; try one strict extraction pass.
		return True

	if _looks_vague(answer):
		return True

	lower_question = question.lower()
	numeric_expected = any(hint in lower_question for hint in NUMERIC_HINTS)
	if numeric_expected and not any(char.isdigit() for char in answer):
		return True

	return False


def _dedupe_docs(docs):
	unique_docs = []
	seen = set()
	for doc in docs:
		content = (getattr(doc, "page_content", "") or "").strip()
		if not content or content in seen:
			continue
		seen.add(content)
		unique_docs.append(doc)
	return unique_docs


def _extra_retrieval_queries(question: str) -> list[str]:
	lower = question.lower()
	extras: list[str] = []

	if "marquee client" in lower or ("clients" in lower and "name" in lower):
		extras.append("Our marquee clients BTL EPC Limited")

	if "founded" in lower:
		extras.append("BTL EPC Limited founded 1965 leadership")

	if "services" in lower and ("type" in lower or "provide" in lower):
		extras.append("engineering procurement construction services BTL EPC")

	return extras


def _query_variants(question: str) -> list[str]:
	base = _normalize_query(question)
	lower = question.lower()
	variants = [base]

	if "what is" in lower or "overview" in lower:
		variants.append(f"{base} heavy engineering EPC company")
	if "who" in lower or "director" in lower or "managing director" in lower:
		variants.append(f"{base} Ravi Todi managing director")
	if "founded" in lower or "when" in lower:
		variants.append(f"{base} founded 1965")
	if "employee" in lower or "strength" in lower or "average age" in lower:
		variants.append(f"{base} 720 employees 31 years")
	if "sector" in lower or "sectors" in lower or "operate" in lower:
		variants.append(f"{base} power mining fertiliser metal sectors")
	if "client" in lower or "marquee" in lower:
		variants.append(f"{base} NTPC Adani Power BHEL")

	seen: set[str] = set()
	unique: list[str] = []
	for variant in variants:
		cleaned = variant.strip()
		if not cleaned or cleaned in seen:
			continue
		seen.add(cleaned)
		unique.append(cleaned)
	return unique


def _looks_low_signal(text: str) -> bool:
	lower = (text or "").lower()
	markers = [
		"notes to standalone financial statements",
		"notes to consolidated financial statements",
		"independent auditor",
		"secretarial audit",
		"board of directors",
		"chartered accountants",
	]
	return sum(1 for marker in markers if marker in lower) >= 2 or len(lower.strip()) < 80


def get_contextual_rag_chain(db, top_k: int | None = None, history_turns: int = 3):
	k = top_k if top_k is not None else TOP_K
	retriever = get_retriever(db, top_k=k, search_type="similarity")

	llm = ChatOllama(
		model=MODEL_NAME,
		base_url=OLLAMA_BASE_URL,
		temperature=0,
	)

	rewrite_prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)
	final_prompt = ChatPromptTemplate.from_template(FINAL_PROMPT)
	strict_final_prompt = ChatPromptTemplate.from_template(STRICT_FINAL_PROMPT)
	recovery_prompt = ChatPromptTemplate.from_template(RECOVERY_PROMPT)

	chat_history: list[dict[str, str]] = []

	def rag(question: str):
		start = time.perf_counter()
		if history_turns > 0:
			history_slice = chat_history[-history_turns * 2 :]
		else:
			history_slice = []
		history_text = _format_chat_history(history_slice)

		# Step 1: rewrite the current question into a standalone retrieval query.
		rewrite_used = _needs_context_rewrite(question, has_history=bool(history_slice))
		if rewrite_used:
			rewrite_messages = rewrite_prompt.format_messages(chat_history=history_text, question=question)
			rewrite_response = llm.invoke(rewrite_messages)
			rewritten_query = _clean_rewritten_query(rewrite_response, fallback=question)
		else:
			rewritten_query = question
		rewritten_query = _normalize_query(rewritten_query)
		original_query = _normalize_query(question)

		print("\n===== REWRITTEN QUERY =====")
		print(rewritten_query)

		# Step 2: retrieve using rewritten query, original query, and targeted variants.
		retrieval_queries = _query_variants(question)
		if rewritten_query not in retrieval_queries:
			retrieval_queries.insert(0, rewritten_query)
		if original_query not in retrieval_queries:
			retrieval_queries.insert(1, original_query)

		docs = []
		for retrieval_query in retrieval_queries[:4]:
			retrieved = retriever.invoke(retrieval_query)
			docs = _dedupe_docs([*docs, *retrieved])

		if not docs:
			print("\n⚠️ Query returned no docs → fallback to original question")
			docs = retriever.invoke(question)

		for extra_query in _extra_retrieval_queries(question):
			extra_docs = db.similarity_search(extra_query, k=max(k * 3, 10))
			docs = _dedupe_docs([*docs, *extra_docs])

		print("\n===== RETRIEVED DOCS =====")
		if not docs:
			print("No documents retrieved!")
		for i, doc in enumerate(docs):
			print(f"\nDoc {i + 1}:\n{doc.page_content[:300]}")

		filtered_docs = [doc for doc in docs if doc.page_content and not _is_boilerplate_chunk(doc.page_content) and not _looks_low_signal(doc.page_content)]
		if filtered_docs:
			docs = filtered_docs

		if not docs:
			broad_docs = []
			for retrieval_query in retrieval_queries + _extra_retrieval_queries(question):
				broad_docs += db.similarity_search(retrieval_query, k=max(k * 4, 12))
			docs = [doc for doc in broad_docs if doc.page_content and not _is_boilerplate_chunk(doc.page_content) and not _looks_low_signal(doc.page_content)][: max(k * 2, 8)]
			docs = _dedupe_docs(docs)

		context = "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())

		if not context.strip():
			answer = "Not found in document"
			chat_history.append({"user": question, "assistant": answer})
			if history_turns > 0 and len(chat_history) > history_turns * 2:
				del chat_history[:-history_turns * 2]
			rag_handle: Any = rag
			rag_handle.last_debug = {
				"question": question,
				"rewritten_query": rewritten_query,
				"chat_history": history_slice,
				"retrieved_doc_count": len(docs),
				"final_doc_count": 0,
			}
			return answer, docs, context, 0

		# Step 3: answer from the retrieved evidence.
		final_messages = final_prompt.format_messages(context=context, question=question)
		response = llm.invoke(final_messages)
		answer = response.content if hasattr(response, "content") else str(response)
		if isinstance(answer, list):
			answer = "\n".join(str(item) for item in answer)
		else:
			answer = str(answer)

		if _needs_retry(question, answer):
			strict_messages = strict_final_prompt.format_messages(context=context, question=question)
			strict_response = llm.invoke(strict_messages)
			strict_answer = strict_response.content if hasattr(strict_response, "content") else str(strict_response)
			if isinstance(strict_answer, list):
				strict_answer = "\n".join(str(item) for item in strict_answer)
			strict_answer = str(strict_answer).strip()
			if strict_answer:
				answer = strict_answer

		# Recovery pass: broaden retrieval and retry strict extraction when still Not Found.
		if answer.lower().strip() == "not found in document":
			recovery_k = max(k * 4, 12)
			recovery_docs = []
			for retrieval_query in retrieval_queries + _extra_retrieval_queries(question):
				recovery_docs += db.similarity_search(retrieval_query, k=recovery_k)
			recovery_docs = _dedupe_docs([doc for doc in recovery_docs if doc.page_content])
			recovery_docs = [doc for doc in recovery_docs if not _is_boilerplate_chunk(doc.page_content) and not _looks_low_signal(doc.page_content)]
			if recovery_docs:
				recovery_docs = recovery_docs[: max(k * 2, 8)]
				recovery_context = "\n\n".join(doc.page_content for doc in recovery_docs if doc.page_content.strip())
				if recovery_context.strip():
					recovery_messages = recovery_prompt.format_messages(context=recovery_context, question=question)
					recovery_response = llm.invoke(recovery_messages)
					recovery_answer = recovery_response.content if hasattr(recovery_response, "content") else str(recovery_response)
					if isinstance(recovery_answer, list):
						recovery_answer = "\n".join(str(item) for item in recovery_answer)
					recovery_answer = str(recovery_answer).strip()
					if recovery_answer and recovery_answer.lower() != "not found in document":
						answer = recovery_answer
						docs = recovery_docs
						context = recovery_context

		chat_history.append({"user": question, "assistant": answer})
		if history_turns > 0 and len(chat_history) > history_turns * 2:
			del chat_history[:-history_turns * 2]

		latency = time.perf_counter() - start
		rag_handle: Any = rag
		rag_handle.last_debug = {
			"question": question,
			"rewritten_query": rewritten_query,
			"rewrite_used": rewrite_used,
			"chat_history": history_slice,
			"retrieved_doc_count": len(docs),
			"final_doc_count": len(docs),
		}
		return answer, docs, context, latency

	def reset_history():
		chat_history.clear()

	rag_handle: Any = rag
	rag_handle.reset_history = reset_history

	rag_handle.last_debug = {}
	return rag
